import math
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from .log_config import get_logger, setup_backend_logging


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


MTX_RTSP_URL = _env("MTX_RTSP_URL", "rtsp://mediamtx:8554/live")
MTX_HOST = _env("MTX_WAIT_HOST", "mediamtx")
MTX_PORT = int(_env("MTX_WAIT_PORT", "8554"))
NVENC_PRESET = _env("NVENC_PRESET", "p4")
VIDEO_ENCODER = _env("VIDEO_ENCODER", "cpu").lower()
# Idle on boot unless AUTO_START=1. Lifespan also calls stop_stream() first to clear stray publishers.
AUTO_START = _env("AUTO_START", "0") == "1"
CUDA_PUBLISHER_BIN = _env("CUDA_PUBLISHER_BIN", "resgrid_plotter")
# NVENC / stdin pacing (--encoder-fps). Default 50 for load testing vs ~200 Hz pipeline; set CUDA_IQ_FPS to override.
CUDA_IQ_FPS = _env("CUDA_IQ_FPS", "50")
CUDA_IQ_CELL_PIXELS = _env("CUDA_IQ_CELL_PIXELS", "6")
CUDA_IQ_PRESET = _env("CUDA_IQ_PRESET", NVENC_PRESET)
CUDA_IQ_PRB = _env("CUDA_IQ_PRB", "273")
CUDA_IQ_SYMBOLS = _env("CUDA_IQ_SYMBOLS", "14")
CUDA_IQ_MAG_SCALE = _env("CUDA_IQ_MAG_SCALE", "1.0")
CUDA_IQ_QUEUE_DEPTH = _env("CUDA_IQ_QUEUE_DEPTH", "32")
CUDA_IQ_RASTER_THREADS = _env("CUDA_IQ_RASTER_THREADS", "16")
# When 1, spawn resource_grid_generator.py with the CUDA publisher. Default 0: use POST /api/stream/iq/generator/start.
CUDA_AUTO_START_IQ_GENERATOR = _env("CUDA_AUTO_START_IQ_GENERATOR", "0") == "1"
CUDA_IQ_USE_FIFO = _env("CUDA_IQ_USE_FIFO", "0") == "1"
CUDA_IQ_FIFO_PATH = _env("CUDA_IQ_FIFO_PATH", "/tmp/resgrid_iq.fifo")
CUDA_IQ_GEN_FPS = _env("CUDA_IQ_GEN_FPS", "200")
CUDA_SLOT_USE_FIFO = _env("CUDA_SLOT_USE_FIFO", "0") == "1"
CUDA_SLOT_FIFO_PATH = _env("CUDA_SLOT_FIFO_PATH", "/tmp/resgrid_slot.fifo")
# Seconds between demo slot records, or "fps"/"sync" (default) → 1/CUDA_IQ_FPS to match plotter frame rate (mode 3).
CUDA_SLOT_DEMO_INTERVAL = _env("CUDA_SLOT_DEMO_INTERVAL", "fps")
# resource_grid_generator.py --slot-pattern: demo | demo-multi | symbol-columns | symbol-windows (latter: one v1 msg per OFDM symbol per slot, random PRB windows at process start).
CUDA_SLOT_DEMO_PATTERN = _env("CUDA_SLOT_DEMO_PATTERN", "symbol-windows").strip().lower()
CUDA_SLOT_WINDOW_MIN_PRB = _env("CUDA_SLOT_WINDOW_MIN_PRB", "50").strip()
# Optional int seed for symbol-windows PRB spans; empty = nondeterministic (passed as --slot-window-seed).
CUDA_SLOT_WINDOW_SEED = _env("CUDA_SLOT_WINDOW_SEED", "").strip()
# Optional int seed for per-slot/per-symbol random RNTI (symbol-windows, symbol-columns); empty = nondeterministic.
CUDA_SLOT_RNTI_SEED = _env("CUDA_SLOT_RNTI_SEED", "").strip()
# Optional override for resgrid_plotter --data-mode (1 = built-in synth, 2 = full-grid IQ FIFO, 3 = slot messages). Empty = derive from STREAM_MODE and CUDA_IQ_USE_FIFO (IQ: 1 or 2; channel → 3).
CUDA_PLOTTER_DATA_MODE = _env("CUDA_PLOTTER_DATA_MODE", "").strip()
# Optional override for resource_grid_generator.py --data-mode. Empty = match effective plotter mode.
CUDA_PRODUCER_DATA_MODE = _env("CUDA_PRODUCER_DATA_MODE", "").strip()
# Passed to resource_grid_generator.py as --log-level when set (e.g. DEBUG). Empty = script default INFO.
IQ_FIFO_PRODUCER_LOG_LEVEL = _env("IQ_FIFO_PRODUCER_LOG_LEVEL", "").strip().upper()

IQ_PRODUCER_FPS_MIN = float(_env("IQ_PRODUCER_FPS_MIN", "1"))
IQ_PRODUCER_FPS_MAX = float(_env("IQ_PRODUCER_FPS_MAX", "600"))

# POST /api/stream/plotter/capturer: file capturer buffers RGBA in RAM then JPEG workers flush to disk (parallel to video encoder).
_cm = _env("RESGRID_CAPTURER_MAX_FRAMES", "").strip()
if not _cm:
    _cm = _env("RESGRID_ENCODER_CAPTURE_MAX_FRAMES", "20000")
_CAPTURER_MAX_FRAMES = max(1, int(_cm))

# Synthetic IQ pattern id (must match resource_grid_generator.py and plotter `kIqPatternMax` in plotter_common.cuh).
IQ_PATTERN_DEFAULT = 0
IQ_PATTERN_SINGLE_TONE = 1
IQ_PATTERN_GAUSSIAN = 2
IQ_PATTERN_VORTEX = 3
IQ_PATTERN_MAX = IQ_PATTERN_VORTEX

_iq_producer_command_override: Optional[float] = None
_iq_pattern_override: Optional[int] = None


def _iq_grid_dims() -> tuple[int, int]:
    try:
        p = int(str(CUDA_IQ_PRB).strip())
    except ValueError:
        p = 273
    try:
        s = int(str(CUDA_IQ_SYMBOLS).strip())
    except ValueError:
        s = 14
    return max(1, p), max(1, s)


def _slot_demo_interval_seconds_str() -> str:
    """String float for resource_grid_generator ``--slot-interval``: wall seconds between slot demo records.

    ``CUDA_SLOT_DEMO_INTERVAL`` of ``fps``, ``sync``, ``iq_fps``, or empty uses ``1/max(1, CUDA_IQ_FPS)``
    so PHY/channel demo updates once per plotter frame. Any positive float is used as literal seconds.
    Non-positive numeric values fall back to the fps-derived interval.
    """
    raw = str(CUDA_SLOT_DEMO_INTERVAL).strip().lower()
    if raw in ("fps", "sync", "iq_fps", ""):
        try:
            hz = float(str(CUDA_IQ_FPS).strip())
        except ValueError:
            hz = 50.0
        return f"{1.0 / max(1.0, hz):.9g}"
    try:
        v = float(str(CUDA_SLOT_DEMO_INTERVAL).strip())
    except ValueError:
        hz = float(str(CUDA_IQ_FPS).strip() or "50")
        return f"{1.0 / max(1.0, hz):.9g}"
    if v <= 0.0:
        hz = float(str(CUDA_IQ_FPS).strip() or "50")
        return f"{1.0 / max(1.0, hz):.9g}"
    return f"{v:.9g}"


_SLOT_DEMO_PATTERNS = frozenset({"demo", "demo-multi", "symbol-columns", "symbol-windows"})


def _slot_demo_pattern_cli() -> str:
    p = CUDA_SLOT_DEMO_PATTERN
    if p in _SLOT_DEMO_PATTERNS:
        return p
    _log.warning(
        "CUDA_SLOT_DEMO_PATTERN=%r invalid; use demo|demo-multi|symbol-columns|symbol-windows; using symbol-windows",
        p,
    )
    return "symbol-windows"


def _slot_window_min_prb_cli() -> str:
    try:
        v = int(str(CUDA_SLOT_WINDOW_MIN_PRB).strip())
        return str(max(1, v))
    except ValueError:
        _log.warning("CUDA_SLOT_WINDOW_MIN_PRB invalid; using 50")
        return "50"


def _slot_window_seed_cli_args() -> list[str]:
    raw = str(CUDA_SLOT_WINDOW_SEED).strip()
    if not raw:
        return []
    try:
        int(raw)
    except ValueError:
        _log.warning("CUDA_SLOT_WINDOW_SEED=%r invalid; omitting --slot-window-seed", raw)
        return []
    return ["--slot-window-seed", raw]


def _slot_rnti_seed_cli_args() -> list[str]:
    raw = str(CUDA_SLOT_RNTI_SEED).strip()
    if not raw:
        return []
    try:
        int(raw)
    except ValueError:
        _log.warning("CUDA_SLOT_RNTI_SEED=%r invalid; omitting --slot-rnti-seed", raw)
        return []
    return ["--slot-rnti-seed", raw]


def _extend_producer_symbol_window_args(cmd: list[str]) -> None:
    cmd.extend(["--slot-window-min-prb", _slot_window_min_prb_cli()])
    cmd.extend(_slot_window_seed_cli_args())
    cmd.extend(_slot_rnti_seed_cli_args())


# Append server + uvicorn logs here (set to none/off to disable file logging only).
LOG_FILE = _env("LOG_FILE", "logs/server.log")
LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()
_log = get_logger()


def _fifo_chown_target() -> tuple[int, int] | None:
    """If set, chown IQ/slot FIFOs after mkfifo so host-mounted pipes match sibling containers (e.g. OAI uid 1000)."""
    if _env("RESGRID_FIFO_CHOWN_DISABLE", "0").strip().lower() in ("1", "true", "yes", "on"):
        return None
    raw = _env("RESGRID_FIFO_CHOWN", "1000:1000").strip()
    if raw.lower() in ("none", "-", "off", "no"):
        return None
    parts = raw.split(":", 1)
    if len(parts) != 2:
        _log.warning("RESGRID_FIFO_CHOWN must be uid:gid (got %r); skipping FIFO chown", raw)
        return None
    try:
        return (int(parts[0].strip()), int(parts[1].strip()))
    except ValueError:
        _log.warning("RESGRID_FIFO_CHOWN invalid (got %r); skipping FIFO chown", raw)
        return None


def _fifo_apply_chown(path: str) -> None:
    t = _fifo_chown_target()
    if t is None:
        return
    u, g = t
    try:
        os.chown(path, u, g, follow_symlinks=False)
    except OSError as e:
        _log.warning("FIFO chown failed for %s to %s:%s: %s", path, u, g, e)


_IQ_FIFO_GENERATOR_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "resource_grid_generator.py"
# Host:port the *browser* uses to reach MediaMTX HTTP (HLS). Fallback when request has no usable Host / Origin (e.g. curl without -H).
MTX_PUBLIC_HTTP_BASE = _env("MTX_PUBLIC_HTTP_BASE", "http://localhost:8888").rstrip("/")
CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
# Browsers on the LAN open Vite at http://<subnet-ip>:<port> — allow private-IP origins on any port.
CORS_ORIGIN_REGEX = _env(
    "CORS_ORIGIN_REGEX",
    r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?",
)


# Stream modes (STREAM_MODE env). REST keeps stable macro names; plotter mapping:
#   SYNTHETIC_IQ — default: built-in C++ synthetic grids (--data-mode 1). With CUDA_IQ_USE_FIFO=1: full-grid FIFO
#       (--data-mode 2) + resource_grid_generator.py.
#   SYNTHETIC_CHANNEL_BASED — slot messages / channel-based (each slot FIFO message is a slot allocation, partial grid)
#       → resgrid_plotter --data-mode 3; optional IQ FIFO is a full-grid sidecar.
PLOTTER_PIPELINE_INTERNAL_SYNTH = 1
PLOTTER_PIPELINE_FULL_GRID_FIFO = 2
PLOTTER_PIPELINE_SLOT_MESSAGES = 3

STREAM_MODE_SYNTHETIC_VIDEO = "SYNTHETIC_VIDEO"
STREAM_MODE_SYNTHETIC_IQ = "SYNTHETIC_IQ"
_LEGACY_STREAM_MODE_CHANNEL_BASED = "CHANNEL_BASED_SYNTHETIC_GENERATION"
STREAM_MODE_CHANNEL_BASED_SYNTHETIC = "SYNTHETIC_CHANNEL_BASED"
_STREAM_MODES_IQ_CUDA = frozenset({STREAM_MODE_SYNTHETIC_IQ, STREAM_MODE_CHANNEL_BASED_SYNTHETIC})
_STREAM_MODES_ALL = frozenset({STREAM_MODE_SYNTHETIC_VIDEO, *_STREAM_MODES_IQ_CUDA})


def _coerce_stream_mode_input(mode: str) -> str:
    """Map legacy/alias tokens to canonical mode strings."""
    m = mode.strip().upper().replace("-", "_")
    if m == _LEGACY_STREAM_MODE_CHANNEL_BASED:
        return STREAM_MODE_CHANNEL_BASED_SYNTHETIC
    return m


def _cuda_plotter_data_mode(stream_mode: str) -> int:
    raw = CUDA_PLOTTER_DATA_MODE.strip()
    if raw == "1":
        return PLOTTER_PIPELINE_INTERNAL_SYNTH
    if raw in ("2", "3"):
        return int(raw)
    return (
        PLOTTER_PIPELINE_SLOT_MESSAGES
        if stream_mode == STREAM_MODE_CHANNEL_BASED_SYNTHETIC
        else PLOTTER_PIPELINE_FULL_GRID_FIFO
    )


def _cuda_producer_data_mode(plot_dm: int) -> int:
    raw = CUDA_PRODUCER_DATA_MODE.strip()
    if raw in ("1", "2", "3"):
        return int(raw)
    return plot_dm


def _parse_stream_mode(raw: str) -> str:
    r = raw.strip().upper().replace("-", "_")
    if r in (_LEGACY_STREAM_MODE_CHANNEL_BASED, "SYNTHETIC_CHANNEL_BASED"):
        return STREAM_MODE_CHANNEL_BASED_SYNTHETIC
    if r == "SYNTHETIC_IQ":
        return STREAM_MODE_SYNTHETIC_IQ
    return STREAM_MODE_SYNTHETIC_VIDEO


STREAM_MODE = _parse_stream_mode(_env("STREAM_MODE", "synthetic_channel_based"))
_mode_lock = threading.Lock()
_active_stream_mode: str = STREAM_MODE

_mtx_path = (urlparse(MTX_RTSP_URL).path or "/live").strip("/") or "live"

_ffmpeg: Optional[subprocess.Popen] = None
_cuda_iq: Optional[subprocess.Popen] = None
_iq_fifo_producer: Optional[subprocess.Popen] = None
_lock = threading.Lock()
_current_lavfi_video: str = ""
# Set True when the last successful CUDA start created / used the slot named pipe (for cleanup on stop).
_cuda_spawn_used_slot_fifo: bool = False
_cuda_plotter_capturer_dir: Optional[str] = None

_SHM_CAPTURER_TOTAL_OFF = 88
_SHM_CAPTURER_INFLIGHT_OFF = 92
_SHM_CAPTURER_WRITTEN_OFF = 96
_SHM_CAPTURER_PHASE_OFF = 100
_SHM_CAPTURER_PATH_OFF = 104
_SHM_CAPTURER_PATH_CAP = 256
_SHM_CAPTURER_QUEUE_DEPTH_OFF = 360
_SHM_CAPTURER_QUEUE_CAP_OFF = 364
_SHM_CAPTURER_JPEG_THREADS_OFF = 368
_SHM_CAPTURER_STOP_OFF = 372
_frame_re = re.compile(r"frame=\s*(\d+)")
_fps_re = re.compile(r"fps=\s*([\d.]+)")
_speed_re = re.compile(r"speed=\s*([\d.]+)x")
_bitrate_re = re.compile(r"bitrate=\s*([\d.]+)\s*kbits/s")


class GenerationStats(BaseModel):
    """Snapshot from FFmpeg's last progress line (stderr)."""

    fps: Optional[float] = Field(
        None,
        description="Encoder output FPS from FFmpeg progress (stderr fps= = stdin frames/s); IQ CUDA matches `--encoder-fps` when the plotter paces writes.",
    )
    speed: Optional[float] = Field(
        None,
        description="Encoding speed vs realtime (e.g. 1.0 = realtime), from FFmpeg progress output.",
    )
    frame: Optional[int] = Field(None, description="Total output frames reported so far.")
    bitrate_kbits: Optional[float] = Field(None, description="Output bitrate in kbit/s, if present in the progress line.")


_stats_lock = threading.Lock()
_stats = GenerationStats()

_IQ_PRODUCER_SHM_BYTES = 32
_IQ_CUDA_SHM_BYTES = 384
_iq_producer_shm: Optional[shared_memory.SharedMemory] = None
_iq_cuda_shm: Optional[shared_memory.SharedMemory] = None


class PlaybackUrls(BaseModel):
    """URLs to read the published stream from MediaMTX (WebRTC for real-time; HLS as fallback)."""

    path: str = Field(description="MediaMTX path name (same as in RTSP publish URL).")
    hls: str = Field(description="HLS playlist URL — higher latency; works everywhere hls.js does.")
    webrtc_whep: str = Field(
        description="WHEP URL for WebRTC playback — lowest latency in Chromium/Firefox; use with MediaMTX reader.",
    )
    note: str = Field(
        default="Prefer WebRTC (webrtc_whep) for real-time; HLS segments always buffer partly.",
    )


class StreamStatus(BaseModel):
    stream_mode: str = Field(
        description="SYNTHETIC_VIDEO (FFmpeg lavfi), SYNTHETIC_IQ (built-in synthesizer or full-grid FIFO with CUDA_IQ_USE_FIFO), or SYNTHETIC_CHANNEL_BASED (slot messages / --data-mode 3)."
    )
    generating: bool = Field(
        description="True while the active mode's publisher is running (FFmpeg or resgrid_plotter).",
    )
    video_encoder: str = Field(description="Backend FFmpeg encoder when video mode; cuda_iq_pipeline when IQ mode.")
    lavfi_video: Optional[str] = Field(
        None,
        description="Lavfi graph when FFmpeg runs; IQ mode shows grid summary when CUDA publisher runs.",
    )
    stats: Optional[GenerationStats] = Field(
        None,
        description=(
            "Synthetic video: FFmpeg stderr progress (fps, speed, frames, bitrate). "
            "IQ CUDA: **fps** ≈ NVENC stdin rate from FFmpeg **fps=** (~`CUDA_IQ_FPS` / `--encoder-fps`); "
            "**frame** = encoded frame count from progress when present, else CUDA SHM (offset 32); speed/bitrate from FFmpeg when present."
        ),
    )
    publish_rtsp: str = Field(description="Internal RTSP URL used by FFmpeg to publish.")
    mtx_path: str = Field(description="MediaMTX path name for this stream.")
    iq_prb: int = Field(description="PRB dimension for the configured IQ CUDA publisher (CUDA_IQ_PRB).")
    iq_symbols: int = Field(description="Symbol dimension for the IQ publisher (CUDA_IQ_SYMBOLS).")
    iq_producer_fps: Optional[float] = Field(
        None,
        description="Measured IQ generation rate (Hz): Python FIFO writer and/or internal CUDA synthetic fill (SHM offset 0).",
    )
    iq_plotter_fps: Optional[float] = Field(
        None,
        description="Measured **Raster** output image rate (Hz), CUDA SHM offset **8** (legacy name); encoder **stdin** image rate may differ (see **encoder_input_image_s**).",
    )
    iq_producer_command_fps: Optional[float] = Field(
        None,
        description="Commanded IQ generation FPS in shared memory (offset 8 Python SHM / 24 CUDA SHM): "
        "**SYNTHETIC_IQ** and **SYNTHETIC_CHANNEL_BASED** (internal synthesizer or optional resource_grid_generator.py).",
    )
    iq_pattern: Optional[int] = Field(
        None,
        description="Synthetic IQ pattern id (Viridis demos); **SYNTHETIC_IQ** only. Not reported for **SYNTHETIC_CHANNEL_BASED** (mode 3). See GET /api/stream/iq/patterns.",
    )
    iq_generator_running: bool = Field(
        default=False,
        description="True when **resource_grid_generator.py** is running (slot and/or IQ FIFO feeder).",
    )
    iq_generator_capable: bool = Field(
        default=False,
        description="True when CUDA is up with slot FIFO and/or **CUDA_IQ_USE_FIFO** so that script can run.",
    )
    assembly_input_msg_s: Optional[float] = Field(
        None,
        description="Slot / IQ / synthetic messages accepted per second (**Assembly** input; CUDA SHM **40**).",
    )
    assembly_output_slot_s: Optional[float] = Field(
        None,
        description="Completed **`GridFrame`** / slot outputs per second from **Assembly** queues (SHM **48**).",
    )
    raster_input_slot_s: Optional[float] = Field(
        None,
        description="SHM **56**: **Assembly** output rate (Hz), same as **assembly_output_slot_s** — chained “input to Raster” = previous stage output.",
    )
    raster_output_image_s: Optional[float] = Field(
        None,
        description="RGBA images **Raster** produced per second (SHM **64**, same as offset **8**).",
    )
    encoder_input_image_s: Optional[float] = Field(
        None,
        description="SHM **72**: **Raster** RGBA output rate (Hz), same as **raster_output_image_s** — chained “input to Encode” = previous stage output.",
    )
    encoder_output_fps: Optional[float] = Field(
        None,
        description="Frames written to FFmpeg stdin per second (NVENC publish cadence; SHM **80**).",
    )
    capturer_capable: bool = Field(
        default=False,
        description="True when this process started **resgrid_plotter** with **`RESGRID_CAPTURER_DIR`** (or legacy **`RESGRID_ENCODER_SNAP_DIR`**) passed as **`--capturer-dir`**. **Encoder** path is RTSP/video; **capturer** path is still JPEG files.",
    )
    capturer_active: Optional[bool] = Field(
        None,
        description="True while file capture is in progress (SHM **phase** 1 = recording / buffering from **raster**, 2 = writing JPEGs).",
    )
    capturer_phase: Optional[str] = Field(
        None,
        description="**idle** | **capturing** | **saving** from CUDA SHM (file capturer tail; same phase encoding as before).",
    )
    capturer_inflight: Optional[int] = Field(
        None,
        description=(
            "**In-flight** frames not yet committed to disk: **pushed − written** "
            "(deque wait + JPEG workers; SHM @ 92). "
            "See **capturer_queue_depth** for deque-only backlog."
        ),
    )
    capturer_written: Optional[int] = Field(
        None,
        description="JPEG files written (**SHM** @ 96) during flush (and final count after **idle**).",
    )
    capturer_total: Optional[int] = Field(
        None,
        description="Frames requested for the armed session (**SHM** @ 88).",
    )
    capturer_out_dir: Optional[str] = Field(
        None,
        description="Output directory (UTF-8 in SHM @ 104…); empty when idle.",
    )
    capturer_queue_depth: Optional[int] = Field(
        None,
        description="Frames **waiting** in the bounded **CapturerFrameQueue** deque; SHM tail @ 360.",
    )
    capturer_queue_cap: Optional[int] = Field(
        None,
        description="Maximum capturer queue slots; SHM @ 364.",
    )
    capturer_jpeg_threads: Optional[int] = Field(
        None,
        description="JPEG writer threads (`RESGRID_CAPTURE_JPEG_THREADS` or plotter default); SHM @ 368.",
    )


class StreamControlResponse(BaseModel):
    status: str
    pid: Optional[int] = None
    lavfi_video: Optional[str] = None
    stream_mode: Optional[str] = None


class StreamModeInfo(BaseModel):
    mode: str = Field(
        description="SYNTHETIC_VIDEO | SYNTHETIC_IQ (built-in / FIFO) | SYNTHETIC_CHANNEL_BASED (slot messages)",
    )
    note: str = Field(description="Operational hint for this mode.")


class StreamModeSet(BaseModel):
    mode: str = Field(
        description="SYNTHETIC_VIDEO | SYNTHETIC_IQ | SYNTHETIC_CHANNEL_BASED (alias CHANNEL_BASED_SYNTHETIC_GENERATION accepted)",
        json_schema_extra={"example": "SYNTHETIC_VIDEO"},
    )


class StreamStartBody(BaseModel):
    """Optional body for POST /api/stream/start (JSON). When `mode` is set, it becomes the active mode before spawning."""

    mode: Optional[str] = Field(
        default=None,
        description="SYNTHETIC_VIDEO | SYNTHETIC_IQ | SYNTHETIC_CHANNEL_BASED. Legacy CHANNEL_BASED_SYNTHETIC_GENERATION is accepted. Omit to use current mode.",
    )


class StreamStopBody(BaseModel):
    """Optional body for POST /api/stream/stop. `mode` is accepted for clients that want a symmetric payload; it does not affect behavior."""

    mode: Optional[str] = Field(default=None, description="Ignored; stop always tears down whichever publisher is running.")


class CapturerStartBody(BaseModel):
    frames: int = Field(
        default=2000,
        ge=1,
        le=50_000,
        description="RGBA frames for this session: held in the **capturer queue** and parallel JPEG workers until flushed to disk (quality ~92). "
        f"Capped by RESGRID_CAPTURER_MAX_FRAMES (this server: {_CAPTURER_MAX_FRAMES}); legacy **RESGRID_ENCODER_CAPTURE_MAX_FRAMES** also honored.",
    )


class CapturerStartResponse(BaseModel):
    out_dir: str = Field(description="Directory where **`frame_*.jpg`** files are written (container path; bind-mount for host access).")
    frames: int


class CapturerStopResponse(BaseModel):
    """Acknowledgement that a stop was written to CUDA IQ SHM; the plotter clears it on the next raster tick."""

    status: str = Field(description='Typically **`"stop_requested"`**.')


class IqProducerFpsResponse(BaseModel):
    fps: float = Field(description="Commanded IQ generation FPS from shared memory or saved override.")
    min_fps: float = Field(description="Minimum allowed generator FPS (env IQ_PRODUCER_FPS_MIN).")
    max_fps: float = Field(description="Maximum allowed generator FPS (env IQ_PRODUCER_FPS_MAX).")
    stream_mode: str
    generating: bool
    iq_fifo_mode: bool = Field(
        description="True when I/Q is read from the IQ named pipe (**CUDA_IQ_USE_FIFO=1**: Python generator or external writer)."
    )


class IqProducerFpsPutBody(BaseModel):
    fps: float = Field(..., description="New commanded IQ generation FPS; written to CUDA control SHM and Python generator SHM when active.")


class IqPatternInfo(BaseModel):
    id: int
    label: str


class IqPatternsResponse(BaseModel):
    patterns: list[IqPatternInfo]


class IqPatternPutBody(BaseModel):
    pattern: int = Field(..., description="Synthetic IQ pattern id (0…max). Written to both IQ shared-memory blocks when active.")


class IqPatternResponse(BaseModel):
    pattern: int


class IqGeneratorControlResponse(BaseModel):
    status: str = Field(description="started | stopped | already_running | not_running")
    pid: Optional[int] = None


class IqChanLegendRgb(BaseModel):
    """sRGB 0…255; matches CUDA `channel_tint_rgb` base coefficients × 255 (rounded)."""

    r: int = Field(..., ge=0, le=255)
    g: int = Field(..., ge=0, le=255)
    b: int = Field(..., ge=0, le=255)


class IqChanLegendEntry(BaseModel):
    """One PHY-style logical channel id (values 1…8 in slot messages / plotter region labels)."""

    id: int = Field(..., ge=1, le=255, description="Logical PHY channel id (1…8 per OAI slot exporter / plotter).")
    label: str = Field(..., description="Short PHY name (same as on-screen region labels).")
    rgb: IqChanLegendRgb
    hex: str = Field(..., description='CSS sRGB "#rrggbb" for the base tint.')


class IqChannelLegendResponse(BaseModel):
    channels: list[IqChanLegendEntry] = Field(
        ...,
        description="PHY ids 1…8 and tints from `resgrid_plotter` `channel_tint_rgb`.",
    )
    channel_zero_note: str = Field(
        ...,
        description="Optional legend footnote for channel id 0 (may be empty).",
    )
    render_note: str = Field(
        ...,
        description="Optional legend footnote on tint vs magnitude (may be empty).",
    )


def _iq_channel_legend() -> IqChannelLegendResponse:
    """Keep in sync with `channel_tint_rgb` in backend/resouce-grid-plotter/src/plotter_raster.cu."""
    specs: tuple[tuple[int, str, tuple[float, float, float]], ...] = (
        (1, "SSB", (0.92, 0.25, 0.28)),
        (2, "PRACH", (0.15, 0.82, 0.82)),
        (3, "PDCCH", (0.95, 0.62, 0.15)),
        (4, "PUCCH", (0.78, 0.38, 0.92)),
        (5, "PDSCH", (0.22, 0.82, 0.38)),
        (6, "PUSCH", (0.28, 0.45, 0.95)),
        (7, "CSI-RS", (0.55, 0.55, 0.55)),
        (8, "SRS", (0.55, 0.55, 0.55)),
    )
    channels: list[IqChanLegendEntry] = []
    for cid, label, (fr, fg, fb) in specs:
        r = int(round(fr * 255.0))
        g = int(round(fg * 255.0))
        b = int(round(fb * 255.0))
        h = f"#{r:02x}{g:02x}{b:02x}"
        channels.append(
            IqChanLegendEntry(id=cid, label=label, rgb=IqChanLegendRgb(r=r, g=g, b=b), hex=h)
        )
    return IqChannelLegendResponse(
        channels=channels,
        channel_zero_note="",
        render_note="",
    )


def _mtx_http_base_from_request(request: Request) -> str:
    """MediaMTX HTTP base (`scheme://host:8888`) inferred from how the client reached this API."""
    raw = (request.headers.get("origin") or request.headers.get("referer") or "").strip()
    if raw:
        p = urlparse(raw)
        if p.hostname:
            scheme = p.scheme or "http"
            return f"{scheme}://{p.hostname}:8888"
    ff = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
    if ff:
        host = ff.split(":", 1)[0].strip()
        if host:
            proto = (request.headers.get("x-forwarded-proto") or "http").split(",")[0].strip()
            if proto not in ("http", "https"):
                proto = "http"
            return f"{proto}://{host}:8888"
    h = (request.headers.get("host") or "").split(",")[0].strip()
    if h:
        host = h.split(":", 1)[0].strip()
        if host:
            scheme = request.url.scheme or "http"
            if scheme not in ("http", "https"):
                scheme = "http"
            return f"{scheme}://{host}:8888"
    if request.url.hostname:
        scheme = request.url.scheme or "http"
        if scheme not in ("http", "https"):
            scheme = "http"
        return f"{scheme}://{request.url.hostname}:8888"
    return MTX_PUBLIC_HTTP_BASE.rstrip("/")


def _playback_for_request(request: Request) -> PlaybackUrls:
    base = _mtx_http_base_from_request(request)
    hls = f"{base}/{_mtx_path}/index.m3u8"
    p = urlparse(base)
    host = p.hostname or "localhost"
    scheme = p.scheme or "http"
    whep_base = f"{scheme}://{host}:8889"
    webrtc_whep = f"{whep_base}/{_mtx_path}/whep"
    return PlaybackUrls(path=_mtx_path, hls=hls, webrtc_whep=webrtc_whep)


# Default lavfi graph is CPU-cheap so NVENC can hit realtime (target ~1.0× speed). Mandelbrot at 720p is CPU-heavy
# and often yields ~0.4–0.5× speed — HLS then stalls and sessions drop as "inactive".
_DEFAULT_LAVFI_VIDEO = "testsrc=size=1280x720:rate=30"


def _lavfi_video() -> str:
    return _env("LAVFI_VIDEO", _DEFAULT_LAVFI_VIDEO)


def _is_iq_cuda_mode(mode: str) -> bool:
    return mode in _STREAM_MODES_IQ_CUDA


def get_stream_mode() -> str:
    with _mode_lock:
        return _active_stream_mode


def _iq_fifo_pipe_active_unlocked() -> bool:
    """True when the IQ named pipe feeds resgrid_plotter (CUDA_IQ_USE_FIFO=1)."""
    return CUDA_IQ_USE_FIFO


def _is_stream_mode_channel_based(sm: str) -> bool:
    return sm in (STREAM_MODE_CHANNEL_BASED_SYNTHETIC, _LEGACY_STREAM_MODE_CHANNEL_BASED)


def set_stream_mode(mode: str) -> str:
    global _active_stream_mode
    m = _coerce_stream_mode_input(mode)
    if m not in _STREAM_MODES_ALL:
        raise ValueError(
            f"mode must be one of {sorted(_STREAM_MODES_ALL)!r}, got {mode!r}"
        )
    with _mode_lock:
        prev = _active_stream_mode
        if prev == m:
            return m
        _active_stream_mode = m
    stop_stream()
    return _active_stream_mode


def _mode_note(mode: str) -> str:
    if mode == STREAM_MODE_SYNTHETIC_VIDEO:
        return "Use Start/Stop to control FFmpeg (lavfi) on this API."
    if mode == STREAM_MODE_CHANNEL_BASED_SYNTHETIC:
        return (
            "Use Start/Stop for resgrid_plotter (channel-based / slot-messages: slot FIFO carries per-slot allocations — PHY channel + RNTI). "
            "Pair with a slot writer (e.g. resource_grid_generator.py --slot-fifo) or external tool on CUDA_SLOT_FIFO_PATH."
        )
    return (
        "Use Start/Stop for resgrid_plotter: **SYNTHETIC_IQ** uses built-in synthetic grids by default, or full-grid IQ FIFO when **CUDA_IQ_USE_FIFO=1** (`--iq-fifo` → CUDA → NVENC)."
    )


def _resolve_mode_for_start_locked(requested_mode: Optional[str]) -> str:
    """Caller must hold _lock. Applies optional requested mode and stops the other publisher if switching."""
    global _active_stream_mode
    if requested_mode is None or not str(requested_mode).strip():
        return get_stream_mode()
    m = _coerce_stream_mode_input(str(requested_mode))
    if m not in _STREAM_MODES_ALL:
        raise ValueError(
            f"mode must be one of {sorted(_STREAM_MODES_ALL)!r}, got {requested_mode!r}"
        )
    with _mode_lock:
        _active_stream_mode = m
    if m == STREAM_MODE_SYNTHETIC_VIDEO and _is_cuda_running():
        _terminate_cuda_unlocked()
    elif _is_iq_cuda_mode(m) and _is_ffmpeg_running():
        _terminate_ffmpeg_unlocked()
    return m


def _wait_tcp(host: str, port: int, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[OSError] = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as e:
            last_err = e
            time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {host}:{port}: {last_err}")


def _build_ffmpeg_cmd(video_lavfi: str) -> list[str]:
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-re",
        "-f",
        "lavfi",
        "-i",
        video_lavfi,
    ]
    if VIDEO_ENCODER == "nvenc":
        cmd.extend(
            [
        "-c:v",
        "h264_nvenc",
        "-preset",
        NVENC_PRESET,
        "-pix_fmt",
        "yuv420p",
                "-g",
                "30",
                "-forced-idr",
                "1",
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-tune",
                "zerolatency",
                "-pix_fmt",
                "yuv420p",
                "-bf",
                "0",
                "-g",
                "30",
                "-keyint_min",
                "30",
                "-x264-params",
                "keyint=30:min-keyint=30:scenecut=0",
            ]
        )
    cmd.extend(
        [
            # Video-only: matches WebRTC reader (H264); avoids MediaMTX warning skipping MPEG-4 Audio in WHEP.
            "-an",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        MTX_RTSP_URL,
    ]
    )
    return cmd


def _is_ffmpeg_running() -> bool:
    global _ffmpeg
    return _ffmpeg is not None and _ffmpeg.poll() is None


def _is_cuda_running() -> bool:
    global _cuda_iq
    return _cuda_iq is not None and _cuda_iq.poll() is None


def _is_running() -> bool:
    return _is_ffmpeg_running() or _is_cuda_running()


def _apply_progress_line(line: str) -> None:
    if "fps=" not in line or "frame=" not in line:
        return
    global _stats
    fm = _frame_re.search(line)
    fpm = _fps_re.search(line)
    sm = _speed_re.search(line)
    bm = _bitrate_re.search(line)
    with _stats_lock:
        prev = _stats
        _stats = GenerationStats(
            frame=int(fm.group(1)) if fm else prev.frame,
            fps=float(fpm.group(1)) if fpm else prev.fps,
            speed=float(sm.group(1)) if sm else prev.speed,
            bitrate_kbits=float(bm.group(1)) if bm else prev.bitrate_kbits,
        )


def _stderr_drain_loop(proc: subprocess.Popen) -> None:
    if proc.stderr is None:
        return
    buf = ""
    try:
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            buf = buf.replace("\r", "\n")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line:
                    _apply_progress_line(line)
        tail = buf.strip()
        if tail:
            _apply_progress_line(tail)
    except Exception:
        _log.exception("FFmpeg stderr reader failed")
    finally:
        try:
            proc.stderr.close()
        except OSError:
            pass


def _spawn_stderr_thread(proc: subprocess.Popen) -> None:
    threading.Thread(target=_stderr_drain_loop, args=(proc,), daemon=True).start()


def _default_iq_producer_command_fps() -> float:
    """SHM-commanded IQ/ slot rate default; separate from encoder ``CUDA_IQ_FPS`` (plotter ``--encoder-fps``).

    Python generator and slot writers follow **CUDA_IQ_GEN_FPS** whenever they are used (IQ FIFO, synthetic IQ,
    or channel / slot-message mode). Encoder output stays **CUDA_IQ_FPS** (e.g. 50 Hz) via plotter ``--encoder-fps``.
    The plotter ``--fps`` knob is the pipeline / internal producer nominal rate (**CUDA_IQ_GEN_FPS**, e.g. 200 Hz).
    """
    sm_fn = get_stream_mode()
    use_gen_default = (
        CUDA_IQ_USE_FIFO
        or sm_fn == STREAM_MODE_SYNTHETIC_IQ
        or sm_fn == STREAM_MODE_CHANNEL_BASED_SYNTHETIC
        or sm_fn == _LEGACY_STREAM_MODE_CHANNEL_BASED
    )
    raw = CUDA_IQ_GEN_FPS if use_gen_default else CUDA_IQ_FPS
    try:
        v = float(str(raw).strip())
    except ValueError:
        v = 200.0 if use_gen_default else 50.0
    return max(IQ_PRODUCER_FPS_MIN, min(IQ_PRODUCER_FPS_MAX, v))


def _validate_producer_command_fps(fps: float) -> float:
    if math.isnan(fps) or math.isinf(fps):
        raise ValueError("fps must be a finite number")
    if fps < IQ_PRODUCER_FPS_MIN or fps > IQ_PRODUCER_FPS_MAX:
        raise ValueError(
            f"fps must be between {IQ_PRODUCER_FPS_MIN:g} and {IQ_PRODUCER_FPS_MAX:g}",
        )
    return float(fps)


def _default_iq_pattern() -> int:
    try:
        v = int(str(_env("IQ_PATTERN", "0")).strip())
    except ValueError:
        v = IQ_PATTERN_DEFAULT
    return max(IQ_PATTERN_DEFAULT, min(IQ_PATTERN_MAX, v))


def _validate_iq_pattern(pid: int) -> int:
    if pid < IQ_PATTERN_DEFAULT or pid > IQ_PATTERN_MAX:
        raise ValueError(f"pattern must be between {IQ_PATTERN_DEFAULT} and {IQ_PATTERN_MAX}")
    return int(pid)


def _effective_iq_pattern_unlocked() -> int:
    if _iq_pattern_override is not None:
        return max(IQ_PATTERN_DEFAULT, min(IQ_PATTERN_MAX, int(_iq_pattern_override)))
    return _default_iq_pattern()


def _read_iq_pattern_from_shm_unlocked() -> int:
    """Current pattern from cuda shm if present, else override/env."""
    if _iq_cuda_shm is not None:
        try:
            (p,) = struct.unpack_from("I", _iq_cuda_shm.buf, 16)
            if p <= IQ_PATTERN_MAX:
                return int(p)
        except (struct.error, TypeError, ValueError):
            pass
    return _effective_iq_pattern_unlocked()


def _write_iq_pattern_shm_unlocked(pattern: int) -> int:
    global _iq_pattern_override
    v = _validate_iq_pattern(pattern)
    _iq_pattern_override = v
    if _iq_producer_shm is not None:
        struct.pack_into("I", _iq_producer_shm.buf, 16, v)
    if _iq_cuda_shm is not None:
        struct.pack_into("I", _iq_cuda_shm.buf, 16, v)
    return v


def _pack_producer_shm_init(cmd: float, pattern: int) -> bytes:
    b = bytearray(_IQ_PRODUCER_SHM_BYTES)
    struct.pack_into("d", b, 0, float("nan"))
    struct.pack_into("d", b, 8, float(cmd))
    struct.pack_into("I", b, 16, pattern & 0xFFFFFFFF)
    return bytes(b)


def _pack_cuda_shm_init(cmd: float, pattern: int) -> bytes:
    b = bytearray(_IQ_CUDA_SHM_BYTES)
    struct.pack_into("d", b, 0, float("nan"))
    struct.pack_into("d", b, 8, float("nan"))
    struct.pack_into("I", b, 16, pattern & 0xFFFFFFFF)
    struct.pack_into("d", b, 24, float(cmd))
    struct.pack_into("Q", b, 32, 0)
    nan = float("nan")
    for off in (40, 48, 56, 64, 72, 80):
        struct.pack_into("d", b, off, nan)
    for off in range(88, _IQ_CUDA_SHM_BYTES):
        b[off] = 0
    return bytes(b)


def _read_capturer_shm_unlocked() -> tuple[int, int, int, int, str, int, int, int]:
    """Returns (total, inflight, written, phase, path, queue_depth, queue_cap, jpeg_threads).

    Phase: 0 idle, 1 capturing, 2 saving. Tail metrics are 0 when SHM is short or idle.
    """
    if _iq_cuda_shm is None or len(_iq_cuda_shm.buf) < _IQ_CUDA_SHM_BYTES:
        return 0, 0, 0, 0, "", 0, 0, 0
    buf = _iq_cuda_shm.buf
    tot, inflight, written, phase = struct.unpack_from("IIII", buf, _SHM_CAPTURER_TOTAL_OFF)
    raw = bytes(buf[_SHM_CAPTURER_PATH_OFF : _SHM_CAPTURER_PATH_OFF + _SHM_CAPTURER_PATH_CAP])
    path = raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    qd, qc, jt = 0, 0, 0
    if len(buf) >= _SHM_CAPTURER_JPEG_THREADS_OFF + 4:
        qd, qc, jt = struct.unpack_from("III", buf, _SHM_CAPTURER_QUEUE_DEPTH_OFF)
    return tot, inflight, written, phase, path, qd, qc, jt


def _arm_capturer_shm_unlocked(out_dir: str, n_frames: int) -> None:
    if _iq_cuda_shm is None or len(_iq_cuda_shm.buf) < _IQ_CUDA_SHM_BYTES:
        raise RuntimeError("CUDA IQ shared memory not available")
    bpath = out_dir.encode("utf-8")
    if len(bpath) >= _SHM_CAPTURER_PATH_CAP:
        raise ValueError("capturer output path exceeds SHM path capacity")
    buf = _iq_cuda_shm.buf
    zerolen = _IQ_CUDA_SHM_BYTES - _SHM_CAPTURER_PATH_OFF
    buf[_SHM_CAPTURER_PATH_OFF : _SHM_CAPTURER_PATH_OFF + zerolen] = b"\x00" * zerolen
    buf[_SHM_CAPTURER_PATH_OFF : _SHM_CAPTURER_PATH_OFF + len(bpath)] = bpath
    struct.pack_into("I", buf, _SHM_CAPTURER_TOTAL_OFF, n_frames & 0xFFFFFFFF)
    struct.pack_into("I", buf, _SHM_CAPTURER_INFLIGHT_OFF, 0)
    struct.pack_into("I", buf, _SHM_CAPTURER_WRITTEN_OFF, 0)
    struct.pack_into("I", buf, _SHM_CAPTURER_PHASE_OFF, 1)
    if len(buf) >= _SHM_CAPTURER_STOP_OFF + 4:
        struct.pack_into("I", buf, _SHM_CAPTURER_STOP_OFF, 0)


def _request_capturer_stop_shm_unlocked() -> None:
    if _iq_cuda_shm is None or len(_iq_cuda_shm.buf) < _IQ_CUDA_SHM_BYTES:
        raise RuntimeError("CUDA IQ shared memory not available")
    buf = _iq_cuda_shm.buf
    if len(buf) < _SHM_CAPTURER_STOP_OFF + 4:
        raise RuntimeError("CUDA IQ shared memory too small for capturer stop flag")
    struct.pack_into("I", buf, _SHM_CAPTURER_STOP_OFF, 1)


def _iq_shm_destroy_unlocked() -> None:
    global _iq_producer_shm, _iq_cuda_shm
    for shm in (_iq_producer_shm, _iq_cuda_shm):
        if shm is None:
            continue
        try:
            shm.close()
        except BufferError:
            pass
        except OSError:
            _log.debug("iq shm close", exc_info=True)
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            _log.debug("iq shm unlink", exc_info=True)
    _iq_producer_shm = None
    _iq_cuda_shm = None


def _iq_producer_shm_create_unlocked(name: str, cmd: float, pattern: int) -> None:
    global _iq_producer_shm
    _iq_producer_shm = shared_memory.SharedMemory(
        create=True, size=_IQ_PRODUCER_SHM_BYTES, name=name
    )
    _iq_producer_shm.buf[:_IQ_PRODUCER_SHM_BYTES] = _pack_producer_shm_init(cmd, pattern)


def _iq_cuda_shm_create_unlocked(name: str, cmd: float, pattern: int) -> None:
    global _iq_cuda_shm
    _iq_cuda_shm = shared_memory.SharedMemory(create=True, size=_IQ_CUDA_SHM_BYTES, name=name)
    _iq_cuda_shm.buf[:_IQ_CUDA_SHM_BYTES] = _pack_cuda_shm_init(cmd, pattern)


def _iq_dual_shm_create_unlocked(
    pattern_override: Optional[int] = None,
    *,
    create_fifo_producer_shm: bool,
) -> tuple[str, Optional[str]]:
    """Create CUDA control SHM; optional POSIX SHM for ``resource_grid_generator.py`` pacing/pattern."""
    _iq_shm_destroy_unlocked()
    ns = time.time_ns()
    cmd = (
        _iq_producer_command_override
        if _iq_producer_command_override is not None
        else _default_iq_producer_command_fps()
    )
    cmd = max(IQ_PRODUCER_FPS_MIN, min(IQ_PRODUCER_FPS_MAX, float(cmd)))
    if pattern_override is not None:
        pattern = _validate_iq_pattern(int(pattern_override))
    else:
        pattern = _effective_iq_pattern_unlocked()
    cuda_name = f"/resgrid_iqcuda_{os.getpid()}_{ns}"
    _iq_cuda_shm_create_unlocked(cuda_name, cmd, pattern)
    prod_name: Optional[str] = None
    if create_fifo_producer_shm:
        prod_name = f"/resgrid_iqprod_{os.getpid()}_{ns}"
        _iq_producer_shm_create_unlocked(prod_name, cmd, pattern)
    return cuda_name, prod_name


def _read_iq_stream_metrics_unlocked() -> tuple[Optional[float], Optional[float], Optional[int]]:
    """Measured producer Hz, Raster output image Hz (SHM offset 8), cumulative published frames (offset 32)."""

    def norm(x: float) -> Optional[float]:
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    prod: Optional[float] = None
    plot: Optional[float] = None
    frames: Optional[int] = None
    if _iq_producer_shm is not None:
        try:
            (pv,) = struct.unpack_from("d", _iq_producer_shm.buf, 0)
            prod = norm(pv)
        except (struct.error, TypeError, ValueError):
            pass
    if _iq_cuda_shm is not None:
        try:
            if _iq_producer_shm is None:
                (pv,) = struct.unpack_from("d", _iq_cuda_shm.buf, 0)
                prod = norm(pv)
            (tv,) = struct.unpack_from("d", _iq_cuda_shm.buf, 8)
            plot = norm(tv)
            if len(_iq_cuda_shm.buf) >= 40:
                (fc,) = struct.unpack_from("Q", _iq_cuda_shm.buf, 32)
                frames = int(fc)
        except (struct.error, TypeError, ValueError):
            pass
    return prod, plot, frames


def _read_cuda_shm_pipeline_stage_rates_unlocked() -> tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Pipeline rates (Hz) from SHM doubles @40…80: asm in/out measured; raster in + encode in duplicate prior stage out (chained telemetry)."""

    def norm(x: float) -> Optional[float]:
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    none6: tuple[Optional[float], ...] = (None, None, None, None, None, None)
    if _iq_cuda_shm is None or len(_iq_cuda_shm.buf) < _IQ_CUDA_SHM_BYTES:
        return none6
    try:
        a_in, a_out, r_in, r_out, e_in, e_out = struct.unpack_from("dddddd", _iq_cuda_shm.buf, 40)
        return (
            norm(a_in),
            norm(a_out),
            norm(r_in),
            norm(r_out),
            norm(e_in),
            norm(e_out),
        )
    except (struct.error, TypeError, ValueError):
        return none6


def _read_iq_producer_command_unlocked() -> float:
    if _iq_producer_shm is not None:
        try:
            (cmd,) = struct.unpack_from("d", _iq_producer_shm.buf, 8)
            if math.isfinite(cmd):
                return max(IQ_PRODUCER_FPS_MIN, min(IQ_PRODUCER_FPS_MAX, float(cmd)))
        except (struct.error, TypeError, ValueError):
            pass
    if _iq_cuda_shm is not None:
        try:
            (cmd,) = struct.unpack_from("d", _iq_cuda_shm.buf, 24)
            if math.isfinite(cmd):
                return max(IQ_PRODUCER_FPS_MIN, min(IQ_PRODUCER_FPS_MAX, float(cmd)))
        except (struct.error, TypeError, ValueError):
            pass
    if _iq_producer_command_override is not None:
        return max(
            IQ_PRODUCER_FPS_MIN,
            min(IQ_PRODUCER_FPS_MAX, float(_iq_producer_command_override)),
        )
    return _default_iq_producer_command_fps()


def _write_iq_producer_command_shm_unlocked(fps: float) -> float:
    global _iq_producer_command_override
    v = _validate_producer_command_fps(fps)
    _iq_producer_command_override = v
    if _iq_producer_shm is not None:
        struct.pack_into("d", _iq_producer_shm.buf, 8, v)
    if _iq_cuda_shm is not None:
        struct.pack_into("d", _iq_cuda_shm.buf, 24, v)
    return v


def _cuda_stderr_drain_loop(proc: subprocess.Popen) -> None:
    """Merge NVENC child stderr (via resgrid_plotter) into shared FFmpeg progress stats."""
    if proc.stderr is None:
        return
    buf = ""
    try:
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            buf = buf.replace("\r", "\n")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if "fps=" in line and "frame=" in line:
                    _apply_progress_line(line)
                else:
                    _log.info("cuda_publisher: %s", line)
        tail = buf.strip()
        if tail:
            if "fps=" in tail and "frame=" in tail:
                _apply_progress_line(tail)
            else:
                _log.info("cuda_publisher: %s", tail)
    except Exception:
        _log.exception("CUDA publisher stderr reader failed")
    finally:
        try:
            proc.stderr.close()
        except OSError:
            pass


def _spawn_cuda_stderr_thread(proc: subprocess.Popen) -> None:
    threading.Thread(target=_cuda_stderr_drain_loop, args=(proc,), daemon=True).start()


def _terminate_ffmpeg_unlocked() -> None:
    global _ffmpeg, _current_lavfi_video, _stats
    if not _is_ffmpeg_running():
        _ffmpeg = None
        _current_lavfi_video = ""
        return
    assert _ffmpeg is not None
    _log.info("Terminating FFmpeg pid=%s", _ffmpeg.pid)
    _ffmpeg.terminate()
    try:
        _ffmpeg.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _ffmpeg.kill()
    _ffmpeg = None
    _current_lavfi_video = ""
    with _stats_lock:
        _stats = GenerationStats()


def _iq_fifo_producer_stderr_loop(proc: subprocess.Popen) -> None:
    if proc.stderr is None:
        return
    try:
        for line in iter(proc.stderr.readline, b""):
            if line:
                txt = line.decode("utf-8", errors="replace").rstrip()
                _log.info("iq_fifo_producer: %s", txt)
    except Exception:
        _log.exception("IQ FIFO generator stderr reader failed")
    finally:
        try:
            proc.stderr.close()
        except OSError:
            pass


def _spawn_iq_fifo_producer_stderr_thread(proc: subprocess.Popen) -> None:
    threading.Thread(target=_iq_fifo_producer_stderr_loop, args=(proc,), daemon=True).start()


def _terminate_iq_fifo_producer_unlocked() -> None:
    global _iq_fifo_producer
    if _iq_fifo_producer is None:
        return
    proc = _iq_fifo_producer
    _iq_fifo_producer = None
    if proc.poll() is not None:
        return
    _log.info("Terminating IQ FIFO generator pid=%s", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _iq_generator_running_unlocked() -> bool:
    global _iq_fifo_producer
    if _iq_fifo_producer is None:
        return False
    return _iq_fifo_producer.poll() is None


def _iq_generator_capable_unlocked() -> bool:
    if not _is_cuda_running():
        return False
    mode = get_stream_mode()
    if mode not in _STREAM_MODES_IQ_CUDA:
        return False
    plot_dm = _cuda_plotter_data_mode(mode)
    if plot_dm == PLOTTER_PIPELINE_SLOT_MESSAGES:
        return True
    return CUDA_IQ_USE_FIFO


def _start_python_iq_producer_unlocked() -> int:
    """Spawn resource_grid_generator.py. Caller must hold _lock; CUDA must be running with producer SHM."""
    global _iq_fifo_producer
    if _iq_fifo_producer is not None and _iq_fifo_producer.poll() is None:
        return _iq_fifo_producer.pid
    if _iq_fifo_producer is not None:
        _iq_fifo_producer = None

    mode = get_stream_mode()
    if mode not in _STREAM_MODES_IQ_CUDA:
        raise RuntimeError("CUDA publisher is not running in an IQ stream mode")
    plot_dm = _cuda_plotter_data_mode(mode)
    prod_dm = _cuda_producer_data_mode(plot_dm)
    use_iq_fifo_pipe = CUDA_IQ_USE_FIFO
    use_slot_fifo = plot_dm == PLOTTER_PIPELINE_SLOT_MESSAGES
    if not use_iq_fifo_pipe and not use_slot_fifo:
        raise RuntimeError("This publisher has no IQ or slot FIFO for the Python generator")
    if _iq_producer_shm is None:
        raise RuntimeError("Producer shared memory missing; restart the publisher")

    producer_shm_name = _iq_producer_shm.name
    fifo_path = CUDA_IQ_FIFO_PATH
    slot_fifo_path = CUDA_SLOT_FIFO_PATH

    if not _IQ_FIFO_GENERATOR_SCRIPT.is_file():
        raise FileNotFoundError(f"resource_grid_generator.py missing: {_IQ_FIFO_GENERATOR_SCRIPT}")

    if use_iq_fifo_pipe:
        py_cmd = [
            sys.executable,
            "-u",
            str(_IQ_FIFO_GENERATOR_SCRIPT),
            "--fifo",
            fifo_path,
            "--data-mode",
            str(prod_dm),
            "--prb",
            CUDA_IQ_PRB,
            "--symbols",
            CUDA_IQ_SYMBOLS,
            "--fps",
            CUDA_IQ_GEN_FPS,
            "--generator-shm",
            producer_shm_name,
        ]
        if use_slot_fifo:
            py_cmd.extend(
                [
                    "--slot-fifo",
                    slot_fifo_path,
                    "--slot-interval",
                    _slot_demo_interval_seconds_str(),
                    "--slot-pattern",
                    _slot_demo_pattern_cli(),
                ]
            )
            _extend_producer_symbol_window_args(py_cmd)
        if IQ_FIFO_PRODUCER_LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR"):
            py_cmd.extend(["--log-level", IQ_FIFO_PRODUCER_LOG_LEVEL])
        try:
            _iq_fifo_producer = subprocess.Popen(
                py_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError:
            _log.exception("Failed to spawn IQ FIFO generator")
            raise
        _spawn_iq_fifo_producer_stderr_thread(_iq_fifo_producer)
        time.sleep(0.15)
        if _iq_fifo_producer.poll() is not None:
            code = _iq_fifo_producer.returncode
            _log.error("IQ FIFO generator exited immediately with %s", code)
            _iq_fifo_producer = None
            raise RuntimeError("IQ FIFO generator failed to start; check server logs")
        return _iq_fifo_producer.pid

    py_cmd = [
        sys.executable,
        "-u",
        str(_IQ_FIFO_GENERATOR_SCRIPT),
        "--slot-only",
        "--data-mode",
        "3",
        "--slot-fifo",
        slot_fifo_path,
        "--fps",
        str(CUDA_IQ_GEN_FPS).strip() or "200",
        "--generator-shm",
        producer_shm_name,
        "--prb",
        CUDA_IQ_PRB,
        "--symbols",
        CUDA_IQ_SYMBOLS,
        "--slot-interval",
        _slot_demo_interval_seconds_str(),
        "--slot-pattern",
        _slot_demo_pattern_cli(),
    ]
    _extend_producer_symbol_window_args(py_cmd)
    if IQ_FIFO_PRODUCER_LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR"):
        py_cmd.extend(["--log-level", IQ_FIFO_PRODUCER_LOG_LEVEL])
    try:
        _iq_fifo_producer = subprocess.Popen(
            py_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError:
        _log.exception("Failed to spawn slot FIFO demo producer")
        raise
    _spawn_iq_fifo_producer_stderr_thread(_iq_fifo_producer)
    time.sleep(0.15)
    if _iq_fifo_producer.poll() is not None:
        code = _iq_fifo_producer.returncode
        _log.error("slot demo producer exited immediately with %s", code)
        _iq_fifo_producer = None
        raise RuntimeError("slot FIFO demo producer failed to start; check server logs")
    return _iq_fifo_producer.pid


def _terminate_cuda_unlocked() -> None:
    global _cuda_iq, _stats, _cuda_spawn_used_slot_fifo, _cuda_plotter_capturer_dir
    _terminate_iq_fifo_producer_unlocked()
    if not _is_cuda_running():
        _cuda_iq = None
        _iq_shm_destroy_unlocked()
        _cuda_plotter_capturer_dir = None
        if _cuda_spawn_used_slot_fifo:
            try:
                os.unlink(CUDA_SLOT_FIFO_PATH)
            except OSError:
                pass
            _cuda_spawn_used_slot_fifo = False
        return
    assert _cuda_iq is not None
    _log.info("Terminating CUDA IQ publisher pid=%s", _cuda_iq.pid)
    _cuda_iq.terminate()
    try:
        _cuda_iq.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _cuda_iq.kill()
    _cuda_iq = None
    _iq_shm_destroy_unlocked()
    _cuda_plotter_capturer_dir = None
    if _cuda_spawn_used_slot_fifo:
        try:
            os.unlink(CUDA_SLOT_FIFO_PATH)
        except OSError:
            pass
        _cuda_spawn_used_slot_fifo = False
    with _stats_lock:
        _stats = GenerationStats()


def _status_lavfi_or_iq() -> Optional[str]:
    sm = get_stream_mode()
    if sm == STREAM_MODE_SYNTHETIC_VIDEO and _current_lavfi_video:
        return _current_lavfi_video
    if _is_iq_cuda_mode(sm) and _is_cuda_running():
        note = f"synthetic IQ {CUDA_IQ_PRB}×{CUDA_IQ_SYMBOLS} @ {CUDA_IQ_FPS}fps (CUDA→NVENC)"
        if sm == STREAM_MODE_CHANNEL_BASED_SYNTHETIC:
            note += ", slot messages / channel-based (PHY tints from slot FIFO)"
        if CUDA_IQ_USE_FIFO and _is_iq_cuda_mode(sm):
            note += f", IQ FIFO {CUDA_IQ_GEN_FPS}Hz"
        if _cuda_plotter_data_mode(sm) == PLOTTER_PIPELINE_SLOT_MESSAGES:
            note += ", slot FIFO"
        return note
    return None


def start_stream(requested_mode: Optional[str] = None, *, plotter_only: bool = False) -> StreamControlResponse:
    global _ffmpeg, _cuda_iq, _iq_fifo_producer, _current_lavfi_video, _stats, _cuda_spawn_used_slot_fifo
    global _cuda_plotter_capturer_dir
    with _lock:
        mode = _resolve_mode_for_start_locked(requested_mode)
        if plotter_only and not _is_iq_cuda_mode(mode):
            raise ValueError(
                "Plotter start requires SYNTHETIC_IQ or SYNTHETIC_CHANNEL_BASED. "
                "Set POST /api/stream/mode first, or pass JSON {\"mode\": \"SYNTHETIC_IQ\"} (or SYNTHETIC_CHANNEL_BASED) on this request."
            )
        if _is_iq_cuda_mode(mode):
            if _is_cuda_running():
                assert _cuda_iq is not None
                return StreamControlResponse(
                    status="already_running",
                    pid=_cuda_iq.pid,
                    stream_mode=mode,
                )
            _terminate_ffmpeg_unlocked()
            _wait_tcp(MTX_HOST, MTX_PORT)
            with _stats_lock:
                _stats = GenerationStats()

            _cuda_spawn_used_slot_fifo = False
            plot_dm = _cuda_plotter_data_mode(mode)
            if CUDA_SLOT_USE_FIFO and plot_dm == PLOTTER_PIPELINE_FULL_GRID_FIFO:
                _log.warning(
                    "CUDA_SLOT_USE_FIFO=1 is ignored for full-grid FIFO (--data-mode 2); "
                    "use SYNTHETIC_CHANNEL_BASED or CUDA_PLOTTER_DATA_MODE=3 for slot messages (channel-based)."
                )
            prod_dm = _cuda_producer_data_mode(plot_dm)

            use_iq_fifo_pipe = CUDA_IQ_USE_FIFO
            if (
                mode == STREAM_MODE_SYNTHETIC_IQ
                and plot_dm == PLOTTER_PIPELINE_FULL_GRID_FIFO
                and not use_iq_fifo_pipe
                and CUDA_PLOTTER_DATA_MODE.strip() != "2"
            ):
                plot_dm = PLOTTER_PIPELINE_INTERNAL_SYNTH

            if use_iq_fifo_pipe and plot_dm == PLOTTER_PIPELINE_INTERNAL_SYNTH:
                raise RuntimeError(
                    "CUDA_IQ_USE_FIFO=1 requires resgrid_plotter --data-mode 2 (full-grid FIFO) or 3 (slot messages). "
                    "Unset CUDA_PLOTTER_DATA_MODE=1 or set CUDA_PLOTTER_DATA_MODE=2."
                )

            if plot_dm not in (
                PLOTTER_PIPELINE_INTERNAL_SYNTH,
                PLOTTER_PIPELINE_FULL_GRID_FIFO,
                PLOTTER_PIPELINE_SLOT_MESSAGES,
            ):
                raise ValueError(
                    f"CUDA_PLOTTER_DATA_MODE must be 1, 2, or 3 (effective {plot_dm})",
                )
            if prod_dm not in (
                PLOTTER_PIPELINE_INTERNAL_SYNTH,
                PLOTTER_PIPELINE_FULL_GRID_FIFO,
                PLOTTER_PIPELINE_SLOT_MESSAGES,
            ):
                raise ValueError(f"CUDA_PRODUCER_DATA_MODE must be 1, 2, or 3 (effective {prod_dm})")

            fifo_path = CUDA_IQ_FIFO_PATH
            if plot_dm == PLOTTER_PIPELINE_FULL_GRID_FIFO and not use_iq_fifo_pipe:
                raise RuntimeError(
                    "resgrid_plotter --data-mode 2 requires CUDA_IQ_USE_FIFO=1 (IQ named pipe + generator). "
                    "For built-in synthetic I/Q without a FIFO, unset CUDA_PLOTTER_DATA_MODE (SYNTHETIC_IQ uses --data-mode 1)."
                )

            use_slot_fifo = plot_dm == PLOTTER_PIPELINE_SLOT_MESSAGES
            # Generator SHM: IQ FIFO path needs pacing/pattern; mode-3 slot-only also needs it so GET/PUT /generator-fps
            # updates the Python slot writer (and matches CUDA SHM @24 for API reads).
            create_fifo_producer_shm = use_iq_fifo_pipe or use_slot_fifo

            if use_iq_fifo_pipe or use_slot_fifo:
                if not _IQ_FIFO_GENERATOR_SCRIPT.is_file():
                    raise FileNotFoundError(f"resource_grid_generator.py missing: {_IQ_FIFO_GENERATOR_SCRIPT}")

            if use_iq_fifo_pipe:
                try:
                    os.unlink(fifo_path)
                except FileNotFoundError:
                    pass
                os.mkfifo(fifo_path, 0o666)
                # umask can turn 0o666 into 0o644; gNB/OAI runs as non-root — need world-writable fifo node.
                os.chmod(fifo_path, 0o666)
                _fifo_apply_chown(fifo_path)

            slot_fifo_path = CUDA_SLOT_FIFO_PATH
            if use_slot_fifo:
                try:
                    os.unlink(slot_fifo_path)
                except FileNotFoundError:
                    pass
                os.mkfifo(slot_fifo_path, 0o666)
                os.chmod(slot_fifo_path, 0o666)
                _fifo_apply_chown(slot_fifo_path)
            _cuda_spawn_used_slot_fifo = use_slot_fifo

            pat_init = None
            cuda_shm_name, producer_shm_name = _iq_dual_shm_create_unlocked(
                pat_init, create_fifo_producer_shm=create_fifo_producer_shm
            )
            _cuda_plotter_capturer_dir = None
            capturer_dir_raw = _env("RESGRID_CAPTURER_DIR", "").strip()
            if not capturer_dir_raw:
                capturer_dir_raw = _env("RESGRID_ENCODER_SNAP_DIR", "").strip()
            capturer_extra: list[str] = []
            if capturer_dir_raw:
                Path(capturer_dir_raw).mkdir(parents=True, exist_ok=True)
                capturer_extra = ["--capturer-dir", capturer_dir_raw]
            cmd = [
                CUDA_PUBLISHER_BIN,
                "--rtsp",
                MTX_RTSP_URL,
                "--fps",
                str(CUDA_IQ_GEN_FPS).strip() or "200",
                "--encoder-fps",
                str(CUDA_IQ_FPS).strip() or "50",
                "--cell-pixels",
                CUDA_IQ_CELL_PIXELS,
                "--preset",
                CUDA_IQ_PRESET,
                "--prb",
                CUDA_IQ_PRB,
                "--symbols",
                CUDA_IQ_SYMBOLS,
                "--mag-scale",
                CUDA_IQ_MAG_SCALE,
                "--queue-depth",
                CUDA_IQ_QUEUE_DEPTH,
                "--raster-threads",
                (CUDA_IQ_RASTER_THREADS or "16").strip() or "16",
                "--data-mode",
                str(plot_dm),
            ]
            if use_iq_fifo_pipe:
                cmd.extend(["--iq-fifo", fifo_path])
            cmd.extend(["--cuda-shm", cuda_shm_name])
            if use_slot_fifo:
                cmd.extend(["--slot-fifo", slot_fifo_path])
            cmd.extend(capturer_extra)
            try:
                _cuda_iq = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except OSError:
                _log.exception("Failed to spawn %s", CUDA_PUBLISHER_BIN)
                _iq_shm_destroy_unlocked()
                if use_iq_fifo_pipe:
                    try:
                        os.unlink(fifo_path)
                    except OSError:
                        pass
                if use_slot_fifo:
                    try:
                        os.unlink(slot_fifo_path)
                    except OSError:
                        pass
                _cuda_spawn_used_slot_fifo = False
                raise
            _spawn_cuda_stderr_thread(_cuda_iq)
            if capturer_dir_raw:
                _cuda_plotter_capturer_dir = os.path.realpath(capturer_dir_raw)

            if (use_iq_fifo_pipe or use_slot_fifo) and not producer_shm_name:
                _terminate_cuda_unlocked()
                raise RuntimeError("internal: missing producer shared memory for IQ/slot FIFO mode")

            if CUDA_AUTO_START_IQ_GENERATOR and (use_iq_fifo_pipe or use_slot_fifo):
                try:
                    _start_python_iq_producer_unlocked()
                except FileNotFoundError:
                    _terminate_cuda_unlocked()
                    if use_iq_fifo_pipe:
                        try:
                            os.unlink(fifo_path)
                        except OSError:
                            pass
                    raise
                except Exception:
                    _log.exception("Failed to spawn IQ / slot FIFO generator")
                    _terminate_cuda_unlocked()
                    if use_iq_fifo_pipe:
                        try:
                            os.unlink(fifo_path)
                        except OSError:
                            pass
                    raise

            _log.info(
                "%s started pid=%s plot_data_mode=%s prod_data_mode=%s iq_fifo_pipe=%s slot_fifo=%s",
                CUDA_PUBLISHER_BIN,
                _cuda_iq.pid,
                plot_dm,
                prod_dm,
                use_iq_fifo_pipe,
                use_slot_fifo,
            )
            return StreamControlResponse(
                status="started",
                pid=_cuda_iq.pid,
                stream_mode=mode,
            )

        if _is_cuda_running():
            _terminate_cuda_unlocked()
        if _is_ffmpeg_running():
            assert _ffmpeg is not None
            _log.info("POST start: already running pid=%s", _ffmpeg.pid)
            return StreamControlResponse(
                status="already_running",
                pid=_ffmpeg.pid,
                lavfi_video=_current_lavfi_video or None,
                stream_mode=get_stream_mode(),
            )
        _wait_tcp(MTX_HOST, MTX_PORT)
        _current_lavfi_video = _lavfi_video()
        cmd = _build_ffmpeg_cmd(_current_lavfi_video)
        with _stats_lock:
            _stats = GenerationStats()
        try:
            _ffmpeg = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
        except OSError:
            _log.exception("Failed to spawn FFmpeg")
            raise
        _spawn_stderr_thread(_ffmpeg)
        _log.info(
            "Generation started pid=%s encoder=%s lavfi=%s",
            _ffmpeg.pid,
            VIDEO_ENCODER,
            _current_lavfi_video,
        )
        return StreamControlResponse(
            status="started",
            pid=_ffmpeg.pid,
            lavfi_video=_current_lavfi_video,
            stream_mode=get_stream_mode(),
        )


def stop_stream(*, plotter_only: bool = False) -> StreamControlResponse:
    global _ffmpeg, _cuda_iq
    with _lock:
        if plotter_only:
            if not _is_cuda_running():
                _log.info("stop_stream(plotter_only): CUDA plotter not running")
                return StreamControlResponse(status="not_running", stream_mode=get_stream_mode())
            assert _cuda_iq is not None
            last_pid = _cuda_iq.pid
            _terminate_cuda_unlocked()
            _log.info("CUDA plotter stopped (pid was %s)", last_pid)
            return StreamControlResponse(status="stopped", pid=last_pid, stream_mode=get_stream_mode())
        if not _is_running():
            _log.info("stop_stream: nothing was running")
            return StreamControlResponse(status="not_running", stream_mode=get_stream_mode())
        last_pid: Optional[int] = None
        if _is_ffmpeg_running():
            assert _ffmpeg is not None
            last_pid = _ffmpeg.pid
            _terminate_ffmpeg_unlocked()
        if _is_cuda_running():
            assert _cuda_iq is not None
            last_pid = _cuda_iq.pid
            _terminate_cuda_unlocked()
        _log.info("Publisher stopped (last pid was %s)", last_pid)
        return StreamControlResponse(status="stopped", pid=last_pid, stream_mode=get_stream_mode())


def _logging_file_settings() -> tuple[int, int, str | None, int, bool]:
    try:
        mx = int(_env("LOG_MAX_BYTES", "5000000"))
        bk = int(_env("LOG_BACKUP_COUNT", "3"))
    except ValueError:
        mx, bk = 5_000_000, 3
    raw_when = _env("LOG_ROTATE_WHEN", "").strip()
    when: str | None = raw_when if raw_when else None
    try:
        interval = int(_env("LOG_ROTATE_INTERVAL", "1"))
    except ValueError:
        interval = 1
    utc = _env("LOG_ROTATE_UTC", "0").strip() == "1"
    return mx, bk, when, interval, utc


@asynccontextmanager
async def lifespan(app: FastAPI):
    mx, bk, rw, ri, ru = _logging_file_settings()
    setup_backend_logging(
        LOG_FILE,
        LOG_LEVEL,
        max_bytes=mx,
        backup_count=bk,
        rotate_when=rw,
        rotate_interval=ri,
        rotate_utc=ru,
    )
    _log.info("Backend starting (AUTO_START=%s STREAM_MODE=%s)", AUTO_START, get_stream_mode())
    try:
        stop_stream()
    except Exception:
        _log.exception("Initial stop_stream failed")
    if AUTO_START:
        try:
            start_stream()
        except Exception:
            _log.exception("AUTO_START failed")
    yield
    _log.info("Backend shutting down; stopping generation if running")
    stop_stream()


def _openapi_description_with_hls(hls_index_url: str) -> str:
    return (
        "Control publication to MediaMTX (RTSP path **`live`**): **SYNTHETIC_VIDEO** (FFmpeg lavfi in this API), "
        "**SYNTHETIC_IQ** (`resgrid_plotter`: built-in synthesizer or full-grid IQ FIFO → NVENC), or **SYNTHETIC_CHANNEL_BASED** (slot messages: "
        "same binary with **slot FIFO always enabled** for PHY channel messages). "
        f"**HLS manifest:** [{hls_index_url}]({hls_index_url}) "
        "(host is taken from **this** request to `/openapi.json`: `Origin` / `Referer`, `X-Forwarded-Host`, `Host`, or the URL host — "
        f"same order as **GET /api/stream/playback**; if none apply, **`{MTX_PUBLIC_HTTP_BASE}`**). "
        "Dual DL/UL stacks use **`live_dl`** / **`live_ul`** — use **GET /api/stream/playback** for this backend’s paths."
    )


def _openapi_tags_with_hls(hls_index_url: str) -> list[dict[str, str]]:
    return [
        {"name": "stream", "description": "Start/stop publisher: FFmpeg (video) or resgrid_plotter (IQ)."},
        {"name": "mode", "description": "Stream mode: (1) synthetic video, (2) full-grid IQ FIFO, (3) slot messages / channel-based."},
        {
            "name": "playback",
            "description": (
                "Subscriber URLs on MediaMTX. Example HLS: "
                f"[{hls_index_url}]({hls_index_url}). Prefer **GET /api/stream/playback**."
            ),
        },
        {"name": "health", "description": "Liveness."},
    ]


app = FastAPI(
    title="MediaMTX stream control",
    description="OpenAPI schema is built per request at [/openapi.json](/openapi.json) so HLS links use your access host (see **/docs**).",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/api/stream/mode",
    response_model=StreamModeInfo,
    tags=["mode"],
    summary="Active stream mode",
)
def api_get_stream_mode():
    m = get_stream_mode()
    return StreamModeInfo(mode=m, note=_mode_note(m))


@app.post(
    "/api/stream/mode",
    response_model=StreamModeInfo,
    tags=["mode"],
    summary="Set stream mode (in-process; compose profile for IQ)",
)
def api_post_stream_mode(body: StreamModeSet):
    try:
        set_stream_mode(body.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    m = get_stream_mode()
    return StreamModeInfo(mode=m, note=_mode_note(m))


@app.get("/health", tags=["health"], summary="Liveness probe")
def health():
    return {"ok": True, "generating": _is_running()}


@app.get(
    "/api/stream/playback",
    response_model=PlaybackUrls,
    tags=["playback"],
    summary="MediaMTX playback URLs for subscribers",
    description=(
        "Returns playback URLs: **webrtc_whep** for real-time WebRTC (lowest latency), and **hls** as a fallback. "
        "The browser reaches MediaMTX directly, not this API. "
        "The **`hls`** host is inferred from **this** request: `Origin` / `Referer`, `X-Forwarded-Host` + `X-Forwarded-Proto`, "
        "`Host`, or the request URL host (then **`:8888`** for MediaMTX HTTP); if none apply, **`MTX_PUBLIC_HTTP_BASE`**."
    ),
)
def api_playback(request: Request):
    return _playback_for_request(request)


def _iq_producer_fps_api_response_unlocked() -> IqProducerFpsResponse:
    return IqProducerFpsResponse(
        fps=_read_iq_producer_command_unlocked(),
        min_fps=IQ_PRODUCER_FPS_MIN,
        max_fps=IQ_PRODUCER_FPS_MAX,
        stream_mode=get_stream_mode(),
        generating=_is_cuda_running(),
        iq_fifo_mode=_iq_fifo_pipe_active_unlocked(),
    )


@app.get(
    "/api/stream/iq/producer-fps",
    response_model=IqProducerFpsResponse,
    tags=["stream"],
    summary="IQ generator FPS (commanded)",
    description=(
        "**Legacy path:** same as GET /api/stream/iq/generator-fps. "
        "Reads commanded generation FPS from IQ control shared memory (when running) or saved override / env default. "
        "Applies to **SYNTHETIC_IQ** and **SYNTHETIC_CHANNEL_BASED** (internal CUDA synthesizer and/or Python FIFO writer)."
    ),
)
@app.get(
    "/api/stream/iq/generator-fps",
    response_model=IqProducerFpsResponse,
    tags=["stream"],
    summary="IQ generator FPS (commanded)",
    description=(
        "Reads commanded generation FPS from IQ control shared memory (when running) or saved override / env default. "
        "Applies to **SYNTHETIC_IQ** and **SYNTHETIC_CHANNEL_BASED** (internal CUDA synthesizer and/or Python FIFO writer)."
    ),
)
def api_iq_producer_fps_get():
    with _lock:
        if not _is_iq_cuda_mode(get_stream_mode()):
            raise HTTPException(
                status_code=400,
                detail="Stream mode must be SYNTHETIC_IQ or SYNTHETIC_CHANNEL_BASED.",
            )
        return _iq_producer_fps_api_response_unlocked()


@app.put(
    "/api/stream/iq/producer-fps",
    response_model=IqProducerFpsResponse,
    tags=["stream"],
    summary="Set IQ generator FPS (commanded)",
    description=(
        "**Legacy path:** same as PUT /api/stream/iq/generator-fps. "
        "Writes **fps** to generator SHM (Python FIFO mode) and CUDA SHM (offset 24, internal synthesizer) when running; "
        "otherwise stores an override applied on the next IQ **Start**."
    ),
)
@app.put(
    "/api/stream/iq/generator-fps",
    response_model=IqProducerFpsResponse,
    tags=["stream"],
    summary="Set IQ generator FPS (commanded)",
    description=(
        "Writes **fps** to generator SHM (Python FIFO mode) and CUDA SHM (offset 24, internal synthesizer) when running; "
        "otherwise stores an override applied on the next IQ **Start**."
    ),
)
def api_iq_producer_fps_put(body: IqProducerFpsPutBody):
    with _lock:
        if not _is_iq_cuda_mode(get_stream_mode()):
            raise HTTPException(
                status_code=400,
                detail="Stream mode must be SYNTHETIC_IQ or SYNTHETIC_CHANNEL_BASED.",
            )
        try:
            v = _write_iq_producer_command_shm_unlocked(body.fps)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _log.info("IQ generator FPS set to %s Hz", v)
        return _iq_producer_fps_api_response_unlocked()


def _generator_start_handler() -> IqGeneratorControlResponse:
    with _lock:
        if _iq_fifo_producer is not None and _iq_fifo_producer.poll() is None:
            return IqGeneratorControlResponse(status="already_running", pid=_iq_fifo_producer.pid)
        if not _iq_generator_capable_unlocked():
            raise HTTPException(
                status_code=400,
                detail="CUDA publisher is not running, or this configuration has no IQ/slot FIFO for the Python generator.",
            )
        try:
            pid = _start_python_iq_producer_unlocked()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return IqGeneratorControlResponse(status="started", pid=pid)


def _generator_stop_handler() -> IqGeneratorControlResponse:
    with _lock:
        if not _iq_generator_running_unlocked():
            return IqGeneratorControlResponse(status="not_running")
        _terminate_iq_fifo_producer_unlocked()
    return IqGeneratorControlResponse(status="stopped")


@app.post(
    "/api/stream/generator/start",
    response_model=IqGeneratorControlResponse,
    tags=["stream"],
    summary="Start Python FIFO generator (dedicated)",
    description=(
        "Same as **POST /api/stream/iq/generator/start**: spawn **resource_grid_generator.py** when **`resgrid_plotter`** "
        "is up with IQ FIFO and/or slot FIFO."
    ),
)
def api_generator_start():
    return _generator_start_handler()


@app.post(
    "/api/stream/generator/stop",
    response_model=IqGeneratorControlResponse,
    tags=["stream"],
    summary="Stop Python FIFO generator (dedicated)",
    description="Same as **POST /api/stream/iq/generator/stop**: stop **resource_grid_generator.py** only.",
)
def api_generator_stop():
    return _generator_stop_handler()


@app.post(
    "/api/stream/iq/generator/start",
    response_model=IqGeneratorControlResponse,
    tags=["stream"],
    summary="Start Python FIFO generator (legacy path)",
    description="Alias of **POST /api/stream/generator/start**.",
)
def api_iq_generator_start():
    return _generator_start_handler()


@app.post(
    "/api/stream/iq/generator/stop",
    response_model=IqGeneratorControlResponse,
    tags=["stream"],
    summary="Stop Python FIFO generator (legacy path)",
    description="Alias of **POST /api/stream/generator/stop**.",
)
def api_iq_generator_stop():
    return _generator_stop_handler()


def _iq_pattern_catalog() -> list[IqPatternInfo]:
    return [
        IqPatternInfo(id=0, label="Waves (default)"),
        IqPatternInfo(id=1, label="Single tone (rotating)"),
        IqPatternInfo(id=2, label="Gaussian blob"),
        IqPatternInfo(id=3, label="Vortex (Gaussian × spiral phase)"),
    ]


@app.get(
    "/api/stream/iq/patterns",
    response_model=IqPatternsResponse,
    tags=["stream"],
    summary="List synthetic IQ pattern ids",
)
def api_iq_patterns_list():
    return IqPatternsResponse(patterns=_iq_pattern_catalog())


@app.get(
    "/api/stream/iq/channel-legend",
    response_model=IqChannelLegendResponse,
    tags=["stream"],
    summary="IQ channel legend",
    description=(
        "Static legend matching **`resgrid_plotter`** CUDA `channel_tint_rgb` "
        "(PHY ids **1…8**, labels, and base sRGB). Does not require an active publisher."
    ),
)
def api_iq_channel_legend():
    return _iq_channel_legend()


@app.put(
    "/api/stream/iq/pattern",
    response_model=IqPatternResponse,
    tags=["stream"],
    summary="Set synthetic IQ pattern",
    description=(
        "Writes pattern id to **generator** and **CUDA** control shared memory (uint32 @ offset 16). "
        "**SYNTHETIC_IQ** only (**built-in** or **FIFO** path). Not used in **SYNTHETIC_CHANNEL_BASED** (mode 3)."
    ),
)
def api_iq_pattern_put(body: IqPatternPutBody):
    with _lock:
        sm = get_stream_mode()
        if not _is_iq_cuda_mode(sm):
            raise HTTPException(
                status_code=400,
                detail="Stream mode must be SYNTHETIC_IQ or SYNTHETIC_CHANNEL_BASED (POST /api/stream/mode).",
            )
        if _is_stream_mode_channel_based(sm):
            raise HTTPException(
                status_code=400,
                detail="Synthetic IQ pattern applies only to SYNTHETIC_IQ, not SYNTHETIC_CHANNEL_BASED (slot messages / mode 3).",
            )
        try:
            v = _write_iq_pattern_shm_unlocked(body.pattern)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _log.info("IQ pattern set to %s", v)
        return IqPatternResponse(pattern=v)


@app.get(
    "/api/stream/status",
    response_model=StreamStatus,
    tags=["stream"],
    summary="Video generation status",
    description=(
        "Whether the backend is generating for MediaMTX, plus encoder/source details. "
        "SYNTHETIC_VIDEO: FFmpeg stderr progress in stats (fps, speed, frames, bitrate). "
        "IQ CUDA: **stats.fps** = FFmpeg progress **fps=** (wall-clock RGBA frames/s **into** NVENC, ~`--encoder-fps`); "
        "if missing, falls back to configured **CUDA_IQ_FPS**. **stats.frame** = encoded frame count from FFmpeg when "
        "present, else SHM counter (offset 32). **iq_plotter_fps** = **Raster** output images/s (SHM offset 8). "
        "Stage rates (**assembly_*** … **encoder_output_fps**) come from CUDA SHM offsets 40…80 when the plotter is new enough. "
        "For a linear pipeline view, use **assembly_output_slot_s** → **raster_output_image_s** → **encoder_output_fps** as each stage’s "
        "**output** rate. **raster_input_slot_s** and **encoder_input_image_s** duplicate the **previous** stage’s output (SHM **56**=**48**, **72**=**64**); **assembly_input_msg_s** (SHM **40**) remains the measured external message rate into **Assembly**."
    ),
)
def api_stream_status(request: Request):
    mode = get_stream_mode()
    iq_prod: Optional[float] = None
    iq_plot: Optional[float] = None
    iq_cmd: Optional[float] = None
    iq_pat: Optional[int] = None
    asm_in: Optional[float] = None
    asm_out: Optional[float] = None
    rast_in: Optional[float] = None
    rast_out: Optional[float] = None
    enc_in: Optional[float] = None
    enc_out: Optional[float] = None
    if mode == "SYNTHETIC_VIDEO":
        gen = _is_ffmpeg_running()
        enc = VIDEO_ENCODER
        with _stats_lock:
            stats_snap = _stats.model_copy() if gen else None
    else:
        gen = _is_cuda_running()
        enc = "cuda_iq_pipeline"
        stats_snap = None
        iq_cmd = _read_iq_producer_command_unlocked()
        iq_pat = _read_iq_pattern_from_shm_unlocked()
        if _is_stream_mode_channel_based(mode):
            iq_pat = None
        if gen:
            iq_prod, iq_plot, iq_pub_frames = _read_iq_stream_metrics_unlocked()
            (
                asm_in,
                asm_out,
                rast_in,
                rast_out,
                enc_in,
                enc_out,
            ) = _read_cuda_shm_pipeline_stage_rates_unlocked()
            with _stats_lock:
                enc_fps = _stats.fps
                enc_frame = _stats.frame
                enc_speed = _stats.speed
                enc_bitrate = _stats.bitrate_kbits
            out_fps = (
                enc_fps
                if enc_fps is not None and math.isfinite(enc_fps)
                else None
            )
            if out_fps is None and gen:
                try:
                    out_fps = float(str(CUDA_IQ_FPS).strip())
                except ValueError:
                    out_fps = 50.0
            out_frame = enc_frame if enc_frame is not None else iq_pub_frames
            if out_fps is not None or out_frame is not None:
                stats_snap = GenerationStats(
                    fps=out_fps,
                    speed=enc_speed,
                    frame=out_frame,
                    bitrate_kbits=enc_bitrate,
                )
    iq_gen_running = False
    iq_gen_capable = False
    if mode != "SYNTHETIC_VIDEO" and gen:
        with _lock:
            iq_gen_running = _iq_generator_running_unlocked()
            iq_gen_capable = _iq_generator_capable_unlocked()
    cap_capable = _cuda_plotter_capturer_dir is not None
    cap_active: Optional[bool] = None
    cap_inflight: Optional[int] = None
    cap_written: Optional[int] = None
    cap_phase: Optional[str] = None
    cap_total: Optional[int] = None
    cap_path: Optional[str] = None
    cap_qd: Optional[int] = None
    cap_qc: Optional[int] = None
    cap_jt: Optional[int] = None
    if mode != "SYNTHETIC_VIDEO" and gen and _iq_cuda_shm is not None and len(_iq_cuda_shm.buf) >= _IQ_CUDA_SHM_BYTES:
        tot, inflight, written, phase, p, qd, qc, jt = _read_capturer_shm_unlocked()
        cap_total = int(tot)
        cap_inflight = int(inflight)
        cap_written = int(written)
        ph = int(phase)
        cap_phase = {0: "idle", 1: "capturing", 2: "saving"}.get(ph, "idle")
        cap_active = ph in (1, 2)
        cap_path = p if p else None
        cap_qd = int(qd)
        cap_qc = int(qc)
        cap_jt = int(jt)
    peer = request.client.host if request.client else "-"
    iq_prb, iq_symbols = _iq_grid_dims()
    _log.info(
        "GET /api/stream/status client=%s generating=%s stream_mode=%s",
        peer,
        gen,
        mode,
        extra={"status_poll": True},
    )
    return StreamStatus(
        stream_mode=mode,
        generating=gen,
        video_encoder=enc,
        lavfi_video=_status_lavfi_or_iq(),
        stats=stats_snap,
        publish_rtsp=MTX_RTSP_URL,
        mtx_path=_mtx_path,
        iq_prb=iq_prb,
        iq_symbols=iq_symbols,
        iq_producer_fps=iq_prod,
        iq_plotter_fps=iq_plot,
        iq_producer_command_fps=iq_cmd,
        iq_pattern=iq_pat,
        iq_generator_running=iq_gen_running,
        iq_generator_capable=iq_gen_capable,
        assembly_input_msg_s=asm_in,
        assembly_output_slot_s=asm_out,
        raster_input_slot_s=rast_in,
        raster_output_image_s=rast_out,
        encoder_input_image_s=enc_in,
        encoder_output_fps=enc_out,
        capturer_capable=cap_capable,
        capturer_active=cap_active,
        capturer_phase=cap_phase,
        capturer_inflight=cap_inflight,
        capturer_written=cap_written,
        capturer_total=cap_total,
        capturer_out_dir=cap_path,
        capturer_queue_depth=cap_qd,
        capturer_queue_cap=cap_qc,
        capturer_jpeg_threads=cap_jt,
    )


@app.post(
    "/api/stream/start",
    response_model=StreamControlResponse,
    tags=["stream"],
    summary="Start publisher",
    description=(
        "SYNTHETIC_VIDEO: spawn FFmpeg lavfi. IQ CUDA: spawn `resgrid_plotter` for **SYNTHETIC_IQ** "
        "(default **--data-mode 1** built-in synthetic grids; **CUDA_IQ_USE_FIFO=1** uses **--data-mode 2** + IQ FIFO) or "
        "**SYNTHETIC_CHANNEL_BASED** (**--data-mode 3**). Channel-based **creates** the slot FIFO and passes "
        "`--slot-fifo` (PHY colors from slot allocation messages). "
        "**resource_grid_generator.py** starts only if **CUDA_AUTO_START_IQ_GENERATOR=1**; else use "
        "**POST /api/stream/generator/start** (or **/api/stream/iq/generator/start**). "
        "For **`resgrid_plotter`** without using this combined endpoint, see **POST /api/stream/plotter/start**. "
        "Optional JSON body `{\"mode\":\"...\"}` sets the active mode before starting; "
        "if the other publisher was running it is stopped first."
    ),
)
def api_stream_start(body: StreamStartBody = StreamStartBody()):
    try:
        return start_stream(body.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/api/stream/stop",
    response_model=StreamControlResponse,
    tags=["stream"],
    summary="Stop publisher",
    description=(
        "Stop FFmpeg or resgrid_plotter, whichever is running. "
        "Optional JSON body may include `mode` for client symmetry; it is ignored."
    ),
)
def api_stream_stop(body: StreamStopBody = StreamStopBody()):
    return stop_stream()


@app.post(
    "/api/stream/plotter/start",
    response_model=StreamControlResponse,
    tags=["stream"],
    summary="Start CUDA plotter only",
    description=(
        "Start **`resgrid_plotter`** (NVENC → RTSP) for **SYNTHETIC_IQ** or **SYNTHETIC_CHANNEL_BASED** only. "
        "Returns **400** if the active stream mode is **SYNTHETIC_VIDEO** (use **POST /api/stream/start** for lavfi). "
        "Optional JSON `{\"mode\": \"SYNTHETIC_IQ\"}` (or **SYNTHETIC_CHANNEL_BASED**) applies **POST /api/stream/mode** semantics "
        "then starts the plotter. Stops FFmpeg first if it was running."
    ),
)
def api_plotter_start(body: StreamStartBody = StreamStartBody()):
    try:
        return start_stream(body.mode, plotter_only=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/api/stream/plotter/stop",
    response_model=StreamControlResponse,
    tags=["stream"],
    summary="Stop CUDA plotter only",
    description=(
        "Terminate **`resgrid_plotter`** only. Does **not** stop synthetic **video** (FFmpeg lavfi); use **POST /api/stream/stop** "
        "to stop all publishers."
    ),
)
def api_plotter_stop(body: StreamStopBody = StreamStopBody()):
    return stop_stream(plotter_only=True)


@app.post(
    "/api/stream/plotter/capturer",
    response_model=CapturerStartResponse,
    tags=["stream"],
    summary="Capture raster frames to JPEG files (parallel to video encoder)",
    description=(
        "Arms the **file capturer** in **`resgrid_plotter`** via CUDA IQ SHM: each **raster** RGBA is pushed to the **video encoder** queue as usual and, in parallel, to a **bounded capturer queue** consumed by JPEG writer threads (**`RESGRID_CAPTURE_JPEG_THREADS`**) writing **`frame_*.jpg`**. "
        "If the capturer queue is full, the **oldest** queued RGBA is **dropped** so the raster never blocks (Swap / backlog). "
        "Progress: **capturer_inflight** / **capturer_written**. Requires **`RESGRID_CAPTURER_DIR`** (legacy **`RESGRID_ENCODER_SNAP_DIR`**). Max: **`RESGRID_CAPTURER_MAX_FRAMES`**."
    ),
)
def api_plotter_capturer(body: CapturerStartBody):
    with _lock:
        if not _is_cuda_running():
            raise HTTPException(
                status_code=503,
                detail="CUDA plotter is not running.",
            )
        if _cuda_plotter_capturer_dir is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "File capturer is not configured. Set environment RESGRID_CAPTURER_DIR "
                    "(or legacy RESGRID_ENCODER_SNAP_DIR) to a writable directory and restart the plotter."
                ),
            )
        if body.frames > _CAPTURER_MAX_FRAMES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"frames exceeds RESGRID_CAPTURER_MAX_FRAMES "
                    f"({_CAPTURER_MAX_FRAMES}); raise the env var or request fewer frames."
                ),
            )
        tot_r, _infl_r, written_r, phase, _p, _qd, _qc, _jt = _read_capturer_shm_unlocked()
        ph = int(phase)
        # Plotter may leave phase "saving" (2) if a JPEG worker exited via queue timeout after the
        # last write — SHM never returns to idle and re-arm would 409 forever. Coerce idle when done.
        if (
            ph in (1, 2)
            and tot_r > 0
            and written_r >= tot_r
            and _iq_cuda_shm is not None
            and len(_iq_cuda_shm.buf) >= _SHM_CAPTURER_PHASE_OFF + 4
        ):
            struct.pack_into("I", _iq_cuda_shm.buf, _SHM_CAPTURER_PHASE_OFF, 0)
            ph = 0
        if ph in (1, 2):
            raise HTTPException(status_code=409, detail="Capturer session already in progress.")
        assert _cuda_plotter_capturer_dir is not None
        base = Path(_cuda_plotter_capturer_dir)
        out = base / f"capture_{int(time.time() * 1000)}"
        try:
            out.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Cannot create capture directory: {e}") from e
        out_abs = str(out.resolve())
        try:
            _arm_capturer_shm_unlocked(out_abs, body.frames)
        except ValueError as e:
            try:
                out.rmdir()
            except OSError:
                pass
            raise HTTPException(status_code=400, detail=str(e)) from e
        _log.info("Capturer armed: %s frames -> %s", body.frames, out_abs)
        return CapturerStartResponse(out_dir=out_abs, frames=body.frames)


@app.post(
    "/api/stream/plotter/capturer/stop",
    response_model=CapturerStopResponse,
    tags=["stream"],
    summary="Stop file capturer early",
    description=(
        "Requests the plotter to **stop capturing** after frames already taken from the raster: "
        "SHM **total** is capped to the current push count, queued RGBA is flushed to JPEG, then the session ends. "
        "No-op if not in **capturing** phase (returns **409** during **saving** or **idle**)."
    ),
)
def api_plotter_capturer_stop():
    with _lock:
        if not _is_cuda_running():
            raise HTTPException(
                status_code=503,
                detail="CUDA plotter is not running.",
            )
        if _cuda_plotter_capturer_dir is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "File capturer is not configured. Set environment RESGRID_CAPTURER_DIR "
                    "(or legacy RESGRID_ENCODER_SNAP_DIR) to a writable directory and restart the plotter."
                ),
            )
        _t, _r, _d, phase, _p, _qd, _qc, _jt = _read_capturer_shm_unlocked()
        ph = int(phase)
        if ph != 1:
            raise HTTPException(
                status_code=409,
                detail="Capturer is not capturing (idle or already saving); stop not applicable.",
            )
        _request_capturer_stop_shm_unlocked()
    _log.info("Capturer stop requested via SHM")
    return CapturerStopResponse(status="stop_requested")


def _generate_openapi_schema(request: Request) -> dict:
    base = _mtx_http_base_from_request(request)
    hls_index = f"{base.rstrip('/')}/{_mtx_path}/index.m3u8"
    return get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        summary=getattr(app, "summary", None),
        description=_openapi_description_with_hls(hls_index),
        routes=app.routes,
        tags=_openapi_tags_with_hls(hls_index),
        servers=[{"url": "/", "description": "This API"}],
    )


@app.get("/openapi.json", include_in_schema=False)
def openapi_endpoint(request: Request):
    return _generate_openapi_schema(request)


@app.get("/docs", include_in_schema=False)
async def swagger_ui_route():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Swagger UI",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_ui_route():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - ReDoc",
    )
