import json
import math
import os
import re
import shlex
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from multiprocessing import shared_memory
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .access_log import HttpAccessLogMiddleware
from .log_config import get_logger, setup_backend_logging


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


MTX_RTSP_URL = _env("MTX_RTSP_URL", "rtsp://mediamtx:8554/live")
MTX_HOST = _env("MTX_WAIT_HOST", "mediamtx")
MTX_PORT = int(_env("MTX_WAIT_PORT", "8554"))
NVENC_PRESET = _env("NVENC_PRESET", "p4")
VIDEO_ENCODER = _env("VIDEO_ENCODER", "cpu").lower()
# Idle on boot unless AUTO_START=1. Lifespan calls stop_plotter_and_generators() first to clear stray plotter / generator children.
AUTO_START = _env("AUTO_START", "0") == "1"
# **resgrid_plotter** path: **RESGRID_PLOTTER_BIN** preferred; **CUDA_PUBLISHER_BIN** retained for existing compose.
PLOTTER_BIN = (_env("RESGRID_PLOTTER_BIN", "") or _env("CUDA_PUBLISHER_BIN", "resgrid_plotter")).strip() or "resgrid_plotter"
# NVENC / stdin pacing (--encoder-fps). Default 24 Hz output; set CUDA_IQ_FPS to override.
CUDA_IQ_FPS = _env("CUDA_IQ_FPS", "24")
CUDA_IQ_CELL_PIXELS = _env("CUDA_IQ_CELL_PIXELS", "6")
CUDA_IQ_PRESET = _env("CUDA_IQ_PRESET", NVENC_PRESET)
CUDA_IQ_PRB = _env("CUDA_IQ_PRB", "273")
CUDA_IQ_SYMBOLS = _env("CUDA_IQ_SYMBOLS", "14")
CUDA_IQ_MAG_SCALE = _env("CUDA_IQ_MAG_SCALE", "1.0")
CUDA_IQ_QUEUE_DEPTH = _env("CUDA_IQ_QUEUE_DEPTH", "32")
CUDA_IQ_RASTER_THREADS = _env("CUDA_IQ_RASTER_THREADS", "16")
# When 1, auto-spawn **generator.py** after the plotter starts. Default 0: use **POST /api/generator/start** with **`feed`**.
CUDA_AUTO_START_IQ_GENERATOR = _env("CUDA_AUTO_START_IQ_GENERATOR", "0") == "1"
# **--fifo-resource-grid** + **--fifo-fapi** + **--fifo-images** are always part of **resgrid_plotter** argv (paths from env; **CUDA_IQ_USE_FIFO** is ignored).
CUDA_IQ_USE_FIFO = True
# Default **--fifo-resource-grid** path (matches **resgrid_plotter** ``kDefaultFifoResourceGridPath``).
_DEFAULT_CUDA_IQ_FIFO_PATH = "/tmp/resource_grid.fifo"
CUDA_IQ_FIFO_PATH = _env("CUDA_IQ_FIFO_PATH", _DEFAULT_CUDA_IQ_FIFO_PATH)
# Default slot / FAPI named pipe (matches **resgrid_plotter** default ``kDefaultFifoFapiPath``).
_DEFAULT_CUDA_SLOT_FIFO_PATH = "/tmp/fapi.fifo"
# Raw RGBA named pipe for **resgrid_plotter** ``--fifo-images`` — finalized after **GENERATION_MODE** (see below).
_RESGRID_FIFO_IMAGES_RAW = (os.environ.get("RESGRID_FIFO_IMAGES") or "").strip()
CUDA_IQ_GEN_FPS = _env("CUDA_IQ_GEN_FPS", "200")
CUDA_SLOT_USE_FIFO = _env("CUDA_SLOT_USE_FIFO", "0") == "1"
CUDA_SLOT_FIFO_PATH = _env("CUDA_SLOT_FIFO_PATH", _DEFAULT_CUDA_SLOT_FIFO_PATH)
# Seconds between demo slot records, or "fps"/"sync" (default) → 1/CUDA_IQ_FPS to match plotter frame rate (mode 3).
CUDA_SLOT_DEMO_INTERVAL = _env("CUDA_SLOT_DEMO_INTERVAL", "fps")
# generator.py --slot-pattern: demo | demo-multi | symbol-columns | symbol-windows (latter: one v1 msg per OFDM symbol per slot, random PRB windows at process start).
CUDA_SLOT_DEMO_PATTERN = _env("CUDA_SLOT_DEMO_PATTERN", "symbol-windows").strip().lower()
CUDA_SLOT_WINDOW_MIN_PRB = _env("CUDA_SLOT_WINDOW_MIN_PRB", "50").strip()
# Optional int seed for symbol-windows PRB spans; empty = nondeterministic (passed as --slot-window-seed).
CUDA_SLOT_WINDOW_SEED = _env("CUDA_SLOT_WINDOW_SEED", "").strip()
# Optional int seed for per-slot/per-symbol random RNTI (symbol-windows, symbol-columns); empty = nondeterministic.
CUDA_SLOT_RNTI_SEED = _env("CUDA_SLOT_RNTI_SEED", "").strip()
# Passed to generator.py as --log-level when set (e.g. DEBUG). Empty = script default INFO.
IQ_FIFO_PRODUCER_LOG_LEVEL = _env("IQ_FIFO_PRODUCER_LOG_LEVEL", "").strip().upper()


def _plotter_padded_rgba_size() -> tuple[int, int]:
    """Match ``resgrid_plotter`` square canvas + NVENC min side + 16-align (no letterbox), for ``--fifo-images`` frame size."""
    try:
        prb = max(1, int(str(CUDA_IQ_PRB).strip()))
        sym = max(1, int(str(CUDA_IQ_SYMBOLS).strip()))
        cell = max(1, int(str(CUDA_IQ_CELL_PIXELS).strip()))
    except ValueError:
        prb, sym, cell = 273, 14, 6
    side = max(prb * cell, sym * cell)
    if side % 2:
        side += 1
    fw = fh = side
    need = max(0, 128 - fw)
    fw += need
    fh += need

    def align16(x: int) -> int:
        return (x + 15) & ~15

    return align16(fw), align16(fh)


def _image_pattern_from_resolved_lavfi(graph: str) -> str:
    g = graph.lower()
    if "randomnoise" in g:
        return "randomnoise"
    if "smptebars" in g:
        return "smptebars"
    if "mandelbrot" in g:
        return "mandelbrot"
    return "testsrc"

IQ_PRODUCER_FPS_MIN = float(_env("IQ_PRODUCER_FPS_MIN", "1"))
IQ_PRODUCER_FPS_MAX = float(_env("IQ_PRODUCER_FPS_MAX", "600"))
# CLI limit aligned with ``resgrid_plotter`` ``kEncoderFpsHardMax`` (``--encoder-fps``).
PLOTTER_ENCODER_FPS_CMD_MAX = 50

# POST /api/plotter/capturer: file capturer buffers RGBA in RAM then JPEG workers flush to disk (parallel to video encoder).
_cm = _env("RESGRID_CAPTURER_MAX_FRAMES", "").strip()
if not _cm:
    _cm = _env("RESGRID_ENCODER_CAPTURE_MAX_FRAMES", "20000")
_CAPTURER_MAX_FRAMES = max(1, int(_cm))

# Synthetic resource-grid pattern id (must match generator.py and plotter `kIqPatternMax` in plotter_common.cuh).
IQ_PATTERN_DEFAULT = 0
IQ_PATTERN_SINGLE_TONE = 1
IQ_PATTERN_GAUSSIAN = 2
IQ_PATTERN_VORTEX = 3
IQ_PATTERN_MAX = IQ_PATTERN_VORTEX

_generator_producer_fps_override: Optional[float] = None
_synthetic_pattern_override: Optional[int] = None


def _plotter_resource_grid_dims() -> tuple[int, int]:
    try:
        p = int(str(CUDA_IQ_PRB).strip())
    except ValueError:
        p = 273
    try:
        s = int(str(CUDA_IQ_SYMBOLS).strip())
    except ValueError:
        s = 14
    return max(1, p), max(1, s)


def _encoder_hz_for_slot_interval(*, encoder_hz_override: Optional[float] = None) -> float:
    if encoder_hz_override is not None:
        return max(1.0, float(encoder_hz_override))
    try:
        return max(1.0, float(str(CUDA_IQ_FPS).strip()))
    except ValueError:
        return 24.0


def _slot_demo_interval_seconds_str(*, encoder_hz_override: Optional[float] = None) -> str:
    """String float for generator.py ``--slot-interval``: wall seconds between slot demo records.

    ``CUDA_SLOT_DEMO_INTERVAL`` of ``fps``, ``sync``, ``iq_fps``, or empty uses ``1/max(1, encoder_hz)``
    so PHY/channel demo updates once per plotter frame (encoder_hz from ``encoder_hz_override``, else
    ``CUDA_IQ_FPS``, else 24). Any positive float is used as literal seconds.
    Non-positive numeric values fall back to the fps-derived interval.
    """
    raw = str(CUDA_SLOT_DEMO_INTERVAL).strip().lower()
    if raw in ("fps", "sync", "iq_fps", ""):
        hz = _encoder_hz_for_slot_interval(encoder_hz_override=encoder_hz_override)
        return f"{1.0 / hz:.9g}"
    try:
        v = float(str(CUDA_SLOT_DEMO_INTERVAL).strip())
    except ValueError:
        hz = _encoder_hz_for_slot_interval(encoder_hz_override=encoder_hz_override)
        return f"{1.0 / hz:.9g}"
    if v <= 0.0:
        hz = _encoder_hz_for_slot_interval(encoder_hz_override=encoder_hz_override)
        return f"{1.0 / hz:.9g}"
    return f"{v:.9g}"


def _plotter_spawn_env_for_log() -> dict[str, str]:
    """Env keys that influence or document ``resgrid_plotter`` (values as seen by the child process)."""
    keys = (
        "GENERATION_MODE",
        "RESGRID_PLOTTER_BIN",
        "CUDA_PUBLISHER_BIN",
        "MTX_RTSP_URL",
        "MTX_WAIT_HOST",
        "MTX_WAIT_PORT",
        "CUDA_IQ_FPS",
        "CUDA_IQ_GEN_FPS",
        "CUDA_IQ_USE_FIFO",
        "CUDA_IQ_FIFO_PATH",
        "CUDA_SLOT_USE_FIFO",
        "CUDA_SLOT_FIFO_PATH",
        "CUDA_IQ_CELL_PIXELS",
        "CUDA_IQ_PRESET",
        "CUDA_IQ_PRB",
        "CUDA_IQ_SYMBOLS",
        "CUDA_IQ_MAG_SCALE",
        "CUDA_IQ_QUEUE_DEPTH",
        "CUDA_IQ_RASTER_THREADS",
        "VIDEO_ENCODER",
        "CUDA_AUTO_START_IQ_GENERATOR",
        "RESGRID_FIFO_IMAGES",
        "RESGRID_CAPTURER_DIR",
        "RESGRID_ENCODER_SNAP_DIR",
        "RESGRID_CAPTURE_JPEG_THREADS",
        "CUDA_SLOT_DEMO_INTERVAL",
        "CUDA_SLOT_DEMO_PATTERN",
        "CUDA_SLOT_WINDOW_MIN_PRB",
        "CUDA_SLOT_WINDOW_SEED",
        "CUDA_SLOT_RNTI_SEED",
        "IQ_FIFO_PRODUCER_LOG_LEVEL",
    )
    return {k: (os.environ.get(k) if os.environ.get(k) is not None else "") for k in keys}


def _coerce_plotter_encoder_fps_cmd(encoder_fps: Optional[float]) -> int:
    """Return validated ``--encoder-fps`` for ``resgrid_plotter`` (pipeline ``--fps`` stays **CUDA_IQ_GEN_FPS**)."""
    try:
        base_e = float(str(CUDA_IQ_FPS).strip() or "24")
    except ValueError:
        base_e = 24.0
    ei = int(round(encoder_fps)) if encoder_fps is not None else int(round(base_e))
    if ei < 1 or ei > PLOTTER_ENCODER_FPS_CMD_MAX:
        raise ValueError(
            f"encoder_fps must be between 1 and {PLOTTER_ENCODER_FPS_CMD_MAX} (got {ei})",
        )
    return ei


def _effective_fifo_images_generator_fps() -> int:
    """RGBA FIFO writer rate: match last plotter ``--encoder-fps`` when CUDA is up, else ``CUDA_IQ_FPS``."""
    if _is_plotter_running() and _cuda_spawn_encoder_fps is not None:
        return max(1, _cuda_spawn_encoder_fps)
    try:
        return max(1, int(float(str(CUDA_IQ_FPS).strip() or "24")))
    except ValueError:
        return 24


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

try:
    _NVENC_PROBE_TIMEOUT_S = float(_env("NVENC_PROBE_TIMEOUT_SEC", "20"))
except ValueError:
    _NVENC_PROBE_TIMEOUT_S = 20.0
_NVENC_PROBE_TIMEOUT_S = max(1.0, _NVENC_PROBE_TIMEOUT_S)
_EFFECTIVE_VIDEO_ENCODER: Optional[str] = None


def _ffmpeg_nvenc_probe_succeeds() -> bool:
    """True if FFmpeg can open h264_nvenc and encode at least one frame (matches runtime lavfi/plotter path)."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=0.05:size=128x128:rate=10",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-preset",
        NVENC_PRESET,
        "-f",
        "null",
        "-",
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1.0, _NVENC_PROBE_TIMEOUT_S),
            stdin=subprocess.DEVNULL,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _init_effective_video_encoder() -> None:
    """Resolve nvenc vs cpu once; warn if NVENC was requested but probe failed."""
    global _EFFECTIVE_VIDEO_ENCODER
    if _EFFECTIVE_VIDEO_ENCODER is not None:
        return
    if VIDEO_ENCODER != "nvenc":
        _EFFECTIVE_VIDEO_ENCODER = VIDEO_ENCODER
        return
    if _ffmpeg_nvenc_probe_succeeds():
        _EFFECTIVE_VIDEO_ENCODER = "nvenc"
        _log.info("NVENC probe OK; using h264_nvenc for FFmpeg lavfi and resgrid_plotter.")
        return
    _EFFECTIVE_VIDEO_ENCODER = "cpu"
    _log.warning(
        "VIDEO_ENCODER=nvenc but FFmpeg cannot use h264_nvenc on this host (one-frame probe failed). "
        "Falling back to libx264 for lavfi and resgrid_plotter. Common on GPUs without NVENC (e.g. many H100 SKUs per "
        "NVIDIA Video Encode and Decode GPU Support Matrix)."
    )


def effective_video_encoder() -> str:
    """Encoder for lavfi RTSP and resgrid_plotter: ``nvenc`` or ``cpu`` (libx264), after NVENC probe when requested."""
    _init_effective_video_encoder()
    assert _EFFECTIVE_VIDEO_ENCODER is not None
    return _EFFECTIVE_VIDEO_ENCODER


def _fifo_chown_target() -> tuple[int, int] | None:
    """If set, chown resource-grid / FAPI FIFOs after mkfifo so host-mounted pipes match sibling containers (e.g. OAI uid 1000)."""
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


def _close_resource_grid_fifo_peer_hold_unlocked() -> None:
    """Release the plotter-bootstrap **O_RDWR** hold on the resource-grid FIFO (caller holds ``_lock``)."""
    global _resource_grid_fifo_peer_fd
    fd = _resource_grid_fifo_peer_fd
    _resource_grid_fifo_peer_fd = None
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def _open_resource_grid_fifo_peer_for_plotter_unlocked(path: str) -> None:
    """Open **path** **O_RDWR** so **resgrid_plotter** can open the read end without **generator.py** yet (Linux FIFO rules)."""
    global _resource_grid_fifo_peer_fd
    _close_resource_grid_fifo_peer_hold_unlocked()
    try:
        _resource_grid_fifo_peer_fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError as e:
        _log.error("resource-grid FIFO peer open O_RDWR failed on %s: %s", path, e)
        raise RuntimeError(
            f"cannot open resource-grid FIFO {path!r} O_RDWR for plotter bootstrap (see server log): {e}"
        ) from e


def _recreate_plotter_fifo_node(path: str) -> None:
    """(Re)create a world-writable named pipe for **resgrid_plotter** / **generator.py** attach."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    os.mkfifo(path, 0o666)
    os.chmod(path, 0o666)
    _fifo_apply_chown(path)


def _append_plotter_fifo_cli_flag(cmd: list[str], flag: str, path: str, default_path: str) -> None:
    """Append **resgrid_plotter** FIFO argv: flag alone when **path** matches the binary default, else **flag path**."""
    if os.path.normpath(path) == os.path.normpath(default_path):
        cmd.append(flag)
    else:
        cmd.extend([flag, path])


def _append_plotter_input_fifo_and_cuda_shm(
    cmd: list[str],
    *,
    cuda_shm_name: str,
    fifo_images_path: str,
    default_fifo_images_path: str,
    resource_grid_path: Optional[str],
    slot_fifo_path: Optional[str],
) -> None:
    """Append **--fifo-images**; then either RGBA-only (**--cuda-shm**) or full stack (**--fifo-resource-grid** + **--cuda-shm** + **--fifo-fapi**)."""
    _append_plotter_fifo_cli_flag(cmd, "--fifo-images", fifo_images_path, default_fifo_images_path)
    if resource_grid_path is None:
        cmd.extend(["--cuda-shm", cuda_shm_name])
        return
    _append_plotter_fifo_cli_flag(cmd, "--fifo-resource-grid", resource_grid_path, _DEFAULT_CUDA_IQ_FIFO_PATH)
    cmd.extend(["--cuda-shm", cuda_shm_name])
    assert slot_fifo_path is not None
    _append_plotter_fifo_cli_flag(cmd, "--fifo-fapi", slot_fifo_path, _DEFAULT_CUDA_SLOT_FIFO_PATH)


def _ensure_named_pipe_for_generator(path: str, *, log_label: str) -> None:
    """Create a FIFO node if missing so ``generator.py`` can open O_WRONLY (plotter may not have recreated it)."""
    if os.path.exists(path):
        try:
            st = os.stat(path)
        except OSError as e:
            raise RuntimeError(f"cannot stat {log_label} path {path!r}: {e}") from e
        if not stat.S_ISFIFO(st.st_mode):
            raise RuntimeError(f"{log_label} path exists but is not a named pipe: {path!r}")
        return
    _log.info("creating %s named pipe %s for generator attach", log_label, path)
    try:
        os.mkfifo(path, 0o666)
    except FileExistsError:
        pass
    os.chmod(path, 0o666)
    _fifo_apply_chown(path)


_GENERATOR_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generator.py"
# Host:port the *browser* uses to reach MediaMTX HTTP (HLS). Fallback when request has no usable Host / Origin (e.g. curl without -H).
MTX_PUBLIC_HTTP_BASE = _env("MTX_PUBLIC_HTTP_BASE", "http://localhost:8888").rstrip("/")
CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
# Browsers on the LAN open Vite at http://<subnet-ip>:<port> — allow private-IP origins on any port.
CORS_ORIGIN_REGEX = _env(
    "CORS_ORIGIN_REGEX",
    r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?",
)


# Generation modes (**GENERATION_MODE** env at process start). Same labels as **POST /api/generator/start** ``mode``:
#   **SYNTHETIC_IMAGES** — **generator.py** → **RESGRID_FIFO_IMAGES** (RGBA FIFO); **resgrid_plotter** still opens all FIFO argv paths from env.
#   **SYNTHETIC_RESOURCE_GRID** — **--fifo-resource-grid** full-grid path + optional **generator.py**;
#       built-in C++ synthesizer when no external writer (compose **CUDA_IQ_USE_FIFO** is ignored; FIFO is always created).
#   **SYNTHETIC_FAPI** — **--fifo-fapi** slot messages / channel-based; optional **--fifo-resource-grid** sidecar I/Q.
# **generator.py** ``--data-mode`` only (one producer shape at a time). **resgrid_plotter** ingests all FIFOs regardless.
_GENERATOR_DATA_MODE_FULL_GRID_FIFO = 2
_GENERATOR_DATA_MODE_SLOT_OR_COMBINED = 3

SYNTHETIC_IMAGES = "SYNTHETIC_IMAGES"
SYNTHETIC_RESOURCE_GRID = "SYNTHETIC_RESOURCE_GRID"
SYNTHETIC_FAPI = "SYNTHETIC_FAPI"

_GENERATION_MODES_CUDA_PLOTTER = frozenset({SYNTHETIC_RESOURCE_GRID, SYNTHETIC_FAPI})
_GENERATION_MODES_ALL = frozenset({SYNTHETIC_IMAGES, SYNTHETIC_RESOURCE_GRID, SYNTHETIC_FAPI})


def _raw_generation_mode_from_env() -> str:
    """Process **GENERATION_MODE** (default **synthetic_channel_based** → **SYNTHETIC_FAPI** after canonicalization)."""
    return _env("GENERATION_MODE", "synthetic_channel_based")


def _canonical_generation_mode_token(raw: str) -> str:
    """Normalize env / API tokens to **SYNTHETIC_IMAGES** | **SYNTHETIC_RESOURCE_GRID** | **SYNTHETIC_FAPI**."""
    r = raw.strip().upper().replace("-", "_")
    if r in (SYNTHETIC_FAPI, "SYNTHETIC_CHANNEL_BASED"):
        return SYNTHETIC_FAPI
    if r in (SYNTHETIC_RESOURCE_GRID, "SYNTHETIC_IQ"):
        return SYNTHETIC_RESOURCE_GRID
    if r in (SYNTHETIC_IMAGES, "SYNTHETIC_VIDEO"):
        return SYNTHETIC_IMAGES
    return SYNTHETIC_IMAGES


def _coerce_generation_mode_input(mode: str) -> str:
    """Accept historic **SYNTHETIC_VIDEO** / **SYNTHETIC_IQ** / **SYNTHETIC_CHANNEL_BASED** spellings."""
    return _canonical_generation_mode_token(mode)


def _parse_generation_mode(raw: str) -> str:
    return _canonical_generation_mode_token(raw)


GENERATION_MODE = _parse_generation_mode(_raw_generation_mode_from_env())
# Default RGBA named pipe when **RESGRID_FIFO_IMAGES** is unset or empty (same path as **resgrid_plotter** default).
_DEFAULT_RESGRID_FIFO_IMAGES_PATH = "/tmp/images.fifo"
RESGRID_FIFO_IMAGES = _RESGRID_FIFO_IMAGES_RAW or _DEFAULT_RESGRID_FIFO_IMAGES_PATH

_mtx_path = (urlparse(MTX_RTSP_URL).path or "/live").strip("/") or "live"

_ffmpeg: Optional[subprocess.Popen] = None
_plotter_process: Optional[subprocess.Popen] = None
_generator_script_proc: Optional[subprocess.Popen] = None
# True while **generator.py** was started with **--fifo-fapi** (slot / combined path), not resource-grid-only.
_generator_script_writes_fapi: bool = False
# True while **generator.py** was started with **--fifo-resource-grid** (full-grid / combined path), not slot-only.
_generator_script_writes_resource_grid: bool = False
_fifo_images_generator: Optional[subprocess.Popen] = None
_lock = threading.Lock()
_current_lavfi_video: str = ""
# Preset id from POST /api/generator/start `source`, or None when graph came from LAVFI_VIDEO / custom body string.
_current_video_generator_source: Optional[str] = None
# Set True when the last successful CUDA start created / used the slot named pipe (for cleanup on stop).
_plotter_fapi_fifo_open: bool = False
# Set True when the last plotter spawn passed **--fifo-resource-grid** (named pipe open for **generator.py** / external writers).
_plotter_resource_grid_fifo_open: bool = False
# Holds **CUDA_IQ_FIFO_PATH** open **O_RDWR** (never read) so **resgrid_plotter**'s blocking **open(O_RDONLY)** on that FIFO
# returns without a real **generator.py** writer — needed for **feed: fapi** (**--slot-only**) so the plotter still starts and
# the sidecar reader blocks in **read** until a combined/resource-grid producer appears (do not read this fd; it is not a consumer).
_resource_grid_fifo_peer_fd: Optional[int] = None
_cuda_spawn_encoder_fps: Optional[int] = None
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
        description="Encoder output FPS from FFmpeg progress (stderr fps= = stdin frames/s); **resgrid_plotter** matches `--encoder-fps` when pacing stdin.",
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
_generator_producer_shm: Optional[shared_memory.SharedMemory] = None
_plotter_cuda_shm: Optional[shared_memory.SharedMemory] = None


class PlaybackUrls(BaseModel):
    """URLs for subscribers to read the MediaMTX stream (WebRTC for real-time; HLS as fallback)."""

    path: str = Field(description="MediaMTX path name (same as in the RTSP URL the plotter / FFmpeg writes to).")
    hls: str = Field(description="HLS playlist URL — higher latency; works everywhere hls.js does.")
    webrtc_whep: str = Field(
        description="WHEP URL for WebRTC playback — lowest latency in Chromium/Firefox; use with MediaMTX reader.",
    )
    note: str = Field(
        default="Prefer WebRTC (webrtc_whep) for real-time; HLS segments always buffer partly.",
    )


class PlotterStatus(BaseModel):
    """**resgrid_plotter** pipeline, NVENC path, and JPEG capturer — excludes **generator.py** and standalone FFmpeg (see **GET /api/generator/status**)."""

    generating: bool = Field(
        description="True while **resgrid_plotter** and/or legacy FFmpeg is driving the MediaMTX path.",
    )
    cuda_plotter_active: bool = Field(
        default=False,
        description="True while this backend's **resgrid_plotter** child is alive.",
    )
    video_encoder: str = Field(
        description="**nvenc** / **cpu** for FFmpeg lavfi path, or **cuda_iq_pipeline** when **resgrid_plotter** is encoding (legacy discriminator string).",
    )
    stats: Optional[GenerationStats] = Field(None, description="Encode / pipeline counters (see **GET /api/plotter/status** OpenAPI on this route).")
    publish_rtsp: str = Field(description="Internal RTSP URL passed to **resgrid_plotter** / FFmpeg as the stream sink.")
    mtx_path: str = Field(description="MediaMTX path name for this stream.")
    iq_prb: int = Field(description="PRB dimension (CUDA_IQ_PRB).")
    iq_symbols: int = Field(description="Symbol dimension (CUDA_IQ_SYMBOLS).")
    iq_producer_fps: Optional[float] = None
    iq_plotter_fps: Optional[float] = None
    iq_producer_command_fps: Optional[float] = None
    iq_pattern: Optional[int] = None
    assembly_input_msg_s: Optional[float] = None
    assembly_output_slot_s: Optional[float] = None
    raster_input_slot_s: Optional[float] = None
    raster_output_image_s: Optional[float] = None
    encoder_input_image_s: Optional[float] = None
    encoder_output_fps: Optional[float] = None
    capturer_capable: bool = False
    capturer_active: Optional[bool] = None
    capturer_phase: Optional[str] = None
    capturer_inflight: Optional[int] = None
    capturer_written: Optional[int] = None
    capturer_total: Optional[int] = None
    capturer_out_dir: Optional[str] = None
    capturer_queue_depth: Optional[int] = None
    capturer_queue_cap: Optional[int] = None
    capturer_jpeg_threads: Optional[int] = None


GeneratorStartFeed = Literal["images", "resource_grid", "fapi"]


class GeneratorResponse(BaseModel):
    """Internal / legacy ack used by **start_plotter_and_generators** (plotter, images FIFO, FFmpeg). Prefer route-specific response models in OpenAPI."""

    status: str
    pid: Optional[int] = None
    lavfi_video: Optional[str] = None
    source: Optional[str] = Field(
        None,
        description="Echo of **video_generator** preset id when applicable; null if the graph came from **LAVFI_VIDEO** or a custom `source` string.",
    )
    encoder_fps: Optional[int] = Field(
        None,
        description=(
            "When starting **resgrid_plotter**: applied **--encoder-fps**. "
            "When starting **SYNTHETIC_IMAGES** FIFO **generator.py**: target write cadence (matches plotter **--encoder-fps** / **CUDA_IQ_FPS**). "
            "Omitted when not applicable."
        ),
    )
    feed: Optional[GeneratorStartFeed] = Field(
        None,
        description=(
            "Echo of **POST /api/generator/start** **`feed`** (**images** | **resource_grid** | **fapi**) when this response concerns that generator path. "
            "Null for CUDA plotter-only or legacy standalone FFmpeg."
        ),
    )


class PlotterControlResponse(BaseModel):
    """JSON from **POST /api/plotter/start** / **stop**, **POST /api/stop**, and **POST /api/generator/stop** (and **feed** acks for **resource_grid** / **fapi** generator start)."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="**started** | **stopped** | **already_running** | **not_running**.")
    pid: Optional[int] = Field(None, description="Relevant child PID when **started** / **stopped** / **already_running**.")
    encoder_fps: Optional[int] = Field(
        None,
        description="Present after **resgrid_plotter** **start** / **already_running**: effective **`--encoder-fps`**.",
    )
    feed: Optional[GeneratorStartFeed] = Field(
        None,
        description="Present on **POST /api/generator/start** for **`feed`** **resource_grid** or **fapi** only.",
    )


class GeneratorLavfiStartResponse(BaseModel):
    """JSON from **POST /api/generator/start** when **`feed`** is **`images`** (RGBA FIFO **generator.py** + lavfi graph)."""

    model_config = ConfigDict(extra="forbid")

    status: str
    pid: Optional[int] = None
    encoder_fps: Optional[int] = None
    lavfi_video: Optional[str] = None
    source: Optional[str] = None
    feed: Literal["images"] = "images"


def _plotter_control_from_generator_response(r: GeneratorResponse) -> PlotterControlResponse:
    return PlotterControlResponse(status=r.status, pid=r.pid, encoder_fps=r.encoder_fps)


def _generator_lavfi_start_from_response(r: GeneratorResponse) -> GeneratorLavfiStartResponse:
    return GeneratorLavfiStartResponse(
        status=r.status,
        pid=r.pid,
        encoder_fps=r.encoder_fps,
        lavfi_video=r.lavfi_video,
        source=r.source,
    )


class VideoGeneratorStatusResponse(BaseModel):
    """State of **images** mode helpers: RGBA FIFO **generator.py** and/or legacy FFmpeg lavfi."""

    running: bool = Field(
        description="True when **generator.py** (RGBA FIFO) **or** legacy FFmpeg lavfi is active."
    )
    backend: Optional[str] = Field(
        None,
        description=(
            "Which child feeds MediaMTX: **`fifo_images`** (generator.py → RESGRID_FIFO_IMAGES → resgrid_plotter), "
            "**`ffmpeg_lavfi`** (legacy FFmpeg lavfi → MediaMTX), or **`null`** when neither runs."
        ),
    )
    pid: Optional[int] = Field(
        None, description="PID of the active child (generator.py or FFmpeg); null when not running."
    )
    source: Optional[str] = Field(
        None,
        description="Preset id from **POST /api/generator/start** `source`, or **null** for a custom graph / default.",
    )
    pattern: Optional[str] = Field(
        None,
        description=(
            "Short pattern name resolved from the active synthetic-images graph (**`testsrc`** | **`smptebars`** | "
            "**`mandelbrot`** | **`randomnoise`**; the last is Python/NumPy, not lavfi), or **null** when no generator is active."
        ),
    )
    lavfi_video: Optional[str] = Field(
        None,
        description="Resolved lavfi graph string driving the active backend, or **null** when not running.",
    )
    fifo_path: Optional[str] = Field(
        None,
        description=(
            "Server-side **RESGRID_FIFO_IMAGES** path (named pipe). Empty/`null` when the RGBA FIFO feeder "
            "is disabled in this process."
        ),
    )


class PlotterStartBody(BaseModel):
    """Optional JSON body for **POST /api/plotter/start**.

    **resgrid_plotter** is always started with **--fifo-images**, **--fifo-resource-grid**, and **--fifo-fapi**
    (paths from **RESGRID_FIFO_IMAGES**, **CUDA_IQ_FIFO_PATH**, **CUDA_SLOT_FIFO_PATH**). The plotter ingests every FIFO;
    **generator.py** chooses a single producer shape via **``--data-mode``** when you **POST /api/generator/start** (not this route).

    Send **`{}`** or omit fields you do not need. Unknown keys (including legacy **`mode`**) are rejected (**422**).

    Optional **encoder_fps** overrides **CUDA_IQ_FPS** for this spawn only (**--encoder-fps**). Pipeline **--fps** remains **CUDA_IQ_GEN_FPS**.
    """

    model_config = ConfigDict(extra="forbid")

    encoder_fps: Optional[float] = Field(
        default=None,
        ge=1,
        le=PLOTTER_ENCODER_FPS_CMD_MAX,
        description=(
            f"Override **CUDA_IQ_FPS** for this start: **resgrid_plotter** **--encoder-fps** (1…{PLOTTER_ENCODER_FPS_CMD_MAX} Hz, "
            "target H.264 output cadence)."
        ),
    )


GeneratorFeedMode = Literal[
    "SYNTHETIC_IMAGES",
    "SYNTHETIC_RESOURCE_GRID",
    "SYNTHETIC_FAPI",
]


class GeneratorImagesOptions(BaseModel):
    """Options for **POST /api/generator/start** when **`feed`** is **`images`** (RGBA FIFO **generator.py**)."""

    pattern: Optional[str] = Field(
        None,
        description=(
            "**testsrc** | **smptebars** | **mandelbrot** | **randomnoise** — mapped to default lavfi preset sizes/fps. "
            "Mutually exclusive with **`preset`**."
        ),
    )
    preset: Optional[str] = Field(
        None,
        description=(
            "A **preset id** (e.g. **testsrc_720p30**, **smptebars_720p30**, …) **or** a full **lavfi** graph string containing `=`. "
            "Replaces the legacy top-level **`source`** field. Mutually exclusive with **`pattern`**."
        ),
    )

    @model_validator(mode="after")
    def _validate_images_opts(self) -> "GeneratorImagesOptions":
        pat = (self.pattern or "").strip()
        pre = (self.preset or "").strip()
        if pat and pre:
            raise ValueError("images: send at most one of `pattern` or `preset` (or omit both for server / **LAVFI_VIDEO** default).")
        return self


class GeneratorFeedStartBody(BaseModel):
    """Internal validated body for lavfi resolution and **start_plotter_and_generators** (uses **SYNTHETIC_**\* tokens)."""

    mode: GeneratorFeedMode = Field(
        ...,
        description=(
            "**SYNTHETIC_IMAGES** — **generator.py** → **RESGRID_FIFO_IMAGES** (RGBA); optional **`pattern`** / **`source`** / neither (env default). "
            "**SYNTHETIC_RESOURCE_GRID** — **generator.py** → **CUDA_IQ_FIFO_PATH** (full-grid resource-grid FIFO only). "
            "**SYNTHETIC_FAPI** — **generator.py** slot / combined path. "
            "CUDA plotter must already be running for all three."
        ),
    )
    source: Optional[str] = Field(
        None,
        description=(
            "**SYNTHETIC_IMAGES** only: a **preset id** (testsrc_720p30, testsrc_480p30, smptebars_720p30, mandelbrot_960p30, randomnoise_960p30), "
            "or a full **lavfi** graph string containing `=`. Omit when using **`pattern`** or when both are omitted for **LAVFI_VIDEO** default."
        ),
    )
    pattern: Optional[str] = Field(
        None,
        description=(
            "**SYNTHETIC_IMAGES** only: **testsrc** | **smptebars** | **mandelbrot** | **randomnoise** — mapped to default preset sizes/fps. "
            "Mutually exclusive with **`source`**."
        ),
    )

    @model_validator(mode="after")
    def _validate_generator_feed_body(self) -> "GeneratorFeedStartBody":
        src = (self.source or "").strip()
        pat = (self.pattern or "").strip()
        if self.mode == SYNTHETIC_IMAGES:
            if pat and src:
                raise ValueError(
                    "SYNTHETIC_IMAGES: send at most one of `pattern` or `source` (or omit both for LAVFI_VIDEO / server default)."
                )
        elif src or pat:
            raise ValueError(
                f"{self.mode!r}: omit `pattern` and `source` (lavfi fields apply only to SYNTHETIC_IMAGES)."
            )
        return self


class GeneratorStartBody(BaseModel):
    """JSON for **POST /api/generator/start** using **`feed`**: **images** (RGBA FIFO), **resource_grid** (full-grid FIFO), **fapi** (slot FIFO)."""

    feed: GeneratorStartFeed = Field(
        ...,
        description=(
            "**images** — **generator.py** → **RESGRID_FIFO_IMAGES**. Optional nested **`images`** (**`pattern`** / **`preset`**). "
            "**fapi** — slot / FAPI FIFO only (**``generator.py``** **``--slot-only``**); no full-grid resource-grid waveform. "
            "CUDA plotter must already be running."
        ),
    )
    images: Optional[GeneratorImagesOptions] = Field(
        default=None,
        description="Only when **`feed`** is **`images`**: lavfi **pattern** short name and/or **preset** (id or full graph).",
    )

    @model_validator(mode="after")
    def _validate_generator_start(self) -> "GeneratorStartBody":
        if self.feed != "images" and self.images is not None:
            raise ValueError("`images` is only allowed when `feed` is `images`.")
        if self.feed == "images" and self.images is not None:
            GeneratorImagesOptions.model_validate(self.images.model_dump())
        return self


class StopAllBody(BaseModel):
    """Optional body for **POST /api/stop**. `mode` is accepted for clients that want a symmetric payload; it does not affect behavior."""

    mode: Optional[str] = Field(default=None, description="Ignored; stop always tears down whichever plotter or generator child is running.")


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
    """Acknowledgement that a stop was written to CUDA plotter SHM; **resgrid_plotter** clears it on the next raster tick."""

    status: str = Field(description='Typically **`"stop_requested"`**.')


class GeneratorFpsResponse(BaseModel):
    fps: float = Field(description="Commanded **generator.py** / built-in producer rate (Hz) from shared memory or saved override.")
    min_fps: float = Field(description="Minimum allowed generator FPS (env IQ_PRODUCER_FPS_MIN).")
    max_fps: float = Field(description="Maximum allowed generator FPS (env IQ_PRODUCER_FPS_MAX).")
    generation_mode: str
    generating: bool
    iq_fifo_mode: bool = Field(
        description="Legacy JSON field name: **True** when **--fifo-resource-grid** is part of this stack’s plotter argv (**CUDA_IQ_USE_FIFO** is always on)."
    )


class GeneratorFpsPutBody(BaseModel):
    fps: float = Field(
        ...,
        description="New commanded generator rate (Hz); written to generator SHM and CUDA control SHM when the plotter is running.",
    )


class GeneratorPatternInfo(BaseModel):
    id: int
    label: str


class GeneratorPatternsResponse(BaseModel):
    patterns: list[GeneratorPatternInfo]


class GeneratorPatternPutBody(BaseModel):
    pattern: int = Field(
        ...,
        description="Synthetic grid pattern id (0…max). Written to generator and CUDA SHM when the plotter is running.",
    )


class GeneratorPatternResponse(BaseModel):
    pattern: int


class GeneratorControlResponse(BaseModel):
    status: str = Field(description="started | stopped | already_running | not_running")
    pid: Optional[int] = None


class SyntheticFifoWriterStatus(BaseModel):
    """Whether **generator.py** is attached to this named pipe and whether the running plotter exposes it."""

    running: bool = Field(description="True while **generator.py** is alive **and** this FIFO path is active for this writer.")
    pid: Optional[int] = Field(None, description="OS pid when **running** is true.")
    capable: bool = Field(
        description="True when **resgrid_plotter** was spawned with this FIFO so **generator.py** may attach.",
    )
    fifo_path: Optional[str] = Field(
        None,
        description=(
            "Configured named pipe path for this writer (**CUDA_IQ_FIFO_PATH** or **CUDA_SLOT_FIFO_PATH**); always echoed for discovery. "
            "**capable** is true only when **resgrid_plotter** on this process opened that FIFO for reads."
        ),
    )


class GeneratorFullStatus(BaseModel):
    """Combined **generator.py** / FFmpeg **images** path state, **resource_grid** / **fapi** FIFO writer state, and process **GENERATION_MODE**."""

    generation_mode: str = Field(description="Process **GENERATION_MODE** from env at startup.")
    synthetic_video: VideoGeneratorStatusResponse = Field(
        description="RGBA FIFO **generator.py** and/or legacy FFmpeg lavfi.",
    )
    synthetic_resource_grid: SyntheticFifoWriterStatus = Field(
        description="Full-grid **--fifo-resource-grid** path (**CUDA_IQ_FIFO_PATH** env on this host).",
    )
    synthetic_fapi: SyntheticFifoWriterStatus = Field(
        description="Slot / FAPI FIFO (**--fifo-fapi** / **CUDA_SLOT_FIFO_PATH**).",
    )


def _generator_control_ack(
    resp: GeneratorControlResponse, *, feed: Optional[GeneratorStartFeed] = None
) -> PlotterControlResponse:
    return PlotterControlResponse(status=resp.status, pid=resp.pid, feed=feed)


class FapiChanLegendRgb(BaseModel):
    """sRGB 0…255; matches CUDA `channel_tint_rgb` base coefficients × 255 (rounded)."""

    r: int = Field(..., ge=0, le=255)
    g: int = Field(..., ge=0, le=255)
    b: int = Field(..., ge=0, le=255)


class FapiChanLegendEntry(BaseModel):
    """One PHY-style logical channel id (values 1…8 in slot messages / plotter region labels)."""

    id: int = Field(..., ge=1, le=255, description="Logical PHY channel id (1…8 per OAI slot exporter / plotter).")
    label: str = Field(..., description="Short PHY name (same as on-screen region labels).")
    rgb: FapiChanLegendRgb
    hex: str = Field(..., description='CSS sRGB "#rrggbb" for the base tint.')


class FapiChannelLegendResponse(BaseModel):
    channels: list[FapiChanLegendEntry] = Field(
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


def _fapi_channel_legend_payload() -> FapiChannelLegendResponse:
    """Keep in sync with `channel_tint_rgb` in backend/plotter/src/plotter_raster.cu."""
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
    channels: list[FapiChanLegendEntry] = []
    for cid, label, (fr, fg, fb) in specs:
        r = int(round(fr * 255.0))
        g = int(round(fg * 255.0))
        b = int(round(fb * 255.0))
        h = f"#{r:02x}{g:02x}{b:02x}"
        channels.append(
            FapiChanLegendEntry(id=cid, label=label, rgb=FapiChanLegendRgb(r=r, g=g, b=b), hex=h)
        )
    return FapiChannelLegendResponse(
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

_LAVFI_PRESET_IDS_DOC = (
    "**`testsrc_720p30`** · **`testsrc_480p30`** · **`smptebars_720p30`** · **`mandelbrot_960p30`** · **`randomnoise_960p30`**"
)

# Preset ids for **POST /api/generator/start** JSON field **`source`** (see `_LAVFI_PRESET_IDS_DOC` in OpenAPI).
_LAVFI_SOURCE_PRESETS: dict[str, str] = {
    "testsrc_720p30": "testsrc=size=1280x720:rate=30",
    "testsrc_480p30": "testsrc=size=854x480:rate=30",
    "smptebars_720p30": "smptebars=size=1280x720:rate=30",
    # Mandelbrot is CPU-heavy at 720p; 960×540 + fps keeps graphs valid for encoders.
    "mandelbrot_960p30": "mandelbrot=size=960x540,fps=30",
    # Not a lavfi graph: marker string so ``_image_pattern_from_resolved_lavfi`` selects Python random RGBA in generator.py.
    "randomnoise_960p30": "randomnoise=size=960x540,fps=30",
}

# Short pattern name → default preset id (used by **POST /api/generator/start** JSON field **`pattern`**).
_LAVFI_PATTERN_DEFAULT_PRESET: dict[str, str] = {
    "testsrc": "testsrc_720p30",
    "smptebars": "smptebars_720p30",
    "mandelbrot": "mandelbrot_960p30",
    "randomnoise": "randomnoise_960p30",
}


def _lavfi_video() -> str:
    return _env("LAVFI_VIDEO", _DEFAULT_LAVFI_VIDEO)


def _resolve_lavfi_graph(source: Optional[str]) -> tuple[str, Optional[str]]:
    """Return (ffmpeg lavfi graph string, preset id or None for env/custom graph)."""
    if source is None or not str(source).strip():
        return _lavfi_video(), None
    raw = str(source).strip()
    if raw in _LAVFI_SOURCE_PRESETS:
        return _LAVFI_SOURCE_PRESETS[raw], raw
    if "=" in raw:
        return raw, None
    raise ValueError(
        f"Unknown lavfi preset id {raw!r}. Use a known preset id ({_LAVFI_PRESET_IDS_DOC}), "
        "omit `source` for LAVFI_VIDEO / default, or pass a full graph string containing '='."
    )


def _coerce_video_generator_source(
    source: Optional[str], pattern: Optional[str]
) -> Optional[str]:
    """Combine ``source`` and ``pattern`` for **SYNTHETIC_IMAGES** **POST /api/generator/start** requests.

    ``source`` wins when both are set. ``pattern`` accepts the short lavfi name
    (``testsrc`` / ``smptebars`` / ``mandelbrot`` / ``randomnoise``) and is mapped to its default
    preset id (see :data:`_LAVFI_PATTERN_DEFAULT_PRESET`). Returns ``None`` when
    both inputs are absent so callers fall back to **LAVFI_VIDEO** / defaults.
    """
    raw_source = (source or "").strip()
    if raw_source:
        return raw_source
    raw_pattern = (pattern or "").strip().lower()
    if not raw_pattern:
        return None
    if raw_pattern not in _LAVFI_PATTERN_DEFAULT_PRESET:
        known = ", ".join(sorted(_LAVFI_PATTERN_DEFAULT_PRESET))
        raise ValueError(
            f"Unknown pattern {raw_pattern!r}. Use one of: {known}. For fine-grained control "
            f"set `source` (preset id {_LAVFI_PRESET_IDS_DOC} or a full lavfi graph)."
        )
    return _LAVFI_PATTERN_DEFAULT_PRESET[raw_pattern]


def _resolved_lavfi_for_synthetic_images(body: GeneratorFeedStartBody) -> Optional[str]:
    """Resolve lavfi preset / graph for **SYNTHETIC_IMAGES**; caller must validate ``body.mode``."""
    return _coerce_video_generator_source(body.source, body.pattern)


def _uses_cuda_plotter(mode: str) -> bool:
    return mode in _GENERATION_MODES_CUDA_PLOTTER


def get_generation_mode() -> str:
    """Fixed for this process from **GENERATION_MODE** at startup (not mutated by plotter / generator APIs)."""
    return GENERATION_MODE


def _resource_grid_fifo_enabled_unlocked() -> bool:
    """True when this stack is configured to expose **--fifo-resource-grid** (always on; see **CUDA_IQ_USE_FIFO** legacy env)."""
    return CUDA_IQ_USE_FIFO


def _is_generation_mode_channel_based(sm: str) -> bool:
    return sm == SYNTHETIC_FAPI


def _resolve_mode_for_start_locked(requested_mode: Optional[str]) -> str:
    """Resolve the generation **mode** for this spawn: **requested_mode** when non-empty (validated), else **GENERATION_MODE**."""
    if requested_mode is not None and str(requested_mode).strip():
        m = _coerce_generation_mode_input(str(requested_mode))
        if m not in _GENERATION_MODES_ALL:
            raise ValueError(
                f"mode must be one of {sorted(_GENERATION_MODES_ALL)!r}, got {requested_mode!r}"
            )
        return m
    return GENERATION_MODE


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
    if effective_video_encoder() == "nvenc":
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


def _is_plotter_running() -> bool:
    global _plotter_process
    return _plotter_process is not None and _plotter_process.poll() is None


def _fifo_images_generator_running() -> bool:
    global _fifo_images_generator
    return _fifo_images_generator is not None and _fifo_images_generator.poll() is None


def _is_running() -> bool:
    return _is_ffmpeg_running() or _is_plotter_running() or _fifo_images_generator_running()


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


def _default_generator_producer_fps() -> float:
    """SHM-commanded producer rate (resource grid / FAPI); separate from encoder ``CUDA_IQ_FPS`` (plotter ``--encoder-fps``).

    Python **generator.py** and slot writers follow **CUDA_IQ_GEN_FPS** whenever they are used (**resource_grid** / **fapi** FIFOs,
    or built-in synthetic grid). Encoder output stays **CUDA_IQ_FPS** (e.g. 24 Hz) via plotter ``--encoder-fps``.
    The plotter ``--fps`` knob is the pipeline / internal producer nominal rate (**CUDA_IQ_GEN_FPS**, e.g. 200 Hz).
    """
    sm_fn = get_generation_mode()
    use_gen_default = (
        CUDA_IQ_USE_FIFO
        or sm_fn == SYNTHETIC_RESOURCE_GRID
        or sm_fn == SYNTHETIC_FAPI
    )
    raw = CUDA_IQ_GEN_FPS if use_gen_default else CUDA_IQ_FPS
    try:
        v = float(str(raw).strip())
    except ValueError:
        v = 200.0 if use_gen_default else 24.0
    return max(IQ_PRODUCER_FPS_MIN, min(IQ_PRODUCER_FPS_MAX, v))


def _validate_producer_command_fps(fps: float) -> float:
    if math.isnan(fps) or math.isinf(fps):
        raise ValueError("fps must be a finite number")
    if fps < IQ_PRODUCER_FPS_MIN or fps > IQ_PRODUCER_FPS_MAX:
        raise ValueError(
            f"fps must be between {IQ_PRODUCER_FPS_MIN:g} and {IQ_PRODUCER_FPS_MAX:g}",
        )
    return float(fps)


def _default_synthetic_pattern() -> int:
    try:
        v = int(str(_env("IQ_PATTERN", "0")).strip())
    except ValueError:
        v = IQ_PATTERN_DEFAULT
    return max(IQ_PATTERN_DEFAULT, min(IQ_PATTERN_MAX, v))


def _validate_synthetic_pattern(pid: int) -> int:
    if pid < IQ_PATTERN_DEFAULT or pid > IQ_PATTERN_MAX:
        raise ValueError(f"pattern must be between {IQ_PATTERN_DEFAULT} and {IQ_PATTERN_MAX}")
    return int(pid)


def _effective_synthetic_pattern_unlocked() -> int:
    if _synthetic_pattern_override is not None:
        return max(IQ_PATTERN_DEFAULT, min(IQ_PATTERN_MAX, int(_synthetic_pattern_override)))
    return _default_synthetic_pattern()


def _read_synthetic_pattern_from_shm_unlocked() -> int:
    """Current pattern from cuda shm if present, else override/env."""
    if _plotter_cuda_shm is not None:
        try:
            (p,) = struct.unpack_from("I", _plotter_cuda_shm.buf, 16)
            if p <= IQ_PATTERN_MAX:
                return int(p)
        except (struct.error, TypeError, ValueError):
            pass
    return _effective_synthetic_pattern_unlocked()


def _write_synthetic_pattern_shm_unlocked(pattern: int) -> int:
    global _synthetic_pattern_override
    v = _validate_synthetic_pattern(pattern)
    _synthetic_pattern_override = v
    if _generator_producer_shm is not None:
        struct.pack_into("I", _generator_producer_shm.buf, 16, v)
    if _plotter_cuda_shm is not None:
        struct.pack_into("I", _plotter_cuda_shm.buf, 16, v)
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
    if _plotter_cuda_shm is None or len(_plotter_cuda_shm.buf) < _IQ_CUDA_SHM_BYTES:
        return 0, 0, 0, 0, "", 0, 0, 0
    buf = _plotter_cuda_shm.buf
    tot, inflight, written, phase = struct.unpack_from("IIII", buf, _SHM_CAPTURER_TOTAL_OFF)
    raw = bytes(buf[_SHM_CAPTURER_PATH_OFF : _SHM_CAPTURER_PATH_OFF + _SHM_CAPTURER_PATH_CAP])
    path = raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    qd, qc, jt = 0, 0, 0
    if len(buf) >= _SHM_CAPTURER_JPEG_THREADS_OFF + 4:
        qd, qc, jt = struct.unpack_from("III", buf, _SHM_CAPTURER_QUEUE_DEPTH_OFF)
    return tot, inflight, written, phase, path, qd, qc, jt


def _arm_capturer_shm_unlocked(out_dir: str, n_frames: int) -> None:
    if _plotter_cuda_shm is None or len(_plotter_cuda_shm.buf) < _IQ_CUDA_SHM_BYTES:
        raise RuntimeError("Plotter CUDA shared memory not available")
    bpath = out_dir.encode("utf-8")
    if len(bpath) >= _SHM_CAPTURER_PATH_CAP:
        raise ValueError("capturer output path exceeds SHM path capacity")
    buf = _plotter_cuda_shm.buf
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
    if _plotter_cuda_shm is None or len(_plotter_cuda_shm.buf) < _IQ_CUDA_SHM_BYTES:
        raise RuntimeError("Plotter CUDA shared memory not available")
    buf = _plotter_cuda_shm.buf
    if len(buf) < _SHM_CAPTURER_STOP_OFF + 4:
        raise RuntimeError("Plotter CUDA shared memory too small for capturer stop flag")
    struct.pack_into("I", buf, _SHM_CAPTURER_STOP_OFF, 1)


def _plotter_shm_destroy_unlocked() -> None:
    global _generator_producer_shm, _plotter_cuda_shm
    for shm in (_generator_producer_shm, _plotter_cuda_shm):
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
    _generator_producer_shm = None
    _plotter_cuda_shm = None


def _generator_producer_shm_create_unlocked(name: str, cmd: float, pattern: int) -> None:
    global _generator_producer_shm
    _generator_producer_shm = shared_memory.SharedMemory(
        create=True, size=_IQ_PRODUCER_SHM_BYTES, name=name
    )
    _generator_producer_shm.buf[:_IQ_PRODUCER_SHM_BYTES] = _pack_producer_shm_init(cmd, pattern)


def _plotter_cuda_shm_create_unlocked(name: str, cmd: float, pattern: int) -> None:
    global _plotter_cuda_shm
    _plotter_cuda_shm = shared_memory.SharedMemory(create=True, size=_IQ_CUDA_SHM_BYTES, name=name)
    _plotter_cuda_shm.buf[:_IQ_CUDA_SHM_BYTES] = _pack_cuda_shm_init(cmd, pattern)


def _plotter_shm_pair_create_unlocked(
    pattern_override: Optional[int] = None,
    *,
    create_fifo_producer_shm: bool,
) -> tuple[str, Optional[str]]:
    """Create CUDA control SHM; optional POSIX SHM for ``generator.py`` pacing/pattern."""
    _plotter_shm_destroy_unlocked()
    ns = time.time_ns()
    cmd = (
        _generator_producer_fps_override
        if _generator_producer_fps_override is not None
        else _default_generator_producer_fps()
    )
    cmd = max(IQ_PRODUCER_FPS_MIN, min(IQ_PRODUCER_FPS_MAX, float(cmd)))
    if pattern_override is not None:
        pattern = _validate_synthetic_pattern(int(pattern_override))
    else:
        pattern = _effective_synthetic_pattern_unlocked()
    cuda_name = f"/resgrid_iqcuda_{os.getpid()}_{ns}"
    _plotter_cuda_shm_create_unlocked(cuda_name, cmd, pattern)
    prod_name: Optional[str] = None
    if create_fifo_producer_shm:
        prod_name = f"/resgrid_iqprod_{os.getpid()}_{ns}"
        _generator_producer_shm_create_unlocked(prod_name, cmd, pattern)
    return cuda_name, prod_name


def _read_plotter_shm_stream_metrics_unlocked() -> tuple[Optional[float], Optional[float], Optional[int]]:
    """Measured generator producer Hz, raster output Hz (SHM offset 8), cumulative encoder frames (offset 32)."""

    def norm(x: float) -> Optional[float]:
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    prod: Optional[float] = None
    plot: Optional[float] = None
    frames: Optional[int] = None
    if _generator_producer_shm is not None:
        try:
            (pv,) = struct.unpack_from("d", _generator_producer_shm.buf, 0)
            prod = norm(pv)
        except (struct.error, TypeError, ValueError):
            pass
    if _plotter_cuda_shm is not None:
        try:
            if _generator_producer_shm is None:
                (pv,) = struct.unpack_from("d", _plotter_cuda_shm.buf, 0)
                prod = norm(pv)
            (tv,) = struct.unpack_from("d", _plotter_cuda_shm.buf, 8)
            plot = norm(tv)
            if len(_plotter_cuda_shm.buf) >= 40:
                (fc,) = struct.unpack_from("Q", _plotter_cuda_shm.buf, 32)
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
    if _plotter_cuda_shm is None or len(_plotter_cuda_shm.buf) < _IQ_CUDA_SHM_BYTES:
        return none6
    try:
        a_in, a_out, r_in, r_out, e_in, e_out = struct.unpack_from("dddddd", _plotter_cuda_shm.buf, 40)
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


def _read_generator_producer_fps_shm_unlocked() -> float:
    if _generator_producer_shm is not None:
        try:
            (cmd,) = struct.unpack_from("d", _generator_producer_shm.buf, 8)
            if math.isfinite(cmd):
                return max(IQ_PRODUCER_FPS_MIN, min(IQ_PRODUCER_FPS_MAX, float(cmd)))
        except (struct.error, TypeError, ValueError):
            pass
    if _plotter_cuda_shm is not None:
        try:
            (cmd,) = struct.unpack_from("d", _plotter_cuda_shm.buf, 24)
            if math.isfinite(cmd):
                return max(IQ_PRODUCER_FPS_MIN, min(IQ_PRODUCER_FPS_MAX, float(cmd)))
        except (struct.error, TypeError, ValueError):
            pass
    if _generator_producer_fps_override is not None:
        return max(
            IQ_PRODUCER_FPS_MIN,
            min(IQ_PRODUCER_FPS_MAX, float(_generator_producer_fps_override)),
        )
    return _default_generator_producer_fps()


def _write_generator_producer_fps_shm_unlocked(fps: float) -> float:
    global _generator_producer_fps_override
    v = _validate_producer_command_fps(fps)
    _generator_producer_fps_override = v
    if _generator_producer_shm is not None:
        struct.pack_into("d", _generator_producer_shm.buf, 8, v)
    if _plotter_cuda_shm is not None:
        struct.pack_into("d", _plotter_cuda_shm.buf, 24, v)
    return v


def _cuda_stderr_drain_loop(proc: subprocess.Popen) -> None:
    """Merge resgrid_plotter stderr (incl. FFmpeg) into shared FFmpeg progress stats."""
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
                    _log.info("plotter: %s", line)
        tail = buf.strip()
        if tail:
            if "fps=" in tail and "frame=" in tail:
                _apply_progress_line(tail)
            else:
                _log.info("plotter: %s", tail)
    except Exception:
        _log.exception("plotter stderr reader failed")
    finally:
        try:
            proc.stderr.close()
        except OSError:
            pass


def _spawn_cuda_stderr_thread(proc: subprocess.Popen) -> None:
    threading.Thread(target=_cuda_stderr_drain_loop, args=(proc,), daemon=True).start()


def _terminate_ffmpeg_unlocked() -> None:
    global _ffmpeg, _current_lavfi_video, _current_video_generator_source, _stats
    if not _is_ffmpeg_running():
        _ffmpeg = None
        _current_lavfi_video = ""
        _current_video_generator_source = None
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
    _current_video_generator_source = None
    with _stats_lock:
        _stats = GenerationStats()


def _generator_script_stderr_loop(proc: subprocess.Popen) -> None:
    if proc.stderr is None:
        return
    try:
        for line in iter(proc.stderr.readline, b""):
            if line:
                txt = line.decode("utf-8", errors="replace").rstrip()
                _log.info("iq_fifo_producer: %s", txt)
    except Exception:
        _log.exception("generator.py stderr reader failed")
    finally:
        try:
            proc.stderr.close()
        except OSError:
            pass


def _spawn_generator_script_stderr_thread(proc: subprocess.Popen) -> None:
    threading.Thread(target=_generator_script_stderr_loop, args=(proc,), daemon=True).start()


def _fifo_images_generator_stderr_loop(proc: subprocess.Popen) -> None:
    if proc.stderr is None:
        return
    try:
        for line in iter(proc.stderr.readline, b""):
            if line:
                txt = line.decode("utf-8", errors="replace").rstrip()
                _log.info("fifo_images_generator: %s", txt)
    except Exception:
        _log.exception("fifo-images generator stderr reader failed")
    finally:
        try:
            proc.stderr.close()
        except OSError:
            pass


def _spawn_fifo_images_generator_stderr_thread(proc: subprocess.Popen) -> None:
    threading.Thread(target=_fifo_images_generator_stderr_loop, args=(proc,), daemon=True).start()


def _terminate_fifo_images_generator_unlocked() -> None:
    global _fifo_images_generator, _current_video_generator_source, _current_lavfi_video
    if _fifo_images_generator is None:
        return
    proc = _fifo_images_generator
    _fifo_images_generator = None
    if proc.poll() is not None:
        _current_video_generator_source = None
        _current_lavfi_video = ""
        return
    _log.info("Terminating fifo-images generator.py pid=%s", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    _current_video_generator_source = None
    _current_lavfi_video = ""


def _start_fifo_images_generator_unlocked(
    *,
    graph: str,
    width: int,
    height: int,
    fps: int,
) -> int:
    """Spawn ``generator.py`` writing RGBA to ``RESGRID_FIFO_IMAGES``. Caller holds ``_lock``."""
    global _fifo_images_generator
    if _fifo_images_generator is not None and _fifo_images_generator.poll() is None:
        return _fifo_images_generator.pid
    if _fifo_images_generator is not None:
        _fifo_images_generator = None
    if not _GENERATOR_SCRIPT.is_file():
        raise FileNotFoundError(f"generator.py missing: {_GENERATOR_SCRIPT}")
    pat = _image_pattern_from_resolved_lavfi(graph)
    cmd: list[str] = [
        sys.executable,
        "-u",
        str(_GENERATOR_SCRIPT),
        "--fifo-images",
        RESGRID_FIFO_IMAGES,
        "--rgba-width",
        str(width),
        "--rgba-height",
        str(height),
        "--fps",
        str(max(1, int(fps))),
        "--image-pattern",
        pat,
    ]
    if _generator_producer_shm is not None:
        cmd.extend(["--generator-shm", _generator_producer_shm.name])
    if IQ_FIFO_PRODUCER_LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR"):
        cmd.extend(["--log-level", IQ_FIFO_PRODUCER_LOG_LEVEL])
    try:
        _fifo_images_generator = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError:
        _log.exception("Failed to spawn fifo-images generator.py")
        raise
    _spawn_fifo_images_generator_stderr_thread(_fifo_images_generator)
    time.sleep(0.15)
    if _fifo_images_generator.poll() is not None:
        code = _fifo_images_generator.returncode
        _log.error("fifo-images generator exited immediately with %s", code)
        _fifo_images_generator = None
        raise RuntimeError("fifo-images generator failed to start; check server logs")
    _log.info(
        "fifo-images generator.py started pid=%s path=%s size=%dx%d pattern=%s",
        _fifo_images_generator.pid,
        RESGRID_FIFO_IMAGES,
        width,
        height,
        pat,
    )
    return _fifo_images_generator.pid


def _terminate_generator_script_unlocked() -> None:
    global _generator_script_proc, _generator_script_writes_fapi, _generator_script_writes_resource_grid
    if _generator_script_proc is None:
        return
    proc = _generator_script_proc
    _generator_script_proc = None
    _generator_script_writes_fapi = False
    _generator_script_writes_resource_grid = False
    if proc.poll() is not None:
        return
    _log.info("Terminating generator.py pid=%s", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _generator_script_running_unlocked() -> bool:
    global _generator_script_proc
    if _generator_script_proc is None:
        return False
    return _generator_script_proc.poll() is None


def _generator_fifo_feed_capable_unlocked() -> bool:
    if not _is_plotter_running():
        return False
    return _plotter_resource_grid_fifo_open or _plotter_fapi_fifo_open


def _generator_resource_grid_fifo_capable_unlocked() -> bool:
    """True when **generator.py** can attach to the resource-grid FIFO (``--fifo-resource-grid``)."""
    if not _is_plotter_running():
        return False
    return bool(_plotter_resource_grid_fifo_open)


def _spawn_generator_slot_only_unlocked(
    *,
    producer_shm_name: str,
    slot_fifo_path: str,
    fps_arg: str,
    slot_interval: str,
) -> int:
    """Spawn **generator.py** with **``--slot-only``** (FAPI FIFO only; no **``--fifo-resource-grid``**). Caller holds ``_lock``."""
    global _generator_script_proc, _generator_script_writes_fapi, _generator_script_writes_resource_grid
    _ensure_named_pipe_for_generator(slot_fifo_path, log_label="slot FIFO")
    py_cmd = [
        sys.executable,
        "-u",
        str(_GENERATOR_SCRIPT),
        "--slot-only",
        "--data-mode",
        str(_GENERATOR_DATA_MODE_SLOT_OR_COMBINED),
        "--fifo-fapi",
        slot_fifo_path,
        "--fps",
        fps_arg,
        "--generator-shm",
        producer_shm_name,
        "--prb",
        CUDA_IQ_PRB,
        "--symbols",
        CUDA_IQ_SYMBOLS,
        "--slot-interval",
        slot_interval,
        "--slot-pattern",
        _slot_demo_pattern_cli(),
    ]
    _extend_producer_symbol_window_args(py_cmd)
    if IQ_FIFO_PRODUCER_LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR"):
        py_cmd.extend(["--log-level", IQ_FIFO_PRODUCER_LOG_LEVEL])
    try:
        _generator_script_proc = subprocess.Popen(
            py_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError:
        _log.exception("Failed to spawn slot FIFO demo producer")
        raise
    _spawn_generator_script_stderr_thread(_generator_script_proc)
    time.sleep(0.15)
    if _generator_script_proc.poll() is not None:
        code = _generator_script_proc.returncode
        _log.error("slot demo producer exited immediately with %s", code)
        _generator_script_proc = None
        _generator_script_writes_fapi = False
        _generator_script_writes_resource_grid = False
        raise RuntimeError("slot FIFO demo producer failed to start; check server logs")
    _generator_script_writes_fapi = True
    _generator_script_writes_resource_grid = False
    return _generator_script_proc.pid


def _start_generator_script_unlocked(
    *,
    encoder_hz_for_slot: Optional[float] = None,
    resource_grid_only: bool = False,
    fapi_slot_only: bool = False,
) -> int:
    """Spawn generator.py. Caller must hold _lock; CUDA must be running with producer SHM.

    **resource_grid_only** — **``--fifo-resource-grid``** + **``--data-mode``** **2** only (no **``--fifo-fapi``** / slot demo args)
    even when the plotter has opened the FAPI named pipe.

    **fapi_slot_only** — **``--slot-only``** + **``--fifo-fapi``** only (no **``--fifo-resource-grid``**) even when the plotter
    has opened the resource-grid named pipe.
    """
    global _generator_script_proc, _generator_script_writes_fapi, _generator_script_writes_resource_grid
    if _generator_script_proc is not None and _generator_script_proc.poll() is None:
        return _generator_script_proc.pid
    if _generator_script_proc is not None:
        _generator_script_proc = None
        _generator_script_writes_fapi = False
        _generator_script_writes_resource_grid = False

    use_resource_grid_fifo = _plotter_resource_grid_fifo_open
    use_fapi_fifo = _plotter_fapi_fifo_open
    if resource_grid_only and fapi_slot_only:
        raise RuntimeError("internal: **resource_grid_only** and **fapi_slot_only** are mutually exclusive")
    if resource_grid_only and not use_resource_grid_fifo:
        raise RuntimeError(
            "Resource-grid **generator.py** needs **resgrid_plotter** started with **--fifo-resource-grid** "
            "(named pipe open). **POST /api/plotter/stop** then **POST /api/plotter/start**."
        )
    if fapi_slot_only and not use_fapi_fifo:
        raise RuntimeError(
            "FAPI / slot **generator.py** needs **resgrid_plotter** started with **--fifo-fapi** "
            "(named pipe open). **POST /api/plotter/stop** then **POST /api/plotter/start**."
        )
    if not use_resource_grid_fifo and not use_fapi_fifo:
        raise RuntimeError("This generator script run has no resource-grid or FAPI FIFO to write.")
    if _generator_producer_shm is None:
        raise RuntimeError("Producer shared memory missing; restart the plotter.")

    producer_shm_name = _generator_producer_shm.name
    fifo_path = CUDA_IQ_FIFO_PATH
    slot_fifo_path = CUDA_SLOT_FIFO_PATH

    fps_arg = str(CUDA_IQ_GEN_FPS).strip() or "200"
    slot_interval = _slot_demo_interval_seconds_str(encoder_hz_override=encoder_hz_for_slot)

    if not _GENERATOR_SCRIPT.is_file():
        raise FileNotFoundError(f"generator.py missing: {_GENERATOR_SCRIPT}")

    if fapi_slot_only:
        return _spawn_generator_slot_only_unlocked(
            producer_shm_name=producer_shm_name,
            slot_fifo_path=slot_fifo_path,
            fps_arg=fps_arg,
            slot_interval=slot_interval,
        )

    if use_resource_grid_fifo:
        if resource_grid_only:
            prod_dm = _GENERATOR_DATA_MODE_FULL_GRID_FIFO
        elif use_fapi_fifo:
            prod_dm = _GENERATOR_DATA_MODE_SLOT_OR_COMBINED
        else:
            prod_dm = _GENERATOR_DATA_MODE_FULL_GRID_FIFO
        _ensure_named_pipe_for_generator(fifo_path, log_label="resource-grid FIFO")
        py_cmd = [
            sys.executable,
            "-u",
            str(_GENERATOR_SCRIPT),
            "--fifo-resource-grid",
            fifo_path,
            "--data-mode",
            str(prod_dm),
            "--prb",
            CUDA_IQ_PRB,
            "--symbols",
            CUDA_IQ_SYMBOLS,
            "--fps",
            fps_arg,
            "--generator-shm",
            producer_shm_name,
        ]
        if use_fapi_fifo and not resource_grid_only:
            _ensure_named_pipe_for_generator(slot_fifo_path, log_label="slot FIFO")
            py_cmd.extend(
                [
                    "--fifo-fapi",
                    slot_fifo_path,
                    "--slot-interval",
                    slot_interval,
                    "--slot-pattern",
                    _slot_demo_pattern_cli(),
                ]
            )
            _extend_producer_symbol_window_args(py_cmd)
        if IQ_FIFO_PRODUCER_LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR"):
            py_cmd.extend(["--log-level", IQ_FIFO_PRODUCER_LOG_LEVEL])
        try:
            _generator_script_proc = subprocess.Popen(
                py_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError:
            _log.exception("Failed to spawn generator.py")
            raise
        _spawn_generator_script_stderr_thread(_generator_script_proc)
        time.sleep(0.15)
        if _generator_script_proc.poll() is not None:
            code = _generator_script_proc.returncode
            _log.error("generator.py exited immediately with %s", code)
            _generator_script_proc = None
            _generator_script_writes_fapi = False
            _generator_script_writes_resource_grid = False
            raise RuntimeError("generator.py failed to start; check server logs")
        _generator_script_writes_fapi = bool(use_fapi_fifo and not resource_grid_only)
        _generator_script_writes_resource_grid = True
        return _generator_script_proc.pid

    return _spawn_generator_slot_only_unlocked(
        producer_shm_name=producer_shm_name,
        slot_fifo_path=slot_fifo_path,
        fps_arg=fps_arg,
        slot_interval=slot_interval,
    )


def _terminate_plotter_unlocked() -> None:
    global _plotter_process, _stats, _plotter_fapi_fifo_open, _plotter_resource_grid_fifo_open, _cuda_plotter_capturer_dir
    global _cuda_spawn_encoder_fps
    _terminate_fifo_images_generator_unlocked()
    _terminate_generator_script_unlocked()
    if not _is_plotter_running():
        _plotter_process = None
        _cuda_spawn_encoder_fps = None
        _plotter_shm_destroy_unlocked()
        _cuda_plotter_capturer_dir = None
        if _plotter_fapi_fifo_open:
            try:
                os.unlink(CUDA_SLOT_FIFO_PATH)
            except OSError:
                pass
            _plotter_fapi_fifo_open = False
        _plotter_resource_grid_fifo_open = False
        _close_resource_grid_fifo_peer_hold_unlocked()
        return
    assert _plotter_process is not None
    _log.info("Terminating CUDA plotter pid=%s", _plotter_process.pid)
    _plotter_process.terminate()
    try:
        _plotter_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _plotter_process.kill()
    _plotter_process = None
    _cuda_spawn_encoder_fps = None
    _plotter_shm_destroy_unlocked()
    _cuda_plotter_capturer_dir = None
    if _plotter_fapi_fifo_open:
        try:
            os.unlink(CUDA_SLOT_FIFO_PATH)
        except OSError:
            pass
        _plotter_fapi_fifo_open = False
    _plotter_resource_grid_fifo_open = False
    _close_resource_grid_fifo_peer_hold_unlocked()
    with _stats_lock:
        _stats = GenerationStats()


def _video_generator_status_unlocked() -> VideoGeneratorStatusResponse:
    """Caller must hold ``_lock`` when FIFO / FFmpeg children may change."""
    fifo_up = _fifo_images_generator_running()
    ff_up = _is_ffmpeg_running()
    if fifo_up:
        backend = "fifo_images"
        assert _fifo_images_generator is not None
        pid = _fifo_images_generator.pid
    elif ff_up:
        backend = "ffmpeg_lavfi"
        assert _ffmpeg is not None
        pid = _ffmpeg.pid
    else:
        backend = None
        pid = None
    graph = _current_lavfi_video if (fifo_up or ff_up) else ""
    source = _current_video_generator_source if (fifo_up or ff_up) else None
    pattern = _image_pattern_from_resolved_lavfi(graph) if graph else None
    return VideoGeneratorStatusResponse(
        running=fifo_up or ff_up,
        backend=backend,
        pid=pid,
        source=source,
        pattern=pattern,
        lavfi_video=graph or None,
        fifo_path=RESGRID_FIFO_IMAGES or None,
    )


def _synthetic_resource_grid_writer_status_unlocked() -> SyntheticFifoWriterStatus:
    """Caller must hold ``_lock``. Full-grid resource-grid FIFO **generator.py** attachment."""
    gen_up = _generator_script_running_unlocked()
    resource_grid_fifo_ok = _plotter_resource_grid_fifo_open
    running = gen_up and resource_grid_fifo_ok and _generator_script_writes_resource_grid
    pid = _generator_script_proc.pid if running and _generator_script_proc is not None else None
    capable = _generator_resource_grid_fifo_capable_unlocked()
    resource_grid_fifo_path = (CUDA_IQ_FIFO_PATH or "").strip() or None
    return SyntheticFifoWriterStatus(
        running=running,
        pid=pid,
        capable=capable,
        fifo_path=resource_grid_fifo_path,
    )


def _synthetic_fapi_writer_status_unlocked() -> SyntheticFifoWriterStatus:
    """Caller must hold ``_lock``. Slot / FAPI FIFO **generator.py** attachment."""
    gen_up = _generator_script_running_unlocked()
    slot_active = _plotter_fapi_fifo_open
    running = gen_up and slot_active and _generator_script_writes_fapi
    pid = _generator_script_proc.pid if running and _generator_script_proc is not None else None
    capable = _is_plotter_running() and _plotter_fapi_fifo_open
    return SyntheticFifoWriterStatus(
        running=running,
        pid=pid,
        capable=capable,
        fifo_path=(CUDA_SLOT_FIFO_PATH or "").strip() or None,
    )


def _generator_full_status_unlocked() -> GeneratorFullStatus:
    """Caller must hold ``_lock``."""
    return GeneratorFullStatus(
        generation_mode=get_generation_mode(),
        synthetic_video=_video_generator_status_unlocked(),
        synthetic_resource_grid=_synthetic_resource_grid_writer_status_unlocked(),
        synthetic_fapi=_synthetic_fapi_writer_status_unlocked(),
    )


def start_plotter_and_generators(
    requested_mode: Optional[str] = None,
    *,
    plotter_only: bool = False,
    synthetic_video_only: bool = False,
    video_generator_source: Optional[str] = None,
    plotter_encoder_fps: Optional[float] = None,
    echo_feed: Optional[GeneratorStartFeed] = None,
) -> GeneratorResponse:
    global _ffmpeg, _plotter_process, _generator_script_proc, _fifo_images_generator, _current_lavfi_video, _current_video_generator_source, _stats, _plotter_fapi_fifo_open
    global _plotter_resource_grid_fifo_open, _cuda_plotter_capturer_dir, _cuda_spawn_encoder_fps
    with _lock:
        if synthetic_video_only:
            mode = _resolve_mode_for_start_locked(SYNTHETIC_IMAGES)
        else:
            mode = _resolve_mode_for_start_locked(requested_mode)
        if (plotter_only or _uses_cuda_plotter(mode)) and not synthetic_video_only:
            if _is_plotter_running():
                assert _plotter_process is not None
                return GeneratorResponse(
                    status="already_running",
                    pid=_plotter_process.pid,
                    encoder_fps=_cuda_spawn_encoder_fps,
                )
            _terminate_ffmpeg_unlocked()
            _wait_tcp(MTX_HOST, MTX_PORT)
            with _stats_lock:
                _stats = GenerationStats()

            _plotter_fapi_fifo_open = False
            _plotter_resource_grid_fifo_open = False
            # **resgrid_plotter** always gets **--fifo-images** + **--fifo-resource-grid** + **--fifo-fapi** + **--cuda-shm** (all named pipes).
            use_resource_grid_fifo = True
            use_fapi_fifo = True

            fifo_path = CUDA_IQ_FIFO_PATH
            slot_fifo_path = CUDA_SLOT_FIFO_PATH
            # Generator SHM: resource-grid / FAPI writers need producer SHM; RGBA FIFO path uses the same knob for symmetry.
            create_fifo_producer_shm = True

            if CUDA_AUTO_START_IQ_GENERATOR and not _GENERATOR_SCRIPT.is_file():
                raise FileNotFoundError(f"generator.py missing: {_GENERATOR_SCRIPT}")

            _close_resource_grid_fifo_peer_hold_unlocked()
            _recreate_plotter_fifo_node(fifo_path)
            _open_resource_grid_fifo_peer_for_plotter_unlocked(fifo_path)
            _recreate_plotter_fifo_node(slot_fifo_path)
            _recreate_plotter_fifo_node(RESGRID_FIFO_IMAGES)

            enc_i = _coerce_plotter_encoder_fps_cmd(plotter_encoder_fps)
            pipe_arg = str(CUDA_IQ_GEN_FPS).strip() or "200"

            pat_init = None
            cuda_shm_name, producer_shm_name = _plotter_shm_pair_create_unlocked(
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
                PLOTTER_BIN,
                "--rtsp",
                MTX_RTSP_URL,
                "--fps",
                pipe_arg,
                "--encoder-fps",
                str(enc_i),
                "--cell-pixels",
                CUDA_IQ_CELL_PIXELS,
                "--preset",
                CUDA_IQ_PRESET,
                "--video-encoder",
                "nvenc" if effective_video_encoder() == "nvenc" else "cpu",
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
            ]
            _append_plotter_input_fifo_and_cuda_shm(
                cmd,
                cuda_shm_name=cuda_shm_name,
                fifo_images_path=RESGRID_FIFO_IMAGES,
                default_fifo_images_path=_DEFAULT_RESGRID_FIFO_IMAGES_PATH,
                resource_grid_path=fifo_path,
                slot_fifo_path=slot_fifo_path,
            )
            cmd.extend(capturer_extra)
            try:
                _plotter_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except OSError:
                _log.exception("Failed to spawn %s", PLOTTER_BIN)
                _close_resource_grid_fifo_peer_hold_unlocked()
                _plotter_shm_destroy_unlocked()
                if use_resource_grid_fifo:
                    try:
                        os.unlink(fifo_path)
                    except OSError:
                        pass
                if use_fapi_fifo:
                    try:
                        os.unlink(slot_fifo_path)
                    except OSError:
                        pass
                _plotter_fapi_fifo_open = False
                _plotter_resource_grid_fifo_open = False
                raise
            _spawn_cuda_stderr_thread(_plotter_process)
            _cuda_spawn_encoder_fps = enc_i
            # All three FIFO argv flags are always passed for this spawn.
            _plotter_resource_grid_fifo_open = True
            _plotter_fapi_fifo_open = True
            if capturer_dir_raw:
                _cuda_plotter_capturer_dir = os.path.realpath(capturer_dir_raw)

            if create_fifo_producer_shm and not producer_shm_name:
                _terminate_plotter_unlocked()
                raise RuntimeError("internal: missing producer shared memory for generator / FIFO modes")

            if CUDA_AUTO_START_IQ_GENERATOR and (use_resource_grid_fifo or use_fapi_fifo):
                try:
                    _start_generator_script_unlocked(
                        encoder_hz_for_slot=float(enc_i),
                    )
                except FileNotFoundError:
                    _terminate_plotter_unlocked()
                    if use_resource_grid_fifo:
                        try:
                            os.unlink(fifo_path)
                        except OSError:
                            pass
                    raise
                except Exception:
                    _log.exception("Failed to spawn generator.py (resource-grid / FAPI FIFO path)")
                    _terminate_plotter_unlocked()
                    if use_resource_grid_fifo:
                        try:
                            os.unlink(fifo_path)
                        except OSError:
                            pass
                    raise

            try:
                ve_cli = cmd[cmd.index("--video-encoder") + 1]
            except (ValueError, IndexError):
                ve_cli = None
            spawn_detail: dict[str, object] = {
                "argv": shlex.join(cmd),
                "binary": PLOTTER_BIN,
                "pid": _plotter_process.pid,
                "cwd": os.getcwd(),
                "plotter_only": plotter_only,
                "generation_mode": mode,
                "plotter_fifo_inputs": "images+resource_grid+fapi",
                "use_resource_grid_fifo": use_resource_grid_fifo,
                "use_fapi_fifo": use_fapi_fifo,
                "create_fifo_producer_shm": create_fifo_producer_shm,
                "resgrid_fifo_images": RESGRID_FIFO_IMAGES or None,
                "resource_grid_fifo_path": fifo_path,
                "fapi_fifo_path": slot_fifo_path,
                "cuda_shm_name": cuda_shm_name,
                "producer_shm_name": producer_shm_name,
                "generator_script": str(_GENERATOR_SCRIPT),
                "api_encoder_fps_request_body": plotter_encoder_fps,
                "effective_encoder_fps": enc_i,
                "pipeline_fps_arg": pipe_arg,
                "capturer_dir": capturer_dir_raw or None,
                "capturer_cli_extra": list(capturer_extra) if capturer_extra else None,
                "auto_start_generator_script": CUDA_AUTO_START_IQ_GENERATOR,
                "video_encoder_cli": ve_cli,
                "effective_video_encoder_after_probe": effective_video_encoder(),
                "env": _plotter_spawn_env_for_log(),
            }
            _log.info("%s started: %s", PLOTTER_BIN, json.dumps(spawn_detail, default=str, sort_keys=True))
            return GeneratorResponse(
                status="started",
                pid=_plotter_process.pid,
                encoder_fps=enc_i,
            )

        if synthetic_video_only:
            if not _is_plotter_running():
                raise ValueError(
                    "Start the CUDA plotter first (**POST /api/plotter/start**), "
                    "then use **POST /api/generator/start** with **`{\"feed\":\"images\",…}`** to run **generator.py** feeding the RGBA FIFO."
                )
            if _fifo_images_generator_running():
                assert _fifo_images_generator is not None
                fps_i = _effective_fifo_images_generator_fps()
                return GeneratorResponse(
                    status="already_running",
                    pid=_fifo_images_generator.pid,
                    lavfi_video=_current_lavfi_video or None,
                    source=_current_video_generator_source,
                    encoder_fps=max(1, int(fps_i)),
                    feed=echo_feed,
                )
            if _is_ffmpeg_running():
                _terminate_ffmpeg_unlocked()
            graph, src_id = _resolve_lavfi_graph(video_generator_source)
            _current_lavfi_video = graph
            _current_video_generator_source = src_id
            w, h = _plotter_padded_rgba_size()
            fps_i = _effective_fifo_images_generator_fps()
            pid = _start_fifo_images_generator_unlocked(graph=graph, width=w, height=h, fps=max(1, fps_i))
            return GeneratorResponse(
                status="started",
                pid=pid,
                lavfi_video=_current_lavfi_video or None,
                source=_current_video_generator_source,
                encoder_fps=max(1, int(fps_i)),
                feed=echo_feed,
            )

        if _is_plotter_running():
            _terminate_plotter_unlocked()
        if _is_ffmpeg_running():
            assert _ffmpeg is not None
            _log.info("POST start: already running pid=%s", _ffmpeg.pid)
            return GeneratorResponse(
                status="already_running",
                pid=_ffmpeg.pid,
                lavfi_video=_current_lavfi_video or None,
                source=_current_video_generator_source,
            )
        _wait_tcp(MTX_HOST, MTX_PORT)
        if synthetic_video_only:
            graph, src_id = _resolve_lavfi_graph(video_generator_source)
        else:
            graph, src_id = _lavfi_video(), None
        _current_lavfi_video = graph
        _current_video_generator_source = src_id
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
            effective_video_encoder(),
            _current_lavfi_video,
        )
        return GeneratorResponse(
            status="started",
            pid=_ffmpeg.pid,
            lavfi_video=_current_lavfi_video,
            source=_current_video_generator_source,
        )


def stop_plotter_and_generators(*, plotter_only: bool = False) -> GeneratorResponse:
    global _ffmpeg, _plotter_process
    with _lock:
        if plotter_only:
            if not _is_plotter_running():
                _log.info("stop_plotter_and_generators(plotter_only): CUDA plotter not running")
                return GeneratorResponse(status="not_running")
            assert _plotter_process is not None
            last_pid = _plotter_process.pid
            _terminate_plotter_unlocked()
            _log.info("CUDA plotter stopped (pid was %s)", last_pid)
            return GeneratorResponse(status="stopped", pid=last_pid)
        if not _is_running():
            _log.info("stop_plotter_and_generators: nothing was running")
            return GeneratorResponse(status="not_running")
        last_pid: Optional[int] = None
        if _is_ffmpeg_running():
            assert _ffmpeg is not None
            last_pid = _ffmpeg.pid
            _terminate_ffmpeg_unlocked()
        if _is_plotter_running():
            assert _plotter_process is not None
            last_pid = _plotter_process.pid
            _terminate_plotter_unlocked()
        _log.info("Plotter stopped (last pid was %s)", last_pid)
        return GeneratorResponse(status="stopped", pid=last_pid)


def stop_generators() -> GeneratorResponse:
    """Stop FFmpeg lavfi (if any), ``generator.py`` RGBA fifo feeder, and/or the resource-grid / FAPI ``generator.py`` child; leaves CUDA plotter running if active."""
    with _lock:
        had = False
        last_pid: Optional[int] = None
        if _is_ffmpeg_running():
            assert _ffmpeg is not None
            last_pid = _ffmpeg.pid
            _terminate_ffmpeg_unlocked()
            had = True
        if _fifo_images_generator_running():
            assert _fifo_images_generator is not None
            last_pid = _fifo_images_generator.pid
            _terminate_fifo_images_generator_unlocked()
            had = True
        if _generator_script_running_unlocked():
            assert _generator_script_proc is not None
            last_pid = _generator_script_proc.pid
            _terminate_generator_script_unlocked()
            had = True
        if not had:
            _log.info("stop_generators: no FFmpeg, fifo-images, or generator.py child running")
            return GeneratorResponse(status="not_running")
        _log.info("Generator / images path stopped (last pid was %s)", last_pid)
        return GeneratorResponse(status="stopped", pid=last_pid)


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
    _init_effective_video_encoder()
    _log.info(
        "Backend starting (AUTO_START=%s GENERATION_MODE=%s video_encoder_requested=%s effective=%s)",
        AUTO_START,
        get_generation_mode(),
        VIDEO_ENCODER,
        effective_video_encoder(),
    )
    try:
        stop_plotter_and_generators()
    except Exception:
        _log.exception("Initial stop_plotter_and_generators failed")
    if AUTO_START:
        try:
            start_plotter_and_generators()
        except Exception:
            _log.exception("AUTO_START failed")
    yield
    _log.info("Backend shutting down; stopping generation if running")
    stop_plotter_and_generators()


def _openapi_description_with_hls(hls_index_url: str) -> str:
    return (
        "MediaMTX (RTSP path **`live`**): **SYNTHETIC_IMAGES** = subscriber-first — with **RESGRID_FIFO_IMAGES**, use **POST /api/plotter/start** then "
        "**POST /api/generator/start** (`generator.py` → RGBA FIFO → CUDA → RTSP; JSON **`feed`** required). "
        "**SYNTHETIC_RESOURCE_GRID** (`resgrid_plotter`: built-in synthesizer or full-grid resource-grid FIFO → H.264), or **SYNTHETIC_FAPI** (slot messages / channel-based). "
        f"**HLS manifest:** [{hls_index_url}]({hls_index_url}) "
        "(host is taken from **this** request to `/openapi.json`: `Origin` / `Referer`, `X-Forwarded-Host`, `Host`, or the URL host — "
        f"same order as **GET /api/plotter/playback**; if none apply, **`{MTX_PUBLIC_HTTP_BASE}`**). "
        "Dual DL/UL stacks use **`live_dl`** / **`live_ul`** — use **GET /api/plotter/playback** for this backend’s paths. "
        "**Start / stop:** **POST /api/generator/start** (RGBA **images** FIFO, needs **RESGRID_FIFO_IMAGES** + running plotter), **POST /api/plotter/start** (always opens **images** + **resource_grid** + **FAPI** FIFO argv), "
        "or **POST /api/generator/start** with **`{{\"feed\":\"resource_grid\"}}`** / **`{{\"feed\":\"fapi\"}}`** for **generator.py** on those FIFOs. **POST /api/stop** stops the plotter and standalone FFmpeg."
    )


def _openapi_tags_with_hls(hls_index_url: str) -> list[dict[str, str]]:
    return [
        {
            "name": "plotter",
            "description": (
                "CUDA **resgrid_plotter**: **POST /api/plotter/start** / **stop**, **POST /api/stop** (plotter + standalone FFmpeg), "
                "**GET /api/plotter/status**, **GET /api/plotter/playback** (subscriber URLs), JPEG capturer."
            ),
        },
        {
            "name": "generator",
            "description": (
                "Unified **/api/generator/** API: **GET /api/generator/status**, **POST /api/generator/start** / **stop**, "
                "**GET**/**PUT /api/generator/fps**, **GET /api/generator/patterns**, **PUT /api/generator/pattern**, **GET /api/generator/channel-legend**."
            ),
        },
        {"name": "health", "description": "Liveness."},
        {
            "name": "OpenAPI",
            "description": "Machine-readable schema (**GET /openapi.json**), **Swagger UI** (**GET /docs**), and **ReDoc** (**GET /redoc**).",
        },
    ]


app = FastAPI(
    title="MediaMTX plotter and generator",
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
app.add_middleware(HttpAccessLogMiddleware)


@app.get("/health", tags=["health"], summary="Liveness probe")
def health():
    return {"ok": True, "generating": _is_running()}


@app.get(
    "/api/generator/status",
    response_model=GeneratorFullStatus,
    tags=["generator"],
    summary="Generator processes and process GENERATION_MODE",
    description=(
        "Process **GENERATION_MODE** plus **synthetic_video** (RGBA FIFO **generator.py** / legacy FFmpeg lavfi) and "
        "**synthetic_resource_grid** and **synthetic_fapi** (resource-grid / FAPI FIFO **generator.py** writers). CUDA **resgrid_plotter** metrics: **GET /api/plotter/status**."
    ),
)
def api_generator_status():
    with _lock:
        return _generator_full_status_unlocked()


@app.get(
    "/api/plotter/playback",
    response_model=PlaybackUrls,
    tags=["plotter"],
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


def _generator_fps_api_response_unlocked() -> GeneratorFpsResponse:
    return GeneratorFpsResponse(
        fps=_read_generator_producer_fps_shm_unlocked(),
        min_fps=IQ_PRODUCER_FPS_MIN,
        max_fps=IQ_PRODUCER_FPS_MAX,
        generation_mode=get_generation_mode(),
        generating=_is_plotter_running(),
        iq_fifo_mode=_resource_grid_fifo_enabled_unlocked(),
    )


@app.get(
    "/api/generator/fps",
    response_model=GeneratorFpsResponse,
    tags=["generator"],
    summary="Generator producer FPS (commanded)",
    description=(
        "Reads commanded producer FPS from generator SHM (when running) or saved override / env default. "
        "Applies to **SYNTHETIC_RESOURCE_GRID** and **SYNTHETIC_FAPI** (internal CUDA synthesizer and/or Python FIFO writer)."
    ),
)
def api_generator_fps_get():
    with _lock:
        if not _is_plotter_running():
            raise HTTPException(
                status_code=400,
                detail="CUDA plotter (**resgrid_plotter**) is not running.",
            )
        return _generator_fps_api_response_unlocked()


@app.put(
    "/api/generator/fps",
    response_model=GeneratorFpsResponse,
    tags=["generator"],
    summary="Set generator producer FPS (commanded)",
    description=(
        "Writes **fps** to generator SHM (Python FIFO mode) and CUDA SHM (offset 24, internal synthesizer) when running; "
        "otherwise stores an override applied on the next plotter **Start**."
    ),
)
def api_generator_fps_put(body: GeneratorFpsPutBody):
    with _lock:
        if not _is_plotter_running():
            raise HTTPException(
                status_code=400,
                detail="CUDA plotter (**resgrid_plotter**) is not running.",
            )
        try:
            v = _write_generator_producer_fps_shm_unlocked(body.fps)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _log.info("Generator producer FPS set to %s Hz", v)
        return _generator_fps_api_response_unlocked()


def _generator_start_handler_fapi_slot() -> GeneratorControlResponse:
    """Spawn **generator.py** with **``--slot-only``** (FAPI / slot FIFO only; no full-grid writes).

    The FastAPI process holds the resource-grid FIFO open **O_RDWR** (see **``_open_resource_grid_fifo_peer_for_plotter_unlocked``**)
    when the plotter starts so **resgrid_plotter**'s blocking **open(O_RDONLY)** on that pipe still succeeds; the plotter sidecar
    then blocks in **read** until a combined/resource-grid producer is used, while slot messages alone drive the raster.
    """
    with _lock:
        if _generator_script_proc is not None and _generator_script_proc.poll() is None:
            return GeneratorControlResponse(status="already_running", pid=_generator_script_proc.pid)
        if not _is_plotter_running():
            raise HTTPException(
                status_code=400,
                detail=(
                    "CUDA plotter (**resgrid_plotter**) is not running. "
                    "Start it with **POST /api/plotter/start**, then start this generator again — it only attaches to an already-running plotter."
                ),
            )
        if not _generator_fifo_feed_capable_unlocked():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Plotter is running but FIFO attach flags are off (internal state). **POST /api/plotter/stop** then **POST /api/plotter/start**."
                ),
            )
        try:
            pid = _start_generator_script_unlocked(
                encoder_hz_for_slot=float(_cuda_spawn_encoder_fps) if _cuda_spawn_encoder_fps is not None else None,
                fapi_slot_only=True,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return GeneratorControlResponse(status="started", pid=pid)


def _generator_start_handler_resource_grid() -> GeneratorControlResponse:
    """Spawn **generator.py** writing **only** the resource-grid FIFO (``--fifo-resource-grid``)."""
    with _lock:
        if _generator_script_proc is not None and _generator_script_proc.poll() is None:
            return GeneratorControlResponse(status="already_running", pid=_generator_script_proc.pid)
        if not _is_plotter_running():
            raise HTTPException(
                status_code=400,
                detail=(
                    "CUDA plotter (**resgrid_plotter**) is not running. "
                    "Start it with **POST /api/plotter/start**, then start this generator again."
                ),
            )
        if not _generator_resource_grid_fifo_capable_unlocked():
            raise HTTPException(
                status_code=400,
                detail=(
                    "The running **resgrid_plotter** did not mark the resource-grid FIFO as open (internal state). "
                    "**POST /api/plotter/stop** then **POST /api/plotter/start** again. "
                    "For slot-only demos use **POST /api/generator/start** with **`{\"feed\":\"fapi\"}`**."
                ),
            )
        try:
            pid = _start_generator_script_unlocked(
                encoder_hz_for_slot=float(_cuda_spawn_encoder_fps) if _cuda_spawn_encoder_fps is not None else None,
                resource_grid_only=True,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return GeneratorControlResponse(status="started", pid=pid)


@app.post(
    "/api/generator/start",
    response_model=Union[PlotterControlResponse, GeneratorLavfiStartResponse],
    response_model_exclude_none=True,
    tags=["generator"],
    summary="Start generator.py (images, resource grid, or FAPI FIFO)",
    description=(
        "Set **`feed`**: **images** (RGBA FIFO **generator.py**; optional nested **`images`** with **`pattern`** / **`preset`**), "
        "**resource_grid** (full-grid resource-grid named pipe writer only), or **fapi** (slot / FAPI FIFO only — no "
        "full-grid I/Q stream from **generator.py**). CUDA plotter must already be running. "
        "**400** on invalid body, missing FIFO for images mode, or plotter not running where required. "
        "**200** JSON omits **null** fields. **`feed`:** **`images`** includes **`lavfi_video`** / **`source`** when applicable; **`resource_grid`** / **`fapi`** return **`status`**, **`pid`**, and **`feed`** only."
    ),
)
def api_generator_start(body: GeneratorStartBody):
    try:
        if body.feed == "images":
            opts = body.images or GeneratorImagesOptions()
            fb = GeneratorFeedStartBody(mode=SYNTHETIC_IMAGES, source=opts.preset, pattern=opts.pattern)
            resolved = _resolved_lavfi_for_synthetic_images(fb)
            return _generator_lavfi_start_from_response(
                start_plotter_and_generators(
                    None,
                    synthetic_video_only=True,
                    video_generator_source=resolved,
                    echo_feed="images",
                )
            )
        if body.feed == "resource_grid":
            return _generator_control_ack(_generator_start_handler_resource_grid(), feed="resource_grid")
        return _generator_control_ack(_generator_start_handler_fapi_slot(), feed="fapi")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/generator/stop",
    response_model=PlotterControlResponse,
    response_model_exclude_none=True,
    tags=["generator"],
    summary="Stop generator.py and/or FFmpeg lavfi (images path)",
    description=(
        "Stops **FFmpeg** lavfi (if any), **generator.py** on **RESGRID_FIFO_IMAGES**, and the **resource_grid** / **fapi** **generator.py** child "
        "when running. Does **not** stop **resgrid_plotter**. Use **POST /api/stop** to stop the plotter too."
    ),
)
def api_generator_stop():
    return _plotter_control_from_generator_response(stop_generators())


def _synthetic_pattern_catalog() -> list[GeneratorPatternInfo]:
    return [
        GeneratorPatternInfo(id=0, label="Waves (default)"),
        GeneratorPatternInfo(id=1, label="Single tone (rotating)"),
        GeneratorPatternInfo(id=2, label="Gaussian blob"),
        GeneratorPatternInfo(id=3, label="Vortex (Gaussian × spiral phase)"),
    ]


@app.get(
    "/api/generator/patterns",
    response_model=GeneratorPatternsResponse,
    tags=["generator"],
    summary="List synthetic resource-grid pattern ids",
)
def api_generator_patterns_list():
    return GeneratorPatternsResponse(patterns=_synthetic_pattern_catalog())


@app.get(
    "/api/generator/channel-legend",
    response_model=FapiChannelLegendResponse,
    tags=["generator"],
    summary="FAPI channel legend",
    description=(
        "Static legend matching **`resgrid_plotter`** CUDA `channel_tint_rgb` "
        "(PHY ids **1…8**, labels, and base sRGB). Does not require a running plotter."
    ),
)
def api_generator_channel_legend():
    return _fapi_channel_legend_payload()


@app.put(
    "/api/generator/pattern",
    response_model=GeneratorPatternResponse,
    tags=["generator"],
    summary="Set synthetic resource-grid pattern",
    description=(
        "Writes pattern id to **generator** and **CUDA** control shared memory (uint32 @ offset 16). "
        "**SYNTHETIC_RESOURCE_GRID** only (**built-in** or **FIFO** path). Not used in **SYNTHETIC_FAPI** (slot mode)."
    ),
)
def api_generator_pattern_put(body: GeneratorPatternPutBody):
    with _lock:
        if not _is_plotter_running():
            raise HTTPException(
                status_code=400,
                detail="CUDA plotter (**resgrid_plotter**) is not running.",
            )
        try:
            v = _write_synthetic_pattern_shm_unlocked(body.pattern)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _log.info("Resource-grid pattern set to %s", v)
        return GeneratorPatternResponse(pattern=v)


@app.get(
    "/api/plotter/status",
    response_model=PlotterStatus,
    tags=["plotter"],
    summary="CUDA plotter and encode pipeline status",
    description=(
        "**resgrid_plotter** (when running), NVENC / pipeline rates from CUDA SHM, and JPEG capturer fields. "
        "Does **not** include **generator.py** / FFmpeg **images** rows — use **GET /api/generator/status**."
    ),
)
def api_plotter_status(request: Request):
    mode = get_generation_mode()
    plotter_shm_prod_hz: Optional[float] = None
    plotter_shm_rast_hz: Optional[float] = None
    plotter_shm_cmd_hz: Optional[float] = None
    resource_grid_pattern_id: Optional[int] = None
    plotter_shm_enc_frames: Optional[int] = None
    asm_in: Optional[float] = None
    asm_out: Optional[float] = None
    rast_in: Optional[float] = None
    rast_out: Optional[float] = None
    enc_in: Optional[float] = None
    enc_out: Optional[float] = None
    if mode == SYNTHETIC_IMAGES:
        if _is_ffmpeg_running():
            gen = True
            enc = effective_video_encoder()
            with _stats_lock:
                stats_snap = _stats.model_copy()
        elif _is_plotter_running():
            gen = True
            enc = "cuda_iq_pipeline"
            stats_snap = None
            plotter_shm_cmd_hz = _read_generator_producer_fps_shm_unlocked()
            resource_grid_pattern_id = _read_synthetic_pattern_from_shm_unlocked()
            plotter_shm_prod_hz, plotter_shm_rast_hz, plotter_shm_enc_frames = _read_plotter_shm_stream_metrics_unlocked()
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
                    out_fps = 24.0
            out_frame = enc_frame if enc_frame is not None else plotter_shm_enc_frames
            if out_fps is not None or out_frame is not None:
                stats_snap = GenerationStats(
                    fps=out_fps,
                    speed=enc_speed,
                    frame=out_frame,
                    bitrate_kbits=enc_bitrate,
                )
        else:
            gen = False
            enc = effective_video_encoder()
            with _stats_lock:
                stats_snap = None
    elif _uses_cuda_plotter(mode):
        gen = _is_plotter_running()
        enc = "cuda_iq_pipeline"
        stats_snap = None
        plotter_shm_cmd_hz = _read_generator_producer_fps_shm_unlocked()
        resource_grid_pattern_id = _read_synthetic_pattern_from_shm_unlocked()
        if _is_generation_mode_channel_based(mode):
            resource_grid_pattern_id = None
        if gen:
            plotter_shm_prod_hz, plotter_shm_rast_hz, plotter_shm_enc_frames = _read_plotter_shm_stream_metrics_unlocked()
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
                    out_fps = 24.0
            out_frame = enc_frame if enc_frame is not None else plotter_shm_enc_frames
            if out_fps is not None or out_frame is not None:
                stats_snap = GenerationStats(
                    fps=out_fps,
                    speed=enc_speed,
                    frame=out_frame,
                    bitrate_kbits=enc_bitrate,
                )
    else:
        gen = False
        enc = effective_video_encoder()
        stats_snap = None
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
    if gen and _is_plotter_running() and _plotter_cuda_shm is not None and len(_plotter_cuda_shm.buf) >= _IQ_CUDA_SHM_BYTES:
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
    grid_prb, grid_symbols = _plotter_resource_grid_dims()
    _log.info(
        "GET /api/plotter/status client=%s generating=%s process_mode=%s",
        peer,
        gen,
        mode,
        extra={"status_poll": True},
    )
    return PlotterStatus(
        generating=gen,
        cuda_plotter_active=_is_plotter_running(),
        video_encoder=enc,
        stats=stats_snap,
        publish_rtsp=MTX_RTSP_URL,
        mtx_path=_mtx_path,
        iq_prb=grid_prb,
        iq_symbols=grid_symbols,
        iq_producer_fps=plotter_shm_prod_hz,
        iq_plotter_fps=plotter_shm_rast_hz,
        iq_producer_command_fps=plotter_shm_cmd_hz,
        iq_pattern=resource_grid_pattern_id,
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
    "/api/stop",
    response_model=PlotterControlResponse,
    response_model_exclude_none=True,
    tags=["plotter"],
    summary="Stop plotter and standalone FFmpeg",
    description=(
        "Stop **FFmpeg** and/or **resgrid_plotter**, whichever is running. "
        "Does **not** stop **generator.py** (use **POST /api/generator/stop**). "
        "Optional JSON body may include `mode` for client symmetry; it is ignored."
    ),
)
def api_stop_all(body: StopAllBody = StopAllBody()):
    return _plotter_control_from_generator_response(stop_plotter_and_generators())


@app.post(
    "/api/plotter/start",
    response_model=PlotterControlResponse,
    response_model_exclude_none=True,
    tags=["plotter"],
    summary="Start CUDA plotter only",
    description=(
        "Start **`resgrid_plotter`** with **all** input FIFOs: **--fifo-images**, **--fifo-resource-grid**, **--fifo-fapi** "
        "(paths from env). The plotter ingests **all** FIFO inputs; **generator.py** chooses one producer at a time via **POST /api/generator/start** ``feed`` / ``--data-mode``. "
        "(**GENERATION_MODE** on **GET /api/generator/status** applies to the Python generator workflow). "
        "Optional JSON body: only **`encoder_fps`** (or empty **`{}`**). It overrides **CUDA_IQ_FPS** for **`--encoder-fps`**; pipeline **`--fps`** stays **CUDA_IQ_GEN_FPS**. "
        "Legacy **`mode`** or any other key returns **422**. "
        "Stops standalone FFmpeg first if it was running. "
        "Response: **`status`**, **`pid`**, optional **`encoder_fps`** (echoes **--encoder-fps** when the plotter starts or is already running). "
        "Process **GENERATION_MODE** is on **GET /api/generator/status**."
    ),
)
def api_plotter_start(body: PlotterStartBody = PlotterStartBody()):
    try:
        return _plotter_control_from_generator_response(
            start_plotter_and_generators(
                None,
                plotter_only=True,
                plotter_encoder_fps=body.encoder_fps,
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/api/plotter/stop",
    response_model=PlotterControlResponse,
    response_model_exclude_none=True,
    tags=["plotter"],
    summary="Stop CUDA plotter only",
    description=(
        "Terminate **`resgrid_plotter`** only. Does **not** stop **generator.py** (RGBA FIFO or FIFO writer); use **POST /api/generator/stop** "
        "or **POST /api/stop** as appropriate. "
        "Response: **`status`**, optional **`pid`**. Process **GENERATION_MODE**: **GET /api/generator/status**."
    ),
)
def api_plotter_stop(body: StopAllBody = StopAllBody()):
    return _plotter_control_from_generator_response(stop_plotter_and_generators(plotter_only=True))


@app.post(
    "/api/plotter/capturer",
    response_model=CapturerStartResponse,
    tags=["plotter"],
    summary="Capture raster frames to JPEG files (parallel to video encoder)",
    description=(
        "Arms the **file capturer** in **`resgrid_plotter`** via plotter CUDA SHM: each **raster** RGBA is pushed to the **video encoder** queue as usual and, in parallel, to a **bounded capturer queue** consumed by JPEG writer threads (**`RESGRID_CAPTURE_JPEG_THREADS`**) writing **`frame_*.jpg`**. "
        "If the capturer queue is full, the **oldest** queued RGBA is **dropped** so the raster never blocks (Swap / backlog). "
        "Progress: **capturer_inflight** / **capturer_written**. Requires **`RESGRID_CAPTURER_DIR`** (legacy **`RESGRID_ENCODER_SNAP_DIR`**). Max: **`RESGRID_CAPTURER_MAX_FRAMES`**."
    ),
)
def api_plotter_capturer(body: CapturerStartBody):
    with _lock:
        if not _is_plotter_running():
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
            and _plotter_cuda_shm is not None
            and len(_plotter_cuda_shm.buf) >= _SHM_CAPTURER_PHASE_OFF + 4
        ):
            struct.pack_into("I", _plotter_cuda_shm.buf, _SHM_CAPTURER_PHASE_OFF, 0)
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
    "/api/plotter/capturer/stop",
    response_model=CapturerStopResponse,
    tags=["plotter"],
    summary="Stop file capturer early",
    description=(
        "Requests the plotter to **stop capturing** after frames already taken from the raster: "
        "SHM **total** is capped to the current push count, queued RGBA is flushed to JPEG, then the session ends. "
        "No-op if not in **capturing** phase (returns **409** during **saving** or **idle**)."
    ),
)
def api_plotter_capturer_stop():
    with _lock:
        if not _is_plotter_running():
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


@app.get("/openapi.json", tags=["OpenAPI"], summary="OpenAPI 3 schema (per-request HLS example URLs)")
def openapi_endpoint(request: Request):
    return _generate_openapi_schema(request)


@app.get("/docs", tags=["OpenAPI"], summary="Swagger UI")
async def swagger_ui_route():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Swagger UI",
    )


@app.get("/redoc", tags=["OpenAPI"], summary="ReDoc")
async def redoc_ui_route():
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - ReDoc",
    )
