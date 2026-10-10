#!/usr/bin/env python3
"""
**generator.py** — Python producer for the resource-grid stack over **POSIX message queues**:
**full I/Q resource grids** (``--mq-resource-grid``), **FAPI / slot** demo records (``--mq-fapi``), and **raw RGBA**
(``--mq-images``) for **resgrid_plotter**.

Synthetic I/Q → chunked **mq** frames (raw float32, interleaved re, im per cell). See **``docs/CHANNEL_BASED_INTERFACE.md``**
for slot-message layout; **``backend/plotter/readme.md``** for plotter **``--mq-*``** argv.

Optional **``--generator-shm``** (``--producer-shm``): 32-byte control block created by the FastAPI backend (measured Hz,
command Hz, pattern id).

Runtime: **``ResgridMqWriter``** opens each mq **O_WRONLY**; **``mq_send``** may block when the queue is full (backpressure).

Default **``--prb``** follows env **``CUDA_IQ_PRB``** (same as **``resgrid_plotter``** in the same container).

Debugging: **``--log-level INFO``** (default) or **``DEBUG``** for per-second pacing snapshots.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from multiprocessing import resource_tracker
from multiprocessing import shared_memory

_LOG = logging.getLogger(__name__)


def _unregister_shared_memory_from_resource_tracker(
    shm: shared_memory.SharedMemory,
) -> None:
    """Child attach registers POSIX SHM for unlink-on-exit; the block is owned by the backend — do not unlink."""
    name = getattr(shm, "_name", "") or ""
    if not name:
        return
    try:
        resource_tracker.unregister(name, "shared_memory")
    except Exception:
        pass


def _attach_generator_shm(generator_shm: str) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(name=generator_shm)
    _unregister_shared_memory_from_resource_tracker(shm)
    return shm


try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    print("generator: install numpy (e.g. pip install numpy)", file=sys.stderr)
    raise SystemExit(1) from e

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from app.resgrid_posix_mq import ResgridMqWriter
from app.resgrid_zmq import ResgridZmqPublisher, is_zmq_bind

# Must match IQ_PATTERN_MAX in backend main.py and plotter_common.cuh (`backend/plotter/`).
_IQ_PATTERN_MAX = 3
# Must match ``kResourceGridSlotIqMagic`` in ``backend/plotter/include/plotter_common.cuh``.
RESOURCE_GRID_SLOT_IQ_MAGIC = 0x52474953
# Default POSIX mq names (match ``app.main`` / ``resgrid_plotter`` defaults).
_DEFAULT_FAPI_MQ_PATH = (os.environ.get("CUDA_SLOT_MQ_PATH") or "").strip() or "/fapi"
_DEFAULT_RESOURCE_GRID_MQ_PATH = (os.environ.get("CUDA_IQ_MQ_PATH") or "").strip() or "/resource_grid"
_DEFAULT_IMAGES_MQ_PATH = (os.environ.get("RESGRID_MQ_IMAGES") or "").strip() or "/images"


def _int_env(name: str, fallback: int) -> int:
    """Parse optional integer env (same convention as ``app.main`` CUDA_* vars)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def chan_phy_name(ch_id: int) -> str:
    """PHY short name for channel id 1…8 (matches ``chan_phy_name`` in plotter_raster.cu)."""
    return {
        1: "SSB",
        2: "PRACH",
        3: "PDCCH",
        4: "PUCCH",
        5: "PDSCH",
        6: "PUSCH",
        7: "CSI-RS",
        8: "SRS",
    }.get(ch_id, str(ch_id))


def log_data_mode3_symbol_phy_plan(
    prb: int,
    n_sym: int,
    slot_pattern: str,
    symbol_windows: list[tuple[int, int]] | None,
) -> None:
    """Log one line per OFDM symbol: PHY name, PRB start/stop (inclusive), PRB count."""
    if slot_pattern not in ("symbol-windows", "symbol-columns"):
        return
    _LOG.info(
        "mode 3 generator: PHY allocation by symbol (%s, %d PRBs × %d symbols)",
        slot_pattern,
        prb,
        n_sym,
    )
    if slot_pattern == "symbol-windows":
        if not symbol_windows or len(symbol_windows) != n_sym:
            _LOG.warning("mode 3: symbol-windows plan skipped (bad window list)")
            return
        for s in range(n_sym):
            p0, pc = symbol_windows[s]
            ch = 1 + (s % 6)
            p_stop = p0 + pc - 1
            _LOG.info(
                "  symbol %2d  %-6s (chan %d)  prb_start=%4d  prb_stop=%4d  n_prb=%d",
                s,
                chan_phy_name(ch),
                ch,
                p0,
                p_stop,
                pc,
            )
        return
    for s in range(n_sym):
        ch = 1 + (s % 6)
        _LOG.info(
            "  symbol %2d  %-6s (chan %d)  prb_start=%4d  prb_stop=%4d  n_prb=%d",
            s,
            chan_phy_name(ch),
            ch,
            0,
            prb - 1,
            prb,
        )


def _p4_guillotine(
    rects: list[tuple[int, int, int, int]],
    s0: int,
    s1: int,
    p0: int,
    p1: int,
    depth: int,
    rng: np.random.Generator,
) -> None:
    sh, pw = s1 - s0, p1 - p0
    stop = depth >= 9 or sh < 3 or pw < 3 or ((sh * pw < 72) and depth >= 5)
    if stop:
        rects.append((s0, s1, p0, p1))
        return
    if sh >= pw and sh >= 6:
        lo, hi = s0 + 2, s1 - 2
        if lo < hi:
            mid = lo + int(rng.integers(0, hi - lo))
            _p4_guillotine(rects, s0, mid, p0, p1, depth + 1, rng)
            _p4_guillotine(rects, mid, s1, p0, p1, depth + 1, rng)
            return
    if pw >= 6:
        lo, hi = p0 + 2, p1 - 2
        if lo < hi:
            mid = lo + int(rng.integers(0, hi - lo))
            _p4_guillotine(rects, s0, s1, p0, mid, depth + 1, rng)
            _p4_guillotine(rects, s0, s1, mid, p1, depth + 1, rng)
            return
    rects.append((s0, s1, p0, p1))


def _p4_subdivide_largest(
    rects: list[tuple[int, int, int, int]], rng: np.random.Generator
) -> None:
    if not rects:
        return
    bi = max(
        range(len(rects)),
        key=lambda i: (rects[i][1] - rects[i][0]) * (rects[i][3] - rects[i][2]),
    )
    s0, s1, p0, p1 = rects.pop(bi)
    sh, pw = s1 - s0, p1 - p0
    if sh >= pw and sh >= 6:
        lo, hi = s0 + 2, s1 - 2
        if lo < hi:
            mid = lo + int(rng.integers(0, hi - lo))
            rects.append((s0, mid, p0, p1))
            rects.append((mid, s1, p0, p1))
            return
    if pw >= 6:
        lo, hi = p0 + 2, p1 - 2
        if lo < hi:
            mid = lo + int(rng.integers(0, hi - lo))
            rects.append((s0, s1, p0, mid))
            rects.append((s0, s1, mid, p1))
            return
    rects.append((s0, s1, p0, p1))


# --- Slot FIFO (docs/CHANNEL_BASED_INTERFACE.md) ---
SLOT_MAGIC = 0x31564153
SLOT_HDR_FMT = "<IHHIHHHHIBBHIII"
assert struct.calcsize(SLOT_HDR_FMT) == 40
SLOT_VERSION_1 = 1
SLOT_VERSION_2 = 2
SLOT_FLAG_MULTI_REGION = 1
SLOT_REGION_FMT = "<HHHHIB3x"
assert struct.calcsize(SLOT_REGION_FMT) == 16


def pack_slot_hdr(
    slot_id: int,
    sym_start: int,
    sym_count: int,
    prb_start: int,
    prb_count: int,
    *,
    rnti: int = 0,
    channel: int = 5,
    n_complex: int | None = None,
    version: int = SLOT_VERSION_1,
    flags: int = 0,
) -> bytes:
    if n_complex is None:
        n_complex = sym_count * prb_count
    return struct.pack(
        SLOT_HDR_FMT,
        SLOT_MAGIC,
        version,
        flags,
        slot_id,
        sym_start,
        sym_count,
        prb_start,
        prb_count,
        rnti,
        channel,
        0,
        0,
        n_complex,
        0,
        0,
    )


def pack_slot_region(
    sym_start: int,
    sym_count: int,
    prb_start: int,
    prb_count: int,
    *,
    rnti: int = 0,
    channel: int = 5,
) -> bytes:
    return struct.pack(
        SLOT_REGION_FMT, sym_start, sym_count, prb_start, prb_count, rnti, channel
    )


def pack_slot_hdr_v2_multi(slot_id: int, n_complex: int) -> bytes:
    return struct.pack(
        SLOT_HDR_FMT,
        SLOT_MAGIC,
        SLOT_VERSION_2,
        SLOT_FLAG_MULTI_REGION,
        slot_id,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        n_complex,
        0,
        0,
    )


def slot_payload_const(sym_count: int, prb_count: int, re: float, im: float) -> bytes:
    n = sym_count * prb_count
    inter = np.empty((n, 2), dtype=np.float32)
    inter[:, 0] = re
    inter[:, 1] = im
    return inter.reshape(-1).tobytes()


def write_slot_frame(
    fd: ResgridMqWriter, hdr: bytes, extension: bytes | None, payload: bytes
) -> None:
    buf = hdr + (extension or b"") + payload
    _write_all(fd, buf)


def make_fixed_symbol_prb_windows(
    prb: int,
    n_sym: int,
    *,
    min_prb_span: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """``n_sym`` tuples ``(prb_start, prb_count)``; each ``prb_count >= min(min_prb_span, prb)``."""
    if prb < 1 or n_sym < 1:
        raise ValueError("invalid prb or symbol count")
    span_lo = max(1, min(int(min_prb_span), prb))
    out: list[tuple[int, int]] = []
    for _ in range(n_sym):
        pc = int(rng.integers(span_lo, prb + 1))
        p0 = int(rng.integers(0, prb - pc + 1))
        out.append((p0, pc))
    return out


def write_slot_symbol_window_batch(
    fd: ResgridMqWriter,
    prb: int,
    n_sym: int,
    slot_id: int,
    windows: list[tuple[int, int]],
    *,
    rnti_rng: np.random.Generator,
    log_labels: bool = True,
) -> None:
    """``n_sym`` v1 **shape-only** records; fresh random RNTI per symbol **each slot** (each call)."""
    if len(windows) != n_sym:
        raise ValueError(f"symbol-windows: need {n_sym} windows, got {len(windows)}")
    for s in range(n_sym):
        p0, pc = windows[s]
        if pc < 1 or p0 < 0 or p0 + pc > prb:
            raise ValueError(
                f"symbol-windows: invalid window s={s} (prb_start={p0}, prb_count={pc}, prb={prb})"
            )
        ch = 1 + (s % 6)
        rn = 0x4600 + int(rnti_rng.integers(0, 9000))
        h = pack_slot_hdr(slot_id, s, 1, p0, pc, rnti=rn, channel=ch, n_complex=0)
        write_slot_frame(fd, h, None, b"")
    if log_labels:
        _LOG.debug("slot fifo symbol-windows slot_id=%s v1 msgs=%s", slot_id, n_sym)


def write_slot_demo_record(
    fd: ResgridMqWriter,
    prb: int,
    sym: int,
    slot_id: int,
    pattern: str,
    *,
    patch_sym: int,
    patch_prb: int,
    log_labels: bool = True,
    per_symbol_rnti: list[int] | None = None,
    rnti_rng: np.random.Generator | None = None,
) -> None:
    """One slot-FIFO record: ``demo`` = v1 rectangle; ``demo-multi`` = v2 two regions (with I/Q payload); ``symbol-columns`` = v2 multi-region **shape-only** (chan + RNTI; random RNTI per symbol when ``rnti_rng`` set)."""
    if pattern in ("demo", "demo-multi"):
        if patch_sym < 1 or patch_prb < 1 or patch_sym > sym or patch_prb > prb:
            raise ValueError(
                f"patch {patch_sym}×{patch_prb} must fit in grid {sym}×{prb}"
            )
    if pattern == "demo":
        re = 0.9 * (0.5 + 0.5 * (slot_id & 1))
        pl = slot_payload_const(patch_sym, patch_prb, re, 0.1)
        channel = 1 + ((slot_id - 1) % 6)
        rnti = 0x4601 + (slot_id % 900)
        h = pack_slot_hdr(
            slot_id, 0, patch_sym, 0, patch_prb, rnti=rnti, channel=channel
        )
        write_slot_frame(fd, h, None, pl)
        if log_labels:
            _LOG.debug(
                "slot fifo v1 slot_id=%s channel=%s rnti=%s", slot_id, channel, rnti
            )
    elif pattern == "demo-multi":
        re = 0.9 * (0.5 + 0.5 * (slot_id & 1))
        r0 = (0, 2, 0, 3, 0x4601 + (slot_id % 900), 1 + ((slot_id - 1) % 6))
        r1 = (4, 2, 10, 4, 0x5500 + slot_id, 1 + ((slot_id + 2) % 6))
        if r0[1] > sym or r0[3] > prb or r1[0] + r1[1] > sym or r1[2] + r1[3] > prb:
            raise ValueError(
                "demo-multi regions exceed grid; use larger --symbols / --prb"
            )
        regions_spec = [r0, r1]
        blobs = [
            pack_slot_region(ss, sc, ps, pc, rnti=rn, channel=ch)
            for (ss, sc, ps, pc, rn, ch) in regions_spec
        ]
        ext = struct.pack("<H", len(regions_spec)) + b"".join(blobs)
        n_complex = sum(
            sc_i * pc_i for (_ss, sc_i, _ps, pc_i, _rn, _ch) in regions_spec
        )
        pl_parts = [
            slot_payload_const(regions_spec[0][1], regions_spec[0][3], re, 0.1),
            slot_payload_const(regions_spec[1][1], regions_spec[1][3], re * 0.7, -0.05),
        ]
        pl = b"".join(pl_parts)
        assert len(pl) == n_complex * 8
        h = pack_slot_hdr_v2_multi(slot_id, n_complex)
        write_slot_frame(fd, h, ext, pl)
        if log_labels:
            for _ss, _sc, _ps, _pc, rn, ch in regions_spec:
                _LOG.debug("slot fifo v2 region channel=%s rnti=%s", ch, rn)
    elif pattern == "symbol-columns":
        if per_symbol_rnti is not None and len(per_symbol_rnti) != sym:
            raise ValueError(
                f"symbol-columns: per_symbol_rnti length {len(per_symbol_rnti)} != sym {sym}"
            )
        # One v2 **shape-only** record (`n_complex`=0): per-symbol full-PRB regions; RNTI random each slot if rnti_rng set.
        regions_spec: list[tuple[int, int, int, int, int, int]] = []
        for s in range(sym):
            ch = 1 + (s % 6)
            if per_symbol_rnti is not None:
                rn = per_symbol_rnti[s]
            elif rnti_rng is not None:
                rn = 0x4600 + int(rnti_rng.integers(0, 9000))
            else:
                rn = 0x4600 + (((slot_id - 1) * sym + s) % 9000)
            regions_spec.append((s, 1, 0, prb, rn, ch))
        blobs = [
            pack_slot_region(ss, sc, ps, pc, rnti=rn, channel=ch)
            for (ss, sc, ps, pc, rn, ch) in regions_spec
        ]
        ext = struct.pack("<H", len(regions_spec)) + b"".join(blobs)
        h = pack_slot_hdr_v2_multi(slot_id, 0)
        write_slot_frame(fd, h, ext, b"")
        if log_labels:
            _LOG.debug(
                "slot fifo v2 symbol-columns slot_id=%s regions=%d (fixed channel + RNTI per symbol)",
                slot_id,
                len(regions_spec),
            )
    else:
        raise ValueError(f"unknown slot pattern {pattern!r}")


def run_slot_fifo_standalone(
    mq_name: str,
    prb: int,
    symbols: int,
    interval: float,
    *,
    pattern: str = "demo",
    quiet: bool = False,
    patch_sym: int = 2,
    patch_prb: int = 3,
    min_prb_span: int = 100,
    window_seed: int | None = None,
    rnti_seed: int | None = None,
    generator_shm: str | None = None,
    default_fps: int = 200,
) -> None:
    """Standalone loop: slot / FAPI POSIX mq only (no IQ writes).

    If ``generator_shm`` is set, sleep between slot batches follows SHM offset 8 (commanded Hz) when finite,
    so REST GET/PUT ``/api/generator/fps`` drives slot rate; offset 0 gets measured slot-batch Hz (~1 Hz).
    """
    if prb < 1 or symbols < 1:
        raise SystemExit("invalid grid")
    if pattern in ("demo", "demo-multi") and (patch_sym > symbols or patch_prb > prb):
        raise SystemExit("demo patch exceeds grid")
    symbol_windows: list[tuple[int, int]] | None = None
    rnti_rng: np.random.Generator | None = None
    if pattern in ("symbol-windows", "symbol-columns"):
        rnti_rng = np.random.default_rng(rnti_seed)
    if pattern == "symbol-windows":
        rng = np.random.default_rng(window_seed)
        symbol_windows = make_fixed_symbol_prb_windows(
            prb, symbols, min_prb_span=min_prb_span, rng=rng
        )
        _LOG.info("symbol-windows: %s PRB windows (fixed for this run)", symbol_windows)
    log_data_mode3_symbol_phy_plan(prb, symbols, pattern, symbol_windows)
    prod_shm: shared_memory.SharedMemory | None = None
    if generator_shm:
        prod_shm = _attach_generator_shm(generator_shm)
        _LOG.info(
            "slot-only: generator control SHM %r (command Hz @8 drives batch interval when set)",
            generator_shm,
        )
    _LOG.info("opening slot POSIX mq %r...", mq_name)
    fd = ResgridMqWriter(mq_name)
    interval_t0 = time.perf_counter()
    slots_in_interval = 0
    try:
        slot_id = 1
        while True:
            try:
                if pattern == "symbol-windows":
                    assert symbol_windows is not None and rnti_rng is not None
                    write_slot_symbol_window_batch(
                        fd,
                        prb,
                        symbols,
                        slot_id,
                        symbol_windows,
                        rnti_rng=rnti_rng,
                        log_labels=not quiet,
                    )
                else:
                    write_slot_demo_record(
                        fd,
                        prb,
                        symbols,
                        slot_id,
                        pattern,
                        patch_sym=patch_sym,
                        patch_prb=patch_prb,
                        log_labels=not quiet,
                        rnti_rng=rnti_rng if pattern == "symbol-columns" else None,
                    )
            except BrokenPipeError:
                _LOG.info(
                    "slot mq reader closed (e.g. resgrid_plotter exited); stopping slot demo writer"
                )
                break
            sleep_s = interval if interval > 0 else 1.0 / max(1, default_fps)
            if prod_shm is not None:
                try:
                    (cmd_read,) = struct.unpack_from("d", prod_shm.buf, 8)
                    if math.isfinite(cmd_read) and cmd_read >= 1.0:
                        sleep_s = 1.0 / max(1.0, min(600.0, float(cmd_read)))
                except struct.error:
                    pass
            time.sleep(sleep_s)
            slot_id += 1
            slots_in_interval += 1
            now = time.perf_counter()
            if prod_shm is not None and now - interval_t0 >= 1.0:
                dt = now - interval_t0
                hz = slots_in_interval / dt if dt > 0 else 0.0
                struct.pack_into("d", prod_shm.buf, 0, hz)
                interval_t0 = now
                slots_in_interval = 0
    except KeyboardInterrupt:
        _LOG.info("exit")
    finally:
        fd.close()
        if prod_shm is not None:
            prod_shm.close()


def _write_all(
    fd: ResgridMqWriter | ResgridZmqPublisher,
    data: bytes,
    *,
    prb: int = 0,
    symbols: int = 0,
) -> None:
    """Write one logical frame on the POSIX mq (optional **prb** / **symbols** in v2 chunk header)."""
    try:
        fd.send_frame(data, prb=prb, symbols=symbols)
    except OSError as e:
        raise BrokenPipeError("MQ reader closed") from e


def synth_frame_u8(frame_idx: int, prb: int, sym: int, pattern: int = 0) -> np.ndarray:
    """Build one IQ frame; ids match ``backend/plotter/src`` (``plotter_raster.cu`` / common) and ``GET /api/generator/patterns``.

    - **0** — Waves (default): smooth phase over (PRB, symbol) like the original demo.
    - **1** — Single tone: identical I/Q in every cell (rotating phasor vs ``frame_idx``); flat magnitude map.
    - **2** — Gaussian blob: 2-D Gaussian envelope on the grid, slow rotating carrier phase.
    - **3** — Vortex: Gaussian × spiral phase ``2·atan2(cy,cx)`` plus frame term.

    PHY **channel** ids **1…8** (tints, RNTI labels) are **not** part of synthetic patterns; they come only from
    **slot FIFO** messages handled inside ``resgrid_plotter``. IQ FIFO carries magnitude/phase only
    (``chan`` plane stays zero → Viridis).

    Unknown ids are treated as **0**.

    Returns a 1D float32 array of length 2 * prb * sym: [re0, im0, re1, im1, …] with cell order
    symbol ``s`` then PRB ``p`` (index ``s * prb + p``). ``frame_idx`` animates over time.
    """
    p = np.arange(prb, dtype=np.float64)
    s = np.arange(sym, dtype=np.float64)
    p_grid, s_grid = np.meshgrid(p, s, indexing="xy")
    pat = int(pattern) if pattern >= 0 else 0
    if pat > _IQ_PATTERN_MAX:
        pat = 0

    if pat == 1:
        ph = 0.03 * float(frame_idx)
        vr = 0.9 * math.cos(ph)
        vi = 0.9 * math.sin(ph)
        re = np.full((sym, prb), vr, dtype=np.float32)
        im = np.full((sym, prb), vi, dtype=np.float32)
    elif pat == 2:
        cx = p_grid / float(prb) - 0.5
        cy = s_grid / float(sym) - 0.5
        g = np.exp(-(cx * cx + cy * cy) / (2.0 * 0.12**2))
        amp = 0.95 * g
        ph = 0.02 * float(frame_idx)
        re = (amp * np.cos(ph)).astype(np.float32)
        im = (amp * np.sin(ph)).astype(np.float32)
    elif pat == 3:
        cx = p_grid / float(prb) - 0.5
        cy = s_grid / float(sym) - 0.5
        r2 = cx * cx + cy * cy
        amp = 0.85 * np.exp(-r2 / (2.0 * 0.35**2))
        th = np.arctan2(cy, cx)
        ph = 2.0 * th + 0.04 * float(frame_idx)
        re = (amp * np.cos(ph)).astype(np.float32)
        im = (amp * np.sin(ph)).astype(np.float32)
    else:
        cx = p_grid / float(prb) - 0.5
        cy = s_grid / float(sym) - 0.5
        ph = 0.03 * float(frame_idx) + 6.2831853 * (cx * 1.3 + cy * 0.9)
        amp = 0.45 + 0.45 * np.sin(ph * 0.7 + cx * 4.0)
        re = (amp * np.cos(ph)).astype(np.float32)
        im = (amp * np.sin(ph)).astype(np.float32)

    inter = np.stack((re, im), axis=-1)
    return inter.reshape(-1)


def _producer_command_hz(
    fps_default: float, prod_shm: shared_memory.SharedMemory | None
) -> float:
    if prod_shm is None:
        return max(1.0, float(fps_default))
    try:
        (cmd_read,) = struct.unpack_from("d", prod_shm.buf, 8)
        if math.isfinite(cmd_read) and cmd_read >= 1.0:
            return float(cmd_read)
    except (struct.error, TypeError, ValueError):
        pass
    return max(1.0, float(fps_default))


def _ffmpeg_binary() -> str:
    raw = (os.environ.get("FFMPEG_BINARY") or "").strip()
    if raw:
        return raw
    w = shutil.which("ffmpeg")
    return w if w else "ffmpeg"


def _lavfi_graph_for_fifo_image_pattern(pattern: str) -> str:
    """Lavfi source strings; sizes must stay aligned with ``_LAVFI_SOURCE_PRESETS`` in ``backend/app/main.py``."""
    if pattern == "mandelbrot":
        return "mandelbrot=size=960x540,fps=30"
    if pattern == "smptebars":
        return "smptebars=size=1280x720:rate=30"
    return "testsrc=size=1280x720:rate=30"


def _wait_fifo_rgba_frame(
    proc: subprocess.Popen[bytes],
    latest_lock: threading.Lock,
    latest: list[bytes | None],
    frame_bytes: int,
    deadline_s: float,
) -> bytes | None:
    """Block until ``latest`` holds one full RGBA frame, FFmpeg exits, or ``deadline_s`` elapses."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < deadline_s:
        with latest_lock:
            buf = latest[0]
        if buf is not None and len(buf) == frame_bytes:
            return buf
        if proc.poll() is not None:
            return None
        time.sleep(0.02)
    return None


def _random_rgba_frame_bytes(width: int, height: int) -> bytes:
    """One frame: row-major RGBA ``uint8``, random RGB and opaque alpha (Python/NumPy, no FFmpeg)."""
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    # ``randint(..., out=)`` is not supported on older NumPy; assign into the view instead.
    rgba[..., :3] = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba.tobytes(order="C")


def run_fifo_images_main(
    mq_path: str,
    width: int,
    height: int,
    fps_default: int,
    pattern: str,
    generator_shm: str | None,
) -> None:
    """Write raw RGBA frames (row-major, ``width * height * 4`` bytes) to the POSIX mq for ``resgrid_plotter --mq-images``.

    For ``testsrc`` / ``smptebars`` / ``mandelbrot``, frames are decoded from **FFmpeg lavfi** (same nominal source sizes
    as backend ``_LAVFI_SOURCE_PRESETS``), scaled to ``width``×``height``. For ``randomnoise``, each frame is filled in
    Python with **NumPy** ``randint`` RGB at the output size (no FFmpeg). All paths use the same SHM / ``fps_default`` pacing.
    """
    if width < 1 or height < 1:
        raise SystemExit("mq-images: invalid rgba dimensions")
    if pattern not in ("testsrc", "smptebars", "mandelbrot", "randomnoise"):
        raise SystemExit(f"mq-images: unknown --image-pattern {pattern!r}")
    frame_bytes = width * height * 4
    prod_shm: shared_memory.SharedMemory | None = None
    if generator_shm:
        prod_shm = _attach_generator_shm(generator_shm)

    stop = False

    def on_stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)

    if pattern == "randomnoise":
        _LOG.info(
            "mq-images writer → %r size=%dx%d pattern=randomnoise (Python/NumPy random RGBA)",
            mq_path,
            width,
            height,
        )
        out = ResgridMqWriter(mq_path)
        try:
            fr = 0
            interval_t0 = time.perf_counter()
            frames_in_interval = 0
            next_t = time.perf_counter()
            while not stop:
                hz = _producer_command_hz(float(fps_default), prod_shm)
                period_s = 1.0 / hz
                raw = _random_rgba_frame_bytes(width, height)
                if len(raw) != frame_bytes:
                    _LOG.error("internal: rgba size mismatch %s != %s", len(raw), frame_bytes)
                    break
                try:
                    _write_all(out, raw, prb=0, symbols=0)
                except BrokenPipeError:
                    _LOG.warning("mq-images: reader closed")
                    return
                fr += 1
                frames_in_interval += 1
                now = time.perf_counter()
                dt = now - interval_t0
                if dt >= 1.0 and prod_shm is not None:
                    fps_m = frames_in_interval / dt if dt > 0 else 0.0
                    struct.pack_into("d", prod_shm.buf, 0, fps_m)
                    interval_t0 = now
                    frames_in_interval = 0
                    _LOG.debug(
                        "mq-images achieved_fps=%.2f target_hz=%.2f frame=%d",
                        fps_m,
                        hz,
                        fr,
                    )
                next_t += period_s
                slack = next_t - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_t = time.perf_counter()
        finally:
            out.close()
            if prod_shm is not None:
                prod_shm.close()
        return

    lavfi = _lavfi_graph_for_fifo_image_pattern(pattern)
    ff_bin = _ffmpeg_binary()
    cmd = [
        ff_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        lavfi,
        "-vf",
        f"scale={width}:{height}:flags=bilinear,format=rgba",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-f",
        "rawvideo",
        "-an",
        "-",
    ]

    _LOG.info(
        "mq-images writer → %r size=%dx%d pattern=%s lavfi=%r",
        mq_path,
        width,
        height,
        pattern,
        lavfi,
    )

    proc: subprocess.Popen[bytes] | None = None
    reader_t: threading.Thread | None = None
    latest_lock = threading.Lock()
    latest: list[bytes | None] = [None]
    fd: ResgridMqWriter | None = None

    def reader_main() -> None:
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                data = proc.stdout.read(frame_bytes)
                if len(data) != frame_bytes:
                    break
                with latest_lock:
                    latest[0] = data
        except Exception:
            _LOG.exception("mq-images ffmpeg stdout reader failed")

    def stderr_main() -> None:
        assert proc is not None and proc.stderr is not None
        try:
            for line in iter(proc.stderr.readline, b""):
                if stop:
                    break
                txt = line.decode("utf-8", errors="replace").rstrip()
                if txt:
                    _LOG.warning("mq-images ffmpeg: %s", txt)
        except Exception:
            _LOG.exception("mq-images ffmpeg stderr reader failed")

    try:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as e:
            raise SystemExit(
                f"mq-images: {ff_bin!r} not found; install ffmpeg or set FFMPEG_BINARY"
            ) from e
        reader_t = threading.Thread(target=reader_main, daemon=True)
        reader_t.start()
        threading.Thread(target=stderr_main, daemon=True).start()

        _LOG.info(
            "mq-images opening POSIX mq write %r (%d bytes/frame)",
            mq_path,
            frame_bytes,
        )
        fd = ResgridMqWriter(mq_path)

        first = _wait_fifo_rgba_frame(proc, latest_lock, latest, frame_bytes, 120.0)
        if first is None:
            rc = proc.poll()
            _LOG.error(
                "mq-images: no RGBA frame from FFmpeg within timeout (exit=%s); check stderr logs",
                rc,
            )
            return

        fr = 0
        interval_t0 = time.perf_counter()
        frames_in_interval = 0
        next_t = time.perf_counter()
        while not stop:
            hz = _producer_command_hz(float(fps_default), prod_shm)
            period_s = 1.0 / hz
            raw: bytes | None = None
            spin0 = time.perf_counter()
            while raw is None and not stop:
                with latest_lock:
                    raw = latest[0]
                if raw is not None:
                    break
                if proc.poll() is not None:
                    _LOG.error(
                        "mq-images ffmpeg exited (rc=%s) before a full frame",
                        proc.returncode,
                    )
                    return
                if time.perf_counter() - spin0 > 30.0:
                    _LOG.error("mq-images: timed out waiting for ffmpeg frame")
                    return
                time.sleep(0.005)
            if stop or raw is None:
                break
            if len(raw) != frame_bytes:
                _LOG.error("internal: rgba size mismatch %s != %s", len(raw), frame_bytes)
                break
            try:
                assert fd is not None
                _write_all(fd, raw, prb=0, symbols=0)
            except BrokenPipeError:
                _LOG.warning("mq-images: reader closed")
                return
            fr += 1
            frames_in_interval += 1
            now = time.perf_counter()
            dt = now - interval_t0
            if dt >= 1.0 and prod_shm is not None:
                fps_m = frames_in_interval / dt if dt > 0 else 0.0
                struct.pack_into("d", prod_shm.buf, 0, fps_m)
                interval_t0 = now
                frames_in_interval = 0
                _LOG.debug(
                    "mq-images achieved_fps=%.2f target_hz=%.2f frame=%d",
                    fps_m,
                    hz,
                    fr,
                )
            next_t += period_s
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.perf_counter()
    finally:
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
        if reader_t is not None:
            reader_t.join(timeout=2.0)
        if fd is not None:
            fd.close()
        if prod_shm is not None:
            prod_shm.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Synthetic resource grid → POSIX message queues for **resgrid_plotter** (**--mq-resource-grid**, **--mq-fapi**). "
            "Full-grid (--data-mode 2): each IQ send is one complete PRB×symbols I/Q raster. "
            "Slot / channel-based (--data-mode 3): demo slot allocation records on **--mq-fapi**; optional IQ sidecar on **--mq-resource-grid**. "
            "Raw RGBA: **--mq-images**. Uses optional SHM for pacing/pattern; **--slot-only** emits slot mq only."
        ),
    )
    ap.add_argument(
        "--mq-resource-grid",
        dest="mq_resource_grid",
        nargs="?",
        const=_DEFAULT_RESOURCE_GRID_MQ_PATH,
        default=None,
        metavar="NAME",
        help=(
            "Full IQ resource-grid POSIX mq (resgrid_plotter --mq-resource-grid). If the flag is present without "
            f"NAME, uses {_DEFAULT_RESOURCE_GRID_MQ_PATH!r} (env CUDA_IQ_MQ_PATH). "
            "Required for --data-mode 2 unless defaulted; optional IQ sidecar for --data-mode 3."
        ),
    )
    ap.add_argument(
        "--mq-resource-grid-mode",
        dest="mq_resource_grid_mode",
        choices=("full", "slot"),
        default="full",
        help=(
            "Standalone --mq-resource-grid only (no --mq-fapi): 'full' = one raw float32 I/Q raster per send. "
            "'slot' = framed slot-IQ stream for resgrid_plotter --mq-resource-grid-mode slot (header + complex "
            "payload; demo sends one full-grid snapshot per slot_id per paced tick)."
        ),
    )
    ap.add_argument(
        "--slot-only",
        action="store_true",
        help="Only write demo slot records to the FAPI mq (no resource-grid mq). Path: --mq-fapi or default in data-mode 3. Same as --data-mode 3 with no --mq-resource-grid.",
    )
    ap.add_argument(
        "--data-mode",
        type=int,
        choices=(2, 3),
        default=None,
        help=(
            "2: full-grid — each --mq-resource-grid send is one complete PRB×symbols I/Q raster "
            "(matches resgrid_plotter --mq-resource-grid). "
            "3: FAPI / slot messages — --mq-fapi optional (defaults to CUDA_SLOT_MQ_PATH); "
            "messages carry partial grids; omit --mq-resource-grid for slot-only demo."
        ),
    )
    ap.add_argument(
        "--prb",
        type=int,
        default=_int_env("CUDA_IQ_PRB", 273),
        help="PRB count (default: env CUDA_IQ_PRB, else 273; matches resgrid_plotter in the same container).",
    )
    ap.add_argument("--symbols", type=int, default=14)
    ap.add_argument("--fps", type=int, default=200, help="Generation / write rate (Hz)")
    ap.add_argument(
        "--generator-shm",
        "--producer-shm",
        dest="generator_shm",
        default=None,
        help="POSIX shared memory object name (parent-created); measured Hz @ 0, command Hz @ 8, pattern @ 16 (uint32)",
    )
    ap.add_argument(
        "--pattern",
        type=int,
        default=0,
        help="With --generator-shm, pattern id is read from SHM @16 each frame; this value is used when SHM is absent, and as fallback if that uint32 cannot be read.",
    )
    ap.add_argument(
        "--mq-fapi",
        dest="mq_fapi",
        nargs="?",
        const=_DEFAULT_FAPI_MQ_PATH,
        default=None,
        metavar="NAME",
        help=(
            "FAPI / slot-message POSIX mq (resgrid_plotter --mq-fapi). If the flag is present without NAME, "
            f"uses {_DEFAULT_FAPI_MQ_PATH!r} (env CUDA_SLOT_MQ_PATH). "
            "Emits demo slot records at --slot-interval."
        ),
    )
    ap.add_argument(
        "--mq-images",
        nargs="?",
        const=_DEFAULT_IMAGES_MQ_PATH,
        default=None,
        metavar="NAME",
        help=(
            "Raw RGBA POSIX mq for resgrid_plotter --mq-images. If the flag is present without NAME, uses "
            f"{_DEFAULT_IMAGES_MQ_PATH!r} (env RESGRID_MQ_IMAGES). "
            "Exclusive with --mq-resource-grid / --mq-fapi. Backend starts this mode from POST /api/generator/start when the plotter is running."
        ),
    )
    ap.add_argument(
        "--rgba-width",
        type=int,
        default=0,
        help="With --mq-images: frame width in pixels (must match padded plotter canvas).",
    )
    ap.add_argument(
        "--rgba-height",
        type=int,
        default=0,
        help="With --mq-images: frame height in pixels (must match padded plotter canvas).",
    )
    ap.add_argument(
        "--image-pattern",
        choices=("testsrc", "smptebars", "mandelbrot", "randomnoise"),
        default="testsrc",
        help=(
            "With --mq-images: image source (maps from API preset). testsrc/smptebars/mandelbrot use FFmpeg lavfi "
            "then scale to --rgba-width/height; randomnoise fills each frame with NumPy random RGB in Python (no FFmpeg)."
        ),
    )
    ap.add_argument(
        "--slot-pattern",
        choices=("demo", "demo-multi", "symbol-columns", "symbol-windows"),
        default="symbol-windows",
        help=(
            "Slot demo: v1 patch; v2 legacy two blobs; symbol-columns (one v2 record, full PRB); "
            "symbol-windows (one v1 record per OFDM symbol per slot, random PRB span per symbol at startup, "
            ">= min(--slot-window-min-prb, prb) wide)."
        ),
    )
    ap.add_argument(
        "--slot-window-min-prb",
        type=int,
        default=_int_env("CUDA_SLOT_WINDOW_MIN_PRB", 50),
        help="symbol-windows: minimum PRB span per symbol (capped by grid --prb; default env CUDA_SLOT_WINDOW_MIN_PRB or 50).",
    )
    ap.add_argument(
        "--slot-window-seed",
        type=int,
        default=None,
        help="symbol-windows: RNG seed for PRB windows only (default: nondeterministic).",
    )
    ap.add_argument(
        "--slot-rnti-seed",
        type=int,
        default=None,
        help="symbol-windows / symbol-columns: RNG seed for per-slot RNTI draws (default: nondeterministic).",
    )
    ap.add_argument(
        "--slot-interval",
        type=float,
        default=2.0,
        help=(
            "Seconds between slot demo records by wall clock; use 0 with --mq-resource-grid to emit one slot record per IQ write "
            "(matches --fps). Backend mode 3 defaults to 1/plotter_fps via CUDA_SLOT_DEMO_INTERVAL=fps."
        ),
    )
    ap.add_argument(
        "--slot-demo-patch-sym",
        type=int,
        default=2,
        help="For --slot-pattern demo: patch symbol count (v1 rectangle).",
    )
    ap.add_argument(
        "--slot-demo-patch-prb",
        type=int,
        default=3,
        help="For --slot-pattern demo: patch PRB count (v1 rectangle).",
    )
    ap.add_argument(
        "--zero-fill",
        action="store_true",
        help="Write all-zero float32 I/Q frames (empty full-grid raster); uses --generator-shm for pacing when set.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO log lines for each slot mq record",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log to stderr; DEBUG logs per-second IQ/pacing snapshot, INFO logs pattern/command Hz changes",
    )
    args = ap.parse_args()

    if args.slot_window_min_prb < 1:
        print("generator: --slot-window-min-prb must be >= 1", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [generator] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    if args.mq_images:
        if args.mq_resource_grid or args.mq_fapi or args.slot_only:
            _LOG.error(
                "--mq-images cannot be combined with --mq-resource-grid, --mq-fapi, or --slot-only"
            )
            sys.exit(1)
        if args.rgba_width < 1 or args.rgba_height < 1:
            _LOG.error(
                "--mq-images requires --rgba-width and --rgba-height (positive integers)"
            )
            sys.exit(1)
        if args.fps < 1:
            _LOG.error("--fps must be >= 1 for --mq-images")
            sys.exit(1)
        run_fifo_images_main(
            args.mq_images,
            args.rgba_width,
            args.rgba_height,
            args.fps,
            args.image_pattern,
            args.generator_shm,
        )
        return

    if args.slot_only and args.mq_resource_grid:
        _LOG.error("--slot-only cannot be combined with --mq-resource-grid")
        sys.exit(1)

    if args.mq_fapi and args.mq_resource_grid_mode == "slot":
        _LOG.error("--mq-resource-grid-mode slot applies only without --mq-fapi")
        sys.exit(1)

    dm = args.data_mode
    if dm is None:
        dm = 3 if args.slot_only else 2
    slot_only_run = args.slot_only or (dm == 3 and args.mq_resource_grid is None)

    if dm == 3 and args.mq_fapi is None:
        args.mq_fapi = _DEFAULT_FAPI_MQ_PATH

    if slot_only_run and args.zero_fill:
        _LOG.error(
            "slot-only (or --data-mode 3 without --mq-resource-grid) conflicts with --zero-fill"
        )
        sys.exit(1)

    if dm == 2 and slot_only_run:
        _LOG.error(
            "--data-mode 2 requires --mq-resource-grid (use --data-mode 3 and --mq-fapi for slot-only)"
        )
        sys.exit(1)

    if slot_only_run:
        interval_eff = float(args.slot_interval)
        if interval_eff <= 0.0:
            interval_eff = 1.0 / max(1, args.fps)
        if args.prb < 1 or args.symbols < 1:
            _LOG.error("invalid prb/symbols")
            sys.exit(1)
        _LOG.info(
            "slot-only demo data_mode=%s slot_fifo=%r interval=%ss pattern=%s grid=%d×%d",
            dm,
            args.mq_fapi,
            interval_eff,
            args.slot_pattern,
            args.prb,
            args.symbols,
        )
        run_slot_fifo_standalone(
            args.mq_fapi,
            args.prb,
            args.symbols,
            interval_eff,
            pattern=args.slot_pattern,
            quiet=args.quiet,
            patch_sym=args.slot_demo_patch_sym,
            patch_prb=args.slot_demo_patch_prb,
            min_prb_span=args.slot_window_min_prb,
            window_seed=args.slot_window_seed,
            rnti_seed=args.slot_rnti_seed,
            generator_shm=args.generator_shm,
            default_fps=max(1, int(args.fps)),
        )
        return

    if dm == 2 and args.mq_resource_grid is None:
        args.mq_resource_grid = _DEFAULT_RESOURCE_GRID_MQ_PATH

    if not args.mq_resource_grid:
        _LOG.error(
            "--mq-resource-grid is required for IQ generation unless using slot-only / --data-mode 3 without mq"
        )
        sys.exit(1)

    if args.prb < 1 or args.symbols < 1 or args.fps < 1:
        print("invalid prb/symbols/fps", file=sys.stderr)
        sys.exit(1)

    if args.mq_fapi and args.slot_interval < 0:
        _LOG.error(
            "--slot-interval must be >= 0 when using --mq-fapi (0 = one slot record per IQ write)"
        )
        sys.exit(1)

    ncells = args.prb * args.symbols
    frame_bytes = ncells * 2 * 4  # two float32 per cell (re, im)

    stop = False

    def on_stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)

    # Pacing: target wall-clock interval between frame starts (updated if SHM command changes).
    period_s = 1.0 / float(args.fps)
    rate_hz = float(args.fps)
    # next_t is the ideal start time for the *next* frame (deadline scheduling).
    next_t = time.perf_counter()
    fr = 0

    # Rolling window (~1 s) to estimate achieved FPS into SHM offset 0 for the API/UI.
    interval_t0 = time.perf_counter()
    frames_in_interval = 0

    prod_shm: shared_memory.SharedMemory | None = None
    if args.generator_shm:
        prod_shm = _attach_generator_shm(args.generator_shm)

    slot_fd: ResgridMqWriter | None = None
    slot_id_counter = 1
    next_slot_deadline = time.perf_counter()

    fd = (
        ResgridZmqPublisher(args.mq_resource_grid)
        if is_zmq_bind(args.mq_resource_grid)
        else ResgridMqWriter(args.mq_resource_grid)
    )
    if args.mq_fapi:
        slot_fd = ResgridMqWriter(args.mq_fapi)
    symbol_windows: list[tuple[int, int]] | None = None
    slot_rnti_rng: np.random.Generator | None = None
    if args.mq_fapi and args.slot_pattern in ("symbol-windows", "symbol-columns"):
        slot_rnti_rng = np.random.default_rng(args.slot_rnti_seed)
    if args.mq_fapi and args.slot_pattern == "symbol-windows":
        rng_sw = np.random.default_rng(args.slot_window_seed)
        symbol_windows = make_fixed_symbol_prb_windows(
            args.prb, args.symbols, min_prb_span=args.slot_window_min_prb, rng=rng_sw
        )
        _LOG.info("symbol-windows: %s PRB windows (fixed for this run)", symbol_windows)
    if (
        dm == 3
        and args.mq_fapi
        and args.slot_pattern in ("symbol-windows", "symbol-columns")
    ):
        log_data_mode3_symbol_phy_plan(
            args.prb, args.symbols, args.slot_pattern, symbol_windows
        )
    try:
        cli_pat = max(0, min(_IQ_PATTERN_MAX, int(args.pattern)))
        _LOG.info(
            "start data_mode=%s fifo=%r grid=%d×%d default_fps=%d frame_bytes=%d generator_shm=%s cli_pattern=%d slot_fifo=%s zero_fill=%s",
            dm,
            args.mq_resource_grid,
            args.prb,
            args.symbols,
            args.fps,
            frame_bytes,
            args.generator_shm or "(none — pattern from --pattern)",
            cli_pat,
            args.mq_fapi or "(off)",
            args.zero_fill,
        )

        last_pattern: int | None = None
        last_rate_hz: float | None = None
        last_raw_pat: int | None = None

        while not stop:
            pattern = max(0, min(_IQ_PATTERN_MAX, int(args.pattern)))
            raw_pat: int | None = None
            cmd_read = float("nan")
            if prod_shm is not None:
                cmd_read = struct.unpack_from("d", prod_shm.buf, 8)[0]
                if math.isfinite(cmd_read) and cmd_read >= 1.0:
                    rate_hz = max(1.0, min(600.0, float(cmd_read)))
                else:
                    rate_hz = float(args.fps)
                    _LOG.debug(
                        "SHM cmd @8 not finite or <1 (%r); pacing fallback --fps=%s",
                        cmd_read,
                        args.fps,
                    )
                period_s = 1.0 / rate_hz
                try:
                    (raw_pat_u,) = struct.unpack_from("I", prod_shm.buf, 16)
                    raw_pat = int(raw_pat_u)
                    if raw_pat <= _IQ_PATTERN_MAX:
                        pattern = raw_pat
                    else:
                        _LOG.warning(
                            "SHM uint32 @16 out of range (%s); keeping pattern=%s (from --pattern baseline)",
                            raw_pat,
                            pattern,
                        )
                except struct.error:
                    pattern = cli_pat
                    _LOG.warning(
                        "unpack pattern @16 failed; using cli_pattern=%s", cli_pat
                    )
            else:
                cmd_read = float("nan")

            if not args.zero_fill:
                if last_pattern is None or pattern != last_pattern:
                    _LOG.info(
                        "pattern %s -> %s (frame=%d raw_shm=%s prod_shm=%s)",
                        last_pattern if last_pattern is not None else "—",
                        pattern,
                        fr,
                        raw_pat,
                        bool(prod_shm),
                    )
                    last_pattern = pattern
            if last_rate_hz is None or abs(rate_hz - last_rate_hz) > 0.5:
                _LOG.info(
                    "commanded pace ~%.2f Hz (SHM cmd @8=%r, frame=%d)",
                    rate_hz,
                    cmd_read,
                    fr,
                )
                last_rate_hz = rate_hz
            if raw_pat is not None and raw_pat != last_raw_pat:
                _LOG.debug(
                    "SHM @16 raw uint32 now %s (interpreted pattern=%s)",
                    raw_pat,
                    pattern,
                )
                last_raw_pat = raw_pat
            # One IQ "frame": all cells for this time step (float32 pairs). CUDA reads the same byte count per cycle.
            if args.zero_fill:
                buf = np.zeros(ncells * 2, dtype=np.float32)
            else:
                buf = synth_frame_u8(fr, args.prb, args.symbols, pattern)
            if buf.nbytes != frame_bytes:
                raise RuntimeError(f"frame size {buf.nbytes} != {frame_bytes}")
            if args.mq_fapi is None and args.mq_resource_grid_mode == "slot":
                sid = (fr + 1) & 0xFFFFFFFF
                pkt = struct.pack(
                    "<IIHHHHI",
                    RESOURCE_GRID_SLOT_IQ_MAGIC,
                    sid,
                    0,
                    args.symbols,
                    0,
                    args.prb,
                    ncells,
                ) + np.ascontiguousarray(buf, dtype=np.float32).tobytes()
                try:
                    _write_all(fd, pkt, prb=args.prb, symbols=args.symbols)
                except BrokenPipeError:
                    _LOG.info("IQ FIFO reader closed (plotter exited); stopping generator")
                    break
            else:
                try:
                    _write_all(fd, buf.tobytes(), prb=args.prb, symbols=args.symbols)
                except BrokenPipeError:
                    _LOG.info("IQ FIFO reader closed (plotter exited); stopping generator")
                    break
            fr += 1

            if slot_fd is not None:
                now = time.perf_counter()
                slot_tick = args.slot_interval == 0.0 or now >= next_slot_deadline
                if slot_tick:
                    try:
                        if args.slot_pattern == "symbol-windows":
                            assert (
                                symbol_windows is not None and slot_rnti_rng is not None
                            )
                            write_slot_symbol_window_batch(
                                slot_fd,
                                args.prb,
                                args.symbols,
                                slot_id_counter,
                                symbol_windows,
                                rnti_rng=slot_rnti_rng,
                                log_labels=not args.quiet,
                            )
                        else:
                            write_slot_demo_record(
                                slot_fd,
                                args.prb,
                                args.symbols,
                                slot_id_counter,
                                args.slot_pattern,
                                patch_sym=args.slot_demo_patch_sym,
                                patch_prb=args.slot_demo_patch_prb,
                                log_labels=not args.quiet,
                                rnti_rng=slot_rnti_rng
                                if args.slot_pattern == "symbol-columns"
                                else None,
                            )
                    except ValueError as e:
                        _LOG.warning("slot fifo: %s", e)
                    except BrokenPipeError:
                        _LOG.info("slot FIFO reader closed; stopping generator")
                        break
                    slot_id_counter += 1
                    if args.slot_interval > 0.0:
                        next_slot_deadline = now + args.slot_interval

            frames_in_interval += 1
            now = time.perf_counter()
            if now - interval_t0 >= 1.0:
                dt = now - interval_t0
                # Achieved IQ write throughput over the last ~1 s (not the commanded rate from SHM offset 16).
                fps_m = frames_in_interval / dt if dt > 0 else 0.0
                if prod_shm is not None:
                    # Offset 0: measured generator FPS for the API/UI; CUDA plotter rate lives on cuda SHM @ 8.
                    struct.pack_into("d", prod_shm.buf, 0, fps_m)
                _LOG.debug(
                    "1s tick frames=%d achieved_fps=%.2f target_hz=%.2f pattern=%d frame_idx=%d shm=%s",
                    frames_in_interval,
                    fps_m,
                    rate_hz,
                    pattern,
                    fr,
                    args.generator_shm or "off",
                )
                interval_t0 = now
                frames_in_interval = 0
            # Sleep until next deadline; if we slipped (slack <= 0), realign next_t to now.
            next_t += period_s
            slack = next_t - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.perf_counter()
    finally:
        fd.close()
        if slot_fd is not None:
            slot_fd.close()
        if prod_shm is not None:
            prod_shm.close()


if __name__ == "__main__":
    main()
