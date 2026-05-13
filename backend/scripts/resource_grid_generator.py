#!/usr/bin/env python3
"""
Synthetic I/Q resource grid → named FIFO (raw float32, interleaved re, im per cell).

Optional **slot FIFO** (see ``docs/CHANNEL_BASED_INTERFACE.md``): pass ``--slot-fifo`` and ``--slot-pattern``
(``demo``, ``demo-multi``, ``symbol-columns``, or ``symbol-windows`` — ``symbol-windows`` / ``symbol-columns`` use
**shape-only** slot messages: ``n_complex=0``, no I/Q bytes; PHY color is flat per ``channel`` in the plotter.
``symbol-windows``: random PRB spans per symbol at startup (**``--slot-window-seed``**). **RNTI** is **random each slot**
(per OFDM symbol) via **``--slot-rnti-seed``** (optional; default nondeterministic).)
``--data-mode 3`` with no ``--fifo`` runs slot-only (same as ``--slot-only``).
``--slot-only`` writes only the slot FIFO (no IQ ``--fifo``).

Wire format (one "frame" per write cycle)
    For each cell index i in [0, prb * symbols):
        bytes[8*i : 8*i+4]   = real part (float32, little-endian)
        bytes[8*i+4 : 8*i+8] = imag part (float32, little-endian)
    Cell order matches resgrid_plotter / C++: row-major by symbol s then PRB p,
    i.e. index i = s * prb + p with float2 at that slot (same layout NumPy builds here).

Runtime
    The writer opens the FIFO O_WRONLY; on Linux this blocks until resgrid_plotter
    opens the read end, so the backend can spawn CUDA first, then this process.
    If CUDA_IQ_GEN_FPS > CUDA_IQ_FPS and the pipe fills, os.write blocks (backpressure).

Default ``--prb`` (when omitted) follows env ``CUDA_IQ_PRB`` (same as the FastAPI backend / ``resgrid_plotter``), else 273.

Optional ``--generator-shm`` (aliases ``--producer-shm``; IQ generator control block, 32 bytes; created by the backend)
    Offset 0:  double — measured generation rate (Hz); this script updates ~once per second.
    Offset 8:  double — commanded generation rate (Hz); backend/API writes; we read every
               loop iteration to adjust pacing. Invalid/missing values fall back to --fps.
    Offset 16: uint32 — synthetic pattern id (0…3); backend/API writes; we read each frame.
    This process must not shm_unlink; only attach, mmap via SharedMemory, and close on exit.
    ``multiprocessing.shared_memory.SharedMemory`` registers the segment with ``resource_tracker``; on process exit
    the tracker would call ``shm_unlink`` and remove the backend-owned segment. We unregister after attach so only
    the backend can unlink when the plotter stops.

Debugging
    ``--log-level INFO`` (default): stderr lines when pattern or commanded Hz changes, plus startup summary.
    ``--log-level DEBUG``: also ~1 Hz lines with achieved fps, target Hz, and active pattern (confirms SHM updates).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import struct
import sys
import time
from multiprocessing import resource_tracker
from multiprocessing import shared_memory

_LOG = logging.getLogger(__name__)


def _unregister_shared_memory_from_resource_tracker(shm: shared_memory.SharedMemory) -> None:
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
    print("resource_grid_generator: install numpy (e.g. pip install numpy)", file=sys.stderr)
    raise SystemExit(1) from e

# Must match IQ_PATTERN_MAX in backend main.py and plotter_common.cuh (resouce-grid-plotter).
_IQ_PATTERN_MAX = 3


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
    _LOG.info("mode 3 generator: PHY allocation by symbol (%s, %d PRBs × %d symbols)", slot_pattern, prb, n_sym)
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


def _p4_subdivide_largest(rects: list[tuple[int, int, int, int]], rng: np.random.Generator) -> None:
    if not rects:
        return
    bi = max(range(len(rects)), key=lambda i: (rects[i][1] - rects[i][0]) * (rects[i][3] - rects[i][2]))
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
    return struct.pack(SLOT_REGION_FMT, sym_start, sym_count, prb_start, prb_count, rnti, channel)


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


def write_slot_frame(fd: int, hdr: bytes, extension: bytes | None, payload: bytes) -> None:
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
    fd: int,
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
            raise ValueError(f"symbol-windows: invalid window s={s} (prb_start={p0}, prb_count={pc}, prb={prb})")
        ch = 1 + (s % 6)
        rn = 0x4600 + int(rnti_rng.integers(0, 9000))
        h = pack_slot_hdr(slot_id, s, 1, p0, pc, rnti=rn, channel=ch, n_complex=0)
        write_slot_frame(fd, h, None, b"")
    if log_labels:
        _LOG.debug("slot fifo symbol-windows slot_id=%s v1 msgs=%s", slot_id, n_sym)


def write_slot_demo_record(
    fd: int,
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
            raise ValueError(f"patch {patch_sym}×{patch_prb} must fit in grid {sym}×{prb}")
    if pattern == "demo":
        re = 0.9 * (0.5 + 0.5 * (slot_id & 1))
        pl = slot_payload_const(patch_sym, patch_prb, re, 0.1)
        channel = 1 + ((slot_id - 1) % 6)
        rnti = 0x4601 + (slot_id % 900)
        h = pack_slot_hdr(slot_id, 0, patch_sym, 0, patch_prb, rnti=rnti, channel=channel)
        write_slot_frame(fd, h, None, pl)
        if log_labels:
            _LOG.debug("slot fifo v1 slot_id=%s channel=%s rnti=%s", slot_id, channel, rnti)
    elif pattern == "demo-multi":
        re = 0.9 * (0.5 + 0.5 * (slot_id & 1))
        r0 = (0, 2, 0, 3, 0x4601 + (slot_id % 900), 1 + ((slot_id - 1) % 6))
        r1 = (4, 2, 10, 4, 0x5500 + slot_id, 1 + ((slot_id + 2) % 6))
        if r0[1] > sym or r0[3] > prb or r1[0] + r1[1] > sym or r1[2] + r1[3] > prb:
            raise ValueError("demo-multi regions exceed grid; use larger --symbols / --prb")
        regions_spec = [r0, r1]
        blobs = [
            pack_slot_region(ss, sc, ps, pc, rnti=rn, channel=ch)
            for (ss, sc, ps, pc, rn, ch) in regions_spec
        ]
        ext = struct.pack("<H", len(regions_spec)) + b"".join(blobs)
        n_complex = sum(sc_i * pc_i for (_ss, sc_i, _ps, pc_i, _rn, _ch) in regions_spec)
        pl_parts = [
            slot_payload_const(regions_spec[0][1], regions_spec[0][3], re, 0.1),
            slot_payload_const(regions_spec[1][1], regions_spec[1][3], re * 0.7, -0.05),
        ]
        pl = b"".join(pl_parts)
        assert len(pl) == n_complex * 8
        h = pack_slot_hdr_v2_multi(slot_id, n_complex)
        write_slot_frame(fd, h, ext, pl)
        if log_labels:
            for (_ss, _sc, _ps, _pc, rn, ch) in regions_spec:
                _LOG.debug("slot fifo v2 region channel=%s rnti=%s", ch, rn)
    elif pattern == "symbol-columns":
        if per_symbol_rnti is not None and len(per_symbol_rnti) != sym:
            raise ValueError(f"symbol-columns: per_symbol_rnti length {len(per_symbol_rnti)} != sym {sym}")
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
            pack_slot_region(ss, sc, ps, pc, rnti=rn, channel=ch) for (ss, sc, ps, pc, rn, ch) in regions_spec
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
    fifo: str,
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
    """Standalone loop: slot FIFO only (no IQ writes).

    If ``generator_shm`` is set, sleep between slot batches follows SHM offset 8 (commanded Hz) when finite,
    so REST GET/PUT ``/api/stream/iq/generator-fps`` drives slot rate; offset 0 gets measured slot-batch Hz (~1 Hz).
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
        symbol_windows = make_fixed_symbol_prb_windows(prb, symbols, min_prb_span=min_prb_span, rng=rng)
        _LOG.info("symbol-windows: %s PRB windows (fixed for this run)", symbol_windows)
    log_data_mode3_symbol_phy_plan(prb, symbols, pattern, symbol_windows)
    prod_shm: shared_memory.SharedMemory | None = None
    if generator_shm:
        prod_shm = _attach_generator_shm(generator_shm)
        _LOG.info("slot-only: generator control SHM %r (command Hz @8 drives batch interval when set)", generator_shm)
    _LOG.info("opening slot fifo %r (blocks until resgrid_plotter opens read side)...", fifo)
    fd = os.open(fifo, os.O_WRONLY)
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
                _LOG.info("slot FIFO reader closed (e.g. resgrid_plotter/FFmpeg exited); stopping slot demo writer")
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
        os.close(fd)
        if prod_shm is not None:
            prod_shm.close()


def _write_all(fd: int, data: bytes) -> None:
    """Write the full buffer to fd (pipes may return short writes; loop until done)."""
    mv = memoryview(data)
    while mv:
        try:
            n = os.write(fd, mv)
        except BrokenPipeError:
            raise BrokenPipeError("FIFO reader closed") from None
        if n == 0:
            raise OSError("write returned 0")
        mv = mv[n:]


def synth_frame_u8(frame_idx: int, prb: int, sym: int, pattern: int = 0) -> np.ndarray:
    """Build one IQ frame; ids match ``backend/resouce-grid-plotter`` (``plotter_raster.cu`` / common) and ``GET /api/stream/iq/patterns``.

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Synthetic resource grid → named pipes. "
            "Full-grid FIFO (--data-mode 2): each IQ write is one complete PRB×symbols I/Q raster. "
            "Slot-message / channel-based (--data-mode 3): demo slot allocation records on --slot-fifo "
            "(partial grids per message); optional IQ sidecar on --fifo. "
            "Uses optional SHM for pacing/pattern; --slot-only emits slot FIFO only."
        ),
    )
    ap.add_argument(
        "--fifo",
        default=None,
        help="IQ named pipe (required for full-grid FIFO / --data-mode 2, or slot-message mode with IQ sidecar / --data-mode 3)",
    )
    ap.add_argument(
        "--slot-only",
        action="store_true",
        help="Only write demo slot records to --slot-fifo (no IQ pipe). Same as --data-mode 3 with no --fifo.",
    )
    ap.add_argument(
        "--data-mode",
        type=int,
        choices=(2, 3),
        default=None,
        help=(
            "2: full-grid FIFO — each --fifo write is one complete PRB×symbols I/Q raster "
            "(matches resgrid_plotter --data-mode 2). "
            "3: slot messages / channel-based — require --slot-fifo; messages carry partial grids; "
            "omit --fifo for slot-only demo."
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
        "--slot-fifo",
        default=None,
        help="Optional slot named pipe (resgrid_plotter --slot-fifo). Emits demo slot records at --slot-interval.",
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
            "Seconds between slot demo records by wall clock; use 0 with --fifo to emit one slot record per IQ write "
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
        help="Write all-zero float32 I/Q frames (empty full-grid raster in FIFO mode); uses --generator-shm for pacing when set.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO log lines for each slot FIFO record",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log to stderr; DEBUG logs per-second IQ/pacing snapshot, INFO logs pattern/command Hz changes",
    )
    args = ap.parse_args()

    if args.slot_window_min_prb < 1:
        print("resource_grid_generator: --slot-window-min-prb must be >= 1", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [resource_grid_generator] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    if args.slot_only and args.fifo:
        _LOG.error("--slot-only cannot be combined with --fifo")
        sys.exit(1)

    dm = args.data_mode
    if dm is None:
        dm = 3 if args.slot_only else 2
    slot_only_run = args.slot_only or (dm == 3 and args.fifo is None)

    if slot_only_run and args.zero_fill:
        _LOG.error("slot-only (or --data-mode 3 without --fifo) conflicts with --zero-fill")
        sys.exit(1)

    if dm == 3 and not args.slot_fifo:
        _LOG.error("--data-mode 3 requires --slot-fifo")
        sys.exit(1)
    if dm == 2 and slot_only_run:
        _LOG.error("--data-mode 2 requires --fifo (use --data-mode 3 and --slot-fifo for slot-only)")
        sys.exit(1)

    if slot_only_run:
        if not args.slot_fifo:
            _LOG.error("slot-only run requires --slot-fifo")
            sys.exit(1)
        interval_eff = float(args.slot_interval)
        if interval_eff <= 0.0:
            interval_eff = 1.0 / max(1, args.fps)
        if args.prb < 1 or args.symbols < 1:
            _LOG.error("invalid prb/symbols")
            sys.exit(1)
        _LOG.info(
            "slot-only demo data_mode=%s slot_fifo=%r interval=%ss pattern=%s grid=%d×%d",
            dm,
            args.slot_fifo,
            interval_eff,
            args.slot_pattern,
            args.prb,
            args.symbols,
        )
        run_slot_fifo_standalone(
            args.slot_fifo,
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

    if not args.fifo:
        _LOG.error("--fifo is required for IQ generation unless using slot-only / --data-mode 3 without fifo")
        sys.exit(1)

    if args.prb < 1 or args.symbols < 1 or args.fps < 1:
        print("invalid prb/symbols/fps", file=sys.stderr)
        sys.exit(1)

    if args.slot_fifo and args.slot_interval < 0:
        _LOG.error("--slot-interval must be >= 0 when using --slot-fifo (0 = one slot record per IQ write)")
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

    slot_fd: int | None = None
    slot_id_counter = 1
    next_slot_deadline = time.perf_counter()

    # Blocks until a reader opens the FIFO for reading.
    fd = os.open(args.fifo, os.O_WRONLY)
    if args.slot_fifo:
        slot_fd = os.open(args.slot_fifo, os.O_WRONLY)
    symbol_windows: list[tuple[int, int]] | None = None
    slot_rnti_rng: np.random.Generator | None = None
    if args.slot_fifo and args.slot_pattern in ("symbol-windows", "symbol-columns"):
        slot_rnti_rng = np.random.default_rng(args.slot_rnti_seed)
    if args.slot_fifo and args.slot_pattern == "symbol-windows":
        rng_sw = np.random.default_rng(args.slot_window_seed)
        symbol_windows = make_fixed_symbol_prb_windows(
            args.prb, args.symbols, min_prb_span=args.slot_window_min_prb, rng=rng_sw
        )
        _LOG.info("symbol-windows: %s PRB windows (fixed for this run)", symbol_windows)
    if dm == 3 and args.slot_fifo and args.slot_pattern in ("symbol-windows", "symbol-columns"):
        log_data_mode3_symbol_phy_plan(args.prb, args.symbols, args.slot_pattern, symbol_windows)
    try:
        cli_pat = max(0, min(_IQ_PATTERN_MAX, int(args.pattern)))
        _LOG.info(
            "start data_mode=%s fifo=%r grid=%d×%d default_fps=%d frame_bytes=%d generator_shm=%s cli_pattern=%d slot_fifo=%s zero_fill=%s",
            dm,
            args.fifo,
            args.prb,
            args.symbols,
            args.fps,
            frame_bytes,
            args.generator_shm or "(none — pattern from --pattern)",
            cli_pat,
            args.slot_fifo or "(off)",
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
                    _LOG.warning("unpack pattern @16 failed; using cli_pattern=%s", cli_pat)
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
                _LOG.debug("SHM @16 raw uint32 now %s (interpreted pattern=%s)", raw_pat, pattern)
                last_raw_pat = raw_pat
            # One IQ "frame": all cells for this time step (float32 pairs). CUDA reads the same byte count per cycle.
            if args.zero_fill:
                buf = np.zeros(ncells * 2, dtype=np.float32)
            else:
                buf = synth_frame_u8(fr, args.prb, args.symbols, pattern)
            if buf.nbytes != frame_bytes:
                raise RuntimeError(f"frame size {buf.nbytes} != {frame_bytes}")
            # Single write of the full frame; FIFO may block if the reader (CUDA) is slower than us.
            try:
                _write_all(fd, buf.tobytes())
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
                            assert symbol_windows is not None and slot_rnti_rng is not None
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
                                rnti_rng=slot_rnti_rng if args.slot_pattern == "symbol-columns" else None,
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
        os.close(fd)
        if slot_fd is not None:
            os.close(slot_fd)
        if prod_shm is not None:
            prod_shm.close()


if __name__ == "__main__":
    main()
