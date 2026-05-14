"""
POSIX message queue helpers for resource-grid visualizer (matches ``plotter_posix_mq.cpp`` / ``plotter_posix_mq.cuh``).

Chunk headers are **v2** by default: 28-byte little-endian layout with optional **prb** / **symbols** after ``byte_off``.
The plotter also accepts legacy **v1** (24-byte) chunks on read.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import struct
from typing import Optional

_O_CREAT = 0o100
_O_RDONLY = 0o0
_O_WRONLY = 0o1

_RESGRID_MQ_CHUNK_MAGIC = 0x52474D43  # 'RGMC' LE
_RESGRID_MQ_CHUNK_VERSION = 2
_CHUNK_HDR_FMT = "<IIIHHIIHH"  # magic, version, frame_seq, chunk_idx, n_chunks, logical_len, byte_off, prb, symbols


def _load_rt() -> ctypes.CDLL:
    for name in ("librt.so.1", "librt.so"):
        try:
            return ctypes.CDLL(name, use_errno=True)
        except OSError:
            continue
    libc = ctypes.util.find_library("c")
    return ctypes.CDLL(libc or "libc.so.6", use_errno=True)


_rt = _load_rt()


class _MqAttr(ctypes.Structure):
    _fields_ = [
        ("mq_flags", ctypes.c_long),
        ("mq_maxmsg", ctypes.c_long),
        ("mq_msgsize", ctypes.c_long),
        ("mq_curmsgs", ctypes.c_long),
    ]


def _read_proc_long(path: str, fallback: int) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return fallback


def effective_msgsize_cap() -> int:
    ev = (os.environ.get("RESGRID_MQ_MSGSIZE") or "").strip()
    if ev:
        try:
            v = int(ev, 10)
            if v >= 256:
                return v
        except ValueError:
            pass
    sysmax = _read_proc_long("/proc/sys/fs/mqueue/msgsize_max", 8192)
    return sysmax if sysmax > 0 else 8192


def effective_maxmsg() -> int:
    ev = (os.environ.get("RESGRID_MQ_MAXMSG") or "").strip()
    if ev:
        try:
            v = int(ev, 10)
            if v >= 1:
                return v
        except ValueError:
            pass
    sysmax = _read_proc_long("/proc/sys/fs/mqueue/msg_max", 10)
    return sysmax if sysmax > 0 else 10


def _setup_mq_functions() -> tuple:
    mq_open_fn = _rt.mq_open
    mq_open_fn.restype = ctypes.c_int

    mq_close = _rt.mq_close
    mq_close.restype = ctypes.c_int
    mq_close.argtypes = [ctypes.c_int]

    mq_unlink = _rt.mq_unlink
    mq_unlink.restype = ctypes.c_int
    mq_unlink.argtypes = [ctypes.c_char_p]

    mq_send = _rt.mq_send
    mq_send.restype = ctypes.c_int
    mq_send.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint]

    mq_getattr = _rt.mq_getattr
    mq_getattr.restype = ctypes.c_int
    mq_getattr.argtypes = [ctypes.c_int, ctypes.POINTER(_MqAttr)]

    def mq_open_simple(name_b: bytes, oflag: int) -> int:
        mq_open_fn.argtypes = [ctypes.c_char_p, ctypes.c_int]
        return int(mq_open_fn(name_b, oflag))

    def mq_open_create(name_b: bytes, oflag: int, mode: int, attr: _MqAttr) -> int:
        mq_open_fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(_MqAttr)]
        return int(mq_open_fn(name_b, oflag, mode, ctypes.byref(attr)))

    return mq_open_simple, mq_open_create, mq_close, mq_unlink, mq_send, mq_getattr


_mq_open_simple, _mq_open_create, _mq_close, _mq_unlink, _mq_send, _mq_getattr = _setup_mq_functions()


def _mq_err() -> OSError:
    err = ctypes.get_errno()
    return OSError(err, os.strerror(err))


def mq_unlink(name: str) -> None:
    if not name or not name.startswith("/"):
        return
    if _mq_unlink(name.encode()) != 0:
        e = ctypes.get_errno()
        if e != errno.ENOENT:
            raise _mq_err()


def mq_create_queue(name: str, maxmsg: Optional[int] = None, msgsize: Optional[int] = None) -> None:
    """Create (replace) a queue with **O_CREAT|O_RDONLY** then close, matching ``resgrid_mq_create_queue`` in C++."""
    if not name or not name.startswith("/"):
        raise ValueError(f"invalid mq name {name!r}")
    if maxmsg is None:
        maxmsg = effective_maxmsg()
    if msgsize is None:
        msgsize = effective_msgsize_cap()
    if maxmsg <= 0 or msgsize <= 0:
        raise ValueError("maxmsg and msgsize must be positive")
    attr = _MqAttr()
    attr.mq_flags = 0
    attr.mq_maxmsg = maxmsg
    attr.mq_msgsize = msgsize
    attr.mq_curmsgs = 0
    mq_unlink(name)
    mqd = _mq_open_create(name.encode(), _O_CREAT | _O_RDONLY, 0o666, attr)
    if mqd < 0:
        raise _mq_err()
    if _mq_close(mqd) != 0:
        raise _mq_err()


def pack_chunk_header(
    *,
    frame_seq: int,
    chunk_idx: int,
    n_chunks: int,
    logical_len: int,
    byte_off: int,
    prb: int = 0,
    symbols: int = 0,
) -> bytes:
    return struct.pack(
        _CHUNK_HDR_FMT,
        _RESGRID_MQ_CHUNK_MAGIC,
        _RESGRID_MQ_CHUNK_VERSION,
        frame_seq & 0xFFFFFFFF,
        chunk_idx & 0xFFFF,
        n_chunks & 0xFFFF,
        logical_len & 0xFFFFFFFF,
        byte_off & 0xFFFFFFFF,
        prb & 0xFFFF,
        symbols & 0xFFFF,
    )


class ResgridMqWriter:
    """Writer side: one logical frame per ``send_frame`` (chunked ``mq_send``)."""

    def __init__(self, name: str) -> None:
        if not name or not name.startswith("/"):
            raise ValueError(f"invalid mq name {name!r}")
        self._mqd: int = -1
        self._msgsize_cap = effective_msgsize_cap()
        mqd = _mq_open_simple(name.encode(), _O_WRONLY)
        if mqd < 0:
            raise _mq_err()
        self._mqd = mqd
        attr = _MqAttr()
        if _mq_getattr(self._mqd, ctypes.byref(attr)) == 0 and attr.mq_msgsize > 0:
            self._msgsize_cap = int(attr.mq_msgsize)
        self._frame_seq = 0

    @property
    def msgsize_cap(self) -> int:
        return self._msgsize_cap

    def close(self) -> None:
        if self._mqd >= 0:
            _mq_close(self._mqd)
            self._mqd = -1

    def __enter__(self) -> ResgridMqWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def send_frame(self, data: bytes, *, prb: int = 0, symbols: int = 0) -> None:
        if self._mqd < 0:
            raise RuntimeError("ResgridMqWriter is closed")
        if prb < 0 or prb > 0xFFFF or symbols < 0 or symbols > 0xFFFF:
            raise ValueError("prb and symbols must be in 0..65535")
        hdr_sz = struct.calcsize(_CHUNK_HDR_FMT)
        cap = self._msgsize_cap
        if cap <= hdr_sz:
            raise RuntimeError(f"mq_msgsize too small ({cap})")
        chunk_cap = cap - hdr_sz
        ln = len(data)
        n_chunks = 1 if ln == 0 else (ln + chunk_cap - 1) // chunk_cap
        if n_chunks > 65535:
            raise ValueError("logical frame too large for chunking")
        self._frame_seq = (self._frame_seq + 1) & 0xFFFFFFFF
        fs = self._frame_seq
        off = 0
        for ci in range(n_chunks):
            remain = ln - off
            take = min(chunk_cap, remain)
            hdr = pack_chunk_header(
                frame_seq=fs,
                chunk_idx=ci,
                n_chunks=n_chunks,
                logical_len=ln,
                byte_off=off,
                prb=prb,
                symbols=symbols,
            )
            msg = hdr + memoryview(data)[off : off + take].tobytes()
            buf = ctypes.create_string_buffer(msg, len(msg))
            if _mq_send(self._mqd, buf, len(msg), 0) != 0:
                raise _mq_err()
            off += take
