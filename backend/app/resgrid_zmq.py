"""ZeroMQ PUB for resource-grid IQ. Bind address is a full ipc:// or tcp:// URL."""

from __future__ import annotations

import zmq


def is_zmq_bind(path: str) -> bool:
    return path.startswith(("ipc://", "tcp://", "inproc://"))


def topic_for(path: str) -> str:
    if "nai" in path:
        return "resgrid_nai"
    if "rx" in path:
        return "resgrid_rx"
    return "resgrid_iq"


class ResgridZmqPublisher:
    """One logical frame per ``send_frame``: text header, then raw bytes."""

    def __init__(self, bind: str) -> None:
        if not is_zmq_bind(bind):
            raise ValueError(f"ZeroMQ bind must be ipc:// or tcp:// (got {bind!r})")
        self._bind = bind
        self._topic = topic_for(bind)
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.bind(bind)
        self._frame_seq = 0

    def close(self) -> None:
        self._pub.close(linger=0)
        self._ctx.term()

    def send_frame(self, data: bytes, *, prb: int = 0, symbols: int = 0) -> None:
        self._frame_seq = (self._frame_seq + 1) & 0xFFFFFFFF
        header = (
            f"{self._topic} seq={self._frame_seq} prb={prb} symbols={symbols} bytes={len(data)}"
        ).encode()
        self._pub.send_multipart([header, data], copy=False)
