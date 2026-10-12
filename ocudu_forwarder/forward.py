"""Publish OCUDU ul.grid slots on the RX resource-grid ZeroMQ contract.

One complete uplink grid becomes one multipart frame:

    resgrid_rx seq=<n> prb=<n> symbols=<n> bytes=<n> sfn=<n> slot=<n>
    float32 little-endian re, im, index = symbol * prb + prb

Each PRB is the mean of its 12 subcarriers on one RX port. Antenna -1 stores
the RMS magnitude in the real part and leaves the imaginary part at 0, matching
the cuPHY plotter export.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time

import numpy as np
import zmq

from ocudu_tools.e3 import E3Tools
from ocudu_tools.joint import SharedPool, StaleObject


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else int(raw)


def cbf16_grid(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    components = (np.frombuffer(payload, "<u2").astype("<u4") << 16).view("<f4")
    iq = components[::2] + 1j * components[1::2]
    return iq.reshape(shape)


def to_plotter_frame(iq: np.ndarray, antenna: int) -> tuple[bytes, int, int]:
    """Reduce [rx_port, symbol, subcarrier] to one float32 re/im sample per PRB."""
    if iq.ndim != 3:
        raise ValueError(f"ul.grid rank {iq.ndim} is not [port, symbol, subcarrier]")
    ports, symbols, subcarriers = iq.shape
    prb = subcarriers // 12
    if ports < 1 or symbols < 1 or prb < 1:
        raise ValueError(f"ul.grid shape {iq.shape} has no PRBs")
    tones = iq[:, :, : prb * 12].reshape(ports, symbols, prb, 12)
    if antenna < 0:
        magnitude = np.sqrt(np.mean(np.abs(tones) ** 2, axis=(0, 3))).astype(np.float32)
        real = magnitude.reshape(-1)
        imag = np.zeros_like(real)
    else:
        port = antenna if antenna < ports else 0
        mean = tones[port].mean(axis=-1)
        real = np.ascontiguousarray(mean.real, dtype=np.float32).reshape(-1)
        imag = np.ascontiguousarray(mean.imag, dtype=np.float32).reshape(-1)
    frame = np.empty(real.size * 2, dtype=np.float32)
    frame[0::2] = real
    frame[1::2] = imag
    return frame.tobytes(), prb, symbols


class RxPublisher:
    def __init__(self, bind: str) -> None:
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.bind(bind)
        self._seq = 0

    def send(self, body: bytes, prb: int, symbols: int, sfn: int, slot: int) -> bool:
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        header = (
            f"resgrid_rx seq={self._seq} prb={prb} symbols={symbols} "
            f"bytes={len(body)} sfn={sfn & 0xFFFF} slot={slot & 0xFFFF}"
        ).encode()
        try:
            self._pub.send(header, zmq.SNDMORE | zmq.DONTWAIT)
            self._pub.send(body, zmq.DONTWAIT, copy=False)
        except zmq.Again:
            return False
        return True

    def close(self) -> None:
        self._pub.close(linger=0)
        self._ctx.term()


def slot_timing(record: dict) -> tuple[int, int]:
    absolute = int(record.get("absolute_slot") or 0)
    per_frame = 10 * (1 << int(record.get("numerology") or 1))
    return (absolute // per_frame) & 0xFFFF, absolute % per_frame


def main() -> None:
    host = os.environ.get("OCUDU_E3_HOST", "127.0.0.1")
    port = env_int("OCUDU_E3_PORT", 36423)
    bind = os.environ.get("OCUDU_ZMQ_BIND", "tcp://0.0.0.0:15556")
    antenna = env_int("OCUDU_RX_ANTENNA", 0)
    shutdown = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())

    publisher = RxPublisher(bind)
    client = None
    pool = None
    sub = None
    print(
        json.dumps(
            {
                "event": "bind",
                "zmq": bind,
                "e3": f"{host}:{port}",
                "antenna": antenna,
            }
        ),
        flush=True,
    )
    try:
        while not shutdown.is_set():
            try:
                client = E3Tools(host, port)
                sub = client.request(
                    {"operation": "joint.subscribe", "cell_id": 0, "rnti": 0}
                )
                break
            except Exception as exc:
                print(json.dumps({"event": "subscribe_retry", "error": str(exc)}), flush=True)
                if client:
                    client.close()
                    client = None
                shutdown.wait(2)
        if shutdown.is_set() or client is None:
            return

        pool = SharedPool(sub)
        print(
            json.dumps(
                {
                    "event": "subscribed",
                    "socket": sub.get("socket"),
                    "capture_session": sub.get("capture_session"),
                    "pool_bytes": sub.get("pool_bytes"),
                }
            ),
            flush=True,
        )
        ack: list[list[int]] = []
        published = dropped = skipped = stale = 0
        previous = start = time.monotonic()
        while not shutdown.is_set():
            result = client.request(
                {"operation": "joint.poll", "token": sub["token"], "ack": ack}
            )
            ack = []
            for record in result.get("records", []):
                if "object_id" in record:
                    ack.append([record["object_id"], record["generation"]])
                if record.get("stream") != "ul.grid" or record.get("state") != "complete":
                    skipped += 1
                    continue
                try:
                    payload = pool.read(record)
                except StaleObject:
                    stale += 1
                    continue
                try:
                    iq = cbf16_grid(payload, tuple(record["shape"]))
                    body, prb, symbols = to_plotter_frame(iq, antenna)
                except (ValueError, KeyError) as exc:
                    skipped += 1
                    print(json.dumps({"event": "drop_grid", "error": str(exc)}), flush=True)
                    continue
                sfn, slot = slot_timing(record)
                if publisher.send(body, prb, symbols, sfn, slot):
                    published += 1
                else:
                    dropped += 1
            now = time.monotonic()
            if now - previous >= 1:
                print(
                    json.dumps(
                        {
                            "elapsed_s": round(now - start, 2),
                            "published": published,
                            "zmq_dropped": dropped,
                            "skipped": skipped,
                            "stale": stale,
                            "producer": result.get("metrics"),
                        }
                    ),
                    flush=True,
                )
                previous = now
            shutdown.wait(0.005)
    finally:
        if client is not None:
            try:
                if sub is not None:
                    client.request(
                        {
                            "operation": "joint.unsubscribe",
                            "token": sub["token"],
                        }
                    )
            except Exception:
                pass
            client.close()
        if pool is not None:
            pool.close()
        publisher.close()


if __name__ == "__main__":
    main()
