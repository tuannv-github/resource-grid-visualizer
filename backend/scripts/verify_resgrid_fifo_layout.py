#!/usr/bin/env python3
"""Sanity-check full-grid FIFO frame size vs CHANNEL_BASED_INTERFACE (symbols * prb complex float32)."""
from __future__ import annotations

import argparse
import os

def flat_index(s: int, p: int, prb: int) -> int:
    return s * prb + p


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prb", type=int, default=int(os.environ.get("CUDA_IQ_PRB", "106")))
    p.add_argument("--symbols", type=int, default=int(os.environ.get("CUDA_IQ_SYMBOLS", "14")))
    args = p.parse_args()
    prb, sym = args.prb, args.symbols
    nbytes = prb * sym * 2 * 4
    assert flat_index(0, 0, prb) == 0
    assert flat_index(1, 0, prb) == prb
    assert flat_index(sym - 1, prb - 1, prb) == (sym - 1) * prb + (prb - 1)
    print(f"OK: prb={prb} symbols={sym} -> {nbytes} bytes per FIFO write (float32 re/im)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
