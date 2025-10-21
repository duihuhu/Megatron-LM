#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-process /dev/shm write bandwidth tester.

Features:
- Parallel writers using multiprocessing (process-based, bypass GIL).
- Each worker writes to a disjoint file region with os.pwrite to avoid lock contention.
- Preallocation with posix_fallocate (fallback to ftruncate).
- Warmup runs to stabilize code paths and page faults.
- Configurable total size, workers, block size, pattern (zeros or random), and fsync.
- Optional NUMA pin via taskset/numactl (user-run external), not handled here.

Example:
  python shm_parallel_bw.py --size-gb 4 --workers 8 --block-mb 64 --warmup 2
"""

import os
import time
import argparse
import multiprocessing as mp
from typing import Tuple

def _preallocate(path: str, total_bytes: int):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        try:
            os.posix_fallocate(fd, 0, total_bytes)  # fast and dense allocation
        except AttributeError:
            # posix_fallocate not available
            os.ftruncate(fd, total_bytes)
        except OSError:
            # fall back if file system doesn't support fallocate (unlikely on tmpfs)
            os.ftruncate(fd, total_bytes)
    finally:
        os.close(fd)

def _make_buffer(size: int, pattern: str) -> bytes:
    if pattern == "zeros":
        return b"\x00" * size
    elif pattern == "random":
        # os.urandom can be slower than memcpy; choose wisely
        return os.urandom(size)
    else:
        # simple repeating pattern to avoid zero-optimization worries
        base = (b"0123456789ABCDEF" * ((size // 16) + 1))[:size]
        return base

def _worker(path: str, offset: int, bytes_to_write: int, block_size: int, pattern: str, do_fsync: bool):
    fd = os.open(path, os.O_WRONLY)
    try:
        buf = _make_buffer(block_size, pattern)
        remaining = bytes_to_write
        off = offset
        # write in blocks to cap per-process memory usage
        while remaining > 0:
            n = block_size if remaining >= block_size else remaining
            if n != len(buf):
                # resize buffer for tail write
                buf = _make_buffer(n, pattern)
            os.pwrite(fd, buf, off)
            off += n
            remaining -= n
        if do_fsync:
            os.fsync(fd)
    finally:
        os.close(fd)

def _run_once(path: str, total_bytes: int, workers: int, block_size: int, pattern: str, fsync: bool) -> Tuple[float, float]:
    # divide work evenly
    per = total_bytes // workers
    # last worker takes the tail
    ranges = [(i * per, per if i < workers - 1 else total_bytes - i * per) for i in range(workers)]

    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        pool.starmap(_worker, [(path, off, sz, block_size, pattern, fsync) for (off, sz) in ranges])
    dt = time.perf_counter() - t0
    gbps = total_bytes / dt / 1e9
    return gbps, dt

def main():
    parser = argparse.ArgumentParser(description="Parallel /dev/shm write bandwidth tester")
    parser.add_argument("--path", type=str, default="/dev/shm/shm_bw_test.bin", help="Target file path in tmpfs")
    parser.add_argument("--size-gb", type=float, default=1.0, help="Total file size to write (GB)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--block-mb", type=int, default=64, help="Per-write block size (MB)")
    parser.add_argument("--pattern", type=str, default="zeros", choices=["zeros", "random", "pattern"],
                        help="Data pattern to write")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (write+delete, not timed)")
    parser.add_argument("--runs", type=int, default=1, help="Timed runs to average")
    parser.add_argument("--fsync", action="store_true", help="Call fsync in each worker (not usually needed for tmpfs)")
    parser.add_argument("--keep", action="store_true", help="Keep the file after test (default remove)")

    args = parser.parse_args()

    total_bytes = int(args.size-gb * 1e9) if hasattr(args, "size-gb") else int(args.size_gb * 1e9)
    # argparse converts '--size-gb' to 'size_gb'
    total_bytes = int(args.size_gb * 1e9)

    block_size = args.block_mb * 1024 * 1024

    # sanity checks
    if total_bytes <= 0:
        raise SystemExit("size-gb must be > 0")
    if args.workers <= 0:
        raise SystemExit("workers must be > 0")
    if block_size <= 0:
        raise SystemExit("block-mb must be > 0")

    print(f"# path={args.path}")
    print(f"# total={args.size_gb:.3f} GB, workers={args.workers}, block={args.block_mb} MB, pattern={args.pattern}")
    print(f"# warmup={args.warmup}, runs={args.runs}, fsync={args.fsync}")
    print("# Preallocating...")
    _preallocate(args.path, total_bytes)

    # warmup
    for i in range(args.warmup):
        g, dt = _run_once(args.path, total_bytes, args.workers, block_size, args.pattern, args.fsync)
        # reset file for next run
        _preallocate(args.path, total_bytes)
        print(f"# warmup {i+1}: {g:.2f} GB/s, {dt:.3f} s")

    # timed runs
    results = []
    for i in range(args.runs):
        g, dt = _run_once(args.path, total_bytes, args.workers, block_size, args.pattern, args.fsync)
        results.append((g, dt))
        print(f"run {i+1}: {g:.2f} GB/s, {dt:.3f} s")

    avg_gbps = sum(g for g, _ in results) / max(1, len(results))
    avg_dt   = sum(dt for _, dt in results) / max(1, len(results))
    print(f"AVG: {avg_gbps:.2f} GB/s over {avg_dt:.3f} s (mean of {len(results)} run(s))")

    if not args.keep:
        try:
            os.remove(args.path)
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    # Use 'spawn' to be safe across platforms (fork-after-CUDA caveats not relevant here but safer anyway)
    mp.set_start_method("spawn", force=True)
    main()
