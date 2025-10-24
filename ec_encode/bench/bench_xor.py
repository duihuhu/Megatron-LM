#!/usr/bin/env python3
"""Benchmark XOR throughput: CPU (ISA-L) vs GPU (torch tensor xor)

CPU mode: spawns multiple processes (1..max_cores) each performing XOR on k+1 blocks
GPU mode: single-process GPU tensor XOR on device; measures throughput

Usage examples:
  python ec_encode/bench/bench_xor.py --mode cpu --k 2 --block-size 1048576 --iters 100 --max-cores 8
  python ec_encode/bench/bench_xor.py --mode gpu --k 2 --block-size 1048576 --iters 100 --gpu-id 0
"""

import argparse
import time
import os
import sys
from typing import List

# Ensure repo root on sys.path for worker processes
try:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    p.add_argument("--k", type=int, default=2, help="number of data blocks (XOR will use k+1 with last as target)")
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--max-cores", type=int, default=8)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--lib-path-isa", type=str, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--prealloc", action="store_true", help="Preallocate buffers and reuse each iteration to remove allocation overhead")
    return p.parse_args()


def aligned_array(size: int, align: int = 32):
    """Allocate a uint8 numpy array of length `size` with `align`-byte alignment.

    Returns a contiguous numpy uint8 view that starts at an address aligned to `align`.
    """
    import numpy as np
    # allocate extra bytes to be able to align
    buf = np.empty(size + align, dtype=np.uint8)
    base = buf.ctypes.data
    offset = (-base) % align
    if offset == 0:
        return buf[:size]
    else:
        return buf[offset:offset + size]


def fill_random_uint8(arr):
    """Fill a numpy uint8 array with random bytes in-place."""
    import numpy as _np
    # generate into a temporary then copy into arr to avoid changing arr's base
    arr[:] = _np.random.randint(0, 256, size=arr.shape, dtype=_np.uint8)


# ---------------- CPU worker (uses ISAL wrapper) ----------------
def cpu_worker(core_id: int, iterations: int, k: int, block_size: int, lib_path: str, q, prealloc: bool = False):
    try:
        os.sched_setaffinity(0, {core_id})
    except Exception:
        pass

    try:
        from ec_encode.cpu.isal_wrapper import ISAL_Lib
    except Exception as e:
        q.put((core_id, 0, 0.0))
        raise ImportError("Failed to import ec_encode.cpu.isal_wrapper") from e

    lib = ISAL_Lib(lib_path) if lib_path else ISAL_Lib()

    total_bytes = 0
    # If prealloc is requested, allocate k+1 buffers once and reuse them to isolate XOR compute time
    if prealloc:
        # use aligned allocations
        blocks = [aligned_array(block_size, align=32) for _ in range(k+1)]
        for b in blocks:
            fill_random_uint8(b)
        t0 = time.time()
        for _ in range(iterations):
            lib.perform_xor(blocks)
            total_bytes += block_size * (k)
        t1 = time.time()
    else:
        # allocate per-iteration (includes allocation cost)
        t0 = time.time()
        for _ in range(iterations):
            blocks = [aligned_array(block_size, align=32) for _ in range(k+1)]
            for b in blocks:
                fill_random_uint8(b)
            lib.perform_xor(blocks)
            total_bytes += block_size * (k)
        t1 = time.time()
    q.put((core_id, total_bytes, t1 - t0))


def run_cpu_benchmark(k: int, block_size: int, iters: int, max_cores: int, lib_path: str, warmup: int, prealloc: bool = False):
    print(f"CPU XOR benchmark: k={k}, block_size={block_size}, iters={iters}, max_cores={max_cores}")
    # warmup
    import multiprocessing as mp
    q = mp.Queue()
    p = mp.Process(target=cpu_worker, args=(0, max(1, warmup), k, block_size, lib_path, q, prealloc))
    p.start(); p.join()
    _ = q.get()

    core_counts = []
    c = 1
    while c <= max_cores:
        core_counts.append(c)
        c *= 2

    results = []
    for cores in core_counts:
        print(f"\nRunning with {cores} worker(s)...")
        procs = []
        q = mp.Queue()
        start = time.time()
        for i in range(cores):
            p = mp.Process(target=cpu_worker, args=(i, iters, k, block_size, lib_path, q, prealloc))
            p.start()
            procs.append(p)

        total_bytes = 0
        max_elapsed = 0
        for _ in procs:
            core_id, b, elapsed = q.get()
            total_bytes += b
            if elapsed > max_elapsed:
                max_elapsed = elapsed

        for p in procs:
            p.join()
    end = time.time()
    wall = end - start
    mb = total_bytes / (1024 * 1024)
    tput_wall = mb / wall if wall > 0 else float('inf')
    # Compute throughput by worker time (max worker elapsed) for comparable compute-only measure
    tput_worker = mb / max_elapsed if max_elapsed > 0 else float('inf')
    print(f"Workers={cores}: total_bytes={total_bytes} bytes ({mb:.2f} MB), wall_time={wall:.3f}s, throughput_wall={tput_wall:.2f} MB/s, max_worker_time={max_elapsed:.3f}s, throughput_worker={tput_worker:.2f} MB/s")
    results.append((cores, total_bytes, wall, tput_wall, max_elapsed))

    return results


# ---------------- GPU benchmark (tensor XOR) ----------------

def run_gpu_benchmark(k: int, block_size: int, iters: int, gpu_id: int, warmup: int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        import torch
    except Exception:
        raise RuntimeError("torch not available for GPU benchmark")
    print(f"GPU XOR benchmark: k={k}, block_size={block_size}, iters={iters}, gpu_id={gpu_id}")

    # warmup + option for preallocation handled by caller
    torch.cuda.empty_cache()
    total_bytes = 0
    # Perform a warmup kernel and synchronize to initialize CUDA and stabilize clocks
    for _ in range(max(1, warmup)):
        data = torch.randint(0, 256, (k+1, block_size), dtype=torch.uint8, device='cuda')
        target = data[-1]
        for i in range(k):
            target ^= data[i]
    torch.cuda.synchronize()

    # Accurate timing: synchronize before starting timer and after finishing
    data = None
    prealloc = False
    # detect if caller requested preallocation via env var (we'll also support CLI flag later)
    # For backward compatibility, check env var BENCH_XOR_PREALLOC
    if os.environ.get('BENCH_XOR_PREALLOC', '0') == '1':
        prealloc = True

    if prealloc:
        # allocate and reuse random data
        data = torch.randint(0, 256, (k+1, block_size), dtype=torch.uint8, device='cuda')

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        if prealloc:
            target = data[-1]
            for i in range(k):
                target ^= data[i]
        else:
            data = torch.randint(0, 256, (k+1, block_size), dtype=torch.uint8, device='cuda')
            target = data[-1]
            for i in range(k):
                target ^= data[i]
        total_bytes += block_size * k
    torch.cuda.synchronize()
    t1 = time.time()

    wall = t1 - t0
    mb = total_bytes / (1024 * 1024)
    tput = mb / wall if wall > 0 else float('inf')
    print(f"GPU: total_bytes={total_bytes} bytes ({mb:.2f} MB), wall_time={wall:.3f}s, throughput={tput:.2f} MB/s")
    return (total_bytes, wall, tput)


def main():
    args = parse_args()
    if args.mode == 'cpu':
        run_cpu_benchmark(args.k, args.block_size, args.iters, args.max_cores, args.lib_path_isa, args.warmup)
    else:
        run_gpu_benchmark(args.k, args.block_size, args.iters, args.gpu_id, args.warmup)


if __name__ == '__main__':
    main()
