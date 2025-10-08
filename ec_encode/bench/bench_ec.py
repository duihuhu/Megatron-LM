#!/usr/bin/env python3
"""Simple benchmark for EC encoding (CPU multi-process and single-GPU).

Usage examples:
  # CPU: test 1,2,4,8 cores (each process binds to one core)
  python ec_encode/bench/bench_ec.py --mode cpu --k 4 --m 2 --block-size 1024 --iters 200 --max-cores 8

  # GPU: single process, single GPU (script will set CUDA_VISIBLE_DEVICES if gpu-id given)
  python ec_encode/bench/bench_ec.py --mode gpu --k 4 --m 2 --block-size 1024 --iters 500 --gpu-id 0

This script uses the ISA-L wrapper for CPU (`ec_encode.cpu.isal_wrapper.ISAL_Lib`) and
the GPU wrapper (`ec_encode.gpu.gcrspcie_wrapper.GCRSPCIEWrapper`).

The CPU benchmark spawns N processes (1..max_cores) and binds each worker to one core
via `os.sched_setaffinity`. Each worker performs a number of encode iterations and reports
its local bytes encoded and elapsed time. The parent reports aggregate throughput.
"""

import argparse
import multiprocessing as mp
import os
import time
from typing import Tuple, List

import numpy as np
import sys

# Ensure repo root is on sys.path so spawned worker processes can import ec_encode
# When this script is run from the repository root this is a no-op, but when run from
# other working directories (or via the user's command) multiprocessing child
# processes may not inherit the correct sys.path. Prepend project root when possible.
try:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
except Exception:
    # best-effort; workers will still attempt to import and emit a clear error
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--max-cores", type=int, default=8, help="Maximum number of CPU workers to test")
    p.add_argument("--gpu-id", type=int, default=0, help="GPU id to use for GPU test")
    p.add_argument("--lib-path-isa", type=str, default=None, help="Optional path to ISA-L shared lib")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations per test")
    p.add_argument("--input-device", choices=("numpy","tensor"), default="numpy", help="Input source for GPU mode: numpy (host) or tensor (torch.cuda.Tensor)")
    p.add_argument("--use-device-ptr", choices=("auto","on","off"), default="auto",
                   help="GPU per-call device-pointer behavior: auto (env+lib), on (force), off (disable)")
    return p.parse_args()


def cpu_worker(core_id: int, iterations: int, k: int, m: int, block_size: int, lib_path: str, q: mp.Queue):
    # Bind to a single core
    try:
        os.sched_setaffinity(0, {core_id})
    except AttributeError:
        # Windows or unavailable - ignore affinity
        pass

    # Import inside worker to avoid pickling issues. Provide a clear error
    # message if the local package cannot be imported.
    try:
        from ec_encode.cpu.isal_wrapper import ISAL_Lib
    except Exception as e:
        q.put((core_id, 0, 0.0))
        # Re-raise with a concise, single-line message so the child process traceback
        # is visible to the parent and user.
        raise ImportError(
            "Failed to import ec_encode.cpu.isal_wrapper. Run from the repository root or add the repo to PYTHONPATH"
        ) from e

    lib = ISAL_Lib(lib_path) if lib_path else ISAL_Lib()

    total_bytes = 0
    t0 = time.time()
    for _ in range(iterations):
        data_blocks = [np.random.randint(0, 256, block_size, dtype=np.uint8) for _ in range(k)]
        coding_blocks = [np.zeros(block_size, dtype=np.uint8) for _ in range(m)]
        lib.perform_ec_encode(data_blocks, coding_blocks, k, m)
        total_bytes += block_size * k
    t1 = time.time()
    q.put((core_id, total_bytes, t1 - t0))


def run_cpu_benchmark(k: int, m: int, block_size: int, iters: int, max_cores: int, lib_path: str, warmup: int):
    print(f"CPU benchmark: k={k}, m={m}, block_size={block_size}, iters={iters}, max_cores={max_cores}")

    # Warmup single-core
    print("Warmup single-core...")
    q = mp.Queue()
    p = mp.Process(target=cpu_worker, args=(0, max(warmup, 1), k, m, block_size, lib_path, q))
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
        procs: List[mp.Process] = []
        q = mp.Queue()
        start = time.time()
        for i in range(cores):
            p = mp.Process(target=cpu_worker, args=(i, iters, k, m, block_size, lib_path, q))
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

        wall_time = end - start
        # Throughput: total bytes processed / wall_time
        mb = total_bytes / (1024 * 1024)
        tput = mb / wall_time if wall_time > 0 else float('inf')
        print(f"Workers={cores}: total_bytes={total_bytes} bytes ({mb:.2f} MB), wall_time={wall_time:.3f}s, throughput={tput:.2f} MB/s, max_worker_time={max_elapsed:.3f}s")
        results.append((cores, total_bytes, wall_time, tput, max_elapsed))

    return results


def run_gpu_benchmark(k: int, m: int, block_size: int, iters: int, gpu_id: int, warmup: int, input_device: str = 'numpy', use_device_ptr: str = 'auto'):
    # Ensure only one GPU is visible
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # Import wrapper after setting env
    from ec_encode.gpu.gcrspcie_wrapper import GCRSPCIEWrapper

    wrapper = GCRSPCIEWrapper()

    # Warmup
    print("GPU warmup...")
    # Note: input_device selection handled by caller via args; default here uses numpy
    # Map use_device_ptr string to per-call parameter: auto -> None, on -> True, off -> False
    per_call_flag = None if use_device_ptr == 'auto' else (True if use_device_ptr == 'on' else False)
    for _ in range(max(1, warmup)):
        if input_device == 'tensor':
            try:
                import torch
                data = torch.randint(0, 256, (k, block_size), dtype=torch.uint8, device='cuda')
            except Exception:
                # fallback to numpy if torch not available
                data = np.random.randint(0, 256, (k, block_size), dtype=np.uint8)
        else:
            data = np.random.randint(0, 256, (k, block_size), dtype=np.uint8)
        _ = wrapper.encode(data, k, m, return_gpu=False, use_device_ptr=per_call_flag)

    print(f"GPU benchmark: k={k}, m={m}, block_size={block_size}, iters={iters}, gpu_id={gpu_id}")
    total_bytes = 0
    t0 = time.time()
    for _ in range(iters):
        if getattr(wrapper, 'debug', False):
            print("Generating input for iteration")
        if input_device == 'tensor':
            try:
                import torch
                data = torch.randint(0, 256, (k, block_size), dtype=torch.uint8, device='cuda')
            except Exception:
                data = np.random.randint(0, 256, (k, block_size), dtype=np.uint8)
        else:
            data = np.random.randint(0, 256, (k, block_size), dtype=np.uint8)
        _ = wrapper.encode(data, k, m, return_gpu=False, use_device_ptr=per_call_flag)
        total_bytes += block_size * k
    t1 = time.time()

    wall = t1 - t0
    mb = total_bytes / (1024 * 1024)
    tput = mb / wall if wall > 0 else float('inf')
    print(f"GPU: total_bytes={total_bytes} bytes ({mb:.2f} MB), wall_time={wall:.3f}s, throughput={tput:.2f} MB/s")
    return (total_bytes, wall, tput)


def main():
    args = parse_args()
    if args.mode == 'cpu':
        run_cpu_benchmark(args.k, args.m, args.block_size, args.iters, args.max_cores, args.lib_path_isa, args.warmup)
    else:
        run_gpu_benchmark(args.k, args.m, args.block_size, args.iters, args.gpu_id, args.warmup, input_device=args.input_device, use_device_ptr=args.use_device_ptr)


if __name__ == '__main__':
    main()
