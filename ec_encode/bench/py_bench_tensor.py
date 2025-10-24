#!/usr/bin/env python3
"""
py_bench_tensor.py

基于 PyTorch Tensor 的 ISA-L Python 基准：比较 xor_gen / ec_encode_data（可选 base 路径）。

用法（从仓库根目录）：
  python3 ec_encode/bench/py_bench_tensor.py xor [--base] -k <k> -n <len> -i <iters> -w <warmup>
  python3 ec_encode/bench/py_bench_tensor.py ec  [--base] -k <k> -d <rows> -n <len> -i <iters> -w <warmup>

说明：
- 预热：默认 warmup=5。
- 复用：EC 模式会预先 build_gftbls 并在循环中复用。
- 对齐：分配对齐的 uint8 CPU tensor。
"""

import argparse
import os
import sys
import time
import math

import numpy as np

try:
    import torch
except Exception:
    torch = None

# 允许直接从仓库根运行脚本：将仓库根加入 sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ec_encode.cpu.isal_wrapper import ISAL_Lib
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import ctypes
from ctypes import POINTER, c_void_p, c_size_t, c_int, c_ubyte
from typing import Optional

# libc helpers for posix_memalign/free (Linux)
_libc = None
if os.name == "posix":
    try:
        _libc = ctypes.CDLL("libc.so.6")
    except Exception:
        try:
            _libc = ctypes.CDLL(None)
        except Exception:
            _libc = None


def posix_alloc(size: int, align: int = 32) -> int:
    """Allocate aligned memory with posix_memalign, return address (int)."""
    if _libc is None:
        raise RuntimeError("libc not available for posix_memalign")
    memptr = c_void_p()
    rc = _libc.posix_memalign(ctypes.byref(memptr), c_size_t(align), c_size_t(size))
    if rc != 0:
        raise OSError(rc, "posix_memalign failed")
    return int(memptr.value)


def posix_free(addr: int) -> None:
    if _libc is None:
        return
    _libc.free(c_void_p(addr))


def mb_per_s(bytes_cnt: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return (bytes_cnt / (1024.0 * 1024.0)) / seconds


def run_xor(args):
    if torch is None:
        raise RuntimeError("需要安装 PyTorch 以运行此基准。")
    lib = ISAL_Lib(prefer_base=args.base)
    k = args.k
    length = args.length
    iters = args.iters
    warmup = args.warmup

    # Single-process fast path (backwards compatible)
    if args.workers <= 1:
        blocks = [ISAL_Lib.aligned_tensor(length, 32) for _ in range(k + 1)]
        for i in range(k):
            blocks[i].random_(0, 256)
        blocks[-1].zero_()

        # warmup
        for _ in range(warmup):
            lib.perform_xor(blocks)

        t0 = time.perf_counter()
        for _ in range(iters):
            lib.perform_xor(blocks)
        t1 = time.perf_counter()

        total_bytes = length * k * iters
        print(
            f"xor ({'base' if args.base else 'opt'}) k={k} len={length} iters={iters} -> MB/s={mb_per_s(total_bytes, t1 - t0):.2f}"
        )
        return

    # Multi-process path (workers > 1), similar to bench_xor.py
    def _xor_worker(core_id: int, iterations: int, k: int, length: int, prefer_base: bool, q, prealloc: bool, pin: bool = False, cpu_start: int = 0, shm_names: Optional[list] = None, posix_ptrs: Optional[list] = None):
        if pin:
            try:
                os.sched_setaffinity(0, {cpu_start + core_id})
            except Exception:
                pass
        try:
            from ec_encode.cpu.isal_wrapper import ISAL_Lib as _ISAL
        except Exception:
            q.put((core_id, 0, 0.0))
            return
        libw = _ISAL(prefer_base=prefer_base)
        total_bytes = 0
        if prealloc and posix_ptrs:
            # posix_memalign path: use addresses inherited via fork
            blocks = []
            for v in range(k + 1):
                base_addr = int(posix_ptrs[v])
                offset = core_id * length
                addr = base_addr + offset
                # create ctypes array at this address
                c_arr = (c_ubyte * length).from_address(addr)
                arr = np.ctypeslib.as_array(c_arr)
                blocks.append(arr)
            t0 = time.time()
            for _ in range(iterations):
                libw.perform_xor(blocks)
                total_bytes += length * k
            t1 = time.time()
        elif prealloc and shm_names:
            # open shared memory buffers and create numpy views to our slice
            blocks = []
            shms = []
            for v in range(k + 1):
                s = shared_memory.SharedMemory(name=shm_names[v])
                shms.append(s)
                offset = core_id * length
                arr = np.ndarray((length,), dtype=np.uint8, buffer=s.buf, offset=offset)
                blocks.append(arr)
            t0 = time.time()
            for _ in range(iterations):
                libw.perform_xor(blocks)
                total_bytes += length * k
            t1 = time.time()
            # close shms (do not unlink here)
            for s in shms:
                try:
                    s.close()
                except Exception:
                    pass
        elif prealloc:
            blocks = [_ISAL.aligned_tensor(length, 32) for _ in range(k + 1)]
            for i in range(k):
                blocks[i].random_(0, 256)
            blocks[-1].zero_()
            t0 = time.time()
            for _ in range(iterations):
                libw.perform_xor(blocks)
                total_bytes += length * k
            t1 = time.time()
        else:
            t0 = time.time()
            for _ in range(iterations):
                blocks = [_ISAL.aligned_tensor(length, 32) for _ in range(k + 1)]
                for i in range(k):
                    blocks[i].random_(0, 256)
                blocks[-1].zero_()
                libw.perform_xor(blocks)
                total_bytes += length * k
            t1 = time.time()
        q.put((core_id, total_bytes, t1 - t0))

    # spawn workers
    q = mp.Queue()
    procs = []
    # choose allocation strategy for prealloc: prefer posix_memalign+fork on Linux
    shm_list = []
    posix_ptrs = []
    use_fork = False
    if args.prealloc and args.workers > 1:
        if _libc is not None and os.name == 'posix':
            # allocate posix_memalign buffers (size = length * workers) for each vector
            for v in range(k + 1):
                size = length * args.workers
                addr = posix_alloc(size, 32)
                # create numpy view and fill per-worker slices
                c_arr = (c_ubyte * size).from_address(addr)
                np_arr = np.ctypeslib.as_array(c_arr)
                for w in range(args.workers):
                    off = w * length
                    if v < k:
                        np_arr[off:off+length] = np.random.randint(0, 256, size=length, dtype=np.uint8)
                    else:
                        np_arr[off:off+length] = 0
                posix_ptrs.append(addr)
            use_fork = True
        else:
            # fallback to shared_memory
            for v in range(k + 1):
                shm = shared_memory.SharedMemory(create=True, size=length * args.workers)
                for w in range(args.workers):
                    off = w * length
                    view = np.ndarray((length,), dtype=np.uint8, buffer=shm.buf, offset=off)
                    if v < k:
                        view[:] = np.random.randint(0, 256, size=length, dtype=np.uint8)
                    else:
                        view[:] = 0
                shm_list.append(shm.name)

    # spawn workers using fork if we used posix allocation (so child inherits memory)
    if use_fork:
        ctx = mp.get_context('fork')
    else:
        ctx = mp

    for i in range(args.workers):
        p = ctx.Process(target=_xor_worker, args=(i, iters, k, length, args.base, q, args.prealloc, args.pin, args.cpu_start, shm_list if shm_list else None, posix_ptrs if posix_ptrs else None))
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

    # cleanup shared memory
    if shm_list:
        for name in shm_list:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except Exception:
                pass

    wall = max_elapsed
    mb = total_bytes / (1024.0 * 1024.0)
    print(f"Workers={args.workers}: total_bytes={total_bytes} bytes ({mb:.2f} MB), wall_time={wall:.3f}s, throughput={mb / wall if wall>0 else float('inf'):.2f} MB/s")


def run_ec(args):
    if torch is None:
        raise RuntimeError("需要安装 PyTorch 以运行此基准。")
    lib = ISAL_Lib(prefer_base=args.base)

    k = args.k
    rows = args.rows
    length = args.length
    iters = args.iters
    warmup = args.warmup

    # Single-process
    if args.workers <= 1:
        src = [ISAL_Lib.aligned_tensor(length, 32) for _ in range(k)]
        for t in src:
            t.random_(0, 256)
        dest = [ISAL_Lib.aligned_tensor(length, 32) for _ in range(rows)]
        for t in dest:
            t.zero_()

        tables = lib.build_gftbls(k, rows)

        # warmup
        for _ in range(warmup):
            lib.ec_encode_with_tables(src, dest, tables)

        t0 = time.perf_counter()
        for _ in range(iters):
            lib.ec_encode_with_tables(src, dest, tables)
        t1 = time.perf_counter()

        total_bytes = length * k * iters
        print(
            f"ec ({'base' if args.base else 'opt'}) k={k} rows={rows} len={length} iters={iters} -> MB/s={mb_per_s(total_bytes, t1 - t0):.2f}"
        )
        return

    # Multi-process path for EC
    def _ec_worker(core_id: int, iterations: int, k: int, rows: int, length: int, prefer_base: bool, q, prealloc: bool, pin: bool = False, cpu_start: int = 0, shm_src: Optional[list] = None, shm_dest: Optional[list] = None, posix_src: Optional[list] = None, posix_dest: Optional[list] = None):
        if pin:
            try:
                os.sched_setaffinity(0, {cpu_start + core_id})
            except Exception:
                pass
        try:
            from ec_encode.cpu.isal_wrapper import ISAL_Lib as _ISAL
        except Exception:
            q.put((core_id, 0, 0.0))
            return
        libw = _ISAL(prefer_base=prefer_base)
        total_bytes = 0
        if prealloc and posix_src is not None and posix_dest is not None:
            # use posix allocated big buffers (inherited via fork)
            src = []
            dest = []
            for sidx in range(k):
                base = int(posix_src[sidx])
                offset = core_id * length
                arr_c = (c_ubyte * length).from_address(base + offset)
                src.append(np.ctypeslib.as_array(arr_c))
            for didx in range(rows):
                base = int(posix_dest[didx])
                offset = core_id * length
                arr_c = (c_ubyte * length).from_address(base + offset)
                dest.append(np.ctypeslib.as_array(arr_c))
            tables = libw.build_gftbls(k, rows)
            t0 = time.time()
            for _ in range(iterations):
                libw.ec_encode_with_tables(src, dest, tables)
                total_bytes += length * k
            t1 = time.time()
        elif prealloc and shm_src is not None and shm_dest is not None:
            src = []
            dest = []
            shms_s = []
            shms_d = []
            for sidx in range(k):
                s = shared_memory.SharedMemory(name=shm_src[sidx])
                shms_s.append(s)
                arr = np.ndarray((length,), dtype=np.uint8, buffer=s.buf, offset=core_id * length)
                src.append(arr)
            for didx in range(rows):
                s = shared_memory.SharedMemory(name=shm_dest[didx])
                shms_d.append(s)
                arr = np.ndarray((length,), dtype=np.uint8, buffer=s.buf, offset=core_id * length)
                dest.append(arr)
            tables = libw.build_gftbls(k, rows)
            t0 = time.time()
            for _ in range(iterations):
                libw.ec_encode_with_tables(src, dest, tables)
                total_bytes += length * k
            t1 = time.time()
        else:
            t0 = time.time()
            for _ in range(iterations):
                src = [_ISAL.aligned_tensor(length, 32) for _ in range(k)]
                for t in src:
                    t.random_(0, 256)
                dest = [_ISAL.aligned_tensor(length, 32) for _ in range(rows)]
                for t in dest:
                    t.zero_()
                tables = libw.build_gftbls(k, rows)
                libw.ec_encode_with_tables(src, dest, tables)
                total_bytes += length * k
            t1 = time.time()
        q.put((core_id, total_bytes, t1 - t0))

    q = mp.Queue()
    procs = []
    # prepare posix or shared_memory prealloc if requested
    posix_src = []
    posix_dest = []
    shm_src = []
    shm_dest = []
    use_fork_ec = False
    if args.prealloc and args.workers > 1:
        if _libc is not None and os.name == 'posix':
            for s in range(k):
                size = length * args.workers
                addr = posix_alloc(size, 32)
                c_arr = (c_ubyte * size).from_address(addr)
                np_arr = np.ctypeslib.as_array(c_arr)
                for w in range(args.workers):
                    np_arr[w*length:(w+1)*length] = np.random.randint(0, 256, size=length, dtype=np.uint8)
                posix_src.append(addr)
            for d in range(rows):
                size = length * args.workers
                addr = posix_alloc(size, 32)
                c_arr = (c_ubyte * size).from_address(addr)
                np_arr = np.ctypeslib.as_array(c_arr)
                for w in range(args.workers):
                    np_arr[w*length:(w+1)*length] = 0
                posix_dest.append(addr)
            use_fork_ec = True
        else:
            for s in range(k):
                shm = shared_memory.SharedMemory(create=True, size=length * args.workers)
                for w in range(args.workers):
                    view = np.ndarray((length,), dtype=np.uint8, buffer=shm.buf, offset=w * length)
                    view[:] = np.random.randint(0, 256, size=length, dtype=np.uint8)
                shm_src.append(shm.name)
            for d in range(rows):
                shm = shared_memory.SharedMemory(create=True, size=length * args.workers)
                for w in range(args.workers):
                    view = np.ndarray((length,), dtype=np.uint8, buffer=shm.buf, offset=w * length)
                    view[:] = 0
                shm_dest.append(shm.name)

    ctx_ec = mp.get_context('fork') if use_fork_ec else mp
    for i in range(args.workers):
        p = ctx_ec.Process(target=_ec_worker, args=(i, iters, k, rows, length, args.base, q, args.prealloc, args.pin, args.cpu_start, shm_src if shm_src else None, shm_dest if shm_dest else None, posix_src if posix_src else None, posix_dest if posix_dest else None))
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

    wall = max_elapsed
    mb = total_bytes / (1024.0 * 1024.0)
    print(f"Workers={args.workers}: total_bytes={total_bytes} bytes ({mb:.2f} MB), wall_time={wall:.3f}s, throughput={mb / wall if wall>0 else float('inf'):.2f} MB/s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ISA-L Python tensor benchmark")
    sub = p.add_subparsers(dest="mode", required=True)

    p_xor = sub.add_parser("xor")
    p_xor.add_argument("--base", action="store_true", help="use *_base symbols if available")
    p_xor.add_argument("-k", type=int, required=True, help="data blocks k")
    p_xor.add_argument("-n", "--length", type=int, required=True, help="bytes per block")
    p_xor.add_argument("-i", "--iters", type=int, default=100)
    p_xor.add_argument("-w", "--warmup", type=int, default=5)
    p_xor.add_argument("--workers", type=int, default=1, help="number of worker processes (1 = single-process)")
    p_xor.add_argument("--prealloc", action="store_true", help="preallocate buffers in each worker and reuse to remove allocation overhead")
    p_xor.add_argument("--pin", action="store_true", help="pin worker processes to consecutive CPUs starting at --cpu-start")
    p_xor.add_argument("--cpu-start", type=int, default=0, help="first cpu id for pinning")

    p_ec = sub.add_parser("ec")
    p_ec.add_argument("--base", action="store_true", help="use *_base symbols if available")
    p_ec.add_argument("-k", type=int, required=True, help="data blocks k")
    p_ec.add_argument("-d", "--rows", type=int, required=True, help="parity rows")
    p_ec.add_argument("-n", "--length", type=int, required=True, help="bytes per block")
    p_ec.add_argument("-i", "--iters", type=int, default=100)
    p_ec.add_argument("-w", "--warmup", type=int, default=5)
    p_ec.add_argument("--workers", type=int, default=1, help="number of worker processes (1 = single-process)")
    p_ec.add_argument("--prealloc", action="store_true", help="preallocate buffers in each worker and reuse to remove allocation overhead")
    p_ec.add_argument("--pin", action="store_true", help="pin worker processes to consecutive CPUs starting at --cpu-start")
    p_ec.add_argument("--cpu-start", type=int, default=0, help="first cpu id for pinning")

    args = p.parse_args()

    if args.mode == "xor":
        run_xor(args)
    elif args.mode == "ec":
        run_ec(args)
