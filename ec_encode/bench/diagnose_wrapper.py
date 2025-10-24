#!/usr/bin/env python3
"""diagnose_wrapper.py

诊断 ISAL Python wrapper 的开销，分别测量：
- 构造 ctypes 指针数组的开销
- 直接调用底层 ctypes 函数的开销（使用预先构造的 pointer arrays）
- 通过 wrapper API 调用（含检查与数组构造）的开销

用法：
  python3 ec_encode/bench/diagnose_wrapper.py

"""
import time
import sys
import math

import numpy as np

try:
    import torch
except Exception:
    torch = None

from ec_encode.cpu.isal_wrapper import ISAL_Lib


def mb_per_s(bytes_cnt: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return (bytes_cnt / (1024.0 * 1024.0)) / seconds


def make_buffers(k, rows, length):
    # try to use ISAL_Lib.aligned_tensor when torch available
    src = []
    dest = []
    if torch is not None:
        for _ in range(k):
            t = ISAL_Lib.aligned_tensor(length, 32)
            t.random_(0, 256)
            src.append(t)
        for _ in range(rows):
            t = ISAL_Lib.aligned_tensor(length, 32)
            t.zero_()
            dest.append(t)
    else:
        # numpy fallback
        for _ in range(k):
            a = np.random.randint(0, 256, size=(length,), dtype=np.uint8)
            src.append(a)
        for _ in range(rows):
            a = np.zeros((length,), dtype=np.uint8)
            dest.append(a)
    return src, dest


def bench_ec(k=8, rows=4, length=1024, iters_ptr=200, iters_call=200, iters_wrapper=200):
    print(f"\n=== EC bench k={k} rows={rows} length={length} bytes per block ===")
    lib = ISAL_Lib()

    src, dest = make_buffers(k, rows, length)

    # build tables once
    tables = lib.build_gftbls(k, rows)

    # 1) pointer-array construction cost
    t0 = time.perf_counter()
    for _ in range(iters_ptr):
        p = lib._get_pointer_array(src)
    t1 = time.perf_counter()
    avg_ptr_ms = (t1 - t0) / iters_ptr * 1000.0
    print(f"pointer-array create: {avg_ptr_ms:.3f} ms/call (avg over {iters_ptr})")

    # 2) direct ctypes call using precomputed pointer arrays
    try:
        src_ptrs = lib._get_pointer_array(src)
        dest_ptrs = lib._get_pointer_array(dest)
    except Exception as e:
        print("Failed to build pointer arrays:", e)
        return

    fn = lib._ec_encode_data or lib._ec_encode_data_base
    if fn is None:
        print("No ec_encode_data symbol available in loaded library; skipping direct-call benchmark")
    else:
        # warmup
        for _ in range(5):
            fn(int(length), int(k), int(rows), tables.gftbls_ptr, src_ptrs, dest_ptrs)
        t0 = time.perf_counter()
        for _ in range(iters_call):
            fn(int(length), int(k), int(rows), tables.gftbls_ptr, src_ptrs, dest_ptrs)
        t1 = time.perf_counter()
        total_bytes = length * k * iters_call
        print(
            f"direct ctypes call: {((t1 - t0) / iters_call * 1000):.3f} ms/call, MB/s={mb_per_s(total_bytes, t1 - t0):.2f} (over {iters_call})"
        )

    # 3) wrapper call which rebuilds pointer arrays every call
    # warmup
    for _ in range(5):
        lib.ec_encode_with_tables(src, dest, tables)
    t0 = time.perf_counter()
    for _ in range(iters_wrapper):
        lib.ec_encode_with_tables(src, dest, tables)
    t1 = time.perf_counter()
    total_bytes = length * k * iters_wrapper
    print(
        f"wrapper call (with checks & ptr build): {((t1 - t0) / iters_wrapper * 1000):.3f} ms/call, MB/s={mb_per_s(total_bytes, t1 - t0):.2f} (over {iters_wrapper})"
    )


def bench_xor(k=8, length=1024, iters_ptr=200, iters_call=200, iters_wrapper=200):
    print(f"\n=== XOR bench k={k} length={length} bytes per block ===")
    lib = ISAL_Lib()

    # blocks: k data + 1 parity
    blocks = [ISAL_Lib.aligned_tensor(length, 32) if torch is not None else np.random.randint(0, 256, size=(length,), dtype=np.uint8) for _ in range(k + 1)]
    if torch is not None:
        for i in range(k):
            blocks[i].random_(0, 256)
        blocks[-1].zero_()

    # pointer creation
    t0 = time.perf_counter()
    for _ in range(iters_ptr):
        p = lib._get_pointer_array(blocks)
    t1 = time.perf_counter()
    print(f"xor pointer-array create: {(t1 - t0)/iters_ptr*1000:.3f} ms/call")

    fn = lib._xor_gen or lib._xor_gen_base
    if fn is None:
        print("No xor_gen symbol available; skipping direct-call xor bench")
    else:
        p = lib._get_pointer_array(blocks)
        # warmup
        for _ in range(5):
            fn(len(blocks), int(length), p)
        t0 = time.perf_counter()
        for _ in range(iters_call):
            fn(len(blocks), int(length), p)
        t1 = time.perf_counter()
        total_bytes = length * k * iters_call
        print(f"direct xor call: {(t1 - t0)/iters_call*1000:.3f} ms/call, MB/s={mb_per_s(total_bytes, t1 - t0):.2f}")

    # wrapper
    for _ in range(5):
        lib.perform_xor(blocks)
    t0 = time.perf_counter()
    for _ in range(iters_wrapper):
        lib.perform_xor(blocks)
    t1 = time.perf_counter()
    total_bytes = length * k * iters_wrapper
    print(f"wrapper xor call: {(t1 - t0)/iters_wrapper*1000:.3f} ms/call, MB/s={mb_per_s(total_bytes, t1 - t0):.2f}")


if __name__ == '__main__':
    # run a few cases: small & large
    try:
        bench_ec(k=8, rows=4, length=256, iters_ptr=500, iters_call=500, iters_wrapper=500)
        bench_ec(k=8, rows=4, length=4 * 1024 * 1024, iters_ptr=20, iters_call=20, iters_wrapper=20)
        bench_xor(k=8, length=256, iters_ptr=500, iters_call=500, iters_wrapper=500)
        bench_xor(k=8, length=4 * 1024 * 1024, iters_ptr=20, iters_call=20, iters_wrapper=20)
    except Exception as e:
        print("Diagnostic script failed:", e)
        raise
