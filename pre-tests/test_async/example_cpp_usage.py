#!/usr/bin/env python3
"""
Complete Example Using C++ Threads + CPUMemoryPool

Demonstrates:
1. Use CPUMemoryPool to allocate memory
2. Copy tensor data to the memory pool
3. Pass the address to C++ thread (no serialization required)
4. The C++ thread accesses memory directly to perform computation

Note: build cpp_thread_example.cpp
"""

import torch
import time
import ctypes
import sys
import os
import numpy as np
import multiprocessing as mp
import pickle

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test_tensors_layout'))
from gpu_cpu_memory_pool import CPUMemoryPool

try:
    import cpp_thread_example
    CPP_AVAILABLE = True
except ImportError:
    print("Warning: cpp_thread_example has not been built, using Python simulated implementation")
    CPP_AVAILABLE = False

def worker_process_pickle(queue_in: mp.Queue, queue_out: mp.Queue):
    """Multiprocessing worker: Receive manually serialized data"""
    while True:
        item = queue_in.get()
        if item is None:
            break

        tensor_idx, serialized = item

        # deserialization
        deser_start = time.perf_counter()
        tensor = pickle.loads(serialized)
        deser_time = time.perf_counter() - deser_start

        # Simulate computation
        # result = tensor.sum().item()
        result = 0

        queue_out.put((tensor_idx, deser_time, result))


def worker_process_auto(queue_in: mp.Queue, queue_out: mp.Queue):
    """Multiprocessing worker: Receive automatically serialized Queue data"""
    while True:
        item = queue_in.get()  # ← automatic Queue deserialization
        receive_time = time.perf_counter()  # ← Record the receive timestamp immediately

        if item is None:
            break

        tensor_idx, send_time, tensor = item

        # Compute elapsed time from send to receive (including serialization+transfer+deserialization)
        transfer_time = receive_time - send_time

        # No need for manual deserialization, Queue has already handled it automatically
        # Simulate computation
        # result = tensor.sum().item()
        result = 0

        queue_out.put((tensor_idx, transfer_time, result))


def test_pickle_multiprocess(tensor_size_mb: float, warmup: bool = False):
    """Test: Python multiprocessing + manual Pickle serialization

    Args:
        tensor_size_mb: Tensor size (MB)
        warmup: whether this is a warm-up test

    Returns:
        Average time (seconds)
    """
    # Test 3 times and average (avoid variance)
    num_iterations = 3

    # Pre-create multiple different tensor (avoid cache effects)
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
    side = int(num_elements ** 0.5)
    tensors = [torch.randn(side, side) for _ in range(num_iterations)]

    # Start worker process
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    process = mp.Process(target=worker_process_pickle, args=(queue_in, queue_out))
    process.start()

    times = []

    for i in range(num_iterations):
        start = time.perf_counter()

        # Main process: manual serialization (use a different tensor)
        ser_start = time.perf_counter()
        serialized = pickle.dumps(tensors[i].contiguous())
        ser_time = time.perf_counter() - ser_start

        # Send to subprocess
        queue_in.put((0, serialized))

        # Wait for results
        _, deser_time, _ = queue_out.get()

        if i != 0:
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    # Stop worker
    queue_in.put(None)
    process.join()

    avg_time = sum(times) / len(times)

    if not warmup:
        print(f"  {tensor_size_mb:>5.0f} MB: {avg_time*1000:>8.2f} ms  (manual Serialization: {ser_time*1000:.2f}ms, manual DeSerialization: {deser_time*1000:.2f}ms)")

    return avg_time


def test_queue_auto_serialize(tensor_size_mb: float, warmup: bool = False):
    """Test: Python multiprocessing + automatic Queue serialization

    Args:
        tensor_size_mb: Tensor size (MB)
        warmup: whether this is a warm-up test

    Returns:
        Average time (seconds)
    """
    # Test 3 times and average (avoid variance)
    num_iterations = 3

    # Pre-create multiple different tensor (avoid cache effects)
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
    side = int(num_elements ** 0.5)
    tensors = [torch.randn(side, side) for _ in range(num_iterations)]

    # Start worker process
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    process = mp.Process(target=worker_process_auto, args=(queue_in, queue_out))
    process.start()

    times = []
    transfer_times = []

    for i in range(num_iterations):
        start = time.perf_counter()

        # Main process: Record the send timestamp, put directly  tensor (use a different tensor)
        cpu_tensor = tensors[i].contiguous()
        send_time = time.perf_counter()  # ← Record the send timestamp
        queue_in.put((0, send_time, cpu_tensor))  # ← automatic Queueally invokes pickle.dumps()

        # Wait for results (the subprocess returns the transfer time)
        _, transfer_time, _ = queue_out.get()

        if i != 0:
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            transfer_times.append(transfer_time)

    # Stop worker
    queue_in.put(None)
    process.join()

    avg_time = sum(times) / len(times)
    avg_transfer = sum(transfer_times) / len(transfer_times)

    if not warmup:
        print(f"  {tensor_size_mb:>5.0f} MB: {avg_time*1000:>8.2f} ms  (Queue transfer: {avg_transfer*1000:.2f}ms)")

    return avg_time


def test_cpp_thread_pool(tensor_size_mb: float, pool: CPUMemoryPool, processor, warmup: bool = False):
    """Test: C++ thread + CPUMemoryPool

    Args:
        tensor_size_mb: Tensor size (MB)
        pool: CPUMemoryPool instance
        processor: C++ processor instance
        warmup: whether this is a warm-up test

    Returns:
        Average time (seconds)
    """
    # Test 3 times and average
    num_iterations = 3

    # Pre-create multiple different tensor (avoid cache effects)
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
    side = int(num_elements ** 0.5)
    tensors = [torch.randn(side, side) for _ in range(num_iterations)]

    times = []

    for i in range(num_iterations):
        start = time.perf_counter()

        # Allocate memory
        size = tensors[i].element_size() * tensors[i].nelement()
        address, tensor_id = pool.allocate(size)

        # Copy to memory pool (use a different tensor)
        cpu_tensor = tensors[i].contiguous()
        ctypes.memmove(address, cpu_tensor.data_ptr(), size)

        # Submit to C++ thread (pass only the address)
        processor.submit(address, size, list(cpu_tensor.shape), 0)

        # Wait for results
        _, _, _ = processor.get_result()

        # Free memory
        pool.deallocate(tensor_id)

        if i != 0:
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    avg_time = sum(times) / len(times)

    if not warmup:
        print(f"  {tensor_size_mb:>5.0f} MB: {avg_time*1000:>8.2f} ms")

    return avg_time


def compare_with_pickle():
    """Compare the performance of three approaches"""
    print("\n" + "=" * 70)
    print("Performance comparison: three approaches")
    print("=" * 70)

    # Test Configuration: different tensor sizes (MB)
    test_sizes = [2, 4, 8, 16, 32, 64]

    print(f"\nTest Configuration:")
    print(f"  - Tensor size: {test_sizes} MB")
    print(f"  - test each size three times and report the average")
    print(f"  - perform a warm-up first to avoid cold-start effects")

    # ==============================================
    # Approach 1: Python multiprocessing + manual Pickle serialization
    # ==============================================
    print("\n" + "-" * 70)
    print("Approach 1: Python multiprocessing + manual Pickle serialization")
    print("  (main process pickle.dumps → Queue.put → Queue.get → subprocess pickle.loads)")
    print("-" * 70)

    print("\n  Test results:")
    pickle_times = []
    for size_mb in test_sizes:
        avg_time = test_pickle_multiprocess(size_mb)
        pickle_times.append(avg_time)

    print(f"\n  Average time: {sum(pickle_times)/len(pickle_times)*1000:.2f} ms")

    # ==============================================
    # Approach 2: Python multiprocessing + automatic Queue serialization
    # ==============================================
    print("\n" + "-" * 70)
    print("Approach 2: Python multiprocessing + automatic Queue serialization")
    print("  (main process Queue.put(tensor) → automatic Queue serialization/deserialization)")
    print("-" * 70)

    print("\n  Test results:")
    queue_auto_times = []
    for size_mb in test_sizes:
        avg_time = test_queue_auto_serialize(size_mb)
        queue_auto_times.append(avg_time)

    print(f"\n  Average time: {sum(queue_auto_times)/len(queue_auto_times)*1000:.2f} ms")

    # ==============================================
    # Approach 3: C++ thread + CPUMemoryPool
    # ==============================================
    if CPP_AVAILABLE:
        print("\n" + "-" * 70)
        print("Approach 3: C++ thread + CPUMemoryPool (zero serialization)")
        print("  (pass the memory address, threads share an address space)")
        print("-" * 70)

        # Initialize
        pool_size = 200 * 1024 * 1024  # 200 MB
        pool = CPUMemoryPool(pool_size)
        processor = cpp_thread_example.AsyncTensorProcessor(num_threads=1)

        print("\n  Test results:")
        cpp_times = []
        for size_mb in test_sizes:
            avg_time = test_cpp_thread_pool(size_mb, pool, processor)
            cpp_times.append(avg_time)

        processor.stop()

        print(f"\n  Average time: {sum(cpp_times)/len(cpp_times)*1000:.2f} ms")

        # ==============================================
        # comparison results
        # ==============================================
        print("\n" + "=" * 70)
        print("📊 Detailed comparison")
        print("=" * 70)
        print(f"\n{'size (MB)':<10} {'manual Pickle (ms)':<16} {'automatic Queue (ms)':<16} {'C++ thread (ms)':<14}")
        print("-" * 70)

        for i, size_mb in enumerate(test_sizes):
            pickle_ms = pickle_times[i] * 1000
            queue_ms = queue_auto_times[i] * 1000
            cpp_ms = cpp_times[i] * 1000

            print(f"{size_mb:<10} {pickle_ms:<16.2f} {queue_ms:<16.2f} {cpp_ms:<14.2f}")

        print("-" * 70)
        avg_pickle = sum(pickle_times) / len(pickle_times) * 1000
        avg_queue = sum(queue_auto_times) / len(queue_auto_times) * 1000
        avg_cpp = sum(cpp_times) / len(cpp_times) * 1000
        print(f"{'Average':<10} {avg_pickle:<16.2f} {avg_queue:<16.2f} {avg_cpp:<14.2f}")

    print("=" * 70)


def main():
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)

    print("\n" + "🚀 " + "C++ Thread + CPUMemoryPool Performance Test".center(58) + " 🚀")

    # Test 1: Performance comparison (primary test)
    compare_with_pickle()

    print("\n" + "=" * 70)
    print("💡 Summary")
    print("=" * 70)
    print("Test Configuration:")
    print("  - Test six different tensor sizes (2-64 MB)")
    print("  - Run each test three times and report the average to reduce variance")
    print("  - Test after warm-up, avoid cold-start effects")
    print("  - Use actual multiprocessing + Queue communication")
    print()
    print("Comparison of the three approaches:")
    print("  1. manual Pickle: main process pickle.dumps → Queue → subprocess pickle.loads")
    print("  2. automatic Queue:  main process Queue.put(tensor) → automatic Queue serialization")
    print("  3. C++ thread:    the main process passes an address → C++ thread directly accesses (zero serialization)")
    print("👉 Recommendation: use C++ threads + CPUMemoryPool to implement asynchronous checkpointing!")
    print("=" * 70)


if __name__ == "__main__":
    main()

