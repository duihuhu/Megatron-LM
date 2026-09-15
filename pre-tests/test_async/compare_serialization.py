#!/usr/bin/env python3
"""
Compare manual serialization with automatic Queue serialization

Test scenario:
- Approach 1: manual serialization → Queue.put(bytes) → Queue.get(bytes) → manual deserialization
- Approach 2: Queue.put(tensor) → Queue.get(tensor)  (automatic Queue serialization/deserialization)
"""

import torch
import multiprocessing as mp
import pickle
import time
from typing import Tuple


def worker_manual_deser(queue_in: mp.Queue, queue_out: mp.Queue):
    """Approach 1: Receive serialized data, manual deserialization"""
    while True:
        item = queue_in.get()
        if item is None:  # stop signal
            break

        tensor_idx, name, serialized = item

        # manual deserialization
        start = time.perf_counter()
        tensor = pickle.loads(serialized)
        deser_time = time.perf_counter() - start

        queue_out.put((tensor_idx, name, deser_time, tensor.shape))


def worker_auto_deser(queue_in: mp.Queue, queue_out: mp.Queue):
    """Approach 2: automatic Queue deserialization"""
    while True:
        item = queue_in.get()
        if item is None:  # stop signal
            break

        tensor_idx, name, tensor = item

        # The Queue has already deserialized the tensor
        queue_out.put((tensor_idx, name, 0, tensor.shape))


def test_manual_serialization(tensors):
    """Test approach 1: manual serialization"""
    print("=" * 60)
    print("Approach 1: manual serialization")
    print("=" * 60)

    queue_in = mp.Queue()
    queue_out = mp.Queue()

    # Start worker
    process = mp.Process(target=worker_manual_deser, args=(queue_in, queue_out))
    process.start()

    total_ser_time = 0
    total_put_time = 0
    total_get_time = 0
    total_deser_time = 0

    for i, tensor in enumerate(tensors):
        # Main process: manual serialization
        start = time.perf_counter()
        serialized = pickle.dumps(tensor.contiguous())
        ser_time = time.perf_counter() - start
        total_ser_time += ser_time

        # Main process: put into queue
        start = time.perf_counter()
        queue_in.put((i, f"tensor_{i}", serialized))
        put_time = time.perf_counter() - start
        total_put_time += put_time

        # Main process: get from the result queue
        start = time.perf_counter()
        tensor_idx, name, deser_time, shape = queue_out.get()
        get_time = time.perf_counter() - start
        total_get_time += get_time
        total_deser_time += deser_time

        print(f"  Tensor {i} ({shape}):")
        print(f"    main process Serialization: {ser_time*1000:>8.3f} ms")
        print(f"    Queue put:    {put_time*1000:>8.3f} ms")
        print(f"    Queue get:    {get_time*1000:>8.3f} ms")
        print(f"    subprocess DeSerialization: {deser_time*1000:>8.3f} ms")
        print(f"    Subtotal:         {(ser_time+put_time+get_time+deser_time)*1000:>8.3f} ms")

    # Stop worker
    queue_in.put(None)
    process.join()

    total_time = total_ser_time + total_put_time + total_get_time + total_deser_time

    print(f"\n  Total:")
    print(f"    main process Serialization: {total_ser_time*1000:>8.3f} ms")
    print(f"    Queue put:    {total_put_time*1000:>8.3f} ms")
    print(f"    Queue get:    {total_get_time*1000:>8.3f} ms")
    print(f"    subprocess DeSerialization: {total_deser_time*1000:>8.3f} ms")
    print(f"    Total time:       {total_time*1000:>8.3f} ms")

    return total_time


def test_auto_serialization(tensors):
    """Test approach 2: automatic Queue serialization"""
    print("\n" + "=" * 60)
    print("Approach 2: automatic Queue serialization")
    print("=" * 60)

    queue_in = mp.Queue()
    queue_out = mp.Queue()

    # Start worker
    process = mp.Process(target=worker_auto_deser, args=(queue_in, queue_out))
    process.start()

    total_put_time = 0
    total_get_time = 0

    for i, tensor in enumerate(tensors):
        # Main process: put directly into queue (automatic Queue serialization)
        start = time.perf_counter()
        queue_in.put((i, f"tensor_{i}", tensor))
        put_time = time.perf_counter() - start
        total_put_time += put_time

        # Main process: get from the result queue
        start = time.perf_counter()
        tensor_idx, name, _, shape = queue_out.get()
        get_time = time.perf_counter() - start
        total_get_time += get_time

        print(f"  Tensor {i} ({shape}):")
        print(f"    Queue put:    {put_time*1000:>8.3f} ms  ← includes automatic serialization")
        print(f"    Queue get:    {get_time*1000:>8.3f} ms  ← includes automatic deserialization")
        print(f"    Subtotal:         {(put_time+get_time)*1000:>8.3f} ms")

    # Stop worker
    queue_in.put(None)
    process.join()

    total_time = total_put_time + total_get_time

    print(f"\n  Total:")
    print(f"    Queue put:    {total_put_time*1000:>8.3f} ms")
    print(f"    Queue get:    {total_get_time*1000:>8.3f} ms")
    print(f"    Total time:       {total_time*1000:>8.3f} ms")

    return total_time


def test_serialization_scaling():
    """Test serialization/deserialization time as tensor size changes"""
    print("\n" + "=" * 70)
    print("📊 Serialization Performance by Tensor Size")
    print("=" * 70)

    # Test different tensor sizes (MB)
    test_sizes = [1, 2, 4, 8, 16, 32, 64]

    print(f"\nTest Configuration:")
    print(f"  - Tensor size: {test_sizes} MB")
    print(f"  - test each size three times and report the average")

    results = []

    for size_mb in test_sizes:
        # Create a tensor of the specified size
        num_elements = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
        side = int(num_elements ** 0.5)

        # Test 3 times and average
        ser_times = []
        deser_times = []

        for _ in range(3):
            tensor = torch.randn(side, side)

            # Test serialization
            start = time.perf_counter()
            serialized = pickle.dumps(tensor.contiguous())
            ser_time = time.perf_counter() - start
            ser_times.append(ser_time)

            # Test deserialization
            start = time.perf_counter()
            _ = pickle.loads(serialized)
            deser_time = time.perf_counter() - start
            deser_times.append(deser_time)

        avg_ser = sum(ser_times) / len(ser_times)
        avg_deser = sum(deser_times) / len(deser_times)
        total = avg_ser + avg_deser

        results.append({
            'size_mb': size_mb,
            'ser_time': avg_ser * 1000,  # ms
            'deser_time': avg_deser * 1000,  # ms
            'total_time': total * 1000  # ms
        })

    # Print results table
    print(f"\n{'size (MB)':<10} {'serialization (ms)':<14} {'deserialization (ms)':<14} {'Total time(ms)':<12} {'MB/s'}")
    print("-" * 70)

    for r in results:
        throughput = r['size_mb'] / (r['total_time'] / 1000)  # MB/s
        print(f"{r['size_mb']:<10} {r['ser_time']:<14.2f} {r['deser_time']:<14.2f} {r['total_time']:<12.2f} {throughput:.1f}")

    # analysis
    print("\n" + "=" * 70)
    print("📈 Performance analysis")
    print("=" * 70)

    # Compute serialization and deserialization share
    total_ser = sum(r['ser_time'] for r in results)
    total_deser = sum(r['deser_time'] for r in results)
    total_all = total_ser + total_deser

    print(f"\n1. Time breakdown:")
    print(f"   - Serialization:   {total_ser:.2f} ms ({total_ser/total_all*100:.1f}%)")
    print(f"   - DeSerialization: {total_deser:.2f} ms ({total_deser/total_all*100:.1f}%)")

    # Compute average throughput
    avg_throughput = sum(r['size_mb'] / (r['total_time'] / 1000) for r in results) / len(results)
    print(f"\n2. Average throughput: {avg_throughput:.1f} MB/s")

    # Analyze time complexity
    small_time_per_mb = results[0]['total_time'] / results[0]['size_mb']
    large_time_per_mb = results[-1]['total_time'] / results[-1]['size_mb']

    print(f"\n3. Time complexity:")
    print(f"   - small tensor ({results[0]['size_mb']}MB): {small_time_per_mb:.2f} ms/MB")
    print(f"   - large tensor ({results[-1]['size_mb']}MB): {large_time_per_mb:.2f} ms/MB")

    if abs(small_time_per_mb - large_time_per_mb) / small_time_per_mb < 0.2:
        print(f"   - ✅ approximately linear O(n): time is proportional to size")
    else:
        print(f"   - ⚠️  nonlinear: large tensor time per unit {'longer' if large_time_per_mb > small_time_per_mb else 'shorter'}")

    print("\n4. Key findings:")
    print(f"   - serialization is {total_ser/total_deser:.1f}x slower than deserialization")
    print(f"   - Serialization overhead becomes more pronounced as tensors grow")
    print(f"   - A 64 MB tensor requires {results[-1]['total_time']:.0f}ms, which is a significant portion of a training iteration")

    return results


def main():
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)

    print("\n" + "🚀 " + "Serialization Performance Test Suite".center(58) + " 🚀")

    # Test 1: Serialization performance by size
    test_serialization_scaling()

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

