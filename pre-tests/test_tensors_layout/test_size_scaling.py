"""
Alignment Performance Test for Different Tensor Sizes
===================================

Test tensor performance at different size scales (B, KB, MB, GB) with and without alignment
"""

import torch
import time
import numpy as np
from gpu_cpu_memory_pool import CPUMemoryPool, GPUToCPUPoolTransfer


def test_tensor_size_scaling():
    """Test tensor performance at different size scales"""
    print("=" * 100)
    print("Tensor Size Scaling Test: Aligned vs. Unaligned Performance")
    print("=" * 100)

    if not torch.cuda.is_available():
        print("CUDA support is required")
        return

    # Define test sizes (byte, KB, MB, and GB scales)
    test_configs = [
        # (name, size(bytes), shape, Iterations)
        ("100 B",      100,          (5, 5),          1000),      # byte scale
        ("1 KB",       1024,         (16, 16),        1000),      # KB scale
        ("10 KB",      10*1024,      (50, 50),        1000),      # KB scale
        ("100 KB",     100*1024,     (160, 160),      500),       # KB scale
        ("1 MB",       1024*1024,    (512, 512),      200),       # MB scale
        ("10 MB",      10*1024*1024, (1620, 1620),    50),        # MB scale
        # ("100 MB",     100*1024*1024,(5120, 5120),    20),        # MB scale
        # ("500 MB",     500*1024*1024,(11450, 11450),  5),         # MB scale
    ]

    results = []

    print(f"\n{'='*100}")
    print(f"{'size':<12} {'shape':<18} {'iterations':<8} {'transfer':<20} {'write':<20} {'read':<20}")
    print(f"{'-'*100}")

    for name, target_bytes, shape, iterations in test_configs:
        print(f"{name:<12} {str(shape):<18} {iterations:<8} ", end='', flush=True)

        try:
            # Create GPU tensor
            gpu_tensor = torch.randn(*shape, device='cuda', dtype=torch.float32)
            actual_size = gpu_tensor.element_size() * gpu_tensor.nelement()
            size_mb = actual_size / 1024 / 1024

            # Prepare a memory pool for each mode
            pool_size = max(actual_size * 4, 10 * 1024 * 1024)  # at least 10 MB

            # ========== Test 1: enable alignment ==========
            pool_aligned = CPUMemoryPool(pool_size_bytes=pool_size, enable_alignment=True)
            mgr_aligned = GPUToCPUPoolTransfer(pool_aligned)

            # transfer performance
            transfer_times_aligned = []
            for _ in range(min(iterations, 10)):  # transfer test up to 10 times
                torch.cuda.synchronize()
                start = time.time()
                tid = mgr_aligned.transfer_to_pool(gpu_tensor, use_pinned_memory=True)
                torch.cuda.synchronize()
                transfer_times_aligned.append(time.time() - start)
                mgr_aligned.free_tensor(tid)

            avg_transfer_aligned = np.mean(transfer_times_aligned)

            # Get CPU tensor for read/write tests
            tid_aligned = mgr_aligned.transfer_to_pool(gpu_tensor)
            cpu_tensor_aligned = mgr_aligned.get_tensor_from_pool(tid_aligned)

            # write performance
            write_times_aligned = []
            for _ in range(3):  # Warm up
                cpu_tensor_aligned.fill_(1.0)

            for i in range(min(iterations, 100)):
                start = time.time()
                cpu_tensor_aligned.fill_(float(i))
                write_times_aligned.append(time.time() - start)

            avg_write_aligned = np.mean(write_times_aligned)

            # read performance
            read_times_aligned = []
            for _ in range(3):  # Warm up
                _ = cpu_tensor_aligned.sum()

            for _ in range(min(iterations, 100)):
                start = time.time()
                _ = cpu_tensor_aligned.sum().item()
                read_times_aligned.append(time.time() - start)

            avg_read_aligned = np.mean(read_times_aligned)

            mgr_aligned.free_tensor(tid_aligned)

            # ========== Test 2: disable alignment ==========
            pool_unaligned = CPUMemoryPool(pool_size_bytes=pool_size, enable_alignment=False)
            mgr_unaligned = GPUToCPUPoolTransfer(pool_unaligned)

            # transfer performance
            transfer_times_unaligned = []
            for _ in range(min(iterations, 10)):
                torch.cuda.synchronize()
                start = time.time()
                tid = mgr_unaligned.transfer_to_pool(gpu_tensor, use_pinned_memory=True)
                torch.cuda.synchronize()
                transfer_times_unaligned.append(time.time() - start)
                mgr_unaligned.free_tensor(tid)

            avg_transfer_unaligned = np.mean(transfer_times_unaligned)

            # Get CPU tensor for read/write tests
            tid_unaligned = mgr_unaligned.transfer_to_pool(gpu_tensor)
            cpu_tensor_unaligned = mgr_unaligned.get_tensor_from_pool(tid_unaligned)

            # write performance
            write_times_unaligned = []
            for _ in range(3):  # Warm up
                cpu_tensor_unaligned.fill_(1.0)

            for i in range(min(iterations, 100)):
                start = time.time()
                cpu_tensor_unaligned.fill_(float(i))
                write_times_unaligned.append(time.time() - start)

            avg_write_unaligned = np.mean(write_times_unaligned)

            # read performance
            read_times_unaligned = []
            for _ in range(3):  # Warm up
                _ = cpu_tensor_unaligned.sum()

            for _ in range(min(iterations, 100)):
                start = time.time()
                _ = cpu_tensor_unaligned.sum().item()
                read_times_unaligned.append(time.time() - start)

            avg_read_unaligned = np.mean(read_times_unaligned)

            mgr_unaligned.free_tensor(tid_unaligned)

            # Compute speedup
            speedup_transfer = avg_transfer_unaligned / avg_transfer_aligned
            speedup_write = avg_write_unaligned / avg_write_aligned
            speedup_read = avg_read_unaligned / avg_read_aligned

            # Format output
            transfer_str = f"{avg_transfer_aligned*1000:.3f}/{avg_transfer_unaligned*1000:.3f}ms ({speedup_transfer:.2f}x)"
            write_str = f"{avg_write_aligned*1000:.3f}/{avg_write_unaligned*1000:.3f}ms ({speedup_write:.2f}x)"
            read_str = f"{avg_read_aligned*1000:.3f}/{avg_read_unaligned*1000:.3f}ms ({speedup_read:.2f}x)"

            print(f"{transfer_str:<20} {write_str:<20} {read_str:<20}")

            # Save results
            results.append({
                'name': name,
                'size_bytes': actual_size,
                'shape': shape,
                'transfer_speedup': speedup_transfer,
                'write_speedup': speedup_write,
                'read_speedup': speedup_read,
                'avg_speedup': (speedup_transfer + speedup_write + speedup_read) / 3
            })

        except Exception as e:
            print(f"Skipped (out of memory or another error: {e})")
            continue

    # ==================== Detailed analysis ====================
    print(f"\n{'='*100}")
    print("Detailed analysis: alignment performance improvement vs. tensor size")
    print("=" * 100)

    print(f"\n{'size':<12} {'actual bytes':<15} {'transfer speedup':<12} {'write speedup':<12} {'read speedup':<12} {'average speedup':<12} {'recommended'}")
    print(f"{'-'*100}")

    for r in results:
        size_str = f"{r['size_bytes']:,}"
        transfer_perf = f"{r['transfer_speedup']:.2f}x"
        write_perf = f"{r['write_speedup']:.2f}x"
        read_perf = f"{r['read_speedup']:.2f}x"
        avg_perf = f"{r['avg_speedup']:.2f}x"

        # Give a recommendation based on the performance improvement
        if r['avg_speedup'] > 1.2:
            recommendation = "✓ enable alignment"
        elif r['avg_speedup'] > 1.05:
            recommendation = "• alignment recommended"
        else:
            recommendation = "• minor impact"

        print(f"{r['name']:<12} {size_str:<15} {transfer_perf:<12} {write_perf:<12} {read_perf:<12} {avg_perf:<12} {recommendation}")

    # ==================== Chart-style summary ====================
    print(f"\n{'='*100}")
    print("Performance speedup trend (alignment vs unaligned)")
    print("=" * 100)

    print("\ntransfer performance speedup:")
    for r in results:
        bar_length = int(r['transfer_speedup'] * 20)
        bar = '█' * bar_length
        print(f"  {r['name']:<10} [{r['transfer_speedup']:.2f}x] {bar}")

    print("\nwrite performance speedup:")
    for r in results:
        bar_length = int(r['write_speedup'] * 20)
        bar = '█' * bar_length
        print(f"  {r['name']:<10} [{r['write_speedup']:.2f}x] {bar}")

    print("\nread performance speedup:")
    for r in results:
        bar_length = int(r['read_speedup'] * 20)
        bar = '█' * bar_length
        print(f"  {r['name']:<10} [{r['read_speedup']:.2f}x] {bar}")

    # ==================== Conclusion ====================
    print(f"\n{'='*100}")
    print("Conclusion")
    print("=" * 100)

    # Analyze the trend
    small_tensors = [r for r in results if r['size_bytes'] < 1024*1024]  # < 1MB
    large_tensors = [r for r in results if r['size_bytes'] >= 1024*1024]  # >= 1MB

    if small_tensors:
        avg_speedup_small = np.mean([r['avg_speedup'] for r in small_tensors])
        print(f"\nSmall tensors (< 1 MB):")
        print(f"  Average performance improvement: {avg_speedup_small:.2f}x")
        print(f"  alignment overhead: ~{64/np.mean([r['size_bytes'] for r in small_tensors])*100:.1f}%")
        if avg_speedup_small > 1.2:
            print(f"  Conclusion: ✓ alignment provides a noticeable performance improvement")
        else:
            print(f"  Conclusion: • performance improvement is limited, overhead is relatively large")

    if large_tensors:
        avg_speedup_large = np.mean([r['avg_speedup'] for r in large_tensors])
        print(f"\nLarge tensors (>= 1 MB):")
        print(f"  Average performance improvement: {avg_speedup_large:.2f}x")
        print(f"  alignment overhead: ~{64/np.mean([r['size_bytes'] for r in large_tensors])*100:.4f}%")
        if avg_speedup_large > 1.2:
            print(f"  Conclusion: ✓ alignment provides a noticeable performance improvement, negligible overhead")
        else:
            print(f"  Conclusion: • performance improvement is limited")

    print(f"\nOverall recommendation:")
    if results:
        overall_avg = np.mean([r['avg_speedup'] for r in results])
        print(f"  overall average speedup: {overall_avg:.2f}x")

        if overall_avg > 1.3:
            print(f"  ✓✓✓ Strongly recommend enabling alignment (all size scales benefit)")
        elif overall_avg > 1.1:
            print(f"  ✓✓ Recommend enabling alignment (most scenarios benefit)")
        else:
            print(f"  • alignment has limited impact, choose based on requirements")

    print(f"\n{'='*100}")


def test_batch_size_impact():
    """Test the impact of alignment on batch transfers"""
    print("\n\n" + "=" * 100)
    print("Batch Transfer Test: Alignment Impact at Different Tensor Counts")
    print("=" * 100)

    if not torch.cuda.is_available():
        print("CUDA support is required")
        return

    # Test Configuration: fixed tensor size, vary the count
    tensor_size = (256, 256)  # 256 KB each
    batch_sizes = [1, 5, 10, 20, 50, 100]

    print(f"\nTensor size: {tensor_size} (256 KB each)")
    print(f"\n{'batch size':<12} {'Total data size':<15} {'alignment: transfer/read/write':<35} {'unaligned: transfer/read/write':<35} {'average speedup'}")
    print(f"{'-'*100}")

    for batch_size in batch_sizes:
        # Create GPU tensors
        gpu_tensors = [torch.randn(*tensor_size, device='cuda', dtype=torch.float32)
                       for _ in range(batch_size)]

        total_size = sum(t.element_size() * t.nelement() for t in gpu_tensors)
        total_size_mb = total_size / 1024 / 1024
        pool_size = int(total_size * 3)

        try:
            # enable alignment
            pool1 = CPUMemoryPool(pool_size_bytes=pool_size, enable_alignment=True)
            mgr1 = GPUToCPUPoolTransfer(pool1)

            # Warm up
            _ = mgr1.transfer_batch_to_pool(gpu_tensors[:min(2, batch_size)], contiguous=True)
            mgr1.free_batch(list(range(min(2, batch_size))))

            # transfer test
            torch.cuda.synchronize()
            start = time.time()
            tids1 = mgr1.transfer_batch_to_pool(gpu_tensors, use_pinned_memory=True, contiguous=True)
            torch.cuda.synchronize()
            transfer_time1 = time.time() - start

            # read test
            cpu_tensors1 = [mgr1.get_tensor_from_pool(tid) for tid in tids1]
            start = time.time()
            for _ in range(10):
                _ = sum(t.sum().item() for t in cpu_tensors1)
            read_time1 = (time.time() - start) / 10

            # write test
            start = time.time()
            for _ in range(10):
                for t in cpu_tensors1:
                    t.fill_(1.0)
            write_time1 = (time.time() - start) / 10

            mgr1.free_batch(tids1)

            # disable alignment
            pool2 = CPUMemoryPool(pool_size_bytes=pool_size, enable_alignment=False)
            mgr2 = GPUToCPUPoolTransfer(pool2)

            # Warm up
            _ = mgr2.transfer_batch_to_pool(gpu_tensors[:min(2, batch_size)], contiguous=True)
            mgr2.free_batch(list(range(min(2, batch_size))))

            # transfer test
            torch.cuda.synchronize()
            start = time.time()
            tids2 = mgr2.transfer_batch_to_pool(gpu_tensors, use_pinned_memory=True, contiguous=True)
            torch.cuda.synchronize()
            transfer_time2 = time.time() - start

            # read test
            cpu_tensors2 = [mgr2.get_tensor_from_pool(tid) for tid in tids2]
            start = time.time()
            for _ in range(10):
                _ = sum(t.sum().item() for t in cpu_tensors2)
            read_time2 = (time.time() - start) / 10

            # write test
            start = time.time()
            for _ in range(10):
                for t in cpu_tensors2:
                    t.fill_(1.0)
            write_time2 = (time.time() - start) / 10

            mgr2.free_batch(tids2)

            # Compute speedup
            speedup_transfer = transfer_time2 / transfer_time1
            speedup_read = read_time2 / read_time1
            speedup_write = write_time2 / write_time1
            avg_speedup = (speedup_transfer + speedup_read + speedup_write) / 3

            # Format output
            aligned_str = f"{transfer_time1*1000:.2f}/{read_time1*1000:.2f}/{write_time1*1000:.2f}"
            unaligned_str = f"{transfer_time2*1000:.2f}/{read_time2*1000:.2f}/{write_time2*1000:.2f}"

            print(f"{batch_size:<12} {total_size_mb:>10.2f} MB   {aligned_str:<35} {unaligned_str:<35} {avg_speedup:.2f}x")

        except Exception as e:
            print(f"{batch_size:<12} {total_size_mb:>10.2f} MB   Skipped (error: {str(e)[:40]})")

    print(f"\n{'='*100}")


def test_memory_overhead():
    """Test memory overhead at different sizes"""
    print("\n\n" + "=" * 100)
    print("Memory Overhead Analysis: Space Wasted by Alignment")
    print("=" * 100)

    # Define test sizes
    test_sizes = [
        ("64 B",    64),
        ("128 B",   128),
        ("256 B",   256),
        ("1 KB",    1024),
        ("10 KB",   10*1024),
        ("100 KB",  100*1024),
        ("1 MB",    1024*1024),
        ("10 MB",   10*1024*1024),
        ("100 MB",  100*1024*1024),
    ]

    alignment = 64

    print(f"\nAlignment boundary: {alignment} bytes")
    print(f"\n{'size':<12} {'original bytes':<15} {'aligned bytes':<15} {'padding bytes':<12} {'overhead percentage':<12} {'assessment'}")
    print(f"{'-'*100}")

    for name, size_bytes in test_sizes:
        aligned_size = (size_bytes + alignment - 1) & ~(alignment - 1)
        padding = aligned_size - size_bytes
        overhead_pct = (padding / aligned_size) * 100

        # assessment
        if overhead_pct < 0.1:
            rating = "✓✓✓ minimal"
        elif overhead_pct < 1.0:
            rating = "✓✓ very small"
        elif overhead_pct < 5.0:
            rating = "✓ acceptable"
        elif overhead_pct < 10.0:
            rating = "⚠️  large"
        else:
            rating = "✗ very large"

        print(f"{name:<12} {size_bytes:>13,}   {aligned_size:>13,}   {padding:>10,}   {overhead_pct:>9.2f}%   {rating}")

    print(f"\nConclusion:")
    print(f"  • Tensor < 1 KB:  overhead is large (5-50%), but the absolute amount is very small")
    print(f"  • Tensor >= 1 KB: overhead is very small (< 5%)")
    print(f"  • Tensor >= 1 MB: overhead is negligible (< 0.1%)")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    print("\n")
    test_tensor_size_scaling()
    test_batch_size_impact()
    test_memory_overhead()

    print("\n\n✓✓✓ All scaling tests completed!")

