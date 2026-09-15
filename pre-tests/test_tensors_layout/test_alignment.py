"""
Test memory contiguity with and without alignment
===================================

Verify whether contiguous allocations remain contiguous with alignment enabled or disabled
"""

import torch
import time
import numpy as np
from gpu_cpu_memory_pool import CPUMemoryPool, GPUToCPUPoolTransfer


def test_alignment_contiguity():
    """Test memory contiguity with and without alignment"""
    print("=" * 80)
    print("Test: Aligned vs. Unaligned - Memory Contiguity Verification")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA support is required")
        return

    # Create test data - intentionally choose unaligned sizes
    test_sizes = [
        (100, 100),   # 40,000 bytes
        (73, 137),    # 40,024 bytes (unusual size)
        (91, 103),    # 37,492 bytes
        (127, 127),   # 64,516 bytes
        (89, 149),    # 53,044 bytes
    ]

    gpu_tensors = []
    actual_sizes = []
    for shape in test_sizes:
        tensor = torch.randn(*shape, device='cuda', dtype=torch.float32)
        gpu_tensors.append(tensor)
        size = tensor.element_size() * tensor.nelement()
        actual_sizes.append(size)

    print(f"\nCreate {len(gpu_tensors)} GPU tensors (sizes intentionally unaligned):")
    for i, size in enumerate(actual_sizes):
        aligned_64 = (size + 63) & ~63
        padding = aligned_64 - size
        print(f"  Tensor {i}: {size:,} bytes, aligned size: {aligned_64:,} bytes, padding: {padding} bytes")

    # ==================== Test 1: enable alignment ====================
    print(f"\n{'='*80}")
    print("Test 1: enable alignment (enable_alignment=True)")
    print("=" * 80)

    pool1 = CPUMemoryPool(
        pool_size_bytes=100 * 1024 * 1024,
        alignment=64,
        enable_alignment=True  # enable alignment
    )
    transfer_mgr1 = GPUToCPUPoolTransfer(pool1)

    print(f"\nmemory pool Configuration:")
    print(f"  alignment: enabled (64 bytes)")
    print(f"  actual alignment: {pool1.alignment}")

    # batch transfer
    tensor_ids1 = transfer_mgr1.transfer_batch_to_pool(
        gpu_tensors,
        use_pinned_memory=False,
        contiguous=True
    )

    # Verify contiguity
    print(f"\nMemory allocation details:")
    print(f"  {'ID':<5} {'address':>18} {'original size':>12} {'allocated size':>12} {'padding':>8} {'gap':>12}")
    print(f"  {'-'*75}")

    all_contiguous1 = True
    prev_end = None
    total_allocated = 0
    total_wasted = 0

    for i, tid in enumerate(tensor_ids1):
        metadata = transfer_mgr1.tensor_metadata[tid]
        addr = metadata['address']
        allocated_size = metadata['size']
        original_size = actual_sizes[i]
        padding = allocated_size - original_size
        total_allocated += allocated_size
        total_wasted += padding

        if prev_end is not None:
            gap = addr - prev_end
            gap_str = f"{gap:,} bytes" if gap > 0 else "contiguous ✓"
            if gap > 0:
                all_contiguous1 = False
        else:
            gap_str = "first"

        print(f"  {tid:<5} 0x{addr:016x} {original_size:>10,} {allocated_size:>10,} {padding:>6} {gap_str:>12}")
        prev_end = addr + allocated_size

    print(f"\nContiguity verification:")
    print(f"  Status: {'✓ all addresses are contiguous' if all_contiguous1 else '✗ gaps exist'}")
    print(f"  total allocated: {total_allocated:,} bytes")
    print(f"  actual data: {sum(actual_sizes):,} bytes")
    print(f"  padding overhead: {total_wasted:,} bytes ({total_wasted/total_allocated*100:.2f}%)")

    # Clean up
    transfer_mgr1.free_batch(tensor_ids1)

    # ==================== Test 2: disable alignment ====================
    print(f"\n{'='*80}")
    print("Test 2: disable alignment (enable_alignment=False)")
    print("=" * 80)

    pool2 = CPUMemoryPool(
        pool_size_bytes=100 * 1024 * 1024,
        alignment=64,  # this value is ignored
        enable_alignment=False  # disable alignment
    )
    transfer_mgr2 = GPUToCPUPoolTransfer(pool2)

    print(f"\nmemory pool Configuration:")
    print(f"  alignment: disabled")
    print(f"  actual alignment: {pool2.alignment} (unaligned)")

    # batch transfer
    tensor_ids2 = transfer_mgr2.transfer_batch_to_pool(
        gpu_tensors,
        use_pinned_memory=False,
        contiguous=True
    )

    # Verify contiguity
    print(f"\nMemory allocation details:")
    print(f"  {'ID':<5} {'address':>18} {'original size':>12} {'allocated size':>12} {'padding':>8} {'gap':>12}")
    print(f"  {'-'*75}")

    all_contiguous2 = True
    prev_end = None
    total_allocated2 = 0
    total_wasted2 = 0

    for i, tid in enumerate(tensor_ids2):
        metadata = transfer_mgr2.tensor_metadata[tid]
        addr = metadata['address']
        allocated_size = metadata['size']
        original_size = actual_sizes[i]
        padding = allocated_size - original_size
        total_allocated2 += allocated_size
        total_wasted2 += padding

        if prev_end is not None:
            gap = addr - prev_end
            gap_str = f"{gap:,} bytes" if gap > 0 else "contiguous ✓"
            if gap > 0:
                all_contiguous2 = False
        else:
            gap_str = "first"

        print(f"  {tid:<5} 0x{addr:016x} {original_size:>10,} {allocated_size:>10,} {padding:>6} {gap_str:>12}")
        prev_end = addr + allocated_size

    print(f"\nContiguity verification:")
    print(f"  Status: {'✓ all addresses are contiguous' if all_contiguous2 else '✗ gaps exist'}")
    print(f"  total allocated: {total_allocated2:,} bytes")
    print(f"  actual data: {sum(actual_sizes):,} bytes")
    print(f"  padding overhead: {total_wasted2:,} bytes ({total_wasted2/total_allocated2*100:.2f}%)")

    # Clean up
    transfer_mgr2.free_batch(tensor_ids2)

    # ==================== Comparison summary ====================
    print(f"\n{'='*80}")
    print("Comparison summary")
    print("=" * 80)

    print(f"\n{'Feature':<30} {'enable alignment':<20} {'disable alignment':<20}")
    print(f"{'-'*70}")
    print(f"{'memory contiguity':<30} {'✓ contiguous' if all_contiguous1 else '✗ not contiguous':<20} {'✓ contiguous' if all_contiguous2 else '✗ not contiguous':<20}")
    print(f"{'total allocated size':<30} {f'{total_allocated:,} bytes':<20} {f'{total_allocated2:,} bytes':<20}")
    print(f"{'padding overhead':<30} {f'{total_wasted:,} bytes':<20} {f'{total_wasted2:,} bytes':<20}")
    print(f"{'overhead percentage':<30} {f'{total_wasted/total_allocated*100:.2f}%':<20} {f'{total_wasted2/total_allocated2*100:.2f}%':<20}")
    print(f"{'memory utilization':<30} {f'{100-total_wasted/total_allocated*100:.2f}%':<20} {f'{100-total_wasted2/total_allocated2*100:.2f}%':<20}")

    space_saved = total_allocated - total_allocated2
    print(f"\nSpace saved:")
    print(f"  Disabling alignment saves: {space_saved:,} bytes ({space_saved/total_allocated*100:.2f}%)")

    print(f"\nConclusion:")
    if all_contiguous1 and all_contiguous2:
        print(f"  ✓✓✓ Both modes keep memory contiguous!")
    else:
        print(f"  ✗ Memory is not contiguous in one mode")

    print(f"  • alignment enabled: wastes {total_wasted/total_allocated*100:.2f}% space, but offers better performance")
    print(f"  • alignment disabled: wastes {total_wasted2/total_allocated2*100:.2f}% space, 100% utilization")

    print(f"\n{'='*80}")
    print("Test completed!")
    print("=" * 80)


def test_alignment_address_details():
    """Show address alignment details"""
    print("\n\n" + "=" * 80)
    print("Detailed test: address alignment analysis")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA support is required")
        return

    # Create several small tensors
    gpu_tensors = [
        torch.randn(50, 50, device='cuda', dtype=torch.float32),   # 10,000 bytes
        torch.randn(50, 50, device='cuda', dtype=torch.float32),   # 10,000 bytes
        torch.randn(50, 50, device='cuda', dtype=torch.float32),   # 10,000 bytes
    ]

    for enable_align in [True, False]:
        mode = "enable alignment" if enable_align else "disable alignment"
        print(f"\n{'-'*80}")
        print(f"{mode}")
        print(f"{'-'*80}")

        pool = CPUMemoryPool(
            pool_size_bytes=10 * 1024 * 1024,
            alignment=64,
            enable_alignment=enable_align
        )
        transfer_mgr = GPUToCPUPoolTransfer(pool)

        tensor_ids = transfer_mgr.transfer_batch_to_pool(gpu_tensors, contiguous=True)

        print(f"\n  Pool base address: 0x{pool.base_address:x}")
        print(f"  Base address % 64 = {pool.base_address % 64} ({'alignment' if pool.base_address % 64 == 0 else 'unaligned'})")

        print(f"\n  Tensor address analysis:")
        for i, tid in enumerate(tensor_ids):
            metadata = transfer_mgr.tensor_metadata[tid]
            addr = metadata['address']
            size = metadata['size']
            offset = addr - pool.base_address

            print(f"    Tensor {i}:")
            print(f"      address:     0x{addr:x}")
            print(f"      offset:     {offset:,} bytes")
            print(f"      address % 64: {addr % 64} ({'✓ alignment' if addr % 64 == 0 else '✗ unaligned'})")
            print(f"      size:     {size:,} bytes")

        transfer_mgr.free_batch(tensor_ids)


def test_alignment_performance():
    """Test alignment vs unaligned read/write performance"""
    print("\n\n" + "=" * 80)
    print("Performance Test: Aligned vs. Unaligned Read/Write Performance")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA support is required")
        return

    # Test Configuration
    num_tensors = 10
    tensor_size = (512, 512)  # each 1 MB
    num_iterations = 100

    print(f"\nTest Configuration:")
    print(f"  Tensor count: {num_tensors}")
    print(f"  Tensor size: {tensor_size}")
    print(f"  Iterations: {num_iterations}")

    # Create GPU tensors
    gpu_tensors = []
    for _ in range(num_tensors):
        tensor = torch.randn(*tensor_size, device='cuda', dtype=torch.float32)
        gpu_tensors.append(tensor)

    total_size_mb = sum(t.element_size() * t.nelement() for t in gpu_tensors) / 1024 / 1024
    print(f"  Total data size: {total_size_mb:.2f} MB")

    # ==================== Test 1: enable alignment ====================
    print(f"\n{'='*80}")
    print("enable alignment (enable_alignment=True)")
    print("=" * 80)

    pool1 = CPUMemoryPool(
        pool_size_bytes=int(total_size_mb * 2 * 1024 * 1024),
        alignment=64,
        enable_alignment=True
    )
    transfer_mgr1 = GPUToCPUPoolTransfer(pool1)

    # transfer to the memory pool
    print("\n1. GPU → CPU transfer...")
    transfer_start = time.time()
    tensor_ids1 = transfer_mgr1.transfer_batch_to_pool(
        gpu_tensors,
        use_pinned_memory=True,
        contiguous=True
    )
    torch.cuda.synchronize()
    transfer_time1 = time.time() - transfer_start

    print(f"   transfer time: {transfer_time1*1000:.2f} ms")
    print(f"   transfer bandwidth: {total_size_mb / transfer_time1:.2f} MB/s")

    # Get CPU tensors (zero-copy)
    cpu_tensors1 = [transfer_mgr1.get_tensor_from_pool(tid) for tid in tensor_ids1]

    # Test write performance
    print("\n2. CPU Tensor write performance...")
    write_times1 = []

    for _ in range(3):  # Warm up
        for cpu_tensor in cpu_tensors1:
            cpu_tensor.fill_(1.0)

    for i in range(num_iterations):
        start = time.time()
        for cpu_tensor in cpu_tensors1:
            cpu_tensor.fill_(float(i))
        elapsed = time.time() - start
        write_times1.append(elapsed)

    avg_write_time1 = np.mean(write_times1)
    std_write_time1 = np.std(write_times1)
    write_bandwidth1 = total_size_mb / avg_write_time1

    print(f"   Average write time: {avg_write_time1*1000:.2f} ± {std_write_time1*1000:.2f} ms")
    print(f"   write bandwidth: {write_bandwidth1:.2f} MB/s")

    # Test read performance
    print("\n3. CPU Tensor read performance...")
    read_times1 = []

    for _ in range(3):  # Warm up
        for cpu_tensor in cpu_tensors1:
            _ = cpu_tensor.sum()

    for i in range(num_iterations):
        start = time.time()
        total_sum = 0.0
        for cpu_tensor in cpu_tensors1:
            total_sum += cpu_tensor.sum().item()
        elapsed = time.time() - start
        read_times1.append(elapsed)

    avg_read_time1 = np.mean(read_times1)
    std_read_time1 = np.std(read_times1)
    read_bandwidth1 = total_size_mb / avg_read_time1

    print(f"   Average read time: {avg_read_time1*1000:.2f} ± {std_read_time1*1000:.2f} ms")
    print(f"   read bandwidth: {read_bandwidth1:.2f} MB/s")

    # Test random access
    print("\n4. CPU Tensor random access performance...")
    random_access_times1 = []

    for i in range(num_iterations):
        start = time.time()
        for cpu_tensor in cpu_tensors1:
            # Randomly access some elements
            for _ in range(100):
                idx1 = np.random.randint(0, tensor_size[0])
                idx2 = np.random.randint(0, tensor_size[1])
                _ = cpu_tensor[idx1, idx2].item()
        elapsed = time.time() - start
        random_access_times1.append(elapsed)

    avg_random_time1 = np.mean(random_access_times1)
    std_random_time1 = np.std(random_access_times1)

    print(f"   Average random access time: {avg_random_time1*1000:.2f} ± {std_random_time1*1000:.2f} ms")

    # Clean up
    transfer_mgr1.free_batch(tensor_ids1)

    # ==================== Test 2: disable alignment ====================
    print(f"\n{'='*80}")
    print("disable alignment (enable_alignment=False)")
    print("=" * 80)

    pool2 = CPUMemoryPool(
        pool_size_bytes=int(total_size_mb * 2 * 1024 * 1024),
        alignment=64,
        enable_alignment=False
    )
    transfer_mgr2 = GPUToCPUPoolTransfer(pool2)

    # transfer to the memory pool
    print("\n1. GPU → CPU transfer...")
    transfer_start = time.time()
    tensor_ids2 = transfer_mgr2.transfer_batch_to_pool(
        gpu_tensors,
        use_pinned_memory=True,
        contiguous=True
    )
    torch.cuda.synchronize()
    transfer_time2 = time.time() - transfer_start

    print(f"   transfer time: {transfer_time2*1000:.2f} ms")
    print(f"   transfer bandwidth: {total_size_mb / transfer_time2:.2f} MB/s")

    # Get CPU tensors (zero-copy)
    cpu_tensors2 = [transfer_mgr2.get_tensor_from_pool(tid) for tid in tensor_ids2]

    # Test write performance
    print("\n2. CPU Tensor write performance...")
    write_times2 = []

    for _ in range(3):  # Warm up
        for cpu_tensor in cpu_tensors2:
            cpu_tensor.fill_(1.0)

    for i in range(num_iterations):
        start = time.time()
        for cpu_tensor in cpu_tensors2:
            cpu_tensor.fill_(float(i))
        elapsed = time.time() - start
        write_times2.append(elapsed)

    avg_write_time2 = np.mean(write_times2)
    std_write_time2 = np.std(write_times2)
    write_bandwidth2 = total_size_mb / avg_write_time2

    print(f"   Average write time: {avg_write_time2*1000:.2f} ± {std_write_time2*1000:.2f} ms")
    print(f"   write bandwidth: {write_bandwidth2:.2f} MB/s")

    # Test read performance
    print("\n3. CPU Tensor read performance...")
    read_times2 = []

    for _ in range(3):  # Warm up
        for cpu_tensor in cpu_tensors2:
            _ = cpu_tensor.sum()

    for i in range(num_iterations):
        start = time.time()
        total_sum = 0.0
        for cpu_tensor in cpu_tensors2:
            total_sum += cpu_tensor.sum().item()
        elapsed = time.time() - start
        read_times2.append(elapsed)

    avg_read_time2 = np.mean(read_times2)
    std_read_time2 = np.std(read_times2)
    read_bandwidth2 = total_size_mb / avg_read_time2

    print(f"   Average read time: {avg_read_time2*1000:.2f} ± {std_read_time2*1000:.2f} ms")
    print(f"   read bandwidth: {read_bandwidth2:.2f} MB/s")

    # Test random access
    print("\n4. CPU Tensor random access performance...")
    random_access_times2 = []

    for i in range(num_iterations):
        start = time.time()
        for cpu_tensor in cpu_tensors2:
            # Randomly access some elements
            for _ in range(100):
                idx1 = np.random.randint(0, tensor_size[0])
                idx2 = np.random.randint(0, tensor_size[1])
                _ = cpu_tensor[idx1, idx2].item()
        elapsed = time.time() - start
        random_access_times2.append(elapsed)

    avg_random_time2 = np.mean(random_access_times2)
    std_random_time2 = np.std(random_access_times2)

    print(f"   Average random access time: {avg_random_time2*1000:.2f} ± {std_random_time2*1000:.2f} ms")

    # Clean up
    transfer_mgr2.free_batch(tensor_ids2)

    # ==================== Performance Comparison Summary ====================
    print(f"\n{'='*80}")
    print("Performance Comparison Summary")
    print("=" * 80)

    print(f"\nData transfer performance (GPU → CPU):")
    print(f"  {'Mode':<30} {'time (ms)':<15} {'bandwidth (MB/s)':<15} {'Relative speed'}")
    print(f"  {'-'*75}")
    print(f"  {'enable alignment':<30} {transfer_time1*1000:>10.2f}      {total_size_mb/transfer_time1:>10.2f}      1.00x")
    speedup_transfer = transfer_time1 / transfer_time2
    print(f"  {'disable alignment':<30} {transfer_time2*1000:>10.2f}      {total_size_mb/transfer_time2:>10.2f}      {speedup_transfer:.2f}x")

    print(f"\nCPU write performance:")
    print(f"  {'Mode':<30} {'time (ms)':<15} {'bandwidth (MB/s)':<15} {'Relative speed'}")
    print(f"  {'-'*75}")
    print(f"  {'enable alignment':<30} {avg_write_time1*1000:>10.2f}      {write_bandwidth1:>10.2f}      1.00x")
    speedup_write = avg_write_time2 / avg_write_time1
    print(f"  {'disable alignment':<30} {avg_write_time2*1000:>10.2f}      {write_bandwidth2:>10.2f}      {speedup_write:.2f}x")

    print(f"\nCPU read performance:")
    print(f"  {'Mode':<30} {'time (ms)':<15} {'bandwidth (MB/s)':<15} {'Relative speed'}")
    print(f"  {'-'*75}")
    print(f"  {'enable alignment':<30} {avg_read_time1*1000:>10.2f}      {read_bandwidth1:>10.2f}      1.00x")
    speedup_read = avg_read_time2 / avg_read_time1
    print(f"  {'disable alignment':<30} {avg_read_time2*1000:>10.2f}      {read_bandwidth2:>10.2f}      {speedup_read:.2f}x")

    print(f"\nCPU random access performance:")
    print(f"  {'Mode':<30} {'time (ms)':<15} {'Relative speed'}")
    print(f"  {'-'*60}")
    print(f"  {'enable alignment':<30} {avg_random_time1*1000:>10.2f}      1.00x")
    speedup_random = avg_random_time2 / avg_random_time1
    print(f"  {'disable alignment':<30} {avg_random_time2*1000:>10.2f}      {speedup_random:.2f}x")

    print(f"\nOverall performance analysis:")
    print(f"  Enabled alignment compared with disabled alignment:")

    if speedup_write > 1.1:
        print(f"    write: ⚡ faster {(speedup_write-1)*100:.1f}%")
    elif speedup_write < 0.9:
        print(f"    write: ⚠️  slower {(1-speedup_write)*100:.1f}%")
    else:
        print(f"    write: ≈ similar ({speedup_write:.2f}x)")

    if speedup_read > 1.1:
        print(f"    read: ⚡ faster {(speedup_read-1)*100:.1f}%")
    elif speedup_read < 0.9:
        print(f"    read: ⚠️  slower {(1-speedup_read)*100:.1f}%")
    else:
        print(f"    read: ≈ similar ({speedup_read:.2f}x)")

    if speedup_random > 1.1:
        print(f"    random access: ⚡ faster {(speedup_random-1)*100:.1f}%")
    elif speedup_random < 0.9:
        print(f"    random access: ⚠️  slower {(1-speedup_random)*100:.1f}%")
    else:
        print(f"    random access: ≈ similar ({speedup_random:.2f}x)")

    avg_speedup = (speedup_write + speedup_read + speedup_random) / 3
    print(f"\n  Average performance improvement: {avg_speedup:.2f}x ({'⚡ alignment is faster' if avg_speedup > 1.05 else '≈ both are similar'})")

    print(f"\nConclusion:")
    if avg_speedup > 1.2:
        print(f"  ✓ enabling alignment provides a significant performance improvement ({avg_speedup:.2f}x)")
        print(f"  ✓ Enable alignment in production")
    elif avg_speedup > 1.05:
        print(f"  ✓ enabling alignment provides some performance improvement ({avg_speedup:.2f}x)")
        print(f"  ✓ enabling alignment is recommended")
    else:
        print(f"  • both modes perform similarly is similar ({avg_speedup:.2f}x)")
        print(f"  • Choose based on memory requirements")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    test_alignment_contiguity()
    test_alignment_address_details()
    test_alignment_performance()

    print("\n\n✓✓✓ All tests completed!")

