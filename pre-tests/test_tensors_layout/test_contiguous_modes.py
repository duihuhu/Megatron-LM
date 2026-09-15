"""
Compare contiguous and independent allocations
==========================

Demonstrate the difference between contiguous=True and contiguous=False
"""

import torch
from gpu_cpu_memory_pool import CPUMemoryPool, GPUToCPUPoolTransfer


def test_contiguous_true():
    """Test contiguous allocation mode (contiguous=True)"""
    print("=" * 80)
    print("Mode 1: contiguous allocation (contiguous=True)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA support is required")
        return

    # Initialize memory pool
    pool = CPUMemoryPool(pool_size_bytes=100 * 1024 * 1024)
    transfer_mgr = GPUToCPUPoolTransfer(pool)

    # Create multiple GPU tensors
    gpu_tensors = [
        torch.randn(100, 100, device='cuda', dtype=torch.float32),  # 40,000 bytes
        torch.randn(50, 50, device='cuda', dtype=torch.float32),    # 10,000 bytes
        torch.randn(200, 200, device='cuda', dtype=torch.float32),  # 160,000 bytes
    ]

    print("\nCreate tensors:")
    for i, t in enumerate(gpu_tensors):
        size = t.element_size() * t.nelement()
        print(f"  Tensor {i}: {t.shape}, {size:,} bytes")

    # contiguous allocation
    print(f"\nRun batch transfer (contiguous=True)...")
    tensor_ids = transfer_mgr.transfer_batch_to_pool(
        gpu_tensors,
        contiguous=True  # ⭐ contiguous allocation
    )

    # Check addresses
    print("\nMemory allocation results:")
    print("  ID | address              | size       | offset     | gap from previous")
    print("  " + "-" * 75)

    prev_end = None
    for i, tid in enumerate(tensor_ids):
        metadata = transfer_mgr.tensor_metadata[tid]
        addr = metadata['address']
        size = metadata['size']
        offset = addr - pool.base_address

        if prev_end is not None:
            gap = addr - prev_end
            gap_str = f"{gap:8,} bytes" if gap > 0 else "contiguous ✓"
        else:
            gap_str = "first"

        print(f"  {tid:2d} | 0x{addr:016x} | {size:9,} | {offset:10,} | {gap_str}")
        prev_end = addr + size

    # Verify contiguity
    print("\nVerification results:")
    all_contiguous = True
    for i in range(len(tensor_ids) - 1):
        addr1 = transfer_mgr.tensor_metadata[tensor_ids[i]]['address']
        size1 = transfer_mgr.tensor_metadata[tensor_ids[i]]['size']
        addr2 = transfer_mgr.tensor_metadata[tensor_ids[i+1]]['address']

        if addr2 != addr1 + size1:
            all_contiguous = False
            print(f"  ✗ Tensor {i} and {i+1} not contiguous!")

    if all_contiguous:
        print(f"  ✓✓✓ all tensor addresses are fully contiguous, with no gaps!")

    # Clean up
    transfer_mgr.free_batch(tensor_ids)
    print()


def test_contiguous_false():
    """Test independent allocation mode (contiguous=False)"""
    print("=" * 80)
    print("Mode 2: independent allocation (contiguous=False)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA support is required")
        return

    # Initialize memory pool
    pool = CPUMemoryPool(pool_size_bytes=100 * 1024 * 1024)
    transfer_mgr = GPUToCPUPoolTransfer(pool)

    # First allocate several tensors, then free some to create memory fragmentation
    print("\nCreate memory fragmentation (simulate a realistic scenario):")
    temp_tensors = [
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
    ]

    temp_ids = transfer_mgr.transfer_batch_to_pool(temp_tensors, contiguous=True)
    print(f"  Allocate four tensors: {temp_ids}")

    # Free the two middle allocations, create gaps
    transfer_mgr.free_tensor(temp_ids[1])
    transfer_mgr.free_tensor(temp_ids[2])
    print(f"  Free tensor {temp_ids[1]} and {temp_ids[2]}")
    print(f"  the memory now contains gaps (fragmentation)")

    # Create new GPU tensors
    gpu_tensors = [
        torch.randn(50, 50, device='cuda', dtype=torch.float32),
        torch.randn(50, 50, device='cuda', dtype=torch.float32),
        torch.randn(50, 50, device='cuda', dtype=torch.float32),
    ]

    print("\nCreate new tensors:")
    for i, t in enumerate(gpu_tensors):
        size = t.element_size() * t.nelement()
        print(f"  Tensor {i}: {t.shape}, {size:,} bytes")

    # independent allocation
    print(f"\nRun batch transfer (contiguous=False)...")
    tensor_ids = transfer_mgr.transfer_batch_to_pool(
        gpu_tensors,
        contiguous=False  # ⭐ independent allocation, not guaranteed to be contiguous
    )

    # Check addresses
    print("\nMemory allocation results:")
    print("  ID | address              | size       | offset     | gap from previous")
    print("  " + "-" * 75)

    prev_end = None
    has_gap = False

    for i, tid in enumerate(tensor_ids):
        metadata = transfer_mgr.tensor_metadata[tid]
        addr = metadata['address']
        size = metadata['size']
        offset = addr - pool.base_address

        if prev_end is not None:
            gap = addr - prev_end
            if gap > 0:
                gap_str = f"{gap:8,} bytes ⚠️"
                has_gap = True
            else:
                gap_str = "contiguous"
        else:
            gap_str = "first"

        print(f"  {tid:2d} | 0x{addr:016x} | {size:9,} | {offset:10,} | {gap_str}")
        prev_end = addr + size

    # Verify
    print("\nVerification results:")
    if has_gap:
        print(f"  ⚠️  Tensors have gaps between them (this is expected because contiguous=False)")
    else:
        print(f"  ℹ️  Although contiguous=False, the addresses happened to be contiguous")
        print(f"      (this depends on the memory pool state, this behavior is not guaranteed)")

    # Clean up
    transfer_mgr.free_batch(tensor_ids)
    transfer_mgr.free_tensor(temp_ids[0])
    transfer_mgr.free_tensor(temp_ids[3])
    print()


def test_comparison():
    """Directly compare both modes"""
    print("=" * 80)
    print("Comparison Test: Same Tensors, Different Allocation Modes")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA support is required")
        return

    # Create test data
    gpu_tensors = [
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
    ]

    print("\nTest data: three 100x100 tensors\n")

    # Test contiguous=True
    pool1 = CPUMemoryPool(pool_size_bytes=50 * 1024 * 1024)
    mgr1 = GPUToCPUPoolTransfer(pool1)

    print("Approach 1: contiguous=True")
    ids1 = mgr1.transfer_batch_to_pool(gpu_tensors, contiguous=True)

    gaps1 = []
    for i in range(len(ids1) - 1):
        addr1 = mgr1.tensor_metadata[ids1[i]]['address']
        size1 = mgr1.tensor_metadata[ids1[i]]['size']
        addr2 = mgr1.tensor_metadata[ids1[i+1]]['address']
        gap = addr2 - (addr1 + size1)
        gaps1.append(gap)
        print(f"  Tensor {i} → {i+1}: gap = {gap} bytes")

    # Test contiguous=False
    pool2 = CPUMemoryPool(pool_size_bytes=50 * 1024 * 1024)
    mgr2 = GPUToCPUPoolTransfer(pool2)

    print("\nApproach 2: contiguous=False")
    ids2 = mgr2.transfer_batch_to_pool(gpu_tensors, contiguous=False)

    gaps2 = []
    for i in range(len(ids2) - 1):
        addr1 = mgr2.tensor_metadata[ids2[i]]['address']
        size1 = mgr2.tensor_metadata[ids2[i]]['size']
        addr2 = mgr2.tensor_metadata[ids2[i+1]]['address']
        gap = addr2 - (addr1 + size1)
        gaps2.append(gap)
        print(f"  Tensor {i} → {i+1}: gap = {gap} bytes")

    # summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  contiguous=True:  all gaps = {gaps1} → guaranteed contiguous ✓")
    print(f"  contiguous=False: all gaps = {gaps2} → not guaranteed to be contiguous ⚠️")
    print("=" * 80)


if __name__ == "__main__":
    test_contiguous_true()
    print("\n\n")
    test_contiguous_false()
    print("\n\n")
    test_comparison()

    print("\n" + "=" * 80)
    print("Conclusion:")
    print("=" * 80)
    print("✓ contiguous=True  → addresses guaranteed to be contiguous, with no gaps")
    print("⚠ contiguous=False → independent allocation, may have gaps (depends on memory pool status)")
    print("=" * 80)

