"""
Verify pinned pool functionality
======================

Verify that the mlock()-based pinned pool works as expected
"""

import torch
from gpu_cpu_memory_pool import CPUMemoryPool, GPUToCPUPoolTransfer


def verify_pinned_pool():
    """Verify pinned pool functionality"""
    print("=" * 80)
    print("Pinned Pool Functionality Verification")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA support is required")
        return

    # Create pinned pool
    print("\nCreate a pinned pool (100 MB)...")
    pool = CPUMemoryPool(
        pool_size_bytes=100 * 1024 * 1024,
        use_pinned_pool=True  # ⭐ key argument
    )

    print(f"  Pool is pinned: {pool.use_pinned_pool}")
    print(f"  Base address: 0x{pool.base_address:x}")

    # Create the transfer manager
    transfer_mgr = GPUToCPUPoolTransfer(pool)

    # Create GPU tensors
    print("\nCreate GPU tensors...")
    gpu_tensors = [
        torch.randn(100, 100, device='cuda'),
        torch.randn(200, 200, device='cuda'),
        torch.randn(150, 150, device='cuda'),
    ]

    # transfer
    print("\nTransfer to the pinned pool...")
    tensor_ids = transfer_mgr.transfer_batch_to_pool(
        gpu_tensors,
        contiguous=True
        # Note: use_pinned_memory=True is unnecessary because the pool itself is pinned
    )

    print(f"  Transfer completed, tensor IDs: {tensor_ids}")

    # Verify data
    print("\nVerify data integrity...")
    all_correct = True
    for i, tid in enumerate(tensor_ids):
        cpu_tensor = transfer_mgr.get_tensor_from_pool(tid)
        gpu_cpu = gpu_tensors[i].cpu()

        max_diff = torch.abs(cpu_tensor - gpu_cpu).max().item()
        status = "✓" if max_diff < 1e-6 else "✗"
        print(f"  Tensor {tid}: max_diff = {max_diff:.2e} {status}")

        if max_diff >= 1e-6:
            all_correct = False

    if all_correct:
        print("\n✓ all data is correct, the pinned pool works correctly!")
    else:
        print("\n✗ data verification failed")

    # Clean up
    transfer_mgr.free_batch(tensor_ids)

    print("\n" + "=" * 80)
    print("Verification completed!")
    print("=" * 80)

    print("\nDescription:")
    print("  • PyTorch  torch.empty(..., pin_memory=True)")
    print("  • More flexible and higher-performance!")


if __name__ == "__main__":
    verify_pinned_pool()

