"""
验证 Pinned Pool 功能
======================

验证使用 mlock() 实现的 pinned pool 是否正常工作
"""

import torch
from gpu_cpu_memory_pool import CPUMemoryPool, GPUToCPUPoolTransfer


def verify_pinned_pool():
    """验证 pinned pool 功能"""
    print("=" * 80)
    print("Pinned Pool 功能验证")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 支持")
        return
    
    # 创建 pinned pool
    print("\n创建 Pinned Pool (100 MB)...")
    pool = CPUMemoryPool(
        pool_size_bytes=100 * 1024 * 1024,
        use_pinned_pool=True  # ⭐ 关键参数
    )
    
    print(f"  Pool 是否 pinned: {pool.use_pinned_pool}")
    print(f"  基地址: 0x{pool.base_address:x}")
    
    # 创建传输管理器
    transfer_mgr = GPUToCPUPoolTransfer(pool)
    
    # 创建 GPU tensors
    print("\n创建 GPU tensors...")
    gpu_tensors = [
        torch.randn(100, 100, device='cuda'),
        torch.randn(200, 200, device='cuda'),
        torch.randn(150, 150, device='cuda'),
    ]
    
    # 传输
    print("\n传输到 pinned pool...")
    tensor_ids = transfer_mgr.transfer_batch_to_pool(
        gpu_tensors,
        contiguous=True
        # 注意：不需要 use_pinned_memory=True，池本身就是 pinned
    )
    
    print(f"  传输完成，tensor IDs: {tensor_ids}")
    
    # 验证数据
    print("\n验证数据完整性...")
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
        print("\n✓ 所有数据正确，Pinned Pool 工作正常！")
    else:
        print("\n✗ 数据验证失败")
    
    # 清理
    transfer_mgr.free_batch(tensor_ids)
    
    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)
    
    print("\n说明:")
    print("  • PyTorch 的 torch.empty(..., pin_memory=True)")
    print("  • 更灵活，性能更好！")


if __name__ == "__main__":
    verify_pinned_pool()

