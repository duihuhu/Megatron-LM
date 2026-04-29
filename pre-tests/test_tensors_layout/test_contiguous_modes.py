"""
对比连续分配 vs 独立分配
==========================

演示 contiguous=True 和 contiguous=False 的区别
"""

import torch
from gpu_cpu_memory_pool import CPUMemoryPool, GPUToCPUPoolTransfer


def test_contiguous_true():
    """测试连续分配模式 (contiguous=True)"""
    print("=" * 80)
    print("模式 1: 连续分配 (contiguous=True)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 支持")
        return
    
    # 初始化内存池
    pool = CPUMemoryPool(pool_size_bytes=100 * 1024 * 1024)
    transfer_mgr = GPUToCPUPoolTransfer(pool)
    
    # 创建多个 GPU tensors
    gpu_tensors = [
        torch.randn(100, 100, device='cuda', dtype=torch.float32),  # 40,000 bytes
        torch.randn(50, 50, device='cuda', dtype=torch.float32),    # 10,000 bytes
        torch.randn(200, 200, device='cuda', dtype=torch.float32),  # 160,000 bytes
    ]
    
    print("\n创建的 tensors:")
    for i, t in enumerate(gpu_tensors):
        size = t.element_size() * t.nelement()
        print(f"  Tensor {i}: {t.shape}, {size:,} bytes")
    
    # 连续分配
    print(f"\n执行批量传输 (contiguous=True)...")
    tensor_ids = transfer_mgr.transfer_batch_to_pool(
        gpu_tensors,
        contiguous=True  # ⭐ 连续分配
    )
    
    # 检查地址
    print("\n内存分配结果:")
    print("  ID | 地址              | 大小       | 偏移量     | 与上一个的间隔")
    print("  " + "-" * 75)
    
    prev_end = None
    for i, tid in enumerate(tensor_ids):
        metadata = transfer_mgr.tensor_metadata[tid]
        addr = metadata['address']
        size = metadata['size']
        offset = addr - pool.base_address
        
        if prev_end is not None:
            gap = addr - prev_end
            gap_str = f"{gap:8,} bytes" if gap > 0 else "连续 ✓"
        else:
            gap_str = "首个"
        
        print(f"  {tid:2d} | 0x{addr:016x} | {size:9,} | {offset:10,} | {gap_str}")
        prev_end = addr + size
    
    # 验证连续性
    print("\n验证结果:")
    all_contiguous = True
    for i in range(len(tensor_ids) - 1):
        addr1 = transfer_mgr.tensor_metadata[tensor_ids[i]]['address']
        size1 = transfer_mgr.tensor_metadata[tensor_ids[i]]['size']
        addr2 = transfer_mgr.tensor_metadata[tensor_ids[i+1]]['address']
        
        if addr2 != addr1 + size1:
            all_contiguous = False
            print(f"  ✗ Tensor {i} 和 {i+1} 不连续!")
    
    if all_contiguous:
        print(f"  ✓✓✓ 所有 tensor 地址完全连续，无间隔！")
    
    # 清理
    transfer_mgr.free_batch(tensor_ids)
    print()


def test_contiguous_false():
    """测试独立分配模式 (contiguous=False)"""
    print("=" * 80)
    print("模式 2: 独立分配 (contiguous=False)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 支持")
        return
    
    # 初始化内存池
    pool = CPUMemoryPool(pool_size_bytes=100 * 1024 * 1024)
    transfer_mgr = GPUToCPUPoolTransfer(pool)
    
    # 先分配一些tensor，然后释放部分，制造内存碎片
    print("\n制造内存碎片（模拟真实场景）:")
    temp_tensors = [
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
    ]
    
    temp_ids = transfer_mgr.transfer_batch_to_pool(temp_tensors, contiguous=True)
    print(f"  分配了 4 个 tensor: {temp_ids}")
    
    # 释放中间的两个，制造间隔
    transfer_mgr.free_tensor(temp_ids[1])
    transfer_mgr.free_tensor(temp_ids[2])
    print(f"  释放了 tensor {temp_ids[1]} 和 {temp_ids[2]}")
    print(f"  现在内存中有间隔（碎片）")
    
    # 创建新的 GPU tensors
    gpu_tensors = [
        torch.randn(50, 50, device='cuda', dtype=torch.float32),
        torch.randn(50, 50, device='cuda', dtype=torch.float32),
        torch.randn(50, 50, device='cuda', dtype=torch.float32),
    ]
    
    print("\n创建的新 tensors:")
    for i, t in enumerate(gpu_tensors):
        size = t.element_size() * t.nelement()
        print(f"  Tensor {i}: {t.shape}, {size:,} bytes")
    
    # 独立分配
    print(f"\n执行批量传输 (contiguous=False)...")
    tensor_ids = transfer_mgr.transfer_batch_to_pool(
        gpu_tensors,
        contiguous=False  # ⭐ 独立分配，不保证连续
    )
    
    # 检查地址
    print("\n内存分配结果:")
    print("  ID | 地址              | 大小       | 偏移量     | 与上一个的间隔")
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
                gap_str = "连续"
        else:
            gap_str = "首个"
        
        print(f"  {tid:2d} | 0x{addr:016x} | {size:9,} | {offset:10,} | {gap_str}")
        prev_end = addr + size
    
    # 验证
    print("\n验证结果:")
    if has_gap:
        print(f"  ⚠️  Tensor 之间存在间隔（这是正常的，因为 contiguous=False）")
    else:
        print(f"  ℹ️  虽然设置了 contiguous=False，但碰巧地址连续了")
        print(f"     （这取决于内存池状态，不是保证的行为）")
    
    # 清理
    transfer_mgr.free_batch(tensor_ids)
    transfer_mgr.free_tensor(temp_ids[0])
    transfer_mgr.free_tensor(temp_ids[3])
    print()


def test_comparison():
    """直接对比两种模式"""
    print("=" * 80)
    print("对比测试：同样的 tensors，不同的分配模式")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 支持")
        return
    
    # 创建测试数据
    gpu_tensors = [
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
        torch.randn(100, 100, device='cuda'),
    ]
    
    print("\n测试数据: 3 个 100x100 tensors\n")
    
    # 测试 contiguous=True
    pool1 = CPUMemoryPool(pool_size_bytes=50 * 1024 * 1024)
    mgr1 = GPUToCPUPoolTransfer(pool1)
    
    print("方式 1: contiguous=True")
    ids1 = mgr1.transfer_batch_to_pool(gpu_tensors, contiguous=True)
    
    gaps1 = []
    for i in range(len(ids1) - 1):
        addr1 = mgr1.tensor_metadata[ids1[i]]['address']
        size1 = mgr1.tensor_metadata[ids1[i]]['size']
        addr2 = mgr1.tensor_metadata[ids1[i+1]]['address']
        gap = addr2 - (addr1 + size1)
        gaps1.append(gap)
        print(f"  Tensor {i} → {i+1}: 间隔 = {gap} bytes")
    
    # 测试 contiguous=False
    pool2 = CPUMemoryPool(pool_size_bytes=50 * 1024 * 1024)
    mgr2 = GPUToCPUPoolTransfer(pool2)
    
    print("\n方式 2: contiguous=False")
    ids2 = mgr2.transfer_batch_to_pool(gpu_tensors, contiguous=False)
    
    gaps2 = []
    for i in range(len(ids2) - 1):
        addr1 = mgr2.tensor_metadata[ids2[i]]['address']
        size1 = mgr2.tensor_metadata[ids2[i]]['size']
        addr2 = mgr2.tensor_metadata[ids2[i+1]]['address']
        gap = addr2 - (addr1 + size1)
        gaps2.append(gap)
        print(f"  Tensor {i} → {i+1}: 间隔 = {gap} bytes")
    
    # 总结
    print("\n" + "=" * 80)
    print("总结:")
    print(f"  contiguous=True:  所有间隔 = {gaps1} → 保证连续 ✓")
    print(f"  contiguous=False: 所有间隔 = {gaps2} → 不保证连续 ⚠️")
    print("=" * 80)


if __name__ == "__main__":
    test_contiguous_true()
    print("\n\n")
    test_contiguous_false()
    print("\n\n")
    test_comparison()
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print("✓ contiguous=True  → 保证地址连续，无间隔")
    print("⚠ contiguous=False → 独立分配，可能有间隔（取决于内存池状态）")
    print("=" * 80)

