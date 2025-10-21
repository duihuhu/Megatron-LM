"""
测试对齐 vs 不对齐的内存连续性
===================================

验证启用/禁用对齐时，连续分配是否真的连续
"""

import torch
import time
import numpy as np
from gpu_cpu_memory_pool import CPUMemoryPool, GPUToCPUPoolTransfer


def test_alignment_contiguity():
    """测试对齐和不对齐情况下的内存连续性"""
    print("=" * 80)
    print("测试：对齐 vs 不对齐 - 内存连续性验证")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 支持")
        return
    
    # 创建测试数据 - 故意选择不对齐的大小
    test_sizes = [
        (100, 100),   # 40,000 bytes
        (73, 137),    # 40,024 bytes (奇怪的大小)
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
    
    print(f"\n创建了 {len(gpu_tensors)} 个 GPU tensors（大小故意不对齐）:")
    for i, size in enumerate(actual_sizes):
        aligned_64 = (size + 63) & ~63
        padding = aligned_64 - size
        print(f"  Tensor {i}: {size:,} bytes, 对齐后: {aligned_64:,} bytes, 填充: {padding} bytes")
    
    # ==================== 测试 1: 启用对齐 ====================
    print(f"\n{'='*80}")
    print("测试 1: 启用对齐 (enable_alignment=True)")
    print("=" * 80)
    
    pool1 = CPUMemoryPool(
        pool_size_bytes=100 * 1024 * 1024,
        alignment=64,
        enable_alignment=True  # 启用对齐
    )
    transfer_mgr1 = GPUToCPUPoolTransfer(pool1)
    
    print(f"\n内存池配置:")
    print(f"  对齐: 启用 (64 字节)")
    print(f"  实际对齐值: {pool1.alignment}")
    
    # 批量传输
    tensor_ids1 = transfer_mgr1.transfer_batch_to_pool(
        gpu_tensors,
        use_pinned_memory=False,
        contiguous=True
    )
    
    # 验证连续性
    print(f"\n内存分配详情:")
    print(f"  {'ID':<5} {'地址':>18} {'原始大小':>12} {'分配大小':>12} {'填充':>8} {'与上一个':>12}")
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
            gap_str = f"{gap:,} bytes" if gap > 0 else "连续 ✓"
            if gap > 0:
                all_contiguous1 = False
        else:
            gap_str = "首个"
        
        print(f"  {tid:<5} 0x{addr:016x} {original_size:>10,} {allocated_size:>10,} {padding:>6} {gap_str:>12}")
        prev_end = addr + allocated_size
    
    print(f"\n连续性验证:")
    print(f"  状态: {'✓ 所有地址连续' if all_contiguous1 else '✗ 存在间隔'}")
    print(f"  总分配: {total_allocated:,} bytes")
    print(f"  实际数据: {sum(actual_sizes):,} bytes")
    print(f"  填充开销: {total_wasted:,} bytes ({total_wasted/total_allocated*100:.2f}%)")
    
    # 清理
    transfer_mgr1.free_batch(tensor_ids1)
    
    # ==================== 测试 2: 禁用对齐 ====================
    print(f"\n{'='*80}")
    print("测试 2: 禁用对齐 (enable_alignment=False)")
    print("=" * 80)
    
    pool2 = CPUMemoryPool(
        pool_size_bytes=100 * 1024 * 1024,
        alignment=64,  # 这个值会被忽略
        enable_alignment=False  # 禁用对齐
    )
    transfer_mgr2 = GPUToCPUPoolTransfer(pool2)
    
    print(f"\n内存池配置:")
    print(f"  对齐: 禁用")
    print(f"  实际对齐值: {pool2.alignment} (不对齐)")
    
    # 批量传输
    tensor_ids2 = transfer_mgr2.transfer_batch_to_pool(
        gpu_tensors,
        use_pinned_memory=False,
        contiguous=True
    )
    
    # 验证连续性
    print(f"\n内存分配详情:")
    print(f"  {'ID':<5} {'地址':>18} {'原始大小':>12} {'分配大小':>12} {'填充':>8} {'与上一个':>12}")
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
            gap_str = f"{gap:,} bytes" if gap > 0 else "连续 ✓"
            if gap > 0:
                all_contiguous2 = False
        else:
            gap_str = "首个"
        
        print(f"  {tid:<5} 0x{addr:016x} {original_size:>10,} {allocated_size:>10,} {padding:>6} {gap_str:>12}")
        prev_end = addr + allocated_size
    
    print(f"\n连续性验证:")
    print(f"  状态: {'✓ 所有地址连续' if all_contiguous2 else '✗ 存在间隔'}")
    print(f"  总分配: {total_allocated2:,} bytes")
    print(f"  实际数据: {sum(actual_sizes):,} bytes")
    print(f"  填充开销: {total_wasted2:,} bytes ({total_wasted2/total_allocated2*100:.2f}%)")
    
    # 清理
    transfer_mgr2.free_batch(tensor_ids2)
    
    # ==================== 对比总结 ====================
    print(f"\n{'='*80}")
    print("对比总结")
    print("=" * 80)
    
    print(f"\n{'特性':<30} {'启用对齐':<20} {'禁用对齐':<20}")
    print(f"{'-'*70}")
    print(f"{'内存连续性':<30} {'✓ 连续' if all_contiguous1 else '✗ 不连续':<20} {'✓ 连续' if all_contiguous2 else '✗ 不连续':<20}")
    print(f"{'总分配大小':<30} {f'{total_allocated:,} bytes':<20} {f'{total_allocated2:,} bytes':<20}")
    print(f"{'填充开销':<30} {f'{total_wasted:,} bytes':<20} {f'{total_wasted2:,} bytes':<20}")
    print(f"{'开销占比':<30} {f'{total_wasted/total_allocated*100:.2f}%':<20} {f'{total_wasted2/total_allocated2*100:.2f}%':<20}")
    print(f"{'内存利用率':<30} {f'{100-total_wasted/total_allocated*100:.2f}%':<20} {f'{100-total_wasted2/total_allocated2*100:.2f}%':<20}")
    
    space_saved = total_allocated - total_allocated2
    print(f"\n空间节省:")
    print(f"  禁用对齐节省: {space_saved:,} bytes ({space_saved/total_allocated*100:.2f}%)")
    
    print(f"\n结论:")
    if all_contiguous1 and all_contiguous2:
        print(f"  ✓✓✓ 两种模式都保持内存连续！")
    else:
        print(f"  ✗ 某种模式下内存不连续")
    
    print(f"  • 启用对齐: 浪费 {total_wasted/total_allocated*100:.2f}% 空间，但性能更好")
    print(f"  • 禁用对齐: 浪费 {total_wasted2/total_allocated2*100:.2f}% 空间，100% 利用率")
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print("=" * 80)


def test_alignment_address_details():
    """详细展示地址对齐情况"""
    print("\n\n" + "=" * 80)
    print("详细测试：地址对齐分析")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 支持")
        return
    
    # 创建几个小 tensor
    gpu_tensors = [
        torch.randn(50, 50, device='cuda', dtype=torch.float32),   # 10,000 bytes
        torch.randn(50, 50, device='cuda', dtype=torch.float32),   # 10,000 bytes
        torch.randn(50, 50, device='cuda', dtype=torch.float32),   # 10,000 bytes
    ]
    
    for enable_align in [True, False]:
        mode = "启用对齐" if enable_align else "禁用对齐"
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
        
        print(f"\n  Pool 基地址: 0x{pool.base_address:x}")
        print(f"  基地址 % 64 = {pool.base_address % 64} ({'对齐' if pool.base_address % 64 == 0 else '不对齐'})")
        
        print(f"\n  Tensor 地址分析:")
        for i, tid in enumerate(tensor_ids):
            metadata = transfer_mgr.tensor_metadata[tid]
            addr = metadata['address']
            size = metadata['size']
            offset = addr - pool.base_address
            
            print(f"    Tensor {i}:")
            print(f"      地址:     0x{addr:x}")
            print(f"      偏移:     {offset:,} bytes")
            print(f"      地址 % 64: {addr % 64} ({'✓ 对齐' if addr % 64 == 0 else '✗ 不对齐'})")
            print(f"      大小:     {size:,} bytes")
        
        transfer_mgr.free_batch(tensor_ids)


def test_alignment_performance():
    """测试对齐 vs 不对齐的读写性能"""
    print("\n\n" + "=" * 80)
    print("性能测试：对齐 vs 不对齐 - 读写性能对比")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 支持")
        return
    
    # 测试配置
    num_tensors = 10
    tensor_size = (512, 512)  # 每个 1 MB
    num_iterations = 100
    
    print(f"\n测试配置:")
    print(f"  Tensor 数量: {num_tensors}")
    print(f"  Tensor 大小: {tensor_size}")
    print(f"  迭代次数: {num_iterations}")
    
    # 创建 GPU tensors
    gpu_tensors = []
    for _ in range(num_tensors):
        tensor = torch.randn(*tensor_size, device='cuda', dtype=torch.float32)
        gpu_tensors.append(tensor)
    
    total_size_mb = sum(t.element_size() * t.nelement() for t in gpu_tensors) / 1024 / 1024
    print(f"  总数据量: {total_size_mb:.2f} MB")
    
    # ==================== 测试 1: 启用对齐 ====================
    print(f"\n{'='*80}")
    print("启用对齐 (enable_alignment=True)")
    print("=" * 80)
    
    pool1 = CPUMemoryPool(
        pool_size_bytes=int(total_size_mb * 2 * 1024 * 1024),
        alignment=64,
        enable_alignment=True
    )
    transfer_mgr1 = GPUToCPUPoolTransfer(pool1)
    
    # 传输到内存池
    print("\n1. GPU → CPU 传输...")
    transfer_start = time.time()
    tensor_ids1 = transfer_mgr1.transfer_batch_to_pool(
        gpu_tensors,
        use_pinned_memory=True,
        contiguous=True
    )
    torch.cuda.synchronize()
    transfer_time1 = time.time() - transfer_start
    
    print(f"   传输时间: {transfer_time1*1000:.2f} ms")
    print(f"   传输带宽: {total_size_mb / transfer_time1:.2f} MB/s")
    
    # 获取 CPU tensors（零拷贝）
    cpu_tensors1 = [transfer_mgr1.get_tensor_from_pool(tid) for tid in tensor_ids1]
    
    # 测试写性能
    print("\n2. CPU Tensor 写入性能...")
    write_times1 = []
    
    for _ in range(3):  # 预热
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
    
    print(f"   平均写入时间: {avg_write_time1*1000:.2f} ± {std_write_time1*1000:.2f} ms")
    print(f"   写入带宽: {write_bandwidth1:.2f} MB/s")
    
    # 测试读性能
    print("\n3. CPU Tensor 读取性能...")
    read_times1 = []
    
    for _ in range(3):  # 预热
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
    
    print(f"   平均读取时间: {avg_read_time1*1000:.2f} ± {std_read_time1*1000:.2f} ms")
    print(f"   读取带宽: {read_bandwidth1:.2f} MB/s")
    
    # 测试随机访问
    print("\n4. CPU Tensor 随机访问性能...")
    random_access_times1 = []
    
    for i in range(num_iterations):
        start = time.time()
        for cpu_tensor in cpu_tensors1:
            # 随机访问一些元素
            for _ in range(100):
                idx1 = np.random.randint(0, tensor_size[0])
                idx2 = np.random.randint(0, tensor_size[1])
                _ = cpu_tensor[idx1, idx2].item()
        elapsed = time.time() - start
        random_access_times1.append(elapsed)
    
    avg_random_time1 = np.mean(random_access_times1)
    std_random_time1 = np.std(random_access_times1)
    
    print(f"   平均随机访问时间: {avg_random_time1*1000:.2f} ± {std_random_time1*1000:.2f} ms")
    
    # 清理
    transfer_mgr1.free_batch(tensor_ids1)
    
    # ==================== 测试 2: 禁用对齐 ====================
    print(f"\n{'='*80}")
    print("禁用对齐 (enable_alignment=False)")
    print("=" * 80)
    
    pool2 = CPUMemoryPool(
        pool_size_bytes=int(total_size_mb * 2 * 1024 * 1024),
        alignment=64,
        enable_alignment=False
    )
    transfer_mgr2 = GPUToCPUPoolTransfer(pool2)
    
    # 传输到内存池
    print("\n1. GPU → CPU 传输...")
    transfer_start = time.time()
    tensor_ids2 = transfer_mgr2.transfer_batch_to_pool(
        gpu_tensors,
        use_pinned_memory=True,
        contiguous=True
    )
    torch.cuda.synchronize()
    transfer_time2 = time.time() - transfer_start
    
    print(f"   传输时间: {transfer_time2*1000:.2f} ms")
    print(f"   传输带宽: {total_size_mb / transfer_time2:.2f} MB/s")
    
    # 获取 CPU tensors（零拷贝）
    cpu_tensors2 = [transfer_mgr2.get_tensor_from_pool(tid) for tid in tensor_ids2]
    
    # 测试写性能
    print("\n2. CPU Tensor 写入性能...")
    write_times2 = []
    
    for _ in range(3):  # 预热
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
    
    print(f"   平均写入时间: {avg_write_time2*1000:.2f} ± {std_write_time2*1000:.2f} ms")
    print(f"   写入带宽: {write_bandwidth2:.2f} MB/s")
    
    # 测试读性能
    print("\n3. CPU Tensor 读取性能...")
    read_times2 = []
    
    for _ in range(3):  # 预热
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
    
    print(f"   平均读取时间: {avg_read_time2*1000:.2f} ± {std_read_time2*1000:.2f} ms")
    print(f"   读取带宽: {read_bandwidth2:.2f} MB/s")
    
    # 测试随机访问
    print("\n4. CPU Tensor 随机访问性能...")
    random_access_times2 = []
    
    for i in range(num_iterations):
        start = time.time()
        for cpu_tensor in cpu_tensors2:
            # 随机访问一些元素
            for _ in range(100):
                idx1 = np.random.randint(0, tensor_size[0])
                idx2 = np.random.randint(0, tensor_size[1])
                _ = cpu_tensor[idx1, idx2].item()
        elapsed = time.time() - start
        random_access_times2.append(elapsed)
    
    avg_random_time2 = np.mean(random_access_times2)
    std_random_time2 = np.std(random_access_times2)
    
    print(f"   平均随机访问时间: {avg_random_time2*1000:.2f} ± {std_random_time2*1000:.2f} ms")
    
    # 清理
    transfer_mgr2.free_batch(tensor_ids2)
    
    # ==================== 性能对比总结 ====================
    print(f"\n{'='*80}")
    print("性能对比总结")
    print("=" * 80)
    
    print(f"\n数据传输性能 (GPU → CPU):")
    print(f"  {'模式':<30} {'时间 (ms)':<15} {'带宽 (MB/s)':<15} {'相对速度'}")
    print(f"  {'-'*75}")
    print(f"  {'启用对齐':<30} {transfer_time1*1000:>10.2f}      {total_size_mb/transfer_time1:>10.2f}      1.00x")
    speedup_transfer = transfer_time1 / transfer_time2
    print(f"  {'禁用对齐':<30} {transfer_time2*1000:>10.2f}      {total_size_mb/transfer_time2:>10.2f}      {speedup_transfer:.2f}x")
    
    print(f"\nCPU 写入性能:")
    print(f"  {'模式':<30} {'时间 (ms)':<15} {'带宽 (MB/s)':<15} {'相对速度'}")
    print(f"  {'-'*75}")
    print(f"  {'启用对齐':<30} {avg_write_time1*1000:>10.2f}      {write_bandwidth1:>10.2f}      1.00x")
    speedup_write = avg_write_time2 / avg_write_time1
    print(f"  {'禁用对齐':<30} {avg_write_time2*1000:>10.2f}      {write_bandwidth2:>10.2f}      {speedup_write:.2f}x")
    
    print(f"\nCPU 读取性能:")
    print(f"  {'模式':<30} {'时间 (ms)':<15} {'带宽 (MB/s)':<15} {'相对速度'}")
    print(f"  {'-'*75}")
    print(f"  {'启用对齐':<30} {avg_read_time1*1000:>10.2f}      {read_bandwidth1:>10.2f}      1.00x")
    speedup_read = avg_read_time2 / avg_read_time1
    print(f"  {'禁用对齐':<30} {avg_read_time2*1000:>10.2f}      {read_bandwidth2:>10.2f}      {speedup_read:.2f}x")
    
    print(f"\nCPU 随机访问性能:")
    print(f"  {'模式':<30} {'时间 (ms)':<15} {'相对速度'}")
    print(f"  {'-'*60}")
    print(f"  {'启用对齐':<30} {avg_random_time1*1000:>10.2f}      1.00x")
    speedup_random = avg_random_time2 / avg_random_time1
    print(f"  {'禁用对齐':<30} {avg_random_time2*1000:>10.2f}      {speedup_random:.2f}x")
    
    print(f"\n综合性能分析:")
    print(f"  启用对齐相比禁用对齐:")
    
    if speedup_write > 1.1:
        print(f"    写入: ⚡ 快 {(speedup_write-1)*100:.1f}%")
    elif speedup_write < 0.9:
        print(f"    写入: ⚠️  慢 {(1-speedup_write)*100:.1f}%")
    else:
        print(f"    写入: ≈ 相近 ({speedup_write:.2f}x)")
    
    if speedup_read > 1.1:
        print(f"    读取: ⚡ 快 {(speedup_read-1)*100:.1f}%")
    elif speedup_read < 0.9:
        print(f"    读取: ⚠️  慢 {(1-speedup_read)*100:.1f}%")
    else:
        print(f"    读取: ≈ 相近 ({speedup_read:.2f}x)")
    
    if speedup_random > 1.1:
        print(f"    随机访问: ⚡ 快 {(speedup_random-1)*100:.1f}%")
    elif speedup_random < 0.9:
        print(f"    随机访问: ⚠️  慢 {(1-speedup_random)*100:.1f}%")
    else:
        print(f"    随机访问: ≈ 相近 ({speedup_random:.2f}x)")
    
    avg_speedup = (speedup_write + speedup_read + speedup_random) / 3
    print(f"\n  平均性能提升: {avg_speedup:.2f}x ({'⚡ 启用对齐更快' if avg_speedup > 1.05 else '≈ 两者相近'})")
    
    print(f"\n结论:")
    if avg_speedup > 1.2:
        print(f"  ✓ 启用对齐带来显著性能提升 ({avg_speedup:.2f}x)")
        print(f"  ✓ 推荐在生产环境中启用对齐")
    elif avg_speedup > 1.05:
        print(f"  ✓ 启用对齐带来一定性能提升 ({avg_speedup:.2f}x)")
        print(f"  ✓ 建议启用对齐")
    else:
        print(f"  • 两种模式性能相近 ({avg_speedup:.2f}x)")
        print(f"  • 可根据内存需求选择")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    test_alignment_contiguity()
    test_alignment_address_details()
    test_alignment_performance()
    
    print("\n\n✓✓✓ 所有测试完成！")

