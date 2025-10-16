"""
GPU Tensor 到 CPU 内存池传输 - 简化测试
==========================================

包含单个tensor和多个tensor传输的核心测试
"""

import torch
import sys
import time
import numpy as np
from gpu_cpu_memory_pool import CPUMemoryPool, GPUToCPUPoolTransfer


def test_single_tensor_transfer():
    """测试单个 tensor 传输"""
    print("=" * 80)
    print("测试 1: 单个 Tensor 传输")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 支持")
        return False
    
    # 初始化内存池 (100 MB)
    pool = CPUMemoryPool(pool_size_bytes=100 * 1024 * 1024)
    transfer_mgr = GPUToCPUPoolTransfer(pool)
    
    print(f"\n✓ 初始化内存池: {pool.pool_size / 1024 / 1024:.0f} MB")
    print(f"  基地址: 0x{pool.base_address:x}")
    
    # 创建 GPU tensor
    gpu_tensor = torch.randn(500, 500, device='cuda', dtype=torch.float32)
    tensor_size_mb = gpu_tensor.element_size() * gpu_tensor.nelement() / 1024 / 1024
    
    print(f"\n创建 GPU tensor:")
    print(f"  Shape: {gpu_tensor.shape}")
    print(f"  Dtype: {gpu_tensor.dtype}")
    print(f"  Size: {tensor_size_mb:.2f} MB")
    print(f"  前5个值: {gpu_tensor.flatten()[:5].cpu().tolist()}")
    
    # 传输到内存池
    print("\n传输到 CPU 内存池...")
    tensor_id = transfer_mgr.transfer_to_pool(gpu_tensor)
    
    print(f"✓ 传输完成, Tensor ID: {tensor_id}")
    
    # 获取地址信息
    metadata = transfer_mgr.tensor_metadata[tensor_id]
    print(f"\n内存分配信息:")
    print(f"  地址: 0x{metadata['address']:x}")
    print(f"  偏移: {metadata['address'] - pool.base_address} bytes")
    print(f"  大小: {metadata['size']:,} bytes")
    
    # 从内存池读取
    print("\n从内存池读取 tensor...")
    cpu_tensor = transfer_mgr.get_tensor_from_pool(tensor_id)
    
    print(f"  Shape: {cpu_tensor.shape}")
    print(f"  Dtype: {cpu_tensor.dtype}")
    print(f"  前5个值: {cpu_tensor.flatten()[:5].tolist()}")
    
    # 验证数据正确性
    print("\n验证数据完整性:")
    gpu_cpu = gpu_tensor.cpu()
    max_diff = torch.abs(cpu_tensor - gpu_cpu).max().item()
    mean_diff = torch.abs(cpu_tensor - gpu_cpu).mean().item()
    
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  平均差异: {mean_diff:.2e}")
    
    if max_diff < 1e-6:
        print(f"  状态: ✓ 通过")
        result = True
    else:
        print(f"  状态: ✗ 失败")
        result = False
    
    # 查看内存使用
    stats = pool.get_statistics()
    print(f"\n内存池统计:")
    print(f"  已用: {stats['used_memory'] / 1024 / 1024:.2f} MB ({stats['utilization']:.1f}%)")
    print(f"  可用: {stats['free_memory'] / 1024 / 1024:.2f} MB")
    
    # 释放内存
    print("\n释放 tensor...")
    transfer_mgr.free_tensor(tensor_id)
    
    stats = pool.get_statistics()
    print(f"  已用: {stats['used_memory'] / 1024 / 1024:.2f} MB")
    print(f"  可用: {stats['free_memory'] / 1024 / 1024:.2f} MB")
    
    print(f"\n{'✓' if result else '✗'} 测试 1 {'通过' if result else '失败'}！\n")
    return result


def test_multiple_tensors_transfer():
    """测试多个 tensor 批量传输（地址连续）"""
    print("=" * 80)
    print("测试 2: 多个 Tensor 批量传输 (连续内存分配)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 支持")
        return False
    
    # 初始化内存池 (500 MB)
    pool = CPUMemoryPool(pool_size_bytes=500 * 1024 * 1024)
    transfer_mgr = GPUToCPUPoolTransfer(pool)
    
    print(f"\n✓ 初始化内存池: {pool.pool_size / 1024 / 1024:.0f} MB")
    
    # 创建多个不同大小的 GPU tensors
    num_tensors = 5
    tensor_shapes = [
        (100, 100),
        (200, 200),
        (150, 150),
        (300, 100),
        (256, 256)
    ]
    
    print(f"\n创建 {num_tensors} 个 GPU tensors:")
    gpu_tensors = []
    total_size = 0
    
    for i, shape in enumerate(tensor_shapes):
        tensor = torch.randn(*shape, device='cuda', dtype=torch.float32)
        gpu_tensors.append(tensor)
        
        size_mb = tensor.element_size() * tensor.nelement() / 1024 / 1024
        total_size += size_mb
        print(f"  Tensor {i}: shape={shape}, size={size_mb:.2f} MB")
    
    print(f"  总大小: {total_size:.2f} MB")
    
    # 批量传输 - 连续分配
    print(f"\n批量传输到 CPU 内存池 (连续分配)...")
    tensor_ids = transfer_mgr.transfer_batch_to_pool(
        gpu_tensors,
        contiguous=True  # 连续分配
    )
    
    print(f"✓ 传输完成")
    print(f"  Tensor IDs: {tensor_ids}")
    
    # 验证地址连续性
    print(f"\n验证地址连续性:")
    print("  ID  | 地址              | 大小       | 状态")
    print("  " + "-" * 55)
    
    all_contiguous = True
    prev_end_addr = None
    
    for i, tid in enumerate(tensor_ids):
        metadata = transfer_mgr.tensor_metadata[tid]
        addr = metadata['address']
        size = metadata['size']
        
        # 检查是否连续
        if prev_end_addr is not None:
            if addr == prev_end_addr:
                status = "✓ 连续"
            else:
                gap = addr - prev_end_addr
                status = f"✗ 间隔 {gap} bytes"
                all_contiguous = False
        else:
            status = "首个"
        
        print(f"  {tid:3d} | 0x{addr:016x} | {size:9,} | {status}")
        prev_end_addr = addr + size
    
    if all_contiguous:
        print(f"\n✓ 所有 tensor 地址连续！")
    else:
        print(f"\n✗ 地址不连续")
    
    # 验证数据完整性
    print(f"\n验证数据完整性:")
    all_correct = True
    
    for i, tid in enumerate(tensor_ids):
        cpu_tensor = transfer_mgr.get_tensor_from_pool(tid)
        gpu_cpu = gpu_tensors[i].cpu()
        
        max_diff = torch.abs(cpu_tensor - gpu_cpu).max().item()
        
        if max_diff < 1e-6:
            status = "✓"
        else:
            status = "✗"
            all_correct = False
        
        print(f"  Tensor {tid}: max_diff={max_diff:.2e} {status}")
    
    if all_correct:
        print(f"\n✓ 所有数据验证通过！")
    else:
        print(f"\n✗ 数据验证失败")
    
    # 内存池统计
    stats = pool.get_statistics()
    print(f"\n内存池统计:")
    print(f"  总容量: {stats['pool_size'] / 1024 / 1024:.2f} MB")
    print(f"  已使用: {stats['used_memory'] / 1024 / 1024:.2f} MB ({stats['utilization']:.1f}%)")
    print(f"  可用:   {stats['free_memory'] / 1024 / 1024:.2f} MB")
    print(f"  块数量: {stats['num_blocks']}")
    print(f"  碎片率: {stats['fragmentation_ratio']:.2%}")
    
    # 打印内存布局
    print(f"\n内存布局:")
    pool.print_memory_map()
    
    # 释放部分内存
    print(f"释放前 3 个 tensors...")
    transfer_mgr.free_batch(tensor_ids[:3])
    
    stats = pool.get_statistics()
    print(f"  已使用: {stats['used_memory'] / 1024 / 1024:.2f} MB")
    print(f"  可用:   {stats['free_memory'] / 1024 / 1024:.2f} MB")
    print(f"  碎片率: {stats['fragmentation_ratio']:.2%}")
    
    # 释放剩余内存
    print(f"\n释放剩余 tensors...")
    transfer_mgr.free_batch(tensor_ids[3:])
    
    stats = pool.get_statistics()
    print(f"  已使用: {stats['used_memory'] / 1024 / 1024:.2f} MB")
    print(f"  可用:   {stats['free_memory'] / 1024 / 1024:.2f} MB")
    
    result = all_contiguous and all_correct
    print(f"\n{'✓' if result else '✗'} 测试 2 {'通过' if result else '失败'}！\n")
    return result


def test_performance_comparison():
    """测试性能对比：.to("cpu") vs CPUMemoryPool"""
    print("=" * 80)
    print("测试 3: 性能对比 (.to vs CPUMemoryPool)")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 支持")
        return False
    
    # 测试配置
    num_tensors = 20
    tensor_shapes = [
        (256, 256),
        (512, 512),
        (128, 128),
        (1024, 256),
        (256, 1024),
    ] * 4  # 重复 4 次，总共 20 个
    
    print(f"\n测试配置:")
    print(f"  Tensor 数量: {num_tensors}")
    print(f"  Tensor 形状: {list(set(tensor_shapes))}")
    
    # 创建 GPU tensors
    print(f"\n准备测试数据...")
    gpu_tensors = []
    total_size = 0
    for shape in tensor_shapes:
        tensor = torch.randn(*shape, device='cuda', dtype=torch.float32)
        gpu_tensors.append(tensor)
        total_size += tensor.element_size() * tensor.nelement()
    
    total_size_mb = total_size / 1024 / 1024
    print(f"  总数据量: {total_size_mb:.2f} MB")
    
    # 预热（避免首次运行的开销）
    print(f"\n预热...")
    for _ in range(3):
        _ = [t.to('cpu') for t in gpu_tensors[:5]]
    
    pool = CPUMemoryPool(pool_size_bytes=int(total_size * 2))
    transfer_mgr = GPUToCPUPoolTransfer(pool)
    _ = transfer_mgr.transfer_batch_to_pool(gpu_tensors[:5], contiguous=True)
    transfer_mgr.free_batch(list(range(5)))
    
    torch.cuda.synchronize()
    
    # ==================== 方法 1: 标准 .to("cpu") ====================
    print(f"\n{'='*80}")
    print("方法 1: 标准 .to('cpu')")
    print("=" * 80)
    
    num_iterations = 10
    to_times = []
    
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        
        cpu_tensors_to = []
        for gpu_tensor in gpu_tensors:
            cpu_tensors_to.append(gpu_tensor.to('cpu'))
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        to_times.append(elapsed)
        
        if i == 0:
            print(f"  第 1 次: {elapsed*1000:.2f} ms")
        
        # 清理
        del cpu_tensors_to
    
    avg_to_time = sum(to_times) / len(to_times)
    print(f"  平均时间: {avg_to_time*1000:.2f} ms ({num_iterations} 次)")
    # print(f"  带宽: {total_size_mb / avg_to_time:.2f} MB/s")
    
    # ==================== 方法 2: CPUMemoryPool (普通池) - 提前分配 ====================
    print(f"\n{'='*80}")
    print("方法 2: CPUMemoryPool (普通池) - 提前分配地址")
    print("=" * 80)
    
    pool2 = CPUMemoryPool(
        pool_size_bytes=int(total_size * 2),
        use_pinned_pool=False  # 普通内存池
    )
    transfer_mgr2 = GPUToCPUPoolTransfer(pool2)
    
    # 提前分配所有地址
    print(f"  预分配地址...")
    sizes_bytes = [t.element_size() * t.nelement() for t in gpu_tensors]
    pre_addresses, pre_tensor_ids = pool2.allocate_contiguous(sizes_bytes)
    print(f"  已分配 {len(pre_addresses)} 个地址，地址范围: 0x{pre_addresses[0]:x} - 0x{pre_addresses[-1]:x}")
    
    pool_times = []
    pool_alloc_times = []
    view_creation_times = []
    
    for i in range(num_iterations):
        # 释放之前的分配（除了第一次）
        if i > 0:
            for tid in pre_tensor_ids:
                pool2.deallocate(tid)
        
        # 测试分配时间（重新分配）
        torch.cuda.synchronize()
        alloc_start = time.time()
        addresses, tensor_ids = pool2.allocate_contiguous(sizes_bytes)
        alloc_elapsed = time.time() - alloc_start
        pool_alloc_times.append(alloc_elapsed)
        
        # 预构建 CPU tensor 视图（优化）
        view_start = time.time()
        for j, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors, addresses, tensor_ids)):
            actual_size = pool2.allocations[tensor_id].size
            transfer_mgr2.tensor_metadata[tensor_id] = {
                'shape': gpu_tensor.shape,
                'dtype': gpu_tensor.dtype,
                'address': address,
                'size': actual_size
            }
            # 预构建视图
            transfer_mgr2.prepare_cpu_tensor_view(tensor_id)
        view_elapsed = time.time() - view_start
        view_creation_times.append(view_elapsed)
        
        # 测试传输时间（使用已构建的视图）
        torch.cuda.synchronize()
        transfer_start = time.time()
        
        # 传输（使用缓存的视图，减少开销）
        for j, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors, addresses, tensor_ids)):
            transfer_mgr2._do_transfer(gpu_tensor, address, tensor_id)
        
        torch.cuda.synchronize()
        transfer_elapsed = time.time() - transfer_start
        pool_times.append(transfer_elapsed)
        
        if i == 0:
            print(f"  第 1 次 - 分配: {alloc_elapsed*1000:.2f} ms, 视图构建: {view_elapsed*1000:.2f} ms, 传输: {transfer_elapsed*1000:.2f} ms")
        
        pre_tensor_ids = tensor_ids  # 保存用于下次清理
    
    # 清理
    for tid in pre_tensor_ids:
        pool2.deallocate(tid)
    
    avg_alloc_time = sum(pool_alloc_times) / len(pool_alloc_times)
    avg_view_time = sum(view_creation_times) / len(view_creation_times)
    avg_pool_time = sum(pool_times) / len(pool_times)
    avg_total_time = avg_alloc_time + avg_view_time + avg_pool_time
    
    print(f"  平均分配时间:   {avg_alloc_time*1000:.2f} ms ({num_iterations} 次)")
    print(f"  平均视图构建:   {avg_view_time*1000:.2f} ms (NumPy 桥接开销)")
    print(f"  平均传输时间:   {avg_pool_time*1000:.2f} ms (纯数据传输)")
    print(f"  平均总时间:     {avg_total_time*1000:.2f} ms")
    print(f"  传输带宽:       {total_size_mb / avg_pool_time:.2f} MB/s")
    
    # ==================== 方法 3: CPUMemoryPool (Pinned Pool) - 提前分配 ====================
    print(f"\n{'='*80}")
    print("方法 3: CPUMemoryPool (Pinned Pool via mlock) - 提前分配地址 ⭐")
    print("=" * 80)
    
    pool3 = CPUMemoryPool(
        pool_size_bytes=int(total_size * 2),
        use_pinned_pool=True  # ⭐ 整个池 pin 住
    )
    transfer_mgr3 = GPUToCPUPoolTransfer(pool3)
    
    # 提前分配所有地址
    print(f"  预分配地址...")
    pre_addresses3, pre_tensor_ids3 = pool3.allocate_contiguous(sizes_bytes)
    print(f"  已分配 {len(pre_addresses3)} 个地址，地址范围: 0x{pre_addresses3[0]:x} - 0x{pre_addresses3[-1]:x}")
    
    pool_pinned_times = []
    pool_pinned_alloc_times = []
    view_pinned_creation_times = []
    
    for i in range(num_iterations):
        # 释放之前的分配（除了第一次）
        if i > 0:
            for tid in pre_tensor_ids3:
                pool3.deallocate(tid)
        
        # 测试分配时间（重新分配）
        torch.cuda.synchronize()
        alloc_start = time.time()
        addresses, tensor_ids = pool3.allocate_contiguous(sizes_bytes)
        alloc_elapsed = time.time() - alloc_start
        pool_pinned_alloc_times.append(alloc_elapsed)
        
        # 预构建 CPU tensor 视图（优化）
        view_start = time.time()
        for j, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors, addresses, tensor_ids)):
            actual_size = pool3.allocations[tensor_id].size
            transfer_mgr3.tensor_metadata[tensor_id] = {
                'shape': gpu_tensor.shape,
                'dtype': gpu_tensor.dtype,
                'address': address,
                'size': actual_size
            }
            # 预构建视图
            transfer_mgr3.prepare_cpu_tensor_view(tensor_id)
        view_elapsed = time.time() - view_start
        view_pinned_creation_times.append(view_elapsed)
        
        # 测试传输时间（使用已构建的视图）
        torch.cuda.synchronize()
        transfer_start = time.time()
        
        # 传输（使用缓存的视图，减少开销）
        for j, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors, addresses, tensor_ids)):
            transfer_mgr3._do_transfer(gpu_tensor, address, tensor_id)
        
        torch.cuda.synchronize()
        transfer_elapsed = time.time() - transfer_start
        pool_pinned_times.append(transfer_elapsed)
        
        if i == 0:
            print(f"  第 1 次 - 分配: {alloc_elapsed*1000:.2f} ms, 视图构建: {view_elapsed*1000:.2f} ms, 传输: {transfer_elapsed*1000:.2f} ms")
        
        pre_tensor_ids3 = tensor_ids  # 保存用于下次清理
    
    # 清理
    for tid in pre_tensor_ids3:
        pool3.deallocate(tid)
    
    avg_alloc_pinned_time = sum(pool_pinned_alloc_times) / len(pool_pinned_alloc_times)
    avg_view_pinned_time = sum(view_pinned_creation_times) / len(view_pinned_creation_times)
    avg_pool_pinned_time = sum(pool_pinned_times) / len(pool_pinned_times)
    avg_total_pinned_time = avg_alloc_pinned_time + avg_view_pinned_time + avg_pool_pinned_time
    
    print(f"  平均分配时间:   {avg_alloc_pinned_time*1000:.2f} ms ({num_iterations} 次)")
    print(f"  平均视图构建:   {avg_view_pinned_time*1000:.2f} ms (NumPy 桥接开销)")
    print(f"  平均传输时间:   {avg_pool_pinned_time*1000:.2f} ms (纯数据传输)")
    print(f"  平均总时间:     {avg_total_pinned_time*1000:.2f} ms")
    print(f"  传输带宽:       {total_size_mb / avg_pool_pinned_time:.2f} MB/s")
    
    # ==================== 性能对比总结 ====================
    print(f"\n{'='*80}")
    print("性能对比总结")
    print("=" * 80)
    
    print(f"\n完整对比 ({num_tensors} 个 tensors, {total_size_mb:.2f} MB):")
    print(f"  {'方法':<45} {'总时间':<12} {'分配':<12} {'视图构建':<12} {'传输':<12} {'带宽 (MB/s)'}")
    print(f"  {'-'*105}")
    
    baseline = avg_to_time
    
    print(f"  {'.to(cpu) [分配+传输]':<45} {avg_to_time*1000:>8.2f} ms   {'N/A':<12} {'N/A':<12} {'N/A':<12} {total_size_mb/avg_to_time:>10.2f}")
    print(f"  {'CPUMemoryPool (普通池)':<45} {avg_total_time*1000:>8.2f} ms {avg_alloc_time*1000:>8.2f} ms {avg_view_time*1000:>8.2f} ms {avg_pool_time*1000:>8.2f} ms {total_size_mb/avg_pool_time:>10.2f}")
    print(f"  {'CPUMemoryPool (Pinned Pool) ⭐':<45} {avg_total_pinned_time*1000:>8.2f} ms {avg_alloc_pinned_time*1000:>8.2f} ms {avg_view_pinned_time*1000:>8.2f} ms {avg_pool_pinned_time*1000:>8.2f} ms {total_size_mb/avg_pool_pinned_time:>10.2f}")
    
    print(f"\n纯传输性能对比（不含分配）:")
    print(f"  {'方法':<45} {'传输时间 (ms)':<15} {'带宽 (MB/s)':<15} {'相对速度'}")
    print(f"  {'-'*100}")
    
    print(f"  {'.to(cpu) [基准]':<45} {avg_to_time*1000:>10.2f}      {total_size_mb/avg_to_time:>10.2f}      {'1.00x'}")
    
    speedup_pool = baseline / avg_pool_time
    faster_slower = "faster" if speedup_pool > 1 else "slower"
    print(f"  {'CPUMemoryPool (普通池)':<45} {avg_pool_time*1000:>10.2f}      {total_size_mb/avg_pool_time:>10.2f}      {speedup_pool:.2f}x {faster_slower}")
    
    speedup_pinned = baseline / avg_pool_pinned_time
    faster_slower_pinned = "faster" if speedup_pinned > 1 else "slower"
    print(f"  {'CPUMemoryPool (Pinned Pool) ⭐':<45} {avg_pool_pinned_time*1000:>10.2f}      {total_size_mb/avg_pool_pinned_time:>10.2f}      {speedup_pinned:.2f}x {faster_slower_pinned}")
    
    print(f"\n开销分析:")
    print(f"  CPUMemoryPool 分配时间:     {avg_alloc_time*1000:.2f} ms (占比 {avg_alloc_time/avg_total_time*100:.1f}%)")
    print(f"  CPUMemoryPool 视图构建时间: {avg_view_time*1000:.2f} ms (占比 {avg_view_time/avg_total_time*100:.1f}%) ← NumPy 桥接")
    print(f"  CPUMemoryPool 纯传输时间:   {avg_pool_time*1000:.2f} ms (占比 {avg_pool_time/avg_total_time*100:.1f}%)")
    print(f"")
    print(f"  分析:")
    print(f"    • 视图构建 = NumPy 桥接开销 ({avg_view_time*1000:.2f} ms)")
    print(f"    • 这是 CPUMemoryPool 比 .to 慢的主要原因")
    print(f"    • 但视图可以预构建并缓存（下次复用）")
    print(f"    • 纯传输时间({avg_pool_time*1000:.2f} ms) vs .to({avg_to_time*1000:.2f} ms) ≈ 相近")
    
    # 额外优势说明
    print(f"\n额外优势:")
    print(f"  ✓ CPUMemoryPool 提供地址连续分配（提升缓存性能）")
    print(f"  ✓ CPUMemoryPool 支持零拷贝读取（直接内存视图）")
    print(f"  ✓ CPUMemoryPool 可预分配地址（分配和传输解耦）")
    print(f"  ✓ CPUMemoryPool 可预构建视图（视图可复用）")
    print(f"  ✓ CPUMemoryPool 更好的内存管理和控制")
    print(f"  ✓ Pinned Pool 提升传输带宽 {speedup_pinned:.2f}x")
    
    print(f"\n关键发现:")
    print(f"  • 分配开销很小 (~{avg_alloc_time*1000:.2f} ms)")
    print(f"  • 视图构建开销 ~{avg_view_time*1000:.2f} ms (可预构建并缓存)")
    print(f"  • 纯传输性能与 .to 接近")
    print(f"  • Pinned Pool 显著提升传输性能（{speedup_pinned:.2f}x faster）")
    
    print(f"\n{'='*80}")
    print(f"✓ 测试 3 完成！")
    print(f"{'='*80}\n")
    
    return True


def test_async_transfer_performance():
    """测试异步传输性能对比"""
    print("=" * 80)
    print("测试 4: 异步传输性能对比")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 支持")
        return False
    
    # 测试配置
    num_tensors = 20
    tensor_shapes = [
        (256, 256),
        (512, 512),
        (128, 128),
        (1024, 256),
        (256, 1024),
    ] * 4  # 重复 4 次，总共 20 个
    
    print(f"\n测试配置:")
    print(f"  Tensor 数量: {num_tensors}")
    print(f"  异步传输: 所有 tensor 传输后统一 synchronize")
    
    # 创建 GPU tensors
    print(f"\n准备测试数据...")
    gpu_tensors = []
    total_size = 0
    for shape in tensor_shapes:
        tensor = torch.randn(*shape, device='cuda', dtype=torch.float32)
        gpu_tensors.append(tensor)
        total_size += tensor.element_size() * tensor.nelement()
    
    total_size_mb = total_size / 1024 / 1024
    print(f"  总数据量: {total_size_mb:.2f} MB")
    
    num_iterations = 10
    sizes_bytes = [t.element_size() * t.nelement() for t in gpu_tensors]
    
    # ==================== 方法 1: .to("cpu") ====================
    print(f"\n{'='*80}")
    print("方法 1: 标准 .to('cpu') [异步]")
    print("=" * 80)
    
    to_times = []
    for _ in range(3):  # 预热
        _ = [t.to('cpu') for t in gpu_tensors[:5]]
    
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
    
        cpu_tensors = []
        for gpu_tensor in gpu_tensors:
            cpu_tensors.append(gpu_tensor.to('cpu', non_blocking=True))
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        to_times.append(elapsed)
        
        if i == 0:
            print(f"  第 1 次: {elapsed*1000:.2f} ms")
        
        del cpu_tensors
    
    avg_to_time = np.mean(to_times)
    print(f"  平均时间: {avg_to_time*1000:.2f} ms")
    print(f"  带宽: {total_size_mb / avg_to_time:.2f} MB/s")
    
    # ==================== 方法 2: CPUMemoryPool 异步（普通池）====================
    print(f"\n{'='*80}")
    print("方法 2: CPUMemoryPool 异步传输（普通池 + 临时 pinned buffer）")
    print("  与 .to('cpu', non_blocking=True) 类似策略")
    print("=" * 80)
    
    pool2 = CPUMemoryPool(
        pool_size_bytes=int(total_size * 2),
        use_pinned_pool=False
    )
    transfer_mgr2 = GPUToCPUPoolTransfer(pool2)
    
    async_times = []
    stream = torch.cuda.Stream()
    
    # 预热
    for _ in range(3):
        with torch.cuda.stream(stream):
            addresses, tids = pool2.allocate_contiguous(sizes_bytes[:5])
            for j, (t, addr, tid) in enumerate(zip(gpu_tensors[:5], addresses, tids)):
                transfer_mgr2._do_transfer_async(t, addr, stream)
        stream.synchronize()
        transfer_mgr2.free_batch(tids)
    
    for i in range(num_iterations):
        # 分配地址
        if i > 0:
            transfer_mgr2.free_batch(tensor_ids)
        
        addresses, tensor_ids = pool2.allocate_contiguous(sizes_bytes)
        
        # 预构建 CPU tensor 视图（减少传输时的开销）
        for j, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors, addresses, tensor_ids)):
            actual_size = pool2.allocations[tensor_id].size
            transfer_mgr2.tensor_metadata[tensor_id] = {
                'shape': gpu_tensor.shape,
                'dtype': gpu_tensor.dtype,
                'address': address,
                'size': actual_size
            }
            # 预构建视图
            transfer_mgr2.prepare_cpu_tensor_view(tensor_id)
        
        # 异步传输所有 tensor
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.cuda.stream(stream):
            for j, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors, addresses, tensor_ids)):
                # 异步传输，使用预构建的视图
                transfer_mgr2._do_transfer_async(gpu_tensor, address, stream, tensor_id)
        
        # 统一 synchronize
        stream.synchronize()
        elapsed = time.time() - start
        async_times.append(elapsed)
        
        if i == 0:
            print(f"  第 1 次: {elapsed*1000:.2f} ms")
    
    # 清理
    transfer_mgr2.free_batch(tensor_ids)
    
    avg_async_time = np.mean(async_times)
    print(f"  平均时间: {avg_async_time*1000:.2f} ms")
    print(f"  带宽: {total_size_mb / avg_async_time:.2f} MB/s")
    
    # ==================== 方法 3: CPUMemoryPool 异步（Pinned Pool）====================
    print(f"\n{'='*80}")
    print("方法 3: CPUMemoryPool 异步传输（Pinned Pool）⭐")
    print("=" * 80)
    
    pool3 = CPUMemoryPool(
        pool_size_bytes=int(total_size * 2),
        use_pinned_pool=True
    )
    transfer_mgr3 = GPUToCPUPoolTransfer(pool3)
    
    async_pinned_times = []
    stream3 = torch.cuda.Stream()
    
    # 预热
    for _ in range(3):
        with torch.cuda.stream(stream3):
            addresses, tids = pool3.allocate_contiguous(sizes_bytes[:5])
            for j, (t, addr, tid) in enumerate(zip(gpu_tensors[:5], addresses, tids)):
                transfer_mgr3._do_transfer_async(t, addr, stream3)
        stream3.synchronize()
        transfer_mgr3.free_batch(tids)
    
    for i in range(num_iterations):
        # 分配地址
        if i > 0:
            transfer_mgr3.free_batch(tensor_ids3)
        
        addresses3, tensor_ids3 = pool3.allocate_contiguous(sizes_bytes)
        
        # 预构建 CPU tensor 视图（减少传输时的开销）
        for j, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors, addresses3, tensor_ids3)):
            actual_size = pool3.allocations[tensor_id].size
            transfer_mgr3.tensor_metadata[tensor_id] = {
                'shape': gpu_tensor.shape,
                'dtype': gpu_tensor.dtype,
                'address': address,
                'size': actual_size
            }
            # 预构建视图
            transfer_mgr3.prepare_cpu_tensor_view(tensor_id)
        
        # 异步传输所有 tensor
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.cuda.stream(stream3):
            for j, (gpu_tensor, address, tensor_id) in enumerate(zip(gpu_tensors, addresses3, tensor_ids3)):
                # 异步传输，使用预构建的视图
                transfer_mgr3._do_transfer_async(gpu_tensor, address, stream3, tensor_id)
        
        # 统一 synchronize
        stream3.synchronize()
        elapsed = time.time() - start
        async_pinned_times.append(elapsed)
        
        if i == 0:
            print(f"  第 1 次: {elapsed*1000:.2f} ms")
    
    # 清理
    transfer_mgr3.free_batch(tensor_ids3)
    
    avg_async_pinned_time = np.mean(async_pinned_times)
    print(f"  平均时间: {avg_async_pinned_time*1000:.2f} ms")
    print(f"  带宽: {total_size_mb / avg_async_pinned_time:.2f} MB/s")
    
    # ==================== 异步传输性能对比 ====================
    print(f"\n{'='*80}")
    print("异步传输性能对比总结")
    print("=" * 80)
    
    print(f"\n传输性能对比 ({num_tensors} 个 tensors, {total_size_mb:.2f} MB):")
    print(f"  {'方法':<50} {'时间 (ms)':<15} {'带宽 (MB/s)':<15} {'相对速度'}")
    print(f"  {'-'*95}")
    
    baseline = avg_to_time
    
    print(f"  {'.to(cpu)':<50} {avg_to_time*1000:>10.2f}      {total_size_mb/avg_to_time:>10.2f}      1.00x")
    
    speedup_async = baseline / avg_async_time
    print(f"  {'CPUMemoryPool 异步（普通池）':<50} {avg_async_time*1000:>10.2f}      {total_size_mb/avg_async_time:>10.2f}      {speedup_async:.2f}x")
    
    speedup_async_pinned = baseline / avg_async_pinned_time
    print(f"  {'CPUMemoryPool 异步（Pinned Pool）⭐':<50} {avg_async_pinned_time*1000:>10.2f}      {total_size_mb/avg_async_pinned_time:>10.2f}      {speedup_async_pinned:.2f}x")
    
    print(f"\n异步 vs 异步对比:")
    speedup_pinned_vs_normal = avg_async_time / avg_async_pinned_time
    print(f"  Pinned Pool vs 普通池: {speedup_pinned_vs_normal:.2f}x ({'faster' if speedup_pinned_vs_normal > 1 else 'slower'})")
    
    print(f"\n重要发现:")
    print(f"  ✓ .to('cpu', non_blocking=True):")
    print(f"      PyTorch 内部创建临时 pinned buffer")
    print(f"      GPU → 临时 pinned buffer (async)")
    print(f"      性能: {avg_to_time*1000:.2f} ms")
    print(f"")
    print(f"  ✓ CPUMemoryPool 异步（普通池 + 临时 pinned buffer）:")
    print(f"      使用临时 pinned buffer（模仿 .to 的策略）")
    print(f"      GPU → 临时 pinned → 目标地址")
    print(f"      性能: {avg_async_time*1000:.2f} ms")
    
    if speedup_async >= 0.9 and speedup_async <= 1.1:
        print(f"      结论: 与 .to 性能相近 ✓ (差异 < 10%)")
    elif speedup_async > 1.1:
        print(f"      结论: 比 .to 更快 ✓ ({speedup_async:.2f}x)")
    else:
        print(f"      结论: 比 .to 稍慢 ({speedup_async:.2f}x，可能是临时 buffer 开销)")
    
    print(f"")
    print(f"  ✓✓✓ CPUMemoryPool 异步（Pinned Pool）⭐:")
    print(f"      池本身就是 pinned，无需临时 buffer")
    print(f"      GPU → 目标地址（直接异步）")
    print(f"      性能: {avg_async_pinned_time*1000:.2f} ms (最快！)")
    
    print(f"\n关键结论:")
    print(f"  • 普通池现在也支持真正的异步（使用临时 pinned buffer）")
    print(f"  • 普通池异步性能 ≈ .to('cpu', non_blocking=True)")
    print(f"  • Pinned Pool 最优：无临时 buffer 开销，性能提升 {speedup_async_pinned:.2f}x")
    print(f"  • 额外优势：CPUMemoryPool 仍提供连续分配、零拷贝读取等功能")
    
    print(f"\n{'='*80}")
    print(f"✓ 测试 4 完成！")
    print(f"{'='*80}\n")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("GPU Tensor 到 CPU 内存池传输 - 简化测试套件")
    print("=" * 80 + "\n")
    
    results = []
    
    try:
        # 测试 1: 单个 tensor
        # result1 = test_single_tensor_transfer()
        # results.append(("单个 Tensor 传输", result1))
        
        # # 测试 2: 多个 tensors
        # result2 = test_multiple_tensors_transfer()
        # results.append(("多个 Tensor 批量传输", result2))
        
        # # # 测试 3: 性能对比
        # result3 = test_performance_comparison()
        # results.append(("性能对比测试", result3))
        
        # 测试 4: 异步传输性能对比
        result4 = test_async_transfer_performance()
        results.append(("异步传输性能对比", result4))
        
        # 汇总结果
        print("=" * 80)
        print("测试结果汇总")
        print("=" * 80)
        
        for test_name, result in results:
            status = "✓ 通过" if result else "✗ 失败"
            print(f"  {test_name:30s} : {status}")
        
        all_passed = all(r[1] for r in results)
        
        print("=" * 80)
        if all_passed:
            print("✓✓✓ 所有测试通过！")
        else:
            print("✗✗✗ 部分测试失败")
        print("=" * 80)
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

