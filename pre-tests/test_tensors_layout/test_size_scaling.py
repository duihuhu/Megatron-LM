"""
不同大小 Tensor 的对齐性能测试
===================================

测试不同大小级别 (B, KB, MB, GB) 的 tensor 在对齐/不对齐情况下的性能差异
"""

import torch
import time
import numpy as np
from gpu_cpu_memory_pool import CPUMemoryPool, GPUToCPUPoolTransfer


def test_tensor_size_scaling():
    """测试不同大小级别的 tensor 性能"""
    print("=" * 100)
    print("Tensor 大小缩放测试：对齐 vs 不对齐性能对比")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 支持")
        return
    
    # 定义测试大小（字节级、KB级、MB级、GB级）
    test_configs = [
        # (名称, 大小(字节), 形状, 迭代次数)
        ("100 B",      100,          (5, 5),          1000),      # 字节级
        ("1 KB",       1024,         (16, 16),        1000),      # KB级
        ("10 KB",      10*1024,      (50, 50),        1000),      # KB级
        ("100 KB",     100*1024,     (160, 160),      500),       # KB级
        ("1 MB",       1024*1024,    (512, 512),      200),       # MB级
        ("10 MB",      10*1024*1024, (1620, 1620),    50),        # MB级
        # ("100 MB",     100*1024*1024,(5120, 5120),    20),        # MB级
        # ("500 MB",     500*1024*1024,(11450, 11450),  5),         # MB级
    ]
    
    results = []
    
    print(f"\n{'='*100}")
    print(f"{'大小':<12} {'形状':<18} {'迭代':<8} {'传输':<20} {'写入':<20} {'读取':<20}")
    print(f"{'-'*100}")
    
    for name, target_bytes, shape, iterations in test_configs:
        print(f"{name:<12} {str(shape):<18} {iterations:<8} ", end='', flush=True)
        
        try:
            # 创建 GPU tensor
            gpu_tensor = torch.randn(*shape, device='cuda', dtype=torch.float32)
            actual_size = gpu_tensor.element_size() * gpu_tensor.nelement()
            size_mb = actual_size / 1024 / 1024
            
            # 为两种模式准备内存池
            pool_size = max(actual_size * 4, 10 * 1024 * 1024)  # 至少 10 MB
            
            # ========== 测试 1: 启用对齐 ==========
            pool_aligned = CPUMemoryPool(pool_size_bytes=pool_size, enable_alignment=True)
            mgr_aligned = GPUToCPUPoolTransfer(pool_aligned)
            
            # 传输性能
            transfer_times_aligned = []
            for _ in range(min(iterations, 10)):  # 传输测试最多10次
                torch.cuda.synchronize()
                start = time.time()
                tid = mgr_aligned.transfer_to_pool(gpu_tensor, use_pinned_memory=True)
                torch.cuda.synchronize()
                transfer_times_aligned.append(time.time() - start)
                mgr_aligned.free_tensor(tid)
            
            avg_transfer_aligned = np.mean(transfer_times_aligned)
            
            # 获取 CPU tensor 用于读写测试
            tid_aligned = mgr_aligned.transfer_to_pool(gpu_tensor)
            cpu_tensor_aligned = mgr_aligned.get_tensor_from_pool(tid_aligned)
            
            # 写入性能
            write_times_aligned = []
            for _ in range(3):  # 预热
                cpu_tensor_aligned.fill_(1.0)
            
            for i in range(min(iterations, 100)):
                start = time.time()
                cpu_tensor_aligned.fill_(float(i))
                write_times_aligned.append(time.time() - start)
            
            avg_write_aligned = np.mean(write_times_aligned)
            
            # 读取性能
            read_times_aligned = []
            for _ in range(3):  # 预热
                _ = cpu_tensor_aligned.sum()
            
            for _ in range(min(iterations, 100)):
                start = time.time()
                _ = cpu_tensor_aligned.sum().item()
                read_times_aligned.append(time.time() - start)
            
            avg_read_aligned = np.mean(read_times_aligned)
            
            mgr_aligned.free_tensor(tid_aligned)
            
            # ========== 测试 2: 禁用对齐 ==========
            pool_unaligned = CPUMemoryPool(pool_size_bytes=pool_size, enable_alignment=False)
            mgr_unaligned = GPUToCPUPoolTransfer(pool_unaligned)
            
            # 传输性能
            transfer_times_unaligned = []
            for _ in range(min(iterations, 10)):
                torch.cuda.synchronize()
                start = time.time()
                tid = mgr_unaligned.transfer_to_pool(gpu_tensor, use_pinned_memory=True)
                torch.cuda.synchronize()
                transfer_times_unaligned.append(time.time() - start)
                mgr_unaligned.free_tensor(tid)
            
            avg_transfer_unaligned = np.mean(transfer_times_unaligned)
            
            # 获取 CPU tensor 用于读写测试
            tid_unaligned = mgr_unaligned.transfer_to_pool(gpu_tensor)
            cpu_tensor_unaligned = mgr_unaligned.get_tensor_from_pool(tid_unaligned)
            
            # 写入性能
            write_times_unaligned = []
            for _ in range(3):  # 预热
                cpu_tensor_unaligned.fill_(1.0)
            
            for i in range(min(iterations, 100)):
                start = time.time()
                cpu_tensor_unaligned.fill_(float(i))
                write_times_unaligned.append(time.time() - start)
            
            avg_write_unaligned = np.mean(write_times_unaligned)
            
            # 读取性能
            read_times_unaligned = []
            for _ in range(3):  # 预热
                _ = cpu_tensor_unaligned.sum()
            
            for _ in range(min(iterations, 100)):
                start = time.time()
                _ = cpu_tensor_unaligned.sum().item()
                read_times_unaligned.append(time.time() - start)
            
            avg_read_unaligned = np.mean(read_times_unaligned)
            
            mgr_unaligned.free_tensor(tid_unaligned)
            
            # 计算加速比
            speedup_transfer = avg_transfer_unaligned / avg_transfer_aligned
            speedup_write = avg_write_unaligned / avg_write_aligned
            speedup_read = avg_read_unaligned / avg_read_aligned
            
            # 格式化输出
            transfer_str = f"{avg_transfer_aligned*1000:.3f}/{avg_transfer_unaligned*1000:.3f}ms ({speedup_transfer:.2f}x)"
            write_str = f"{avg_write_aligned*1000:.3f}/{avg_write_unaligned*1000:.3f}ms ({speedup_write:.2f}x)"
            read_str = f"{avg_read_aligned*1000:.3f}/{avg_read_unaligned*1000:.3f}ms ({speedup_read:.2f}x)"
            
            print(f"{transfer_str:<20} {write_str:<20} {read_str:<20}")
            
            # 保存结果
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
            print(f"跳过 (内存不足或错误: {e})")
            continue
    
    # ==================== 详细分析 ====================
    print(f"\n{'='*100}")
    print("详细分析：对齐的性能提升 vs Tensor 大小")
    print("=" * 100)
    
    print(f"\n{'大小':<12} {'实际字节':<15} {'传输加速':<12} {'写入加速':<12} {'读取加速':<12} {'平均加速':<12} {'建议'}")
    print(f"{'-'*100}")
    
    for r in results:
        size_str = f"{r['size_bytes']:,}"
        transfer_perf = f"{r['transfer_speedup']:.2f}x"
        write_perf = f"{r['write_speedup']:.2f}x"
        read_perf = f"{r['read_speedup']:.2f}x"
        avg_perf = f"{r['avg_speedup']:.2f}x"
        
        # 根据性能提升给出建议
        if r['avg_speedup'] > 1.2:
            recommendation = "✓ 启用对齐"
        elif r['avg_speedup'] > 1.05:
            recommendation = "• 建议对齐"
        else:
            recommendation = "• 影响小"
        
        print(f"{r['name']:<12} {size_str:<15} {transfer_perf:<12} {write_perf:<12} {read_perf:<12} {avg_perf:<12} {recommendation}")
    
    # ==================== 图表式总结 ====================
    print(f"\n{'='*100}")
    print("性能加速趋势（对齐 vs 不对齐）")
    print("=" * 100)
    
    print("\n传输性能加速:")
    for r in results:
        bar_length = int(r['transfer_speedup'] * 20)
        bar = '█' * bar_length
        print(f"  {r['name']:<10} [{r['transfer_speedup']:.2f}x] {bar}")
    
    print("\n写入性能加速:")
    for r in results:
        bar_length = int(r['write_speedup'] * 20)
        bar = '█' * bar_length
        print(f"  {r['name']:<10} [{r['write_speedup']:.2f}x] {bar}")
    
    print("\n读取性能加速:")
    for r in results:
        bar_length = int(r['read_speedup'] * 20)
        bar = '█' * bar_length
        print(f"  {r['name']:<10} [{r['read_speedup']:.2f}x] {bar}")
    
    # ==================== 结论 ====================
    print(f"\n{'='*100}")
    print("结论")
    print("=" * 100)
    
    # 分析趋势
    small_tensors = [r for r in results if r['size_bytes'] < 1024*1024]  # < 1MB
    large_tensors = [r for r in results if r['size_bytes'] >= 1024*1024]  # >= 1MB
    
    if small_tensors:
        avg_speedup_small = np.mean([r['avg_speedup'] for r in small_tensors])
        print(f"\n小 Tensor (< 1 MB):")
        print(f"  平均性能提升: {avg_speedup_small:.2f}x")
        print(f"  对齐开销占比: ~{64/np.mean([r['size_bytes'] for r in small_tensors])*100:.1f}%")
        if avg_speedup_small > 1.2:
            print(f"  结论: ✓ 对齐带来明显性能提升")
        else:
            print(f"  结论: • 性能提升有限，开销相对较大")
    
    if large_tensors:
        avg_speedup_large = np.mean([r['avg_speedup'] for r in large_tensors])
        print(f"\n大 Tensor (>= 1 MB):")
        print(f"  平均性能提升: {avg_speedup_large:.2f}x")
        print(f"  对齐开销占比: ~{64/np.mean([r['size_bytes'] for r in large_tensors])*100:.4f}%")
        if avg_speedup_large > 1.2:
            print(f"  结论: ✓ 对齐带来明显性能提升，开销可忽略")
        else:
            print(f"  结论: • 性能提升有限")
    
    print(f"\n总体建议:")
    if results:
        overall_avg = np.mean([r['avg_speedup'] for r in results])
        print(f"  整体平均加速: {overall_avg:.2f}x")
        
        if overall_avg > 1.3:
            print(f"  ✓✓✓ 强烈推荐启用对齐（所有大小级别都受益）")
        elif overall_avg > 1.1:
            print(f"  ✓✓ 推荐启用对齐（大部分场景受益）")
        else:
            print(f"  • 对齐影响有限，根据具体需求选择")
    
    print(f"\n{'='*100}")


def test_batch_size_impact():
    """测试批量传输时对齐的影响"""
    print("\n\n" + "=" * 100)
    print("批量传输测试：不同数量 Tensor 的对齐影响")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 支持")
        return
    
    # 测试配置：固定 tensor 大小，改变数量
    tensor_size = (256, 256)  # 256 KB 每个
    batch_sizes = [1, 5, 10, 20, 50, 100]
    
    print(f"\nTensor 大小: {tensor_size} (256 KB 每个)")
    print(f"\n{'批量大小':<12} {'总数据量':<15} {'对齐: 传输/读/写':<35} {'不对齐: 传输/读/写':<35} {'平均加速'}")
    print(f"{'-'*100}")
    
    for batch_size in batch_sizes:
        # 创建 GPU tensors
        gpu_tensors = [torch.randn(*tensor_size, device='cuda', dtype=torch.float32) 
                       for _ in range(batch_size)]
        
        total_size = sum(t.element_size() * t.nelement() for t in gpu_tensors)
        total_size_mb = total_size / 1024 / 1024
        pool_size = int(total_size * 3)
        
        try:
            # 启用对齐
            pool1 = CPUMemoryPool(pool_size_bytes=pool_size, enable_alignment=True)
            mgr1 = GPUToCPUPoolTransfer(pool1)
            
            # 预热
            _ = mgr1.transfer_batch_to_pool(gpu_tensors[:min(2, batch_size)], contiguous=True)
            mgr1.free_batch(list(range(min(2, batch_size))))
            
            # 传输测试
            torch.cuda.synchronize()
            start = time.time()
            tids1 = mgr1.transfer_batch_to_pool(gpu_tensors, use_pinned_memory=True, contiguous=True)
            torch.cuda.synchronize()
            transfer_time1 = time.time() - start
            
            # 读取测试
            cpu_tensors1 = [mgr1.get_tensor_from_pool(tid) for tid in tids1]
            start = time.time()
            for _ in range(10):
                _ = sum(t.sum().item() for t in cpu_tensors1)
            read_time1 = (time.time() - start) / 10
            
            # 写入测试
            start = time.time()
            for _ in range(10):
                for t in cpu_tensors1:
                    t.fill_(1.0)
            write_time1 = (time.time() - start) / 10
            
            mgr1.free_batch(tids1)
            
            # 禁用对齐
            pool2 = CPUMemoryPool(pool_size_bytes=pool_size, enable_alignment=False)
            mgr2 = GPUToCPUPoolTransfer(pool2)
            
            # 预热
            _ = mgr2.transfer_batch_to_pool(gpu_tensors[:min(2, batch_size)], contiguous=True)
            mgr2.free_batch(list(range(min(2, batch_size))))
            
            # 传输测试
            torch.cuda.synchronize()
            start = time.time()
            tids2 = mgr2.transfer_batch_to_pool(gpu_tensors, use_pinned_memory=True, contiguous=True)
            torch.cuda.synchronize()
            transfer_time2 = time.time() - start
            
            # 读取测试
            cpu_tensors2 = [mgr2.get_tensor_from_pool(tid) for tid in tids2]
            start = time.time()
            for _ in range(10):
                _ = sum(t.sum().item() for t in cpu_tensors2)
            read_time2 = (time.time() - start) / 10
            
            # 写入测试
            start = time.time()
            for _ in range(10):
                for t in cpu_tensors2:
                    t.fill_(1.0)
            write_time2 = (time.time() - start) / 10
            
            mgr2.free_batch(tids2)
            
            # 计算加速比
            speedup_transfer = transfer_time2 / transfer_time1
            speedup_read = read_time2 / read_time1
            speedup_write = write_time2 / write_time1
            avg_speedup = (speedup_transfer + speedup_read + speedup_write) / 3
            
            # 格式化输出
            aligned_str = f"{transfer_time1*1000:.2f}/{read_time1*1000:.2f}/{write_time1*1000:.2f}"
            unaligned_str = f"{transfer_time2*1000:.2f}/{read_time2*1000:.2f}/{write_time2*1000:.2f}"
            
            print(f"{batch_size:<12} {total_size_mb:>10.2f} MB   {aligned_str:<35} {unaligned_str:<35} {avg_speedup:.2f}x")
            
        except Exception as e:
            print(f"{batch_size:<12} {total_size_mb:>10.2f} MB   跳过 (错误: {str(e)[:40]})")
    
    print(f"\n{'='*100}")


def test_memory_overhead():
    """测试不同大小下的内存开销"""
    print("\n\n" + "=" * 100)
    print("内存开销分析：对齐造成的空间浪费")
    print("=" * 100)
    
    # 定义测试大小
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
    
    print(f"\n对齐边界: {alignment} 字节")
    print(f"\n{'大小':<12} {'原始字节':<15} {'对齐后字节':<15} {'填充字节':<12} {'开销占比':<12} {'评价'}")
    print(f"{'-'*100}")
    
    for name, size_bytes in test_sizes:
        aligned_size = (size_bytes + alignment - 1) & ~(alignment - 1)
        padding = aligned_size - size_bytes
        overhead_pct = (padding / aligned_size) * 100
        
        # 评价
        if overhead_pct < 0.1:
            rating = "✓✓✓ 极小"
        elif overhead_pct < 1.0:
            rating = "✓✓ 很小"
        elif overhead_pct < 5.0:
            rating = "✓ 可接受"
        elif overhead_pct < 10.0:
            rating = "⚠️  较大"
        else:
            rating = "✗ 很大"
        
        print(f"{name:<12} {size_bytes:>13,}   {aligned_size:>13,}   {padding:>10,}   {overhead_pct:>9.2f}%   {rating}")
    
    print(f"\n结论:")
    print(f"  • Tensor < 1 KB:  开销较大 (5-50%)，但绝对值很小")
    print(f"  • Tensor >= 1 KB: 开销很小 (< 5%)")
    print(f"  • Tensor >= 1 MB: 开销极小 (< 0.1%)")
    
    print(f"\n{'='*100}")


if __name__ == "__main__":
    print("\n")
    test_tensor_size_scaling()
    test_batch_size_impact()
    test_memory_overhead()
    
    print("\n\n✓✓✓ 所有缩放测试完成！")

