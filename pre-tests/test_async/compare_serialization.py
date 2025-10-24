#!/usr/bin/env python3
"""
对比手动序列化 vs Queue 自动序列化的性能

测试场景：
- 方式1：手动序列化 → Queue.put(bytes) → Queue.get(bytes) → 手动反序列化
- 方式2：Queue.put(tensor) → Queue.get(tensor) （Queue 自动序列化/反序列化）
"""

import torch
import multiprocessing as mp
import pickle
import time
from typing import Tuple


def worker_manual_deser(queue_in: mp.Queue, queue_out: mp.Queue):
    """方式1：接收序列化数据，手动反序列化"""
    while True:
        item = queue_in.get()
        if item is None:  # 停止信号
            break
        
        tensor_idx, name, serialized = item
        
        # 手动反序列化
        start = time.perf_counter()
        tensor = pickle.loads(serialized)
        deser_time = time.perf_counter() - start
        
        queue_out.put((tensor_idx, name, deser_time, tensor.shape))


def worker_auto_deser(queue_in: mp.Queue, queue_out: mp.Queue):
    """方式2：Queue 自动反序列化"""
    while True:
        item = queue_in.get()
        if item is None:  # 停止信号
            break
        
        tensor_idx, name, tensor = item
        
        # Queue 已经自动反序列化了
        queue_out.put((tensor_idx, name, 0, tensor.shape))


def test_manual_serialization(tensors):
    """测试方式1：手动序列化"""
    print("=" * 60)
    print("方式 1: 手动序列化")
    print("=" * 60)
    
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    
    # 启动 worker
    process = mp.Process(target=worker_manual_deser, args=(queue_in, queue_out))
    process.start()
    
    total_ser_time = 0
    total_put_time = 0
    total_get_time = 0
    total_deser_time = 0
    
    for i, tensor in enumerate(tensors):
        # 主进程：手动序列化
        start = time.perf_counter()
        serialized = pickle.dumps(tensor.contiguous())
        ser_time = time.perf_counter() - start
        total_ser_time += ser_time
        
        # 主进程：放入队列
        start = time.perf_counter()
        queue_in.put((i, f"tensor_{i}", serialized))
        put_time = time.perf_counter() - start
        total_put_time += put_time
        
        # 主进程：从结果队列获取
        start = time.perf_counter()
        tensor_idx, name, deser_time, shape = queue_out.get()
        get_time = time.perf_counter() - start
        total_get_time += get_time
        total_deser_time += deser_time
        
        print(f"  Tensor {i} ({shape}):")
        print(f"    主进程序列化: {ser_time*1000:>8.3f} ms")
        print(f"    Queue put:    {put_time*1000:>8.3f} ms")
        print(f"    Queue get:    {get_time*1000:>8.3f} ms")
        print(f"    子进程反序列化: {deser_time*1000:>8.3f} ms")
        print(f"    小计:         {(ser_time+put_time+get_time+deser_time)*1000:>8.3f} ms")
    
    # 停止 worker
    queue_in.put(None)
    process.join()
    
    total_time = total_ser_time + total_put_time + total_get_time + total_deser_time
    
    print(f"\n  总计:")
    print(f"    主进程序列化: {total_ser_time*1000:>8.3f} ms")
    print(f"    Queue put:    {total_put_time*1000:>8.3f} ms")
    print(f"    Queue get:    {total_get_time*1000:>8.3f} ms")
    print(f"    子进程反序列化: {total_deser_time*1000:>8.3f} ms")
    print(f"    总时间:       {total_time*1000:>8.3f} ms")
    
    return total_time


def test_auto_serialization(tensors):
    """测试方式2：Queue 自动序列化"""
    print("\n" + "=" * 60)
    print("方式 2: Queue 自动序列化")
    print("=" * 60)
    
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    
    # 启动 worker
    process = mp.Process(target=worker_auto_deser, args=(queue_in, queue_out))
    process.start()
    
    total_put_time = 0
    total_get_time = 0
    
    for i, tensor in enumerate(tensors):
        # 主进程：直接放入队列（Queue 自动序列化）
        start = time.perf_counter()
        queue_in.put((i, f"tensor_{i}", tensor))
        put_time = time.perf_counter() - start
        total_put_time += put_time
        
        # 主进程：从结果队列获取
        start = time.perf_counter()
        tensor_idx, name, _, shape = queue_out.get()
        get_time = time.perf_counter() - start
        total_get_time += get_time
        
        print(f"  Tensor {i} ({shape}):")
        print(f"    Queue put:    {put_time*1000:>8.3f} ms  ← 包含自动序列化")
        print(f"    Queue get:    {get_time*1000:>8.3f} ms  ← 包含自动反序列化")
        print(f"    小计:         {(put_time+get_time)*1000:>8.3f} ms")
    
    # 停止 worker
    queue_in.put(None)
    process.join()
    
    total_time = total_put_time + total_get_time
    
    print(f"\n  总计:")
    print(f"    Queue put:    {total_put_time*1000:>8.3f} ms")
    print(f"    Queue get:    {total_get_time*1000:>8.3f} ms")
    print(f"    总时间:       {total_time*1000:>8.3f} ms")
    
    return total_time


def test_serialization_scaling():
    """测试序列化/反序列化时间随 tensor 大小的变化"""
    print("\n" + "=" * 70)
    print("📊 序列化性能随 Tensor 大小的变化")
    print("=" * 70)
    
    # 测试不同大小的 tensor (MB)
    test_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    print(f"\n测试配置:")
    print(f"  - Tensor 大小: {test_sizes} MB")
    print(f"  - 每个大小测试 3 次取平均")
    
    results = []
    
    for size_mb in test_sizes:
        # 创建指定大小的 tensor
        num_elements = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
        side = int(num_elements ** 0.5)
        
        # 测试 3 次取平均
        ser_times = []
        deser_times = []
        
        for _ in range(3):
            tensor = torch.randn(side, side)
            
            # 测试序列化
            start = time.perf_counter()
            serialized = pickle.dumps(tensor.contiguous())
            ser_time = time.perf_counter() - start
            ser_times.append(ser_time)
            
            # 测试反序列化
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
    
    # 打印结果表格
    print(f"\n{'大小(MB)':<10} {'序列化(ms)':<14} {'反序列化(ms)':<14} {'总时间(ms)':<12} {'MB/s'}")
    print("-" * 70)
    
    for r in results:
        throughput = r['size_mb'] / (r['total_time'] / 1000)  # MB/s
        print(f"{r['size_mb']:<10} {r['ser_time']:<14.2f} {r['deser_time']:<14.2f} {r['total_time']:<12.2f} {throughput:.1f}")
    
    # 分析
    print("\n" + "=" * 70)
    print("📈 性能分析")
    print("=" * 70)
    
    # 计算序列化和反序列化的占比
    total_ser = sum(r['ser_time'] for r in results)
    total_deser = sum(r['deser_time'] for r in results)
    total_all = total_ser + total_deser
    
    print(f"\n1. 时间占比:")
    print(f"   - 序列化:   {total_ser:.2f} ms ({total_ser/total_all*100:.1f}%)")
    print(f"   - 反序列化: {total_deser:.2f} ms ({total_deser/total_all*100:.1f}%)")
    
    # 计算平均吞吐量
    avg_throughput = sum(r['size_mb'] / (r['total_time'] / 1000) for r in results) / len(results)
    print(f"\n2. 平均吞吐量: {avg_throughput:.1f} MB/s")
    
    # 分析时间复杂度
    small_time_per_mb = results[0]['total_time'] / results[0]['size_mb']
    large_time_per_mb = results[-1]['total_time'] / results[-1]['size_mb']
    
    print(f"\n3. 时间复杂度:")
    print(f"   - 小tensor ({results[0]['size_mb']}MB): {small_time_per_mb:.2f} ms/MB")
    print(f"   - 大tensor ({results[-1]['size_mb']}MB): {large_time_per_mb:.2f} ms/MB")
    
    if abs(small_time_per_mb - large_time_per_mb) / small_time_per_mb < 0.2:
        print(f"   - ✅ 接近线性 O(n)：时间与大小成正比")
    else:
        print(f"   - ⚠️  非线性：大 tensor 的单位时间 {'更长' if large_time_per_mb > small_time_per_mb else '更短'}")
    
    print("\n4. 关键发现:")
    print(f"   - 序列化比反序列化慢 {total_ser/total_deser:.1f}x")
    print(f"   - Tensor 越大，序列化开销越明显")
    print(f"   - 64MB tensor 需要 {results[-1]['total_time']:.0f}ms，约占训练迭代的显著部分")
    
    return results


def main():
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    print("\n" + "🚀 " + "序列化性能测试套件".center(58) + " 🚀")
    
    # 测试 1: 序列化性能随大小的变化
    test_serialization_scaling()
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

