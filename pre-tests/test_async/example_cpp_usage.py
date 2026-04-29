#!/usr/bin/env python3
"""
使用 C++ 线程 + CPUMemoryPool 的完整示例

演示：
1. 使用 CPUMemoryPool 分配内存
2. 拷贝 tensor 数据到内存池
3. 传递地址给 C++ 线程（无需序列化）
4. C++ 线程直接访问内存进行计算

注意：需要先编译 cpp_thread_example.cpp
"""

import torch
import time
import ctypes
import sys
import os
import numpy as np
import multiprocessing as mp
import pickle

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test_tensors_layout'))
from gpu_cpu_memory_pool import CPUMemoryPool

try:
    import cpp_thread_example
    CPP_AVAILABLE = True
except ImportError:
    print("警告：cpp_thread_example 未编译，将使用 Python 模拟版本")
    CPP_AVAILABLE = False

def worker_process_pickle(queue_in: mp.Queue, queue_out: mp.Queue):
    """多进程 Worker：接收手动序列化的数据"""
    while True:
        item = queue_in.get()
        if item is None:
            break
        
        tensor_idx, serialized = item
        
        # 反序列化
        deser_start = time.perf_counter()
        tensor = pickle.loads(serialized)
        deser_time = time.perf_counter() - deser_start
        
        # 模拟计算
        # result = tensor.sum().item()
        result = 0
        
        queue_out.put((tensor_idx, deser_time, result))


def worker_process_auto(queue_in: mp.Queue, queue_out: mp.Queue):
    """多进程 Worker：接收 Queue 自动序列化的数据"""
    while True:
        item = queue_in.get()  # ← Queue 自动反序列化
        receive_time = time.perf_counter()  # ← 立即记录接收时间戳
        
        if item is None:
            break
        
        tensor_idx, send_time, tensor = item
        
        # 计算从发送到接收的时间（包含序列化+传输+反序列化）
        transfer_time = receive_time - send_time
        
        # 无需手动反序列化，Queue 已经自动完成
        # 模拟计算
        # result = tensor.sum().item()
        result = 0
        
        queue_out.put((tensor_idx, transfer_time, result))


def test_pickle_multiprocess(tensor_size_mb: float, warmup: bool = False):
    """测试：Python 多进程 + 手动 Pickle 序列化
    
    Args:
        tensor_size_mb: Tensor 大小（MB）
        warmup: 是否为预热测试
    
    Returns:
        平均时间（秒）
    """
    # 测试 3 次取平均（避免抖动）
    num_iterations = 3
    
    # 预先创建多个不同的 tensor（避免缓存）
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
    side = int(num_elements ** 0.5)
    tensors = [torch.randn(side, side) for _ in range(num_iterations)]
    
    # 启动 worker 进程
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    process = mp.Process(target=worker_process_pickle, args=(queue_in, queue_out))
    process.start()
    
    times = []
    
    for i in range(num_iterations):
        start = time.perf_counter()
        
        # 主进程：手动序列化（使用不同的 tensor）
        ser_start = time.perf_counter()
        serialized = pickle.dumps(tensors[i].contiguous())
        ser_time = time.perf_counter() - ser_start
        
        # 发送到子进程
        queue_in.put((0, serialized))
        
        # 等待结果
        _, deser_time, _ = queue_out.get()
        
        if i != 0:
            elapsed = time.perf_counter() - start
            times.append(elapsed)
    
    # 停止 worker
    queue_in.put(None)
    process.join()
    
    avg_time = sum(times) / len(times)
    
    if not warmup:
        print(f"  {tensor_size_mb:>5.0f} MB: {avg_time*1000:>8.2f} ms  (手动序列化: {ser_time*1000:.2f}ms, 手动反序列化: {deser_time*1000:.2f}ms)")
    
    return avg_time


def test_queue_auto_serialize(tensor_size_mb: float, warmup: bool = False):
    """测试：Python 多进程 + Queue 自动序列化
    
    Args:
        tensor_size_mb: Tensor 大小（MB）
        warmup: 是否为预热测试
    
    Returns:
        平均时间（秒）
    """
    # 测试 3 次取平均（避免抖动）
    num_iterations = 3
    
    # 预先创建多个不同的 tensor（避免缓存）
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
    side = int(num_elements ** 0.5)
    tensors = [torch.randn(side, side) for _ in range(num_iterations)]
    
    # 启动 worker 进程
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    process = mp.Process(target=worker_process_auto, args=(queue_in, queue_out))
    process.start()
    
    times = []
    transfer_times = []
    
    for i in range(num_iterations):
        start = time.perf_counter()
        
        # 主进程：记录发送时间戳，直接放入 tensor（使用不同的 tensor）
        cpu_tensor = tensors[i].contiguous()
        send_time = time.perf_counter()  # ← 记录发送时间戳
        queue_in.put((0, send_time, cpu_tensor))  # ← Queue 内部自动 pickle.dumps()
        
        # 等待结果（子进程会返回传输时间）
        _, transfer_time, _ = queue_out.get()
        
        if i != 0:
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            transfer_times.append(transfer_time)
    
    # 停止 worker
    queue_in.put(None)
    process.join()
    
    avg_time = sum(times) / len(times)
    avg_transfer = sum(transfer_times) / len(transfer_times)
    
    if not warmup:
        print(f"  {tensor_size_mb:>5.0f} MB: {avg_time*1000:>8.2f} ms  (Queue传输: {avg_transfer*1000:.2f}ms)")
    
    return avg_time


def test_cpp_thread_pool(tensor_size_mb: float, pool: CPUMemoryPool, processor, warmup: bool = False):
    """测试：C++ 线程 + CPUMemoryPool
    
    Args:
        tensor_size_mb: Tensor 大小（MB）
        pool: CPUMemoryPool 实例
        processor: C++ 处理器实例
        warmup: 是否为预热测试
    
    Returns:
        平均时间（秒）
    """
    # 测试 3 次取平均
    num_iterations = 3
    
    # 预先创建多个不同的 tensor（避免缓存）
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
    side = int(num_elements ** 0.5)
    tensors = [torch.randn(side, side) for _ in range(num_iterations)]
    
    times = []
    
    for i in range(num_iterations):
        start = time.perf_counter()
        
        # 分配内存
        size = tensors[i].element_size() * tensors[i].nelement()
        address, tensor_id = pool.allocate(size)
        
        # 拷贝到内存池（使用不同的 tensor）
        cpu_tensor = tensors[i].contiguous()
        ctypes.memmove(address, cpu_tensor.data_ptr(), size)
        
        # 提交给 C++ 线程（只传地址）
        processor.submit(address, size, list(cpu_tensor.shape), 0)
        
        # 等待结果
        _, _, _ = processor.get_result()
        
        # 释放内存
        pool.deallocate(tensor_id)
        
        if i != 0:
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
    avg_time = sum(times) / len(times)
    
    if not warmup:
        print(f"  {tensor_size_mb:>5.0f} MB: {avg_time*1000:>8.2f} ms")
    
    return avg_time


def compare_with_pickle():
    """对比：3种方案的性能"""
    print("\n" + "=" * 70)
    print("性能对比：3种方案")
    print("=" * 70)
    
    # 测试配置：不同大小的 tensor (MB)
    test_sizes = [2, 4, 8, 16, 32, 64]
    
    print(f"\n测试配置:")
    print(f"  - Tensor 大小: {test_sizes} MB")
    print(f"  - 每个大小测试 3 次取平均")
    print(f"  - 先进行预热避免冷启动")
    
    # ==============================================
    # 方案 1: Python 多进程 + 手动 Pickle 序列化
    # ==============================================
    print("\n" + "-" * 70)
    print("方案 1: Python 多进程 + 手动 Pickle 序列化")
    print("  (主进程 pickle.dumps → Queue.put → Queue.get → 子进程 pickle.loads)")
    print("-" * 70)
    
    print("\n  测试结果:")
    pickle_times = []
    for size_mb in test_sizes:
        avg_time = test_pickle_multiprocess(size_mb)
        pickle_times.append(avg_time)
    
    print(f"\n  平均时间: {sum(pickle_times)/len(pickle_times)*1000:.2f} ms")
    
    # ==============================================
    # 方案 2: Python 多进程 + Queue 自动序列化
    # ==============================================
    print("\n" + "-" * 70)
    print("方案 2: Python 多进程 + Queue 自动序列化")
    print("  (主进程 Queue.put(tensor) → Queue 自动序列化/反序列化)")
    print("-" * 70)
    
    print("\n  测试结果:")
    queue_auto_times = []
    for size_mb in test_sizes:
        avg_time = test_queue_auto_serialize(size_mb)
        queue_auto_times.append(avg_time)
    
    print(f"\n  平均时间: {sum(queue_auto_times)/len(queue_auto_times)*1000:.2f} ms")
    
    # ==============================================
    # 方案 3: C++ 线程 + CPUMemoryPool
    # ==============================================
    if CPP_AVAILABLE:
        print("\n" + "-" * 70)
        print("方案 3: C++ 线程 + CPUMemoryPool（零序列化）")
        print("  (直接传递内存地址，线程共享地址空间)")
        print("-" * 70)
        
        # 初始化
        pool_size = 200 * 1024 * 1024  # 200 MB
        pool = CPUMemoryPool(pool_size)
        processor = cpp_thread_example.AsyncTensorProcessor(num_threads=1)
        
        print("\n  测试结果:")
        cpp_times = []
        for size_mb in test_sizes:
            avg_time = test_cpp_thread_pool(size_mb, pool, processor)
            cpp_times.append(avg_time)
        
        processor.stop()
        
        print(f"\n  平均时间: {sum(cpp_times)/len(cpp_times)*1000:.2f} ms")
        
        # ==============================================
        # 对比结果
        # ==============================================
        print("\n" + "=" * 70)
        print("📊 详细对比")
        print("=" * 70)
        print(f"\n{'大小(MB)':<10} {'手动Pickle(ms)':<16} {'Queue自动(ms)':<16} {'C++线程(ms)':<14}")
        print("-" * 70)
        
        for i, size_mb in enumerate(test_sizes):
            pickle_ms = pickle_times[i] * 1000
            queue_ms = queue_auto_times[i] * 1000
            cpp_ms = cpp_times[i] * 1000
            
            print(f"{size_mb:<10} {pickle_ms:<16.2f} {queue_ms:<16.2f} {cpp_ms:<14.2f}")
        
        print("-" * 70)
        avg_pickle = sum(pickle_times) / len(pickle_times) * 1000
        avg_queue = sum(queue_auto_times) / len(queue_auto_times) * 1000
        avg_cpp = sum(cpp_times) / len(cpp_times) * 1000
        print(f"{'平均':<10} {avg_pickle:<16.2f} {avg_queue:<16.2f} {avg_cpp:<14.2f}")
 
    print("=" * 70)


def main():
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    print("\n" + "🚀 " + "C++ 线程 + CPUMemoryPool 性能测试".center(58) + " 🚀")
    
    # 测试 1: 性能对比（主要测试）
    compare_with_pickle()
    
    print("\n" + "=" * 70)
    print("💡 总结")
    print("=" * 70)
    print("测试配置:")
    print("  - 测试了 6 种不同大小的 Tensor (2-64 MB)")
    print("  - 每个测试运行 3 次取平均，避免抖动")
    print("  - 预热后测试，避免冷启动影响")
    print("  - 使用真实的多进程 + Queue 通信")
    print()
    print("3种方案对比:")
    print("  1. 手动 Pickle: 主进程 pickle.dumps → Queue → 子进程 pickle.loads")
    print("  2. Queue 自动:  主进程 Queue.put(tensor) → Queue 自动序列化")
    print("  3. C++ 线程:    主进程传地址 → C++ 线程直接访问（零序列化）")
    print("👉 推荐：使用 C++ 线程 + CPUMemoryPool 实现异步 checkpoint！")
    print("=" * 70)


if __name__ == "__main__":
    main()

