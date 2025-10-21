"""
分布式异步 Checkpoint 系统 - 原生 Torch 方式
==========================================

场景：
- 两个独立启动的程序（rank 0 和 rank 1）
- 每个程序有 3 个 GPU tensor
- 流水线架构:
  1. 主进程: 负责 GPU→CPU 传输和序列化（D2H 阶段）
  2. Exchange Worker（子进程）: 使用 send/recv 与另一个程序交换数据
  3. XOR Worker（子进程）: 对 tensor 数据按位异或
- 3 个 tensor 采用流水线方式处理

实现方式：
- 主进程在前台执行 D2H（避免跨进程传递 GPU tensor）
- 两个子进程并行处理后续阶段
- 详细的序列化性能统计
"""

import torch
import torch.distributed as dist
import time
import multiprocessing as mp
import pickle
import numpy as np
import os
import argparse
from typing import Dict


# ============================================================================
# Worker 函数（子进程）- 原生 Torch 方式
# ============================================================================

def exchange_worker_torch(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    stats_queue: mp.Queue,
    rank: int,
    world_size: int,
    ready_event: mp.Event
):
    """
    Exchange Worker - 原生 torch 方式
    使用 torch.distributed send/recv 交换数据
    
    统计项：
    - deserialize_time: CPU tensor 反序列化时间
    - comm_time: 分布式通信时间
    - serialize_time: 结果序列化时间
    - total_time: 总时间
    """
    # 子进程需要设置环境变量并初始化分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # 初始化分布式环境
    try:
        dist.init_process_group(
            backend='gloo',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        print(f"[Rank {rank}] Exchange Worker (torch): 分布式环境初始化成功")
    except Exception as e:
        print(f"[Rank {rank}] Exchange Worker (torch): 分布式初始化失败: {e}")
        stats_queue.put(('ready', 'exchange_worker_torch_failed'))
        return
    
    # 发送就绪信号
    stats_queue.put(('ready', 'exchange_worker_torch'))
    print(f"[Rank {rank}] Exchange Worker (torch): 已就绪")
    
    # 等待所有 worker 就绪
    ready_event.wait()
    
    peer_rank = 1 - rank  # 交换对象
    while True:
        try:
            item = input_queue.get()
            if item is None:
                break
            
            tensor_idx, name, cpu_tensor_serialized = item
            total_start = time.perf_counter()
            
            # ===== 1. 反序列化 CPU tensor =====
            deser_start = time.perf_counter()
            cpu_tensor = pickle.loads(cpu_tensor_serialized)
            deser_time = time.perf_counter() - deser_start
            
            # ===== 2. 分布式通信 =====
            comm_start = time.perf_counter()
            if rank == 0:
                dist.send(cpu_tensor, dst=peer_rank)
                received_tensor = torch.zeros_like(cpu_tensor)
                dist.recv(received_tensor, src=peer_rank)
            else:
                received_tensor = torch.zeros_like(cpu_tensor)
                dist.recv(received_tensor, src=peer_rank)
                dist.send(cpu_tensor, dst=peer_rank)
            comm_time = time.perf_counter() - comm_start
            
            # ===== 3. 序列化结果 =====
            ser_start = time.perf_counter()
            received_serialized = pickle.dumps(received_tensor)
            ser_time = time.perf_counter() - ser_start
            
            total_time = time.perf_counter() - total_start
            
            # 传递到下一阶段
            output_queue.put((tensor_idx, name, cpu_tensor_serialized, received_serialized))
            
            # 发送详细统计
            stats_queue.put(('exchange_detailed', tensor_idx, {
                'deserialize_time': deser_time,
                'comm_time': comm_time,
                'serialize_time': ser_time,
                'total_time': total_time,
                'tensor_size_bytes': cpu_tensor.element_size() * cpu_tensor.nelement()
            }))
            stats_queue.put(('exchange', tensor_idx, total_time))
            
        except Exception as e:
            if "Empty" not in str(type(e).__name__):
                print(f"[Rank {rank}] Exchange Worker Error: {e}")
    
    # 清理分布式环境
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[Rank {rank}] Exchange Worker (torch): 分布式环境已清理")
    except Exception as e:
        print(f"[Rank {rank}] Exchange Worker (torch): 清理失败: {e}")


def xor_worker_torch(
    input_queue: mp.Queue,
    stats_queue: mp.Queue,
    rank: int,
    ready_event: mp.Event
):
    """
    XOR Worker - 原生 torch 方式
    对本地和接收的 tensor 按位异或
    
    统计项：
    - deserialize_local_time: 本地 tensor 反序列化时间
    - deserialize_received_time: 接收 tensor 反序列化时间
    - xor_time: XOR 计算时间
    - total_time: 总时间
    """
    # 发送就绪信号
    stats_queue.put(('ready', 'xor_worker_torch'))
    print(f"[Rank {rank}] XOR Worker (torch): 已就绪")
    
    # 等待所有 worker 就绪
    ready_event.wait()
    
    while True:
        try:
            item = input_queue.get(timeout=0.1)
            if item is None:
                break
            
            tensor_idx, name, local_serialized, received_serialized = item
            total_start = time.perf_counter()
            
            # ===== 1. 反序列化本地 tensor =====
            deser_local_start = time.perf_counter()
            local_tensor = pickle.loads(local_serialized)
            deser_local_time = time.perf_counter() - deser_local_start
            
            # ===== 2. 反序列化接收 tensor =====
            deser_received_start = time.perf_counter()
            received_tensor = pickle.loads(received_serialized)
            deser_received_time = time.perf_counter() - deser_received_start
            
            # ===== 3. 按位异或 =====
            xor_start = time.perf_counter()
            local_int = local_tensor.view(torch.int32)
            received_int = received_tensor.view(torch.int32)
            xor_result = torch.bitwise_xor(local_int, received_int)
            xor_time = time.perf_counter() - xor_start
            
            total_time = time.perf_counter() - total_start
            
            # 发送详细统计
            stats_queue.put(('xor_detailed', tensor_idx, {
                'deserialize_local_time': deser_local_time,
                'deserialize_received_time': deser_received_time,
                'xor_time': xor_time,
                'total_time': total_time,
                'tensor_size_bytes': local_tensor.element_size() * local_tensor.nelement()
            }))
            stats_queue.put(('xor', tensor_idx, total_time))
            
        except Exception as e:
            if "Empty" not in str(type(e).__name__):
                print(f"[Rank {rank}] XOR Worker Error: {e}")


# ============================================================================
# 流水线管理类
# ============================================================================

class AsyncPipelineTorch:
    """异步流水线 - 原生 torch 方式（主进程负责 D2H）"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        
        # 两个队列（主进程直接处理 D2H，不需要 d2h_queue）
        self.exchange_queue = mp.Queue()
        self.xor_queue = mp.Queue()
        self.stats_queue = mp.Queue()
        
        # 同步事件（确保所有 worker 就绪后再开始）
        self.ready_event = mp.Event()
        
        # 两个子进程（主进程负责 D2H）
        self.exchange_process = None
        self.xor_process = None
        
        # 统计
        self.stats = {
            'd2h_times': [],
            'exchange_times': [],
            'xor_times': [],
            'd2h_detailed': [],       # D2H 详细统计（主进程）
            'exchange_detailed': [],  # Exchange 详细统计
            'xor_detailed': []        # XOR 详细统计
        }
    
    def start_pipeline(self):
        """启动两个 worker 子进程（主进程负责 D2H）"""
        print(f"[Rank {self.rank}] 启动两个 worker 子进程...")
        
        # 只启动 Exchange 和 XOR workers（D2H 由主进程负责）
        self.exchange_process = mp.Process(
            target=exchange_worker_torch,
            args=(self.exchange_queue, self.xor_queue, self.stats_queue, self.rank, self.world_size, self.ready_event),
            daemon=True
        )
        self.exchange_process.start()
        
        self.xor_process = mp.Process(
            target=xor_worker_torch,
            args=(self.xor_queue, self.stats_queue, self.rank, self.ready_event),
            daemon=True
        )
        self.xor_process.start()
        
        # 等待所有 worker 就绪
        print(f"[Rank {self.rank}] 等待所有 worker 初始化...")
        ready_count = 0
        expected_workers = 2  # 只有 2 个子进程
        
        while ready_count < expected_workers:
            try:
                while not self.stats_queue.empty():
                    stat = self.stats_queue.get_nowait()
                    if len(stat) == 2 and stat[0] == 'ready':
                        ready_count += 1
                        print(f"[Rank {self.rank}]   {stat[1]} 就绪 ({ready_count}/{expected_workers})")
            except:
                pass
            time.sleep(0.1)
        
        if ready_count == expected_workers:
            print(f"[Rank {self.rank}] ✓ 所有 worker 已就绪，主进程开始执行 D2H")
            self.ready_event.set()  # 通知所有 worker 可以开始
        else:
            print(f"[Rank {self.rank}] ✗ 警告: 只有 {ready_count}/{expected_workers} 个 worker 就绪")
            self.ready_event.set()  # 仍然继续
    
    def submit_tensor(self, tensor_idx: int, name: str, gpu_tensor: torch.Tensor):
        """
        提交 tensor 到流水线（主进程执行 D2H）
        
        在主进程执行：
        1. GPU → CPU 传输
        2. CPU tensor 序列化
        3. 放入 exchange_queue
        """
        print(f"[Rank {self.rank}] 主进程: 开始处理 tensor {tensor_idx}")
        total_start = time.perf_counter()
        
        # ===== 1. D2H 传输（主进程有完整的 CUDA 上下文）=====
        d2h_start = time.perf_counter()
        cpu_tensor = gpu_tensor.to('cpu', non_blocking=False)
        torch.cuda.synchronize()
        d2h_time = time.perf_counter() - d2h_start
        print(f"[Rank {self.rank}] 主进程: D2H 传输完成，耗时 {d2h_time*1000:.2f} ms")
        
        # ===== 2. 序列化 CPU tensor =====
        ser_start = time.perf_counter()
        cpu_tensor_serialized = pickle.dumps(cpu_tensor.contiguous())
        ser_time = time.perf_counter() - ser_start
        print(f"[Rank {self.rank}] 主进程: 序列化完成，耗时 {ser_time*1000:.2f} ms")
        
        total_time = time.perf_counter() - total_start
        
        # ===== 3. 放入 exchange_queue（供 Exchange Worker 处理）=====
        self.exchange_queue.put((tensor_idx, name, cpu_tensor_serialized))
        print(f"[Rank {self.rank}] 主进程: tensor {tensor_idx} 已提交到 Exchange Worker")
        
        # 记录 D2H 阶段的详细统计（主进程执行）
        self.stats['d2h_detailed'].append((tensor_idx, {
            'd2h_time': d2h_time,
            'serialize_time': ser_time,
            'total_time': total_time,
            'tensor_size_bytes': cpu_tensor.element_size() * cpu_tensor.nelement()
        }))
        self.stats['d2h_times'].append((tensor_idx, total_time))
    
    def stop_pipeline(self):
        """停止流水线（只需停止 2 个子进程）"""
        # 发送终止信号
        self.exchange_queue.put(None)
        self.xor_queue.put(None)
        
        # 等待子进程结束
        if self.exchange_process:
            self.exchange_process.join(timeout=2.0)
            if self.exchange_process.is_alive():
                self.exchange_process.terminate()
                print(f"[Rank {self.rank}] Exchange Worker 被强制终止")
        
        if self.xor_process:
            self.xor_process.join(timeout=2.0)
            if self.xor_process.is_alive():
                self.xor_process.terminate()
                print(f"[Rank {self.rank}] XOR Worker 被强制终止")
    
    def collect_stats(self):
        """收集统计信息"""
        while not self.stats_queue.empty():
            try:
                stat = self.stats_queue.get_nowait()
                # 处理不同格式的消息
                if len(stat) == 2:
                    # 就绪消息：('ready', worker_name)
                    continue
                elif len(stat) == 3:
                    stage, idx, data = stat
                    # 统计消息：(stage, idx, elapsed) 或 (stage_detailed, idx, dict)
                    if stage == 'd2h':
                        self.stats['d2h_times'].append((idx, data))
                    elif stage == 'exchange':
                        self.stats['exchange_times'].append((idx, data))
                    elif stage == 'xor':
                        self.stats['xor_times'].append((idx, data))
                    elif stage == 'd2h_detailed':
                        self.stats['d2h_detailed'].append((idx, data))
                    elif stage == 'exchange_detailed':
                        self.stats['exchange_detailed'].append((idx, data))
                    elif stage == 'xor_detailed':
                        self.stats['xor_detailed'].append((idx, data))
            except Exception as e:
                pass
        
        return self.stats


# ============================================================================
# 测试函数
# ============================================================================

def run_test(rank: int, world_size: int):
    """
    运行异步流水线测试
    
    Args:
        rank: 当前进程的 rank (0 or 1)
        world_size: 总进程数 (2)
    """
    mode_name = "原生 torch 异步流水线"
    
    print(f"\n{'='*80}")
    print(f"[Rank {rank}] {mode_name}")
    print(f"{'='*80}")
    
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 支持")
        return
    
    # 创建 3 个 GPU tensors
    num_tensors = 3
    tensor_shapes = [
        (512, 512),
        (1024, 1024),
        (256, 256)
    ]
    
    print(f"\n[Rank {rank}] 创建 {num_tensors} 个 GPU tensors")
    gpu_tensors = []
    total_size = 0
    for i, shape in enumerate(tensor_shapes):
        tensor = torch.randn(*shape, device='cuda', dtype=torch.float32)
        gpu_tensors.append(tensor)
        total_size += tensor.element_size() * tensor.nelement()
        size_mb = tensor.element_size() * tensor.nelement() / 1024 / 1024
        print(f"  Tensor {i}: {shape} ({size_mb:.2f} MB)")
    
    total_size_mb = total_size / 1024 / 1024
    print(f"  总大小: {total_size_mb:.2f} MB")
    
    # 创建流水线
    pipeline = AsyncPipelineTorch(rank, world_size)
    
    # 启动流水线
    pipeline.start_pipeline()
    
    # 提交 tensors（流水线处理）
    print(f"\n[Rank {rank}] 提交 tensors 到流水线...")
    start_time = time.time()
    
    for i, gpu_tensor in enumerate(gpu_tensors):
        name = f"tensor_{i}"
        pipeline.submit_tensor(i, name, gpu_tensor)
        print(f"  [Rank {rank}] 提交 Tensor {i}")
    
    # 等待完成（使用更智能的等待机制）
    print(f"\n[Rank {rank}] 等待流水线完成...")
    expected_completions = num_tensors  # 期望完成的 tensor 数量
    completed_tensors = set()
    max_wait_time = 15.0
    check_start = time.time()
    
    while len(completed_tensors) < expected_completions:
        # 收集统计信息
        pipeline.collect_stats()
        
        # 检查 XOR 阶段完成的 tensor（最后一个阶段）
        xor_completed = {idx for idx, _ in pipeline.stats['xor_times']}
        completed_tensors = xor_completed
        
        # 超时检查
        if time.time() - check_start > max_wait_time:
            print(f"[Rank {rank}] ⚠️ 等待超时，已完成 {len(completed_tensors)}/{expected_completions} 个 tensor")
            break
        
        if len(completed_tensors) < expected_completions:
            time.sleep(0.1)
    
    # 最后再收集一次统计
    time.sleep(0.5)
    pipeline.collect_stats()
    
    total_time = time.time() - start_time
    print(f"[Rank {rank}] ✓ 流水线处理完成 ({len(completed_tensors)}/{expected_completions} tensors)")
    
    # 收集统计
    stats = pipeline.collect_stats()
    
    # 停止流水线
    pipeline.stop_pipeline()
    
    # 打印详细统计
    print_detailed_stats(rank, stats, total_time)
    
    return stats, total_time


def print_detailed_stats(rank: int, stats: Dict, total_time: float):
    """打印详细的性能统计"""
    print(f"\n[Rank {rank}] ================== 异步流水线性能统计 ==================")
    print(f"  总时间: {total_time*1000:.2f} ms\n")
    
    # ===== D2H 阶段详细统计 =====
    if stats.get('d2h_detailed'):
        print(f"  【D2H 阶段详细统计】（主进程执行）")
        d2h_detailed = sorted(stats['d2h_detailed'], key=lambda x: x[0])
        
        total_d2h = 0
        total_ser = 0
        total_stage = 0
        
        for idx, details in d2h_detailed:
            print(f"    Tensor {idx}:")
            print(f"      D2H传输:   {details['d2h_time']*1000:>8.2f} ms")
            print(f"      序列化:    {details['serialize_time']*1000:>8.2f} ms")
            print(f"      总计:      {details['total_time']*1000:>8.2f} ms")
            
            total = details['total_time']
            ser_percent = details['serialize_time'] / total * 100
            print(f"      序列化占比: {ser_percent:.1f}%")
            print(f"      Tensor大小: {details['tensor_size_bytes']/1024/1024:.2f} MB\n")
            
            total_d2h += details['d2h_time']
            total_ser += details['serialize_time']
            total_stage += details['total_time']
        
        # 平均统计
        n = len(d2h_detailed)
        ser_overhead = total_ser / total_stage * 100
        print(f"    平均:")
        print(f"      D2H传输:   {total_d2h/n*1000:>8.2f} ms")
        print(f"      序列化:    {total_ser/n*1000:>8.2f} ms")
        print(f"      总计:      {total_stage/n*1000:>8.2f} ms")
        print(f"      序列化占比: {ser_overhead:.1f}%  ← 关键指标")
        print(f"      ✅ 优势: 主进程直接持有 GPU tensor，无反序列化开销！\n")
    
    # ===== Exchange 阶段详细统计 =====
    if stats.get('exchange_detailed'):
        print(f"  【Exchange 阶段详细统计】")
        exchange_detailed = sorted(stats['exchange_detailed'], key=lambda x: x[0])
        
        total_deser = 0
        total_comm = 0
        total_ser = 0
        total_stage = 0
        
        for idx, details in exchange_detailed:
            print(f"    Tensor {idx}:")
            print(f"      反序列化:  {details['deserialize_time']*1000:>8.2f} ms")
            print(f"      通信:      {details['comm_time']*1000:>8.2f} ms")
            print(f"      序列化:    {details['serialize_time']*1000:>8.2f} ms")
            print(f"      总计:      {details['total_time']*1000:>8.2f} ms")
            
            total = details['total_time']
            ser_time = details['deserialize_time'] + details['serialize_time']
            ser_percent = ser_time / total * 100
            print(f"      序列化占比: {ser_percent:.1f}%\n")
            
            total_deser += details['deserialize_time']
            total_comm += details['comm_time']
            total_ser += details['serialize_time']
            total_stage += details['total_time']
        
        # 平均统计
        n = len(exchange_detailed)
        ser_overhead = (total_deser + total_ser) / total_stage * 100
        print(f"    平均:")
        print(f"      反序列化:  {total_deser/n*1000:>8.2f} ms")
        print(f"      通信:      {total_comm/n*1000:>8.2f} ms")
        print(f"      序列化:    {total_ser/n*1000:>8.2f} ms")
        print(f"      总计:      {total_stage/n*1000:>8.2f} ms")
        print(f"      序列化占比: {ser_overhead:.1f}%  ← 关键指标\n")
    
    # ===== XOR 阶段详细统计 =====
    if stats.get('xor_detailed'):
        print(f"  【XOR 阶段详细统计】")
        xor_detailed = sorted(stats['xor_detailed'], key=lambda x: x[0])
        
        total_deser_local = 0
        total_deser_recv = 0
        total_xor = 0
        total_stage = 0
        
        for idx, details in xor_detailed:
            print(f"    Tensor {idx}:")
            print(f"      反序列化(本地): {details['deserialize_local_time']*1000:>8.2f} ms")
            print(f"      反序列化(接收): {details['deserialize_received_time']*1000:>8.2f} ms")
            print(f"      XOR计算:       {details['xor_time']*1000:>8.2f} ms")
            print(f"      总计:          {details['total_time']*1000:>8.2f} ms")
            
            total = details['total_time']
            deser_time = details['deserialize_local_time'] + details['deserialize_received_time']
            ser_percent = deser_time / total * 100
            print(f"      序列化占比: {ser_percent:.1f}%\n")
            
            total_deser_local += details['deserialize_local_time']
            total_deser_recv += details['deserialize_received_time']
            total_xor += details['xor_time']
            total_stage += details['total_time']
        
        # 平均统计
        n = len(xor_detailed)
        ser_overhead = (total_deser_local + total_deser_recv) / total_stage * 100
        print(f"    平均:")
        print(f"      反序列化(本地): {total_deser_local/n*1000:>8.2f} ms")
        print(f"      反序列化(接收): {total_deser_recv/n*1000:>8.2f} ms")
        print(f"      XOR计算:       {total_xor/n*1000:>8.2f} ms")
        print(f"      总计:          {total_stage/n*1000:>8.2f} ms")
        print(f"      序列化占比: {ser_overhead:.1f}%  ← 关键指标\n")
    
    # ===== 总体序列化开销分析 =====
    print(f"  【总体序列化开销分析】")
    
    if stats.get('d2h_detailed') and stats.get('exchange_detailed') and stats.get('xor_detailed'):
        # 计算所有阶段的序列化总时间
        total_serialization = 0
        total_computation = 0
        total_execution = 0
        
        for idx, details in stats['d2h_detailed']:
            total_serialization += details['serialize_time']  # 主进程执行，无 deserialize
            total_computation += details['d2h_time']
            total_execution += details['total_time']
        
        for idx, details in stats['exchange_detailed']:
            total_serialization += details['deserialize_time'] + details['serialize_time']
            total_computation += details['comm_time']
            total_execution += details['total_time']
        
        for idx, details in stats['xor_detailed']:
            total_serialization += details['deserialize_local_time'] + details['deserialize_received_time']
            total_computation += details['xor_time']
            total_execution += details['total_time']
        
        ser_percent_overall = total_serialization / total_execution * 100
        
        print(f"    序列化总时间:   {total_serialization*1000:.2f} ms")
        print(f"    计算总时间:     {total_computation*1000:.2f} ms")
        print(f"    执行总时间:     {total_execution*1000:.2f} ms")
        print(f"    序列化占比:     {ser_percent_overall:.1f}%  ← 🔑 整体序列化开销")
        print(f"    计算占比:       {total_computation/total_execution*100:.1f}%\n")
    
    # ===== 流水线并行效率 =====
    if stats['d2h_times'] and stats['exchange_times'] and stats['xor_times']:
        d2h_total = sum([t[1] for t in stats['d2h_times']])
        exchange_total = sum([t[1] for t in stats['exchange_times']])
        xor_total = sum([t[1] for t in stats['xor_times']])
        
        sequential_time = d2h_total + exchange_total + xor_total
        parallel_efficiency = (sequential_time / total_time - 1) * 100 if total_time > 0 else 0
        
        print(f"  【流水线并行效率】")
        print(f"    顺序执行时间（估算）: {sequential_time*1000:.2f} ms")
        print(f"    实际并行执行时间:     {total_time*1000:.2f} ms")
        print(f"    流水线加速比:         {sequential_time/total_time if total_time > 0 else 0:.2f}x")
        print(f"    并行效率提升:         {parallel_efficiency:.1f}%\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分布式异步 Checkpoint 测试')
    parser.add_argument('--rank', type=int, required=True, help='进程 rank (0 or 1)')
    parser.add_argument('--world-size', type=int, default=2, help='总进程数')
    
    args = parser.parse_args()
    
    # 设置多进程启动模式
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    
    print(f"\n{'='*80}")
    print(f"分布式异步 Checkpoint 测试 - 原生 Torch 方式")
    print(f"{'='*80}")
    print(f"Rank: {args.rank} / {args.world_size}")
    print(f"\n实现方式:")
    print(f"  • torch.to('cpu') - GPU→CPU 传输")
    print(f"  • pickle - tensor 序列化")
    print(f"  • dist.send/recv - 分布式通信")
    print(f"  • 三进程异步流水线 - D2H | Exchange | XOR")
    
    # 运行测试
    stats, total_time = run_test(args.rank, args.world_size)
    
    print(f"\n[Rank {args.rank}] 测试完成！\n")


if __name__ == "__main__":
    main()

