#!/usr/bin/env python3
"""
对比测试：直接GPU到CPU传输 vs 预先创建的进程池传输
使用静态函数避免multiprocessing问题
"""

from telnetlib import TSPEED
import torch
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
import argparse
from typing import List, Tuple, Any

# 设置multiprocessing启动方法为spawn
mp.set_start_method("spawn", force=True)

class PoolTensors:
    def __init__(self, num_workers: int = None, device_id: int = 0):
        self.num_workers = num_workers or 2
        self.device_id = device_id
        self.workers: List[mp.Process] = []
        self.task_queue: mp.Queue = None
        self.result_queue: mp.Queue = None
        self.running = True
        
        self._init_worker_pool()
    
    @staticmethod
    def worker_function(task_queue: mp.Queue, result_queue: mp.Queue, device_id: int):
        """静态工作函数 - 在子进程中执行GPU到CPU传输"""
        try:
            # if torch.cuda.is_available():
            #     torch.cuda.set_device(device_id)
            #     device = f'cuda:{device_id}'
            # else:
            #     device = 'cpu'
            
            while True:
                try:
                    # 获取任务
                    task_data = task_queue.get()
                    if task_data is None:  # 结束信号
                        break
                    # task_id, tensor_shape, tensor_data = task_data
                    
                    # # 重建tensor到GPU
                    # gpu_tensor = torch.tensor(tensor_data, device=device).reshape(tensor_shape)
                    
                    # GPU到CPU传输
                    start_time = time.time()
                    for tensor in task_data:
                        cpu_tensor = tensor.cpu()
                    end_time = time.time()
                    
                    transfer_time = end_time - start_time
                    result_queue.put((0, transfer_time))
                    print("pool transfer time: ", transfer_time)
                    
                    # # 清理
                    # del gpu_tensor, cpu_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Worker error: {e}")
                    break
                    
        except Exception as e:
            print(f"Worker initialization error: {e}")

        
    def _init_worker_pool(self):
        """Initialize the worker process pool."""
        ctx = mp.get_context('spawn')
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        
        for worker_id in range(self.num_workers):
            worker = ctx.Process(
                target=PoolTensors.worker_function,
                args=(self.task_queue, self.result_queue, self.device_id)
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"已创建 {self.num_workers} 个工作进程")
    
    def add_tensor(self, gpu_tensor: List[torch.Tensor], task_id: int = 0) -> Tuple[int, float]:
        """将tensor发送给工作进程处理"""
        if not self.running:
            raise RuntimeError("进程池已关闭")
        
        # # 将GPU tensor转换为可序列化的数据
        # tensor_shape = gpu_tensor.shape
        # tensor_data = gpu_tensor.cpu().numpy().flatten().tolist()
        
        # 发送任务
        self.task_queue.put((gpu_tensor))
        
        # 获取结果
        result_task_id, transfer_time = self.result_queue.get()
        return result_task_id, transfer_time
    
    def close(self):
        """关闭进程池"""
        if not self.running:
            return
            
        self.running = False
        
        # 发送结束信号
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # 等待所有进程结束
        for worker in self.workers:
            worker.join()
        
        print("进程池已关闭")

def test_direct_transfer(tensor_size, num_iterations, device_id):
    """测试1：直接GPU到CPU传输"""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = f'cuda:{device_id}'
    else:
        print("CUDA不可用，使用CPU模拟")
        device = 'cpu'
    
    times = []
    for i in range(num_iterations):
        # 创建tensor
        gpu_tensors = []
        for i in range(100):
            gpu_tensor = torch.randn(tensor_size, device=device)
            gpu_tensors.append(gpu_tensor)
        # 测量传输时间
        start_time = time.time()
        for tensor in gpu_tensors:
            cpu_tensor = tensor.cpu()
        end_time = time.time()
        
        times.append(end_time - start_time)
        
        print("direct transfer time ", end_time-start_time)
        # 清理
        del gpu_tensor, cpu_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return times

def test_pool_transfer(tensor_size, num_iterations, device_id, pool):
    """测试2：进程池传输"""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = f'cuda:{device_id}'
    else:
        print("CUDA不可用，使用CPU模拟")
        device = 'cpu'
    
    times = []
    for i in range(num_iterations):
        # 创建tensor
        # 创建tensor
        gpu_tensors = []
        for i in range(100):
            gpu_tensor = torch.randn(tensor_size, device=device)
            gpu_tensors.append(gpu_tensor)
        # 测量总时间（包括进程间通信和传输）
        start_time = time.time()
        task_id, transfer_time = pool.add_tensor(gpu_tensors, i)
        end_time = time.time()
        
        total_time = end_time - start_time
        times.append(total_time)
        
        # 清理
        del gpu_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return times

def main():
    parser = argparse.ArgumentParser(description='GPU到CPU传输性能对比测试')
    parser.add_argument('--tensor_size', type=int, nargs=2, default=[1, 4096, 1024], 
                       help='Tensor大小 (height, width)')
    parser.add_argument('--num_iterations', type=int, default=20, 
                       help='测试迭代次数')
    parser.add_argument('--device_id', type=int, default=0, 
                       help='GPU设备ID')
    parser.add_argument('--num_workers', type=int, default=1, 
                       help='进程池工作进程数量')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GPU到CPU传输性能对比测试")
    print("="*70)
    print(f"Tensor大小: {args.tensor_size}")
    print(f"迭代次数: {args.num_iterations}")
    print(f"GPU设备: {args.device_id}")
    print(f"工作进程数: {args.num_workers}")
    print()
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA可用，使用GPU: {args.device_id}")
    else:
        print("CUDA不可用，将使用CPU进行测试")
    
    # 预热
    print("预热中...")
    warmup_times = test_direct_transfer(args.tensor_size, 3, args.device_id)
    print(f"预热完成，平均时间: {np.mean(warmup_times):.6f}s")
    
    # 创建预先的进程池
    print("创建预先的进程池...")
    pool = PoolTensors(args.num_workers, args.device_id)
    
    try:
        # 测试1：直接传输
        print("测试1: 当前进程直接GPU到CPU传输...")
        direct_times = test_direct_transfer(args.tensor_size, args.num_iterations, args.device_id)
        
        # 测试2：预先创建的进程池传输
        print("测试2: 通过预先创建的进程池传输...")
        pool_times = test_pool_transfer(args.tensor_size, args.num_iterations, args.device_id, pool)
    
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n关闭进程池...")
        pool.close()
        print("测试完成!")

if __name__ == "__main__":
    main()
