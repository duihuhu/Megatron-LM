#!/usr/bin/env python3
"""
Comparison test: direct GPU-to-CPU transfer vs. transfer through a preinitialized process pool
Use a static function to avoid multiprocessing issues
"""

from telnetlib import TSPEED
import torch
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
import argparse
from typing import List, Tuple, Any

# Set the multiprocessing start method to spawn
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
        """Static worker function that performs GPU-to-CPU transfers in a subprocess"""
        try:
            # if torch.cuda.is_available():
            #     torch.cuda.set_device(device_id)
            #     device = f'cuda:{device_id}'
            # else:
            #     device = 'cpu'

            while True:
                try:
                    # Get a task
                    task_data = task_queue.get()
                    if task_data is None:  # termination signal
                        break
                    # task_id, tensor_shape, tensor_data = task_data

                    # # Rebuild tensor on GPU
                    # gpu_tensor = torch.tensor(tensor_data, device=device).reshape(tensor_shape)

                    # GPU-to-CPU transfer
                    start_time = time.time()
                    for tensor in task_data:
                        cpu_tensor = tensor.cpu()
                    end_time = time.time()

                    transfer_time = end_time - start_time
                    result_queue.put((0, transfer_time))
                    print("pool transfer time: ", transfer_time)

                    # # Clean up
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

        print(f"Created {self.num_workers} worker processes")

    def add_tensor(self, gpu_tensor: List[torch.Tensor], task_id: int = 0) -> Tuple[int, float]:
        """Send tensors to a worker process"""
        if not self.running:
            raise RuntimeError("Process pool is closed")

        # # Convert the GPU tensor to serializable data
        # tensor_shape = gpu_tensor.shape
        # tensor_data = gpu_tensor.cpu().numpy().flatten().tolist()

        # Send task
        self.task_queue.put((gpu_tensor))

        # Get result
        result_task_id, transfer_time = self.result_queue.get()
        return result_task_id, transfer_time

    def close(self):
        """Close the process pool"""
        if not self.running:
            return

        self.running = False

        # Send termination signals
        for _ in range(self.num_workers):
            self.task_queue.put(None)

        # Wait for all processes to finish
        for worker in self.workers:
            worker.join()

        print("Process pool is closed")

def test_direct_transfer(tensor_size, num_iterations, device_id):
    """Test 1: Direct GPU-to-CPU Transfer"""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = f'cuda:{device_id}'
    else:
        print("CUDA unavailable; simulating with CPU")
        device = 'cpu'

    times = []
    for i in range(num_iterations):
        # Create tensors
        gpu_tensors = []
        for i in range(100):
            gpu_tensor = torch.randn(tensor_size, device=device)
            gpu_tensors.append(gpu_tensor)
        # Measure transfer time
        start_time = time.time()
        for tensor in gpu_tensors:
            cpu_tensor = tensor.cpu()
        end_time = time.time()

        times.append(end_time - start_time)

        print("direct transfer time ", end_time-start_time)
        # Clean up
        del gpu_tensor, cpu_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return times

def test_pool_transfer(tensor_size, num_iterations, device_id, pool):
    """Test 2: Process Pool Transfer"""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = f'cuda:{device_id}'
    else:
        print("CUDA unavailable; simulating with CPU")
        device = 'cpu'

    times = []
    for i in range(num_iterations):
        # Create tensors
        # Create tensors
        gpu_tensors = []
        for i in range(100):
            gpu_tensor = torch.randn(tensor_size, device=device)
            gpu_tensors.append(gpu_tensor)
        # Measure Total time (including interprocess communication and transfer)
        start_time = time.time()
        task_id, transfer_time = pool.add_tensor(gpu_tensors, i)
        end_time = time.time()

        total_time = end_time - start_time
        times.append(total_time)

        # Clean up
        del gpu_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return times

def main():
    parser = argparse.ArgumentParser(description='GPU-to-CPU Transfer Performance Comparison')
    parser.add_argument('--tensor_size', type=int, nargs=2, default=[1, 4096, 1024],
                       help='Tensor size (height, width)')
    parser.add_argument('--num_iterations', type=int, default=20,
                       help='Number of test iterations')
    parser.add_argument('--device_id', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of process-pool workers')

    args = parser.parse_args()

    print("="*70)
    print("GPU-to-CPU Transfer Performance Comparison")
    print("="*70)
    print(f"Tensor size: {args.tensor_size}")
    print(f"Iterations: {args.num_iterations}")
    print(f"GPU device: {args.device_id}")
    print(f"Worker processes: {args.num_workers}")
    print()

    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA available, using GPU: {args.device_id}")
    else:
        print("CUDA unavailable; testing on CPU")

    # Warm up
    print("Warming up...")
    warmup_times = test_direct_transfer(args.tensor_size, 3, args.device_id)
    print(f"Warm-up completed, average time: {np.mean(warmup_times):.6f}s")

    # Creating the preinitialized process pool
    print("Creating the preinitialized process pool...")
    pool = PoolTensors(args.num_workers, args.device_id)

    try:
        # Test1: directlytransfer
        print("Test1: Direct GPU-to-CPU transfer in the current process...")
        direct_times = test_direct_transfer(args.tensor_size, args.num_iterations, args.device_id)

        # Test2: transfer through a preinitialized process pool
        print("Test2: Transfer through the preinitialized process pool...")
        pool_times = test_pool_transfer(args.tensor_size, args.num_iterations, args.device_id, pool)

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nClose the process pool...")
        pool.close()
        print("Test completed!")

if __name__ == "__main__":
    main()
