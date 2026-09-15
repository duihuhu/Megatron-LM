"""
Distributed Asynchronous Checkpoint System - Native Torch Implementation
==========================================

Scenario:
- two independently launched programs (rank 0 and rank 1)
- each program has three GPU tensors
- pipeline architecture:
  1. Main process: responsible for GPU→CPU transfer and serialization (D2H stage)
  2. Exchange Worker (subprocess): use send/recv with the other program to exchange data
  3. XOR Worker (subprocess): Perform bitwise XOR on tensor data
- Process three tensors through a pipeline

Implementation:
- the main process performs D2H in the foreground (avoid passing across processes GPU tensor)
- Two subprocesses process subsequent stages in parallel
- Detailed serialization performance statistics
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
# Worker functions (subprocesses) - native Torch approach
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
    Exchange Worker - native Torch approach
    use torch.distributed send/recv exchangedata

    metrics:
    - deserialize_time: CPU tensor deserialization time
    - comm_time: distributed communicationtime
    - serialize_time: result serialization time
    - total_time: Total time
    """
    # The subprocess must set environment variables and initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Initialize distributed environment
    try:
        dist.init_process_group(
            backend='gloo',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        print(f"[Rank {rank}] Exchange Worker (torch): distributed environment initialized successfully")
    except Exception as e:
        print(f"[Rank {rank}] Exchange Worker (torch): distributed initialization failed: {e}")
        stats_queue.put(('ready', 'exchange_worker_torch_failed'))
        return

    # Send a ready signal
    stats_queue.put(('ready', 'exchange_worker_torch'))
    print(f"[Rank {rank}] Exchange Worker (torch): ready")

    # Wait for all workers are ready
    ready_event.wait()

    peer_rank = 1 - rank  # exchange peer
    while True:
        try:
            item = input_queue.get()
            if item is None:
                break

            tensor_idx, name, cpu_tensor_serialized = item
            total_start = time.perf_counter()

            # ===== 1. deserialize the CPU tensor =====
            deser_start = time.perf_counter()
            cpu_tensor = pickle.loads(cpu_tensor_serialized)
            deser_time = time.perf_counter() - deser_start

            # ===== 2. distributed communication =====
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

            # ===== 3. serialize the result =====
            ser_start = time.perf_counter()
            received_serialized = pickle.dumps(received_tensor)
            ser_time = time.perf_counter() - ser_start

            total_time = time.perf_counter() - total_start

            # Pass to the next stage
            output_queue.put((tensor_idx, name, cpu_tensor_serialized, received_serialized))

            # Send detailed statistics
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

    # Clean up the distributed environment
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[Rank {rank}] Exchange Worker (torch): distributed environment cleaned up")
    except Exception as e:
        print(f"[Rank {rank}] Exchange Worker (torch): Cleanup failed: {e}")


def xor_worker_torch(
    input_queue: mp.Queue,
    stats_queue: mp.Queue,
    rank: int,
    ready_event: mp.Event
):
    """
    XOR Worker - native Torch approach
    Perform bitwise XOR on local and received tensors

    metrics:
    - deserialize_local_time: local tensor deserialization time
    - deserialize_received_time: Receive tensor deserialization time
    - xor_time: XOR computation time
    - total_time: Total time
    """
    # Send a ready signal
    stats_queue.put(('ready', 'xor_worker_torch'))
    print(f"[Rank {rank}] XOR Worker (torch): ready")

    # Wait for all workers are ready
    ready_event.wait()

    while True:
        try:
            item = input_queue.get(timeout=0.1)
            if item is None:
                break

            tensor_idx, name, local_serialized, received_serialized = item
            total_start = time.perf_counter()

            # ===== 1. deserialize the local tensor =====
            deser_local_start = time.perf_counter()
            local_tensor = pickle.loads(local_serialized)
            deser_local_time = time.perf_counter() - deser_local_start

            # ===== 2. Deserialize the received tensor =====
            deser_received_start = time.perf_counter()
            received_tensor = pickle.loads(received_serialized)
            deser_received_time = time.perf_counter() - deser_received_start

            # ===== 3. bitwise XOR =====
            xor_start = time.perf_counter()
            local_int = local_tensor.view(torch.int32)
            received_int = received_tensor.view(torch.int32)
            xor_result = torch.bitwise_xor(local_int, received_int)
            xor_time = time.perf_counter() - xor_start

            total_time = time.perf_counter() - total_start

            # Send detailed statistics
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
# Pipeline Management Class
# ============================================================================

class AsyncPipelineTorch:
    """Asynchronous pipeline - native Torch approach (the main process is responsible for D2H)"""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

        # Two queues (main process handles D2H, does not require d2h_queue)
        self.exchange_queue = mp.Queue()
        self.xor_queue = mp.Queue()
        self.stats_queue = mp.Queue()

        # synchronization event (ensure all workers are ready before starting)
        self.ready_event = mp.Event()

        # Two subprocesses (the main process is responsible for D2H)
        self.exchange_process = None
        self.xor_process = None

        # statistics
        self.stats = {
            'd2h_times': [],
            'exchange_times': [],
            'xor_times': [],
            'd2h_detailed': [],       # D2H detailed statistics (main process)
            'exchange_detailed': [],  # Exchange detailed statistics
            'xor_detailed': []        # XOR detailed statistics
        }

    def start_pipeline(self):
        """Starting two worker subprocesses (the main process is responsible for D2H)"""
        print(f"[Rank {self.rank}] Starting two worker subprocesses...")

        # Only start Exchange and XOR workers (D2H handled by the main process)
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

        # Wait for all workers are ready
        print(f"[Rank {self.rank}] Wait for all workers to initialize...")
        ready_count = 0
        expected_workers = 2  # only two subprocesses

        while ready_count < expected_workers:
            try:
                while not self.stats_queue.empty():
                    stat = self.stats_queue.get_nowait()
                    if len(stat) == 2 and stat[0] == 'ready':
                        ready_count += 1
                        print(f"[Rank {self.rank}]   {stat[1]} ready ({ready_count}/{expected_workers})")
            except:
                pass
            time.sleep(0.1)

        if ready_count == expected_workers:
            print(f"[Rank {self.rank}] ✓ all workers are ready, the main process begins D2H processing")
            self.ready_event.set()  # Notify all workers that processing can begin
        else:
            print(f"[Rank {self.rank}] ✗ Warning: only {ready_count}/{expected_workers} workers are ready")
            self.ready_event.set()  # continue anyway

    def submit_tensor(self, tensor_idx: int, name: str, gpu_tensor: torch.Tensor):
        """
        Submit tensor to the pipeline (main process execution D2H)

        Executed in the Main process:
        1. GPU → CPU transfer
        2. CPU tensor serialization
        3. put the tensor into exchange_queue
        """
        print(f"[Rank {self.rank}] Main process: begin processing tensor {tensor_idx}")
        total_start = time.perf_counter()

        # ===== 1. D2H transfer (the main process has a complete CUDA context)=====
        d2h_start = time.perf_counter()
        cpu_tensor = gpu_tensor.to('cpu', non_blocking=False)
        torch.cuda.synchronize()
        d2h_time = time.perf_counter() - d2h_start
        print(f"[Rank {self.rank}] Main process: D2H Transfer completed, elapsed time {d2h_time*1000:.2f} ms")

        # ===== 2. serialize the CPU tensor =====
        ser_start = time.perf_counter()
        cpu_tensor_serialized = pickle.dumps(cpu_tensor.contiguous())
        ser_time = time.perf_counter() - ser_start
        print(f"[Rank {self.rank}] Main process: serialization completed, elapsed time {ser_time*1000:.2f} ms")

        total_time = time.perf_counter() - total_start

        # ===== 3. put the tensor into exchange_queue (for processing by the Exchange Worker)=====
        self.exchange_queue.put((tensor_idx, name, cpu_tensor_serialized))
        print(f"[Rank {self.rank}] Main process: tensor {tensor_idx} submitted to Exchange Worker")

        # Record D2H stage details (main process execution)
        self.stats['d2h_detailed'].append((tensor_idx, {
            'd2h_time': d2h_time,
            'serialize_time': ser_time,
            'total_time': total_time,
            'tensor_size_bytes': cpu_tensor.element_size() * cpu_tensor.nelement()
        }))
        self.stats['d2h_times'].append((tensor_idx, total_time))

    def stop_pipeline(self):
        """Stop the pipeline (only two subprocesses need to be stopped)"""
        # Send termination signals
        self.exchange_queue.put(None)
        self.xor_queue.put(None)

        # Wait for subprocesses to exit
        if self.exchange_process:
            self.exchange_process.join(timeout=2.0)
            if self.exchange_process.is_alive():
                self.exchange_process.terminate()
                print(f"[Rank {self.rank}] Exchange Worker was forcefully terminated")

        if self.xor_process:
            self.xor_process.join(timeout=2.0)
            if self.xor_process.is_alive():
                self.xor_process.terminate()
                print(f"[Rank {self.rank}] XOR Worker was forcefully terminated")

    def collect_stats(self):
        """Collect statistical information"""
        while not self.stats_queue.empty():
            try:
                stat = self.stats_queue.get_nowait()
                # Process messages in different formats
                if len(stat) == 2:
                    # Ready message: ('ready', worker_name)
                    continue
                elif len(stat) == 3:
                    stage, idx, data = stat
                    # Statistics message: (stage, idx, elapsed) or (stage_detailed, idx, dict)
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
# Test Functions
# ============================================================================

def run_test(rank: int, world_size: int):
    """
    Run the asynchronous pipeline test

    Args:
        rank: Rank of the current process (0 or 1)
        world_size: Total number of processes (2)
    """
    mode_name = "Native Torch asynchronous pipeline"

    print(f"\n{'='*80}")
    print(f"[Rank {rank}] {mode_name}")
    print(f"{'='*80}")

    if not torch.cuda.is_available():
        print("Error: CUDA support is required")
        return

    # Create three GPU tensors
    num_tensors = 3
    tensor_shapes = [
        (512, 512),
        (1024, 1024),
        (256, 256)
    ]

    print(f"\n[Rank {rank}] Create {num_tensors} GPU tensors")
    gpu_tensors = []
    total_size = 0
    for i, shape in enumerate(tensor_shapes):
        tensor = torch.randn(*shape, device='cuda', dtype=torch.float32)
        gpu_tensors.append(tensor)
        total_size += tensor.element_size() * tensor.nelement()
        size_mb = tensor.element_size() * tensor.nelement() / 1024 / 1024
        print(f"  Tensor {i}: {shape} ({size_mb:.2f} MB)")

    total_size_mb = total_size / 1024 / 1024
    print(f"  total size: {total_size_mb:.2f} MB")

    # Create the pipeline
    pipeline = AsyncPipelineTorch(rank, world_size)

    # Start the pipeline
    pipeline.start_pipeline()

    # Submit tensors (pipeline processing)
    print(f"\n[Rank {rank}] Submit tensors to the pipeline...")
    start_time = time.time()

    for i, gpu_tensor in enumerate(gpu_tensors):
        name = f"tensor_{i}"
        pipeline.submit_tensor(i, name, gpu_tensor)
        print(f"  [Rank {rank}] Submit Tensor {i}")

    # Wait for completion (use a more robust waiting mechanism)
    print(f"\n[Rank {rank}] Wait for pipeline completion...")
    expected_completions = num_tensors  # expected number of completed tensors
    completed_tensors = set()
    max_wait_time = 15.0
    check_start = time.time()

    while len(completed_tensors) < expected_completions:
        # Collect statistical information
        pipeline.collect_stats()

        # Check XOR stage-completed tensor (final stage)
        xor_completed = {idx for idx, _ in pipeline.stats['xor_times']}
        completed_tensors = xor_completed

        # Check for timeout
        if time.time() - check_start > max_wait_time:
            print(f"[Rank {rank}] ⚠️ Wait timed out, completed {len(completed_tensors)}/{expected_completions} tensors")
            break

        if len(completed_tensors) < expected_completions:
            time.sleep(0.1)

    # Collect statistics one final time
    time.sleep(0.5)
    pipeline.collect_stats()

    total_time = time.time() - start_time
    print(f"[Rank {rank}] ✓ pipeline processing completed ({len(completed_tensors)}/{expected_completions} tensors)")

    # Collect statistics
    stats = pipeline.collect_stats()

    # Stop the pipeline
    pipeline.stop_pipeline()

    # Print detailed statistics
    print_detailed_stats(rank, stats, total_time)

    return stats, total_time


def print_detailed_stats(rank: int, stats: Dict, total_time: float):
    """Print detailed performance statistics"""
    print(f"\n[Rank {rank}] ================== Asynchronous Pipeline Performance Statistics ==================")
    print(f"  Total time: {total_time*1000:.2f} ms\n")

    # ===== D2H stage details =====
    if stats.get('d2h_detailed'):
        print(f"  [D2H stage details] (main process execution)")
        d2h_detailed = sorted(stats['d2h_detailed'], key=lambda x: x[0])

        total_d2h = 0
        total_ser = 0
        total_stage = 0

        for idx, details in d2h_detailed:
            print(f"    Tensor {idx}:")
            print(f"      D2H transfer:   {details['d2h_time']*1000:>8.2f} ms")
            print(f"      Serialization:    {details['serialize_time']*1000:>8.2f} ms")
            print(f"      Total:      {details['total_time']*1000:>8.2f} ms")

            total = details['total_time']
            ser_percent = details['serialize_time'] / total * 100
            print(f"      Serialization share: {ser_percent:.1f}%")
            print(f"      Tensor size: {details['tensor_size_bytes']/1024/1024:.2f} MB\n")

            total_d2h += details['d2h_time']
            total_ser += details['serialize_time']
            total_stage += details['total_time']

        # Average statistics
        n = len(d2h_detailed)
        ser_overhead = total_ser / total_stage * 100
        print(f"    Average:")
        print(f"      D2H transfer:   {total_d2h/n*1000:>8.2f} ms")
        print(f"      Serialization:    {total_ser/n*1000:>8.2f} ms")
        print(f"      Total:      {total_stage/n*1000:>8.2f} ms")
        print(f"      Serialization share: {ser_overhead:.1f}%  ← key metric")
        print(f"      ✅ advantage: the main process holds the GPU tensor directly, with no deserialization overhead!\n")

    # ===== Exchange stage details =====
    if stats.get('exchange_detailed'):
        print(f"  [Exchange stage details]")
        exchange_detailed = sorted(stats['exchange_detailed'], key=lambda x: x[0])

        total_deser = 0
        total_comm = 0
        total_ser = 0
        total_stage = 0

        for idx, details in exchange_detailed:
            print(f"    Tensor {idx}:")
            print(f"      DeSerialization:  {details['deserialize_time']*1000:>8.2f} ms")
            print(f"      communication:      {details['comm_time']*1000:>8.2f} ms")
            print(f"      Serialization:    {details['serialize_time']*1000:>8.2f} ms")
            print(f"      Total:      {details['total_time']*1000:>8.2f} ms")

            total = details['total_time']
            ser_time = details['deserialize_time'] + details['serialize_time']
            ser_percent = ser_time / total * 100
            print(f"      Serialization share: {ser_percent:.1f}%\n")

            total_deser += details['deserialize_time']
            total_comm += details['comm_time']
            total_ser += details['serialize_time']
            total_stage += details['total_time']

        # Average statistics
        n = len(exchange_detailed)
        ser_overhead = (total_deser + total_ser) / total_stage * 100
        print(f"    Average:")
        print(f"      DeSerialization:  {total_deser/n*1000:>8.2f} ms")
        print(f"      communication:      {total_comm/n*1000:>8.2f} ms")
        print(f"      Serialization:    {total_ser/n*1000:>8.2f} ms")
        print(f"      Total:      {total_stage/n*1000:>8.2f} ms")
        print(f"      Serialization share: {ser_overhead:.1f}%  ← key metric\n")

    # ===== XOR stage details =====
    if stats.get('xor_detailed'):
        print(f"  [XOR stage details]")
        xor_detailed = sorted(stats['xor_detailed'], key=lambda x: x[0])

        total_deser_local = 0
        total_deser_recv = 0
        total_xor = 0
        total_stage = 0

        for idx, details in xor_detailed:
            print(f"    Tensor {idx}:")
            print(f"      Deserialization (local): {details['deserialize_local_time']*1000:>8.2f} ms")
            print(f"      Deserialization (received): {details['deserialize_received_time']*1000:>8.2f} ms")
            print(f"      XOR computation:       {details['xor_time']*1000:>8.2f} ms")
            print(f"      Total:          {details['total_time']*1000:>8.2f} ms")

            total = details['total_time']
            deser_time = details['deserialize_local_time'] + details['deserialize_received_time']
            ser_percent = deser_time / total * 100
            print(f"      Serialization share: {ser_percent:.1f}%\n")

            total_deser_local += details['deserialize_local_time']
            total_deser_recv += details['deserialize_received_time']
            total_xor += details['xor_time']
            total_stage += details['total_time']

        # Average statistics
        n = len(xor_detailed)
        ser_overhead = (total_deser_local + total_deser_recv) / total_stage * 100
        print(f"    Average:")
        print(f"      Deserialization (local): {total_deser_local/n*1000:>8.2f} ms")
        print(f"      Deserialization (received): {total_deser_recv/n*1000:>8.2f} ms")
        print(f"      XOR computation:       {total_xor/n*1000:>8.2f} ms")
        print(f"      Total:          {total_stage/n*1000:>8.2f} ms")
        print(f"      Serialization share: {ser_overhead:.1f}%  ← key metric\n")

    # ===== Overall Serialization Overhead Analysis =====
    print(f"  [Overall Serialization Overhead Analysis]")

    if stats.get('d2h_detailed') and stats.get('exchange_detailed') and stats.get('xor_detailed'):
        # Compute total serialization time across all stages
        total_serialization = 0
        total_computation = 0
        total_execution = 0

        for idx, details in stats['d2h_detailed']:
            total_serialization += details['serialize_time']  # main process execution, no deserialize
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

        print(f"    total serialization time:   {total_serialization*1000:.2f} ms")
        print(f"    total computation time:     {total_computation*1000:.2f} ms")
        print(f"    total execution time:     {total_execution*1000:.2f} ms")
        print(f"    Serialization share:     {ser_percent_overall:.1f}%  ← 🔑 overall serialization overhead")
        print(f"    computation share:       {total_computation/total_execution*100:.1f}%\n")

    # ===== Pipeline Parallel Efficiency =====
    if stats['d2h_times'] and stats['exchange_times'] and stats['xor_times']:
        d2h_total = sum([t[1] for t in stats['d2h_times']])
        exchange_total = sum([t[1] for t in stats['exchange_times']])
        xor_total = sum([t[1] for t in stats['xor_times']])

        sequential_time = d2h_total + exchange_total + xor_total
        parallel_efficiency = (sequential_time / total_time - 1) * 100 if total_time > 0 else 0

        print(f"  [Pipeline Parallel Efficiency]")
        print(f"    Sequential execution time (estimated): {sequential_time*1000:.2f} ms")
        print(f"    Actual parallel execution time:     {total_time*1000:.2f} ms")
        print(f"    pipeline speedup:         {sequential_time/total_time if total_time > 0 else 0:.2f}x")
        print(f"    parallel efficiency improvement:         {parallel_efficiency:.1f}%\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='distributed asynchronous Checkpoint Test')
    parser.add_argument('--rank', type=int, required=True, help='Process rank (0 or 1)')
    parser.add_argument('--world-size', type=int, default=2, help='Total number of processes')

    args = parser.parse_args()

    # Set the multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass

    print(f"\n{'='*80}")
    print(f"Distributed Asynchronous Checkpoint Test - Native Torch Implementation")
    print(f"{'='*80}")
    print(f"Rank: {args.rank} / {args.world_size}")
    print(f"\nImplementation:")
    print(f"  • torch.to('cpu') - GPU→CPU transfer")
    print(f"  • pickle - tensor serialization")
    print(f"  • dist.send/recv - distributed communication")
    print(f"  • three-stage asynchronous pipeline - D2H | Exchange | XOR")

    # Run the test
    stats, total_time = run_test(args.rank, args.world_size)

    print(f"\n[Rank {args.rank}] Test completed!\n")


if __name__ == "__main__":
    main()

