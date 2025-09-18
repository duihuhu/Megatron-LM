import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

import os
import shutil
import logging
from functools import partial
import signal, sys
from megatron.core.dist_checkpointing.strategies.pipeline_utils import DummyWriteItem
try:
    from torch.distributed.checkpoint.planner import WriteItemType
except Exception:
    class WriteItemType:
        BYTE_IO = 0

# 假设此文件与 filesystem_async_pipeline.py 在同一目录下
# import sys
# sys.path.append('/path/to/your/megatron/core/dist_checkpointing/strategies')
from megatron.core.dist_checkpointing.strategies.filesystem_async_pipeline import FileSystemWriterAsyncPipeline, WriteBucket
from megatron.core.dist_checkpointing.strategies.filesystem_async import _get_write_results_queue
import io


# Simple synchronous future-like wrapper used for test fs.async_write
class DummyFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


# Minimal filesystem adapter that provides async_write used by the pipeline tests.
class DummyFS:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def async_write(self, data, file_path: str, mode: str = "wb", prev_offset: int = None):
        # Ensure parent dir
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        to_write = None
        if hasattr(data, 'read'):
            # file-like
            data.seek(0)
            to_write = data.read()
        elif isinstance(data, bytes):
            to_write = data
        else:
            # try converting
            try:
                to_write = bytes(data)
            except Exception:
                to_write = str(data).encode('utf-8')

        # write to file synchronously
        with open(file_path, mode) as f:
            f.write(to_write)
            offset = f.tell()

        return DummyFuture(offset)

# --- 1. 配置测试参数 ---
WORLD_SIZE = 4  # 模拟的GPU数量
NUM_TENSOR_GROUPS = 2  # 流水线阶段数
ENABLE_EC = True  # 是否开启EC
EC_K = 2  # EC数据块
EC_M = 2  # EC校验块 (K+M 必须等于 WORLD_SIZE)

# 模拟保存的目录
CHECKPOINT_DIR = "/tmp/megatron_standalone_test"


def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '34567'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized on device {torch.cuda.current_device()}")

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def create_dummy_data(rank, device) -> list[WriteBucket]:
    """为每个rank创建一些模拟的待保存数据"""
    write_buckets = []

    # 模拟一个大的权重文件，所有rank都会贡献一部分
    common_path = os.path.join(CHECKPOINT_DIR, "model_weights.bin")
    tensor1 = torch.randn(1024 * 1024, device=device) * (rank + 1) # 2MB
    bucket1 = (common_path, f"layer_{rank}_weight", ([], [(DummyWriteItem(None), tensor1)]))
    write_buckets.append(bucket1)

    # 模拟一个小的、只有rank 0写入的文件
    if rank == 0:
        exclusive_path = os.path.join(CHECKPOINT_DIR, "optimizer.bin")
        tensor2 = torch.ones(512 * 1024, device=device) # 1MB
        bucket2 = (exclusive_path, "optimizer_state", ([], [(DummyWriteItem(None), tensor2)]))
        write_buckets.append(bucket2)
        
    # 模拟另一个大的权重文件
    common_path_2 = os.path.join(CHECKPOINT_DIR, "another_model_weights.bin")
    tensor3 = torch.randn(2 * 1024 * 1024, device=device) * (rank + 1) # 4MB
    bucket3 = (common_path_2, f"layer_{rank}_attention", ([], [(DummyWriteItem(None), tensor3)]))
    write_buckets.append(bucket3)

    return write_buckets

def run_test_for_rank(rank, world_size, num_pipeline_stages, enable_ec, ec_k, ec_m):
    """
    为单个rank运行测试的核心逻辑。
    """
    # --- 1. 初始化 ---
    setup_distributed(rank, world_size)
    print(f"Rank {rank} initialized on device {rank}")

    # --- 2. 清理和准备目录 ---
    if rank == 0:
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dist.barrier() # 确保所有rank都看到目录已创建

    # --- 3. 实例化异步写入器 ---
    writer = FileSystemWriterAsyncPipeline(
        path=CHECKPOINT_DIR,
        enable_pipeline=True,
        num_tensor_groups=num_pipeline_stages,
        enable_ec=enable_ec,
        ec_k=ec_k,
        ec_m=ec_m,
    )

    # --- 4. 创建模拟数据并保存 ---
    dummy_data = create_dummy_data(rank, torch.cuda.current_device())
    print(f"Rank {rank}: Created {len(dummy_data)} data buckets to save.")
    
    # The test environment does not use the external async interface. Instead,
    # drive the pipeline synchronously using the writer's helper functions.
    writer.write_buckets = dummy_data
    writer.results_queue = _get_write_results_queue()

    # Provide a minimal filesystem adapter expected by the pipeline
    writer.fs = DummyFS(CHECKPOINT_DIR)
    # mimic attribute that may be inspected by serializer
    writer.use_msc = False

    save_fn, preload_fn, save_args = writer.get_save_function_and_args()
    # Optionally run the preload function to stage tensors (pipeline uses its own preload internally too)
    if preload_fn:
        try:
            preload_fn()
        except Exception:
            # preload may be a no-op or rely on other state; ignore if it fails here
            pass

    # Call the save function synchronously in this process
    save_fn(*save_args)

    # Try to collect result from the shared results queue (may be None for EC path)
    result = None
    try:
        if writer.results_queue is not None:
            result = writer.results_queue.get(timeout=30)
    except Exception:
        result = None

    print(f"Rank {rank}: Save operation completed with result: {result}")
    
    dist.barrier() # 等待所有rank完成保存

    # --- 6. (可选) 验证文件 ---
    if rank == 0:
        print("\n--- File Validation (Rank 0) ---")
        expected_files = ["model_weights.bin", "another_model_weights.bin", "optimizer.bin"]
        if enable_ec:
            # 如果EC开启，我们期望看到数据块和校验块
            for i in range(ec_k):
                expected_files.append(f"model_weights.bin.ec_chunk_{i}")
                expected_files.append(f"another_model_weights.bin.ec_chunk_{i}")
            for i in range(ec_m):
                expected_files.append(f"model_weights.bin.ec_parity_{i}")
                expected_files.append(f"another_model_weights.bin.ec_parity_{i}")
        
        all_files = os.listdir(CHECKPOINT_DIR)
        print(f"Files found in {CHECKPOINT_DIR}: {all_files}")
        
        # 简单的检查，看预期文件是否都以某种形式存在
        for f_base in ["model_weights.bin", "another_model_weights.bin", "optimizer.bin"]:
             any_match = any(f.startswith(f_base) for f in all_files)
             if any_match:
                 print(f"  [V] Found file/chunks for: {f_base}")
             else:
                 print(f"  [X] Missing file/chunks for: {f_base}")


    # --- 7. 清理 ---
    cleanup_distributed()
    print(f"Rank {rank} finished and cleaned up.")


def _term_handler(signum, frame):
    try:
        cleanup_distributed()
    except Exception:
        pass
    sys.exit(1)

signal.signal(signal.SIGINT, _term_handler)
signal.signal(signal.SIGTERM, _term_handler)

if __name__ == "__main__":
    if WORLD_SIZE > torch.cuda.device_count():
        print(f"Error: WORLD_SIZE ({WORLD_SIZE}) is greater than available GPUs ({torch.cuda.device_count()})")
        exit(1)
        
    if ENABLE_EC and (EC_K + EC_M != WORLD_SIZE):
        print(f"Error: EC config is invalid. K({EC_K}) + M({EC_M}) != WORLD_SIZE({WORLD_SIZE})")
        exit(1)

    print("--- Starting Standalone Pipeline Test ---")
    print(f"World Size: {WORLD_SIZE}, Pipeline Stages: {NUM_TENSOR_GROUPS}, EC: {ENABLE_EC} (k={EC_K}, m={EC_M})")
    
    # 使用mp.spawn启动多进程测试
    global_results_queue = Queue()
    mp.spawn(
        run_test_for_rank,
        args=(WORLD_SIZE, NUM_TENSOR_GROUPS, ENABLE_EC, EC_K, EC_M),
        nprocs=WORLD_SIZE,
        join=True,
        daemon=False,
        start_method="spawn",
    )
    print("--- Test Finished ---")

import logging

logging.basicConfig(
    level=logging.DEBUG,  # 或 INFO
    format='%(asctime)s %(levelname)s %(process)d %(threadName)s %(name)s: %(message)s'
)
