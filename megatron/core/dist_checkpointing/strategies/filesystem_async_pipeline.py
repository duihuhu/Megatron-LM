# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Enhanced Storage writer for PyT Distributed format with corrected grouped pipeline checkpoint functionality."""

import threading
import math
from typing import List, Tuple, Optional, Callable, Dict, Any
from time import time
import logging
import torch
from torch import multiprocessing as mp
from functools import partial

from .filesystem_async import FileSystemWriterAsync, WriteBucket, _disable_gc, _process_memory

from .async_utils import AsyncRequest

logger = logging.getLogger(__name__)

# Type definitions for grouped pipeline functionality
TensorGroup = List[WriteBucket]
GroupedWriteBuckets = List[TensorGroup]


class PipelineStage:
    """Represents a pipeline stage for checkpoint data transfer with correct sequencing."""
    
    def __init__(self, stage_id: int, group_data: TensorGroup):
        self.stage_id = stage_id
        self.group_data = group_data
        self.gpu_to_cpu_done = threading.Event()
        self.cpu_to_disk_done = threading.Event()
        self.preloaded_data: Optional[List[WriteBucket]] = None
        self.error: Optional[Exception] = None


class FileSystemWriterAsyncPipeline(FileSystemWriterAsync):
    """
    Enhanced FileSystemWriterAsync with corrected grouped pipeline checkpoint functionality.
    
    This class extends the base FileSystemWriterAsync to support:
    1. Grouping checkpoint parameters into multiple tensor groups
    2. Sequential GPU to CPU transfer between groups (Stage N waits for Stage N-1)
    3. Pipeline CPU->disk write of Stage N overlaps with GPU->CPU transfer of Stage N+1
    4. CPU->disk write of Stage N waits for CPU->disk write of Stage N-1 to complete
    5. Markers to distinguish between new and old functionality
    """
    
    def __init__(self, *args, enable_pipeline: bool = False, num_tensor_groups: int = 2, **kwargs):
        """
        Initialize the pipeline-enabled writer.
        
        Args:
            enable_pipeline: Flag to enable pipeline functionality (marker for new vs old)
            num_tensor_groups: Number of groups to split tensors into for pipeline processing
            *args, **kwargs: Arguments passed to parent FileSystemWriterAsync
        """
        super().__init__(*args, **kwargs)
        self.enable_pipeline = enable_pipeline
        self.num_tensor_groups = max(2, num_tensor_groups)  # At least 2 groups for pipeline
        
    def get_save_function_and_args(self) -> Tuple[Optional[Callable], Optional[Callable], List]:
        """
        Enhanced version that returns pipeline or standard functions based on marker.
        
        Returns: None (if nothing to write) or tuple of:
            1) the function that saves the data (pipeline or standard)
            2) the function that stages GPU tensors (pipeline or standard) 
            3) arguments to the save function
        """
        if not self.write_buckets:
            return None, None, []
            
        transform_list = [self.transforms] if hasattr(self, "transforms") else []

        if self.enable_pipeline:
            # Return pipeline-enabled functions
            return (
                partial(self.write_preloaded_data_multiproc_pipeline, transform_list, self.use_msc),
                partial(self.preload_tensors_pipeline, self.write_buckets, True, self.num_tensor_groups),
                [torch.distributed.get_rank(), self.write_buckets, self.results_queue, self.num_tensor_groups],
            )
        else:
            # Return standard functions (existing behavior)
            return super().get_save_function_and_args()

    @staticmethod
    def group_write_buckets_by_size(write_buckets: List[WriteBucket], num_groups: int) -> GroupedWriteBuckets:
        """
        Group write buckets into multiple groups based on data paths and tensor sizes for optimal pipeline processing.
        """
        if num_groups <= 1:
            return [write_buckets]

        print(f"DEBUG: num_groups={num_groups}, write_buckets_count={len(write_buckets)}")
        
        # 按数据路径分组，并合并相同路径的bytes_data和tensor_data
        path_data = {}
        for bucket in write_buckets:
            file_path, file_name, (bytes_data, tensor_data) = bucket
            if file_path not in path_data:
                path_data[file_path] = {
                    'file_path': file_path,
                    'file_name': file_name,
                    'bytes_data': [],
                    'tensor_data': [],
                    'total_size': 0
                }
            
            # 合并bytes_data
            path_data[file_path]['bytes_data'].extend(bytes_data)
            
            # 合并tensor_data
            path_data[file_path]['tensor_data'].extend(tensor_data)
            
            # 计算这个bucket的数据量
            bucket_size = 0
            for item, data in bytes_data:
                if isinstance(data, bytes):
                    bucket_size += len(data)
                else:
                    bucket_size += 1
            for item, tensor in tensor_data:
                if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                    bucket_size += tensor.numel() * tensor.element_size()
                elif hasattr(tensor, 'nbytes'):
                    bucket_size += tensor.nbytes
                else:
                    bucket_size += 1
            
            path_data[file_path]['total_size'] += bucket_size
        
        print(f"DEBUG: found {len(path_data)} unique paths: {list(path_data.keys())}")
        
        # 按路径数据量排序（最大的在前）
        sorted_paths = sorted(path_data.items(), key=lambda x: x[1]['total_size'], reverse=True)
        
        # 计算每个路径应该分配多少个组
        total_data_size = sum(data['total_size'] for data in path_data.values())
        groups = [[] for _ in range(num_groups)]
        group_sizes = [0] * num_groups
        
        for path, data in sorted_paths:
            path_size = data['total_size']
            # 计算这个路径应该分配多少个组
            path_groups_count = max(1, round((path_size / total_data_size) * num_groups))
            
            print(f"DEBUG: path={path}, size={path_size}, allocated_groups={path_groups_count}")
            
            # 如果该路径只需要1个组，直接分配到最小的组
            if path_groups_count == 1:
                min_group_idx = min(range(num_groups), key=lambda i: group_sizes[i])
                # 创建一个新的WriteBucket，包含合并后的数据
                merged_bucket = (
                    data['file_path'],
                    data['file_name'],
                    (data['bytes_data'], data['tensor_data'])
                )
                groups[min_group_idx].append(merged_bucket)
                group_sizes[min_group_idx] += path_size
                print(f"DEBUG: path={path} -> group[{min_group_idx}], added_size={path_size}")
            else:
                # 如果该路径需要多个组，将数据项分配到多个组
                # 创建所有数据项的列表，按大小排序
                all_items = []
                
                # 添加bytes_data项
                for item, data_item in data['bytes_data']:
                    item_size = len(data_item) if isinstance(data_item, bytes) else 1
                    all_items.append(('bytes', item, data_item, item_size))
                
                # 添加tensor_data项
                for item, tensor in data['tensor_data']:
                    if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                        item_size = tensor.numel() * tensor.element_size()
                    elif hasattr(tensor, 'nbytes'):
                        item_size = tensor.nbytes
                    else:
                        item_size = 1
                    all_items.append(('tensor', item, tensor, item_size))
                
                # 按大小排序（最大的在前）
                all_items.sort(key=lambda x: x[3], reverse=True)
                
                # 使用贪心算法分配数据项到各组
                group_items = [[] for _ in range(path_groups_count)]
                group_sizes_temp = [0] * path_groups_count
                
                for item_type, item, data_item, item_size in all_items:
                    # 找到当前总大小最小的组
                    min_group_idx = min(range(path_groups_count), key=lambda i: group_sizes_temp[i])
                    group_items[min_group_idx].append((item_type, item, data_item, item_size))
                    group_sizes_temp[min_group_idx] += item_size
                
                # 将分配好的数据项转换为WriteBucket并分配到实际组中
                for i, items in enumerate(group_items):
                    if items:  # 如果这个组有数据项
                        # 找到当前总大小最小的实际组
                        min_group_idx = min(range(num_groups), key=lambda i: group_sizes[i])
                        
                        # 分离bytes_data和tensor_data
                        group_bytes_data = []
                        group_tensor_data = []
                        group_size = 0
                        
                        for item_type, item, data_item, item_size in items:
                            if item_type == 'bytes':
                                group_bytes_data.append((item, data_item))
                            else:  # tensor
                                group_tensor_data.append((item, data_item))
                            group_size += item_size
                        
                        # 创建新的WriteBucket
                        merged_bucket = (
                            data['file_path'],
                            data['file_name'],
                            (group_bytes_data, group_tensor_data)
                        )
                        groups[min_group_idx].append(merged_bucket)
                        group_sizes[min_group_idx] += group_size
                        
                        print(f"DEBUG: path={path}, group_items={len(items)} -> group[{min_group_idx}], added_size={group_size}")
        
        print(f"DEBUG: final groups={[len(group) for group in groups]}")
        print(f"DEBUG: final group_sizes={group_sizes}")
        
        # 返回所有组
        return groups

    @staticmethod
    def preload_tensors_pipeline(
        write_buckets: List[WriteBucket], 
        non_blocking: bool = True, 
        num_groups: int = 2
    ) -> List[WriteBucket]:
        """
        Pipeline-enabled tensor preloading with grouped sequential GPU->CPU transfer.
        
        This function groups tensors and performs sequential GPU->CPU transfer within each group,
        which is then used in the pipeline with CPU->disk writing.
        
        Args:
            write_buckets: List of WriteBucket objects
            non_blocking: Enable non-blocking D2H transfer
            num_groups: Number of groups for pipeline processing
            
        Returns:
            List of preloaded WriteBucket objects with grouped metadata
        """
        start_time = time()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.debug(f"rank: {rank}, starting pipeline preload with {num_groups} groups")
        
        # Group the write buckets
        grouped_buckets = FileSystemWriterAsyncPipeline.group_write_buckets_by_size(
            write_buckets, num_groups
        )
        
        logger.debug(f"rank: {rank}, created {len(grouped_buckets)} groups for pipeline processing")
                
        result = []
        
        # Process each group sequentially for GPU->CPU transfer
        for group_idx, group_buckets in enumerate(grouped_buckets):
            group_start = time()
            
            group_result = []
            
            # Sequential GPU->CPU transfer within the group
            for bucket in group_buckets:
                file_name, storage_key, (bytes_data, tensor_data) = bucket
                
                # Transfer tensor data from GPU to CPU
                transferred_tensor_data = []
                for item, tensor in tensor_data:
                    if tensor.device.type != "cpu":
                        cpu_tensor = tensor.to("cpu", non_blocking=non_blocking)
                        transferred_tensor_data.append((item, cpu_tensor))
                    else:
                        transferred_tensor_data.append((item, tensor))
                        
                group_result.append((file_name, storage_key, (bytes_data, transferred_tensor_data)))
            
            # Synchronize GPU operations for this group
            if non_blocking and torch.cuda.is_available():
                torch.cuda.synchronize()
                
            group_end = time()
            logger.debug(f"rank: {rank}, group {group_idx} D2H completed in {group_end - group_start:.3f}s")
            
            result.extend(group_result)
        
        end_time = time()
        logger.debug(f"rank: {rank}, pipeline preload completed in {end_time - start_time:.3f}s")
        
        return result

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_multiproc_pipeline(
        transform_list: List[Any],
        use_msc: bool,
        rank: int,
        write_buckets: List[WriteBucket],
        global_results_queue: mp.Queue,
        num_groups: int = 2,
        preload_fn: Optional[Callable] = None,
    ) -> None:
        """
        CORRECTED Pipeline-enabled multiprocess data writing with proper stage sequencing.
        """
        logger = logging.getLogger(__name__)
        start_time = time()
        logger.debug(f"rank: {rank}, starting CORRECTED pipeline multiprocess write")
        
        # Re-group the write buckets for pipeline processing
        grouped_buckets = FileSystemWriterAsyncPipeline.group_write_buckets_by_size(
            write_buckets, num_groups
        )
        print("grouped_buckets ", len(grouped_buckets))
        
        # 记录原始路径到结果的映射关系
        original_path_mapping = {}
        for i, bucket in enumerate(write_buckets):
            file_path, file_name, (bytes_data, tensor_data) = bucket
            if file_path not in original_path_mapping:
                original_path_mapping[file_path] = []
            original_path_mapping[file_path].append(i)
        
        print(f"DEBUG: original_path_mapping={original_path_mapping}")
        
        # Initialize pipeline stages
        stages = []
        for i, group_data in enumerate(grouped_buckets):
            stage = PipelineStage(i, group_data)
            stages.append(stage)
        
        write_results_or_exc: Dict[str, Any] = {}
        pipeline_threads = []
        
        def gpu_to_cpu_worker(stage: PipelineStage):
            """Worker function for GPU->CPU transfer with proper sequencing."""
            try:
                stage_start = time()
                
                # CORRECTED: Wait for previous stage's GPU->CPU transfer to complete
                if stage.stage_id > 0:
                    prev_stage = stages[stage.stage_id - 1]
                    prev_stage.gpu_to_cpu_done.wait()
                
                print(f"rank: {rank}, stage {stage.stage_id} start for GPU->CPU", time())
                # 在子进程中执行GPU->CPU转换（合并preload_tensors_pipeline的功能）
                stage.preloaded_data = []
                for bucket in stage.group_data:
                    file_name, storage_key, (bytes_data, tensor_data) = bucket
                    
                    # Transfer tensor data from GPU to CPU
                    transferred_tensor_data = []
                    for item, tensor in tensor_data:
                        if tensor.device.type != "cpu":
                            cpu_tensor = tensor.to("cpu", non_blocking=True)
                            transferred_tensor_data.append((item, cpu_tensor))
                        else:
                            transferred_tensor_data.append((item, tensor))
                            
                    stage.preloaded_data.append((file_name, storage_key, (bytes_data, transferred_tensor_data)))
                
                # Synchronize GPU operations for this stage
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                stage.gpu_to_cpu_done.set()
                
                stage_end = time()
                print(f"rank: {rank}, stage {stage.stage_id} GPU->CPU completed in {stage_end - stage_start:.3f}s", time())
                
            except Exception as e:
                stage.error = e
                stage.gpu_to_cpu_done.set()
                logger.error(f"rank: {rank}, stage {stage.stage_id} GPU->CPU failed: {e}")
        
        def cpu_to_disk_worker(stage: PipelineStage):
            """Worker function for CPU->disk write with proper sequencing."""
            try:
                # Wait for current stage's GPU->CPU transfer to complete
                stage.gpu_to_cpu_done.wait()
                
                # CORRECTED: Wait for previous stage's CPU->disk write to complete
                if stage.stage_id > 0:
                    prev_stage = stages[stage.stage_id - 1]
                    logger.debug(f"rank: {rank}, stage {stage.stage_id} waiting for stage {stage.stage_id - 1} CPU->disk")
                    prev_stage.cpu_to_disk_done.wait()
                    logger.debug(f"rank: {rank}, stage {stage.stage_id} stage {stage.stage_id - 1} CPU->disk completed")
            
                print(f"rank: {rank}, stage {stage.stage_id} start for CPU->Disk ", time())

                stage_start = time()
                
                # Use the existing write_preloaded_data function for actual disk writing
                ctx = mp.get_context("fork")
                local_results_queue = ctx.Queue()
                count_queue = ctx.JoinableQueue()
                processes = []
                
                for i, write_bucket in enumerate(stage.preloaded_data):
                    count_queue.put(i)
                    
                    kwargs = {
                        "local_proc_idx": f"{stage.stage_id}_{i}",
                        "write_bucket": write_bucket,
                        "results_queue": local_results_queue,
                        "count_queue": count_queue,
                        "use_fsync": True,
                    }
                    
                    if use_msc:
                        kwargs["use_msc"] = use_msc
                    
                    p = ctx.Process(
                        target=partial(FileSystemWriterAsync.write_preloaded_data, transform_list),
                        kwargs=kwargs,
                    )
                    processes.append(p)
                    p.start()
                
                # Wait for all processes to complete
                count_queue.join()
                
                # Collect results
                stage_results = {}
                for _ in range(len(stage.preloaded_data)):
                    proc_idx, local_results_or_exc = local_results_queue.get()
                    if isinstance(local_results_or_exc, Exception):
                        raise local_results_or_exc
                    stage_results[proc_idx] = local_results_or_exc
                
                for p in processes:
                    p.join()
                    
                write_results_or_exc[f"stage_{stage.stage_id}"] = stage_results
                stage.cpu_to_disk_done.set()
                
                stage_end = time()
                logger.debug(f"rank: {rank}, stage {stage.stage_id} CPU->disk completed in {stage_end - stage_start:.3f}s")
                print(f"rank: {rank}, stage {stage.stage_id} CPU->disk completed in {stage_end - stage_start:.3f}s", time())
                
            except Exception as e:
                stage.error = e
                write_results_or_exc[f"stage_{stage.stage_id}"] = e
                stage.cpu_to_disk_done.set()
                logger.error(f"rank: {rank}, stage {stage.stage_id} CPU->disk failed: {e}")
        
        # Start pipeline execution with CORRECTED sequencing
        try:
            # Start all GPU->CPU transfers (they will wait for previous stages)
            for stage in stages:
                thread = threading.Thread(target=gpu_to_cpu_worker, args=(stage,))
                thread.start()
                pipeline_threads.append(thread)
                logger.debug(f"rank: {rank}, started GPU->CPU thread for stage {stage.stage_id}")
            
            # Start all CPU->disk writes (they will wait for proper dependencies)
            for stage in stages:
                thread = threading.Thread(target=cpu_to_disk_worker, args=(stage,))
                thread.start()
                pipeline_threads.append(thread)
                logger.debug(f"rank: {rank}, started CPU->disk thread for stage {stage.stage_id}")
            
            # Wait for all pipeline stages to complete
            for thread in pipeline_threads:
                thread.join()
            
            # Check for any errors in the pipeline stages
            for stage in stages:
                if stage.error:
                    write_results_or_exc = stage.error
                    break
                    
        except Exception as e:
            logger.error(f"rank: {rank}, pipeline execution failed: {e}")
            write_results_or_exc = e
        
        # 修改：按相同路径合并结果
        if isinstance(write_results_or_exc, dict):
            # 按路径收集结果
            path_results = {}
            
            for stage_key, stage_results in write_results_or_exc.items():
                if isinstance(stage_results, dict):
                    for proc_idx, proc_results in stage_results.items():
                        # 从stage_data中获取对应的bucket信息
                        stage_idx = int(stage_key.split('_')[1])
                        stage_data = stages[stage_idx].group_data
                        
                        # 正确解析proc_idx
                        if isinstance(proc_idx, str) and '_' in proc_idx:
                            # proc_idx格式为 "stage_idx_bucket_idx"
                            bucket_idx = int(proc_idx.split('_')[1])
                        else:
                            # 如果proc_idx是整数，直接使用
                            bucket_idx = int(proc_idx)
                        
                        if bucket_idx < len(stage_data):
                            bucket = stage_data[bucket_idx]
                            file_path, file_name, (bytes_data, tensor_data) = bucket
                            
                            if file_path not in path_results:
                                path_results[file_path] = []
                            path_results[file_path].extend(proc_results)
            
            print(f"DEBUG: path_results={list(path_results.keys())}")
            
            # 将结果映射回原始write_buckets格式
            remapped_results = {}
            for path, results in path_results.items():
                if path in original_path_mapping:
                    # 将结果分配给该路径对应的所有原始bucket索引
                    for bucket_idx in original_path_mapping[path]:
                        remapped_results[bucket_idx] = results
            
            print(f"DEBUG: remapped_results keys={list(remapped_results.keys())}")
            write_results_or_exc = remapped_results
        
        # Put final results in global queue (same format as standard method)
        global_results_queue.put(write_results_or_exc)
        
        end_time = time()
        logger.debug(f"rank: {rank}, CORRECTED pipeline multiprocess write completed in {end_time - start_time:.3f}s")


# Integration marker functions for backward compatibility

def is_pipeline_enabled(writer_or_request) -> bool:
    """
    Check if pipeline functionality is enabled for a writer or request.
    
    This function serves as a marker to distinguish between old and new functionality.
    
    Args:
        writer_or_request: FileSystemWriterAsync instance or AsyncRequest object
        
    Returns:
        True if pipeline is enabled, False otherwise
    """
    if hasattr(writer_or_request, 'enable_pipeline'):
        return writer_or_request.enable_pipeline
    return False

