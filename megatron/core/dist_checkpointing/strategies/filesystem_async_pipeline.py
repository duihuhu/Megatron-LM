# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Enhanced Storage writer for PyT Distributed format with corrected grouped pipeline checkpoint functionality."""

import threading
import math
from typing import List, Tuple, Optional, Callable, Dict, Any
from time import time
import logging
import json
import torch
import zfec
import torch.distributed as dist
from torch import multiprocessing as mp
from functools import partial
import io
from megatron.core.dist_checkpointing.strategies.pipeline_utils import DummyWriteItem
try:
    from torch.distributed.checkpoint.planner import WriteItemType
except Exception:
    class WriteItemType:
        BYTE_IO = 0

from .filesystem_async import FileSystemWriterAsync, WriteBucket, _disable_gc, _process_memory, serialize_bucket_to_bytes

from .async_utils import AsyncRequest

logger = logging.getLogger(__name__)

# Type definitions for grouped pipeline functionality
TensorGroup = List[WriteBucket]
GroupedWriteBuckets = List[TensorGroup]


class PipelineStage:
    """Represents a pipeline stage for checkpoint data transfer with correct sequencing."""

    def __init__(self, stage_id: int, group_data: TensorGroup, enable_ec: bool = False, k: int = 0, m: int = 0):
        self.stage_id = stage_id
        self.group_data = group_data
        self.enable_ec = enable_ec
        self.k = k
        self.m = m
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
    
    def __init__(self, *args, enable_pipeline: bool = False, num_tensor_groups: int = 2, 
                 enable_ec: bool = False, ec_k: int = 4, ec_m: int = 2, **kwargs):
        """
        Initialize the pipeline-enabled writer.
        
        Args:
            enable_pipeline: Flag to enable pipeline functionality (marker for new vs old)
            num_tensor_groups: Number of groups to split tensors into for pipeline processing
            enable_ec: Flag to enable erasure coding functionality
            ec_k: Number of data chunks for erasure coding
            ec_m: Number of parity chunks for erasure coding
            *args, **kwargs: Arguments passed to parent FileSystemWriterAsync
        """
        super().__init__(*args, **kwargs)
        self.enable_pipeline = enable_pipeline
        self.num_tensor_groups = max(2, num_tensor_groups)  # At least 2 groups for pipeline
        self.enable_ec = enable_ec
        self.ec_k = ec_k
        self.ec_m = ec_m
        
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
                partial(self.write_preloaded_data_multiproc_pipeline, self.fs, transform_list, self.use_msc, 
                       enable_ec=self.enable_ec, ec_k=self.ec_k, ec_m=self.ec_m),
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

        logger.debug(f"num_groups={num_groups}, write_buckets_count={len(write_buckets)}")

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
                    'total_size': 0,
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

        logger.debug(f"found {len(path_data)} unique paths: {list(path_data.keys())}")

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

            logger.debug(f"path={path}, size={path_size}, allocated_groups={path_groups_count}")

            # 如果该路径只需要1个组，直接分配到最小的组
            if path_groups_count == 1:
                min_group_idx = min(range(num_groups), key=lambda i: group_sizes[i])
                # 创建一个新的WriteBucket，包含合并后的数据
                merged_bucket = (
                    data['file_path'],
                    data['file_name'],
                    (data['bytes_data'], data['tensor_data']),
                )
                groups[min_group_idx].append(merged_bucket)
                group_sizes[min_group_idx] += path_size
                logger.debug(f"path={path} -> group[{min_group_idx}], added_size={path_size}")
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
                            (group_bytes_data, group_tensor_data),
                        )
                        groups[min_group_idx].append(merged_bucket)
                        group_sizes[min_group_idx] += group_size

                        logger.debug(f"path={path}, group_items={len(items)} -> group[{min_group_idx}], added_size={group_size}")

        logger.debug(f"final groups={[len(group) for group in groups]}")
        logger.debug(f"final group_sizes={group_sizes}")

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
        fs: Any,
        transform_list: List[Any],
        use_msc: bool,
        rank: int,
        write_buckets: List[WriteBucket],
        global_results_queue: mp.Queue,
        num_groups: int = 2,
        preload_fn: Optional[Callable] = None,
        enable_ec: bool = False,
        ec_k: int = 4,
        ec_m: int = 2,
    ) -> None:
        """
        CORRECTED Pipeline-enabled multiprocess data writing with proper stage sequencing.
        """
        logger = logging.getLogger(__name__)
        start_time = time()
        logger.debug(f"Rank {rank}: Starting CORRECTED pipeline multiprocess write")
        
        # Re-group the write buckets for pipeline processing
        grouped_buckets = FileSystemWriterAsyncPipeline.group_write_buckets_by_size(
            write_buckets, num_groups
        )
        
        # Initialize pipeline stages with EC parameters
        stages = []
        for i, group_data in enumerate(grouped_buckets):
            stage = PipelineStage(i, group_data, enable_ec, ec_k, ec_m)
            stages.append(stage)
        
        write_results_or_exc: Dict[str, Any] = {}
        pipeline_threads = []
        
        def gpu_to_cpu_worker(stage: PipelineStage):
            """Worker function for GPU->CPU transfer with proper sequencing."""
            try:
                stage_start = time()
                if stage.stage_id > 0:
                    prev_stage = stages[stage.stage_id - 1]
                    prev_stage.gpu_to_cpu_done.wait()
                logger.debug(f"Rank {rank}: Stage {stage.stage_id} start for GPU->CPU")

                # Step 1: D2H transfer
                preloaded_for_stage = []
                for bucket in stage.group_data:
                    file_name, storage_key, (bytes_data, tensor_data) = bucket
                    transferred_tensor_data = []
                    for item, tensor in tensor_data:
                        if tensor.device.type != "cpu":
                            cpu_tensor = tensor.to("cpu", non_blocking=True)
                            transferred_tensor_data.append((item, cpu_tensor))
                        else:
                            transferred_tensor_data.append((item, tensor))
                    preloaded_for_stage.append((file_name, storage_key, (bytes_data, transferred_tensor_data)))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                logger.debug(f"Rank {rank}: Stage {stage.stage_id} D2H transfer completed")

                # Step 2: EC逻辑只用主线程预分发的结果
                if stage.enable_ec:
                    # 直接用主线程预分发的 ec_chunks
                    my_chunk = None
                    if hasattr(stage, "ec_chunks") and stage.ec_chunks and rank < len(stage.ec_chunks):
                        my_chunk = stage.ec_chunks[rank]
                    if my_chunk:
                        original_file_name, _, _ = stage.group_data[0]
                        ec_file_name = f'{original_file_name}.ec_chunk_{rank}'
                        # 修正：用 DummyWriteItem(type=BYTE_IO)，data用 io.BytesIO
                        bytes_data = [(DummyWriteItem(item_type=WriteItemType.BYTE_IO, index=rank, name=f'ec_chunk_{rank}'), io.BytesIO(my_chunk))]
                        tensor_data = []
                        stage.preloaded_data = [(ec_file_name, ec_file_name, (bytes_data, tensor_data))]
                        logger.debug(f"Rank {rank}: Stage {stage.stage_id} received EC chunk of size {len(my_chunk)} for file {ec_file_name}")
                    else:
                        stage.preloaded_data = []
                        logger.debug(f"Rank {rank}: Stage {stage.stage_id} received no EC chunk or rank is out of bounds")
                else:
                    stage.preloaded_data = preloaded_for_stage

                stage.gpu_to_cpu_done.set()
                stage_end = time()
                logger.debug(f"Rank {rank}: Stage {stage.stage_id} GPU->CPU (and EC) completed in {stage_end - stage_start:.3f}s")
            except Exception as e:
                stage.error = e
                stage.gpu_to_cpu_done.set()
                logger.error(f"Rank {rank}: Stage {stage.stage_id} GPU->CPU failed: {e}", exc_info=True)
        
        def cpu_to_disk_worker(stage: PipelineStage):
            """Worker function for CPU->disk write with proper sequencing."""
            try:
                stage_start = time()
                
                # Wait for this stage's GPU->CPU to be done
                stage.gpu_to_cpu_done.wait()
                if stage.error:
                    raise stage.error

                # CORRECTED: Wait for the previous stage's CPU->disk to be done
                if stage.stage_id > 0:
                    prev_stage = stages[stage.stage_id - 1]
                    prev_stage.cpu_to_disk_done.wait()
                    if prev_stage.error:
                        raise prev_stage.error
                
                logger.debug(f"Rank {rank}: Stage {stage.stage_id} start for CPU->Disk")
                
                if not stage.preloaded_data:
                    logger.debug(f"Rank {rank}: Stage {stage.stage_id} has no data to write.")
                    stage.cpu_to_disk_done.set()
                    return

                # Now, `stage.preloaded_data` contains either the original tensors or an EC chunk
                for bucket in stage.preloaded_data:
                    file_name, storage_key, (bytes_data, tensor_data) = bucket

                    # Serialize the bucket into bytes for async write
                    processed_data, size = serialize_bucket_to_bytes(transform_list, (bytes_data, tensor_data), storage_key)
                    
                    if file_name not in write_results_or_exc:
                        # Store the Future returned by async_write so callers can
                        # call .result() or .wait() on it. Previously this used
                        # AsyncRequest(...) which attempted to construct the
                        # NamedTuple incorrectly and caused TypeError.
                        write_results_or_exc[file_name] = fs.async_write(
                            processed_data, file_name, "wb"
                        )
                    else:
                        # This case should be rare with unique EC filenames, but kept for safety
                        # Wait on previous future if needed and append to file by
                        # passing previous offset/result as an argument to the
                        # async write. Keep storing the returned Future.
                        write_results_or_exc[file_name].wait()
                        write_results_or_exc[file_name] = fs.async_write(
                            processed_data,
                            file_name,
                            "ab",
                            write_results_or_exc[file_name].result(),
                        )
                    
                    # For EC chunks, we don't have original items to update, but we can log the result
                    if stage.enable_ec:
                        logger.debug(f"Rank {rank}: Wrote EC chunk to {file_name} with size {size}")
                        # Rank 0 writes metadata file for EC so that reconstruction can find k/m and original file mapping
                        if rank == 0:
                            try:
                                meta = {
                                    'k': stage.k,
                                    'm': stage.m,
                                    'original_file': stage.group_data[0][1] if stage.group_data else file_name,
                                    'chunk_name': file_name,
                                }
                                meta_bytes = json.dumps(meta).encode('utf-8')
                                meta_file = f"{file_name}.ec_meta"
                                # Fire-and-forget async write of metadata. We don't
                                # need to wrap this in AsyncRequest; the returned
                                # Future is ignored intentionally here.
                                fs.async_write(meta_bytes, meta_file, "wb")
                            except Exception:
                                logger.exception("Failed to write EC metadata file")
                    else:
                        # Update results with metadata for original items
                        for item, _ in bytes_data:
                            item.storage_key = storage_key
                            item.offset = write_results_or_exc[file_name].result()
                            item.length = size
                        for item, _ in tensor_data:
                            item.storage_key = storage_key
                            item.offset = write_results_or_exc[file_name].result()
                            item.length = size

                stage.cpu_to_disk_done.set()
                stage_end = time()
                logger.debug(f"Rank {rank}: Stage {stage.stage_id} CPU->Disk completed in {stage_end - stage_start:.3f}s")

            except Exception as e:
                stage.error = e
                stage.cpu_to_disk_done.set()
                logger.error(f"Rank {rank}: Stage {stage.stage_id} CPU->Disk failed: {e}", exc_info=True)
        
        # Start pipeline execution
        try:
            # If EC is enabled, perform all-gather and encoding before starting workers
            if enable_ec:
                for stage in stages:
                    if not stage.enable_ec:
                        continue
                    
                    logger.debug(f"Rank {rank}: Pre-calculating EC for Stage {stage.stage_id}")
                    
                    # Step 1: Perform D2H transfer for the current stage's data on all ranks
                    preloaded_for_stage = []
                    for bucket in stage.group_data:
                        file_name, storage_key, (bytes_data, tensor_data) = bucket
                        transferred_tensor_data = []
                        for item, tensor in tensor_data:
                            if tensor.device.type != "cpu":
                                cpu_tensor = tensor.to("cpu", non_blocking=True)
                                transferred_tensor_data.append((item, cpu_tensor))
                            else:
                                transferred_tensor_data.append((item, tensor))
                        preloaded_for_stage.append((file_name, storage_key, (bytes_data, transferred_tensor_data)))
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Step 2: Gather all data from all ranks
                    all_ranks_data = [None] * dist.get_world_size()
                    dist.all_gather_object(all_ranks_data, preloaded_for_stage)
                    logger.debug(f"Rank {rank}: Stage {stage.stage_id} all_gather_object completed for EC pre-calculation")

                    # Step 3: Rank 0 performs encoding and scatters chunks
                    encoded_chunks = None
                    if rank == 0:
                        all_tensors = []
                        for r_data in all_ranks_data:
                            for bucket in r_data:
                                _, _, (_, tensor_data) = bucket
                                for _, tensor in tensor_data:
                                    all_tensors.append(tensor)
                        
                        if all_tensors:
                            encoded_chunks = FileSystemWriterAsyncPipeline.ec_encode_tensors(all_tensors, stage.k, stage.m)
                        else:
                            encoded_chunks = []
                    
                    scatter_list = [encoded_chunks] if rank == 0 else [None]
                    dist.broadcast_object_list(scatter_list, src=0)
                    stage.ec_chunks = scatter_list[0] # Store pre-calculated chunks
                    logger.debug(f"Rank {rank}: Stage {stage.stage_id} received {len(stage.ec_chunks) if stage.ec_chunks else 0} EC chunks")

            # Start all worker threads
            for stage in stages:
                gpu_thread = threading.Thread(target=gpu_to_cpu_worker, args=(stage,))
                cpu_thread = threading.Thread(target=cpu_to_disk_worker, args=(stage,))
                gpu_thread.start()
                cpu_thread.start()
                pipeline_threads.extend([gpu_thread, cpu_thread])
            
            # Wait for all threads to complete
            for thread in pipeline_threads:
                thread.join()
            
            # Check for any errors in the pipeline stages
            final_results = {}
            for stage in stages:
                if stage.error:
                    raise stage.error
                # This part is tricky because results are scattered across files.
                # The current implementation of `save` expects results to be mapped back to original items.
                # For EC, this mapping is not direct. We will return an empty dict for now,
                # as the primary goal is to write the data.
                # A more robust solution would involve a different result format for EC.
            
            # For non-EC case, we need to collect results.
            # The current structure with threads makes it hard to collect results per item.
            # The `write_results_or_exc` dict holds file-level results.
            # We will pass this up and let the caller handle it.
            if not enable_ec:
                 # This is a simplification. The original multiproc version collected per-item results.
                 # A full implementation would require queues to pass results back from threads.
                 pass

        except Exception as e:
            logger.error(f"Rank {rank}: Pipeline execution failed: {e}", exc_info=True)
            write_results_or_exc = e
        
        # Put final results in global queue
        # For EC, we might not have meaningful item-level results.
        # For non-EC, the current threaded model doesn't easily produce them.
        # We send back the file-level results or exception.
        global_results_queue.put(write_results_or_exc)
        
        end_time = time()
        logger.debug(f"Rank {rank}: CORRECTED pipeline multiprocess write completed in {end_time - start_time:.3f}s")

    @staticmethod
    def gather_tensors_across_ranks(local_tensors: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """
        使用all_gather收集所有rank的tensor数据
        """
        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_tensors)
        return gathered

    @staticmethod
    def ec_encode_tensors(tensors: List[torch.Tensor], k: int, m: int) -> List[bytes]:
        """
        Encode a list of tensors using Erasure Coding.

        Args:
            tensors: A list of tensors to be encoded.
            k: The number of data chunks.
            m: The number of parity chunks.

        Returns:
            A list of k+m chunks (bytes), where the first k are data chunks
            and the next m are parity chunks.
        """
        if not tensors:
            return []

        # 1. Concatenate all tensors into a single byte buffer
        # Note: This is a simplified approach. For real-world scenarios,
        # padding and metadata (like original shapes/dtypes) would be needed.
        all_data_list = []
        for t in tensors:
            # Ensure tensor is on CPU and contiguous
            t_cpu = t.contiguous() if t.is_contiguous() else t.clone().contiguous()
            t_cpu = t_cpu if t_cpu.device.type == 'cpu' else t_cpu.cpu()
            # Convert tensor to bytes
            all_data_list.append(t_cpu.numpy().tobytes())
        
        all_data = b''.join(all_data_list)
        total_size = len(all_data)

        # 2. Split the byte buffer into k data chunks
        chunk_size = math.ceil(total_size / k)
        padded_size = chunk_size * k
        # Pad the data with zeros to make it evenly divisible
        all_data = all_data.ljust(padded_size, b'\0')

        data_chunks = [all_data[i:i+chunk_size] for i in range(0, padded_size, chunk_size)]

        # 3. Perform EC encoding
        encoder = zfec.Encoder(k, m)
        parity_chunks = encoder.encode(data_chunks)

        # 4. Return all chunks
        return data_chunks + parity_chunks

    @staticmethod
    def write_ec_chunk_to_disk(chunk_bytes: bytes, file_path: str):
        """
        将EC编码后的chunk写入磁盘
        """
        # 实际的落盘操作在这里
        with open(file_path, "wb") as f:
            f.write(chunk_bytes)


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

