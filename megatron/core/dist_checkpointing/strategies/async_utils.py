# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
This module provides an async utilities which allow to start
a checkpoint save process in the background.
"""
import gc
import io
import logging
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from queue import Empty
from time import sleep, time
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import multiprocessing as mp

from ..utils import debug_time

logger = logging.getLogger(__name__)

# Module-level cache for pair process groups
_pair_process_groups_cache: Dict[int, torch.distributed.ProcessGroup] = {}
_gloo_backend_initialized: bool = False


def _ensure_gloo_backend_available() -> None:
    """Ensure gloo backend is available for CPU tensor communication.
    
    When main process group uses NCCL, we may need to ensure gloo backend
    is properly initialized before creating gloo sub-groups.
    """
    global _gloo_backend_initialized
    
    if _gloo_backend_initialized:
        return
    
    try:
        current_backend = torch.distributed.get_backend()
        logger.debug(f"Current backend: {current_backend}")
        _gloo_backend_initialized = True
    except Exception as e:
        logger.warning(f"Could not verify gloo backend availability: {e}")


def _create_all_pair_process_groups() -> None:
    """Create all pair process groups at once.
    
    IMPORTANT: torch.distributed.new_group requires ALL ranks to call it,
    not just the ranks participating in the group. This function ensures
    all ranks call new_group for all pair groups.
    """
    global _pair_process_groups_cache
    
    if _pair_process_groups_cache:
        return  # Already created
    
    rank = torch.distributed.get_rank()
    
    # Ensure gloo backend is available
    _ensure_gloo_backend_available()
    
    # Define all pairs: rank 0<->2, rank 1<->3
    all_pairs = [[0, 2], [1, 3]]
    
    logger.info(f"rank: {rank}, creating all pair process groups")
    
    # All ranks must call new_group for each pair, even if they're not in that pair
    for pair_ranks in all_pairs:
        group_key = min(pair_ranks)
        try:
            logger.info(f"rank: {rank}, calling new_group for ranks {pair_ranks}")
            # ALL ranks must call this, not just the ranks in pair_ranks
            pair_group = torch.distributed.new_group(ranks=pair_ranks, backend='gloo')
            
            # Only store the group if this rank is in it
            if rank in pair_ranks:
                _pair_process_groups_cache[group_key] = pair_group
                logger.info(f"rank: {rank}, stored pair process group for ranks {pair_ranks}")
            else:
                logger.info(f"rank: {rank}, participated in creating group for ranks {pair_ranks} (not a member)")
        except Exception as e:
            logger.error(f"rank: {rank}, failed to create pair process group for ranks {pair_ranks}: {e}", exc_info=True)
            raise
    
    logger.info(f"rank: {rank}, finished creating all pair process groups")


def get_or_create_pair_process_group(rank: int, paired_rank: int) -> torch.distributed.ProcessGroup:
    """Get or create a process group for paired ranks.
    
    This is a standalone function that can be used without creating a TemporalAsyncCaller instance.
    Uses module-level cache to store process groups.
    
    Args:
        rank: Current rank
        paired_rank: Paired rank to communicate with
        
    Returns:
        ProcessGroup for the pair
    """
    # Create all pair groups if not already created
    # This ensures all ranks participate in creating all groups
    _create_all_pair_process_groups()
    
    # Use the lower rank as the key to retrieve the group
    group_key = min(rank, paired_rank)
    
    if group_key not in _pair_process_groups_cache:
        raise RuntimeError(
            f"rank: {rank}, pair process group for ranks [{min(rank, paired_rank)}, {max(rank, paired_rank)}] not found"
        )
    
    return _pair_process_groups_cache[group_key]


@contextmanager
def _disable_gc():
    """Temporarily disables GC."""
    gc_enabled = gc.isenabled()
    try:
        if gc_enabled:
            gc.disable()
        yield
    finally:
        if gc_enabled:
            gc.enable()


class AsyncRequest(NamedTuple):
    """Represents an async request that needs to be scheduled for execution.

    Args:
        async_fn (Callable, optional): async function to call. None represents noop.
        async_fn_args (Tuple): args to pass to `async_fn`.
        finalize_fns (List[Callable]): list of functions to call to finalize the request.
            These functions will be called synchronously after `async_fn` is done
            *on all ranks*.
        async_fn_kwargs (Tuple): kwargs to pass to `async_fn`.
        preload_fn (Callable): preload function to stage tensors from GPU to Host.
            This should be self-contained with a proper list of arguments with  `partial`.
        is_frozen (Bool): a flag to indicate this async request can be modified or not.
        call_idx (int): index variable used to order async requests for synchronization
                        in preloading and writing tensors on the async caller

    """

    async_fn: Optional[Callable]
    async_fn_args: Tuple
    finalize_fns: List[Callable]
    async_fn_kwargs: Dict = {}
    preload_fn: Callable = None
    is_frozen: bool = False
    call_idx: int = 0

    def add_finalize_fn(self, fn: Callable) -> None:
        """Adds a new finalize function to the request.

        Args:
            fn (Callable): function to add to the async request. This function
                will be called *after* existing finalization functions.

        Returns:
            None
        """
        if self.is_frozen:
            raise RuntimeError('Cannot add finalization functions to a frozen AsyncRequest')
        self.finalize_fns.append(fn)

    def execute_sync(self) -> None:
        """Helper to synchronously execute the request.

        This logic is equivalent to what should happen in case of the async call.
        """
        if self.async_fn is not None:
            self.async_fn(*self.async_fn_args)
        torch.distributed.barrier()
        for finalize_fn in self.finalize_fns:
            finalize_fn()

    def freeze(self) -> 'AsyncRequest':
        """Freezes the async request, disallowing adding new finalization functions.

        Returns:
            AsyncRequest: new async request with all same fields except for the
                `is_frozen` flag.
        """
        return self._replace(is_frozen=True)


class AsyncCaller(ABC):
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    @abstractmethod
    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Schedule `async_req` with some process forking or reusing
           persistent worker

        This method must be called on all ranks.

        Args:
            async_req (AsyncRequest): `AsyncRequest` object containing to
                                       start async process
        """
        raise NotImplementedError("This should be implemented")

    @abstractmethod
    def is_current_async_call_done(self, blocking: bool, no_dist: bool) -> bool:
        """Check if async save is finished on all ranks.

        For semantic correctness, requires rank synchronization in each check.
        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until the call is done
                on all ranks. Otherwise, returns immediately if at least one rank
                is still active. Defaults to False.
            no_dist (bool, Optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.

        Returns:
            bool: True if all ranks are done (immediately of after active wait
                if `blocking` is True), False if at least one rank is still active.

        """
        raise NotImplementedError("This should be implemented")

    def sync_all_async_calls(self, is_alive: int) -> bool:
        """Check if all ranks have completed async checkpoint writing

        Args:
            is_alive (bool): if True, the current async request is not completed

        Returns:
            bool: True if all ranks are done, False if at least one rank is still active.

        """
        ten = torch.tensor([is_alive], dtype=torch.int, device=torch.cuda.current_device())
        torch.distributed.all_reduce(ten)
        return ten[0] == 0

    @abstractmethod
    def close(self):
        """Terminate the async caller at exit of an application or some termination conditions"""
        logger.info(f"AsyncCaller: {torch.distributed.get_rank()}, Destroying Async Caller")

    def __del__(self):
        raise NotImplementedError("This should be implemented")


class TemporalAsyncCaller(AsyncCaller):
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    def __init__(self):
        self.process: Optional[mp.Process] = None
        self.replica_process: Optional[mp.Process] = None
        self.start_time: Optional[float] = None
        
        # Reusable buffers for data exchange to avoid repeated allocations
        self._serialization_buffer: Optional[io.BytesIO] = None
        self._local_tensor_buffer: Optional[torch.Tensor] = None
        self._remote_tensor_buffer: Optional[torch.Tensor] = None
        self._local_array_buffer: Optional[np.ndarray] = None
        self._last_local_size: int = 0
        self._last_remote_size: int = 0
        
        self.pairing_map = {0: 2, 2: 0, 1: 3, 3: 1}

    def calculate_buckets_data_size(self, write_buckets):
        """Calculate data size by traversing write_buckets structure.
        
        Args:
            write_buckets: List of write buckets (file_name, storage_key, (bytes_data, tensor_data))
            
        Returns:
            int: Total data size in bytes
        """
        total_size_bytes = 0
        
        if write_buckets is None:
            return 0
        
        # Traverse each bucket
        for bucket in write_buckets:
            if isinstance(bucket, tuple) and len(bucket) >= 3:
                _, _, (bytes_data, tensor_data) = bucket
                
                # Calculate bytes_data size
                if isinstance(bytes_data, list):
                    for item in bytes_data:
                        if isinstance(item, (bytes, bytearray)):
                            total_size_bytes += len(item)
                        elif isinstance(item, io.BytesIO):
                            total_size_bytes += len(item.getvalue())
                        elif isinstance(item, tuple) and len(item) >= 2:
                            # Handle (key, value) pairs
                            key, value = item[0], item[1]
                            if isinstance(value, (bytes, bytearray)):
                                total_size_bytes += len(value)
                            elif isinstance(value, io.BytesIO):
                                total_size_bytes += len(value.getvalue())
                
                # Calculate tensor_data size
                if isinstance(tensor_data, list):
                    for item_tuple in tensor_data:
                        if isinstance(item_tuple, tuple) and len(item_tuple) >= 2:
                            tensor = item_tuple[1]
                            if isinstance(tensor, torch.Tensor):
                                total_size_bytes += tensor.numel() * tensor.element_size()
        
        return total_size_bytes
    
    def exchange_checkpoint_data(self, write_buckets):
        """Exchange checkpoint data between paired ranks.
        
        Step 1: Calculate local data size by traversing buckets
        Step 2: Exchange data sizes with paired rank
        Step 3: Exchange actual data
        
        Args:
            write_buckets: The checkpoint data to exchange (list of buckets)
            
        Returns:
            Tuple[list, int, int]: (replica_buckets, local_size_bytes, remote_size_bytes)
        """
        rank = torch.distributed.get_rank()
        
        # Pairwise communication: rank 0<->rank 2, rank 1<->rank 3
        paired_rank = self.pairing_map.get(rank, None)
        
        if paired_rank is None or write_buckets is None:
            logger.warning(f"rank: {rank}, no paired rank or no data, skipping exchange")
            return None, 0, 0
        
        # Create or get pair process group
        pair_group = get_or_create_pair_process_group(rank, paired_rank)
        
        # Step 1: Serialize local data first to get actual size
        serialize_start = time()
        
        # Reuse serialization buffer if possible
        if self._serialization_buffer is None:
            self._serialization_buffer = io.BytesIO()
        else:
            # Reset buffer for reuse
            self._serialization_buffer.seek(0)
            self._serialization_buffer.truncate(0)
        
        torch.save(write_buckets, self._serialization_buffer)
        buffer_view = self._serialization_buffer.getbuffer()
        local_size = buffer_view.nbytes
        serialize_time = time() - serialize_start
        logger.info(f"rank: {rank}, serialized local data: {local_size / (1024**2):.2f} MB in {serialize_time:.4f}s")
        
        # Step 2: Exchange actual data sizes using all_gather
        size_exchange_start = time()
        size_tensor = torch.tensor([local_size], dtype=torch.long, device='cpu')
        gathered_sizes = [torch.zeros_like(size_tensor) for _ in range(2)]
        torch.distributed.all_gather(gathered_sizes, size_tensor, group=pair_group)
        
        # Determine paired rank's data size
        pair_ranks = [min(rank, paired_rank), max(rank, paired_rank)]
        my_idx = pair_ranks.index(rank)
        paired_idx = 1 - my_idx
        remote_size = gathered_sizes[paired_idx].item()
        size_exchange_time = time() - size_exchange_start
        
        logger.info(f"rank: {rank}, size exchange took {size_exchange_time:.4f}s, "
                   f"paired rank {paired_rank} data size: {remote_size / (1024**2):.2f} MB")
        
        # Step 3: Exchange actual data using broadcast
        data_exchange_start = time()
        
        # Convert bytes to tensor for communication
        # Reuse or allocate local tensor buffer
        if self._local_tensor_buffer is None or self._local_tensor_buffer.numel() < local_size:
            # Allocate new buffer with some extra space (10% overhead) to reduce reallocations
            # buffer_size = int(local_size * 1.1)
            self._local_tensor_buffer = torch.empty(local_size, dtype=torch.uint8, device='cpu')
            logger.debug(f"rank: {rank}, allocated local tensor buffer: {local_size / (1024**2):.2f} MB")
        
        # Copy data from buffer_view to tensor
        local_array = np.frombuffer(buffer_view, dtype=np.uint8)
        # Use the preallocated buffer (only the needed portion)
        self._local_tensor_buffer.copy_(torch.from_numpy(local_array))
        local_tensor = self._local_tensor_buffer
        data_end_copy = time()
        logger.info(f"rank: {rank}, data end copy took {data_end_copy - data_exchange_start:.4f}s")
        # Reuse or allocate remote tensor buffer
        if self._remote_tensor_buffer is None or self._remote_tensor_buffer.numel() < remote_size:
            # Allocate new buffer with some extra space (10% overhead)
            # buffer_size = int(remote_size * 1.1)
            self._remote_tensor_buffer = torch.empty(remote_size, dtype=torch.uint8, device='cpu')
            logger.debug(f"rank: {rank}, allocated remote tensor buffer: {remote_size / (1024**2):.2f} MB")
        
        # Use the preallocated buffer (only the needed portion)
        remote_tensor = self._remote_tensor_buffer
        
        # Determine the global ranks for src in broadcast
        # pair_ranks is already sorted: [min_rank, max_rank]
        lower_global_rank = pair_ranks[0]
        higher_global_rank = pair_ranks[1]
        
        # Each rank broadcasts its data to the other
        # Use global rank as src (PyTorch will map it to group rank internally)
        if rank == lower_global_rank:
            # I'm the lower rank, broadcast my data first
            torch.distributed.broadcast(local_tensor, src=lower_global_rank, group=pair_group)
            # Then receive from higher rank
            torch.distributed.broadcast(remote_tensor, src=higher_global_rank, group=pair_group)
        else:
            # I'm the higher rank, receive from lower rank first
            torch.distributed.broadcast(remote_tensor, src=lower_global_rank, group=pair_group)
            # Then broadcast my data
            torch.distributed.broadcast(local_tensor, src=higher_global_rank, group=pair_group)

        data_exchange_time = time() - data_end_copy
        logger.info(f"rank: {rank}, data exchange took {data_exchange_time:.4f}s, "
                   f"received {remote_size / (1024**2):.2f} MB from rank {paired_rank}")
                
        # Get remote bytes directly (no deserialization needed)
        remote_bytes = remote_tensor.numpy().tobytes()
        

        # Return remote_bytes instead of deserialized replica_buckets
        # This avoids deserialization overhead and preserves exact size
        return remote_bytes, local_size, remote_size
    
    @_disable_gc()
    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Spawn a process with `async_fn` as the target.

        This method must be called on all ranks.

        Args:
            async_fn (Callable, optional): async function to call. If None,
                no process will be started.
            async_req (AsyncRequest): `AsyncRequest` object containing to
                                       start async process
        """
        if async_req.async_fn is None:
            return  # nothing to do

        async_fn_args = list(async_req.async_fn_args)
        start_sync = time()
        if async_req.preload_fn:
            # If there's a preload_fn in `async_req`, we call this func
            # to do the defined action in `async_req.preload_fn` to
            # stage GPU tensors to its defined destination
            async_fn_args[1] = async_req.preload_fn()
        
        # print("preload_fn preload_fn ")
        
        rank = torch.distributed.get_rank()
        # logger.info(f"EC-CHECK: Synchronizing CUDA")
        torch.cuda.synchronize()
        # logger.info(f"EC-CHECK: CUDA synchronized")
        end_sync = time()
        logger.info(f"rank: {rank}, takes {end_sync - start_sync} to finish D2H ")
        
        # Exchange checkpoint data with paired rank
        # Step 1: Calculate size by traversing buckets
        # Step 2: Exchange sizes
        # Step 3: Exchange actual data
        # exchange_start = time()
        ctx = mp.get_context('fork')
        
        from megatron.training import get_args
        args = get_args()
        
        # Check if we should use optimized version (no serialization)
        use_optimized = getattr(args, 'use_gemini_optimized', False)
        
        if args.use_gemini:
            
            if use_optimized:
                # Use optimized version: preload already completed exchange
                # async_fn_args[1] now contains [local_bucket, remote_bucket]
                logger.info(f"Gemini rank {rank}: Using optimized mode (exchange completed in preload)")
                
                write_buckets = async_fn_args[1]
                
                # Check if we have both local and remote buckets
                if len(write_buckets) == 2:
                    local_bucket = write_buckets[0]
                    remote_bucket = write_buckets[1]
                    
                    # Extract remote buffer and metadata
                    if remote_bucket[1] == 'gemini_optimized_remote':
                        _, _, (bytes_data, _) = remote_bucket
                        
                        remote_metadata = None
                        remote_buffer = None
                        
                        for item in bytes_data:
                            if isinstance(item, tuple) and len(item) == 2:
                                key, value = item
                                if key == 'gemini_metadata':
                                    remote_metadata = value
                                elif key == 'gemini_buffer':
                                    remote_buffer = value
                        
                        if remote_buffer is not None and remote_metadata is not None:
                            # Get replica file path
                            replica_file_path = self._get_replica_file_path([local_bucket], rank)
                            
                            # Serialize metadata
                            metadata_buffer = io.BytesIO()
                            torch.save(remote_metadata, metadata_buffer)
                            remote_metadata_bytes = metadata_buffer.getvalue()
                            
                            # Convert buffer to bytes
                            remote_buffer_bytes = remote_buffer.numpy().tobytes()
                            
                            # Combine: [metadata_size (8 bytes)] + [metadata_bytes] + [buffer_bytes]
                            metadata_size = len(remote_metadata_bytes)
                            header = metadata_size.to_bytes(8, byteorder='little')
                            combined_bytes = header + remote_metadata_bytes + remote_buffer_bytes
                            
                            # Start process to write replica checkpoint
                            self.replica_process = ctx.Process(
                                target=self._write_bytes_to_file,
                                args=(combined_bytes, replica_file_path)
                            )
                            self.replica_process.start()
                            
                            logger.info(
                                f"Gemini rank {rank}: Started replica save process, "
                                f"buffer: {len(remote_buffer_bytes) / (1024**2):.2f} MB, "
                                f"metadata: {metadata_size / 1024:.2f} KB"
                            )
                        else:
                            self.replica_process = None
                            logger.warning(f"Gemini rank {rank}: Failed to extract remote buffer/metadata")
                    else:
                        self.replica_process = None
                        logger.warning(f"Gemini rank {rank}: Remote bucket not in expected format")
                else:
                    self.replica_process = None
                    logger.warning(f"Gemini rank {rank}: Expected 2 buckets, got {len(write_buckets)}")
            else:
                # Use original version with serialization
                logger.info(f"Gemini rank {rank}: Using original exchange (with serialization)")
                remote_bytes, local_size, remote_size = self.exchange_checkpoint_data(async_fn_args[1])
                
                exchange_end = time()
                logger.info(f"rank: {rank}, takes {exchange_end - start_sync} to schedule async ckpt {exchange_end} {start_sync}")

                if remote_bytes is not None:
                    # Get replica file path from original write_buckets
                    replica_file_path = self._get_replica_file_path(async_fn_args[1], rank)

                    # Start process to directly write remote_bytes to file
                    self.replica_process = ctx.Process(
                        target=self._write_bytes_to_file,
                        args=(remote_bytes, replica_file_path)
                    )
                    self.replica_process.start()

                    # logger.info(f"rank: {rank}, started replica checkpoint save process, "
                            # f"will write {remote_size / (1024**2):.2f} MB to {replica_file_path}")
                else:
                    self.replica_process = None
                
        # exchange_end = time()
        # logger.info(f"rank: {rank}, total exchange (size + data) took {exchange_end - exchange_start:.2f}s")

        self.start_time = time()
        
        # Start process to save original checkpoint
        if args.use_gemini:
            # Get original checkpoint file path
            original_file_path = self._get_original_file_path(async_fn_args[1])
            
            # Get global_results_queue from async_fn_args (it's the 3rd element)
            # async_fn_args structure: [rank, write_buckets, global_results_queue]
            global_results_queue = async_fn_args[2] if len(async_fn_args) > 2 else None
            
            if use_optimized:
                # Use optimized version: extract local buffer and metadata from preloaded write_buckets
                logger.info(f"Gemini rank {rank}: Preparing optimized original checkpoint save")
                
                # Extract buffer and metadata from async_fn_args[1] (already preloaded)
                write_buckets = async_fn_args[1]
                
                # Check if we have the expected format: [local_bucket, remote_bucket]
                if len(write_buckets) >= 1:
                    local_bucket = write_buckets[0]
                    
                    # Extract local buffer and metadata
                    if local_bucket[1] == 'gemini_optimized_local':
                        _, _, (bytes_data, _) = local_bucket
                        
                        original_metadata = None
                        original_buffer = None
                        
                        for item in bytes_data:
                            if isinstance(item, tuple) and len(item) == 2:
                                key, value = item
                                if key == 'gemini_metadata':
                                    original_metadata = value
                                elif key == 'gemini_buffer':
                                    original_buffer = value
                        
                        if original_buffer is not None and original_metadata is not None:
                            # Serialize metadata
                            metadata_buffer = io.BytesIO()
                            torch.save(original_metadata, metadata_buffer)
                            original_metadata_bytes = metadata_buffer.getvalue()
                            
                            # Convert buffer to bytes
                            original_buffer_bytes = original_buffer.numpy().tobytes()
                            
                            # Combine: [metadata_size (8 bytes)] + [metadata_bytes] + [buffer_bytes]
                            metadata_size = len(original_metadata_bytes)
                            header = metadata_size.to_bytes(8, byteorder='little')
                            original_bytes = header + original_metadata_bytes + original_buffer_bytes
                            
                            logger.info(
                                f"Gemini rank {rank}: Original checkpoint prepared, "
                                f"buffer: {len(original_buffer_bytes) / (1024**2):.2f} MB, "
                                f"metadata: {metadata_size / 1024:.2f} KB"
                            )
                        else:
                            # Fallback to serialization
                            logger.warning(f"Gemini rank {rank}: Failed to extract buffer/metadata, falling back to serialization")
                            original_buffer = io.BytesIO()
                            torch.save(async_fn_args[1], original_buffer)
                            original_bytes = original_buffer.getvalue()
                    else:
                        # Not in optimized format, fallback to serialization
                        logger.warning(f"Gemini rank {rank}: Local bucket not in expected format, falling back to serialization")
                        original_buffer = io.BytesIO()
                        torch.save(async_fn_args[1], original_buffer)
                        original_bytes = original_buffer.getvalue()
                else:
                    # Empty write_buckets, fallback to serialization
                    logger.warning(f"Gemini rank {rank}: Empty write_buckets, falling back to serialization")
                    original_buffer = io.BytesIO()
                    torch.save(async_fn_args[1], original_buffer)
                    original_bytes = original_buffer.getvalue()
                
                # Start process to write original checkpoint
                self.process = ctx.Process(
                    target=self._write_bytes_to_file_with_queue,
                    args=(original_bytes, original_file_path, len(async_fn_args[1]), global_results_queue, 
                          args.use_gemini, args.use_gemini_optimized)
                )
                self.process.start()
            else:
                # Use original serialization method
                original_buffer = io.BytesIO()
                torch.save(async_fn_args[1], original_buffer)
                original_buffer_view = original_buffer.getbuffer()
                original_bytes = original_buffer_view.tobytes()
                
                # Start process to directly write original_bytes to file
                self.process = ctx.Process(
                    target=self._write_bytes_to_file_with_queue,
                    args=(original_bytes, original_file_path, len(async_fn_args[1]), global_results_queue,
                          args.use_gemini, False)  # use_gemini=True but use_gemini_optimized=False
                )
                self.process.start()
        else:
            # Use original async function for non-gemini mode
            self.process = ctx.Process(
                target=async_req.async_fn, args=async_fn_args, kwargs=async_req.async_fn_kwargs
            )
            self.process.start()
        
        # init_time = time()
        # logger.info(f"rank: {rank}, takes {init_time - start_sync} to schedule async ckpt ", init_time, " ", start_sync)
    
    def _get_original_file_path(self, write_buckets):
        """Get original checkpoint file path from write_buckets.
        
        Args:
            write_buckets: List of write buckets (file_name, storage_key, data)
            
        Returns:
            str: Original file path
        """
        import os
        
        if not write_buckets or len(write_buckets) == 0:
            raise ValueError("write_buckets is empty, cannot determine file path")
        
        # Get the first bucket's file name (assuming all buckets are in the same directory)
        first_bucket = write_buckets[0]
        if isinstance(first_bucket, tuple) and len(first_bucket) >= 3:
            file_name, _, _ = first_bucket
            return file_name
        else:
            raise ValueError(f"Invalid bucket format: {first_bucket}")
    
    def _get_replica_file_path(self, write_buckets, rank):
        """Get replica file path from write_buckets by adding _replica_on_rank{id} suffix.
        
        Args:
            write_buckets: List of write buckets (file_name, storage_key, data)
            rank: Current rank id
            
        Returns:
            str: Replica file path
        """
        import os
        
        if not write_buckets or len(write_buckets) == 0:
            raise ValueError("write_buckets is empty, cannot determine file path")
        
        # Get the first bucket's file name (assuming all buckets are in the same directory)
        first_bucket = write_buckets[0]
        pair_rank = self.pairing_map.get(rank)
        if isinstance(first_bucket, tuple) and len(first_bucket) >= 3:
            file_name, _, _ = first_bucket
            
            # Extract directory and filename
            file_dir = os.path.dirname(file_name)
            base_name = os.path.basename(file_name)
            
            # Add replica suffix
            base_name_no_ext, ext = os.path.splitext(base_name)
            replica_file_name = f"{base_name_no_ext}_replica{pair_rank}_rank{rank}{ext}"
            
            # Construct full path
            if file_dir:
                replica_file_path = os.path.join(file_dir, replica_file_name)
            else:
                replica_file_path = replica_file_name
            
            return replica_file_path
        else:
            raise ValueError(f"Invalid bucket format: {first_bucket}")
    
    @staticmethod
    def _write_bytes_to_file(data_bytes: bytes, file_path: str):
        """Write bytes directly to file.
        
        This function is designed to be called in a separate process to write
        the received checkpoint data directly to disk without deserialization.
        Used for replica checkpoint saving (no results queue needed).
        
        Args:
            data_bytes: Bytes data to write
            file_path: Target file path
        """
        import os
        import logging
        
        # Get logger for this module
        _logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        file_dir = os.path.dirname(file_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
        
        # Write bytes directly to file
        with open(file_path, 'wb') as f:
            f.write(data_bytes)

        _logger.info(f"Successfully wrote {len(data_bytes) / (1024**2):.2f} MB to {file_path}")
    
    @staticmethod
    def _write_bytes_to_file_with_queue(data_bytes: bytes, file_path: str, num_buckets: int, global_results_queue, 
                                       use_gemini: bool = False, use_gemini_optimized: bool = False):
        """Write bytes directly to file and put results to queue.
        
        This function is designed to be called in a separate process to write
        the original checkpoint data directly to disk without deserialization.
        It also reports write results to the global_results_queue.
        
        Args:
            data_bytes: Bytes data to write
            file_path: Target file path
            num_buckets: Number of write buckets (for results reporting)
            global_results_queue: Queue to report write results
            use_gemini: Whether Gemini mode is enabled
            use_gemini_optimized: Whether Gemini optimized mode is enabled
        """
        import os
        import logging
        from time import time
        
        # Get logger for this module
        _logger = logging.getLogger(__name__)
        
        w_start = time()
        
        try:
            # Ensure directory exists
            file_dir = os.path.dirname(file_path)
            if file_dir:
                os.makedirs(file_dir, exist_ok=True)
            
            # Write bytes directly to file
            with open(file_path, 'wb') as f:
                f.write(data_bytes)

            _logger.info(f"Successfully wrote {len(data_bytes) / (1024**2):.2f} MB to {file_path}")
            
            # Create write results compatible with FileSystemWriterAsync.retrieve_write_results
            if use_gemini and use_gemini_optimized:
                # In Gemini optimized mode, we write serialized bytes directly as a single operation
                # So we only report one result entry (key 0), regardless of how many buckets
                # were in the original write_buckets before preload
                write_results_or_exc = {0: []}
            else:
                # In normal mode, report results for each bucket
                write_results_or_exc = {}
                for i in range(num_buckets):
                    write_results_or_exc[i] = []
            
            # Put results to queue if provided
            if global_results_queue is not None:
                global_results_queue.put(write_results_or_exc)
                _logger.info(f"Put write results to queue: {len(write_results_or_exc)} entries")
            else:
                _logger.warning("global_results_queue is None, cannot report write results")
            
            w_end = time()
            _logger.info(f"Write with queue took {w_end - w_start:.2f}s, process exiting normally")
            
        except Exception as e:
            _logger.error(f"Failed to write bytes to file: {e}")
            # Put exception to queue
            if global_results_queue is not None:
                global_results_queue.put(RuntimeError(f"Write failed: {e}"))
            raise

    def is_current_async_call_done(self, blocking: bool = False, no_dist: bool = False) -> bool:
        """Check if async save is finished on all ranks.

        For semantic correctness, requires rank synchronization in each check.
        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until the call is done
                on all ranks. Otherwise, returns immediately if at least one rank
                is still active. Defaults to False.
            no_dist (bool, Optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.

        Returns:
            bool: True if all ranks are done (immediately of after active wait
                if `blocking` is True), False if at least one rank is still active.
        """
        # The following takes the same overhead
        # as torch.distributed.barrier (single integer all-reduce)
        is_alive = int(self.process.is_alive()) if self.process is not None else 0
        
        # Also check replica process if it exists
        if hasattr(self, 'replica_process') and self.replica_process is not None:
            is_alive = max(is_alive, int(self.replica_process.is_alive()))
        
        is_done = not is_alive if no_dist else self.sync_all_async_calls(is_alive)

        if is_done or blocking:
            # Process join is called in the following cases
            # 1. blocking == True -> regardless of is_done
            # 2. blocking == False (non-blocking)
            #    -> is_done == True: async requests on all ranks are identified to be finished
            #    `self.close()` makes sure the async callers terminated
            self.close()
            is_done = True
        return is_done

    def close(self):
        """For TemporalAsyncCaller, this method is called explictly in `is_current_async_calls_done`

        This method make sure the TemporalAsyncCaller terminated
        with all its assigned async request completed
        """
        if self.process:
            logger.debug(f"rank: {torch.distributed.get_rank()}, joining self.process")
            self.process.join()
            self.process = None
            logger.debug(
                "TemporalAsyncCaller: Async process join finished "
                f"after {time() - self.start_time:.2f}s from forking"
            )
        
        # Also join replica process if it exists
        if hasattr(self, 'replica_process') and self.replica_process is not None:
            logger.debug(f"rank: {torch.distributed.get_rank()}, joining replica_process")
            self.replica_process.join()
            self.replica_process = None
            logger.debug(f"rank: {torch.distributed.get_rank()}, replica process join finished")
        
        self.start_time = None

    def __del__(self):
        pass


class PersistentAsyncCaller(AsyncCaller):
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    def __init__(self):
        self.process: mp.Process = None
        self.start_time: Optional[float] = None
        ctx = mp.get_context('spawn')
        # main queue to deliver `AsyncRequest` from host to the ckpt worker
        self.queue: mp.JoinableQueue = ctx.JoinableQueue()
        # Queue used to synchronize for the completion of preloading tensors to host
        # between a trainer and ckpt worker
        self.preload_q: mp.JoinableQueue = ctx.JoinableQueue()
        # Queue used to inform trainer when the saving is completed
        self.comp_q: mp.Queue = ctx.Queue()
        self.cur_item: int = None
        self.cur_idx: int = -1

    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Put `AsyncRequest` to the Persistent Async Caller

        This method must be called on all ranks.

        Args:
            async_fn (Callable, optional): async function to call. If None,
                no process will be started.
            async_req (AsyncRequest): `AsyncRequest` object containing to
                                       schedule a checkpointing request
        """
        if async_req.async_fn is None:
            return  # nothing to do

        start_sync = end_sync = None

        self.start_time = time()
        if self.process is None:
            ctx = mp.get_context('spawn')
            logger.info(
                f"PersistentAsyncCaller: {torch.distributed.get_rank()}, Starting Async Caller"
            )
            self.process: mp.Process = ctx.Process(
                target=PersistentAsyncCaller.async_loop,
                args=(
                    torch.distributed.get_rank(),
                    self.queue,
                    self.preload_q,
                    self.comp_q,
                    logger.getEffectiveLevel(),
                ),
            )
            self.process.start()
            logger.info(
                f"PersistentAsyncCaller: {torch.distributed.get_rank()}, Started Async Caller"
            )

        if async_req.preload_fn:
            self.preload_q.put(async_req.call_idx)
        self.queue.put(async_req)
        logger.debug(f"rank: {torch.distributed.get_rank()}, put {async_req.call_idx}")

        if async_req.preload_fn:
            start_sync = time()
            # Synchronize for pre-staging tensors
            self.preload_q.join()
            end_sync = time()
            logger.debug(
                f"rank: {torch.distributed.get_rank()}, "
                f"takes {end_sync - start_sync} to finish D2H "
            )

        init_time = time()
        logger.debug(
            f"rank: {torch.distributed.get_rank()}, takes {init_time - self.start_time} "
            "to schedule async ckpt "
        )

    def is_current_async_call_done(self, blocking: bool = False, no_dist: bool = False) -> bool:
        """Check if async save is finished on all ranks.

        For semantic correctness, requires rank synchronization in each check.
        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until the call is done
                on all ranks. Otherwise, returns immediately if at least one rank
                is still active. Defaults to False.
            no_dist (bool, Optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.

        Returns:
            bool: True if all ranks are done (immediately of after active wait
                if `blocking` is True), False if at least one rank is still active.
        """

        is_alive: bool = False

        if self.process:
            while self.cur_item is None:
                try:
                    # Retrieve comp call_idx without waiting
                    self.cur_item = self.comp_q.get_nowait()
                except Empty:
                    # This method is called after any `AsyncRequest` is pushed to the main loop
                    # So, the background writing is still active
                    # before the worker put call_idx to `comp_q`
                    if not blocking:
                        is_alive = True
                        break
                    sleep(0.1)

        if self.cur_item is not None:
            logger.debug(
                f"rank: {torch.distributed.get_rank()}, item: {self.cur_item}"
                f" is completed, {is_alive}"
            )

        is_done = not is_alive if no_dist else self.sync_all_async_calls(is_alive)
        # This is set to False when blocking == False so this routine is called again
        # to simply call `sync_all_async_calls` to check if other ranks complete the writing
        if is_done:
            # The current request is completed globally. Reset the current item for polling.
            logger.debug(
                f"rank: {torch.distributed.get_rank()}, item: {self.cur_item}"
                f" is completed globally, {is_done}"
            )
            self.cur_item = None

        return is_done

    def close(self):
        """Wait on the left async requests and terminate the PersistentAsyncCaller

        Signals the PersistentAsyncCaller by sending a 'DONE' message to make it terminated
        """
        logger.info(
            f"PersistentAsyncCaller: {torch.distributed.get_rank()}, Destroying Async Caller"
        )
        if self.process:
            self.queue.put('DONE')
            self.queue.join()
            self.process.join()
            self.process = None

    def __del__(self):
        self.close()

    @staticmethod
    @_disable_gc()
    def async_loop(
        rank: int,
        queue: mp.JoinableQueue,
        preload_q: mp.JoinableQueue,
        comp_q: mp.Queue,
        log_level: int = logging.INFO,
    ):
        """Main function for the persistent checkpoint worker

        The persisent worker is created once and terminated at exit or
        when application calls `close()` explictily

        This routine receives `AsyncRequest` and does `preload_fn` first and
        put the integer value in `preload_q` to inform the trainer to proceed.
        When the `async_fn` from the request` is completed (background saving is done),
        it puts a integer value to `comp_q` to notify the trainer the completion.

        Args:
            rank (int): the rank of the trainer where the persistent worker is created.
            queue (mp.JoinableQueue): the main queue used to receive `AsyncRequest
                                      from the training rank
            preload_q (mp.JoinableQueue): a queue to inform trainer that preloading of tensors
                                          from GPU to Host or dedicated location is completed
            comp_q (mp.Queue): a queue to inform the training rank the completion of scheduled
                               async checkpoint request
            log_level (int, Optional): an integer to set log-level in this spawned process
                                       to get aligned with the training rank's logging level

        """
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        logger.info(f"PersistentAsyncCaller: persistent ckpt worker for {rank} has started")
        while True:
            item = queue.get()
            if isinstance(item, str) and item == 'DONE':
                queue.task_done()
                break
            elif isinstance(item, AsyncRequest):
                async_fn_args = list(item.async_fn_args)
                if item.preload_fn:
                    call_idx = preload_q.get()
                    # the 2nd arg is state dict
                    async_fn_args[1] = item.preload_fn()
                    logger.debug(f"{rank} has completed D2H of {call_idx}")
                    preload_q.task_done()
                item.async_fn(*async_fn_args, **item.async_fn_kwargs)
                logger.debug(f"{rank} has completed saving {item.call_idx}")
                comp_q.put(item.call_idx)
                queue.task_done()

        logger.info(f"PersistentAsyncCaller: persistent ckpt worker for {rank}  has terminated")


class _ActiveAsyncRequest(NamedTuple):
    """Helper to represent an active async call.

    Args:
        idx (int): index of the call (starting from 0)
        async_caller (DistributedAsyncCaller): async caller instance that represents
            the async process handling the async request
        async_request (AsyncRequest):  async request that is being called
    """

    idx: int
    async_caller: AsyncCaller
    async_request: AsyncRequest


class AsyncCallsQueue:
    """Manages a queue of async calls.

    Allows adding a new async call with `schedule_async_request` and finalizing
    active calls with `maybe_finalize_async_calls`.
    """

    def __init__(self, persistent: bool = False):
        self.async_calls: deque[_ActiveAsyncRequest] = deque([])
        self.call_idx: int = -1
        self.persistent: bool = persistent
        self.persistent_caller: AsyncCaller = None

    def _get_async_caller(self):
        if not self.persistent:
            return TemporalAsyncCaller()
        if self.persistent_caller is None:
            self.persistent_caller = PersistentAsyncCaller()
        return self.persistent_caller

    def schedule_async_request(self, async_request: AsyncRequest) -> int:
        """Start a new async call and add it to a queue of active async calls.

        This method must be called on all ranks.

        Args:
            async_request (AsyncRequest): async request to start.

        Returns:
            int: index of the async call that was started.
                This can help the user keep track of the async calls.
        """
        self.call_idx += 1
        async_caller = self._get_async_caller()
        # Backward compatibility for local checkpointing built with the old AsyncRequest
        if len(async_request._fields) != len(AsyncRequest._fields):
            async_request = AsyncRequest(**async_request._asdict())
        async_request = async_request.freeze()
        async_caller.schedule_async_call(
            async_request._replace(call_idx=self.call_idx, finalize_fns=[])
        )
        self.async_calls.append(_ActiveAsyncRequest(self.call_idx, async_caller, async_request))
        return self.call_idx

    def maybe_finalize_async_calls(self, blocking=False, no_dist=False) -> List[int]:
        """Finalizes all available calls.

        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until all active requests
                are done. Otherwise, finalizes only the async request that already
                finished. Defaults to False.
        Returns:
            List[int]: list of indices (as returned by `schedule_async_request`)
                of async calls that have been successfully finalized.
        """
        call_idx_finalized = []
        while self.async_calls:
            next_async_done = self.async_calls[0].async_caller.is_current_async_call_done(
                blocking, no_dist
            )
            if not next_async_done:
                break
            with debug_time("finalize", logger):
                call_idx, _, async_request = self.async_calls.popleft()
                for finalize_fn in async_request.finalize_fns:
                    finalize_fn()
                ten = torch.tensor([call_idx], dtype=torch.int, device=torch.cuda.current_device())
                torch.distributed.all_reduce(ten, op=torch.distributed.ReduceOp.MAX)
                assert ten.item() == call_idx, 'Unmatched async calls. '
                'That probably means not all ranks are participating in async finalization'
                call_idx_finalized.append(call_idx)
        return call_idx_finalized

    def get_num_unfinalized_calls(self):
        """Get the number of active async calls."""
        return len(self.async_calls)

    def close(self):
        """Finalize all calls upon closing."""
        self.maybe_finalize_async_calls(blocking=True)
        if self.persistent and self.persistent_caller:
            self.persistent_caller.close()
