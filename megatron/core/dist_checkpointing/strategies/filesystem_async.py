# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Storage writer for PyT Distributed format allowing asynchronous save."""

import dataclasses
import inspect
import logging
import os
import pickle
import queue
from functools import partial
from heapq import heappop, heappush
from itertools import chain
from operator import itemgetter
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import multiprocessing as mp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX, _StoragePrefix, _write_item
from torch.distributed.checkpoint.metadata import Metadata

try:
    from torch.distributed.checkpoint.filesystem import _StorageWriterTransforms
except ImportError:
    _StorageWriterTransforms = Any

from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

from .async_utils import _disable_gc
from .state_dict_decomposer import (
    DecomposedStateDict,
    decompose_state_dict,
    organize_tensor_data_in_cpu_memory,
)

from .state_dict_decomposer import TensorMetadata

logger = logging.getLogger(__name__)

WriteBucket = Tuple[Path, str, Tuple[list, list]]  # represents writes to a single file


@dataclasses.dataclass
class EccheckMappedFile:
    """Container for mmap file information used for NCCL send/recv operations.
    
    Attributes:
        mmap_object: mmap object that must be kept alive for the memory to remain valid
        memory_address: starting memory address (can be used with NCCL)
        file_size: total size of the mapped file in bytes
    """
    mmap_object: Any  # mmap.mmap object
    memory_address: int
    file_size: int
    local_metadata: List[TensorMetadata]
    non_tensor_data: Dict[str, Any]
    
    def close(self) -> None:
        """Close the mmap object to release resources."""
        if self.mmap_object is not None:
            self.mmap_object.close()
            self.mmap_object = None


@dataclasses.dataclass
class EclatinMappedFile:
    """Container for mmap file information used for ECLATIN load operations.
    
    Similar to EccheckMappedFile but for ECLATIN format.
    
    Attributes:
        mmap_object: mmap object that must be kept alive for the memory to remain valid
        memory_address: starting memory address
        file_size: total size of the mapped file in bytes
        local_metadata: List of TensorMetadata extracted from Component 2
        non_tensor_data: Dict of non-tensor data extracted from Component 1
        tensor_infos: List of TensorInfo with offset information (for recovery buffer extraction)
    """
    mmap_object: Any  # mmap.mmap object
    memory_address: int
    file_size: int
    local_metadata: List[TensorMetadata]
    non_tensor_data: Dict[str, Any]
    tensor_infos: Optional[List[Any]] = None  # List[TensorInfo] with offset information
    
    def close(self) -> None:
        """Close the mmap object to release resources."""
        if self.mmap_object is not None:
            self.mmap_object.close()
            self.mmap_object = None

try:
    import psutil

    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

_results_queue = None


def _get_write_results_queue():
    global _results_queue
    if _results_queue is None:
        ctx = mp.get_context("spawn")
        _results_queue = ctx.Manager().Queue()
    return _results_queue


class FileSystemWriterAsync(FileSystemWriter):
    """
    Async-enabled implementation of FileSystemWriter using file I/O.

    This class does not spawn the async process itself but relies on an external async mechanism.

    **Flow:**

    1. Call `write_data`
    2. Externally start an async process with `get_save_function_and_args` and its arguments.
    3. The async function `writer_proxy_func` calls `write_preloaded_data` across multiple
        processes.
    4. Once saving is finalized on all ranks, call `super().finish` with the results stored
        in `self.writer_result`.

    **Note:** Step (3) can also be executed synchronously.

    Currently, it is assumed that a separate writer is created for each ckpt save
    (intermediate state is stored as writer attributes).
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        *args,
        separation_hint: Optional[str] = None,
        use_msc: bool = False,
        use_eccheck: bool = False,
        eccheck_pin_memory: bool = True,
        eccheck_k: int = 2,
        eccheck_m: int = 2,
        eccheck_buffer_size: int = 64 * 1024 * 1024,
        eccheck_native: Optional[Any] = None,  # Pre-initialized C++ module
        eccheck_buffers: Optional[Dict] = None,  # Pre-allocated buffers
        use_eclatin: bool = False,
        eclatin_pin_memory: bool = True,
        eclatin_buffer_size: int = 64 * 1024 * 1024,
        eclatin_native: Optional[Any] = None,  # Pre-initialized C++ module
        eclatin_buffers: Optional[Dict] = None,  # Pre-allocated buffers
        use_gemini: bool = False,
        gemini_native: Optional[Any] = None,  # Pre-initialized C++ module
        **kwargs,
    ):
        self.checkpoint_dir = path
        self.use_msc = use_msc
        
        # EC-CHECK configuration
        self.use_eccheck = use_eccheck
        self.eccheck_pin_memory = eccheck_pin_memory
        
        # EC-CHECK encoding parameters (configurable)
        self.eccheck_k = eccheck_k  # Number of data nodes
        self.eccheck_m = eccheck_m  # Number of encoded packets per data packet

        self.eccheck_buffer_size = eccheck_buffer_size  # Buffer size in bytes
        
        # ECLATIN configuration
        self.use_eclatin = use_eclatin
        self.eclatin_pin_memory = eclatin_pin_memory
        self.eclatin_buffer_size = eclatin_buffer_size  # Buffer size in bytes
        
        # Gemini configuration
        self.use_gemini = use_gemini

        super().__init__(path, *args, **kwargs)
        if not self.single_file_per_rank:
            raise NotImplementedError(
                "single_file_per_rank flag not supported for FileSystemWriterAsync"
            )

        self.can_run_decentralized_global_plan: bool = True

        # Intermediate state between preparation and finalization
        self.write_buckets: Optional[List[WriteBucket]] = None
        self.results_queue: Optional[mp.Queue] = None
        self.separation_hint = separation_hint
        
        # EC-CHECK intermediate state
        self.decomposed_state_dict: Optional[DecomposedStateDict] = None
        self.tensor_buffer: Optional[torch.Tensor] = None
        self.preallocated_cpu_buffer: Optional[torch.Tensor] = None
        self.eccheck_serialized_metadata: Optional[Dict] = None
        
        # EC-CHECK Phase 2 & 3 state
        self.eccheck_global_registry = None  # GlobalMetadataRegistry from all ranks
        self.eccheck_data_buffers = None  # List of data buffers
        self.eccheck_encoding_buffers = None  # List of encoding buffers
        self.eccheck_recv_encoding_buffers = None  # Tuple of two large receive buffers (thread1, thread2)
        self.eccheck_parity_buffers = None  # List of parity buffers for XOR results
        
        self.eccheck_p2p_buffers = None
        
        # EC-CHECK buffer poller thread (persistent, created once)
        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None  # Controls when polling is active
        
        self.ecc_write_buckets = None
        # Initialize C++ native module if available
        if eccheck_native is not None:
            # Use pre-initialized C++ module from strategy
            self._eccheck_native = eccheck_native
            self._eccheck_shared = True  # Mark as shared module
            logger.info("EC-CHECK: Using pre-initialized C++ native module from strategy")
            
            # Use pre-allocated buffers from strategy
            if eccheck_buffers is not None:
                self._setup_eccheck_buffers_from_strategy(eccheck_buffers)
        else:
            self._eccheck_native = None
            self._eccheck_shared = False
        
        # ECLATIN intermediate state
        self.eclatin_serialized_metadata: Optional[Dict] = None
        
        # ECLATIN Phase 2 & 3 state
        self.eclatin_global_registry = None  # GlobalMetadataRegistry from all ranks
        self.eclatin_data_buffers = None  # List of data buffers (pooled)
        self.eclatin_recv_buffers = None  # List of recv buffers (pooled)
        # Note: 4 persistent blocks (data_block_1/2, parity_block_1/2) will be allocated in strategy
        
        # ECLATIN buffer poller thread (persistent, created once, shared with ECCHECK if both enabled)
        self._eclatin_buffer_poller_thread = None
        self._eclatin_buffer_poller_stop_event = None
        self._eclatin_buffer_poller_active_event = None  # Controls when polling is active
        
        self.ecl_write_buckets = None
        # Initialize C++ native module if available
        if eclatin_native is not None:
            # Use pre-initialized C++ module from strategy
            self._eclatin_native = eclatin_native
            self._eclatin_shared = True  # Mark as shared module
            logger.info("ECLATIN: Using pre-initialized C++ native module from strategy")
            
            # Use pre-allocated buffers from strategy
            if eclatin_buffers is not None:
                self._setup_eclatin_buffers_from_strategy(eclatin_buffers)
        else:
            self._eclatin_native = None
            self._eclatin_shared = False
        
        # Gemini intermediate state
        # Note: Gemini reuses decomposed_state_dict and preallocated_cpu_buffer from EC-CHECK
        # Initialize C++ native module if available
        if gemini_native is not None:
            # Use pre-initialized C++ module from strategy
            self._gemini_native = gemini_native
            self._gemini_shared = True  # Mark as shared module
            logger.info("Gemini: Using pre-initialized C++ native module from strategy")
        else:
            self._gemini_native = None
            self._gemini_shared = False

    def __del__(self):
        """
        Destructor to ensure proper cleanup of EC-CHECK resources.
        Note: Only cleanup if this writer owns the C++ module (not shared).
        """
        try:
            if hasattr(self, 'use_eccheck') and self.use_eccheck and hasattr(self, '_eccheck_native') and self._eccheck_native is not None:
                # Only cleanup if we own the module (not shared from strategy)
                # The strategy will handle cleanup of shared modules
                # Buffer poller thread is managed by strategy, not by writer
                if not hasattr(self, '_eccheck_shared') or not self._eccheck_shared:
                    self._stop_phase3_workers()
        except Exception as e:
            # Log but don't raise exceptions in destructor
            logger.warning(f"EC-CHECK: Error during cleanup in destructor: {e}")


    def _setup_eccheck_buffers_from_strategy(self, buffers):
        """Set up EC-CHECK buffers from pre-allocated strategy buffers.
        
        Note: Sets up data, encoding, and parity buffers from strategy.
        Receive buffers will be allocated later after metadata exchange.
        """
        self.eccheck_data_buffers = buffers['data_buffers']
        self.eccheck_encoding_buffers = buffers['encoding_buffers']
        self.eccheck_parity_buffers = buffers.get('parity_buffers')
        self._free_data_buffer_queue = buffers['free_data_buffer_queue']
        self._free_encoding_buffer_queue = buffers['free_encoding_buffer_queue']
        self._free_parity_buffer_queue = buffers.get('free_parity_buffer_queue')
        
        # Use buffer poller from strategy (already running)
        self._buffer_poller_active_event = buffers.get('buffer_poller_active_event')
        # Store the strategy's poll method with a different name to avoid conflict
        self._strategy_poll_and_release_buffers = buffers.get('poll_and_release_buffers')
        
        # Mark that we're using shared buffer poller (don't start our own)
        self._buffer_poller_shared = True
        
        # Receive buffers are NOT set here
        # They will be allocated after metadata exchange when peer data size is known
        
        logger.info(
            f"EC-CHECK: Using pre-allocated buffers from strategy - "
            f"Data: {len(self.eccheck_data_buffers)}, "
            f"Encoding: {len(self.eccheck_encoding_buffers)}, "
            f"Parity: {len(self.eccheck_parity_buffers) if self.eccheck_parity_buffers else 0}, "
            f"Buffer poller: {'shared from strategy' if self._buffer_poller_active_event else 'will create own'}"
        )

    def _setup_eclatin_buffers_from_strategy(self, buffers):
        """Set up ECLATIN buffers from pre-allocated strategy buffers.
        
        Note: Sets up data and recv buffers (pooled) from strategy.
        The 4 persistent blocks (data_block_1/2, parity_block_1/2) will be allocated
        in strategy after metadata exchange.
        """
        self.eclatin_data_buffers = buffers['data_buffers']
        self.eclatin_recv_buffers = buffers['recv_buffers']
        self._free_eclatin_data_buffer_queue = buffers['free_data_buffer_queue']
        self._free_eclatin_recv_buffer_queue = buffers['free_recv_buffer_queue']
        
        # Use buffer poller from strategy (already running)
        self._eclatin_buffer_poller_active_event = buffers.get('buffer_poller_active_event')
        # Store the strategy's poll method with a different name to avoid conflict
        self._eclatin_strategy_poll_and_release_buffers = buffers.get('poll_and_release_buffers')
        
        # Mark that we're using shared buffer poller (don't start our own)
        self._eclatin_buffer_poller_shared = True
        
        # Persistent blocks are NOT set here
        # They will be allocated in strategy after metadata exchange
        
        logger.info(
            f"ECLATIN: Using pre-allocated buffers from strategy - "
            f"Data: {len(self.eclatin_data_buffers)}, "
            f"Recv: {len(self.eclatin_recv_buffers)}, "
            f"Buffer poller: {'shared from strategy' if self._eclatin_buffer_poller_active_event else 'will create own'}"
        )


    def prepare_write_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        First stage of async saving. Copy data to CPU and plan the local saving.

        Args:
            plan (SavePlan): save plan generated by the PyT Distributed compatible planner
            planner (SavePlanner): save planner used to resolve the bytes and tensor data

        Returns: None, but stores the save plan in `self.write_buckets`
        """
        # EC-CHECK mode: use decomposed state_dict passed from strategy
        if self.use_eccheck:
            # In EC-CHECK mode, decomposed_state_dict and metadata are already prepared
            # by the strategy layer (torch.py). We just need to prepare write_buckets.
            self._prepare_eccheck_write_buckets(plan)
            return
        
        storage_plan: _StoragePrefix = plan.storage_data
        start = time()
        logger.debug(f"thread_count: {self.thread_count}, time: {start}")
        if self.separation_hint:
            assert (
                self.thread_count > 1
            ), "thread_count must be at least 2 if separation_hint is provided"
        bins = self.thread_count // 2 if self.separation_hint is not None else self.thread_count
        item_buckets = _split_by_size_and_type(bins, plan.items)
        logger.debug(f"bucket_prep, time: {time() - start}")

        start = time()
        # move tensors from GPU to CPU before starting async writing
        # We do D2H synchronously for now
        file_count = 0

        def gen_file(prefix=""):
            nonlocal file_count
            file_name = f"{prefix}{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        def _clone_if_needed(ten: torch.Tensor):
            """Clone if we detect incontiguous storage for CPU tensors

            Makes sure we perform a `clone` only if we detect incontiguous storage,
            so that we don't blow up host memory unnecessarily.

            TODO: For persistent worker, this work should be changed to move the cpu tensor
            to shared_memory.
            """
            ten = ten.detach()
            if ten.device.type != "cpu":
                # We do D2H later when the async_request is scheduled for both sync / async
                # checkpointing
                return ten
            is_view = ten.untyped_storage().size() != ten.numel() * ten.itemsize
            return ten.clone() if is_view else ten

        # Prepare bytes / tensor data in each bucket, which will be assigned to each writer process
        self.write_buckets = []
        for group_name, group_buckets in _split_by_separation_hint(
            item_buckets, self.separation_hint
        ).items():
            for bucket in group_buckets:
                bytes_data = [
                    (item, planner.resolve_data(item))
                    for item in bucket
                    if item.type == WriteItemType.BYTE_IO
                ]
                tensor_data = [
                    (item, _clone_if_needed(planner.resolve_data(item)))
                    for item in bucket
                    if item.type != WriteItemType.BYTE_IO
                ]
                if len(bytes_data) > 0 or len(tensor_data) > 0:
                    file_name = gen_file(prefix=group_name)
                    self.write_buckets.append(
                        (  # type: ignore[arg-type]
                            os.path.join(self.checkpoint_dir, file_name),
                            file_name,
                            (bytes_data, tensor_data),
                        )
                    )

        # Check if there is anything to write on this rank
        if len(self.write_buckets) > 0:
            assert len(self.write_buckets) <= self.thread_count, (
                len(self.write_buckets),
                self.thread_count,
            )
            self.results_queue = _get_write_results_queue()
        else:
            self.results_queue = None
        end = time()
        logger.debug(f"D2H and push, time: {end - start}")

    def get_save_function_and_args(self) -> Tuple[Optional[Callable], Optional[Callable], List]:
        """
        Get function that saves the data to storage along with its arguments.
        Allows the external caller to apply the save function synchronously or asynchronously.

        Returns: None (if there is nothing to write on this rank) or a tuple of:
            1) the function that saves the data.
            2) the function that stages the GPU tensors to a destination for async checkpointing.
               This function should be self-contained.
            3) arguments to that function in 1).
        """
        if not self.write_buckets:
            return None, None, []
        
        transform_list = [self.transforms] if hasattr(self, "transforms") else []
        
        # ECLATIN mode: use special preload function
        # The preload function will embed ECLATIN data in write_buckets
        if self.use_eclatin:
            # Ensure eclatin blocks are available
            if self.eclatin_blocks is None:
                logger.warning("ECLATIN: eclatin_blocks not set, falling back to normal mode")
                return (
                    partial(self.write_preloaded_data_multiproc, transform_list, self.use_msc),
                    partial(self.preload_tensors, self.write_buckets, True),
                    [torch.distributed.get_rank(), self.write_buckets, self.results_queue],
                )
            
            return (
                partial(self.write_preloaded_data_multiproc, transform_list, self.use_msc),
                partial(self._eclatin_preload_tensors_to_buffer, True),
                [torch.distributed.get_rank(), self.write_buckets, self.results_queue],
            )
        
        # EC-CHECK mode: use special preload function
        # The preload function will embed EC-CHECK data in write_buckets
        if self.use_eccheck:
            # Ensure eccheck_file_path is set in eccheck_serialized_metadata
            if self.eccheck_serialized_metadata and self.eccheck_serialized_metadata.get('eccheck_file_path') is None:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                eccheck_file = self.eccheck_serialized_metadata.get('eccheck_file', f'__{rank}_0.distcp')
                eccheck_path = os.path.join(self.checkpoint_dir, eccheck_file)
                self.eccheck_serialized_metadata['eccheck_file_path'] = eccheck_path
                logger.debug(f"EC-CHECK: Set eccheck_file_path to {eccheck_path}")
            
            return (
                partial(self.write_preloaded_data_multiproc, transform_list, self.use_msc),
                partial(self._eccheck_preload_tensors_to_buffer, True),
                [torch.distributed.get_rank(), self.write_buckets, self.results_queue],
            )
        from megatron.training import get_args
        args = get_args()
        
        # Gemini optimized mode: use continuous buffer preload to avoid serialization
        if hasattr(args, 'use_gemini') and args.use_gemini and \
           hasattr(args, 'use_gemini_optimized') and args.use_gemini_optimized:
            logger.info("Gemini: Using optimized preload (continuous buffer, no serialization)")
            return (
                partial(self.write_preloaded_data_multiproc, transform_list, self.use_msc),
                partial(self._gemini_preload_to_continuous_buffer, self.write_buckets, True),
                [torch.distributed.get_rank(), self.write_buckets, self.results_queue],
            )
        
        if args.use_layer_transfer:
            return (
                partial(self.write_preloaded_data_multiproc, transform_list, self.use_msc),
                partial(self.preload_tensors_layerwise_cpp, self.write_buckets, True),
                [torch.distributed.get_rank(), self.write_buckets, self.results_queue],
            )
        # Normal mode
        return (
            partial(self.write_preloaded_data_multiproc, transform_list, self.use_msc),
            partial(self.preload_tensors, self.write_buckets, True),
            [torch.distributed.get_rank(), self.write_buckets, self.results_queue],
        )

    @staticmethod
    def preload_tensors(write_buckets: List[WriteBucket], non_blocking=True) -> List[WriteBucket]:
        """
        Preloads tensors in `state_dict` to host memory via CPU memory.

        Args:
            write_buckets (List): List of `WriteBucket` objects that define what to
                save in a checkpoint.
            non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
        """
        result = []

        for bucket in write_buckets:
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            tensor_data = [
                (item, tensor.to("cpu", non_blocking=non_blocking)) for item, tensor in tensor_data
            ]
            result.append((file_name, storage_key, (bytes_data, tensor_data)))
        if non_blocking:
            torch.cuda.synchronize()
        return result

    def _gemini_preload_to_continuous_buffer(self, write_buckets: List[WriteBucket], non_blocking=True) -> List[WriteBucket]:
        """
        Gemini optimized preload: Transfer tensors to continuous CPU buffer and exchange with peer rank.
        
        This method is designed for Gemini checkpointing to eliminate torch.save serialization overhead.
        It performs the following operations in one pass:
        1. GPU→CPU: Copy tensor data to continuous CPU buffer (no serialization)
        2. Exchange: Swap buffers with paired rank
        3. Return: Both local and remote buffers in write_buckets format
        
        Strategy:
        1. Calculate total size of all data in write_buckets
        2. Allocate a single continuous CPU buffer (pinned memory for faster transfer)
        3. Copy all tensor and bytes data sequentially to the buffer
        4. Generate lightweight metadata for reconstruction
        5. Exchange buffer and metadata with paired rank
        6. Return write_buckets containing both local and remote data
        
        Args:
            write_buckets (List[WriteBucket]): Original write buckets with tensors
            non_blocking (bool): Use non-blocking GPU-to-CPU transfer
            
        Returns:
            List[WriteBucket]: Two buckets - [local_bucket, remote_bucket]
                - local_bucket: Contains local buffer and metadata for original checkpoint
                - remote_bucket: Contains remote buffer and metadata for replica checkpoint
        """
        if not write_buckets or len(write_buckets) == 0:
            return write_buckets
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        start_time = time()
        
        logger.info(f"Gemini rank {rank}: Starting optimized preload using decomposed_state_dict...")
        
        # Phase 1: Use decomposed_state_dict if available (prepared by strategy)
        if not hasattr(self, 'decomposed_state_dict') or self.decomposed_state_dict is None:
            logger.error(f"Gemini rank {rank}: decomposed_state_dict not available, falling back to normal mode")
            return self.preload_tensors(write_buckets, non_blocking)
        
        # Get total size from decomposed_state_dict
        total_size = self.decomposed_state_dict.total_tensor_size_bytes
        
        logger.info(
            f"Gemini rank {rank}: Using decomposed_state_dict with {len(self.decomposed_state_dict.tensor_infos)} tensors, "
            f"total size: {total_size / (1024**2):.2f} MB"
        )
        
        # Phase 2: Allocate continuous buffer (reuse preallocated buffer if available)
        if self.preallocated_cpu_buffer is not None:
            buffer = self.preallocated_cpu_buffer
            # If preallocated buffer exists, ensure it's large enough
            if buffer.numel() < total_size:
                logger.warning(
                    f"Gemini rank {rank}: Preallocated buffer ({buffer.numel() / (1024**3):.2f} GB) "
                    f"is smaller than total_size ({total_size / (1024**3):.2f} GB). Reallocating..."
                )
                if torch.cuda.is_available():
                    buffer = torch.empty(total_size, dtype=torch.uint8).pin_memory()
            else:
                # Use slice of preallocated buffer
                buffer = buffer[:total_size]
                logger.info(f"Gemini rank {rank}: Reusing preallocated buffer")
        else:
            # Allocate new buffer with pinned memory for faster GPU-CPU transfer
            if torch.cuda.is_available():
                buffer = torch.empty(total_size, dtype=torch.uint8).pin_memory()
                logger.info(f"Gemini rank {rank}: Allocated new pinned memory buffer")
            else:
                buffer = torch.empty(total_size, dtype=torch.uint8)
                logger.info(f"Gemini rank {rank}: Allocated new CPU buffer")
        
        # Phase 3: Copy tensor data to buffer using decomposed_state_dict (EC-CHECK style)
        num_gpu_tensors = 0
        
        for info, tensor in zip(
            self.decomposed_state_dict.tensor_infos,
            self.decomposed_state_dict.tensor_data
        ):
            # Get view of buffer at current offset
            buffer_view = buffer[info.offset:info.offset + info.size_bytes]
            
            # Flatten and copy tensor to continuous buffer (same as EC-CHECK)
            tensor_flat = tensor.flatten().contiguous().view(torch.uint8)
            buffer_view.copy_(tensor_flat, non_blocking=non_blocking)
            
            if tensor.device.type != 'cpu':
                num_gpu_tensors += 1
        
        # Synchronize GPU operations
        if non_blocking and num_gpu_tensors > 0:
            torch.cuda.synchronize()
        
        preload_time = time() - start_time
        bandwidth = (total_size / (1024**3)) / preload_time if preload_time > 0 else 0
        
        logger.info(
            f"Gemini rank {rank}: Preload completed in {preload_time:.4f}s, "
            f"copied {total_size / (1024**2):.2f} MB, "
            f"bandwidth: {bandwidth:.2f} GB/s, "
            f"GPU tensors: {num_gpu_tensors}"
        )
        
        # Package local metadata using decomposed_state_dict
        local_metadata = {
            'total_size': total_size,
            'num_tensors': len(self.decomposed_state_dict.tensor_infos),
            'non_tensor_data': self.decomposed_state_dict.non_tensor_data,
            'tensor_infos': [
                {
                    'key': info.key,
                    'shape': list(info.shape),
                    'dtype': str(info.dtype),
                    'offset': info.offset,
                    'size_bytes': info.size_bytes,
                }
                for info in self.decomposed_state_dict.tensor_infos
            ],
        }
        
        # ===== Phase 2: Exchange buffer and metadata with paired rank =====
        from megatron.training import get_args
        args = get_args()
        
        # Check if Gemini exchange is enabled
        if not (hasattr(args, 'use_gemini') and args.use_gemini):
            # No exchange needed, just return local data
            result_bucket = (
                write_buckets[0][0] if write_buckets else 'gemini_optimized.distcp',
                'gemini_optimized',
                (
                    [('gemini_metadata', local_metadata), ('gemini_buffer', buffer)],
                    []
                )
            )
            return [result_bucket]
        
        # Perform exchange with paired rank
        logger.info(f"Gemini rank {rank}: Starting buffer exchange with paired rank...")
        exchange_start = time()
        
        # Serialize metadata (small, overhead acceptable)
        import io
        metadata_buffer = io.BytesIO()
        torch.save(local_metadata, metadata_buffer)
        local_metadata_bytes = metadata_buffer.getvalue()
        local_metadata_size = len(local_metadata_bytes)
        local_buffer_size = buffer.numel()
        
        logger.info(
            f"Gemini rank {rank}: Local buffer size: {local_buffer_size / (1024**2):.2f} MB, "
            f"metadata size: {local_metadata_size / 1024:.2f} KB"
        )
        
        # Check if C++ native module is available
        if self._gemini_native is not None:
            # Use C++ ASIO-based exchange (optimized path)
            logger.info(f"Gemini rank {rank}: Using C++ ASIO-based exchange")
            
            # Step 1: Exchange buffer sizes first via torch.distributed
            # (needed to allocate remote buffer before C++ exchange)
            paired_rank = self._gemini_native.get_partner_rank()
            
            from ..strategies.async_utils import get_or_create_pair_process_group
            pair_group = get_or_create_pair_process_group(rank, paired_rank)
            
            size_tensor = torch.tensor([local_buffer_size, local_metadata_size], dtype=torch.long, device='cpu')
            gathered_sizes = [torch.zeros_like(size_tensor) for _ in range(2)]
            torch.distributed.all_gather(gathered_sizes, size_tensor, group=pair_group)
            
            pair_ranks = [min(rank, paired_rank), max(rank, paired_rank)]
            my_idx = pair_ranks.index(rank)
            paired_idx = 1 - my_idx
            remote_buffer_size = gathered_sizes[paired_idx][0].item()
            remote_metadata_size = gathered_sizes[paired_idx][1].item()
            
            logger.info(
                f"Gemini rank {rank}: Remote buffer size: {remote_buffer_size / (1024**2):.2f} MB, "
                f"metadata size: {remote_metadata_size / 1024:.2f} KB"
            )
            
            # Step 2: Allocate remote buffer
            remote_buffer = torch.empty(remote_buffer_size, dtype=torch.uint8, device='cpu')
            
            # Step 3: Exchange buffers using C++ ASIO (simultaneous send/recv)
            logger.info(f"Gemini rank {rank}: Starting C++ ASIO buffer exchange...")
            asio_start = time()
            
            try:
                # Get raw memory addresses and sizes from tensors
                send_buffer_addr = buffer.data_ptr()
                send_buffer_size = buffer.numel()
                recv_buffer_addr = remote_buffer.data_ptr()
                recv_buffer_size = remote_buffer.numel()
                
                # Call C++ exchange with raw memory addresses
                received_size = self._gemini_native.exchange_buffers(
                    send_buffer_addr, send_buffer_size,
                    recv_buffer_addr, recv_buffer_size
                )
                
                asio_time = time() - asio_start
                asio_bandwidth = ((send_buffer_size + received_size) / (1024**3)) / asio_time if asio_time > 0 else 0
                
                logger.info(
                    f"Gemini rank {rank}: C++ ASIO buffer exchange completed in {asio_time:.4f}s, "
                    f"sent: {send_buffer_size / (1024**2):.2f} MB, "
                    f"received: {received_size / (1024**2):.2f} MB, "
                    f"bandwidth: {asio_bandwidth:.2f} GB/s"
                )
            except Exception as e:
                logger.error(f"Gemini rank {rank}: C++ ASIO exchange failed: {e}")
                raise
            
            # Step 4: Exchange metadata using torch.distributed (small, overhead acceptable)
            remote_metadata_tensor = torch.empty(remote_metadata_size, dtype=torch.uint8, device='cpu')
            local_metadata_tensor = torch.frombuffer(local_metadata_bytes, dtype=torch.uint8).clone()
            
            lower_global_rank = pair_ranks[0]
            higher_global_rank = pair_ranks[1]
            
            if rank == lower_global_rank:
                torch.distributed.broadcast(local_metadata_tensor, src=lower_global_rank, group=pair_group)
                torch.distributed.broadcast(remote_metadata_tensor, src=higher_global_rank, group=pair_group)
            else:
                torch.distributed.broadcast(remote_metadata_tensor, src=lower_global_rank, group=pair_group)
                torch.distributed.broadcast(local_metadata_tensor, src=higher_global_rank, group=pair_group)
            
            # Deserialize remote metadata
            remote_metadata_bytes = remote_metadata_tensor.numpy().tobytes()
            remote_metadata_buffer = io.BytesIO(remote_metadata_bytes)
            remote_metadata = torch.load(remote_metadata_buffer)
            
        else:
            # Fallback to torch.distributed broadcast (original path)
            logger.info(f"Gemini rank {rank}: Using torch.distributed broadcast (C++ module not available)")
            
            # Get paired rank (rank 0<->2, 1<->3)
            pairing_map = {0: 2, 2: 0, 1: 3, 3: 1}
            paired_rank = pairing_map.get(rank, None)
            
            if paired_rank is None:
                logger.warning(f"Gemini rank {rank}: No paired rank found, skipping exchange")
                result_bucket = (
                    write_buckets[0][0] if write_buckets else 'gemini_optimized.distcp',
                    'gemini_optimized',
                    (
                        [('gemini_metadata', local_metadata), ('gemini_buffer', buffer)],
                        []
                    )
                )
                return [result_bucket]
            
            # Create or get pair process group
            from ..strategies.async_utils import get_or_create_pair_process_group
            pair_group = get_or_create_pair_process_group(rank, paired_rank)
            
            # Exchange sizes (buffer size + metadata size)
            size_tensor = torch.tensor([local_buffer_size, local_metadata_size], dtype=torch.long, device='cpu')
            gathered_sizes = [torch.zeros_like(size_tensor) for _ in range(2)]
            torch.distributed.all_gather(gathered_sizes, size_tensor, group=pair_group)
            
            # Get remote sizes
            pair_ranks = [min(rank, paired_rank), max(rank, paired_rank)]
            my_idx = pair_ranks.index(rank)
            paired_idx = 1 - my_idx
            remote_buffer_size = gathered_sizes[paired_idx][0].item()
            remote_metadata_size = gathered_sizes[paired_idx][1].item()
            
            logger.info(
                f"Gemini rank {rank}: Exchanging with rank {paired_rank}, "
                f"local: {local_buffer_size / (1024**2):.2f} MB, "
                f"remote: {remote_buffer_size / (1024**2):.2f} MB"
            )
            
            # Allocate remote buffers
            remote_buffer = torch.empty(remote_buffer_size, dtype=torch.uint8, device='cpu')
            remote_metadata_tensor = torch.empty(remote_metadata_size, dtype=torch.uint8, device='cpu')
            
            # Convert local metadata to tensor
            local_metadata_tensor = torch.frombuffer(local_metadata_bytes, dtype=torch.uint8).clone()
            
            # Determine broadcast order
            lower_global_rank = pair_ranks[0]
            higher_global_rank = pair_ranks[1]
            
            # Exchange buffers
            if rank == lower_global_rank:
                torch.distributed.broadcast(buffer, src=lower_global_rank, group=pair_group)
                torch.distributed.broadcast(remote_buffer, src=higher_global_rank, group=pair_group)
                torch.distributed.broadcast(local_metadata_tensor, src=lower_global_rank, group=pair_group)
                torch.distributed.broadcast(remote_metadata_tensor, src=higher_global_rank, group=pair_group)
            else:
                torch.distributed.broadcast(remote_buffer, src=lower_global_rank, group=pair_group)
                torch.distributed.broadcast(buffer, src=higher_global_rank, group=pair_group)
                torch.distributed.broadcast(remote_metadata_tensor, src=lower_global_rank, group=pair_group)
                torch.distributed.broadcast(local_metadata_tensor, src=higher_global_rank, group=pair_group)
            
            # Deserialize remote metadata
            remote_metadata_bytes = remote_metadata_tensor.numpy().tobytes()
            remote_metadata_buffer = io.BytesIO(remote_metadata_bytes)
            remote_metadata = torch.load(remote_metadata_buffer)
        
        exchange_time = time() - exchange_start
        exchange_bandwidth = ((local_buffer_size + remote_buffer_size) / (1024**3)) / exchange_time if exchange_time > 0 else 0
        
        logger.info(
            f"Gemini rank {rank}: Exchange completed in {exchange_time:.4f}s, "
            f"bandwidth: {exchange_bandwidth:.2f} GB/s"
        )
        
        total_time = time() - start_time
        logger.info(
            f"Gemini rank {rank}: Total time: {total_time:.4f}s "
            f"(preload: {preload_time:.4f}s, exchange: {exchange_time:.4f}s)"
        )
        
        # Return two buckets: local (for original checkpoint) and remote (for replica checkpoint)
        local_bucket = (
            write_buckets[0][0] if write_buckets else 'gemini_optimized.distcp',
            'gemini_optimized_local',
            (
                [('gemini_metadata', local_metadata), ('gemini_buffer', buffer)],
                []
            )
        )
        
        remote_bucket = (
            write_buckets[0][0] if write_buckets else 'gemini_optimized_replica.distcp',
            'gemini_optimized_remote',
            (
                [('gemini_metadata', remote_metadata), ('gemini_buffer', remote_buffer)],
                []
            )
        )
        
        return [local_bucket, remote_bucket]

    @staticmethod
    def preload_tensors_layerwise_cpp(write_buckets: List[WriteBucket], non_blocking=True) -> List[WriteBucket]:
        """
        Preloads tensors layer-by-layer using C++ thread for coordination.
        
        This function organizes model parameters by layer and coordinates their transfer
        from GPU to CPU using a dedicated C++ worker thread. The actual transfer can be
        done either by PyTorch (default) or by CUDA in C++ (if compiled with USE_CUDA).
        
        Transfer Modes:
            1. PyTorch mode (default, no CUDA needed in C++):
               - Python: Allocates CPU buffers and initiates async GPU->CPU copy via PyTorch
               - C++ thread: Ensures layers complete sequentially
               - Best for: Easy compilation, works everywhere
            
            2. CUDA mode (requires C++ compiled with USE_CUDA):
               - Python: Only allocates CPU buffers and passes pointers
               - C++ thread: Performs actual cudaMemcpy for each layer
               - Best for: Direct control, potentially lower overhead
        
        Args:
            write_buckets (List): List of `WriteBucket` objects that define what to
                save in a checkpoint.
            non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
        
        Returns:
            List[WriteBucket]: Same structure as input but with tensors moved to CPU.
        
        Implementation Flow:
            1. Group write_buckets by layer (using file_name)
            2. For each layer:
               a. Allocate pinned CPU buffers for GPU tensors
               b. Initiate async transfer (PyTorch) or pass to C++ (CUDA mode)
               c. Submit layer info to C++ thread
            3. C++ thread processes layers sequentially
            4. Wait for all layers to complete
            5. Construct result with CPU tensors
        """
        # Load layer_transfer_cpp module directly from .so file
        layer_transfer_cpp = None
        try:
            # Direct import .so file without modifying sys.path or affecting other packages
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Find .so file
            import glob as _glob_module
            so_files = _glob_module.glob(os.path.join(current_dir, "layer_transfer_cpp*.so"))
            
            if not so_files:
                raise ImportError(f"No layer_transfer_cpp.so file found in {current_dir}")
            
            # Load .so file directly using importlib
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("layer_transfer_cpp", so_path)
            layer_transfer_cpp = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(layer_transfer_cpp)
            logger.debug(f"Loaded layer_transfer_cpp from {so_path}")
            
        except Exception as e:
            logger.warning(
                f"layer_transfer_cpp module not found: {e}. Falling back to standard preload_tensors. "
                "To build the C++ module, run: bash build_layer_transfer.sh"
            )
            return FileSystemWriterAsync.preload_tensors(write_buckets, non_blocking)
        
        logger.info("Starting layer-wise tensor preloading using C++ thread")
        start_time = time()
        
        # Initialize the C++ processor
        processor = layer_transfer_cpp.LayerTransferProcessor()
        
        # Check if C++ module was compiled with CUDA support
        # If USE_CUDA is defined in C++, it will handle the actual GPU->CPU transfer
        # Otherwise, PyTorch handles the transfer and C++ only coordinates
        use_cuda_in_cpp = hasattr(layer_transfer_cpp, 'USE_CUDA') and layer_transfer_cpp.USE_CUDA
        
        if use_cuda_in_cpp:
            logger.info("Using C++ CUDA mode: C++ thread performs GPU->CPU transfer")
        else:
            logger.info("Using PyTorch mode: PyTorch performs GPU->CPU transfer, C++ coordinates")
        
        # Organize tensors by layer for layer-wise transfer
        # Extract layer numbers from tensor FQNs (Fully Qualified Names)
        layer_groups = {}
        
        # Map to store CPU tensors by (bucket_idx, tensor_idx)
        cpu_tensor_map = {}
        
        # Helper function to extract layer number from FQN
        def extract_layer_number(fqn: str) -> int:
            """Extract layer number from FQN like 'decoder.layers.0.weight' -> 0
            Returns -1 for non-layer tensors (embeddings, output layers, etc.)
            
            Supports patterns:
            - decoder.layers.N.
            - encoder.layers.N.
            - transformer.layers.N.
            - model.layers.N.
            - layers.N.
            """
            import re
            # Match patterns like .layers.N. or .layer.N. (with or without leading component)
            # Also match patterns at start of string or after underscore
            patterns = [
                r'\.layers\.(\d+)\.',      # .layers.N.
                r'^layers\.(\d+)\.',       # layers.N. at start
                r'\.layer\.(\d+)\.',       # .layer.N.
                r'^layer\.(\d+)\.',        # layer.N. at start
                r'_layers_(\d+)_',         # _layers_N_
                r'_layer_(\d+)_',          # _layer_N_
                r'\.blocks\.(\d+)\.',      # .blocks.N.
                r'^blocks\.(\d+)\.',       # blocks.N. at start
                r'_blocks_(\d+)_',         # _blocks_N_
            ]
            for pattern in patterns:
                match = re.search(pattern, fqn)
                if match:
                    return int(match.group(1))
            return -1  # Non-layer tensor
        
        # Group tensors within each bucket by layer
        # Track sample FQNs and item attributes for debugging
        sample_fqns = []
        
        # Strategy: Since FQN doesn't contain layer number (e.g., "decoder.layers.xxx"),
        # we need to infer layer number from tensor order and FQN patterns.
        # For ShardedTensors, same FQN appears multiple times for different layers.
        fqn_to_occurrences = {}  # Track how many times each FQN appears (indicates number of layers)
        
        for bucket_idx, bucket in enumerate(write_buckets):
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            
            # First pass: count occurrences of each FQN pattern
            for tensor_idx, (item, tensor) in enumerate(tensor_data):
                if hasattr(item, 'index') and hasattr(item.index, 'fqn'):
                    fqn = item.index.fqn
                    # Extract base FQN pattern (normalize to pattern without layer number)
                    import re
                    base_fqn = fqn
                    # If FQN contains .layers.N. (with number), remove the number
                    if re.search(r'\.layers\.\d+\.', fqn):
                        base_fqn = re.sub(r'\.layers\.\d+\.', '.layers.', fqn)
                    elif re.search(r'^layers\.\d+\.', fqn):
                        base_fqn = re.sub(r'^layers\.\d+\.', 'layers.', fqn)
                    # If FQN contains .layers. but no number (like decoder.layers.xxx), use as-is
                    # This is already the base pattern
                    
                    fqn_to_occurrences[base_fqn] = fqn_to_occurrences.get(base_fqn, 0) + 1
        
        # Determine if this is a layer-based FQN pattern
        # If same FQN appears multiple times (e.g., 12 times for 12 layers), it's a layer tensor
        layer_fqn_patterns = set()
        for fqn, count in fqn_to_occurrences.items():
            if count > 1 and ('layers.' in fqn or 'layer.' in fqn):
                layer_fqn_patterns.add(fqn)
        
        logger.info(f"Found {len(layer_fqn_patterns)} layer FQN patterns (appearing multiple times)")
        if layer_fqn_patterns and logger.isEnabledFor(logging.DEBUG):
            for pattern in sorted(list(layer_fqn_patterns))[:5]:
                logger.debug(f"  Layer pattern: {pattern} (appears {fqn_to_occurrences[pattern]} times)")
        
        # Second pass: assign layer numbers based on FQN pattern and occurrence order
        fqn_to_layer_counter = {}  # Track current layer number for each FQN pattern
        
        for bucket_idx, bucket in enumerate(write_buckets):
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            
            # Process each tensor in this bucket
            for tensor_idx, (item, tensor) in enumerate(tensor_data):
                # Extract layer number from FQN or infer from pattern
                if hasattr(item, 'index') and hasattr(item.index, 'fqn'):
                    fqn = item.index.fqn
                    
                    # Try direct extraction first
                    layer_num = extract_layer_number(fqn)
                    
                    # If not found, try to infer from FQN pattern
                    if layer_num == -1:
                        import re
                        base_fqn = fqn
                        # Normalize to base pattern (remove layer number if present)
                        if re.search(r'\.layers\.\d+\.', fqn):
                            base_fqn = re.sub(r'\.layers\.\d+\.', '.layers.', fqn)
                        elif re.search(r'^layers\.\d+\.', fqn):
                            base_fqn = re.sub(r'^layers\.\d+\.', 'layers.', fqn)
                        # If FQN contains .layers. but no number, use as-is
                        
                        # If this is a layer pattern (appears multiple times), assign layer number based on occurrence
                        if base_fqn in layer_fqn_patterns:
                            if base_fqn not in fqn_to_layer_counter:
                                fqn_to_layer_counter[base_fqn] = 0
                            layer_num = fqn_to_layer_counter[base_fqn]
                            fqn_to_layer_counter[base_fqn] += 1
                    
                    # Collect sample FQNs for debugging (first 10)
                    if len(sample_fqns) < 10:
                        sample_fqns.append(f"{fqn} -> layer_{layer_num}")
                else:
                    # Fallback if no FQN available
                    fqn = str(item)
                    layer_num = -1
                    if len(sample_fqns) < 10:
                        sample_fqns.append(f"{fqn} -> no FQN")
                
                # Use layer number as key, group non-layer tensors together
                layer_key = f"layer_{layer_num}" if layer_num >= 0 else "non_layer"
                
                if layer_key not in layer_groups:
                    layer_groups[layer_key] = []
                
                # Store (bucket_idx, tensor_idx, item, tensor) for this layer
                layer_groups[layer_key].append((bucket_idx, tensor_idx, item, tensor))
        
        # Log sample FQNs for debugging
        if sample_fqns:
            logger.info(f"Sample tensor FQNs and layer extraction:")
            for sample in sample_fqns:
                logger.info(f"  {sample}")
        
        total_tensors = sum(len(tensors) for tensors in layer_groups.values())
        logger.info(f"Organized {total_tensors} tensors into {len(layer_groups)} layer groups")
        
        # Log layer distribution for debugging
        if logger.isEnabledFor(logging.DEBUG):
            for layer_key, tensors in sorted(layer_groups.items()):
                logger.debug(f"  {layer_key}: {len(tensors)} tensors")
        
        # Process each layer group - prepare tensors and submit to C++ thread
        # Sort layer groups by layer number for sequential processing
        sorted_layer_groups = sorted(
            layer_groups.items(),
            key=lambda x: int(x[0].split('_')[1]) if x[0] != "non_layer" else -1
        )
        
        for layer_id, (layer_key, tensor_list) in enumerate(sorted_layer_groups):
            layer_start_time = time()
            
            # Collect all tensors for this layer
            layer_tensors_info = []
            
            for bucket_idx, tensor_idx, item, tensor in tensor_list:
                if tensor.is_cuda:
                    # Allocate CPU buffer (pinned memory for faster transfer)
                    cpu_tensor = torch.empty_like(tensor, device='cpu', pin_memory=True)
                    
                    if not use_cuda_in_cpp:
                        # PyTorch mode: Initiate async transfer now
                        # The C++ thread will just ensure layer-by-layer completion
                        cpu_tensor.copy_(tensor, non_blocking=True)
                    # else: CUDA mode - C++ will do the actual transfer
                    
                    # Store CPU tensor for later result construction using index
                    cpu_tensor_map[(bucket_idx, tensor_idx)] = cpu_tensor
                    
                    # Get FQN for logging
                    fqn = item.index.fqn if hasattr(item, 'index') and hasattr(item.index, 'fqn') else str(item)
                    
                    # Prepare info for C++ thread
                    tensor_info = (
                        tensor.data_ptr(),           # GPU pointer (source)
                        cpu_tensor.data_ptr(),       # CPU pointer (destination)
                        cpu_tensor.numel() * cpu_tensor.element_size(),  # Size in bytes
                        list(cpu_tensor.shape),      # Shape
                        fqn                          # Name (FQN)
                    )
                    layer_tensors_info.append(tensor_info)
                else:
                    # Already on CPU, store directly using index
                    cpu_tensor_map[(bucket_idx, tensor_idx)] = tensor
            
            # Submit this layer to C++ processor for GPU->CPU transfer
            if layer_tensors_info:
                processor.submit_layer(layer_id, layer_tensors_info)
                total_bytes = sum(info[2] for info in layer_tensors_info)
                logger.info(f"Layer {layer_id} ({layer_key}): submitted {len(layer_tensors_info)} tensors ({total_bytes/(1024**2):.2f} MB) for transfer")
                
                # Log each tensor's name and size in this layer
                for tensor_info in layer_tensors_info:
                    tensor_name = tensor_info[4]  # FQN is the 5th element (index 4)
                    tensor_size_bytes = tensor_info[2]  # Size is the 3rd element (index 2)
                    tensor_shape = tensor_info[3]  # Shape is the 4th element (index 3)
                    logger.info(f"  Tensor: {tensor_name}, Size: {tensor_size_bytes/(1024**2):.2f} MB, Shape: {tensor_shape}")
        
        # Wait for C++ thread to finish processing all layers
        logger.info("Waiting for C++ thread to complete all layer transfers...")
        processor.wait_all_complete()
        
        # Synchronize CUDA to ensure all transfers are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Get statistics
        stats = processor.get_stats()
        cpp_results = processor.get_results()
        
        # Clean up processor
        processor.stop()
        
        # Log detailed results
        for layer_result in cpp_results:
            layer_id, success, transfer_time, total_bytes, error_msg = layer_result
            if success:
                logger.debug(f"Layer {layer_id}: transferred {total_bytes/1e6:.2f} MB in {transfer_time:.4f}s")
            else:
                logger.warning(f"Layer {layer_id} transfer failed: {error_msg}")
        
        # Now construct the result buckets with CPU tensors
        result = []
        for bucket_idx, bucket in enumerate(write_buckets):
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            
            # Build tensor_data with CPU tensors
            cpu_tensor_data = []
            for tensor_idx, (item, original_tensor) in enumerate(tensor_data):
                # Get the CPU tensor from our map using index
                cpu_tensor = cpu_tensor_map.get((bucket_idx, tensor_idx), original_tensor)
                cpu_tensor_data.append((item, cpu_tensor))
            
            result.append((file_name, storage_key, (bytes_data, cpu_tensor_data)))
        
        total_time = time() - start_time
        logger.info(
            f"Layer-wise tensor preloading completed: "
            f"{stats['tasks_completed']} layers, {len(write_buckets)} buckets in {total_time:.4f}s"
        )
        
        return result

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_multiproc(
        transform_list: List[_StorageWriterTransforms],
        use_msc: bool,
        rank: int,
        write_buckets: List[WriteBucket],
        global_results_queue: mp.Queue,
    ) -> None:
        """
        Performs saving data to storage with multiple processes.

        Starts predefined number of processes and uses 2 queues to make sure the results
        are complete:
        - local_results_queue - to send the actual results
        - count_queue - small queue to mark worker as completed

        Using just one queue disallowed proper exception handling.

        This method is meant to be run in a forked subprocess.
        Triggering GC during execution leads to CUDA errors
        (cleaning up tensors owned by the parent process).
        To prevent this, we disable the GC explicitly for this function with _disable_gc.

        Args:
            write_buckets (List[WriteBucket]): write plan (may contain EC-CHECK data)
            global_results_queue (mp.Queue): mp.Queue to collect Dict[List[WriteResults]]
                (or an Exception) from parallel write processes to the main training process
        Returns: None
        """
        import sys
        # print(f"EC-CHECK: Write preloaded data multiproc started", file=sys.stdout)
        logger = logging.getLogger(__name__)
        # logger.info(f"EC-CHECK: Write preloaded data multiproc started")
        w_start = time()
        write_results_or_exc: Union[dict, Exception] = dict()
        ctx = mp.get_context("fork")
        local_results_queue = ctx.Queue()
        count_queue = ctx.JoinableQueue()
        p_list = []
        for i, write_bucket in enumerate(write_buckets):
            try:
                count_queue.put(i)

                kwargs = {
                    "local_proc_idx": i,
                    "write_bucket": write_bucket,
                    "results_queue": local_results_queue,
                    "count_queue": count_queue,
                    "use_fsync": True,
                }

                if use_msc:
                    import inspect

                    # Remove the inspect after the test_async_save.py is fixed.
                    signature = inspect.signature(FileSystemWriterAsync.write_preloaded_data)
                    if len(signature.parameters) > 6:
                        kwargs["use_msc"] = use_msc

                p_list.append(
                    ctx.Process(
                        target=partial(FileSystemWriterAsync.write_preloaded_data, transform_list),
                        kwargs=kwargs,
                    )
                )
            except Exception as e:
                err_msg = f"An error is caught while a proc {i} is created, error: {e}"
                logger.error(err_msg)
                write_results_or_exc = RuntimeError(err_msg)

        if not isinstance(write_results_or_exc, Exception):
            # logger.info(f"EC-CHECK: Starting {len(p_list)} write processes...")
            for p in p_list:
                p.start()
                logger.info(f"EC-CHECK: Started process {p.pid}")

            # logger.debug("FileSystemWriterAsync: collecting worker results...")

            # To make sure all nodes are completed
            # logger.info("EC-CHECK: Waiting for all processes to complete (count_queue.join)...")
            count_queue.join()
            # logger.info("EC-CHECK: All processes completed (count_queue.join returned)")

            # At this point, all workers completed, so the queue should have exactly
            # `len(write_buckets)` items
            for proc_idx in range(len(write_buckets)):
                try:
                    local_proc_idx, local_results_or_exc = local_results_queue.get()
                except queue.Empty:
                    write_results_or_exc = RuntimeError(
                        "Unexpected empty `local_results_queue`"
                        f" (got only {proc_idx}/{len(write_buckets)} items)"
                    )
                    break
                else:
                    if isinstance(local_results_or_exc, Exception):
                        err_msg = (
                            f"Local process {local_proc_idx} encountered"
                            f" an error: {local_results_or_exc}"
                        )
                        logger.error(err_msg)
                        write_results_or_exc = local_results_or_exc
                        break
                    assert isinstance(local_results_or_exc, list), type(local_results_or_exc)
                    write_results_or_exc[local_proc_idx] = local_results_or_exc
                    p_list[local_proc_idx].join()

            logger.debug("FileSystemWriterAsync: collected worker results successfully")

        global_results_queue.put(write_results_or_exc)

        w_end = time()
        logger.debug(f"{w_end}, rank: {rank}, write(sync,parallel): {w_end - w_start}")
        print(f"{w_end}, rank: {rank}, write(sync,parallel): {w_end - w_start}")
    
    @staticmethod
    @_disable_gc()
    def write_preloaded_data(
        transform_list: List[_StorageWriterTransforms],
        local_proc_idx: int,
        write_bucket: WriteBucket,
        results_queue: mp.SimpleQueue,
        count_queue: mp.JoinableQueue,
        use_fsync: bool,
        **kwargs,
    ) -> None:
        """
        Performs actual data saving to storage.

        Args:
            local_proc_idx (int): index of a local process that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (mp.Queue): queue to return the write results
                to the proxy checkpoint process.
            count_queue (mp.JoinableQueue): queue to marks worker task as completed
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None, the write result are put into the `queue`
        """
        logger = logging.getLogger(__name__)
        logger.info(f"EC-CHECK: Process {local_proc_idx} started (write_preloaded_data)")
        logger.debug(f"{local_proc_idx} started")
        mem_before = _process_memory()
        use_msc = kwargs.get("use_msc", False)

        local_results = []
        try:
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            
            # Check if this is EC-CHECK mode by detecting special markers in bytes_data
            eccheck_metadata = None
            eccheck_continuous_buffer = None
            if len(bytes_data) > 0 and bytes_data[0][0] == 'eccheck_metadata':
                # EC-CHECK mode detected
                logger.info(f"EC-CHECK: Process {local_proc_idx} detected EC-CHECK mode")
                for key, value in bytes_data:
                    if key == 'eccheck_metadata':
                        eccheck_metadata = value
                    elif key == 'eccheck_continuous_buffer':
                        eccheck_continuous_buffer = value
            
            # Check if this is ECLATIN mode by detecting special markers in bytes_data
            eclatin_metadata = None
            eclatin_continuous_buffer = None
            if len(bytes_data) > 0 and bytes_data[0][0] == 'eclatin_metadata':
                # ECLATIN mode detected
                logger.info(f"ECLATIN: Process {local_proc_idx} detected ECLATIN mode")
                for key, value in bytes_data:
                    if key == 'eclatin_metadata':
                        eclatin_metadata = value
                    elif key == 'eclatin_continuous_buffer':
                        eclatin_continuous_buffer = value
            
            # ECLATIN mode: save three components to ONE file (similar to ECCHECK)
            if eclatin_metadata is not None:
                if use_msc:
                    import multistorageclient as msc
                    open_file = msc.open
                else:
                    open_file = open
                
                write_start = time()
                logger.info("ECLATIN: Saving three components to single file...")
                
                # Get file path (file_name is the full path)
                eclatin_file_path = str(file_name)
                
                # Prepare header with component sizes
                import struct
                non_tensor_size = eclatin_metadata['non_tensor_size']
                tensor_keys_size = eclatin_metadata['tensor_keys_size']
                tensor_buffer_size = eclatin_metadata['tensor_buffer_size']
                
                # Determine if this is a main file, data block, or parity block
                is_main_file = storage_key.endswith('_0.distcp') or '_0.distcp' in storage_key
                is_data_block = 'data_block' in storage_key
                is_parity_block = 'parity_block' in storage_key
                block_type = "main" if is_main_file else ("data" if is_data_block else ("parity" if is_parity_block else "unknown"))
                
                if eclatin_continuous_buffer is not None:
                    buffer_size = eclatin_continuous_buffer.numel()
                    
                    if is_main_file:
                        # Main file: write full tensor_buffer_size (actual data only, exclude padding)
                        write_size = tensor_buffer_size
                        if write_size > buffer_size:
                            logger.warning(
                                f"ECLATIN: Write size ({write_size}) > buffer size ({buffer_size}), "
                                f"writing entire buffer"
                            )
                            write_size = buffer_size
                    elif is_data_block:
                        # Data blocks: write only actual data portion (half of total tensor_buffer_size)
                        # Each data block stores half of the total data
                        write_size = tensor_buffer_size // 2
                        if write_size > buffer_size:
                            logger.warning(
                                f"ECLATIN: Write size ({write_size}) > buffer size ({buffer_size}), "
                                f"writing entire buffer"
                            )
                            write_size = buffer_size
                    elif is_parity_block:
                        # Parity blocks: write full aligned half size (aligned_half_block_size)
                        write_size = buffer_size
                    else:
                        # Fallback: write actual data size
                        write_size = min(tensor_buffer_size, buffer_size)
                        logger.warning(
                            f"ECLATIN: Unknown block type in storage_key '{storage_key}', "
                            f"writing {write_size} bytes"
                        )
                    
                    # Header format: magic(4) + padding(4) + 3 sizes(8 each) = 32 bytes
                    # Magic number: 'ECLT' (ECLATIN)
                    header = struct.pack(
                        '4sQQQ',
                        b'ECLT',              # Magic number
                        non_tensor_size,      # Component 1 size
                        tensor_keys_size,     # Component 2 size
                        write_size,           # Component 3 size (actual or aligned)
                    )
                    
                    # Write all three components to one file
                    with open_file(eclatin_file_path, "wb") as f:
                        # Write header
                        header_start = time()
                        f.write(header)
                        logger.debug(f"ECLATIN: Wrote header in {time() - header_start:.4f}s")
                        
                        # Write Component 1: Non-tensor key-value pairs
                        comp1_start = time()
                        f.write(eclatin_metadata['non_tensor_data'])
                        comp1_time = time() - comp1_start
                        logger.debug(f"ECLATIN: Wrote Component 1 ({non_tensor_size / 1024:.2f} KB) in {comp1_time:.4f}s")
                        
                        # Write Component 2: Tensor keys
                        comp2_start = time()
                        f.write(eclatin_metadata['tensor_keys_data'])
                        comp2_time = time() - comp2_start
                        logger.debug(f"ECLATIN: Wrote Component 2 ({tensor_keys_size / 1024:.2f} KB) in {comp2_time:.4f}s")
                        
                        # Write Component 3: Block data
                        component3_start = time()
                        import numpy as np
                        np_array = eclatin_continuous_buffer[:write_size].numpy()  # Zero-copy view
                        mv = memoryview(np_array)
                        
                        # Write data at once
                        f.write(mv)
                        component3_size = mv.nbytes
                        
                        # Verify size matches
                        if component3_size != write_size:
                            logger.warning(
                                f"ECLATIN: Size mismatch: wrote {component3_size} bytes, "
                                f"expected {write_size} bytes"
                            )
                        
                        component3_time = time() - component3_start
                        bandwidth = (component3_size / (1024**3)) / component3_time if component3_time > 0 else 0
                        if is_main_file:
                            logger.info(
                                f"ECLATIN: Wrote Component 3 ({component3_size / (1024**3):.2f} GB) "
                                f"in {component3_time:.2f}s ({bandwidth:.2f} GB/s), "
                                f"main file (actual data only, excluded {buffer_size - write_size} bytes padding)"
                            )
                        else:
                            logger.info(
                                f"ECLATIN: Wrote Component 3 ({component3_size / (1024**3):.2f} GB) "
                                f"in {component3_time:.2f}s ({bandwidth:.2f} GB/s), "
                                f"{block_type} block (excluded {buffer_size - write_size} bytes padding)"
                            )
                        
                        # Flush to disk
                        if use_fsync:
                            if use_msc:
                                f.fsync()
                            else:
                                os.fsync(f.fileno())
                    
                    total_size = len(header) + non_tensor_size + tensor_keys_size + component3_size
                    total_write_time = time() - write_start
                    overall_bandwidth = (total_size / (1024**3)) / total_write_time if total_write_time > 0 else 0
                    
                    logger.info(
                        f"ECLATIN: Saved all components in {total_write_time:.2f}s:\n"
                        f"  File: {eclatin_file_path}\n"
                        f"  Block type: {block_type}\n"
                        f"  Total size: {total_size / (1024**3):.2f} GB\n"
                        f"  Overall bandwidth: {overall_bandwidth:.2f} GB/s\n"
                        f"  Breakdown:\n"
                        f"    Header: 32 bytes\n"
                        f"    Component 1: {non_tensor_size / 1024:.2f} KB ({comp1_time:.4f}s)\n"
                        f"    Component 2: {tensor_keys_size / 1024:.2f} KB ({comp2_time:.4f}s)\n"
                        f"    Component 3: {component3_size / (1024**3):.2f} GB ({component3_time:.2f}s)"
                    )
                else:
                    logger.error("ECLATIN: Continuous buffer is None, cannot write Component 3")
                
                # Create dummy results for compatibility
                local_results = []
            
            # EC-CHECK mode: save three components to ONE file
            elif eccheck_metadata is not None:
                if use_msc:
                    import multistorageclient as msc
                    open_file = msc.open
                else:
                    open_file = open
                
                write_start = time()
                logger.info("EC-CHECK: Saving three components to single file...")
                
                # Get file path (file_name is the eccheck_file_path)
                eccheck_file_path = eccheck_metadata['eccheck_file_path']
                
                # Prepare header with component sizes
                import struct
                non_tensor_size = eccheck_metadata['non_tensor_size']
                tensor_keys_size = eccheck_metadata['tensor_keys_size']
                tensor_buffer_size = eccheck_metadata['tensor_buffer_size']
                
                # Header format: magic(4) + padding(4) + 3 sizes(8 each) = 32 bytes
                # Magic number: 'ECCK' (EC-CHECK)
                # Default format includes padding for alignment
                header = struct.pack(
                    '4sQQQ',
                    b'ECCK',              # Magic number
                    non_tensor_size,      # Component 1 size
                    tensor_keys_size,     # Component 2 size
                    tensor_buffer_size,   # Component 3 size
                )
                
                # Write all three components to one file
                with open_file(eccheck_file_path, "wb") as f:
                    # Write header
                    header_start = time()
                    f.write(header)
                    logger.debug(f"EC-CHECK: Wrote header in {time() - header_start:.4f}s")
                    
                    # Write Component 1: Non-tensor key-value pairs
                    comp1_start = time()
                    f.write(eccheck_metadata['non_tensor_data'])
                    comp1_time = time() - comp1_start
                    logger.debug(f"EC-CHECK: Wrote Component 1 ({non_tensor_size / 1024:.2f} KB) in {comp1_time:.4f}s")
                    
                    # Write Component 2: Tensor keys
                    comp2_start = time()
                    f.write(eccheck_metadata['tensor_keys_data'])
                    comp2_time = time() - comp2_start
                    logger.debug(f"EC-CHECK: Wrote Component 2 ({tensor_keys_size / 1024:.2f} KB) in {comp2_time:.4f}s")
                    
                    # Write Component 3: Tensor data
                    # Only write actual data (exclude padding zeros for pipeline synchronization)
                    component3_start = time()
                    component3_size = 0
                    
                    if eccheck_continuous_buffer is not None:
                        # Write only actual data portion (exclude padding zeros)
                        import numpy as np
                        # tensor_buffer_size from metadata is the actual data size
                        actual_size = tensor_buffer_size
                        buffer_size = eccheck_continuous_buffer.numel()
                        
                        if actual_size > buffer_size:
                            logger.warning(
                                f"EC-CHECK: Actual size ({actual_size}) > buffer size ({buffer_size}), "
                                f"writing entire buffer"
                            )
                            actual_size = buffer_size
                        
                        # Only write the actual data portion (exclude padding)
                        np_array = eccheck_continuous_buffer[:actual_size].numpy()  # Zero-copy view
                        mv = memoryview(np_array)
                        
                        # Write actual data at once
                        f.write(mv)
                        component3_size = mv.nbytes
                        
                        # Verify size matches
                        if component3_size != tensor_buffer_size:
                            logger.warning(
                                f"EC-CHECK: Size mismatch: wrote {component3_size} bytes, "
                                f"expected {tensor_buffer_size} bytes"
                            )
                        
                        component3_time = time() - component3_start
                        bandwidth = (component3_size / (1024**3)) / component3_time if component3_time > 0 else 0
                        logger.info(
                            f"EC-CHECK: Wrote Component 3 ({component3_size / (1024**3):.2f} GB) "
                            f"in {component3_time:.2f}s ({bandwidth:.2f} GB/s), "
                            f"actual data only (excluded {buffer_size - actual_size} bytes padding)"
                        )
                    else:
                        logger.error("EC-CHECK: Continuous buffer is None, cannot write Component 3")
                    
                    # Flush to disk
                    if use_fsync:
                        if use_msc:
                            f.fsync()
                        else:
                            os.fsync(f.fileno())
                
                total_size = len(header) + non_tensor_size + tensor_keys_size + component3_size
                total_write_time = time() - write_start
                overall_bandwidth = (total_size / (1024**3)) / total_write_time if total_write_time > 0 else 0
                
                logger.info(
                    f"EC-CHECK: Saved all components in {total_write_time:.2f}s:\n"
                    f"  File: {eccheck_file_path}\n"
                    f"  Total size: {total_size / (1024**3):.2f} GB\n"
                    f"  Overall bandwidth: {overall_bandwidth:.2f} GB/s\n"
                    f"  Breakdown:\n"
                    f"    Header: 28 bytes\n"
                    f"    Component 1: {non_tensor_size / 1024:.2f} KB ({comp1_time:.4f}s)\n"
                    f"    Component 2: {tensor_keys_size / 1024:.2f} KB ({comp2_time:.4f}s)\n"
                    f"    Component 3: {component3_size / (1024**3):.2f} GB ({component3_time:.2f}s)"
                )
                
                # Create dummy results for compatibility
                local_results = []
            
            # Normal mode: standard write
            else:
                extra_kwargs = {}
                if "serialization_format" in inspect.signature(_write_item).parameters:
                    from torch.distributed.checkpoint.filesystem import SerializationFormat

                    extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE
                if use_msc:
                    import multistorageclient as msc

                    open_file = msc.open
                else:
                    open_file = open
                with open_file(file_name, "wb") as stream:
                    for write_item, data in bytes_data:
                        local_results.append(
                            _write_item(
                                *transform_list, stream, data, write_item, storage_key, **extra_kwargs
                            )
                        )

                    for write_item, tensor in tensor_data:
                        assert tensor.is_cpu
                        local_results.append(
                            _write_item(
                                *transform_list, stream, tensor, write_item, storage_key, **extra_kwargs
                            )
                        )

                    if use_fsync:
                        if use_msc:
                            stream.fsync()
                        else:
                            os.fsync(stream.fileno())
            
            local_output = (local_proc_idx, local_results)
        except Exception as e:
            logger.debug(f"{local_proc_idx} failed")
            local_output = (local_proc_idx, e)  # type: ignore[assignment]

        results_queue.put(local_output)
        # Signal this process is done.
        count_queue.get()
        count_queue.task_done()

        mem_after = _process_memory()
        logger.debug(
            f"{local_proc_idx} consumed: {mem_after - mem_before},"
            f" before: {mem_before}, after: {mem_after}"
        )

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[List[WriteResult]]:
        """Write all items from ``plan``."""
        raise NotImplementedError("write_data not implemented for FileSystemWriterAsync")

    def retrieve_write_results(self) -> List[WriteResult]:
        """
        Turn the latest dict including write results from `self.results_queue`
            into a single results lists. Includes error check.

        Returns (List[WriteResult]): the list of write results
            from all local processes performing the save.

        """
        assert self.write_buckets is not None

        if self.results_queue is None:
            write_results_or_exc = {}
        else:
            try:
                write_results_or_exc = self.results_queue.get_nowait()
            except queue.Empty:
                raise RuntimeError("results_queue should not be empty")

        if isinstance(write_results_or_exc, Exception):
            raise RuntimeError(f"Worker failure: {write_results_or_exc}") from write_results_or_exc
        write_results: dict = write_results_or_exc
        if len(write_results) != len(self.write_buckets):
            raise RuntimeError(
                f"Incomplete worker results (expected {len(self.write_buckets)},"
                f" got {len(write_results)}. This probably indicates a worker failure."
            )
        return list(chain.from_iterable(write_results.values()))

    def prepare_decentralized_global_plan(self, local_plan: SavePlan) -> SavePlan:
        """Instead of assigning indices by plan order, uses PyT rank (same outcome).

        Args:
            local_plan (SavePlan): local plan to turn to a global plan
                (without interactions with other ranks)

        Returns:
            SavePlan - locally transformed plan equivalent to the plan that would be
                created by the coordinator
        """
        return dataclasses.replace(
            local_plan, storage_data=_StoragePrefix(f"__{torch.distributed.get_rank()}_")
        )

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        """
        Finish the checkpointing process.

        Args:
            metadata (Metadata): metadata to save
            results (List[List[WriteResult]]): results to save
        """
        if self.use_msc:
            import multistorageclient as msc

            storage_md = dict()
            for wr_list in results:
                storage_md.update({wr.index: wr.storage_data for wr in wr_list})

            metadata.storage_data = storage_md
            metadata.storage_meta = self.storage_meta()

            path = os.path.join(self.checkpoint_dir, ".metadata")

            with msc.open(path, "wb") as metadata_file:
                pickle.dump(metadata, metadata_file)
        else:
            super().finish(metadata, results)

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Prepare the local plan for the checkpointing process.
        """
        if self.use_msc:
            import multistorageclient as msc

            msc.os.makedirs(str(self.checkpoint_dir), exist_ok=True)
        else:
            super().prepare_local_plan(plan)

        return plan

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
        return str(self.checkpoint_dir)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        Validate the checkpoint_id that will be used to save the checkpoint.

        This method is available in PyTorch 2.3 and above.
        """
        if checkpoint_id.startswith("msc://"):
            return True

        if hasattr(FileSystemWriter, "validate_checkpoint_id"):
            return FileSystemWriter.validate_checkpoint_id(checkpoint_id)

        return False
    
    def _prepare_eccheck_write_buckets(self, plan: SavePlan) -> None:
        """
        Prepare write buckets for EC-CHECK mode.
        
        In EC-CHECK mode, the serialized metadata is already prepared by torch.py.
        We just need to create write_buckets using that metadata.
        
        Args:
            plan (SavePlan): save plan
        """
        storage_plan: _StoragePrefix = plan.storage_data
        
        self.write_buckets = []
        
        # Use eccheck_serialized_metadata passed from torch.py
        if self.eccheck_serialized_metadata is None:
            raise RuntimeError("EC-CHECK: eccheck_serialized_metadata not set by strategy")
        
        # Get file path from metadata
        eccheck_path = self.eccheck_serialized_metadata.get('eccheck_file_path')
        if eccheck_path is None:
            # Build path if not set
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            eccheck_file = f"__{rank}_0.distcp"
            eccheck_path = os.path.join(self.checkpoint_dir, eccheck_file)
            self.eccheck_serialized_metadata['eccheck_file_path'] = eccheck_path
        
        logger.debug(
            f"EC-CHECK: Using serialized metadata from strategy:\n"
            f"  File: {eccheck_path}\n"
            f"  Component 1 size: {self.eccheck_serialized_metadata['non_tensor_size'] / 1024:.2f} KB\n"
            f"  Component 2 size: {self.eccheck_serialized_metadata['tensor_keys_size'] / 1024:.2f} KB\n"
            f"  Component 3 size: {self.eccheck_serialized_metadata['tensor_buffer_size'] / (1024**3):.2f} GB"
        )
        
        # Create a single write bucket for EC-CHECK
        # The actual data will be written by custom logic
        self.write_buckets.append((
            eccheck_path,  # Single file path
            storage_plan.prefix,
            ([], [])  # Will be handled specially in write_preloaded_data
        ))
        
        # Add P2P write buckets if P2P buffers are available
        # P2P buckets are tuples: (file_path, storage_key, (bytes_data, tensor_data))
        if hasattr(self, 'eccheck_p2p_buffers') and self.eccheck_p2p_buffers is not None:
            if 'own_write_bucket' in self.eccheck_p2p_buffers:
                # Unpack the tuple to get file_path and storage_key
                old_own_file_path, own_storage_key, own_eccheck_bytes_data = self.eccheck_p2p_buffers['own_write_bucket']
                # Update path with current checkpoint_dir
                own_file_name = Path(old_own_file_path).name if isinstance(old_own_file_path, (str, Path)) else str(old_own_file_path).split('/')[-1]
                own_file_path = Path(self.checkpoint_dir) / own_file_name
                self.write_buckets.append((own_file_path, own_storage_key, ([own_eccheck_bytes_data], [])))
                # logger.debug(f"EC-CHECK: Added own P2P write bucket to write_buckets with path {own_file_path}")
            if 'partner_write_bucket' in self.eccheck_p2p_buffers:
                # Unpack the tuple to get file_path and storage_key
                old_partner_file_path, partner_storage_key, partner_eccheck_bytes_data = self.eccheck_p2p_buffers['partner_write_bucket']
                # Update path with current checkpoint_dir
                partner_file_name = Path(old_partner_file_path).name if isinstance(old_partner_file_path, (str, Path)) else str(old_partner_file_path).split('/')[-1]
                partner_file_path = Path(self.checkpoint_dir) / partner_file_name
                self.write_buckets.append((partner_file_path, partner_storage_key, ([partner_eccheck_bytes_data], [])))
                # logger.debug(f"EC-CHECK: Added partner P2P write bucket to write_buckets with path {partner_file_path}")
        
        # Set up results queue
        if len(self.write_buckets) > 0:
            self.results_queue = _get_write_results_queue()
        else:
            self.results_queue = None 
    
    def _poll_and_release_buffers(self):
        """Poll C++ for buffers ready to be released and put them back to queues."""
        # If using shared buffers from strategy, use strategy's poll method
        if hasattr(self, '_strategy_poll_and_release_buffers') and self._strategy_poll_and_release_buffers:
            self._strategy_poll_and_release_buffers()
            return

    def _execute_phase3_encoding(self, m=None):
        """
        Execute Phase 3: Complete tensor data exchange and encoding process with pipeline.
        
        Args:
            m: Encoding parameter (uses self.eccheck_m if None)
        """
        if m is None:
            m = self.eccheck_m
            
        logger.info(f"EC-CHECK: Starting Phase 3 - Tensor data exchange and encoding (k={self.eccheck_k}, m={m})")
        phase3_start = time()
    
        # Phase 3.1: Copy tensor data to data buffers (producer)
        self._copy_tensor_data_to_buffers()
        
        phase3_time = time() - phase3_start
        logger.warning(f"EC-CHECK: Phase 3 completed in {phase3_time:.2f}s")
    
    # ===== Phase 3 - Pipeline implementation =====
    def _stop_phase3_workers(self) -> None:
        """Signal workers to stop and join them safely."""
        if self._eccheck_native is not None:
            # C++ implementation
            self._eccheck_native.stop_pipeline()
        else:
            # No fallback - C++ module is required
            raise RuntimeError("EC-CHECK: C++ native module is required but not available")
    
    
    def _copy_tensor_data_to_buffers(self) -> None:
        """Producer: memcpy from continuous tensor buffer into free data buffers, emit to encode queue.
        
        Only proceeds when a free data buffer is available. Emits (data_buf_index, used_size).
        """
        logger.info("EC-CHECK: Phase 3.1 - memcpy to data buffers with backpressure")
        
        if self._eccheck_native is None:
            raise RuntimeError("EC-CHECK: C++ native module is required but not available")
        
        # Direct implementation
        self._copy_tensor_data_to_buffers_pipeline()
    
    def _copy_tensor_data_to_buffers_pipeline(self) -> None:
        """
        Python memcpy implementation with C++ encoding coordination.
        
        Key improvements:
        1. Data buffers are released as soon as both C++ threads copy the data
        2. No need to wait for encoding completion to release data buffers
        3. Better resource utilization and reduced risk of deadlock
        """
        # logger.info("EC-CHECK: Phase 3.1 - Python memcpy to data buffers with C++ encoding")
        
        # Reset completion flags for new encoding round
        self._eccheck_native.reset_encoding_completion_flags()
        
        # Ensure we're in save mode (not load mode)
        self._eccheck_native.set_load_mode(False, -1)
        
        # Activate the persistent buffer poller at the start of pipeline
        # This ensures buffers can be released as soon as C++ threads finish using them
        if self._buffer_poller_active_event:
            self._buffer_poller_active_event.set()
            logger.info("EC-CHECK: Activated buffer poller for pipeline operation")
        
        try:
            self._copy_tensor_data_to_buffers_pipeline_impl()
        finally:
            # Deactivate the buffer poller
            if self._buffer_poller_active_event:
                self._buffer_poller_active_event.clear()
                logger.info("EC-CHECK: Deactivated buffer poller after pipeline completion")
            
            # Final poll to ensure all buffers are released
            self._poll_and_release_buffers()
    
    def _copy_tensor_data_to_buffers_pipeline_impl(self) -> None:
        """Implementation of the pipeline logic (called within try-finally block)."""
        
        def get_free_data_buffer():
            """Get a free data buffer address, blocking if none available."""
            # Poll for released buffers before trying to get one
            self._poll_and_release_buffers()
            
            try:
                return self._free_data_buffer_queue.get(timeout=5.0)
            except queue.Empty:
                logger.error("EC-CHECK: TIMEOUT waiting for free data buffer - possible deadlock!")
                # Print queue status for debugging
                logger.error(f"EC-CHECK: Data buffer queue size: {self._free_data_buffer_queue.qsize()}")
                return self._free_data_buffer_queue.get()
                # raise RuntimeError("EC-CHECK: Timeout waiting for data buffer")
        
        def get_free_encoding_buffer():
            """Get a free encoding buffer address, blocking if none available."""
            # Poll for released buffers before trying to get one
            self._poll_and_release_buffers()
            
            try:
                return self._free_encoding_buffer_queue.get(timeout=5.0)
            except queue.Empty:
                logger.error("EC-CHECK: TIMEOUT waiting for free encoding buffer - possible deadlock!")
                # Print queue status for debugging
                # logger.error(f"EC-CHECK: Encoding buffer queue size: {self._free_encoding_buffer_queue.qsize()}")
                return self._free_encoding_buffer_queue.get()
                # raise RuntimeError("EC-CHECK: Timeout waiting for encoding buffer")
        
        def get_free_parity_buffer():
            """Get a free parity buffer address, blocking if none available."""
            # Poll for released buffers before trying to get one
            self._poll_and_release_buffers()
            
            try:
                return self._free_parity_buffer_queue.get(timeout=5.0)
            except queue.Empty:
                logger.error("EC-CHECK: TIMEOUT waiting for free parity buffer - possible deadlock!")
                return self._free_parity_buffer_queue.get()
        
        # Process continuous tensor buffer sequentially
        # Copy data from self.tensor_buffer (continuous CPU buffer) to data buffers
        # Use pipeline_total_bytes to ensure all ranks have same iterations
        if hasattr(self, 'pipeline_total_bytes'):
            total_bytes = self.pipeline_total_bytes
        else:
            total_bytes = self.decomposed_state_dict.total_tensor_size_bytes
            logger.warning(
                "EC-CHECK: pipeline_total_bytes not set, using actual size. "
                "This may cause pipeline synchronization issues."
            )
        
        # Get actual data size for padding logic
        actual_data_bytes = getattr(self, 'actual_tensor_buffer_size', total_bytes)
        
        src_pos = 0  # Current position in continuous tensor buffer
        # chunk_count = 0
        
        # Get base addresses of TWO receive buffers (one per encoding thread)
        recv_buffer_thread1, recv_buffer_thread2 = self.eccheck_recv_encoding_buffers
        recv_buffer_base_addr_thread1 = int(recv_buffer_thread1.data_ptr())
        recv_buffer_base_addr_thread2 = int(recv_buffer_thread2.data_ptr())
        recv_buffer_offset_thread1 = 0  # Current offset in thread1's receive buffer
        recv_buffer_offset_thread2 = 0  # Current offset in thread2's receive buffer
        
        # Get base addresses of P2P buffers (own_buffer and partner_buffer)
        if self.eccheck_p2p_buffers is not None:
            own_buffer = self.eccheck_p2p_buffers['own_buffer']
            partner_buffer = self.eccheck_p2p_buffers['partner_buffer']
            p2p_own_buffer_base_addr = int(own_buffer.data_ptr())
            p2p_partner_buffer_base_addr = int(partner_buffer.data_ptr())
            p2p_own_buffer_offset = 0  # Current offset in own_buffer
            p2p_partner_buffer_offset = 0  # Current offset in partner_buffer
        else:
            logger.warning("EC-CHECK: P2P buffers not set, using 0 addresses")
            p2p_own_buffer_base_addr = 0
            p2p_partner_buffer_base_addr = 0
            p2p_own_buffer_offset = 0
            p2p_partner_buffer_offset = 0

        while src_pos < total_bytes:
            # Get a free data buffer (with timeout to detect deadlocks)
            cur_buffer_addr = get_free_data_buffer()
            
            # Calculate how much data to copy to this buffer
            remaining_in_source = total_bytes - src_pos
            take = min(self.eccheck_buffer_size, remaining_in_source)
            
            # Check recv_buffer bounds BEFORE copying data
            # This ensures we don't copy more data than can fit in recv buffers
            recv_buffer_size_thread1 = recv_buffer_thread1.numel()
            recv_buffer_size_thread2 = recv_buffer_thread2.numel()
            
            # Calculate aligned offsets to check available space
            recv_buffer_offset_thread1_aligned = ((recv_buffer_offset_thread1 + 63) // 64) * 64
            recv_buffer_offset_thread2_aligned = ((recv_buffer_offset_thread2 + 63) // 64) * 64
            
            remaining_space_thread1 = recv_buffer_size_thread1 - recv_buffer_offset_thread1_aligned
            remaining_space_thread2 = recv_buffer_size_thread2 - recv_buffer_offset_thread2_aligned
            max_available_recv_space = min(remaining_space_thread1, remaining_space_thread2)
            
            # Adjust 'take' if needed to fit within recv buffer bounds
            if take > max_available_recv_space:
                if max_available_recv_space < 64:
                    # Not even 64 bytes available - recv buffers are exhausted
                    logger.warning(
                        f"EC-CHECK: Recv buffers exhausted. "
                        f"Thread1 remaining: {remaining_space_thread1} bytes, "
                        f"Thread2 remaining: {remaining_space_thread2} bytes. "
                        f"Processed {src_pos / (1024**3):.2f} GB of {total_bytes / (1024**3):.2f} GB. "
                        f"Stopping data processing."
                    )
                    break  # Exit the loop
                
                # Adjust take to fit available space
                take = max_available_recv_space
                logger.debug(
                    f"EC-CHECK: Adjusted 'take' from {min(self.eccheck_buffer_size, remaining_in_source)} "
                    f"to {take} to fit recv buffer bounds"
                )
            
            # Python memcpy: copy from continuous tensor buffer to data buffer
            import ctypes
            buffer_ptr = ctypes.cast(cur_buffer_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer_array = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
            
            # Check if we need to pad with zeros (for ranks with smaller data)
            if src_pos < actual_data_bytes:
                # Still have actual data to copy
                bytes_to_copy = min(take, actual_data_bytes - src_pos)
                src_data = self.tensor_buffer[src_pos: src_pos + bytes_to_copy].numpy()
                
                # Copy actual data
                ctypes.memmove(buffer_array.contents, src_data.ctypes.data, bytes_to_copy)
                
                # Fill remaining space with zeros if needed
                if take > bytes_to_copy:
                    padding_size = take - bytes_to_copy
                    padding_ptr = ctypes.cast(
                        ctypes.addressof(buffer_array.contents) + bytes_to_copy,
                        ctypes.POINTER(ctypes.c_uint8)
                    )
                    ctypes.memset(padding_ptr, 0, padding_size)
            else:
                # Already past actual data, fill entire chunk with zeros
                ctypes.memset(buffer_array.contents, 0, take)
            
            # Get two encoding buffers (with timeout to detect deadlocks)
            enc_addr1 = get_free_encoding_buffer()
            enc_addr2 = get_free_encoding_buffer()
            
            # Get two parity buffers for XOR results
            parity_addr1 = get_free_parity_buffer()
            parity_addr2 = get_free_parity_buffer()
            
            # Allocate receive addresses from TWO recv_encoding_buffers
            # Each encoding thread gets its own receive address
            # 
            # CRITICAL: Addresses must be 64-byte aligned for ISA-L AVX512 XOR operations
            # Size does NOT need to be a multiple of 64 bytes (ISA-L handles this)
            # recv_chunk_size should match 'take' (already adjusted for buffer bounds above)
            recv_chunk_size = take
            
            # Align offsets to 64-byte boundary (address alignment requirement)
            # Note: We already checked bounds above, so aligned offset + recv_chunk_size should be safe
            recv_buffer_offset_thread1_aligned = ((recv_buffer_offset_thread1 + 63) // 64) * 64
            recv_buffer_offset_thread2_aligned = ((recv_buffer_offset_thread2 + 63) // 64) * 64
            
            # Thread1 receive address (guaranteed 64-byte aligned and within bounds)
            recv_addr_thread1 = recv_buffer_base_addr_thread1 + recv_buffer_offset_thread1_aligned
            recv_buffer_offset_thread1 = recv_buffer_offset_thread1_aligned + recv_chunk_size  # Update offset after processing
            
            # Thread2 receive address (guaranteed 64-byte aligned and within bounds)
            recv_addr_thread2 = recv_buffer_base_addr_thread2 + recv_buffer_offset_thread2_aligned
            recv_buffer_offset_thread2 = recv_buffer_offset_thread2_aligned + recv_chunk_size  # Update offset after processing
            
            # Verify address alignment and bounds (for debugging)
            assert recv_addr_thread1 % 64 == 0, (
                f"recv_addr_thread1 not 64-byte aligned: {hex(recv_addr_thread1)}, "
                f"base={hex(recv_buffer_base_addr_thread1)}, offset={recv_buffer_offset_thread1_aligned}"
            )
            assert recv_addr_thread2 % 64 == 0, (
                f"recv_addr_thread2 not 64-byte aligned: {hex(recv_addr_thread2)}, "
                f"base={hex(recv_buffer_base_addr_thread2)}, offset={recv_buffer_offset_thread2_aligned}"
            )
            assert recv_buffer_offset_thread1_aligned + recv_chunk_size <= recv_buffer_size_thread1, (
                f"recv_addr_thread1 out of bounds: offset={recv_buffer_offset_thread1_aligned}, "
                f"size={recv_chunk_size}, buffer_size={recv_buffer_size_thread1}"
            )
            assert recv_buffer_offset_thread2_aligned + recv_chunk_size <= recv_buffer_size_thread2, (
                f"recv_addr_thread2 out of bounds: offset={recv_buffer_offset_thread2_aligned}, "
                f"size={recv_chunk_size}, buffer_size={recv_buffer_size_thread2}"
            )
            
            # Calculate P2P write addresses (similar to recv addresses)
            # Both thread1 and thread2 use the same P2P addresses for the same chunk
            # CRITICAL: Addresses must be 64-byte aligned for ISA-L AVX512 operations
            if p2p_own_buffer_base_addr != 0:
                # Align offsets to 64-byte boundary (address alignment requirement)
                p2p_own_buffer_offset_aligned = ((p2p_own_buffer_offset + 63) // 64) * 64
                p2p_partner_buffer_offset_aligned = ((p2p_partner_buffer_offset + 63) // 64) * 64
                
                # Calculate aligned addresses
                p2p_own_write_addr = p2p_own_buffer_base_addr + p2p_own_buffer_offset_aligned
                p2p_partner_write_addr = p2p_partner_buffer_base_addr + p2p_partner_buffer_offset_aligned
                
                # Update offsets after processing (size does NOT need to be aligned)
                p2p_own_buffer_offset = p2p_own_buffer_offset_aligned + take
                p2p_partner_buffer_offset = p2p_partner_buffer_offset_aligned + take
                
                # Verify address alignment (for debugging)
                assert p2p_own_write_addr % 64 == 0, (
                    f"p2p_own_write_addr not 64-byte aligned: {hex(p2p_own_write_addr)}, "
                    f"base={hex(p2p_own_buffer_base_addr)}, offset={p2p_own_buffer_offset_aligned}"
                )
                assert p2p_partner_write_addr % 64 == 0, (
                    f"p2p_partner_write_addr not 64-byte aligned: {hex(p2p_partner_write_addr)}, "
                    f"base={hex(p2p_partner_buffer_base_addr)}, offset={p2p_partner_buffer_offset_aligned}"
                )
            else:
                p2p_own_write_addr = 0
                p2p_partner_write_addr = 0
            
            # Submit to BOTH encoding threads with their respective receive addresses and parity buffers
            # The C++ threads will mark the data buffer as copied immediately after reading
            # Once both threads mark it as copied, the data buffer will be released
            # recv_addr will be used by recv_worker to receive peer data
            # parity_addr will be used by XOR worker to store XOR results
            # p2p_own_write_addr and p2p_partner_write_addr will be used by P2P worker after XOR
            self._eccheck_native.submit_data_for_encoding_thread1(
                cur_buffer_addr, take, enc_addr1, recv_addr_thread1, recv_chunk_size, parity_addr1,
                p2p_own_write_addr, p2p_partner_write_addr
            )
            
            self._eccheck_native.submit_data_for_encoding_thread2(
                cur_buffer_addr, take, enc_addr2, recv_addr_thread2, recv_chunk_size, parity_addr2,
                p2p_own_write_addr, p2p_partner_write_addr
            )
            
            src_pos += take
            
        logger.info(
            f"  Thread1 receive buffer used: {recv_buffer_offset_thread1 / (1024**3):.2f} GB\n"
            f"  Thread2 receive buffer used: {recv_buffer_offset_thread2 / (1024**3):.2f} GB\n"
            f"  Total receive buffer used: {(recv_buffer_offset_thread1 + recv_buffer_offset_thread2) / (1024**3):.2f} GB"
        )
        
        # Mark end of stream for both encoders (with P2P addresses set to 0)
        self._eccheck_native.submit_data_for_encoding_thread1(0, 0, 0, 0, 0, 0, 0, 0)  # Sentinel for thread 1
        self._eccheck_native.submit_data_for_encoding_thread2(0, 0, 0, 0, 0, 0, 0, 0)  # Sentinel for thread 2
        
        # Wait for both encoding threads to complete
        # Buffer poller is already active (activated at function start)
        logger.info("EC-CHECK: Waiting for encoding threads to complete (with buffer polling)...")
        
        # Wait for encoding completion (this may block)
        # The buffer poller will continue running in the background
        self._eccheck_native.wait_for_encoding_completion()
        torch.cuda.synchronize()
        logger.info(f"EC-CHECK: Pipeline CUDA synchronized")
        
        # Buffer poller will be deactivated in the outer finally block
        # logger.info("EC-CHECK: All encoding operations completed")
    
    def _eclatin_preload_tensors_to_buffer(self, non_blocking: bool = True) -> List[WriteBucket]:
        """
        ECLATIN version: Transfer tensors from GPU to preallocated CPU buffer and submit to C++ pipelines.
        
        This method transfers tensor data from GPU to the preallocated CPU buffer
        in a pipelined manner, enabling overlap with subsequent encoding operations.
        
        Args:
            non_blocking (bool): if True, use non-blocking GPU-to-CPU transfer
        
        Returns:
            List[WriteBucket]: List of WriteBuckets for the 4 blocks
        """
        if not self.decomposed_state_dict:
            raise RuntimeError("ECLATIN: State dict not decomposed yet")
        
        logger.info("ECLATIN: Starting GPU-to-CPU tensor transfer...")
        start = time()
        
        # Step 1: Get actual data size for this rank
        actual_total_size = self.decomposed_state_dict.total_tensor_size_bytes
        
        # Step 2: Calculate maximum data size across all ranks
        if (torch.distributed.is_initialized() and 
            hasattr(self, 'eclatin_global_registry') and 
            self.eclatin_global_registry is not None):
            all_total_bytes_list = []
            for r in range(torch.distributed.get_world_size()):
                rank_metadata = self.eclatin_global_registry.rank_metadata.get(r, [])
                rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
                all_total_bytes_list.append(rank_total_size)
            max_total_bytes = max(all_total_bytes_list)
        else:
            max_total_bytes = actual_total_size
        
        # Step 3: Allocate buffer with maximum size (for pipeline synchronization)
        if self.preallocated_cpu_buffer is not None:
            buffer = self.preallocated_cpu_buffer
            if buffer.numel() < max_total_bytes:
                logger.warning(
                    f"ECLATIN: Preallocated buffer ({buffer.numel() / (1024**3):.2f} GB) "
                    f"is smaller than max_total_bytes ({max_total_bytes / (1024**3):.2f} GB). "
                    f"Reallocating..."
                )
                if self.eclatin_pin_memory and torch.cuda.is_available():
                    buffer = torch.empty(max_total_bytes, dtype=torch.uint8).pin_memory()
                else:
                    buffer = torch.empty(max_total_bytes, dtype=torch.uint8)
        else:
            if self.eclatin_pin_memory and torch.cuda.is_available():
                buffer = torch.empty(max_total_bytes, dtype=torch.uint8).pin_memory()
            else:
                buffer = torch.empty(max_total_bytes, dtype=torch.uint8)
        
        logger.info(
            f"ECLATIN: Allocated continuous CPU buffer: {max_total_bytes / (1024**3):.2f} GB "
            f"(actual data: {actual_total_size / (1024**3):.2f} GB, "
            f"padding: {(max_total_bytes - actual_total_size) / (1024**3):.2f} GB)"
        )
        
        # Step 4: Transfer tensors from GPU to continuous CPU buffer
        num_gpu_tensors = 0
        offset = 0
        for info, tensor in zip(
            self.decomposed_state_dict.tensor_infos,
            self.decomposed_state_dict.tensor_data
        ):
            tensor_size = info.size_bytes
            buffer_view = buffer[offset:offset + tensor_size]
            tensor_flat = tensor.flatten().contiguous().view(torch.uint8)
            buffer_view.copy_(tensor_flat, non_blocking=non_blocking)
            
            if tensor.device.type != 'cpu':
                num_gpu_tensors += 1
            
            info.offset = offset
            info.device = torch.device('cpu')
            offset += tensor_size
        
        # Step 5: Fill remaining space with zeros (for pipeline synchronization)
        if offset < max_total_bytes:
            padding_size = max_total_bytes - offset
            buffer[offset:max_total_bytes].fill_(0)
            logger.debug(
                f"ECLATIN: Filled {padding_size / (1024**2):.2f} MB with zeros "
                f"for pipeline synchronization"
            )
        
        # Synchronize if using non-blocking transfers
        if non_blocking and num_gpu_tensors > 0:
            torch.cuda.synchronize()
        
        # Step 6: Store the continuous buffer
        self.tensor_buffer = buffer
        self.actual_tensor_buffer_size = actual_total_size
        self.pipeline_total_bytes = max_total_bytes
        
        transfer_time = time() - start
        total_gb = actual_total_size / (1024**3)
        bandwidth = total_gb / transfer_time if transfer_time > 0 else 0
        
        logger.info(
            f"ECLATIN: Transferred {total_gb:.2f} GB in {transfer_time:.2f}s "
            f"({bandwidth:.2f} GB/s), {num_gpu_tensors} tensors from GPU to CPU"
        )
        
        # Step 7: Verify blocks and buffers are set
        if not hasattr(self, 'eclatin_blocks') or self.eclatin_blocks is None:
            raise RuntimeError(
                "ECLATIN: Blocks not set. Should be passed from strategy "
                "after _prepare_eclatin_data completes."
            )
        
        if not hasattr(self, 'eclatin_data_buffers') or self.eclatin_data_buffers is None:
            raise RuntimeError(
                "ECLATIN: Data buffers not set. Should be passed from strategy."
            )
        
        if not hasattr(self, 'eclatin_recv_buffers') or self.eclatin_recv_buffers is None:
            raise RuntimeError(
                "ECLATIN: Recv buffers not set. Should be passed from strategy."
            )
        
        # Step 8: Execute pipelines
        exec_start = time()
        self._execute_eclatin_pipelines()
        exec_time = time() - exec_start
        
        logger.info(f"ECLATIN: Pipeline execution completed in {exec_time:.2f}s")
        duration = time() - start
        logger.warning(f"ECLATIN: Pipeline execution completed in {duration:.2f}s")

        # Step 9: Update self.write_buckets and return (consistent with ECCHECK)
        # This ensures retrieve_write_results() can check the correct count
        if hasattr(self, 'ecl_write_buckets') and self.ecl_write_buckets:
            # Update paths with current checkpoint_dir (similar to ECCHECK)
            result_buckets = []
            
            # Extract metadata from first block (all blocks use the same metadata)
            first_bucket = self.ecl_write_buckets[0]
            _, _, (first_bytes_data, _) = first_bucket
            
            # Extract metadata for main file
            main_file_metadata = None
            for key, value in first_bytes_data:
                if key == 'eclatin_metadata':
                    main_file_metadata = value
                    break
            
            # Add main file bucket (similar to ECCHECK)
            if main_file_metadata:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                eclatin_main_file = f"__{rank}_0.distcp"
                eclatin_main_path = Path(self.checkpoint_dir) / eclatin_main_file
                
                # Use same metadata but with full tensor_buffer instead of block tensor
                eclatin_main_bytes_data = [
                    ('eclatin_metadata', main_file_metadata),
                    ('eclatin_continuous_buffer', self.tensor_buffer),  # Full tensor_buffer
                ]
                result_buckets.append((eclatin_main_path, eclatin_main_file, (eclatin_main_bytes_data, [])))
                logger.debug(f"ECLATIN: Added main file bucket: {eclatin_main_path}")
            
            # Add 4 block buckets
            for bucket in self.ecl_write_buckets:
                file_path, storage_key, data = bucket
                # Extract file name from path
                if isinstance(file_path, (str, Path)):
                    file_path_obj = Path(file_path)
                    file_name = file_path_obj.name
                else:
                    file_name = str(file_path).split('/')[-1] if '/' in str(file_path) else str(file_path)
                
                # Build new path with current checkpoint_dir
                new_file_path = Path(self.checkpoint_dir) / file_name
                result_buckets.append((new_file_path, storage_key, data))
            
            # Update self.write_buckets so retrieve_write_results() can check the correct count
            self.write_buckets = result_buckets
            return result_buckets
        else:
            return []

    def _execute_eclatin_pipelines(self) -> None:
        """
        Execute ECLATIN pipelines: Copy data to blocks and submit to 6 C++ pipelines.
        """
        logger.info("ECLATIN: Starting pipeline execution...")
        
        # Activate buffer poller if available
        if hasattr(self, '_eclatin_buffer_poller_active_event'):
            if self._eclatin_buffer_poller_active_event is not None:
                self._eclatin_buffer_poller_active_event.set()
                logger.debug("ECLATIN: Activated buffer poller")
        
        try:
            # Copy tensor data to blocks and submit to pipelines
            self._copy_tensor_data_to_eclatin_blocks()
        finally:
            # Deactivate buffer poller
            if hasattr(self, '_eclatin_buffer_poller_active_event'):
                if self._eclatin_buffer_poller_active_event is not None:
                    self._eclatin_buffer_poller_active_event.clear()
                    logger.debug("ECLATIN: Deactivated buffer poller")

    def _copy_tensor_data_to_eclatin_blocks(self) -> None:
        """Copy tensor data to 4 blocks and submit to 6 C++ pipelines."""
        
        def get_free_data_buffer():
            """Get a free data buffer address, blocking if none available."""
            if hasattr(self, '_eclatin_strategy_poll_and_release_buffers'):
                self._eclatin_strategy_poll_and_release_buffers()
            
            try:
                return self._free_eclatin_data_buffer_queue.get(timeout=5.0)
            except queue.Empty:
                logger.error("ECLATIN: TIMEOUT waiting for free data buffer!")
                logger.error(f"ECLATIN: Data buffer queue size: {self._free_eclatin_data_buffer_queue.qsize()}")
                return self._free_eclatin_data_buffer_queue.get()
        
        def get_free_recv_buffer():
            """Get a free recv buffer address, blocking if none available."""
            if hasattr(self, '_eclatin_strategy_poll_and_release_buffers'):
                self._eclatin_strategy_poll_and_release_buffers()
            
            try:
                return self._free_eclatin_recv_buffer_queue.get(timeout=5.0)
            except queue.Empty:
                logger.error("ECLATIN: TIMEOUT waiting for free recv buffer!")
                return self._free_eclatin_recv_buffer_queue.get()
        
        # Get 4 blocks from eclatin_blocks
        data_block_1 = self.eclatin_blocks['data_block_1']
        data_block_2 = self.eclatin_blocks['data_block_2']
        parity_block_1 = self.eclatin_blocks['parity_block_1']
        parity_block_2 = self.eclatin_blocks['parity_block_2']
        
        # Calculate base addresses
        data_block_1_base = int(data_block_1.data_ptr())
        data_block_2_base = int(data_block_2.data_ptr())
        parity_block_1_base = int(parity_block_1.data_ptr())
        parity_block_2_base = int(parity_block_2.data_ptr())
        
        # Initialize offsets (will be 64-byte aligned when used)
        data_block_1_offset = 0
        data_block_2_offset = 0
        parity_block_1_offset = 0
        parity_block_2_offset = 0
        
        # Get block sizes (all should be the same - aligned_size)
        aligned_block_size = self.eclatin_blocks['aligned_size']
        block_size = aligned_block_size
        
        # Process continuous tensor buffer sequentially
        # Use pipeline_total_bytes (padded size) to ensure all ranks have same iterations
        total_bytes = self.pipeline_total_bytes
        actual_data_bytes = self.actual_tensor_buffer_size
        
        # Split total_bytes into two halves
        half_total = total_bytes // 2  # Divide pipeline_total_bytes into two halves
        
        src_pos = 0  # Current position in continuous tensor buffer (for iteration)
        
        logger.info(
            f"ECLATIN: Processing {total_bytes / (1024**3):.2f} GB "
            f"(actual: {actual_data_bytes / (1024**3):.2f} GB) "
            f"in chunks of {self.eclatin_buffer_size / (1024**2):.0f} MB, "
            f"split into two halves of {half_total / (1024**3):.2f} GB each"
        )
        
        import ctypes
        
        while src_pos < half_total:
            # Calculate chunk size
            remaining_in_source = total_bytes - src_pos
            take = min(self.eclatin_buffer_size, remaining_in_source)
            
            # Get 4 data buffers (to avoid release coordination issues in C++)
            buffer1_addr = get_free_data_buffer()  # For parity1_send1 (data_block_1 part)
            buffer2_addr = get_free_data_buffer()  # For parity1_send2 (data_block_2 part)
            buffer3_addr = get_free_data_buffer()  # For parity2_send1 (data_block_1 part, same as buffer1)
            buffer4_addr = get_free_data_buffer()  # For parity2_send2 (data_block_2 part, same as buffer2)
            
            # Copy from first half to buffer1 and buffer3 (data_block_1 part)
            buffer1_ptr = ctypes.cast(buffer1_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer1_array = ctypes.cast(buffer1_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
            buffer3_ptr = ctypes.cast(buffer3_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer3_array = ctypes.cast(buffer3_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
            
            # Source position in first half
            src_pos_half1 = src_pos  # Position in first half
            if src_pos_half1 < half_total:
                bytes_to_copy_half1 = min(take, half_total - src_pos_half1)
                if src_pos_half1 < actual_data_bytes:
                    actual_bytes_half1 = min(bytes_to_copy_half1, actual_data_bytes - src_pos_half1)
                    src_data_half1 = self.tensor_buffer[src_pos_half1: src_pos_half1 + actual_bytes_half1].numpy()
                    ctypes.memmove(buffer1_array.contents, src_data_half1.ctypes.data, actual_bytes_half1)
                    ctypes.memmove(buffer3_array.contents, src_data_half1.ctypes.data, actual_bytes_half1)
                    
                    if bytes_to_copy_half1 > actual_bytes_half1:
                        padding_size = bytes_to_copy_half1 - actual_bytes_half1
                        padding_ptr1 = ctypes.cast(
                            ctypes.addressof(buffer1_array.contents) + actual_bytes_half1,
                            ctypes.POINTER(ctypes.c_uint8)
                        )
                        padding_ptr3 = ctypes.cast(
                            ctypes.addressof(buffer3_array.contents) + actual_bytes_half1,
                            ctypes.POINTER(ctypes.c_uint8)
                        )
                        ctypes.memset(padding_ptr1, 0, padding_size)
                        ctypes.memset(padding_ptr3, 0, padding_size)
                    
                    if take > bytes_to_copy_half1:
                        # Fill remaining with zeros
                        remaining_padding = take - bytes_to_copy_half1
                        remaining_ptr1 = ctypes.cast(
                            ctypes.addressof(buffer1_array.contents) + bytes_to_copy_half1,
                            ctypes.POINTER(ctypes.c_uint8)
                        )
                        remaining_ptr3 = ctypes.cast(
                            ctypes.addressof(buffer3_array.contents) + bytes_to_copy_half1,
                            ctypes.POINTER(ctypes.c_uint8)
                        )
                        ctypes.memset(remaining_ptr1, 0, remaining_padding)
                        ctypes.memset(remaining_ptr3, 0, remaining_padding)
                else:
                    ctypes.memset(buffer1_array.contents, 0, take)
                    ctypes.memset(buffer3_array.contents, 0, take)
            else:
                # Past first half, fill with zeros
                ctypes.memset(buffer1_array.contents, 0, take)
                ctypes.memset(buffer3_array.contents, 0, take)
            
            # Copy from second half to buffer2 and buffer4 (data_block_2 part)
            buffer2_ptr = ctypes.cast(buffer2_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer2_array = ctypes.cast(buffer2_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
            buffer4_ptr = ctypes.cast(buffer4_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer4_array = ctypes.cast(buffer4_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
            
            # Source position in second half
            src_pos_half2 = src_pos  # Same src_pos, but we read from second half
            src_pos_in_second_half = half_total + src_pos_half2  # Position in second half
            
            if src_pos_in_second_half < total_bytes:
                bytes_to_copy_half2 = min(take, total_bytes - src_pos_in_second_half)
                if src_pos_in_second_half < actual_data_bytes:
                    actual_bytes_half2 = min(bytes_to_copy_half2, actual_data_bytes - src_pos_in_second_half)
                    src_data_half2 = self.tensor_buffer[src_pos_in_second_half: src_pos_in_second_half + actual_bytes_half2].numpy()
                    ctypes.memmove(buffer2_array.contents, src_data_half2.ctypes.data, actual_bytes_half2)
                    ctypes.memmove(buffer4_array.contents, src_data_half2.ctypes.data, actual_bytes_half2)
                    
                    if bytes_to_copy_half2 > actual_bytes_half2:
                        padding_size = bytes_to_copy_half2 - actual_bytes_half2
                        padding_ptr2 = ctypes.cast(
                            ctypes.addressof(buffer2_array.contents) + actual_bytes_half2,
                            ctypes.POINTER(ctypes.c_uint8)
                        )
                        padding_ptr4 = ctypes.cast(
                            ctypes.addressof(buffer4_array.contents) + actual_bytes_half2,
                            ctypes.POINTER(ctypes.c_uint8)
                        )
                        ctypes.memset(padding_ptr2, 0, padding_size)
                        ctypes.memset(padding_ptr4, 0, padding_size)
                    
                    if take > bytes_to_copy_half2:
                        # Fill remaining with zeros
                        remaining_padding = take - bytes_to_copy_half2
                        remaining_ptr2 = ctypes.cast(
                            ctypes.addressof(buffer2_array.contents) + bytes_to_copy_half2,
                            ctypes.POINTER(ctypes.c_uint8)
                        )
                        remaining_ptr4 = ctypes.cast(
                            ctypes.addressof(buffer4_array.contents) + bytes_to_copy_half2,
                            ctypes.POINTER(ctypes.c_uint8)
                        )
                        ctypes.memset(remaining_ptr2, 0, remaining_padding)
                        ctypes.memset(remaining_ptr4, 0, remaining_padding)
                else:
                    ctypes.memset(buffer2_array.contents, 0, take)
                    ctypes.memset(buffer4_array.contents, 0, take)
            else:
                # Past second half, fill with zeros
                ctypes.memset(buffer2_array.contents, 0, take)
                ctypes.memset(buffer4_array.contents, 0, take)
            
            # Write to data_block_1 and data_block_2 (for persistence)
            # Calculate write addresses for data blocks (64-byte aligned)
            data_block_1_offset_aligned = ((data_block_1_offset + 63) // 64) * 64
            data_block_2_offset_aligned = ((data_block_2_offset + 63) // 64) * 64
            
            # Check bounds
            if data_block_1_offset_aligned + take > block_size:
                logger.warning(f"ECLATIN: data_block_1 exhausted")
                break
            if data_block_2_offset_aligned + take > block_size:
                logger.warning(f"ECLATIN: data_block_2 exhausted")
                break
            
            data_block_1_write_addr = data_block_1_base + data_block_1_offset_aligned
            data_block_2_write_addr = data_block_2_base + data_block_2_offset_aligned
            
            # Copy to persistent data blocks (only actual data part, not padding)
            data_block_1_ptr = ctypes.cast(data_block_1_write_addr, ctypes.POINTER(ctypes.c_uint8))
            data_block_2_ptr = ctypes.cast(data_block_2_write_addr, ctypes.POINTER(ctypes.c_uint8))
            
            # CRITICAL FIX: Use actual_data_bytes // 2 as split point for data blocks
            # Pipeline uses half_total (pipeline_total_bytes // 2) for synchronization,
            # but data blocks should split actual data at actual_data_bytes // 2
            half_actual_data = actual_data_bytes // 2  # Split point for actual data
            
            # Copy from buffers to data blocks (only the actual data portion)
            # data_block_1: first half of actual data [0, half_actual_data)
            if src_pos_half1 < half_actual_data:
                bytes_to_write_half1 = min(take, half_actual_data - src_pos_half1)
                if src_pos_half1 < actual_data_bytes:
                    actual_write_half1 = min(bytes_to_write_half1, actual_data_bytes - src_pos_half1)
                    ctypes.memmove(data_block_1_ptr, buffer1_array.contents, actual_write_half1)
                else:
                    ctypes.memmove(data_block_1_ptr, buffer1_array.contents, bytes_to_write_half1)
            else:
                # Past first half of actual data, no data for data_block_1
                ctypes.memset(data_block_1_ptr, 0, take)
            
            # data_block_2: second half of actual data [half_actual_data, actual_data_bytes)
            # Map src_pos_half1 to second half position
            if src_pos_half1 >= half_actual_data and src_pos_half1 < actual_data_bytes:
                src_pos_in_second_half_mapped = src_pos_half1  # Same position in second half
                bytes_to_write_half2 = min(take, actual_data_bytes - src_pos_in_second_half_mapped)
                actual_write_half2 = min(bytes_to_write_half2, actual_data_bytes - src_pos_in_second_half_mapped)
                # Use buffer2 which contains data from second half of pipeline
                ctypes.memmove(data_block_2_ptr, buffer2_array.contents, actual_write_half2)
            else:
                # Before second half or past actual data, no data for data_block_2
                ctypes.memset(data_block_2_ptr, 0, take)
            
            # Update offsets (use take for data blocks, as they store full chunks)
            data_block_1_offset = data_block_1_offset_aligned + take
            data_block_2_offset = data_block_2_offset_aligned + take
            
            # Get recv buffers for parity1 (2 buffers for recv_xor)
            recv1_addr_parity1 = get_free_recv_buffer()
            recv2_addr_parity1 = get_free_recv_buffer()
            
            # Get recv buffers for parity2 (2 buffers for recv_xor)
            recv1_addr_parity2 = get_free_recv_buffer()
            recv2_addr_parity2 = get_free_recv_buffer()
            
            # Calculate write addresses for parity blocks (64-byte aligned)
            parity_block_1_offset_aligned = ((parity_block_1_offset + 63) // 64) * 64
            parity_block_2_offset_aligned = ((parity_block_2_offset + 63) // 64) * 64
            
            # Check bounds (parity blocks use take size)
            if parity_block_1_offset_aligned + take > block_size:
                logger.warning(f"ECLATIN: parity_block_1 exhausted")
                break
            if parity_block_2_offset_aligned + take > block_size:
                logger.warning(f"ECLATIN: parity_block_2 exhausted")
                break
            
            parity_block_1_write_addr = parity_block_1_base + parity_block_1_offset_aligned
            parity_block_2_write_addr = parity_block_2_base + parity_block_2_offset_aligned
            
            # Update offsets (parity blocks use take size)
            parity_block_1_offset = parity_block_1_offset_aligned + take
            parity_block_2_offset = parity_block_2_offset_aligned + take
            
            # Verify address alignment
            assert parity_block_1_write_addr % 64 == 0
            assert parity_block_2_write_addr % 64 == 0
            
            # Submit to 6 pipelines - all use take size (full chunk)
            # Parity 1 pipelines: send data_block_1 and data_block_2 parts
            self._eclatin_native.submit_parity1_send1(buffer1_addr, take)  # data_block_1 part, full chunk size
            self._eclatin_native.submit_parity1_send2(buffer2_addr, take)  # data_block_2 part, full chunk size
            self._eclatin_native.submit_parity1_recv_xor(
                recv1_addr_parity1,      # recv1_addr
                recv2_addr_parity1,      # recv2_addr
                parity_block_1_write_addr,  # parity_addr
                take                      # size
            )
            
            # Parity 2 pipelines: send data_block_1 and data_block_2 parts (same data, different buffers)
            self._eclatin_native.submit_parity2_send1(buffer3_addr, take)  # data_block_1 part, full chunk size
            self._eclatin_native.submit_parity2_send2(buffer4_addr, take)  # data_block_2 part, full chunk size
            self._eclatin_native.submit_parity2_recv_xor(
                recv1_addr_parity2,      # recv1_addr
                recv2_addr_parity2,      # recv2_addr
                parity_block_2_write_addr,  # parity_addr
                take                      # size
            )
            
            src_pos += take
        
        logger.info(
            f"ECLATIN: Processed {src_pos / (1024**3):.2f} GB\n"
            f"  data_block_1 used: {data_block_1_offset / (1024**3):.2f} GB\n"
            f"  data_block_2 used: {data_block_2_offset / (1024**3):.2f} GB\n"
            f"  parity_block_1 used: {parity_block_1_offset / (1024**3):.2f} GB\n"
            f"  parity_block_2 used: {parity_block_2_offset / (1024**3):.2f} GB"
        )
        
        # Mark end of stream for all 6 pipelines (sentinels)
        self._eclatin_native.submit_parity1_send1(0, 0)
        self._eclatin_native.submit_parity1_send2(0, 0)
        self._eclatin_native.submit_parity1_recv_xor(0, 0, 0, 0)  # Changed from (0, 0, 0, 0, 0) to (0, 0, 0, 0)
        self._eclatin_native.submit_parity2_send1(0, 0)
        self._eclatin_native.submit_parity2_send2(0, 0)
        self._eclatin_native.submit_parity2_recv_xor(0, 0, 0, 0)  # Changed from (0, 0, 0, 0, 0) to (0, 0, 0, 0)
        
        # Wait for all pipelines to complete
        logger.info("ECLATIN: Waiting for all pipelines to complete...")
        self._eclatin_native.wait_for_encoding_completion()
        torch.cuda.synchronize()
        logger.info("ECLATIN: All pipelines completed and CUDA synchronized")
    
    def _eccheck_preload_tensors_to_buffer(self, non_blocking: bool = True) -> List[WriteBucket]:
        """
        EC-CHECK version: Transfer tensors from GPU to preallocated CPU buffer.
        
        This method transfers tensor data from GPU to the preallocated CPU buffer
        in a pipelined manner, enabling overlap with subsequent encoding operations.
        
        To ensure all ranks have the same pipeline iterations, this method:
        1. Calculates the maximum data size across all ranks
        2. Allocates a buffer of maximum size
        3. Fills remaining space with zeros for ranks with smaller data
        
        Args:
            non_blocking (bool): if True, use non-blocking GPU-to-CPU transfer
        
        Returns:
            torch.Tensor: continuous CPU buffer containing all tensor data
        """
        if not self.decomposed_state_dict:
            raise RuntimeError("EC-CHECK: State dict not decomposed yet")
        
        logger.info("EC-CHECK: Starting GPU-to-CPU tensor transfer...")
        start = time()
        
        # Step 1: Get actual data size for this rank
        actual_total_size = self.decomposed_state_dict.total_tensor_size_bytes
        
        # Step 2: Calculate maximum data size across all ranks
        # Use global_registry if available (no communication needed, all ranks have same registry)
        if (torch.distributed.is_initialized() and 
            hasattr(self, 'eccheck_global_registry') and 
            self.eccheck_global_registry is not None):
            # Get all ranks' data sizes from global_registry (no communication needed)
            all_total_bytes_list = []
            for r in range(torch.distributed.get_world_size()):
                rank_metadata = self.eccheck_global_registry.rank_metadata.get(r, [])
                rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
                all_total_bytes_list.append(rank_total_size)
            
            # Compute maximum locally (all ranks have the same global_registry)
            max_total_bytes = max(all_total_bytes_list)
        else:
            max_total_bytes = actual_total_size
        
        # Step 3: Allocate buffer with maximum size (for pipeline synchronization)
        if self.preallocated_cpu_buffer is not None:
            buffer = self.preallocated_cpu_buffer
            # If preallocated buffer exists, ensure it's large enough
            if buffer.numel() < max_total_bytes:
                logger.warning(
                    f"EC-CHECK: Preallocated buffer ({buffer.numel() / (1024**3):.2f} GB) "
                    f"is smaller than max_total_bytes ({max_total_bytes / (1024**3):.2f} GB). "
                    f"Reallocating..."
                )
                if self.eccheck_pin_memory and torch.cuda.is_available():
                    buffer = torch.empty(max_total_bytes, dtype=torch.uint8).pin_memory()
                else:
                    buffer = torch.empty(max_total_bytes, dtype=torch.uint8)
        else:
            if self.eccheck_pin_memory and torch.cuda.is_available():
                buffer = torch.empty(max_total_bytes, dtype=torch.uint8).pin_memory()
            else:
                buffer = torch.empty(max_total_bytes, dtype=torch.uint8)
        
        logger.info(
            f"EC-CHECK: Allocated continuous CPU buffer: {max_total_bytes / (1024**3):.2f} GB "
            f"(actual data: {actual_total_size / (1024**3):.2f} GB, "
            f"padding: {(max_total_bytes - actual_total_size) / (1024**3):.2f} GB)"
        )
        
        # Step 4: Transfer tensors from GPU to continuous CPU buffer
        num_gpu_tensors = 0
        offset = 0
        for info, tensor in zip(
            self.decomposed_state_dict.tensor_infos,
            self.decomposed_state_dict.tensor_data
        ):
            # Calculate size for this tensor
            tensor_size = info.size_bytes
            
            # Get view of buffer at current offset
            buffer_view = buffer[offset:offset + tensor_size]
            
            # Flatten and copy tensor to continuous buffer
            tensor_flat = tensor.flatten().contiguous().view(torch.uint8)
            buffer_view.copy_(tensor_flat, non_blocking=non_blocking)
            
            if tensor.device.type != 'cpu':
                num_gpu_tensors += 1
            
            # Update tensor info
            info.offset = offset
            info.device = torch.device('cpu')
            
            # Move to next tensor position
            offset += tensor_size
        
        # Step 5: Fill remaining space with zeros (for pipeline synchronization)
        if offset < max_total_bytes:
            padding_size = max_total_bytes - offset
            buffer[offset:max_total_bytes].fill_(0)
            logger.debug(
                f"EC-CHECK: Filled {padding_size / (1024**2):.2f} MB with zeros "
                f"for pipeline synchronization"
            )
        
        # Synchronize if using non-blocking transfers
        if non_blocking and num_gpu_tensors > 0:
            torch.cuda.synchronize()
        
        # Step 6: Store the continuous buffer and metadata
        self.tensor_buffer = buffer
        self.actual_tensor_buffer_size = actual_total_size  # Actual data size
        self.pipeline_total_bytes = max_total_bytes  # Maximum size for pipeline
        
        transfer_time = time() - start
        total_gb = actual_total_size / (1024**3)
        bandwidth = total_gb / transfer_time if transfer_time > 0 else 0
        
        logger.info(
            f"EC-CHECK: Transferred {total_gb:.2f} GB in {transfer_time:.2f}s "
            f"({bandwidth:.2f} GB/s), {num_gpu_tensors} tensors from GPU to CPU"
        )
        logger.info(
            f"EC-CHECK: Updated tensor_infos device info - "
            f"{num_gpu_tensors} tensors now on CPU"
        )
        logger.info(
            f"EC-CHECK: Pipeline will use {max_total_bytes / (1024**3):.2f} GB "
            f"to ensure all ranks have same iterations"
        )
        
        # Validate that decomposition is still correct after transfer
        if not self.validate_eccheck_decomposition():
            logger.warning("EC-CHECK: Validation warning after GPU-to-CPU transfer")
        
        # ===== Verify that metadata and buffers are passed from strategy =====
        # Metadata exchange and buffer allocation are now done in torch.py
        # The global_registry and receive buffers should already be set by the strategy
        
        if not hasattr(self, 'eccheck_global_registry') or self.eccheck_global_registry is None:
            raise RuntimeError(
                "EC-CHECK: Global registry not set. Should be passed from strategy "
                "after _prepare_eccheck_data completes."
            )
        
        if not hasattr(self, 'eccheck_recv_encoding_buffers') or self.eccheck_recv_encoding_buffers is None:
            raise RuntimeError(
                "EC-CHECK: Receive buffers not set. Should be passed from strategy "
                "after _prepare_eccheck_data completes."
            )
        
        logger.info(
            "EC-CHECK: Using metadata and receive buffers from strategy (already allocated in torch.py)"
        )
        
        # Execute Phase 3: Tensor data exchange and encoding
        exec_start = time()
        self._execute_phase3_encoding()
        exec_time = time() - exec_start
        
        # Return write_buckets with EC-CHECK continuous buffer
        # Buffer contains all tensor data in continuous memory
        
        # todo(hucc):  mul write_buckets is for mul process write ,but here is one process write ,so we need to change the write_buckets to a list of write_buckets, leave it future
        result_buckets = []
        for i, bucket in enumerate(self.write_buckets):
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            # Add EC-CHECK metadata and continuous buffer
            if self.use_eccheck:
                if i == 0:
                    eccheck_bytes_data = [
                        ('eccheck_metadata', self.eccheck_serialized_metadata),
                        ('eccheck_continuous_buffer', self.tensor_buffer),  # Continuous buffer
                    ]
                    result_buckets.append((file_name, storage_key, (eccheck_bytes_data, [])))
                else:
                    continue
            else:
                eccheck_bytes_data = [
                    ('eccheck_metadata', self.eccheck_serialized_metadata),
                    ('eccheck_continuous_buffer', self.tensor_buffer),  # Continuous buffer
                ]
                result_buckets.append((file_name, storage_key, (eccheck_bytes_data, [])))
            
        if self.use_eccheck and self.ecc_write_buckets is not None:
            # Update bucket paths with current checkpoint_dir before adding to result_buckets
            # This ensures paths are updated for each iteration
            for bucket in self.ecc_write_buckets:
                file_path, storage_key, data = bucket
                # Extract file name from path
                if isinstance(file_path, (str, Path)):
                    file_path_obj = Path(file_path)
                    file_name = file_path_obj.name
                else:
                    file_name = str(file_path).split('/')[-1] if '/' in str(file_path) else str(file_path)
                
                # Build new path with current checkpoint_dir
                new_file_path = Path(self.checkpoint_dir) / file_name
                
                # Update eccheck_metadata['eccheck_file_path'] in data
                # This is critical because write_preloaded_data uses metadata['eccheck_file_path'] (line 550)
                updated_data = data
                if isinstance(data, tuple) and len(data) > 0:
                    bytes_data_list = data[0] if isinstance(data[0], list) else []
                    updated_bytes_data = []
                    for item in bytes_data_list:
                        if isinstance(item, tuple) and len(item) == 2:
                            key, value = item
                            if key == 'eccheck_metadata' and isinstance(value, dict):
                                # Update the file path in metadata
                                updated_metadata = value.copy()
                                updated_metadata['eccheck_file_path'] = str(new_file_path)
                                updated_bytes_data.append((key, updated_metadata))
                            else:
                                updated_bytes_data.append(item)
                        else:
                            updated_bytes_data.append(item)
                    updated_data = (updated_bytes_data, data[1] if len(data) > 1 else [])
                
                # Create updated bucket with new path and updated metadata
                updated_bucket = (new_file_path, storage_key, updated_data)
                result_buckets.append(updated_bucket)
                
        logger.info(f"EC-CHECK: eccheck preload tensor to buffer {exec_time:.2f}s")

        return result_buckets
    
    @staticmethod
    def load_eccheck_bytes_from_file(file_path: Union[str, os.PathLike], my_rank: int = 0) -> Tuple[EccheckMappedFile, Dict[str, Any], List[Any]]:
        """
        Load EC-CHECK file using mmap and return memory address and size for NCCL send/recv.
        Also extracts non_tensor_data and tensor metadata for preparing local_metadata.
        
        File structure:
        [Header: 32 bytes] [Component 1] [Component 2] [Component 3]
        
        Header format:
        - Magic number: 4 bytes ('ECCK')
        - Padding: 4 bytes (for alignment)
        - Component 1 size: 8 bytes (uint64)
        - Component 2 size: 8 bytes (uint64)
        - Component 3 size: 8 bytes (uint64)
        
        Args:
            file_path: path to the EC-CHECK file
            my_rank: current rank (used for preparing local_metadata), default 0
        
        Returns:
            Tuple[EccheckMappedFile, Dict[str, Any], List[TensorMetadata]]: tuple containing:
                - EccheckMappedFile: dataclass containing:
                    - mmap_object: mmap object that must be kept alive for the memory to remain valid
                    - memory_address: starting memory address (can be used with NCCL)
                    - file_size: total size of the mapped file in bytes
                - non_tensor_data: Dict[str, Any] extracted from Component 1
                - local_metadata: List[TensorMetadata] for preparing local metadata
        
        Note:
            The mmap object must be kept alive (not garbage collected) while using the memory
            for NCCL operations. The caller should call mapped_file.close() when done.
        """
        import mmap
        import struct
        import pickle
        from .state_dict_decomposer import TensorMetadata
        
        # Open file and get size
        f = open(file_path, "rb")
        mm = None
        try:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Seek back to start
            
            # Memory-map the entire file (zero-copy for /dev/shm)
            # Note: We don't use 'with' statement to keep mmap alive for NCCL operations
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Close file handle - mmap is independent of the file handle
            f.close()
            f = None
            
            # Parse header to extract Component 1 (non_tensor_data) and Component 2 (tensor_infos)
            header_bytes = mm[:32]
            if len(header_bytes) != 32:
                raise RuntimeError(f"EC-CHECK: Invalid file header (expected 32 bytes, got {len(header_bytes)})")
            
            # Parse header (default format includes padding for alignment)
            magic, non_tensor_size, tensor_keys_size, tensor_buffer_size = struct.unpack('4sQQQ', header_bytes)
            
            # Validate magic number
            if magic != b'ECCK':
                raise RuntimeError(f"EC-CHECK: Invalid magic number (expected b'ECCK', got {magic})")
            
            # Extract Component 1: non_tensor_data
            offset = 32  # After header
            
            non_tensor_bytes = mm[offset:offset + non_tensor_size]
            if len(non_tensor_bytes) != non_tensor_size:
                raise RuntimeError(
                    f"EC-CHECK: Failed to read Component 1 "
                    f"(expected {non_tensor_size} bytes, got {len(non_tensor_bytes)})"
                )
            
            # Deserialize non_tensor_data
            non_tensor_data = pickle.loads(non_tensor_bytes)
            logger.debug(f"EC-CHECK: Extracted non_tensor_data from Component 1 ({non_tensor_size / 1024:.2f} KB)")
            
            offset += non_tensor_size  # Move to Component 2
            
            # Extract Component 2: tensor_infos (for preparing local_metadata)
            tensor_keys_bytes = mm[offset:offset + tensor_keys_size]
            if len(tensor_keys_bytes) != tensor_keys_size:
                raise RuntimeError(
                    f"EC-CHECK: Failed to read Component 2 "
                    f"(expected {tensor_keys_size} bytes, got {len(tensor_keys_bytes)})"
                )
            
            # Deserialize tensor_infos
            tensor_infos = pickle.loads(tensor_keys_bytes)
            logger.debug(f"EC-CHECK: Extracted {len(tensor_infos)} tensor infos from Component 2")
            
            # Convert tensor_infos to local_metadata (List[TensorMetadata])
            local_metadata = []
            for info in tensor_infos:
                # print("info.target_rank, info.source_rank: ", info)
                # Create TensorMetadata for each TensorInfo
                data_meta = TensorMetadata(
                    key=info.key,
                    shape=info.shape,
                    dtype=str(info.dtype),
                    size_bytes=info.size_bytes,
                    global_offset=info.global_offset if info.global_offset is not None else (),
                    shard_index=info.shard_index if info.shard_index is not None else 0,
                    chunk_type=info.chunk_type,
                    target_rank=info.target_rank,  # Data stays on same rank
                    source_rank=info.source_rank,
                )
                local_metadata.append(data_meta)
            
            logger.debug(f"EC-CHECK: Prepared {len(local_metadata)} TensorMetadata entries for local_metadata")
            
            # Get memory address for NCCL operations
            # Use numpy.frombuffer to get address from read-only mmap (doesn't require write access)
            # This creates a read-only numpy view and extracts its memory address
            import numpy as np
            
            # Create minimal numpy view to get buffer address (works with read-only buffers)
            np_view = np.frombuffer(mm, dtype=np.uint8, count=min(1, file_size))
            memory_address = np_view.ctypes.data
            
            # Note: The mmap object itself implements the buffer protocol and can be used
            # directly with NCCL. The memory_address is provided for cases where a raw
            # pointer is needed.
            
            logger.info(
                f"EC-CHECK: Mapped file {file_path} for NCCL operations\n"
                f"  File size: {file_size / (1024**3):.2f} GB\n"
                f"  Memory address: {hex(memory_address)}\n"
                f"  Non-tensor data: {len(non_tensor_data)} keys\n"
                f"  Tensor metadata: {len(local_metadata)} entries\n"
                f"  Note: mmap object must be kept alive during NCCL operations"
            )
            
            mapped_file = EccheckMappedFile(
                mmap_object=mm,
                memory_address=memory_address,
                file_size=file_size,
                local_metadata=local_metadata,
                non_tensor_data=non_tensor_data
            )
            
            return mapped_file
            
        except Exception as e:
            if mm is not None:
                try:
                    mm.close()
                except:
                    pass
            if f is not None:
                f.close()
            raise RuntimeError(f"EC-CHECK: Failed to map file {file_path} for NCCL: {e}") from e
    
    @staticmethod
    def load_eclatin_bytes_from_file(file_path: Union[str, os.PathLike], my_rank: int = 0) -> EclatinMappedFile:
        """
        Load ECLATIN file using mmap and extract metadata.
        
        Similar to load_eccheck_bytes_from_file but for ECLATIN format (ECLT magic).
        
        File structure:
        [Header: 32 bytes] [Component 1] [Component 2] [Component 3]
        
        Args:
            file_path: path to the ECLATIN file
            my_rank: current rank (used for preparing local_metadata), default 0
        
        Returns:
            EclatinMappedFile: dataclass containing mmap object, memory address, file size,
                              local_metadata, and non_tensor_data
        """
        import mmap
        import struct
        import pickle
        from .state_dict_decomposer import TensorMetadata
        
        # Open file and get size
        f = open(file_path, "rb")
        mm = None
        try:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Seek back to start
            
            # Memory-map the entire file
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Close file handle - mmap is independent of the file handle
            f.close()
            f = None
            
            # Parse header to extract Component 1 (non_tensor_data) and Component 2 (tensor_infos)
            header_bytes = mm[:32]
            if len(header_bytes) != 32:
                raise RuntimeError(f"ECLATIN: Invalid file header (expected 32 bytes, got {len(header_bytes)})")
            
            # Parse header
            magic, non_tensor_size, tensor_keys_size, tensor_buffer_size = struct.unpack('4sQQQ', header_bytes)
            
            # Validate magic number
            if magic != b'ECLT':
                raise RuntimeError(f"ECLATIN: Invalid magic number (expected b'ECLT', got {magic})")
            
            # Extract Component 1: non_tensor_data
            offset = 32  # After header
            
            non_tensor_bytes = mm[offset:offset + non_tensor_size]
            if len(non_tensor_bytes) != non_tensor_size:
                raise RuntimeError(
                    f"ECLATIN: Failed to read Component 1 "
                    f"(expected {non_tensor_size} bytes, got {len(non_tensor_bytes)})"
                )
            
            # Deserialize non_tensor_data
            non_tensor_data = pickle.loads(non_tensor_bytes)
            logger.debug(f"ECLATIN: Extracted non_tensor_data from Component 1 ({non_tensor_size / 1024:.2f} KB)")
            
            offset += non_tensor_size  # Move to Component 2
            
            # Extract Component 2: tensor_infos (for preparing local_metadata)
            tensor_keys_bytes = mm[offset:offset + tensor_keys_size]
            if len(tensor_keys_bytes) != tensor_keys_size:
                raise RuntimeError(
                    f"ECLATIN: Failed to read Component 2 "
                    f"(expected {tensor_keys_size} bytes, got {len(tensor_keys_bytes)})"
                )
            
            # Deserialize tensor_infos
            tensor_infos = pickle.loads(tensor_keys_bytes)
            logger.debug(f"ECLATIN: Extracted {len(tensor_infos)} tensor infos from Component 2")
            
            # Convert tensor_infos to local_metadata (List[TensorMetadata])
            # Saved objects may be TensorInfo (no chunk_type/target/source), so fill defaults.
            local_metadata = []
            for info in tensor_infos:
                chunk_type = getattr(info, "chunk_type", "data")
                target_rank = getattr(info, "target_rank", my_rank)
                source_rank = getattr(info, "source_rank", my_rank)
                data_meta = TensorMetadata(
                    key=info.key,
                    shape=info.shape,
                    dtype=str(info.dtype),
                    size_bytes=info.size_bytes,
                    global_offset=info.global_offset if info.global_offset is not None else (),
                    shard_index=info.shard_index if info.shard_index is not None else 0,
                    chunk_type=chunk_type,
                    target_rank=target_rank,
                    source_rank=source_rank,
                )
                local_metadata.append(data_meta)
            
            logger.debug(f"ECLATIN: Prepared {len(local_metadata)} TensorMetadata entries for local_metadata")
            
            # Get memory address
            import numpy as np
            np_view = np.frombuffer(mm, dtype=np.uint8, count=min(1, file_size))
            memory_address = np_view.ctypes.data
            
            logger.info(
                f"ECLATIN: Mapped file {file_path}\n"
                f"  File size: {file_size / (1024**3):.2f} GB\n"
                f"  Memory address: {hex(memory_address)}\n"
                f"  Non-tensor data: {len(non_tensor_data)} keys\n"
                f"  Tensor metadata: {len(local_metadata)} entries"
            )
            
            mapped_file = EclatinMappedFile(
                mmap_object=mm,
                memory_address=memory_address,
                file_size=file_size,
                local_metadata=local_metadata,
                non_tensor_data=non_tensor_data,
                tensor_infos=tensor_infos  # Preserve original tensor_infos with offset information
            )
            
            return mapped_file
            
        except Exception as e:
            if mm is not None:
                try:
                    mm.close()
                except:
                    pass
            if f is not None:
                f.close()
            raise RuntimeError(f"ECLATIN: Failed to map file {file_path}: {e}") from e
    
    @staticmethod
    def load_eccheck_components_from_file(file_path: Union[str, os.PathLike]) -> DecomposedStateDict:
        """
        Load three components from a single EC-CHECK file.
        
        File structure:
        [Header: 32 bytes] [Component 1] [Component 2] [Component 3]
        
        Header format:
        - Magic number: 4 bytes ('ECCK')
        - Padding: 4 bytes (for alignment)
        - Component 1 size: 8 bytes (uint64)
        - Component 2 size: 8 bytes (uint64)
        - Component 3 size: 8 bytes (uint64)
        
        Args:
            file_path: path to the EC-CHECK file
        
        Returns:
            DecomposedStateDict: reconstructed decomposed structure
        """
        import struct
        import numpy as np
        import mmap
        
        # Optimization for /dev/shm: Use mmap for zero-copy access
        # Since data is in shared memory, mmap provides direct memory access without copying
        with open(file_path, "rb") as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Seek back to start
            
            # Memory-map the entire file (zero-copy for /dev/shm)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            try:
                # Read header (32 bytes: 4 for magic + 4 for padding + 8*3 for sizes)
                header_bytes = mm[:32]
                if len(header_bytes) != 32:
                    raise RuntimeError(f"EC-CHECK: Invalid file header (expected 32 bytes, got {len(header_bytes)})")
                
                # Parse header (default format includes padding for alignment)
                magic, non_tensor_size, tensor_keys_size, tensor_buffer_size = struct.unpack('4sQQQ', header_bytes)
                
                # Validate magic number
                if magic != b'ECCK':
                    raise RuntimeError(f"EC-CHECK: Invalid magic number (expected b'ECCK', got {magic})")
                
                logger.info(
                    f"EC-CHECK: Loading from {file_path} (using mmap for zero-copy)\n"
                    f"  Component 1 size: {non_tensor_size / 1024:.2f} KB\n"
                    f"  Component 2 size: {tensor_keys_size / 1024:.2f} KB\n"
                    f"  Component 3 size: {tensor_buffer_size / (1024**3):.2f} GB"
                )
                
                # Calculate offsets for each component
                offset = 32  # After header
                
                t1 = time()
                # Component 1: Non-tensor key-value pairs (direct slice from mmap)
                non_tensor_bytes = mm[offset:offset + non_tensor_size]
                if len(non_tensor_bytes) != non_tensor_size:
                    raise RuntimeError(
                        f"EC-CHECK: Failed to read Component 1 "
                        f"(expected {non_tensor_size} bytes, got {len(non_tensor_bytes)})"
                    )
                offset += non_tensor_size
                
                t2 = time()
                non_tensor_data = pickle.loads(non_tensor_bytes)
                logger.debug(f"EC-CHECK: Loaded Component 1 ({non_tensor_size / 1024:.2f} KB)")
            
                t3 = time()
                # Component 2: Tensor keys (direct slice from mmap)
                tensor_keys_bytes = mm[offset:offset + tensor_keys_size]
                if len(tensor_keys_bytes) != tensor_keys_size:
                    raise RuntimeError(
                        f"EC-CHECK: Failed to read Component 2 "
                        f"(expected {tensor_keys_size} bytes, got {len(tensor_keys_bytes)})"
                    )
                offset += tensor_keys_size
                
                tensor_infos = pickle.loads(tensor_keys_bytes)
                logger.debug(f"EC-CHECK: Loaded Component 2 ({tensor_keys_size / 1024:.2f} KB)")
                t4 = time()
                
                # Component 3: Tensor data buffer (zero-copy numpy view from mmap)
                # This is the key optimization: np.frombuffer on mmap creates a zero-copy view
                tensor_buffer_start = offset
                tensor_buffer_end = offset + tensor_buffer_size
                
                if tensor_buffer_end > file_size:
                    raise RuntimeError(
                        f"EC-CHECK: File truncated - expected {tensor_buffer_end} bytes, got {file_size}"
                    )
                
                # Create zero-copy numpy array view directly from mmap
                buffer_np = np.frombuffer(mm, dtype=np.uint8, count=tensor_buffer_size, offset=tensor_buffer_start)
                
                t5 = time()
                # Extract individual tensors from the zero-copy buffer
                tensor_data = []
                for info in tensor_infos:
                    # Calculate byte offset range for this tensor (relative to buffer start)
                    start = info.offset
                    end = start + info.size_bytes
                    
                    # Extract numpy slice (view, not copy) from buffer
                    tensor_bytes_np = buffer_np[start:end]
                    
                    # Create torch tensor directly from bytes
                    # Use frombuffer to create view, then clone to make writable
                    tensor_view = torch.frombuffer(
                        memoryview(tensor_bytes_np), 
                        dtype=info.dtype
                    )
                    # Clone to create writable copy and reshape to original shape
                    tensor = tensor_view.clone().reshape(info.shape)
                    tensor_data.append(tensor)
                
                t6 = time()
                print("time t6 , t5, t4 , t3, t2, t1: ", t6 - t5, t5 - t4, t4 - t3, t3 - t2, t2 - t1, t6 - t1)
                logger.debug(f"EC-CHECK: Loaded Component 3 ({tensor_buffer_size / (1024**3):.2f} GB) and extracted {len(tensor_data)} tensors")
            
            finally:
                # Close mmap
                # mm.close()
                pass
        
        # Create DecomposedStateDict
        decomposed = DecomposedStateDict(
            non_tensor_data=non_tensor_data,
            tensor_infos=tensor_infos,
            tensor_data=tensor_data,
        )
        
        logger.info(
            f"EC-CHECK: Successfully loaded all components from {file_path}\n"
            f"  Component 1: {len(non_tensor_data)} keys\n"
            f"  Component 2: {len(tensor_infos)} tensor infos\n"
            f"  Component 3: {len(tensor_data)} tensors"
        )
        
        return decomposed
    
    @staticmethod
    def load_eclatin_components_from_file(file_path: Union[str, os.PathLike]) -> DecomposedStateDict:
        """
        Load three components from a single ECLATIN file.
        
        Similar to load_eccheck_components_from_file but for ECLATIN format (ECLT magic).
        
        File structure:
        [Header: 32 bytes] [Component 1] [Component 2] [Component 3]
        
        Args:
            file_path: path to the ECLATIN file
        
        Returns:
            DecomposedStateDict: reconstructed decomposed structure
        """
        import struct
        import numpy as np
        import mmap
        import pickle
        from .state_dict_decomposer import DecomposedStateDict
        
        # Optimization for /dev/shm: Use mmap for zero-copy access
        with open(file_path, "rb") as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Seek back to start
            
            # Memory-map the entire file (zero-copy for /dev/shm)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            try:
                # Read header (32 bytes: 4 for magic + 4 for padding + 8*3 for sizes)
                header_bytes = mm[:32]
                if len(header_bytes) != 32:
                    raise RuntimeError(f"ECLATIN: Invalid file header (expected 32 bytes, got {len(header_bytes)})")
                
                # Parse header
                magic, non_tensor_size, tensor_keys_size, tensor_buffer_size = struct.unpack('4sQQQ', header_bytes)
                
                # Validate magic number
                if magic != b'ECLT':
                    raise RuntimeError(f"ECLATIN: Invalid magic number (expected b'ECLT', got {magic})")
                
                logger.info(
                    f"ECLATIN: Loading from {file_path}\n"
                    f"  Component 1 size: {non_tensor_size / 1024:.2f} KB\n"
                    f"  Component 2 size: {tensor_keys_size / 1024:.2f} KB\n"
                    f"  Component 3 size: {tensor_buffer_size / (1024**3):.2f} GB"
                )
                
                # Calculate offsets for each component
                offset = 32  # After header
                
                # Component 1: Non-tensor key-value pairs
                non_tensor_bytes = mm[offset:offset + non_tensor_size]
                if len(non_tensor_bytes) != non_tensor_size:
                    raise RuntimeError(
                        f"ECLATIN: Failed to read Component 1 "
                        f"(expected {non_tensor_size} bytes, got {len(non_tensor_bytes)})"
                    )
                offset += non_tensor_size
                
                non_tensor_data = pickle.loads(non_tensor_bytes)
                logger.debug(f"ECLATIN: Loaded Component 1 ({non_tensor_size / 1024:.2f} KB)")
            
                # Component 2: Tensor keys
                tensor_keys_bytes = mm[offset:offset + tensor_keys_size]
                if len(tensor_keys_bytes) != tensor_keys_size:
                    raise RuntimeError(
                        f"ECLATIN: Failed to read Component 2 "
                        f"(expected {tensor_keys_size} bytes, got {len(tensor_keys_bytes)})"
                    )
                offset += tensor_keys_size
                
                tensor_infos = pickle.loads(tensor_keys_bytes)
                logger.debug(f"ECLATIN: Loaded Component 2 ({tensor_keys_size / 1024:.2f} KB)")
                
                # Component 3: Tensor data buffer (zero-copy numpy view from mmap)
                tensor_buffer_start = offset
                tensor_buffer_end = offset + tensor_buffer_size
                
                if tensor_buffer_end > file_size:
                    raise RuntimeError(
                        f"ECLATIN: File truncated - expected {tensor_buffer_end} bytes, got {file_size}"
                    )
                
                # Create zero-copy numpy array view directly from mmap
                buffer_np = np.frombuffer(mm, dtype=np.uint8, count=tensor_buffer_size, offset=tensor_buffer_start)
                
                # Extract individual tensors from the zero-copy buffer
                tensor_data = []
                for info in tensor_infos:
                    # Calculate byte offset range for this tensor (relative to buffer start)
                    start = info.offset
                    end = start + info.size_bytes
                    
                    # Extract numpy slice (view, not copy) from buffer
                    tensor_bytes_np = buffer_np[start:end]
                    
                    # Create torch tensor directly from bytes
                    tensor_view = torch.frombuffer(
                        memoryview(tensor_bytes_np), 
                        dtype=info.dtype
                    )
                    # Clone to create writable copy and reshape to original shape
                    tensor = tensor_view.clone().reshape(info.shape)
                    tensor_data.append(tensor)
                
                logger.debug(f"ECLATIN: Loaded Component 3 ({tensor_buffer_size / (1024**3):.2f} GB) and extracted {len(tensor_data)} tensors")
            
            finally:
                # Close mmap
                pass
        
        # Create DecomposedStateDict
        decomposed = DecomposedStateDict(
            non_tensor_data=non_tensor_data,
            tensor_infos=tensor_infos,
            tensor_data=tensor_data,
        )
        
        logger.info(
            f"ECLATIN: Successfully loaded all components from {file_path}\n"
            f"  Component 1: {len(non_tensor_data)} keys\n"
            f"  Component 2: {len(tensor_infos)} tensor infos\n"
            f"  Component 3: {len(tensor_data)} tensors"
        )
        
        return decomposed
    
    def validate_eccheck_decomposition(self) -> bool:
        """
        Simple validation: check if state_dict is correctly decomposed into three components.
        
        Validates:
        1. non_tensor_data is a dict
        2. tensor_infos is a list (tensor keys)
        3. tensor_data is a list of tensors
        4. Counts match between tensor_infos and tensor_data
        
        Returns:
            bool: True if decomposition is valid, False otherwise
        """
        if not self.use_eccheck:
            logger.warning("EC-CHECK: Validation skipped - EC-CHECK is not enabled")
            return False
        
        if not self.decomposed_state_dict:
            logger.error("EC-CHECK: Validation failed - State dict not decomposed yet")
            return False
        
        decomposed = self.decomposed_state_dict
        
        # Check 1: Non-tensor key-value pairs (dict)
        if not isinstance(decomposed.non_tensor_data, dict):
            logger.error(
                f"EC-CHECK: Component 1 failed - non_tensor_data should be dict, "
                f"got {type(decomposed.non_tensor_data).__name__}"
            )
            return False
        
        # Check 2: Tensor keys (list)
        if not isinstance(decomposed.tensor_infos, list):
            logger.error(
                f"EC-CHECK: Component 2 failed - tensor_infos should be list, "
                f"got {type(decomposed.tensor_infos).__name__}"
            )
            return False
        
        # Check 3: Tensor data (list)
        if not isinstance(decomposed.tensor_data, list):
            logger.error(
                f"EC-CHECK: Component 3 failed - tensor_data should be list, "
                f"got {type(decomposed.tensor_data).__name__}"
            )
            return False
        
        # Check 4: Counts match
        if len(decomposed.tensor_infos) != len(decomposed.tensor_data):
            logger.error(
                f"EC-CHECK: Count mismatch - {len(decomposed.tensor_infos)} tensor_infos "
                f"vs {len(decomposed.tensor_data)} tensors"
            )
            return False
        
        # All checks passed
        logger.info(
            f"EC-CHECK: Decomposition validation passed ✓\n"
            f"  Component 1 (non-tensor dict): {len(decomposed.non_tensor_data)} keys\n"
            f"  Component 2 (tensor keys list): {len(decomposed.tensor_infos)} tensors\n"
            f"  Component 3 (tensor data list): {len(decomposed.tensor_data)} tensors"
        )
            
        # Log device info for verification
        if len(decomposed.tensor_infos) > 0:
            devices = set(info.device.type for info in decomposed.tensor_infos)
            logger.debug(f"EC-CHECK: Tensor devices: {devices}")
        
        return True


def _split_by_size_and_type(bins: int, items: List[WriteItem]) -> List[List[WriteItem]]:
    """
    Splits write items according to item size into close to uniform bins.

    Same as torch.distributed.checkpoint.filesystem._split_by_size_and_type,
    but with a fixed _item_size function.

    Args:
        bins (int): numbers of bins to split to
        items (List[WriteItem]): list of write items

    Returns (List[List[WriteItem]]): write items split to bins
    """
    if bins == 1:
        return [items]

    bytes_items: List[WriteItem] = []
    tensor_items: List[WriteItem] = []
    for wi in items:
        container = bytes_items if wi.type == WriteItemType.BYTE_IO else tensor_items
        container.append(wi)

    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    # Assign bytes with a simple round-robin
    for i, item in enumerate(bytes_items):
        buckets[i % bins].append(item)

    # Sort tensor items by size in decreasing order once and store the size with item
    sized_tensors = [(item, _item_size(item)) for item in tensor_items]
    sized_tensors.sort(key=itemgetter(1), reverse=True)

    # Use a min heap for bin assignment
    # Store (total_size_of_bin, bin_index) tuples
    heap: List[Tuple[int, int]] = [(0, i) for i in range(bins)]

    # Assign tensors using heap
    for item, size in sized_tensors:
        total_bin_size, bin_idx = heappop(heap)
        buckets[bin_idx].append(item)
        heappush(heap, (total_bin_size + size, bin_idx))

    return buckets


def _split_by_separation_hint(
    buckets: List[List[WriteItem]], separation_hint: Optional[str] = None
) -> Dict[str, List[List[WriteItem]]]:
    """
    Splits buckets into those whose keys begin with the separation_hint and those whose keys do not

    Args:
        buckets (List[List[WriteItem]]): buckets to split
        separation_hint (Optional[str]): optional prefix to split on

    Returns (Dict[str, List[List[WriteItem]]]): a dictionary
        mapping the prefix to the relevant buckets
    """
    bins = len(buckets)
    buckets_with_separation_hint = {}
    if separation_hint is not None:
        buckets_default = [[] for _ in range(bins)]
        buckets_hint = [[] for _ in range(bins)]
        for i in range(bins):
            for item in buckets[i]:
                if item.index.fqn.startswith(separation_hint):
                    buckets_hint[i].append(item)
                else:
                    buckets_default[i].append(item)
        buckets_with_separation_hint[""] = buckets_default
        buckets_with_separation_hint[separation_hint] = buckets_hint
    else:
        buckets_with_separation_hint[""] = buckets
    return buckets_with_separation_hint


def _item_size(item: WriteItem) -> int:
    """
    Calculates size (in bytes) of a single write item.

    Same as torch.distributed.checkpoint.filesystem._item_size,
    but fixes computing chunk size (with item.tensor_data.chunk.sizes)

    Args:
        item (WriteItem): write item to compute the size of

    Returns (int): size of an item in bytes
    """
    size = 1
    assert item.tensor_data is not None
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.chunk.sizes:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _process_memory() -> int:
    """
    Get memory used by current process.

    Returns (int): memory used by current process
    """
    if not HAVE_PSUTIL:
        raise RuntimeError("psutil is not installed, please install it with `pip install psutil`")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
