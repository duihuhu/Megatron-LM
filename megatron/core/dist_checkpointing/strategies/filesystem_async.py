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

logger = logging.getLogger(__name__)

WriteBucket = Tuple[Path, str, Tuple[list, list]]  # represents writes to a single file

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
        
        # EC-CHECK buffer poller thread (persistent, created once)
        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None  # Controls when polling is active
        
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
        
        Note: Only sets up data and encoding buffers from strategy.
        Receive and parity buffers will be allocated later after metadata exchange.
        """
        self.eccheck_data_buffers = buffers['data_buffers']
        self.eccheck_encoding_buffers = buffers['encoding_buffers']
        self._free_data_buffer_queue = buffers['free_data_buffer_queue']
        self._free_encoding_buffer_queue = buffers['free_encoding_buffer_queue']
        
        # Use buffer poller from strategy (already running)
        self._buffer_poller_active_event = buffers.get('buffer_poller_active_event')
        # Store the strategy's poll method with a different name to avoid conflict
        self._strategy_poll_and_release_buffers = buffers.get('poll_and_release_buffers')
        
        # Mark that we're using shared buffer poller (don't start our own)
        self._buffer_poller_shared = True
        
        # Receive and parity buffers are NOT set here
        # They will be allocated after metadata exchange when peer data size is known
        
        logger.info(
            f"EC-CHECK: Using pre-allocated buffers from strategy - "
            f"Data: {len(self.eccheck_data_buffers)}, "
            f"Encoding: {len(self.eccheck_encoding_buffers)}, "
            f"Buffer poller: {'shared from strategy' if self._buffer_poller_active_event else 'will create own'}"
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
        logger = logging.getLogger(__name__)
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
            for p in p_list:
                p.start()

            logger.debug("FileSystemWriterAsync: collecting worker results...")

            # To make sure all nodes are completed
            count_queue.join()
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
                for key, value in bytes_data:
                    if key == 'eccheck_metadata':
                        eccheck_metadata = value
                    elif key == 'eccheck_continuous_buffer':
                        eccheck_continuous_buffer = value
            
            # EC-CHECK mode: save three components to ONE file
            if eccheck_metadata is not None:
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
                    # Optimize: write directly without extra copies
                    component3_start = time()
                    component3_size = 0
                    
                    if eccheck_continuous_buffer is not None:
                        # Write continuous buffer directly (most efficient - single write)
                        import numpy as np
                        np_array = eccheck_continuous_buffer.numpy()  # Zero-copy view
                        mv = memoryview(np_array)
                        
                        # Write entire buffer at once
                        f.write(mv)
                        component3_size = mv.nbytes
                        
                        component3_time = time() - component3_start
                        bandwidth = (component3_size / (1024**3)) / component3_time if component3_time > 0 else 0
                        logger.info(
                            f"EC-CHECK: Wrote Component 3 ({component3_size / (1024**3):.2f} GB) "
                            f"in {component3_time:.2f}s ({bandwidth:.2f} GB/s), "
                            f"continuous buffer"
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
        
        # Process continuous tensor buffer sequentially
        # Copy data from self.tensor_buffer (continuous CPU buffer) to data buffers
        total_bytes = self.decomposed_state_dict.total_tensor_size_bytes
        src_pos = 0  # Current position in continuous tensor buffer
        # chunk_count = 0
        
        # Get base addresses of TWO receive buffers (one per encoding thread)
        recv_buffer_thread1, recv_buffer_thread2 = self.eccheck_recv_encoding_buffers
        recv_buffer_base_addr_thread1 = int(recv_buffer_thread1.data_ptr())
        recv_buffer_base_addr_thread2 = int(recv_buffer_thread2.data_ptr())
        recv_buffer_offset_thread1 = 0  # Current offset in thread1's receive buffer
        recv_buffer_offset_thread2 = 0  # Current offset in thread2's receive buffer

        while src_pos < total_bytes:
            # Get a free data buffer (with timeout to detect deadlocks)
            cur_buffer_addr = get_free_data_buffer()
            
            # Calculate how much data to copy to this buffer
            remaining_in_source = total_bytes - src_pos
            take = min(self.eccheck_buffer_size, remaining_in_source)
            
            # Python memcpy: copy from continuous tensor buffer to data buffer
            import ctypes
            buffer_ptr = ctypes.cast(cur_buffer_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer_array = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
            
            # Get source data from continuous tensor buffer
            src_data = self.tensor_buffer[src_pos: src_pos + take].numpy()
            
            # Direct memory copy using ctypes
            ctypes.memmove(buffer_array.contents, src_data.ctypes.data, take)
            
            # Get two encoding buffers (with timeout to detect deadlocks)
            enc_addr1 = get_free_encoding_buffer()
            enc_addr2 = get_free_encoding_buffer()
            
            # Allocate receive addresses from TWO recv_encoding_buffers (按实际数据大小分配)
            # Each encoding thread gets its own receive address
            # 如果剩余数据能够填满固定chunk size，则按照chunk size分配
            # 否则按照实际剩余大小分配
            recv_chunk_size = min(self.eccheck_buffer_size, remaining_in_source)
            
            # Thread1 receive address
            recv_addr_thread1 = recv_buffer_base_addr_thread1 + recv_buffer_offset_thread1
            recv_buffer_offset_thread1 += recv_chunk_size  # 按照实际大小移动
            
            # Thread2 receive address
            recv_addr_thread2 = recv_buffer_base_addr_thread2 + recv_buffer_offset_thread2
            recv_buffer_offset_thread2 += recv_chunk_size  # 按照实际大小移动
            
            # Submit to BOTH encoding threads with their respective receive addresses
            # The C++ threads will mark the data buffer as copied immediately after reading
            # Once both threads mark it as copied, the data buffer will be released
            # recv_addr will be used by recv_worker to receive peer data
            self._eccheck_native.submit_data_for_encoding_thread1(
                cur_buffer_addr, take, enc_addr1, recv_addr_thread1, recv_chunk_size
            )
            
            self._eccheck_native.submit_data_for_encoding_thread2(
                cur_buffer_addr, take, enc_addr2, recv_addr_thread2, recv_chunk_size
            )
            
            src_pos += take
            
        logger.info(
            f"  Thread1 receive buffer used: {recv_buffer_offset_thread1 / (1024**3):.2f} GB\n"
            f"  Thread2 receive buffer used: {recv_buffer_offset_thread2 / (1024**3):.2f} GB\n"
            f"  Total receive buffer used: {(recv_buffer_offset_thread1 + recv_buffer_offset_thread2) / (1024**3):.2f} GB"
        )
        
        # Mark end of stream for both encoders
        self._eccheck_native.submit_data_for_encoding_thread1(0, 0, 0, 0, 0)  # Sentinel for thread 1
        self._eccheck_native.submit_data_for_encoding_thread2(0, 0, 0, 0, 0)  # Sentinel for thread 2`
        
        # Wait for both encoding threads to complete
        # Buffer poller is already active (activated at function start)
        logger.info("EC-CHECK: Waiting for encoding threads to complete (with buffer polling)...")
        
        # Wait for encoding completion (this may block)
        # The buffer poller will continue running in the background
        self._eccheck_native.wait_for_encoding_completion()
        
        # Buffer poller will be deactivated in the outer finally block
        # logger.info("EC-CHECK: All encoding operations completed")
    
    def _eccheck_preload_tensors_to_buffer(self, non_blocking: bool = True) -> List[WriteBucket]:
        """
        EC-CHECK version: Transfer tensors from GPU to preallocated CPU buffer.
        
        This method transfers tensor data from GPU to the preallocated CPU buffer
        in a pipelined manner, enabling overlap with subsequent encoding operations.
        
        Args:
            non_blocking (bool): if True, use non-blocking GPU-to-CPU transfer
        
        Returns:
            torch.Tensor: continuous CPU buffer containing all tensor data
        """
        if not self.decomposed_state_dict:
            raise RuntimeError("EC-CHECK: State dict not decomposed yet")
        
        logger.info("EC-CHECK: Starting GPU-to-CPU tensor transfer...")
        start = time()
        # Allocate continuous CPU buffer if not already allocated
        if self.preallocated_cpu_buffer is not None:
            buffer = self.preallocated_cpu_buffer
        else:
            total_size = self.decomposed_state_dict.total_tensor_size_bytes
            if self.eccheck_pin_memory and torch.cuda.is_available():
                buffer = torch.empty(total_size, dtype=torch.uint8).pin_memory()
            else:
                buffer = torch.empty(total_size, dtype=torch.uint8)
            logger.info(f"EC-CHECK: Allocated continuous CPU buffer: {total_size / (1024**3):.2f} GB")
        
        # Transfer tensors from GPU to continuous CPU buffer
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
        
        # Synchronize if using non-blocking transfers
        if non_blocking and num_gpu_tensors > 0:
            torch.cuda.synchronize()
        
        # Store the continuous buffer
        self.tensor_buffer = buffer
        
        transfer_time = time() - start
        total_gb = self.decomposed_state_dict.total_tensor_size_bytes / (1024**3)
        bandwidth = total_gb / transfer_time if transfer_time > 0 else 0
        
        logger.info(
            f"EC-CHECK: Transferred {total_gb:.2f} GB in {transfer_time:.2f}s "
            f"({bandwidth:.2f} GB/s), {num_gpu_tensors} tensors from GPU to CPU"
        )
        logger.info(
            f"EC-CHECK: Updated tensor_infos device info - "
            f"{num_gpu_tensors} tensors now on CPU"
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
        logger.info(f"EC-CHECK: Phase 3 completed in {exec_time:.2f}s")
        
        # Return write_buckets with EC-CHECK continuous buffer
        # Buffer contains all tensor data in continuous memory
        
        # todo(hucc):  mul write_buckets is for mul process write ,but here is one process write ,so we need to change the write_buckets to a list of write_buckets, leave it future
        result_buckets = []
        for bucket in self.write_buckets:
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            # Add EC-CHECK metadata and continuous buffer
            eccheck_bytes_data = [
                ('eccheck_metadata', self.eccheck_serialized_metadata),
                ('eccheck_continuous_buffer', self.tensor_buffer),  # Continuous buffer
            ]
            result_buckets.append((file_name, storage_key, (eccheck_bytes_data, [])))
        
        return result_buckets
    
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
        
        with open(file_path, "rb") as f:
            # Read header (32 bytes: 4 for magic + 4 for padding + 8*3 for sizes)
            header_bytes = f.read(32)
            if len(header_bytes) != 32:
                raise RuntimeError(f"EC-CHECK: Invalid file header (expected 32 bytes, got {len(header_bytes)})")
            
            # Parse header (default format includes padding for alignment)
            magic, non_tensor_size, tensor_keys_size, tensor_buffer_size = struct.unpack('4sQQQ', header_bytes)
            
            # Validate magic number
            if magic != b'ECCK':
                raise RuntimeError(f"EC-CHECK: Invalid magic number (expected b'ECCK', got {magic})")
            
            logger.info(
                f"EC-CHECK: Loading from {file_path}\n"
                f"  Component 1 size: {non_tensor_size / 1024:.2f} KB\n"
                f"  Component 2 size: {tensor_keys_size / 1024:.2f} KB\n"
                f"  Component 3 size: {tensor_buffer_size / (1024**3):.2f} GB"
            )
            
            # Read Component 1: Non-tensor key-value pairs
            non_tensor_bytes = f.read(non_tensor_size)
            if len(non_tensor_bytes) != non_tensor_size:
                raise RuntimeError(
                    f"EC-CHECK: Failed to read Component 1 "
                    f"(expected {non_tensor_size} bytes, got {len(non_tensor_bytes)})"
                )
            non_tensor_data = pickle.loads(non_tensor_bytes)
            logger.debug(f"EC-CHECK: Loaded Component 1 ({non_tensor_size / 1024:.2f} KB)")
            
            # Read Component 2: Tensor keys
            tensor_keys_bytes = f.read(tensor_keys_size)
            if len(tensor_keys_bytes) != tensor_keys_size:
                raise RuntimeError(
                    f"EC-CHECK: Failed to read Component 2 "
                    f"(expected {tensor_keys_size} bytes, got {len(tensor_keys_bytes)})"
                )
            tensor_infos = pickle.loads(tensor_keys_bytes)
            logger.debug(f"EC-CHECK: Loaded Component 2 ({tensor_keys_size / 1024:.2f} KB)")
            
            # Read Component 3: Tensor data buffer
            tensor_buffer_bytes = f.read(tensor_buffer_size)
            if len(tensor_buffer_bytes) != tensor_buffer_size:
                raise RuntimeError(
                    f"EC-CHECK: Failed to read Component 3 "
                    f"(expected {tensor_buffer_size} bytes, got {len(tensor_buffer_bytes)})"
                )
            
            # Convert bytes to tensor buffer
            # Use copy to make tensor writable
            tensor_buffer = torch.from_numpy(np.frombuffer(tensor_buffer_bytes, dtype=np.uint8).copy())
            logger.debug(f"EC-CHECK: Loaded Component 3 ({tensor_buffer_size / (1024**3):.2f} GB)")
        
        # Extract individual tensors from buffer
        from .state_dict_decomposer import extract_tensors_from_continuous_buffer
        tensor_data = extract_tensors_from_continuous_buffer(tensor_buffer, tensor_infos)
        
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
