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
        eccheck_use_continuous_buffer: bool = True,
        eccheck_pin_memory: bool = True,
        eccheck_preallocate_cpu_buffer: bool = True,
        eccheck_k: int = 2,
        eccheck_m: int = 2,
        eccheck_data_buffers_count: int = 12,
        eccheck_encoding_buffers_count: Optional[int] = None,
        eccheck_buffer_size: int = 64 * 1024 * 1024,
        eccheck_config_path: Optional[Union[str, os.PathLike]] = None,
        eccheck_native: Optional[Any] = None,  # Pre-initialized C++ module
        eccheck_buffers: Optional[Dict] = None,  # Pre-allocated buffers
        **kwargs,
    ):
        self.checkpoint_dir = path
        self.use_msc = use_msc
        
        # EC-CHECK configuration
        self.use_eccheck = use_eccheck
        self.eccheck_use_continuous_buffer = eccheck_use_continuous_buffer
        self.eccheck_pin_memory = eccheck_pin_memory
        self.eccheck_preallocate_cpu_buffer = eccheck_preallocate_cpu_buffer
        
        # EC-CHECK encoding parameters (configurable)
        self.eccheck_k = eccheck_k  # Number of data nodes
        self.eccheck_m = eccheck_m  # Number of encoded packets per data packet
        self.eccheck_data_buffers_count = eccheck_data_buffers_count  # Number of data buffers per worker
        self.eccheck_encoding_buffers_count = eccheck_encoding_buffers_count or (eccheck_data_buffers_count * eccheck_m)  # Number of encoding buffers per worker
        self.eccheck_buffer_size = eccheck_buffer_size  # Buffer size in bytes
        self.eccheck_config_path = str(eccheck_config_path) if eccheck_config_path is not None else None

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
        self.eccheck_recv_encoding_buffers = None  # List of large receive buffers (one per column)
        self.eccheck_parity_buffers = None  # List of parity buffers for XOR results
        # Persistent stores (in-memory): final recv and parity results
        self.eccheck_persist_recv_store: Optional[torch.Tensor] = None
        self.eccheck_persist_parity_store: Optional[torch.Tensor] = None
        self.eccheck_persist_recv_enabled: bool = True
        self.eccheck_persist_parity_enabled: bool = True
        
        # EC-CHECK buffer poller thread (persistent, created once)
        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None  # Controls when polling is active
        
        # Try to load ECCHECK external configuration (YAML/JSON)
        self._load_eccheck_config()
        
        # Initialize C++ native module if available
        if eccheck_native is not None:
            # Use pre-initialized C++ module from strategy
            self._eccheck_native = eccheck_native
            self._eccheck_shared = True  # Mark as shared module
            logger.info("EC-CHECK: Using pre-initialized C++ native module from strategy")
            
            # Use pre-allocated buffers from strategy
            if eccheck_buffers is not None:
                self._setup_eccheck_buffers_from_strategy(eccheck_buffers)
        elif self.use_eccheck:
            # Initialize C++ module here (fallback for direct usage)
            self._init_eccheck_native()
            self._eccheck_shared = False  # Mark as owned module
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
                # Stop buffer poller thread
                self._stop_buffer_poller_thread()
                
                # Only cleanup if we own the module (not shared from strategy)
                # The strategy will handle cleanup of shared modules
                if not hasattr(self, '_eccheck_shared') or not self._eccheck_shared:
                    self._stop_phase3_workers()
        except Exception as e:
            # Log but don't raise exceptions in destructor
            logger.warning(f"EC-CHECK: Error during cleanup in destructor: {e}")

    def _init_eccheck_native(self) -> None:
        """Initialize C++ native module during class construction."""
        try:
            import eccheck_native
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            paired_rank = self._get_paired_rank(rank, world_size)
            
            self._eccheck_native = eccheck_native.ECCHECKNative(rank, world_size, paired_rank)
            logger.info(f"EC-CHECK: C++ native module initialized (rank={rank}, world_size={world_size}, paired_rank={paired_rank})")
            
            # Initialize buffer allocation for EC-CHECK
            self._init_eccheck_buffers()
            
            # Start persistent buffer poller thread
            self._start_buffer_poller_thread()
            
            # Note: No longer using callbacks to avoid GIL deadlock
            # Instead, Python will poll C++ for buffers ready to be released
            
        except ImportError:
            logger.warning("EC-CHECK: C++ native module not available, EC-CHECK functionality will not work")
            self._eccheck_native = None
        except Exception as e:
            logger.warning(f"EC-CHECK: Failed to initialize C++ native module: {e}, EC-CHECK functionality will not work")
            self._eccheck_native = None

    def _load_eccheck_config(self) -> None:
        """Load optional ECCHECK config (YAML/JSON) to override defaults.
        Supported keys:
          - persist:
              recv: bool
              parity: bool
        Future (not required now): columns mapping, coefficients, peers per column.
        """
        try:
            cfg_path = self.eccheck_config_path or os.environ.get('ECCHECK_CONFIG_PATH')
            if not cfg_path or not os.path.isfile(cfg_path):
                return
            with open(cfg_path, 'r') as f:
                text = f.read()
            data = None
            # Try YAML first, then JSON
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(text)
            except Exception:
                pass
            if data is None:
                import json
                data = json.loads(text)
            if not isinstance(data, dict):
                return
            persist = data.get('persist', {}) if isinstance(data.get('persist', {}), dict) else {}
            if 'recv' in persist:
                self.eccheck_persist_recv_enabled = bool(persist['recv'])
            if 'parity' in persist:
                self.eccheck_persist_parity_enabled = bool(persist['parity'])
            logger.info(
                f"EC-CHECK: Config loaded from {cfg_path} (persist.recv={self.eccheck_persist_recv_enabled}, "
                f"persist.parity={self.eccheck_persist_parity_enabled})"
            )
        except Exception as e:
            logger.warning(f"EC-CHECK: Failed to load config: {e}")

    def _setup_eccheck_buffers_from_strategy(self, buffers):
        """Set up EC-CHECK buffers from pre-allocated strategy buffers.
        
        Note: Only sets up data and encoding buffers from strategy.
        Receive and parity buffers will be allocated later after metadata exchange.
        """
        self.eccheck_data_buffers = buffers['data_buffers']
        self.eccheck_encoding_buffers = buffers['encoding_buffers']
        self._free_data_buffer_queue = buffers['free_data_buffer_queue']
        self._free_encoding_buffer_queue = buffers['free_encoding_buffer_queue']
        
        # Receive and parity buffers are NOT set here
        # They will be allocated after metadata exchange when peer data size is known
        
        logger.info(
            f"EC-CHECK: Using pre-allocated buffers from strategy - "
            f"Data: {len(self.eccheck_data_buffers)}, "
            f"Encoding: {len(self.eccheck_encoding_buffers)}"
        )

    def _init_eccheck_buffers(self):
        """Initialize EC-CHECK buffers during C++ module initialization.
        
        Note: Only allocates data and encoding buffers here.
        Receive and parity buffers are allocated later after metadata exchange,
        when we know the peer's data size.
        """
        logger.info("EC-CHECK: Initializing buffers for EC-CHECK (data and encoding only)")
        
        # Allocate data buffers for storing original tensor data
        # Each worker reserves 12 data buffers, each 64MB in size
        self.eccheck_data_buffers = self._allocate_data_buffers()
        
        # Allocate encoding buffers for encoded packets
        # Each worker reserves 24 encoding buffers, each 64MB in size
        self.eccheck_encoding_buffers = self._allocate_encoding_buffers()
        
        # Initialize free buffer queues for Phase 3
        # Store buffer addresses directly in queue for easier management
        import queue
        self._free_data_buffer_queue = queue.Queue()
        for buffer in self.eccheck_data_buffers:
            self._free_data_buffer_queue.put(int(buffer.data_ptr()))
        
        self._free_encoding_buffer_queue = queue.Queue()
        for buffer in self.eccheck_encoding_buffers:
            self._free_encoding_buffer_queue.put(int(buffer.data_ptr()))
        
        logger.info(f"EC-CHECK: Initial buffer allocation completed - "
                   f"Data buffers: {len(self.eccheck_data_buffers)}, "
                   f"Encoding buffers: {len(self.eccheck_encoding_buffers)}")

    def prepare_write_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        First stage of async saving. Copy data to CPU and plan the local saving.

        Args:
            plan (SavePlan): save plan generated by the PyT Distributed compatible planner
            planner (SavePlanner): save planner used to resolve the bytes and tensor data

        Returns: None, but stores the save plan in `self.write_buckets`
        """
        # EC-CHECK mode: decompose state_dict and preallocate CPU memory
        if self.use_eccheck:
            self._prepare_eccheck_data(plan, planner)
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
    
    def _prepare_eccheck_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        EC-CHECK preparation: organize data for serialization-free encoding.
        
        This method performs the following steps:
        1. Process plan items like normal mode (separate bytes and tensors)
        2. Organize tensors for EC-CHECK (extract metadata and data)
        3. Preallocate CPU memory buffer for tensors
        4. Prepare write buckets for async transfer
        
        Args:
            plan (SavePlan): save plan from PyTorch distributed checkpoint
            planner (SavePlanner): save planner to resolve data
        """
        start_total = time()
        logger.info("EC-CHECK: Starting serialization-free checkpoint preparation")
        
        # Step 1: Process plan items (similar to normal mode)
        start = time()
        storage_plan: _StoragePrefix = plan.storage_data
        
        # Separate items into BYTE_IO (non-tensor) and TENSOR
        non_tensor_data = {}
        tensor_infos = []
        tensor_data_list = []
        
        logger.info(f"EC-CHECK: Processing {len(plan.items)} items from SavePlan")
        byte_io_count = 0
        tensor_count = 0
        none_data_count = 0
        
        for item in plan.items:
            data = planner.resolve_data(item)
            
            # Debug: check for None data
            if data is None:
                none_data_count += 1
                if none_data_count <= 5:
                    logger.warning(f"EC-CHECK SAVE: Found None data for item: fqn={item.index.fqn}, type={item.type}")
                continue  # Skip None data items
            
            if item.type == WriteItemType.BYTE_IO:
                # Non-tensor data (e.g., extra_state)
                # BytesIO objects need special handling to preserve format
                import io
                if isinstance(data, io.BytesIO):
                    # Store the BytesIO content directly as bytes
                    # We'll also store metadata to indicate this was a BytesIO
                    non_tensor_data[item.index.fqn] = {
                        '_eccheck_type': 'BytesIO',
                        '_eccheck_data': data.getvalue()
                    }
                else:
                    non_tensor_data[item.index.fqn] = data
                byte_io_count += 1
            else:
                # Tensor data - create TensorInfo
                # Extract and store serializable fields from WriteItem.index
                from .state_dict_decomposer import TensorInfo
                
                tensor_info = TensorInfo(
                    key=item.index.fqn,  # Keep FQN as base key
                    shape=tuple(data.shape),
                    dtype=data.dtype,
                    device=data.device,
                    numel=data.numel(),
                    size_bytes=data.numel() * data.element_size(),
                    offset=0,  # Will be calculated below
                    global_offset=tuple(item.index.offset),  # Extract offset as tuple (serializable)
                    shard_index=item.index.index,  # Extract index (serializable)
                )
                tensor_infos.append(tensor_info)
                tensor_data_list.append(data)
                tensor_count += 1
        
        logger.info(
            f"EC-CHECK: Processed {byte_io_count} BytesIO items, {tensor_count} tensor items"
            + (f", skipped {none_data_count} None items" if none_data_count > 0 else "")
        )
        
        # Calculate offsets for tensor data
        offset = 0
        for info in tensor_infos:
            info.offset = offset
            offset += info.size_bytes
        
        # Create decomposed structure
        self.decomposed_state_dict = DecomposedStateDict(
            non_tensor_data=non_tensor_data,
            tensor_infos=tensor_infos,
            tensor_data=tensor_data_list,
        )
        
        process_time = time() - start
        
        # Log statistics
        stats = self.decomposed_state_dict.get_statistics()
        logger.info(
            f"EC-CHECK: Processed plan items in {process_time:.2f}s\n"
            f"  Non-tensor items: {len(non_tensor_data)}\n"
            f"  Tensor items: {len(tensor_data_list)}\n"
            f"  Non-tensor data: {stats['non_tensor_size_bytes'] / 1024:.2f} KB "
            f"({stats['non_tensor_percentage']:.4f}%)\n"
            f"  Tensor keys: {stats['tensor_keys_size_bytes'] / 1024:.2f} KB "
            f"({stats['tensor_keys_percentage']:.4f}%)\n"
            f"  Tensor data: {stats['tensor_data_size_bytes'] / (1024**3):.2f} GB "
            f"({stats['tensor_data_percentage']:.2f}%)"
        )
        
        # Step 2: Preallocate CPU memory buffer if enabled
        if self.eccheck_preallocate_cpu_buffer:
            start = time()
            total_size = self.decomposed_state_dict.total_tensor_size_bytes
            logger.info(f"EC-CHECK: Preallocating CPU buffer of {total_size / (1024**3):.2f} GB")
            
            if self.eccheck_pin_memory and torch.cuda.is_available():
                self.preallocated_cpu_buffer = torch.empty(
                    total_size, dtype=torch.uint8
                ).pin_memory()
                logger.debug("EC-CHECK: Using pinned memory for CPU buffer")
            else:
                self.preallocated_cpu_buffer = torch.empty(
                    total_size, dtype=torch.uint8
                )
            
            prealloc_time = time() - start
            logger.debug(f"EC-CHECK: CPU buffer preallocation took {prealloc_time:.2f}s")
        else:
            prealloc_time = 0
        
        # Step 3: Prepare write buckets for async transfer
        start = time()
        self._prepare_eccheck_write_buckets(plan)
        bucket_time = time() - start
        logger.debug(f"EC-CHECK: Write bucket preparation took {bucket_time:.2f}s")
        
        total_time = time() - start_total
        logger.info(
            f"EC-CHECK: Preparation completed in {total_time:.2f}s\n"
            f"  Item processing: {process_time:.2f}s\n"
            f"  Preallocation: {prealloc_time:.2f}s\n"
            f"  Bucket prep: {bucket_time:.2f}s"
        )
        
        # Validate decomposition
        if not self.validate_eccheck_decomposition():
            raise RuntimeError("EC-CHECK: Decomposition validation failed")
    
    def _prepare_eccheck_write_buckets(self, plan: SavePlan) -> None:
        """
        Prepare write buckets for EC-CHECK mode.
        
        In EC-CHECK mode, each node stores three components in ONE file:
        1. Serialized non-tensor key-value pairs
        2. Serialized tensor keys (tensor_infos)
        3. Tensor data buffer (will be filled during preload)
        
        File structure:
        [Header: sizes of 3 components] [Component 1] [Component 2] [Component 3]
        
        Args:
            plan (SavePlan): save plan
        """
        storage_plan: _StoragePrefix = plan.storage_data
        
        self.write_buckets = []
        
        # Serialize Components 1 & 2 in CPU memory
        non_tensor_data = pickle.dumps(self.decomposed_state_dict.non_tensor_data)
        tensor_keys_data = pickle.dumps(self.decomposed_state_dict.tensor_infos)
        
        # Calculate sizes for header
        non_tensor_size = len(non_tensor_data)
        tensor_keys_size = len(tensor_keys_data)
        tensor_buffer_size = self.decomposed_state_dict.total_tensor_size_bytes
        
        # Create single file for all three components
        # Use standard .distcp extension for compatibility, but with EC-CHECK content
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # Use the same naming convention as standard distributed checkpoint
        # Format: __{rank}_{thread_id}.distcp
        eccheck_file = f"__{rank}_0.distcp"  # Single file per rank, thread 0
        eccheck_path = os.path.join(self.checkpoint_dir, eccheck_file)
        
        # Store the serialized metadata in CPU memory for async save
        self.eccheck_serialized_metadata = {
            'non_tensor_data': non_tensor_data,
            'tensor_keys_data': tensor_keys_data,
            'non_tensor_size': non_tensor_size,
            'tensor_keys_size': tensor_keys_size,
            'tensor_buffer_size': tensor_buffer_size,
            'eccheck_file_path': eccheck_path,
        }
        
        logger.debug(
            f"EC-CHECK: Prepared single file structure:\n"
            f"  File: {eccheck_file}\n"
            f"  Component 1 size: {non_tensor_size / 1024:.2f} KB\n"
            f"  Component 2 size: {tensor_keys_size / 1024:.2f} KB\n"
            f"  Component 3 size: {tensor_buffer_size / (1024**3):.2f} GB"
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
    
    def _prepare_local_metadata_for_broadcast(self, my_rank: int, world_size: int):
        """
        Prepare local tensor metadata for broadcasting to all ranks.
        
        Currently implements simple strategy:
        - Each rank keeps its own data chunks locally (target_rank = my_rank)
        - Future: Add parity chunk generation for redundancy
        
        Args:
            my_rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            List[TensorMetadata]: Serializable metadata for broadcasting
        """
        from .state_dict_decomposer import TensorMetadata
        
        local_metadata = []
        
        for info in self.decomposed_state_dict.tensor_infos:
            # Create metadata for data chunk
            data_meta = TensorMetadata(
                key=info.key,
                shape=info.shape,
                dtype=str(info.dtype),
                size_bytes=info.size_bytes,
                global_offset=info.global_offset if info.global_offset is not None else (),
                shard_index=info.shard_index if info.shard_index is not None else 0,
                chunk_type='data',
                target_rank=my_rank,  # Data stays on same rank
                source_rank=my_rank,
            )
            local_metadata.append(data_meta)
        
        return local_metadata
    
    def _broadcast_and_exchange_metadata(self):
        """
        All-to-all metadata exchange using torch.distributed.all_gather.
        
        Each rank broadcasts its metadata to all other ranks.
        After this call, all ranks have complete metadata from all peers.
        
        Returns:
            GlobalMetadataRegistry: Complete metadata from all ranks
        """
        from .state_dict_decomposer import GlobalMetadataRegistry
        import pickle
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        logger.info(f"EC-CHECK: [Rank {rank}] Starting metadata exchange with {world_size} ranks")
        
        # ===== Step 1: Prepare local metadata (both tensor and non-tensor) =====
        local_tensor_metadata = self._prepare_local_metadata_for_broadcast(rank, world_size)
        local_non_tensor_data = self.decomposed_state_dict.non_tensor_data
        
        # Package both together
        local_package = {
            'tensor_metadata': local_tensor_metadata,
            'non_tensor_data': local_non_tensor_data,
        }
        
        logger.info(
            f"EC-CHECK: [Rank {rank}] Local metadata: "
            f"{len(local_tensor_metadata)} tensor items, "
            f"{len(local_non_tensor_data)} non-tensor items"
        )
        
        # ===== Step 2: All-gather complete metadata using all_gather_object =====
        # This automatically handles serialization, padding, and deserialization
        # Transmits both tensor_metadata and non_tensor_data
        all_packages = [None] * world_size
        torch.distributed.all_gather_object(all_packages, local_package)
        
        # ===== Step 3: Build rank_metadata and rank_non_tensor_data dicts =====
        rank_metadata = {}
        rank_non_tensor_data = {}
        for i, package in enumerate(all_packages):
            rank_metadata[i] = package['tensor_metadata']
            rank_non_tensor_data[i] = package['non_tensor_data']
        
        logger.info(
            f"EC-CHECK: [Rank {rank}] Received metadata from all {world_size} ranks"
        )
        
        # Create registry with both tensor and non-tensor metadata
        registry = GlobalMetadataRegistry(
            rank_metadata=rank_metadata,
            rank_non_tensor_data=rank_non_tensor_data
        )
        
        # Log statistics
        stats = registry.get_statistics()
        logger.info(
            f"EC-CHECK: Global metadata exchange complete:\n"
            f"  Total ranks: {stats['total_ranks']}\n"
            f"  Total tensor items: {stats['total_tensor_chunks']}\n"
            f"  Total non-tensor items: {stats['total_non_tensor_items']}\n"
            f"  Metadata size: {stats['total_metadata_bytes'] / 1024:.2f} KB (actual transmitted)\n"
            f"  Tensor data size: {stats['total_tensor_data_bytes'] / (1024**3):.2f} GB (referenced, not transmitted)\n"
            f"  Per-rank tensor items: {stats['per_rank_tensor_items']}\n"
            f"  Per-rank non-tensor items: {stats['per_rank_non_tensor_items']}"
        )
        
        return registry
    
    def _get_paired_rank(self, my_rank: int, world_size: int) -> int:
        """
        Get the paired rank for parity exchange.
        
        Pairing strategy:
        - 2 ranks: rank0 ↔ rank1
        - 4 ranks: rank0 ↔ rank2, rank1 ↔ rank3
        - General: rank_i ↔ rank_{i + world_size/2}
        
        Args:
            my_rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            int: Paired rank ID
        """
        if world_size % 2 != 0:
            raise ValueError(f"EC-CHECK: World size must be even for pairing, got {world_size}")
        
        half_size = world_size // 2
        
        if my_rank < half_size:
            # First half pairs with second half
            paired_rank = my_rank + half_size
        else:
            # Second half pairs with first half
            paired_rank = my_rank - half_size
        
        logger.debug(f"EC-CHECK: Rank {my_rank} paired with Rank {paired_rank}")
        return paired_rank
    
    def _allocate_data_buffers(self):
        """
        Allocate data buffers for storing original tensor data.
        Configurable number of data buffers per worker.
        
        Returns:
            List[torch.Tensor]: List of data buffers
        """
        logger.info(f"EC-CHECK: Allocating data buffers ({self.eccheck_data_buffers_count} buffers, {self.eccheck_buffer_size // (1024*1024)}MB each)")
        
        data_buffers = []
        
        for i in range(self.eccheck_data_buffers_count):
            buffer = torch.empty(self.eccheck_buffer_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
            data_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated data buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.info(f"EC-CHECK: Allocated {len(data_buffers)} data buffers")
        return data_buffers
    
    def _allocate_encoding_buffers(self):
        """
        Allocate encoding buffers for encoded packets.
        Configurable number of encoding buffers per worker (data_count * m).
        
        Args:
            global_registry (GlobalMetadataRegistry): Complete metadata from all ranks
            
        Returns:
            List[torch.Tensor]: List of encoding buffers
        """
        logger.info(f"EC-CHECK: Allocating encoding buffers ({self.eccheck_encoding_buffers_count} buffers, {self.eccheck_buffer_size // (1024*1024)}MB each)")
        
        encoding_buffers = []
        
        for i in range(self.eccheck_encoding_buffers_count):
            buffer = torch.empty(self.eccheck_buffer_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
            encoding_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated encoding buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.info(f"EC-CHECK: Allocated {len(encoding_buffers)} encoding buffers")
        return encoding_buffers
    
    def _allocate_recv_encoding_buffers(self, global_registry):
        """
        Allocate large receive buffers for peer encoded packets (one per column).
        
        Each buffer is equal to peer's total data size, aligned to buffer_size (64MB).
        
        Args:
            global_registry (GlobalMetadataRegistry): Complete metadata from all ranks
            
        Returns:
            List[torch.Tensor]: List of receive buffers (one per column)
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        paired_rank = self._get_paired_rank(rank, world_size)
        
        # Get peer's total data size from global registry
        peer_metadata = global_registry.rank_metadata.get(paired_rank, [])
        peer_total_size = sum(meta.size_bytes for meta in peer_metadata)
        
        # Align peer's data size to buffer_size (64MB)
        aligned_size = ((peer_total_size + self.eccheck_buffer_size - 1) // self.eccheck_buffer_size) * self.eccheck_buffer_size
        
        # Determine number of columns
        columns_cfg = getattr(self, 'eccheck_columns_config', None)
        num_columns = len(columns_cfg) if isinstance(columns_cfg, list) and len(columns_cfg) > 0 else 2
        
        logger.info(
            f"EC-CHECK: Allocating {num_columns} receive buffers based on peer data size\n"
            f"  Paired rank: {paired_rank}\n"
            f"  Peer data size: {peer_total_size / (1024**3):.2f} GB\n"
            f"  Aligned buffer size (per buffer): {aligned_size / (1024**3):.2f} GB\n"
            f"  Total receive memory: {num_columns * aligned_size / (1024**3):.2f} GB"
        )
        
        # Allocate large continuous buffers (one for each column)
        recv_buffers = []
        for i in range(num_columns):
            recv_buffer = torch.empty(aligned_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
            recv_buffers.append(recv_buffer)
        
        logger.info(
            f"EC-CHECK: Allocated {num_columns} receive buffers: {aligned_size / (1024**3):.2f} GB each "
            f"({aligned_size / (1024**2):.0f} MB each)"
        )
        
        return recv_buffers
    
    def _allocate_parity_buffers(self):
        """
        Allocate parity buffers for XOR computation results.
        These buffers will store the parity packets after XOR reduction.
        
        Note: Each chunk requires 2 parity buffers (one per encoding thread).
        We allocate enough buffers to support concurrent XOR operations.
        
        Returns:
            List[torch.Tensor]: List of parity buffers
        """
        # Calculate number of chunks based on total tensor size
        total_bytes = self.decomposed_state_dict.total_tensor_size_bytes
        num_chunks = (total_bytes + self.eccheck_buffer_size - 1) // self.eccheck_buffer_size
        
        # Determine number of columns
        columns_cfg = getattr(self, 'eccheck_columns_config', None)
        num_columns = len(columns_cfg) if isinstance(columns_cfg, list) and len(columns_cfg) > 0 else 2
        
        # Each chunk requires num_columns parity buffers (one per column)
        # We allocate at least as many as encoding buffers to ensure sufficient concurrency
        # Use max to ensure we have enough even if calculation is off
        min_parity_buffers = max(
            self.eccheck_encoding_buffers_count,  # At least as many as encoding buffers
            num_chunks * num_columns,  # At least enough for all chunks (num_columns per chunk)
            self.eccheck_data_buffers_count * num_columns  # At least num_columns x data buffers
        )
        
        logger.info(
            f"EC-CHECK: Allocating parity buffers\n"
            f"  Total data size: {total_bytes / (1024**3):.2f} GB\n"
            f"  Estimated chunks: {num_chunks}\n"
            f"  Required parity buffers: {num_chunks * 2} (2 per chunk)\n"
            f"  Allocating: {min_parity_buffers} buffers"
        )
        
        parity_buffers = []
        
        # Allocate parity buffers for storing XOR results
        for i in range(min_parity_buffers):
            buffer = torch.empty(self.eccheck_buffer_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
            parity_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated parity buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.info(f"EC-CHECK: Allocated {len(parity_buffers)} parity buffers")
        return parity_buffers
    
    def _poll_and_release_buffers(self):
        """Poll C++ for buffers ready to be released and put them back to queues."""
        if self._eccheck_native is None:
            return
        
        # Get data buffers ready for release
        data_buffers = self._eccheck_native.get_data_buffers_to_release()
        for data_addr in data_buffers:
            try:
                self._free_data_buffer_queue.put_nowait(data_addr)
                logger.debug(f"EC-CHECK: Released data buffer at address {data_addr}")
            except queue.Full:
                logger.error(f"EC-CHECK: Data buffer queue is full, cannot release buffer {data_addr}")
        
        # Get encoding buffers ready for release
        encoding_buffers = self._eccheck_native.get_encoding_buffers_to_release()
        for encoding_addr in encoding_buffers:
            try:
                self._free_encoding_buffer_queue.put_nowait(encoding_addr)
                logger.debug(f"EC-CHECK: Released encoding buffer at address {encoding_addr}")
            except queue.Full:
                logger.error(f"EC-CHECK: Encoding buffer queue is full, cannot release buffer {encoding_addr}")
        
        # Get parity buffers ready for release
        if hasattr(self, '_free_parity_buffer_queue'):
            parity_buffers = self._eccheck_native.get_parity_buffers_to_release()
            for parity_addr in parity_buffers:
                try:
                    self._free_parity_buffer_queue.put_nowait(parity_addr)
                    logger.debug(f"EC-CHECK: Released parity buffer at address {parity_addr}")
                except queue.Full:
                    logger.error(f"EC-CHECK: Parity buffer queue is full, cannot release buffer {parity_addr}")
    
    def _start_buffer_poller_thread(self):
        """Start a persistent background thread to poll and release buffers."""
        import threading
        
        if self._buffer_poller_thread is not None:
            logger.warning("EC-CHECK: Buffer poller thread already started")
            return
        
        # Create control events
        self._buffer_poller_stop_event = threading.Event()
        self._buffer_poller_active_event = threading.Event()
        
        def buffer_poller_worker():
            """Persistent background thread that polls for buffer releases."""
            logger.info("EC-CHECK: Buffer poller thread started")
            poll_count = 0
            
            while not self._buffer_poller_stop_event.is_set():
                # Only poll when active
                if self._buffer_poller_active_event.is_set():
                    self._poll_and_release_buffers()
                    poll_count += 1
                    if poll_count % 1000 == 0:
                        logger.debug(f"EC-CHECK: Buffer poller running (polled {poll_count} times)")
                
                # Sleep briefly to avoid busy waiting
                import time
                time.sleep(0.001)  # 1ms
            
            logger.info("EC-CHECK: Buffer poller thread stopping")
        
        # Start the daemon thread
        self._buffer_poller_thread = threading.Thread(target=buffer_poller_worker, daemon=True)
        self._buffer_poller_thread.start()
        logger.info("EC-CHECK: Buffer poller thread created and started")
    
    def _stop_buffer_poller_thread(self):
        """Stop the persistent buffer poller thread."""
        if self._buffer_poller_thread is None:
            return
        
        logger.info("EC-CHECK: Stopping buffer poller thread...")
        
        # Signal the thread to stop
        if self._buffer_poller_stop_event:
            self._buffer_poller_stop_event.set()
        
        # Wait for thread to finish
        if self._buffer_poller_thread.is_alive():
            self._buffer_poller_thread.join(timeout=2.0)
            if self._buffer_poller_thread.is_alive():
                logger.warning("EC-CHECK: Buffer poller thread did not stop in time")
            else:
                logger.info("EC-CHECK: Buffer poller thread stopped successfully")
        
        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None

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
        
        # Ensure buffer poller is active before starting
        if self._buffer_poller_active_event:
            self._buffer_poller_active_event.set()
        
        # Poll a few times to ensure all buffers from previous round are released
        for _ in range(10):
            self._poll_and_release_buffers()
            import time
            time.sleep(0.01)  # Small delay to let C++ release buffers
        
        def get_free_data_buffer():
            """Get a free data buffer address, blocking if none available."""
            # Poll for released buffers before trying to get one
            self._poll_and_release_buffers()
            
            try:
                return self._free_data_buffer_queue.get(timeout=30.0)
            except queue.Empty:
                logger.warning("EC-CHECK: Waiting longer for free data buffer (system under load)...")
                # Print queue status for debugging
                logger.warning(f"EC-CHECK: Data buffer queue size: {self._free_data_buffer_queue.qsize()}")
                # Try polling again before giving up
                self._poll_and_release_buffers()
                try:
                    return self._free_data_buffer_queue.get(timeout=10.0)
                except queue.Empty:
                    raise RuntimeError("EC-CHECK: Failed to get data buffer after extended wait - possible deadlock")
        
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
                # Try polling again before giving up
                self._poll_and_release_buffers()
                logger.error("EC-CHECK: TIMEOUT waiting for free parity buffer - possible deadlock!")
                logger.error(f"EC-CHECK: Parity buffer queue size: {self._free_parity_buffer_queue.qsize()}")
                return self._free_parity_buffer_queue.get()
        
        # Process continuous tensor buffer sequentially
        # Copy data from self.tensor_buffer (continuous CPU buffer) to data buffers
        total_bytes = self.decomposed_state_dict.total_tensor_size_bytes
        src_pos = 0  # Current position in continuous tensor buffer
        # chunk_count = 0
        
        # Determine columns config
        columns_cfg = getattr(self, 'eccheck_columns_config', None)
        num_columns = len(columns_cfg) if isinstance(columns_cfg, list) and len(columns_cfg) > 0 else 2
        
        # Get base addresses of receive buffers (one per column)
        recv_buffers = self.eccheck_recv_encoding_buffers
        if len(recv_buffers) < num_columns:
            raise RuntimeError(f"EC-CHECK: Mismatch: {num_columns} columns configured but only {len(recv_buffers)} receive buffers allocated")
        
        recv_buffer_base_addrs = [int(buf.data_ptr()) for buf in recv_buffers[:num_columns]]
        recv_buffer_offsets = [0] * num_columns  # Current offset in each column's receive buffer

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
            
            # Prepare per-column resources
            # Check if we have pipeline hints (advanced mode)
            pipeline_hints = getattr(self, 'eccheck_pipeline_hints', {})
            zero_parity_buffers = getattr(self, 'eccheck_zero_parity_buffers', {})
            
            # IMPORTANT: All columns share the SAME parity buffer for this chunk
            # This enables incremental XOR updates: parity = parity XOR recv_encoding
            shared_parity_addr = get_free_parity_buffer()
            
            per_column = []  # list of tuples (col_idx, enc_addr, recv_addr, recv_chunk_size, parity_addr, zero_parity_addr)
            recv_chunk_size = min(self.eccheck_buffer_size, remaining_in_source)
            
            # Track chunk index for zero parity buffer selection
            chunk_idx = src_pos // self.eccheck_buffer_size
            
            # Loop through all columns - all share the same parity buffer
            for col_idx in range(num_columns):
                enc_addr = get_free_encoding_buffer()
                # All columns use the same parity buffer for incremental XOR
                parity_addr = shared_parity_addr
                
                # Determine recv_addr based on pipeline hints
                hints = pipeline_hints.get(col_idx, {})
                needs_recv = hints.get('needs_recv', True)  # default: needs recv
                
                if needs_recv:
                    recv_addr = recv_buffer_base_addrs[col_idx] + recv_buffer_offsets[col_idx]
                    recv_buffer_offsets[col_idx] += recv_chunk_size
                    recv_size = recv_chunk_size
                else:
                    recv_addr = 0  # No recv needed for this column
                    recv_size = 0
                
                # Get zero-initialized parity buffer if needed (only for column 0)
                zero_parity_addr = 0
                if col_idx == 0 and hints.get('needs_zero_parity', False):
                    col_zero_buffers = zero_parity_buffers.get(col_idx, [])
                    if chunk_idx < len(col_zero_buffers):
                        zero_parity_addr = int(col_zero_buffers[chunk_idx].data_ptr())
                
                per_column.append((col_idx, enc_addr, recv_addr, recv_size, parity_addr, zero_parity_addr))
            
            # Submit per column via generic API
            for (col_idx, enc_addr, recv_addr, rsz, parity_addr, zero_parity_addr) in per_column:
                # Determine XOR mode based on column index and pipeline hints
                hints = pipeline_hints.get(col_idx, {})
                xor_mode_str = hints.get('xor_mode', 'with_recv_encoding')
                
                # For shared parity model:
                # Column 0: with_zero_parity (parity = encoding XOR zero_parity)
                # Column 1+: incremental (parity = parity XOR recv_encoding)
                if col_idx == 0:
                    # First column: initialize parity with zero
                    if zero_parity_addr != 0:
                        xor_mode_int = 1  # WITH_ZERO_PARITY
                    else:
                        # Fallback: direct write (if no zero parity provided)
                        xor_mode_int = 1
                else:
                    # Subsequent columns: incremental update
                    # Always use INCREMENTAL mode for shared parity updates
                    xor_mode_int = 2  # INCREMENTAL
                    # Override if explicitly specified in config
                    if xor_mode_str == 'with_recv_encoding':
                        # If config explicitly says with_recv_encoding, use it but treat as incremental
                        # (since we're updating shared parity)
                        xor_mode_int = 2
                
                # Pass zero_parity_addr and xor_mode to C++
                # Get send_count and recv_count from pipeline hints
                hints = pipeline_hints.get(col_idx, {})
                send_count = hints.get('send_count', 1)
                recv_count = hints.get('recv_count', 1)
                
                self._eccheck_native.submit_data_for_encoding(
                    col_idx, cur_buffer_addr, take, enc_addr, recv_addr, rsz, parity_addr,
                    zero_parity_addr, xor_mode_int, send_count, recv_count
                )
                
                if zero_parity_addr != 0:
                    logger.debug(f"EC-CHECK: Column {col_idx} chunk {chunk_idx} initializing parity with zero buffer at 0x{zero_parity_addr:x}")
                elif col_idx > 0:
                    logger.debug(f"EC-CHECK: Column {col_idx} chunk {chunk_idx} incrementally updating shared parity")
            
            src_pos += take
        
        # Log buffer usage per column
        total_recv_used = sum(recv_buffer_offsets)
        buffer_usage_lines = [f"  Column {i} receive buffer used: {recv_buffer_offsets[i] / (1024**3):.2f} GB" 
                              for i in range(num_columns)]
        logger.info(
            "\n".join(buffer_usage_lines) + 
            f"\n  Total receive buffer used: {total_recv_used / (1024**3):.2f} GB"
        )
        
        # Mark end of stream for all encoders (send sentinel to each column)
        for col_idx in range(num_columns):
            self._eccheck_native.submit_data_for_encoding(
                col_idx, 0, 0, 0, 0, 0, 0
            )
        
        # Wait for all encoding threads to complete
        # Activate persistent buffer poller while waiting
        logger.info(f"EC-CHECK: Waiting for {num_columns} encoding threads to complete (with buffer polling)...")
        
        # Activate the persistent buffer poller
        if self._buffer_poller_active_event:
            self._buffer_poller_active_event.set()
        
        try:
            # Wait for encoding completion (this may block)
            self._eccheck_native.wait_for_encoding_completion()
            
            # Execute post_xor_steps if configured
            self._execute_post_xor_steps()
        finally:
            # Deactivate the buffer poller
            if self._buffer_poller_active_event:
                self._buffer_poller_active_event.clear()
        
        # Final poll to ensure all buffers are released
        # Poll multiple times with delays to ensure all buffers are released
        for _ in range(20):
            self._poll_and_release_buffers()
            import time
            time.sleep(0.01)
        
        # Verify all buffers are back in queues
        expected_data_buffers = len(self.eccheck_data_buffers)
        expected_encoding_buffers = len(self.eccheck_encoding_buffers)
        actual_data_buffers = self._free_data_buffer_queue.qsize()
        actual_encoding_buffers = self._free_encoding_buffer_queue.qsize()
        
        if actual_data_buffers < expected_data_buffers:
            logger.warning(
                f"EC-CHECK: Data buffer mismatch - expected {expected_data_buffers}, "
                f"got {actual_data_buffers} in free queue"
            )
        
        if actual_encoding_buffers < expected_encoding_buffers:
            logger.warning(
                f"EC-CHECK: Encoding buffer mismatch - expected {expected_encoding_buffers}, "
                f"got {actual_encoding_buffers} in free queue"
            )
        
        # If buffers are missing, try one more aggressive poll
        if actual_data_buffers < expected_data_buffers or actual_encoding_buffers < expected_encoding_buffers:
            logger.warning("EC-CHECK: Performing aggressive buffer recovery...")
            for _ in range(50):
                self._poll_and_release_buffers()
                import time
                time.sleep(0.02)
        
        # Final check
        final_data = self._free_data_buffer_queue.qsize()
        final_encoding = self._free_encoding_buffer_queue.qsize()
        if hasattr(self, '_free_parity_buffer_queue'):
            final_parity = self._free_parity_buffer_queue.qsize()
            logger.debug(
                f"EC-CHECK: Final buffer counts - Data: {final_data}/{expected_data_buffers}, "
                f"Encoding: {final_encoding}/{expected_encoding_buffers}, "
                f"Parity: {final_parity}"
            )
        
        # logger.info("EC-CHECK: All encoding operations completed")
    
    def _validate_pairing_compatibility(self, own_total_size: int, peer_total_size: int, my_rank: int, paired_rank: int):
        """
        Validate that paired ranks have compatible data sizes for XOR encoding.
        
        For XOR encoding to work properly, we need to handle size differences:
        - If sizes differ, we'll need padding or per-tensor XOR
        - Log warnings if significant size mismatch
        
        Args:
            own_total_size: Total size of own tensor data
            peer_total_size: Total size of paired rank's tensor data
            my_rank: Current rank
            paired_rank: Paired rank
        """
        own_gb = own_total_size / (1024**3)
        peer_gb = peer_total_size / (1024**3)
        
        logger.info(
            f"EC-CHECK: [Rank {my_rank}] Pairing validation:\n"
            f"  Own data size: {own_gb:.2f} GB\n"
            f"  Peer data size (Rank {paired_rank}): {peer_gb:.2f} GB"
        )
        
        if own_total_size != peer_total_size:
            size_diff = abs(own_total_size - peer_total_size)
            diff_percent = (size_diff / max(own_total_size, peer_total_size)) * 100
            
            logger.warning(
                f"EC-CHECK: [Rank {my_rank}] Size mismatch with Rank {paired_rank}:\n"
                f"  Difference: {size_diff / (1024**2):.2f} MB ({diff_percent:.1f}%)\n"
                f"  Will use per-tensor XOR (not continuous buffer XOR)"
            )
            
            # For now, we'll use per-tensor XOR which handles different sizes
            # Future optimization: padding for continuous buffer XOR
            return False  # Sizes don't match, use per-tensor XOR
        else:
            logger.info(
                f"EC-CHECK: [Rank {my_rank}] Perfect match with Rank {paired_rank}, "
                f"can use optimized continuous buffer XOR"
            )
            return True  # Sizes match, can use continuous buffer XOR
 
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
        
        # ===== Phase 2: Metadata Broadcast =====
        logger.warning("EC-CHECK: Phase 2 - Broadcasting metadata")
        phase2_start = time()
        
        # Step 2.1: Broadcast and exchange metadata (tensor + non-tensor)
        global_registry = self._broadcast_and_exchange_metadata()
        
        # Store global registry for Phase 3
        self.eccheck_global_registry = global_registry
        
        phase2_time = time() - phase2_start
        logger.warning(f"EC-CHECK: Phase 2 completed in {phase2_time:.2f}s")
        
        # ===== Phase 2.5: Allocate receive and parity buffers based on peer data size =====
        logger.info("EC-CHECK: Phase 2.5 - Allocating receive and parity buffers based on peer data")
        buffer_alloc_start = time()
        
        # Allocate receive buffers for peer encoded packets (based on peer's data size)
        self.eccheck_recv_encoding_buffers = self._allocate_recv_encoding_buffers(global_registry)
        
        # Allocate parity buffers for XOR computation results
        self.eccheck_parity_buffers = self._allocate_parity_buffers()
        
        # Initialize free parity buffer queue
        self._free_parity_buffer_queue = queue.Queue()
        for buffer in self.eccheck_parity_buffers:
            self._free_parity_buffer_queue.put(int(buffer.data_ptr()))
        
        # Allocate persistent stores (recv/parity) for in-memory final results (no write-out here)
        # recv_store size equals peer_total_size (aligned), parity_store equals own total size (aligned)
        try:
            # Peer total size already computed in _allocate_recv_encoding_buffers; recompute here
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            paired_rank = self._get_paired_rank(rank, world_size)
            peer_metadata = self.eccheck_global_registry.rank_metadata.get(paired_rank, []) if self.eccheck_global_registry else []
            peer_total_size = sum(meta.size_bytes for meta in peer_metadata)
            own_total_size = self.decomposed_state_dict.total_tensor_size_bytes
            aligned_peer = ((peer_total_size + self.eccheck_buffer_size - 1) // self.eccheck_buffer_size) * self.eccheck_buffer_size
            aligned_own = ((own_total_size + self.eccheck_buffer_size - 1) // self.eccheck_buffer_size) * self.eccheck_buffer_size
            
            if self.eccheck_persist_recv_enabled:
                self.eccheck_persist_recv_store = torch.empty(aligned_peer, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
                logger.info(f"EC-CHECK: Allocated persistent recv store: {aligned_peer / (1024**3):.2f} GB")
            if self.eccheck_persist_parity_enabled:
                self.eccheck_persist_parity_store = torch.empty(aligned_own, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
                logger.info(f"EC-CHECK: Allocated persistent parity store: {aligned_own / (1024**3):.2f} GB")
            
            # Pass persistent store addresses to C++ so that recv/xor workers memcpy into them
            recv_base = int(self.eccheck_persist_recv_store.data_ptr()) if self.eccheck_persist_recv_store is not None else 0
            parity_base = int(self.eccheck_persist_parity_store.data_ptr()) if self.eccheck_persist_parity_store is not None else 0
            persist_recv_flag = 1 if self.eccheck_persist_recv_enabled and recv_base != 0 else 0
            persist_parity_flag = 1 if self.eccheck_persist_parity_enabled and parity_base != 0 else 0
            if hasattr(self._eccheck_native, 'set_persist_stores'):
                # Pass capacities to guard memcpy in C++
                self._eccheck_native.set_persist_stores(
                    recv_base,
                    parity_base,
                    int(aligned_peer),
                    int(aligned_own),
                    persist_recv_flag,
                    persist_parity_flag,
                )
        except Exception as e:
            logger.warning(f"EC-CHECK: Failed to allocate or register persistent stores: {e}")
        
        buffer_alloc_time = time() - buffer_alloc_start
        logger.info(f"EC-CHECK: Phase 2.5 completed in {buffer_alloc_time:.2f}s")
        
        # Parse and apply column configuration (supports simple and advanced modes)
        logger.info("EC-CHECK: Phase 2.5 - Starting configuration parsing")
        try:
            cfg_path = self.eccheck_config_path or os.environ.get('ECCHECK_CONFIG_PATH')
            logger.info(f"EC-CHECK: Config path check - self.eccheck_config_path={self.eccheck_config_path}, env ECCHECK_CONFIG_PATH={os.environ.get('ECCHECK_CONFIG_PATH')}, final cfg_path={cfg_path}")
            logger.info(f"EC-CHECK: Config path: {cfg_path}, exists: {os.path.isfile(cfg_path) if cfg_path else False}")
            if cfg_path and os.path.isfile(cfg_path):
                logger.info(f"EC-CHECK: Loading configuration from {cfg_path}")
                import json
                with open(cfg_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"EC-CHECK: Configuration loaded successfully")
                
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                paired_rank = self._get_paired_rank(rank, world_size)
                logger.info(f"EC-CHECK: Rank={rank}, world_size={world_size}, paired_rank={paired_rank}")
                
                # Check for advanced mode: per-rank configuration
                ranks_config = data.get('ranks')
                logger.info(f"EC-CHECK: Checking config mode, ranks_config type: {type(ranks_config)}, is dict: {isinstance(ranks_config, dict)}")
                if isinstance(ranks_config, dict):
                    # Advanced mode: per-rank configuration
                    rank_str = str(rank)
                    logger.info(f"EC-CHECK: Looking for rank {rank_str} in ranks_config, available keys: {list(ranks_config.keys())}")
                    rank_config = ranks_config.get(rank_str)
                    if rank_config and isinstance(rank_config, dict):
                        columns_config = rank_config.get('columns', [])
                        post_xor_steps = rank_config.get('post_xor_steps', [])
                        logger.info(f"EC-CHECK: Using advanced config mode for rank {rank}, columns={len(columns_config)}, post_xor_steps={len(post_xor_steps)}")
                        norm_cols = self._parse_advanced_columns_config(columns_config, rank, world_size, paired_rank)
                        if norm_cols and hasattr(self._eccheck_native, 'set_columns_config'):
                            self._eccheck_native.set_columns_config(norm_cols)
                            logger.info(f"EC-CHECK: Applied advanced columns config ({len(norm_cols)} entries)")
                            # Store pipeline configs for future use (if C++ supports it)
                            self._store_pipeline_configs(columns_config)
                        # Store post_xor_steps for future use
                        if post_xor_steps:
                            if not hasattr(self, 'eccheck_post_xor_steps'):
                                self.eccheck_post_xor_steps = []
                            self.eccheck_post_xor_steps = post_xor_steps
                            logger.info(f"EC-CHECK: Stored post_xor_steps config ({len(post_xor_steps)} steps)")
                            logger.info(f"EC-CHECK: Post-XOR steps content: {post_xor_steps}")
                        else:
                            logger.warning(f"EC-CHECK: No post_xor_steps found in config for rank {rank}")
                        
                        # Allocate zero-initialized parity buffer if needed
                        self._allocate_zero_parity_buffers_if_needed()
                        logger.info(f"EC-CHECK: Configuration parsing completed for rank {rank}")
                    else:
                        logger.warning(f"EC-CHECK: No configuration found for rank {rank} in advanced mode, falling back to simple mode")
                        # Fall through to simple mode
                        ranks_config = None
                
                # Also allocate zero parity buffers for simple mode (check if needed)
                if ranks_config is None:
                    # For simple mode, check if we need zero parity (default: no)
                    if not hasattr(self, 'eccheck_zero_parity_buffers'):
                        self.eccheck_zero_parity_buffers = {}
                
                # Simple mode: backward compatible (all ranks use same config)
                if ranks_config is None:
                    columns = data.get('columns')
                    if isinstance(columns, list) and len(columns) > 0 and hasattr(self._eccheck_native, 'set_columns_config'):
                        # Normalize entries: {coefficient:int, send_peer:int, recv_peer:int}
                        # Note: -1 means "auto-calculate based on current rank" (paired_rank)
                        norm_cols = []
                        # Initialize pipeline hints for simple mode (default: needs send and recv)
                        if not hasattr(self, 'eccheck_pipeline_hints'):
                            self.eccheck_pipeline_hints = {}
                        
                        for col_idx, col in enumerate(columns):
                            if not isinstance(col, dict):
                                continue
                            coef = int(col.get('coefficient', col_idx))
                            # If send_peer is -1 or not specified, use paired_rank (auto-calculate)
                            send_peer_raw = col.get('send_peer', -1)
                            send_peer = int(send_peer_raw) if send_peer_raw is not None else -1
                            if send_peer < 0:
                                send_peer = paired_rank
                            # If recv_peer is -1 or not specified, use paired_rank (auto-calculate)
                            recv_peer_raw = col.get('recv_peer', -1)
                            recv_peer = int(recv_peer_raw) if recv_peer_raw is not None else -1
                            if recv_peer < 0:
                                recv_peer = paired_rank
                            norm_cols.append({'coefficient': coef, 'send_peer': send_peer, 'recv_peer': recv_peer})
                            
                            # Set default pipeline hints for simple mode
                            self.eccheck_pipeline_hints[col_idx] = {
                                'needs_send': True,
                                'needs_recv': True,
                                'xor_mode': 'with_recv_encoding',
                                'needs_zero_parity': False,
                                'send_count': 1,
                                'recv_count': 1
                            }
                        if norm_cols:
                            self._eccheck_native.set_columns_config(norm_cols)
                            logger.info(f"EC-CHECK: Applied simple columns config ({len(norm_cols)} entries)")
        except Exception as e:
            logger.warning(f"EC-CHECK: Failed to apply columns config: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("EC-CHECK: Phase 2.5 configuration parsing section completed")
        # IMPORTANT: Return write_buckets for multiprocessing
        # This is called from async_utils.PipelineAsyncCaller and must return write_buckets
        if self.write_buckets is None:
            logger.warning("EC-CHECK: write_buckets is None, returning empty list")
            return []
        
        return self.write_buckets
    
    def _parse_pipeline_step(self, pipeline, column_idx):
        """
        Parse pipeline steps for a column to extract execution hints.
        Returns a dict with execution flags and parameters.
        """
        if not pipeline or not isinstance(pipeline, list):
            return {
                'needs_send': True,
                'needs_recv': True,
                'xor_mode': 'with_recv_encoding',
                'needs_zero_parity': False,
                'send_count': 1,
                'recv_count': 1
            }
        
        result = {
            'needs_send': False,
            'needs_recv': False,
            'xor_mode': 'with_recv_encoding',  # default
            'needs_zero_parity': False,
            'send_count': 1,
            'recv_count': 1
        }
        
        for step in pipeline:
            if not isinstance(step, dict):
                continue
            
            step_type = step.get('type', '')
            step_name = step.get('step', '')
            
            if step_type == 'nccl_send' or step_name == 'send':
                result['needs_send'] = True
                result['send_count'] = int(step.get('count', 1))
            
            elif step_type == 'nccl_recv' or step_name == 'recv':
                result['needs_recv'] = True
                result['recv_count'] = int(step.get('count', 1))
            
            elif step_type == 'xor_update' or step_name == 'xor':
                mode = step.get('mode', 'with_recv_encoding')
                result['xor_mode'] = mode
                if mode == 'with_zero_parity':
                    result['needs_zero_parity'] = True
                    result['needs_recv'] = False  # with_zero_parity doesn't need recv
        
        return result
    
    def _parse_advanced_columns_config(self, columns_config, rank, world_size, paired_rank):
        """
        Parse advanced column configuration with pipeline support.
        Returns normalized columns compatible with simple mode API.
        
        For now, extracts basic info (coefficient, send_peer, recv_peer) from pipeline.
        Also stores pipeline execution hints for Python-side decision making.
        """
        norm_cols = []
        
        # Initialize pipeline hints storage
        if not hasattr(self, 'eccheck_pipeline_hints'):
            self.eccheck_pipeline_hints = {}
        
        for col_cfg in columns_config:
            if not isinstance(col_cfg, dict):
                continue
            
            column_idx = col_cfg.get('column_idx', len(norm_cols))
            coefficient = int(col_cfg.get('coefficient', column_idx))
            
            # Extract send_peer and recv_peer from pipeline
            send_peer = -1
            recv_peer = -1
            pipeline = col_cfg.get('pipeline', [])
            
            for step in pipeline:
                if not isinstance(step, dict):
                    continue
                step_type = step.get('type', '')
                if step_type == 'nccl_send':
                    target = step.get('target_rank', -1)
                    if target >= 0:
                        send_peer = target
                elif step_type == 'nccl_recv':
                    source = step.get('source_rank', -1)
                    if source >= 0:
                        recv_peer = source
            
            # Auto-calculate if not found in pipeline
            if send_peer < 0:
                send_peer = paired_rank
            if recv_peer < 0:
                recv_peer = paired_rank
            
            norm_cols.append({
                'coefficient': coefficient,
                'send_peer': send_peer,
                'recv_peer': recv_peer
            })
            
            # Store full pipeline config for future use
            if not hasattr(self, 'eccheck_pipeline_configs'):
                self.eccheck_pipeline_configs = {}
            self.eccheck_pipeline_configs[column_idx] = pipeline
            
            # Parse pipeline to get execution hints
            self.eccheck_pipeline_hints[column_idx] = self._parse_pipeline_step(pipeline, column_idx)
        
        return norm_cols
    
    def _execute_post_xor_steps(self):
        """
        Execute post-XOR steps (sync, send, recv) after all XOR operations complete.
        These steps are configured in post_xor_steps section of the config.
        """
        post_xor_steps = getattr(self, 'eccheck_post_xor_steps', [])
        logger.info(f"EC-CHECK: _execute_post_xor_steps called, post_xor_steps={post_xor_steps}, type={type(post_xor_steps)}, hasattr={hasattr(self, 'eccheck_post_xor_steps')}")
        if not post_xor_steps:
            logger.info("EC-CHECK: No post-xor steps configured, skipping")
            return  # No post-xor steps configured
        
        logger.info(f"EC-CHECK: Executing {len(post_xor_steps)} post-XOR steps")
        
        # Allocate post-xor buffers if needed
        # These buffers are for exchanging data/parity after XOR
        if not hasattr(self, 'eccheck_post_xor_buffers'):
            self.eccheck_post_xor_buffers = {}
        
        for step_idx, step in enumerate(post_xor_steps):
            if not isinstance(step, dict):
                continue
            
            step_name = step.get('step', '')
            step_type = step.get('type', '')
            
            if step_name == 'sync' and step_type == 'barrier':
                # Synchronization barrier
                sync_point = step.get('sync_point', 'after_all_xor')
                ranks = step.get('ranks', [])
                logger.info(f"EC-CHECK: Post-XOR sync point: {sync_point}, ranks: {ranks}")
                # TODO: Implement barrier synchronization (might need C++ support or use torch.distributed.barrier)
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
                    logger.info(f"EC-CHECK: Post-XOR barrier completed")
            
            elif step_name == 'send' and step_type == 'nccl_send':
                # Send data/parity to target rank
                target_rank = step.get('target_rank', -1)
                data_type = step.get('data_type', 'parity')
                data_source = step.get('data_source', 'local_parity')
                logger.info(f"EC-CHECK: Processing post-XOR send step: target_rank={target_rank}, data_type={data_type}, data_source={data_source}")
                
                if target_rank < 0:
                    logger.warning(f"EC-CHECK: Post-XOR send: Invalid target_rank {target_rank}")
                    continue
                
                # Determine source buffer address and size
                send_addr = 0
                send_size = 0
                
                if data_source == 'local_parity' and hasattr(self, 'eccheck_persist_parity_store'):
                    if self.eccheck_persist_parity_store is not None:
                        send_addr = int(self.eccheck_persist_parity_store.data_ptr())
                        send_size = self.eccheck_persist_parity_store.numel()
                        logger.info(f"EC-CHECK: Post-XOR send: target_rank={target_rank}, data_type={data_type}, source={data_source}, size={send_size}, addr=0x{send_addr:x}")
                    else:
                        logger.warning(f"EC-CHECK: Post-XOR send: parity store not allocated")
                        continue
                elif data_source == 'local_data' and hasattr(self, 'tensor_buffer'):
                    if self.tensor_buffer is not None:
                        send_addr = int(self.tensor_buffer.data_ptr())
                        send_size = self.tensor_buffer.numel()
                        logger.info(f"EC-CHECK: Post-XOR send: target_rank={target_rank}, data_type={data_type}, source={data_source}, size={send_size}, addr=0x{send_addr:x}")
                    else:
                        logger.warning(f"EC-CHECK: Post-XOR send: tensor buffer not available")
                        continue
                else:
                    logger.warning(f"EC-CHECK: Post-XOR send: Unknown data_source '{data_source}'")
                    continue
                
                # Call C++ post_xor_send
                if hasattr(self._eccheck_native, 'post_xor_send') and send_addr > 0 and send_size > 0:
                    try:
                        logger.info(f"EC-CHECK: Calling C++ post_xor_send(target_rank={target_rank}, addr=0x{send_addr:x}, size={send_size})")
                        self._eccheck_native.post_xor_send(target_rank, send_addr, send_size)
                        logger.info(f"EC-CHECK: Post-XOR send completed to rank {target_rank}")
                    except Exception as e:
                        logger.error(f"EC-CHECK: Post-XOR send failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning(f"EC-CHECK: Post-XOR send: C++ API not available or invalid params (hasattr={hasattr(self._eccheck_native, 'post_xor_send')}, send_addr={send_addr}, send_size={send_size})")
            
            elif step_name == 'recv' and step_type == 'nccl_recv':
                # Receive data/parity from source rank
                source_rank = step.get('source_rank', -1)
                data_type = step.get('data_type', 'data')
                data_target = step.get('data_target', 'peer_data_buffer')
                logger.info(f"EC-CHECK: Processing post-XOR recv step: source_rank={source_rank}, data_type={data_type}, data_target={data_target}")
                
                if source_rank < 0:
                    logger.warning(f"EC-CHECK: Post-XOR recv: Invalid source_rank {source_rank}")
                    continue
                
                # Allocate or get receive buffer
                recv_addr = 0
                recv_size = 0
                
                if data_target == 'peer_data_buffer':
                    # Allocate buffer for peer data if not exists
                    if not hasattr(self, 'eccheck_post_xor_buffers'):
                        self.eccheck_post_xor_buffers = {}
                    
                    if 'peer_data_buffer' not in self.eccheck_post_xor_buffers:
                        # Allocate buffer based on peer's data size
                        if hasattr(self, 'eccheck_global_registry') and self.eccheck_global_registry:
                            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                            peer_metadata = self.eccheck_global_registry.rank_metadata.get(source_rank, [])
                            peer_total_size = sum(meta.size_bytes for meta in peer_metadata)
                            aligned_size = ((peer_total_size + self.eccheck_buffer_size - 1) // self.eccheck_buffer_size) * self.eccheck_buffer_size
                            
                            self.eccheck_post_xor_buffers['peer_data_buffer'] = torch.empty(
                                aligned_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory
                            )
                            logger.info(f"EC-CHECK: Allocated post-xor peer_data_buffer: {aligned_size / (1024**3):.2f} GB")
                    
                    buffer = self.eccheck_post_xor_buffers.get('peer_data_buffer')
                    if buffer is not None:
                        recv_addr = int(buffer.data_ptr())
                        recv_size = buffer.numel()
                        logger.info(f"EC-CHECK: Post-XOR recv: source_rank={source_rank}, data_type={data_type}, target={data_target}, size={recv_size}")
                    else:
                        logger.warning(f"EC-CHECK: Post-XOR recv: Failed to allocate peer_data_buffer")
                        continue
                elif data_target == 'peer_parity_buffer':
                    # Use peer's parity store if available, or allocate new buffer
                    if hasattr(self, 'eccheck_persist_recv_store') and self.eccheck_persist_recv_store is not None:
                        recv_addr = int(self.eccheck_persist_recv_store.data_ptr())
                        recv_size = self.eccheck_persist_recv_store.numel()
                        logger.info(f"EC-CHECK: Post-XOR recv: source_rank={source_rank}, data_type={data_type}, target={data_target}, size={recv_size}")
                    else:
                        logger.warning(f"EC-CHECK: Post-XOR recv: recv_store not available")
                        continue
                else:
                    logger.warning(f"EC-CHECK: Post-XOR recv: Unknown data_target '{data_target}'")
                    continue
                
                # Call C++ post_xor_recv
                if hasattr(self._eccheck_native, 'post_xor_recv') and recv_addr > 0 and recv_size > 0:
                    try:
                        logger.info(f"EC-CHECK: Calling C++ post_xor_recv(source_rank={source_rank}, addr=0x{recv_addr:x}, size={recv_size})")
                        self._eccheck_native.post_xor_recv(source_rank, recv_addr, recv_size)
                        logger.info(f"EC-CHECK: Post-XOR recv completed from rank {source_rank}")
                    except Exception as e:
                        logger.error(f"EC-CHECK: Post-XOR recv failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning(f"EC-CHECK: Post-XOR recv: C++ API not available or invalid params (hasattr={hasattr(self._eccheck_native, 'post_xor_recv')}, recv_addr={recv_addr}, recv_size={recv_size})")
        
        logger.info("EC-CHECK: Post-XOR steps execution completed")
    
    def _allocate_zero_parity_buffers_if_needed(self):
        """
        Allocate zero-initialized parity buffers for columns that need them.
        These buffers are used for the first column that XORs with zero-initialized parity.
        """
        if not hasattr(self, 'eccheck_pipeline_hints'):
            return
        
        if not hasattr(self, 'eccheck_zero_parity_buffers'):
            self.eccheck_zero_parity_buffers = {}
        
        total_bytes = self.decomposed_state_dict.total_tensor_size_bytes
        num_chunks = (total_bytes + self.eccheck_buffer_size - 1) // self.eccheck_buffer_size
        
        for column_idx, hints in self.eccheck_pipeline_hints.items():
            if hints.get('needs_zero_parity', False):
                # Allocate one zero-initialized buffer per chunk for this column
                # In practice, we might reuse one buffer if XOR is sequential, but for now allocate per chunk
                buffers = []
                for chunk_idx in range(num_chunks):
                    # Use torch.zeros to create zero-initialized buffer
                    zero_buffer = torch.zeros(self.eccheck_buffer_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
                    buffers.append(zero_buffer)
                    logger.debug(f"EC-CHECK: Allocated zero-initialized parity buffer for column {column_idx}, chunk {chunk_idx}")
                
                self.eccheck_zero_parity_buffers[column_idx] = buffers
                logger.info(f"EC-CHECK: Allocated {len(buffers)} zero-initialized parity buffers for column {column_idx}")
    
    def _store_pipeline_configs(self, columns_config):
        """Store pipeline configurations for potential future C++ pipeline engine."""
        if not hasattr(self, 'eccheck_pipeline_configs'):
            self.eccheck_pipeline_configs = {}
        
        for col_cfg in columns_config:
            if not isinstance(col_cfg, dict):
                continue
            column_idx = col_cfg.get('column_idx', len(self.eccheck_pipeline_configs))
            pipeline = col_cfg.get('pipeline', [])
            if pipeline:
                self.eccheck_pipeline_configs[column_idx] = pipeline
                logger.debug(f"EC-CHECK: Stored pipeline config for column {column_idx}: {len(pipeline)} steps")
        
        exec_start = time()
        # Execute Phase 3: Tensor data exchange and encoding
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
