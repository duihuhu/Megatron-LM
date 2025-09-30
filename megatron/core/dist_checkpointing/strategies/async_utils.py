# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
This module provides an async utilities which allow to start
a checkpoint save process in the background.
"""
import dis
import gc
import logging
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from queue import Empty
from time import sleep, time
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, get_args

import torch
from torch import multiprocessing as mp
import os

from ..utils import debug_time

logger = logging.getLogger(__name__)


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
        self.start_time: Optional[float] = None

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
        if async_req.preload_fn:
            # If there's a preload_fn in `async_req`, we call this func
            # to do the defined action in `async_req.preload_fn` to
            # stage GPU tensors to its defined destination
            async_fn_args[1] = async_req.preload_fn()
            
        rank = torch.distributed.get_rank()
        start_sync = time()
        torch.cuda.synchronize()
        end_sync = time()
        logger.debug(f"rank: {rank}, takes {end_sync - start_sync} to finish D2H ")

        ctx = mp.get_context('fork')
        self.start_time = time()
        self.process = ctx.Process(
            target=async_req.async_fn, args=async_fn_args, kwargs=async_req.async_fn_kwargs
        )
        self.process.start()
        init_time = time()
        logger.debug(f"rank: {rank}, takes {init_time - self.start_time} to schedule async ckpt ")

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

    def execute_sync(self, async_req: AsyncRequest) -> None:
        """Execute async request synchronously using pipeline workers.
        
        This method provides the same pipeline benefits (parallel GPU->CPU transfer
        and pipelined execution) but waits for completion before returning.
        This allows using pipeline optimization even without --async-save.
        
        Args:
            async_req (AsyncRequest): async request to execute synchronously
        """
        if async_req.async_fn is None:
            return
        
        # Schedule the async call
        self.schedule_async_call(async_req)
        
        # Wait for completion (blocking) with timeout
        logger.debug("PipelineAsyncCaller: Waiting for synchronous completion...")
        max_wait_time = 600  # 10 minutes timeout for sync execution
        wait_count = 0
        
        while not self.is_current_async_call_done(blocking=False, no_dist=True):
            import time
            time.sleep(0.1)  # Small sleep to avoid busy waiting
            wait_count += 1
            
            if wait_count >= max_wait_time * 10:  # 0.1s * 10 * 600 = 600s
                logger.error("PipelineAsyncCaller: Timeout in synchronous execution")
                # Try to force completion by checking one more time with blocking=True
                try:
                    self.is_current_async_call_done(blocking=True, no_dist=True)
                except Exception as e:
                    logger.error(f"Failed to force completion: {e}")
                break
            
            if wait_count % 100 == 0:  # Log every 10 seconds
                logger.debug(f"PipelineAsyncCaller: Still waiting for completion ({wait_count/10:.1f}s elapsed)")
        
        # Execute finalization functions
        try:
            torch.distributed.barrier()
            for finalize_fn in async_req.finalize_fns:
                finalize_fn()
            logger.debug("PipelineAsyncCaller: Synchronous execution completed")
        except Exception as e:
            logger.error(f"Error in finalization: {e}")
            raise

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            # Avoid exceptions in __del__ which can cause issues
            import sys
            print(f"Warning: Error in PipelineAsyncCaller.__del__: {e}", file=sys.stderr)

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

    def __init__(self, persistent: bool = False, pipeline: bool = False, num_workers: int = None):
        self.async_calls: deque[_ActiveAsyncRequest] = deque([])
        self.call_idx: int = -1
        self.persistent: bool = persistent
        self.pipeline: bool = pipeline
        self.num_workers: int = num_workers
        self.persistent_caller: AsyncCaller = None
        self.pipeline_caller: AsyncCaller = None

    def _get_async_caller(self):
        if self.pipeline:
            if self.pipeline_caller is None:
                self.pipeline_caller = PipelineAsyncCaller(self.num_workers)
            return self.pipeline_caller
        elif self.persistent:
            if self.persistent_caller is None:
                self.persistent_caller = PersistentAsyncCaller()
            return self.persistent_caller
        else:
            return TemporalAsyncCaller()

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

    def execute_sync_request(self, async_request: AsyncRequest) -> None:
        """Execute async request synchronously using pipeline workers.
        
        This method provides pipeline benefits but waits for completion.
        Useful for sync checkpointing with pipeline optimization.

        Args:
            async_request (AsyncRequest): async request to execute synchronously
        """
        async_caller = self._get_async_caller()
        
        # For pipeline caller, use the execute_sync method
        if isinstance(async_caller, PipelineAsyncCaller):
            async_caller.execute_sync(async_request)
        else:
            # For other callers, fall back to the standard sync execution
            async_request.execute_sync()

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
        if self.pipeline and self.pipeline_caller:
            self.pipeline_caller.close()


class PipelineAsyncCaller(AsyncCaller):
    """Pipeline-based async caller with pre-created worker processes.
    
    This implementation creates a pool of worker processes during initialization
    to avoid cold start overhead. It implements pipelining between GPU-to-CPU
    data transfer and disk writing operations.
    """

    def __init__(self, num_workers: int = None):
        """Initialize pipeline async caller with worker pool.
        
        Args:
            num_workers (int): Number of worker processes to create. 
                             Defaults to number of write buckets or 2.
        """
        self.num_workers = num_workers or 2
        from megatron.training import get_args
        args = get_args()
        if hasattr(args, 'instance_seq'):
            self.instance_seq = args.instance_seq
        if hasattr(args, 'gpus_per_node'):
            self.gpus_per_node = args.gpus_per_node
        self.workers: List[mp.Process] = []
        self.task_queues: List[mp.Queue] = []
        self.result_queues: List[mp.Queue] = []
        self.completion_queue: mp.Queue = None
        self.stage_sync_queue: mp.Queue = None  # For pipeline stage synchronization
        self.active_requests: Dict[int, AsyncRequest] = {}
        self.collected_results: Dict[int, Dict[int, List]] = {}  # call_id -> {worker_id -> results}
        self.next_worker_idx = 0
        self.call_counter = 0
        
        self._init_worker_pool()
    def _init_worker_pool(self):
        """Initialize the worker process pool."""
        if hasattr(self, 'workers') and self.workers:
            return  # 避免重复创建
        ctx = mp.get_context('spawn')
        self.completion_queue = ctx.Queue()
        self.stage_sync_queue = ctx.Queue()  # For pipeline stage synchronization
        
        for worker_id in range(self.num_workers):
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()
            
            worker = ctx.Process(
                target=self._worker_main_loop,
                args=(
                    worker_id,
                    task_queue,
                    result_queue,
                    self.completion_queue,
                    self.stage_sync_queue,
                    torch.distributed.get_rank(),
                    logging.getLogger().getEffectiveLevel(),
                    self.instance_seq,
                    self.num_workers,
                    self.gpus_per_node
                )
            )
            worker.start()
            
            self.workers.append(worker)
            self.task_queues.append(task_queue)
            self.result_queues.append(result_queue)
        # logger.info(f"PipelineAsyncCaller: Initialized {self.num_workers} worker processes")

    @staticmethod
    def _init_workers_comm(worker_id: int, rank: int, instance_seq: int, num_workers: int, gpus_per_node: int):
        # print(f"Worker {worker_id} for rank {rank} has instance_seq {instance_seq} num_workers {num_workers} gpus_per_node {gpus_per_node}")

        device_id = rank % gpus_per_node
        torch.cuda.set_device(device_id)
        worker_rank = (rank * num_workers) + worker_id

        world_size = int(os.environ["WORLD_SIZE"])
        worker_port = int(os.environ.get('MASTER_PORT', '29500')) + 1000
        addr = os.environ["MASTER_ADDR"]
        print(f"Worker {worker_id} for rank {rank} has instance_seq {instance_seq} worker_rank {worker_rank} gpus_per_node {gpus_per_node}  num_workers {num_workers} world_size {world_size} worker_port {worker_port} master_addr {addr}")

        # worker_rank = (instance_seq  * gpus_per_node * num_workers) + rank + worker_id
        torch.distributed.init_process_group(backend='gloo', rank=worker_rank, world_size=num_workers * world_size)
        print(f"Worker xxxxxxxxxxxxxxxxxxx")
        if instance_seq % 2 == 0:
            peer_rank = worker_rank + num_workers * gpus_per_node
        else:
            peer_rank = worker_rank - num_workers * gpus_per_node
        worker_comm = torch.distributed.new_group(ranks=[worker_rank, peer_rank])
        # torch.distributed.barrier()
        print(f"Worker {worker_id} for rank {rank} has instance_seq {instance_seq} worker_rank {worker_rank} gpus_per_node {gpus_per_node} peer_rank {peer_rank} num_workers {num_workers}")

        return worker_rank, peer_rank, worker_comm


    @staticmethod
    def _worker_main_loop(
        worker_id: int,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        completion_queue: mp.Queue,
        stage_sync_queue: mp.Queue,
        rank: int,
        log_level: int,
        instance_seq: int,
        num_workers: int,
        gpus_per_node: int
    ):
        """Main loop for worker processes with pipeline stage synchronization."""
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        logger.info(f"Worker {worker_id} for rank {rank} started")
        
        src_rank, peer_rank, comm = PipelineAsyncCaller._init_workers_comm(worker_id, rank, instance_seq, num_workers, gpus_per_node)
        
        while True:
            try:
                task = task_queue.get()  # Add timeout to allow periodic checks
                if task is None:  # Shutdown signal
                    logger.info(f"Worker {worker_id}: Received shutdown signal")
                    break
                
                task_type, call_id, data = task
                
                if task_type == "pipeline_preload_and_write":
                    # Execute preload and write in true pipeline fashion
                    async_req, write_buckets_slice, total_workers = data
                    
                    logger.debug(f"Worker {worker_id}: Processing call {call_id} with {len(write_buckets_slice)} buckets")
                    
                    # Handle empty bucket slice case
                    if not write_buckets_slice:
                        logger.debug(f"Worker {worker_id}: No buckets to process for call {call_id}, completing immediately")
                        completion_queue.put((call_id, worker_id, "completed", []))
                        continue
                    
                    # Step 1: Wait for turn to do GPU to CPU transfer
                    logger.debug(f"Worker {worker_id}: Waiting for GPU->CPU turn for call {call_id}")
                    try:
                        PipelineAsyncCaller._wait_for_gpu_transfer_turn(
                            worker_id, call_id, stage_sync_queue, total_workers, logger
                        )
                    except RuntimeError as e:
                        if "Shutdown signal received" in str(e):
                            logger.info(f"Worker {worker_id}: Exiting due to shutdown signal")
                            break  # Exit the main loop
                        else:
                            logger.error(f"Worker {worker_id}: Error waiting for GPU transfer turn: {e}")
                            raise  # Re-raise other RuntimeErrors
                    
                    # Step 2: GPU to CPU transfer (preload) - only this worker does it now
                    logger.debug(f"Worker {worker_id}: Starting GPU->CPU transfer for call {call_id}")
                    try:
                        t1 = time()
                        preloaded_buckets = PipelineAsyncCaller._preload_bucket_slice(
                            write_buckets_slice, non_blocking=True
                        )
                        t2 = time()
                        print(f"Worker {worker_id}: GPU->CPU transfer completed for call {call_id} {t2 - t1} seconds", t1, t2)
                        logger.debug(f"Worker {worker_id}: GPU->CPU transfer completed for call {call_id}")
                    except Exception as e:
                        logger.error(f"Worker {worker_id}: Error in GPU->CPU transfer for call {call_id}: {e}")
                        raise
                    
                    # Step 3: Signal that GPU->CPU is done, next worker can start
                    logger.debug(f"Worker {worker_id}: Signaling GPU->CPU completion for call {call_id}")
                    try:
                        PipelineAsyncCaller._signal_gpu_transfer_done(
                            worker_id, call_id, stage_sync_queue, logger
                        )
                    except Exception as e:
                        logger.error(f"Worker {worker_id}: Error signaling completion for call {call_id}: {e}")
                        raise
                    
                    # Step 4: Choose between data transfer or disk write
                    # use_data_transfer = async_req.async_fn_kwargs.get('use_data_transfer', False)
                    use_data_transfer = False
                    
                    
                    if use_data_transfer:
                        # Data transfer between workers (can overlap with next worker's GPU->CPU transfer)
                        logger.debug(f"Worker {worker_id}: Starting data transfer for call {call_id}")
                        try:
                            transfer_results = PipelineAsyncCaller._transfer_bucket_slice(
                                worker_id, preloaded_buckets, async_req, src_rank, peer_rank, instance_seq, comm
                            )
                            logger.debug(f"Worker {worker_id}: Data transfer completed for call {call_id}, {len(transfer_results)} results")
                            write_results = transfer_results
                        except Exception as e:
                            logger.error(f"Worker {worker_id}: Error in data transfer for call {call_id}: {e}")
                            raise
                    else:
                        # Write to disk (can overlap with next worker's GPU->CPU transfer)
                        logger.debug(f"Worker {worker_id}: Starting disk write for call {call_id}")
                        try:
                            write_results = PipelineAsyncCaller._write_bucket_slice(
                                worker_id, preloaded_buckets, async_req
                            )
                            logger.debug(f"Worker {worker_id}: Disk write completed for call {call_id}, {len(write_results)} results")
                        except Exception as e:
                            logger.error(f"Worker {worker_id}: Error in disk write for call {call_id}: {e}")
                            raise
                    
                    # Step 5: Notify completion with write results
                    completion_queue.put((call_id, worker_id, "completed", write_results))
                    logger.debug(f"Worker {worker_id}: Completed call {call_id} successfully")
                
            except Exception as e:
                if "timeout" in str(e).lower() or "Empty" in str(e):
                    # Timeout waiting for task, continue loop to check for shutdown
                    continue
                else:
                    # Log detailed error information
                    import traceback
                    error_details = f"Worker {worker_id} error: {e}\nTraceback: {traceback.format_exc()}"
                    logger.error(error_details)
                    
                    # Only put error in completion_queue if we have a valid call_id
                    if 'call_id' in locals():
                        completion_queue.put((call_id, worker_id, f"error: {e}", []))
                    else:
                        # Error occurred outside of task processing (e.g., during initialization)
                        logger.error(f"Worker {worker_id}: Error outside task processing: {e}")
        
        logger.info(f"Worker {worker_id} for rank {rank} terminated")

    @staticmethod
    def _wait_for_gpu_transfer_turn(
        worker_id: int, call_id: int, stage_sync_queue: mp.Queue, 
        total_workers: int, logger
    ):
        """Wait for this worker's turn to perform GPU->CPU transfer."""
        if worker_id == 0:
            # First worker can start immediately
            return
        
        # Wait for previous worker to signal completion with timeout
        expected_signal = f"gpu_done_{call_id}_{worker_id - 1}"
        max_wait_time = 300  # 5 minutes timeout
        wait_count = 0
        
        while True:
            try:
                signal = stage_sync_queue.get()
                if signal == expected_signal:
                    logger.debug(f"Worker {worker_id}: Received expected signal {signal}")
                    break
                elif signal == "SHUTDOWN":
                    # Shutdown signal received
                    logger.info(f"Worker {worker_id}: Received shutdown signal, exiting wait")
                    raise RuntimeError("Shutdown signal received")
                else:
                    # Put back unexpected signal for other workers
                    stage_sync_queue.put(signal)
                    logger.debug(f"Worker {worker_id}: Received unexpected signal {signal}, putting back")
            except Exception as e:
                # Timeout or other error
                wait_count += 1
                if wait_count >= max_wait_time:
                    logger.error(f"Worker {worker_id}: Timeout waiting for signal {expected_signal} after {max_wait_time}s")
                    raise RuntimeError(f"Timeout waiting for GPU transfer turn: {expected_signal}")
                if wait_count % 30 == 0:  # Log every 30 seconds
                    logger.warning(f"Worker {worker_id}: Still waiting for signal {expected_signal} ({wait_count}s elapsed)")
                continue

    @staticmethod
    def _signal_gpu_transfer_done(
        worker_id: int, call_id: int, stage_sync_queue: mp.Queue, logger
    ):
        """Signal that GPU->CPU transfer is complete, next worker can start."""
        signal = f"gpu_done_{call_id}_{worker_id}"
        stage_sync_queue.put(signal)
        logger.debug(f"Worker {worker_id}: Sent signal {signal}")

    @staticmethod
    def _preload_bucket_slice(write_buckets_slice: List, non_blocking=True):
        """Preload a slice of write buckets from GPU to CPU."""
        result = []
        for bucket in write_buckets_slice:
            file_name, storage_key, (bytes_data, tensor_data) = bucket          
            tensor_data = [
                (item, tensor.to("cpu", non_blocking=non_blocking)) 
                for item, tensor in tensor_data
            ]
            result.append((file_name, storage_key, (bytes_data, tensor_data)))
        
        if non_blocking:
            torch.cuda.synchronize()
        return result

    @staticmethod
    def _write_bucket_slice(worker_id: int, preloaded_buckets: List, async_req: AsyncRequest):
        """Write a slice of preloaded buckets to disk and return write results."""
        from megatron.core.dist_checkpointing.strategies.filesystem_async import _write_item
        import inspect
        import os
        
        # Import necessary components for writing
        try:
            from torch.distributed.checkpoint.filesystem import SerializationFormat
            extra_kwargs = {"serialization_format": SerializationFormat.TORCH_SAVE}
        except ImportError:
            extra_kwargs = {}
        
        # Extract parameters from async_req.async_fn_kwargs
        transform_list = async_req.async_fn_kwargs.get('transform_list', [])
        use_msc = async_req.async_fn_kwargs.get('use_msc', False)
        
        # Collect write results for this worker
        local_results = []
        
        for bucket in preloaded_buckets:
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            
            # Determine file opening method
            if use_msc:
                import multistorageclient as msc
                open_file = msc.open
            else:
                open_file = open
            # Write data to file
            t1 = time()
            with open_file(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    write_result = _write_item(
                        *transform_list, stream, data, write_item, storage_key, **extra_kwargs
                    )
                    local_results.append(write_result)
                
                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu, f"Tensor should be on CPU, got {tensor.device}"
                    write_result = _write_item(
                        *transform_list, stream, tensor, write_item, storage_key, **extra_kwargs
                    )
                    local_results.append(write_result)
                
                # Ensure data is written to disk
                if use_msc:
                    stream.fsync()
                else:
                    os.fsync(stream.fileno())
            t2 = time()
            print(f"Worker {worker_id}: Write data to file {t2 - t1} seconds")
        return local_results

    @staticmethod
    def _transfer_bucket_slice(
        worker_id: int, preloaded_buckets: List, async_req: AsyncRequest,
        worker_rank: int, peer_rank: int, instance_seq: int, comm: torch.distributed.ProcessGroup
    ):
        """Transfer preloaded bucket data between workers using distributed communication.
        
        Args:
            worker_id: ID of the worker process
            preloaded_buckets: List of buckets with data already on CPU
            async_req: AsyncRequest containing transfer parameters
            worker_rank: Current worker's rank in the communication group
            peer_rank: Peer worker's rank to communicate with
            worker_comm: PyTorch distributed communication group
            
        Returns:
            List of transfer result descriptions
        """
        import pickle
        
        # current_rank = torch.distributed.get_rank()
        transfer_results = []
        
        logger = logging.getLogger(__name__)
        logger.info(f"Worker {worker_id}: Starting data transfer worker_rank={worker_rank}, peer_rank={peer_rank}")
        
        # 确定是发送方还是接收方（基于worker_rank）
        is_sender = (instance_seq % 2 == 0)
        
        try:
            for bucket_idx, bucket in enumerate(preloaded_buckets):
                file_name, storage_key, (bytes_data, tensor_data) = bucket
                
                t1 = time()
                
                if is_sender:
                    # 发送数据到对等rank（数据保持在CPU上）
                    logger.debug(f"Worker {worker_id}: Sending bucket {bucket_idx} to peer_rank {peer_rank}")
                    
                    # 1. 发送元数据
                    metadata = {
                        'file_name': file_name,
                        'storage_key': storage_key,
                        'num_bytes_items': len(bytes_data),
                        'num_tensor_items': len(tensor_data),
                    }
                    metadata_bytes = pickle.dumps(metadata)
                    
                    # 发送元数据长度（使用CPU tensor）
                    meta_len_tensor = torch.tensor([len(metadata_bytes)], dtype=torch.long)
                    torch.distributed.send(meta_len_tensor, dst=peer_rank, group=comm)
                    
                    # 发送元数据内容（使用CPU tensor）
                    meta_tensor = torch.tensor(list(metadata_bytes), dtype=torch.uint8)
                    torch.distributed.send(meta_tensor, dst=peer_rank, group=comm)
                    
                    # # 2. 发送bytes_data
                    # if bytes_data:
                    #     bytes_pickle = pickle.dumps(bytes_data)
                    #     bytes_len_tensor = torch.tensor([len(bytes_pickle)], dtype=torch.long)
                    #     torch.distributed.send(bytes_len_tensor, dst=peer_rank, group=worker_comm)
                        
                    #     bytes_tensor = torch.tensor(list(bytes_pickle), dtype=torch.uint8)
                    #     torch.distributed.send(bytes_tensor, dst=peer_rank, group=worker_comm)
                    # else:
                    #     # 发送0长度表示没有bytes_data
                    #     zero_len = torch.tensor([0], dtype=torch.long)
                    #     torch.distributed.send(zero_len, dst=peer_rank, group=worker_comm)
                    
                    # # 3. 发送tensor_data（数据保持在CPU上）
                    # for tensor_idx, (item, tensor) in enumerate(tensor_data):
                    #     # 发送tensor元数据
                    #     tensor_meta = {
                    #         'item': item,
                    #         'shape': list(tensor.shape),
                    #         'dtype': str(tensor.dtype)
                    #     }
                    #     tensor_meta_bytes = pickle.dumps(tensor_meta)
                        
                    #     tensor_meta_len = torch.tensor([len(tensor_meta_bytes)], dtype=torch.long)
                    #     torch.distributed.send(tensor_meta_len, dst=peer_rank, group=worker_comm)
                        
                    #     tensor_meta_tensor = torch.tensor(list(tensor_meta_bytes), dtype=torch.uint8)
                    #     torch.distributed.send(tensor_meta_tensor, dst=peer_rank, group=worker_comm)
                        
                    #     # 发送tensor数据（使用CPU tensor直接发送）
                    #     assert tensor.is_cpu, f"Tensor should be on CPU, got {tensor.device}"
                    #     torch.distributed.send(tensor.flatten(), dst=peer_rank, group=worker_comm)
                    #     logger.debug(f"Worker {worker_id}: Sent tensor {tensor_idx} with shape {tensor.shape}")
                    
                    # transfer_results.append(f"sent_bucket_{bucket_idx}")
                    
                else:
                    # 从对等rank接收数据（数据保持在CPU上）
                    logger.debug(f"Worker {worker_id}: Receiving bucket {bucket_idx} from peer_rank {peer_rank}")
                    
                    # 1. 接收元数据
                    meta_len_tensor = torch.empty(1, dtype=torch.long)
                    torch.distributed.recv(meta_len_tensor, src=peer_rank, group=comm)
                    
                    meta_tensor = torch.empty(meta_len_tensor.item(), dtype=torch.uint8)
                    torch.distributed.recv(meta_tensor, src=peer_rank, group=comm)
                    
                    metadata = pickle.loads(bytes(meta_tensor.tolist()))
                    logger.debug(f"Worker {worker_id}: Received metadata: {metadata}")
                    
                    # # 2. 接收bytes_data
                    # bytes_len_tensor = torch.empty(1, dtype=torch.long)
                    # torch.distributed.recv(bytes_len_tensor, src=worker_rank, group=worker_comm)
                    
                    # received_bytes_data = None
                    # if bytes_len_tensor.item() > 0:
                    #     bytes_tensor = torch.empty(bytes_len_tensor.item(), dtype=torch.uint8)
                    #     torch.distributed.recv(bytes_tensor, src=worker_rank, group=worker_comm)
                    #     received_bytes_data = pickle.loads(bytes(bytes_tensor.tolist()))
                    
                    # # 3. 接收tensor_data
                    # received_tensor_data = []
                    # for tensor_idx in range(metadata['num_tensor_items']):
                    #     # 接收tensor元数据
                    #     tensor_meta_len = torch.empty(1, dtype=torch.long)
                    #     torch.distributed.recv(tensor_meta_len, src=worker_rank, group=worker_comm)
                        
                    #     tensor_meta_tensor = torch.empty(tensor_meta_len.item(), dtype=torch.uint8)
                    #     torch.distributed.recv(tensor_meta_tensor, src=worker_rank, group=worker_comm)
                        
                    #     tensor_meta = pickle.loads(bytes(tensor_meta_tensor.tolist()))
                        
                    #     # 接收tensor数据（保持在CPU上）
                    #     tensor_numel = 1
                    #     for dim in tensor_meta['shape']:
                    #         tensor_numel *= dim
                        
                    #     received_tensor_flat = torch.empty(tensor_numel, dtype=getattr(torch, tensor_meta['dtype'].split('.')[-1]))
                    #     torch.distributed.recv(received_tensor_flat, src=worker_rank, group=worker_comm)
                        
                    #     # 重塑为原始形状（保持在CPU上）
                    #     received_tensor = received_tensor_flat.reshape(tensor_meta['shape'])
                    #     received_tensor_data.append((tensor_meta['item'], received_tensor))
                        
                    #     logger.debug(f"Worker {worker_id}: Received tensor {tensor_idx} with shape {tensor_meta['shape']}")
                    
                    # # 构建接收到的bucket
                    # received_bucket = (
                    #     metadata['file_name'],
                    #     metadata['storage_key'], 
                    #     (received_bytes_data or [], received_tensor_data)
                    # )
                    
                    # transfer_results.append(f"received_bucket_{bucket_idx}")
                
                t2 = time()
                logger.info(f"Worker {worker_id}: {'Sent' if is_sender else 'Received'} bucket {bucket_idx} in {t2 - t1:.3f} seconds")
        
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error in data transfer: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        logger.info(f"Worker {worker_id}: Completed data transfer with {len(transfer_results)} operations")
        return transfer_results

    def _split_write_buckets(self, write_buckets: List) -> List[List]:
        """Split write buckets among workers."""
        if not write_buckets:
            return [[] for _ in range(self.num_workers)]
        
        # Distribute buckets round-robin style
        worker_buckets = [[] for _ in range(self.num_workers)]
        for i, bucket in enumerate(write_buckets):
            worker_idx = i % self.num_workers
            worker_buckets[worker_idx].append(bucket)
        
        return worker_buckets

    @_disable_gc()
    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Schedule async call using the worker pool."""
        if async_req.async_fn is None:
            return
        
        self.call_counter += 1
        call_id = self.call_counter
        
        # Store the request for later finalization
        self.active_requests[call_id] = async_req
        
        # Extract parameters from the original async request
        # For FileSystemWriterAsync, args are: [rank, write_buckets, results_queue]
        rank, write_buckets, results_queue = async_req.async_fn_args

        # If preload_fn is provided, call it to get preloaded buckets, in pipeline mode, do not execute 
        # if async_req.preload_fn is not None:
            # write_buckets = async_req.preload_fn()
        
        # Split buckets among workers
        worker_bucket_slices = self._split_write_buckets(write_buckets)
        
        # Extract additional parameters from the original async_fn if it's write_preloaded_data_multiproc
        transform_list = []
        use_msc = False
        
        # Try to extract parameters from the original function (write_preloaded_data_multiproc)
        if hasattr(async_req.async_fn, 'func'):
            # This is a partial function, get the original args
            original_args = getattr(async_req.async_fn, 'args', ())
            if len(original_args) >= 2:
                transform_list = original_args[0] if original_args[0] else []
                use_msc = original_args[1] if len(original_args) > 1 else False
        
        # Extract use_data_transfer option from async_req
        # use_data_transfer = async_req.async_fn_kwargs.get('use_data_transfer', False)
        
        # Create a simplified async request for workers with necessary parameters
        worker_async_req = AsyncRequest(
            async_fn=None,
            async_fn_args=async_req.async_fn_args,
            finalize_fns=[],
            async_fn_kwargs={
                'transform_list': transform_list,
                'use_msc': use_msc,
                # 'use_data_transfer': use_data_transfer,
                **async_req.async_fn_kwargs
            },
            preload_fn=None,
            is_frozen=True
        )
        
        # Schedule tasks to workers with pipeline synchronization
        active_workers = 0
        for worker_idx, bucket_slice in enumerate(worker_bucket_slices):
            if bucket_slice:  # Only send non-empty slices
                task = ("pipeline_preload_and_write", call_id, (worker_async_req, bucket_slice, self.num_workers))
                self.task_queues[worker_idx].put(task)
                active_workers += 1
        
        # If no workers were assigned (empty buckets), immediately mark as completed
        if active_workers == 0:
            logger.debug(f"No active workers for call {call_id}, marking as completed")
            # Put a dummy completion result
            if len(async_req.async_fn_args) >= 3:
                rank, write_buckets, results_queue = async_req.async_fn_args
                if results_queue is not None:
                    results_queue.put({})  # Empty results
            # Mark as completed
            self.active_requests.pop(call_id, None)
        
        logger.debug(f"Scheduled pipeline call {call_id} across {active_workers} workers")

    def is_current_async_call_done(self, blocking: bool = False, no_dist: bool = False) -> bool:
        """Check if async calls are completed."""
        completed_calls = set()
        
        # Check for completed tasks and collect results
        while True:
            try:
                result = self.completion_queue.get_nowait()
                if len(result) == 4:  # New format with write_results
                    call_id, worker_id, status, write_results = result
                else:  # Legacy format without write_results
                    call_id, worker_id, status = result
                    write_results = []
                
                if status == "completed":
                    # Collect write results for this call
                    if call_id not in self.collected_results:
                        self.collected_results[call_id] = {}
                    self.collected_results[call_id][worker_id] = write_results
                    
                    # Check if all workers for this call have completed
                    if call_id in self.active_requests:
                        expected_workers = self._count_active_workers_for_call(call_id)
                        if len(self.collected_results[call_id]) >= expected_workers:
                            # All workers completed, put results in the FileSystemWriterAsync results_queue
                            self._finalize_call_results(call_id)
                            completed_calls.add(call_id)
                else:
                    logger.error(f"Worker {worker_id} failed for call {call_id}: {status}")
                    # For failed workers, still need to finalize the call to avoid hanging
                    if call_id in self.active_requests:
                        async_req = self.active_requests[call_id]
                        rank, write_buckets, results_queue = async_req.async_fn_args
                        if results_queue is not None:
                            # Put an error result to avoid hanging in retrieve_write_results
                            error_exception = RuntimeError(f"Worker {worker_id} failed: {status}")
                            results_queue.put(error_exception)
                    completed_calls.add(call_id)  # Mark as done even if failed
            except:
                break
        
        # Remove completed requests
        for call_id in completed_calls:
            if call_id in self.active_requests:
                del self.active_requests[call_id]
            if call_id in self.collected_results:
                del self.collected_results[call_id]
        
        # Check if all requests are done
        has_active_requests = len(self.active_requests) > 0
        
        if blocking and has_active_requests:
            # Wait for completion if blocking
            while self.active_requests:
                try:
                    result = self.completion_queue.get(timeout=0.1)
                    if len(result) == 4:
                        call_id, worker_id, status, write_results = result
                    else:
                        call_id, worker_id, status = result
                        write_results = []
                    
                    if call_id in self.active_requests:
                        if status == "completed":
                            if call_id not in self.collected_results:
                                self.collected_results[call_id] = {}
                            self.collected_results[call_id][worker_id] = write_results
                            
                            expected_workers = self._count_active_workers_for_call(call_id)
                            if len(self.collected_results[call_id]) >= expected_workers:
                                self._finalize_call_results(call_id)
                                del self.active_requests[call_id]
                                del self.collected_results[call_id]
                        else:
                            del self.active_requests[call_id]
                except:
                    continue
        
        # Synchronize across ranks if needed
        is_done = not has_active_requests
        if not no_dist:
            is_alive = int(has_active_requests)
            is_done = self.sync_all_async_calls(is_alive)
        
        return is_done

    def _count_active_workers_for_call(self, call_id: int) -> int:
        """Count the number of active workers for a specific call."""
        if call_id not in self.active_requests:
            return 0
        
        # Count non-empty bucket slices that were sent to workers
        async_req = self.active_requests[call_id]
        rank, write_buckets, results_queue = async_req.async_fn_args
        
        # if async_req.preload_fn is not None:
        #     write_buckets = async_req.preload_fn()
        
        worker_bucket_slices = self._split_write_buckets(write_buckets)
        return sum(1 for bucket_slice in worker_bucket_slices if bucket_slice)

    def _finalize_call_results(self, call_id: int):
        """Finalize results for a completed call by putting them in the FileSystemWriterAsync results_queue."""
        if call_id not in self.active_requests or call_id not in self.collected_results:
            return
        
        async_req = self.active_requests[call_id]
        rank, write_buckets, results_queue = async_req.async_fn_args
        
        if results_queue is not None:
            # Combine all worker results into the format expected by FileSystemWriterAsync
            # Format: {worker_id: [write_results]}
            combined_results = self.collected_results[call_id]
            
            # Put the combined results in the FileSystemWriterAsync results_queue
            results_queue.put(combined_results)
            logger.debug(f"Finalized results for call {call_id} with {len(combined_results)} workers")

    def execute_sync(self, async_req: AsyncRequest) -> None:
        """Execute async request synchronously using pipeline workers.
        
        This method provides the same pipeline benefits (parallel GPU->CPU transfer
        and pipelined execution) but waits for completion before returning.
        This allows using pipeline optimization even without --async-save.
        
        Args:
            async_req (AsyncRequest): async request to execute synchronously
        """
        if async_req.async_fn is None:
            return
        
        # Schedule the async call
        self.schedule_async_call(async_req)
        
        # Wait for completion (blocking)
        logger.debug("PipelineAsyncCaller: Waiting for synchronous completion...")
        while not self.is_current_async_call_done(blocking=True, no_dist=True):
            import time
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        # Execute finalization functions
        torch.distributed.barrier()
        for finalize_fn in async_req.finalize_fns:
            finalize_fn()
        
        logger.debug("PipelineAsyncCaller: Synchronous execution completed")

    def close(self):
        """Shutdown the worker pool."""
        logger.info(f"PipelineAsyncCaller: Shutting down {self.num_workers} workers")
        
        # Send shutdown signal to stage_sync_queue to unblock waiting workers
        if self.stage_sync_queue is not None:
            for _ in range(self.num_workers * 2):  # Send multiple signals to ensure all workers get it
                self.stage_sync_queue.put("SHUTDOWN")
        
        # Send shutdown signal to all task queues
        for task_queue in self.task_queues:
            task_queue.put(None)
        
        # Wait for workers to terminate with timeout
        for i, worker in enumerate(self.workers):
            try:
                worker.join(timeout=10.0)  # 10 second timeout
                if worker.is_alive():
                    logger.warning(f"Worker {i} did not terminate gracefully, forcing termination")
                    worker.terminate()
                    worker.join(timeout=5.0)  # Give it 5 more seconds
                    if worker.is_alive():
                        logger.error(f"Worker {i} could not be terminated, may become zombie process")
                else:
                    logger.debug(f"Worker {i} terminated successfully")
            except Exception as e:
                logger.error(f"Error terminating worker {i}: {e}")
                try:
                    worker.terminate()
                except:
                    pass
        
        # Clear all queues to ensure no hanging references
        try:
            # Drain queues to prevent blocking
            while not self.completion_queue.empty():
                self.completion_queue.get_nowait()
        except:
            pass
        
        try:
            while not self.stage_sync_queue.empty():
                self.stage_sync_queue.get_nowait()
        except:
            pass
        
        for task_queue in self.task_queues:
            try:
                while not task_queue.empty():
                    task_queue.get_nowait()
            except:
                pass
        
        self.workers.clear()
        self.task_queues.clear()
        self.result_queues.clear()
        
        logger.info("PipelineAsyncCaller: All workers shut down")

    def __del__(self):
        self.close()
