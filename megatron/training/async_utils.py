# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
This module provides a singleton instance of AsyncCallsQueue which manages
the async checkpoint save calls.
"""
import logging

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, AsyncRequest
from megatron.training import get_args
from megatron.training.utils import print_rank_0

logger = logging.getLogger(__name__)

# Singleton manager of async calls
# The default is `TemporalAsyncCaller`
_async_calls_queue = AsyncCallsQueue()


def init_persistent_async_worker():
    global _async_calls_queue
    # Recreate the async_calls_queue for persistent worker
    # This duplicate step is for backward compatiblity
    _async_calls_queue = AsyncCallsQueue(persistent=True)


def init_pipeline_async_worker(num_workers: int = 4):
    """Initialize pipeline async worker with pre-created process pool.
    
    Args:
        num_workers (int, optional): Number of worker processes to create.
                                   If None, will use a default value based on system.
    """
    global _async_calls_queue
    _async_calls_queue = AsyncCallsQueue(pipeline=True, num_workers=num_workers)


def schedule_async_save(async_request: AsyncRequest):
    """Schedule the async save request.

    Args:
        async_request (AsyncRequest): the async save request.
    """
    _async_calls_queue.schedule_async_request(async_request)


def maybe_finalize_async_save(blocking: bool = False, terminate=False):
    """Finalizes active async save calls.

    Args:
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
        terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
    """
    args = get_args()
    
    # For pipeline mode, we need to finalize even in sync mode to ensure proper cleanup
    queue_info = get_async_queue_info()
    if queue_info['queue_type'] == 'pipeline' or args.async_save:
        if blocking and not is_empty_async_queue():
            print_rank_0('Unfinalized checkpoint operations. Finalizing them synchronously now.')

        _async_calls_queue.maybe_finalize_async_calls(blocking, no_dist=False)

        if terminate:
            print_rank_0('Closing async checkpoint workers...')
            _async_calls_queue.close()
            print_rank_0('Async checkpoint workers closed.')


def is_empty_async_queue() -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks.

    Returns:
        bool: True if there is any ongoing async call.
    """
    return _async_calls_queue.get_num_unfinalized_calls() == 0


def get_async_calls_queue():
    """Get the configured AsyncCallsQueue instance.
    
    Returns:
        AsyncCallsQueue: The currently configured async calls queue instance
    """
    global _async_calls_queue
    return _async_calls_queue


def get_async_queue_info() -> dict:
    """Get information about the current async queue configuration.
    
    Returns:
        dict: Information about async queue configuration including:
            - queue_type: 'temporal', 'persistent', or 'pipeline'
            - num_workers: number of workers (for pipeline mode)
            - active_calls: number of active async calls
    """
    global _async_calls_queue
    
    queue_type = 'temporal'  # default
    num_workers = None
    
    if _async_calls_queue.persistent:
        queue_type = 'persistent'
    elif _async_calls_queue.pipeline:
        queue_type = 'pipeline'
        num_workers = _async_calls_queue.num_workers
    
    return {
        'queue_type': queue_type,
        'num_workers': num_workers,
        'active_calls': _async_calls_queue.get_num_unfinalized_calls()
    }


# Usage example for pipeline async worker:
# 
# To enable pipeline async checkpointing in your training script:
# 
# 1. Initialize the pipeline worker early in training:
#    from megatron.training.async_utils import init_pipeline_async_worker
#    init_pipeline_async_worker(num_workers=4)  # Use 4 worker processes
# 
# 2. The rest of the checkpointing code remains the same:
#    - schedule_async_save() will automatically use the pipeline workers
#    - maybe_finalize_async_save() will handle completion checking
# 
# Benefits of pipeline mode:
# - No process creation overhead during training (workers are pre-created)
# - Parallel GPU-to-CPU transfer across multiple workers
# - Pipeline execution: while one worker writes to disk, others can do GPU-to-CPU transfer
# - Better resource utilization and potentially faster checkpointing
#
# Configuration recommendations:
# - num_workers should typically be 2-8 depending on your I/O bandwidth and GPU memory
# - More workers help with parallel data transfer but may compete for I/O resources
# - Monitor memory usage as each worker may hold copied tensor data temporarily
