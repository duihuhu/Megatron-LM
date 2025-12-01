#!/usr/bin/env python3
"""
Standalone test script for EC-CHECK load pipeline.

This script tests the load pipeline for rank2 recovery scenario without requiring
actual checkpoint files. It creates mock data and directly calls the pipeline.

Usage:
    # Using torchrun (recommended)
    torchrun --nproc_per_node=4 test_eccheck_load_pipeline.py
    
    # Using mpirun
    mpirun -np 4 python test_eccheck_load_pipeline.py
    
    # Set environment variables for ASIO mode (optional)
    export ECCHECK_USE_ASIO=true
    torchrun --nproc_per_node=4 test_eccheck_load_pipeline.py
"""

import os
import sys
import torch
import torch.distributed as dist
import logging

# Setup environment variables similar to test_eccheck.sh
os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth0')
os.environ.setdefault('GLOO_SOCKET_IFNAME', 'eth0')
os.environ.setdefault('ECCHECK_USE_ASIO', 'true')

# Setup simple logging without rank in format string
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def init_distributed():
    """Initialize distributed environment for testing."""
    # Get rank and world_size from environment (set by launcher)
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 4))
    # Use same defaults as test_eccheck.sh
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = int(os.environ.get('MASTER_PORT', '6000'))
    
    # Initialize process group
    if torch.cuda.is_available():
        backend = 'nccl'
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        torch.cuda.set_device(device)
    else:
        backend = 'gloo'
        device = torch.device('cpu')
    
    # Use the same init method as test_eccheck.sh
    init_method = f'tcp://{master_addr}:{master_port}'
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=init_method
    )
    
    logger.info(f"[Rank {rank}/{world_size}] Initialized (backend={backend}, device={device}, master={master_addr}:{master_port})")
    return rank, world_size, device


def setup_args():
    """Setup mock args for EC-CHECK."""
    try:
        from megatron.training.arguments import parse_args
        
        # Create minimal args for testing
        # We'll modify sys.argv temporarily to avoid errors
        original_argv = sys.argv
        sys.argv = ['test_eccheck_load_pipeline.py', '--use-eccheck']
        
        try:
            args = parse_args(ignore_unknown_args=True)
            args.use_eccheck = True
            logger.info("EC-CHECK TEST: Args setup complete (use_eccheck=True)")
            return args
        finally:
            sys.argv = original_argv
    except Exception as e:
        logger.warning(f"EC-CHECK TEST: Could not setup args via parse_args: {e}")
        logger.warning("  Creating minimal args object...")
        
        # Create minimal args object
        class MockArgs:
            use_eccheck = True
        
        return MockArgs()


def main():
    """Main test function."""
    rank = 0
    world_size = 4
    
    try:
        # === Step 1: Initialize distributed environment ===
        rank, world_size, device = init_distributed()
        
        # === Step 2: Setup args ===
        args = setup_args()
        
        # Store args in global vars if needed
        try:
            from megatron.training.global_vars import set_args
            set_args(args)
            logger.info("EC-CHECK TEST: Args stored in global_vars")
        except Exception as e:
            logger.warning(f"EC-CHECK TEST: Could not set global args: {e}")
        
        # === Step 3: Synchronize before creating strategy ===
        logger.info(f"[Rank {rank}] Synchronizing all ranks before creating strategy...")
        dist.barrier()
        
        # === Step 4: Create load strategy ===
        logger.info(f"[Rank {rank}] Creating TorchDistLoadShardedStrategy...")
        from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy
        strategy = TorchDistLoadShardedStrategy()
        
        # === Step 5: Run test ===
        logger.info(f"[Rank {rank}] Starting load pipeline test...")
        test_data_size = 64 * 1024 * 1024  # 64MB
        
        success = strategy.test_load_pipeline(test_data_size=test_data_size)
        
        if success:
            logger.info(f"[Rank {rank}] Test completed successfully!")
        else:
            logger.error(f"[Rank {rank}] Test failed!")
            sys.exit(1)
        
        # === Step 6: Synchronize and exit ===
        dist.barrier()
        logger.info(f"[Rank {rank}] All ranks completed, test passed!")
        
    except Exception as e:
        logger.error(f"[Rank {rank}] Test failed with error: {e}", exc_info=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)
    finally:
        # Cleanup
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass


if __name__ == '__main__':
    main()

