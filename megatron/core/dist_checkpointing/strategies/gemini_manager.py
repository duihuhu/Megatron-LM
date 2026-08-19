# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Gemini manager for replica-level data transfer with C++ ASIO or RDMA implementation."""

import os
import queue
import threading
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch

from .state_dict_decomposer import DecomposedStateDict, TensorInfo, assign_tensor_offsets
from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip

logger = getLogger(__name__)


class GeminiManager:
    """Shared manager for Gemini replica-level data transfer.
    
    This class provides a singleton instance that manages:
    - Gemini C++ native module (_gemini_native) for ASIO or RDMA communication
    - Buffer allocation and management for replica data exchange
    - Decomposed state dict for efficient GPU-to-CPU transfer
    - Buffer registration for RDMA (when use_rdma is enabled)
    
    Similar to ECCHECKManager, but focused on replica-level data exchange
    between paired ranks (0<->2, 1<->3) using ASIO or RDMA for network communication.
    """
    
    _instance: Optional['GeminiManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the manager (only once due to singleton)."""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._gemini_native = None
        self.use_gemini = False
        self.use_gemini_optimized = False
        self.use_rdma = False
        
        # Buffer configuration
        self.gemini_pin_memory = True
        self.preallocated_cpu_buffer: Optional[torch.Tensor] = None
        
        # Decomposed state dict for efficient transfer
        self.decomposed_state_dict: Optional[DecomposedStateDict] = None
        
        # Replica data buffers (for received data from peer)
        self.replica_buffer: Optional[torch.Tensor] = None
        self.replica_metadata: Optional[dict] = None
        
        # Track registered buffers (for RDMA)
        self.registered_buffers: Dict[int, Tuple[int, int]] = {}  # {buffer_addr: (size, iteration)}
        self.current_iteration: int = 0
        
        self._initialized = True
    
    RANKS_PER_GROUP = 4

    def _get_gemini_paired_rank(self, my_rank: int, world_size: int) -> int:
        """Get the paired rank for Gemini replica exchange.

        Pairing rules (same as EC grouping: one group spans different nodes).
        Within each 4-rank group: rank_in_group 0<->2, 1<->3.
        Formula: group_id = rank % num_groups, rank_in_group = rank // num_groups,
        paired_rank_in_group = (rank_in_group + 2) % 4, paired_rank = group_id + num_groups * paired_rank_in_group.

        Args:
            my_rank (int): Current rank
            world_size (int): Total number of ranks

        Returns:
            int: Paired rank for replica exchange
        """
        if world_size < 4:
            raise ValueError(f"Gemini: World size must be at least 4, got {world_size}")
        if world_size % self.RANKS_PER_GROUP != 0:
            raise ValueError(
                f"Gemini: World size must be divisible by {self.RANKS_PER_GROUP}, got {world_size}"
            )
        num_groups = world_size // self.RANKS_PER_GROUP
        group_id = my_rank % num_groups
        rank_in_group = my_rank // num_groups
        paired_rank_in_group = (rank_in_group + 2) % self.RANKS_PER_GROUP
        paired_rank = group_id + num_groups * paired_rank_in_group
        logger.debug(
            f"Gemini: Rank {my_rank} (group_id={group_id}, rank_in_group={rank_in_group}) "
            f"paired with Rank {paired_rank}"
        )
        return paired_rank
    
    def _get_gemini_network_config(self, rank: int, world_size: int) -> dict:
        """
        Get network configuration for Gemini ASIO connections.
        
        Similar to EC-CHECK, but only needs one connection per rank (to paired rank).
        
        Args:
            rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            dict: Network configuration with keys:
                - 'my_ip': str - This rank's IP address
                - 'base_port': int - Base port number
                - 'partner_ip': str - Partner's IP address
                - 'ports': dict - Port numbers
                    - 'send': int - Port for sending data
                    - 'recv': int - Port for receiving data
        """
        # Step 1: Get base IP address (with multi-NIC per-rank support)
        base_ip = resolve_ip("GEMINI", rank=rank)
        
        # Step 2: Get base port
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('GEMINI_BASE_PORT', master_port + 20000))
        
        # Step 3: Calculate ports for this rank
        # Port allocation: base_port + rank * 2 + offset
        # offset: 0=send, 1=recv
        ports = {
            'send': base_port + rank * 2 + 0,
            'recv': base_port + rank * 2 + 1,
        }
        
        # Step 4: Get partner rank
        partner_rank = self._get_gemini_paired_rank(rank, world_size)
        
        # Step 5: Exchange IP addresses via torch.distributed.all_gather
        rank_ips = {}
        
        if torch.distributed.is_initialized():
            try:
                # Use all_gather_object for IP exchange (works with any backend)
                # This is more reliable than tensor-based all_gather for string data
                ip_list = [None] * world_size
                torch.distributed.all_gather_object(ip_list, base_ip)
                
                # Convert list to dict
                for r, ip in enumerate(ip_list):
                    rank_ips[r] = ip
                
                logger.info(
                    f"Gemini: [Rank {rank}] IP exchange completed - "
                    f"All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"Gemini: Failed to exchange IPs via all_gather, using local IP: {e}"
                )
                # Fallback to using local IP for all ranks
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            # Single rank mode - use local IP
            logger.info("Gemini: Distributed not initialized, using local IP for all ranks")
            rank_ips[rank] = base_ip
        
        # Get partner IP from rank_ips
        partner_ip = rank_ips.get(partner_rank, base_ip)
        
        config = {
            'my_ip': base_ip,
            'base_port': base_port,
            'partner_ip': partner_ip,
            'rank_ips': rank_ips,
            'ports': ports,
        }
        
        logger.info(
            f"Gemini: [Rank {rank}] Network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  Partner IP: {config['partner_ip']}\n"
            f"  Ports: {config['ports']}\n"
            f"  All rank IPs: {config['rank_ips']}"
        )
        
        return config
    
    def init_gemini_if_enabled(self):
        """Initialize Gemini C++ module if enabled and distributed environment is ready."""
        if self._gemini_native is not None:
            logger.debug("Gemini: Already initialized, skipping")
            return
        
        try:
            from megatron.training import get_args as input_args
            args = input_args()
            self.use_gemini = getattr(args, 'use_gemini', False)
            self.use_gemini_optimized = getattr(args, 'use_gemini_optimized', False)
            self.use_rdma = getattr(args, 'use_rdma', False)
            
            if not self.use_gemini or not self.use_gemini_optimized:
                return
            
            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning("Gemini: Distributed environment not initialized, skipping initialization")
                return
            
            # Initialize Gemini C++ module
            self._init_gemini_native()
            
        except Exception as e:
            logger.warning(f"Gemini: Failed to initialize: {e}")
            self._gemini_native = None
    
    def _init_gemini_native(self):
        """Initialize Gemini C++ native module with ASIO or RDMA."""
        gemini_native = None
        try:
            # Load .so file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            import glob as _glob_module
            so_files = _glob_module.glob(os.path.join(current_dir, "gemini_native*.so"))
            
            if not so_files:
                logger.warning(f"Gemini: No gemini_native.so file found in {current_dir}, will use fallback")
                return
            
            # Load .so file directly
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("gemini_native", so_path)
            gemini_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(gemini_native)
            logger.debug(f"Gemini: Loaded .so file from {so_path}")
            
            # Check RDMA availability if RDMA mode is requested (strict mode: fail if not available)
            if self.use_rdma:
                try:
                    rdma_available = gemini_native.is_rdma_available()
                    if not rdma_available:
                        raise RuntimeError(
                            "RDMA mode requested (--use-rdma) but RDMA is not available on this system.\n"
                            "Possible causes:\n"
                            "  1. No RDMA devices installed (check: ibv_devices)\n"
                            "  2. RDMA drivers not loaded (try: modprobe rdma_cm ib_uverbs)\n"
                            "  3. Insufficient permissions\n"
                            "  4. RDMA services not running (check: systemctl status rdma)\n"
                            "\n"
                            "To use standard TCP/IP networking instead, remove --use-rdma from your training script."
                        )
                except RuntimeError:
                    raise  # Re-raise the RuntimeError we just created
                except Exception as check_err:
                    logger.warning(f"Gemini: Could not check RDMA availability: {check_err}")
                    logger.warning("Gemini: Will attempt to initialize RDMA anyway...")
            
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            partner_rank = self._get_gemini_paired_rank(rank, world_size)
            
            # Get network configuration
            net_config = self._get_gemini_network_config(rank, world_size)
            
            # Synchronize all ranks before creating C++ instances
            transport_mode = "RDMA" if self.use_rdma else "ASIO"
            logger.info(f"Gemini: [Rank {rank}] Synchronizing all ranks before creating C++ native module ({transport_mode})...")
            torch.distributed.barrier()
            logger.info(f"Gemini: [Rank {rank}] All ranks synchronized, creating C++ native module with {transport_mode}...")
            
            # Calculate partner's recv port (where we send to)
            base_port = net_config['base_port']
            partner_recv_port = base_port + partner_rank * 2 + 1  # partner's recv port
            
            # Create C++ instance (Phase 1: start listener only)
            logger.info(f"Gemini: Creating C++ native module with {transport_mode} (Phase 1: listener)...")
            print(f"Gemini: [Rank {rank}] Creating C++ native module with {transport_mode} (Phase 1: starting listener)...")
            
            self._gemini_native = gemini_native.GeminiNative(
                rank, world_size, partner_rank,
                # Connection: (partner_ip, partner_recv_port, my_ip, my_recv_port, use_rdma)
                net_config['partner_ip'], partner_recv_port,
                net_config['my_ip'], net_config['ports']['recv'],
                self.use_rdma
            )
            
            logger.info(f"Gemini: C++ native module created (listener ready) for rank {rank}")
            print(f"Gemini: [Rank {rank}] Listener ready, waiting for all ranks...")
            
            # Synchronize all ranks before connecting (Phase 2)
            torch.distributed.barrier()
            logger.info(f"Gemini: [Rank {rank}] All ranks ready, starting Phase 2 (connecting)...")
            print(f"Gemini: [Rank {rank}] Phase 2: Connecting to partner rank {partner_rank} via {transport_mode}...")
            
            # Phase 2: Connect to partner
            self._gemini_native.finalize_connections()
            
            logger.info(f"Gemini: C++ native module fully initialized (rank={rank}, partner_rank={partner_rank}, transport={transport_mode})")
            print(f"Gemini: [Rank {rank}] C++ native module fully initialized - {transport_mode} connection ready")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini: Failed to initialize C++ native module: {error_msg}")
            
            # Provide helpful guidance for RDMA-specific errors
            if "RDMA" in error_msg and "event channel" in error_msg:
                logger.error("")
                logger.error("=" * 80)
                logger.error("RDMA INITIALIZATION FAILED")
                logger.error("=" * 80)
                logger.error("Diagnostic steps:")
                logger.error("  1. Check RDMA devices: ibv_devices")
                logger.error("  2. Check RDMA links: rdma link")
                logger.error("  3. Check kernel modules: lsmod | grep -E '(rdma|ib_)'")
                logger.error("  4. Load modules if needed: modprobe rdma_cm ib_uverbs ib_core")
                logger.error("  5. Check RDMA service: systemctl status rdma")
                logger.error("=" * 80)
            
            import traceback
            traceback.print_exc()
            self._gemini_native = None
            raise  # Re-raise the exception to fail fast
    
    def prepare_decomposed_state_dict(self, plan, planner):
        """
        Prepare decomposed state dict from SavePlan for efficient GPU-to-CPU transfer.
        
        Similar to EC-CHECK's _prepare_eccheck_data, this method:
        1. Extracts tensor data from plan items
        2. Creates TensorInfo list with metadata
        3. Stores in decomposed_state_dict for efficient iteration
        
        Args:
            plan: SavePlan from PyTorch distributed checkpoint
            planner: SavePlanner instance
        """
        from time import time
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        start = time()
        
        logger.info(f"Gemini: [Rank {rank}] Preparing decomposed state dict from plan...")
        
        # Extract tensor data from plan
        non_tensor_data = {}
        tensor_infos = []
        tensor_data_list = []
        
        byte_io_count = 0
        tensor_count = 0
        none_data_count = 0
        
        for item in plan.items:
            # Resolve data using planner
            data = planner.resolve_data(item)
            
            if data is None:
                none_data_count += 1
                continue
            
            if item.type == 1:  # WriteItemType.BYTE_IO
                # Store non-tensor data
                # Convert BytesIO to bytes for proper serialization
                if hasattr(data, 'getvalue'):  # BytesIO object
                    # Extract bytes content from BytesIO
                    data.seek(0)
                    data_bytes = data.getvalue()
                    non_tensor_data[item.index.fqn] = data_bytes
                else:
                    non_tensor_data[item.index.fqn] = data
                byte_io_count += 1
            else:
                # Tensor data - create TensorInfo
                tensor_info = TensorInfo(
                    key=item.index.fqn,
                    shape=tuple(data.shape),
                    dtype=data.dtype,
                    device=data.device,
                    numel=data.numel(),
                    size_bytes=data.numel() * data.element_size(),
                    offset=0,  # Will be calculated below
                    global_offset=tuple(item.index.offset) if hasattr(item.index, 'offset') else None,
                    shard_index=item.index.index if hasattr(item.index, 'index') else None,
                )
                tensor_infos.append(tensor_info)
                tensor_data_list.append(data)
                tensor_count += 1
        
        logger.info(
            f"Gemini: [Rank {rank}] Processed {byte_io_count} BytesIO items, {tensor_count} tensor items"
            + (f", skipped {none_data_count} None items" if none_data_count > 0 else "")
        )
        
        # Calculate aligned offsets for tensor data.
        total_tensor_size_bytes = assign_tensor_offsets(tensor_infos)
        
        # Create decomposed structure
        self.decomposed_state_dict = DecomposedStateDict(
            non_tensor_data=non_tensor_data,
            tensor_infos=tensor_infos,
            tensor_data=tensor_data_list,
            total_tensor_size_bytes=total_tensor_size_bytes,
        )
        
        process_time = time() - start
        
        # Log statistics
        stats = self.decomposed_state_dict.get_statistics()
        logger.info(
            f"Gemini: [Rank {rank}] Prepared decomposed state dict in {process_time:.2f}s\n"
            f"  Non-tensor data: {stats['non_tensor_size_bytes'] / 1024:.2f} KB\n"
            f"  Tensor data: {stats['tensor_data_size_bytes'] / (1024**3):.2f} GB\n"
            f"  Total tensors: {stats['num_tensors']}"
        )
    
    def allocate_preallocated_buffer(self, size_bytes: int):
        """Allocate preallocated CPU buffer for data transfer.
        
        Args:
            size_bytes: Size of buffer to allocate in bytes
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        if self.preallocated_cpu_buffer is not None:
            if self.preallocated_cpu_buffer.numel() >= size_bytes:
                logger.info(f"Gemini: [Rank {rank}] Reusing existing preallocated buffer")
                return
        
        logger.info(f"Gemini: [Rank {rank}] Allocating preallocated buffer: {size_bytes / (1024**3):.2f} GB")
        
        if self.gemini_pin_memory and torch.cuda.is_available():
            self.preallocated_cpu_buffer = torch.empty(size_bytes, dtype=torch.uint8).pin_memory()
            logger.info(f"Gemini: [Rank {rank}] Allocated pinned memory buffer")
        else:
            self.preallocated_cpu_buffer = torch.empty(size_bytes, dtype=torch.uint8)
            logger.info(f"Gemini: [Rank {rank}] Allocated regular CPU buffer")
        
        # Register buffer for RDMA if enabled
        if self.use_rdma and self._gemini_native is not None:
            self.register_buffer(self.preallocated_cpu_buffer)
    
    def register_buffer(self, buffer: torch.Tensor):
        """Register buffer for RDMA operations (called on first allocation in save phase).
        
        Args:
            buffer: PyTorch tensor to register
        """
        if not self.use_rdma or self._gemini_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        buffer_size = buffer.numel() * buffer.element_size()
        
        # Check if already registered
        if buffer_addr in self.registered_buffers:
            logger.debug(f"Gemini: [Rank {rank}] Buffer already registered at 0x{buffer_addr:x} (size: {buffer_size / (1024**2):.2f} MB)")
            return
        
        try:
            logger.info(f"Gemini: [Rank {rank}] Registering buffer at 0x{buffer_addr:x}, size: {buffer_size / (1024**3):.2f} GB, numel: {buffer.numel()}, dtype: {buffer.dtype} (iteration {self.current_iteration})")
            self._gemini_native.register_buffer(buffer_addr, buffer_size)
            self.registered_buffers[buffer_addr] = (buffer_size, self.current_iteration)
            logger.info(f"Gemini: [Rank {rank}] Buffer registered successfully (total registered: {len(self.registered_buffers)})")
            
            # Print all registered buffers
            logger.info(f"Gemini: [Rank {rank}] All registered buffers:")
            # for addr, (size, iteration) in self.registered_buffers.items():
                # logger.info(f"  - 0x{addr:x}: {size / (1024**2):.2f} MB (iteration {iteration})")
        except Exception as e:
            logger.error(f"Gemini: [Rank {rank}] Failed to register buffer: {e}")
            raise
    
    def unregister_buffer(self, buffer: torch.Tensor):
        """Unregister buffer for RDMA operations.
        
        Args:
            buffer: PyTorch tensor to unregister
        """
        if not self.use_rdma or self._gemini_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        
        if buffer_addr not in self.registered_buffers:
            logger.debug(f"Gemini: [Rank {rank}] Buffer not registered at 0x{buffer_addr:x}")
            return
        
        try:
            logger.info(f"Gemini: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x}")
            self._gemini_native.unregister_buffer(buffer_addr)
            del self.registered_buffers[buffer_addr]
            logger.info(f"Gemini: [Rank {rank}] Buffer unregistered successfully")
        except Exception as e:
            logger.error(f"Gemini: [Rank {rank}] Failed to unregister buffer: {e}")
    
    def get_native_module(self):
        """Get the C++ native module instance."""
        return self._gemini_native
    
    def is_initialized(self) -> bool:
        """Check if Gemini native module is initialized."""
        return self._gemini_native is not None
    
    def cleanup(self):
        """Cleanup resources."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Unregister all buffers for RDMA
        if self.use_rdma and self._gemini_native is not None:
            for buffer_addr in list(self.registered_buffers.keys()):
                try:
                    logger.info(f"Gemini: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x} during cleanup")
                    self._gemini_native.unregister_buffer(buffer_addr)
                except Exception as e:
                    logger.warning(f"Gemini: [Rank {rank}] Failed to unregister buffer during cleanup: {e}")
            self.registered_buffers.clear()
        
        if self._gemini_native is not None:
            logger.info("Gemini: Cleaning up native module")
            self._gemini_native = None
        
        self.preallocated_cpu_buffer = None
        self.decomposed_state_dict = None
        self.replica_buffer = None
        self.replica_metadata = None

