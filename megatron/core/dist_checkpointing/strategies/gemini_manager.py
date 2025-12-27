# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Gemini manager for replica-level data transfer with C++ ASIO implementation."""

import os
import queue
import threading
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch

from .state_dict_decomposer import DecomposedStateDict, TensorInfo

logger = getLogger(__name__)


class GeminiManager:
    """Shared manager for Gemini replica-level data transfer.
    
    This class provides a singleton instance that manages:
    - Gemini C++ native module (_gemini_native) for ASIO-based communication
    - Buffer allocation and management for replica data exchange
    - Decomposed state dict for efficient GPU-to-CPU transfer
    
    Similar to ECCHECKManager, but focused on replica-level data exchange
    between paired ranks (0<->2, 1<->3) using ASIO for network communication.
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
        
        # Buffer configuration
        self.gemini_pin_memory = True
        self.preallocated_cpu_buffer: Optional[torch.Tensor] = None
        
        # Decomposed state dict for efficient transfer
        self.decomposed_state_dict: Optional[DecomposedStateDict] = None
        
        # Replica data buffers (for received data from peer)
        self.replica_buffer: Optional[torch.Tensor] = None
        self.replica_metadata: Optional[dict] = None
        
        self._initialized = True
    
    def _get_gemini_paired_rank(self, my_rank: int, world_size: int) -> int:
        """Get the paired rank for Gemini replica exchange.
        
        Pairing rules (same as EC-CHECK XOR pairing):
        - Rank 0 ↔ Rank 2
        - Rank 1 ↔ Rank 3
        
        Args:
            my_rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            int: Paired rank for replica exchange
        """
        if world_size < 4:
            raise ValueError(f"Gemini: World size must be at least 4, got {world_size}")
        
        pairing_map = {0: 2, 2: 0, 1: 3, 3: 1}
        
        if my_rank in pairing_map:
            paired_rank = pairing_map[my_rank]
            logger.debug(f"Gemini: Rank {my_rank} paired with Rank {paired_rank}")
            return paired_rank
        else:
            raise ValueError(f"Gemini: Unsupported rank {my_rank} for 4-rank setup")
    
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
        import socket
        
        # Step 1: Get base IP address
        # Priority: GEMINI_BASE_IP > auto-detect > MASTER_ADDR (fallback)
        base_ip = os.environ.get('GEMINI_BASE_IP')
        
        # If GEMINI_BASE_IP is not set, try to auto-detect actual IP
        if not base_ip:
            try:
                # Get IP of the interface used for distributed training
                # Connect to a remote address (doesn't actually send data)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # Use a public DNS server IP to determine the default route interface
                s.connect(('8.8.8.8', 80))
                base_ip = s.getsockname()[0]
                s.close()
                logger.info(f"Gemini: Auto-detected IP address: {base_ip}")
            except Exception as e:
                logger.warning(f"Gemini: Failed to auto-detect IP: {e}")
                # Fallback to MASTER_ADDR or localhost
                base_ip = os.environ.get('MASTER_ADDR', '127.0.0.1')
                logger.warning(f"Gemini: Using fallback IP: {base_ip}")
        
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
        """Initialize Gemini C++ native module with ASIO."""
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
            
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            partner_rank = self._get_gemini_paired_rank(rank, world_size)
            
            # Get network configuration
            net_config = self._get_gemini_network_config(rank, world_size)
            
            # Synchronize all ranks before creating C++ instances
            logger.info(f"Gemini: [Rank {rank}] Synchronizing all ranks before creating C++ native module...")
            torch.distributed.barrier()
            logger.info(f"Gemini: [Rank {rank}] All ranks synchronized, creating C++ native module with ASIO...")
            
            # Calculate partner's recv port (where we send to)
            base_port = net_config['base_port']
            partner_recv_port = base_port + partner_rank * 2 + 1  # partner's recv port
            
            # Create C++ instance with ASIO parameters (Phase 1: start acceptor only)
            logger.info(f"Gemini: Creating C++ native module with ASIO (Phase 1: acceptor)...")
            print(f"Gemini: [Rank {rank}] Creating C++ native module (Phase 1: starting acceptor)...")
            
            self._gemini_native = gemini_native.GeminiNative(
                rank, world_size, partner_rank,
                # Connection: (partner_ip, partner_recv_port, my_ip, my_recv_port)
                net_config['partner_ip'], partner_recv_port,
                net_config['my_ip'], net_config['ports']['recv']
            )
            
            logger.info(f"Gemini: C++ native module created (acceptor ready) for rank {rank}")
            print(f"Gemini: [Rank {rank}] Acceptor ready, waiting for all ranks...")
            
            # Synchronize all ranks before connecting (Phase 2)
            torch.distributed.barrier()
            logger.info(f"Gemini: [Rank {rank}] All ranks ready, starting Phase 2 (connecting)...")
            print(f"Gemini: [Rank {rank}] Phase 2: Connecting to partner rank {partner_rank}...")
            
            # Phase 2: Connect to partner
            self._gemini_native.finalize_connections()
            
            logger.info(f"Gemini: C++ native module fully initialized (rank={rank}, partner_rank={partner_rank})")
            print(f"Gemini: [Rank {rank}] C++ native module fully initialized - ASIO connection ready")
            
        except Exception as e:
            logger.error(f"Gemini: Failed to initialize C++ native module: {e}")
            import traceback
            traceback.print_exc()
            self._gemini_native = None
    
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
    
    def get_native_module(self):
        """Get the C++ native module instance."""
        return self._gemini_native
    
    def is_initialized(self) -> bool:
        """Check if Gemini native module is initialized."""
        return self._gemini_native is not None
    
    def cleanup(self):
        """Cleanup resources."""
        if self._gemini_native is not None:
            logger.info("Gemini: Cleaning up native module")
            self._gemini_native = None
        
        self.preallocated_cpu_buffer = None
        self.decomposed_state_dict = None
        self.replica_buffer = None
        self.replica_metadata = None

