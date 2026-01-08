# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Gemini Replicas manager for multi-replica data transfer with C++ ASIO implementation."""

import os
import queue
import threading
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch

from .state_dict_decomposer import DecomposedStateDict, TensorInfo

logger = getLogger(__name__)


class GeminiReplicasManager:
    """Shared manager for Gemini Replicas multi-replica data transfer.
    
    This class provides a singleton instance that manages:
    - Gemini Replicas C++ native module (_gemini_replicas_native) for ASIO-based communication
    - Buffer allocation and management for multi-replica data exchange
    - Decomposed state dict for efficient GPU-to-CPU transfer
    
    Unlike Gemini (2 replicas), this supports configurable number of replicas (default: 3)
    with round-robin placement strategy:
    - 3 replicas, 4 ranks: rank0 -> [0,1,2], rank1 -> [1,2,3], rank2 -> [2,3,0], rank3 -> [3,0,1]
    """
    
    _instance: Optional['GeminiReplicasManager'] = None
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
        
        self._gemini_replicas_native = None
        self.use_gemini_replicas = False
        self.use_gemini_replicas_optimized = False
        
        # Replica configuration
        self.num_replicas = 3  # Default: 3 replicas (including local)
        
        # Buffer configuration
        self.gemini_replicas_pin_memory = True
        self.preallocated_cpu_buffer: Optional[torch.Tensor] = None
        
        # Decomposed state dict for efficient transfer
        self.decomposed_state_dict: Optional[DecomposedStateDict] = None
        
        # Target ranks for replicas (calculated based on round-robin)
        self.target_ranks: List[int] = []
        
        # Replica data buffers (for received data from peers)
        self.replica_buffers: List[torch.Tensor] = []
        self.replica_metadata: List[dict] = []
        
        self._initialized = True
    
    def _calculate_target_ranks(self, my_rank: int, world_size: int) -> List[int]:
        """Calculate target ranks for replicas using round-robin strategy.
        
        Round-robin placement:
        - rank0 (3 replicas): [0, 1, 2]
        - rank1 (3 replicas): [1, 2, 3]
        - rank2 (3 replicas): [2, 3, 0]
        - rank3 (3 replicas): [3, 0, 1]
        
        Args:
            my_rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            List[int]: List of target ranks (including self)
        """
        if self.num_replicas > world_size:
            raise ValueError(
                f"Gemini Replicas: num_replicas ({self.num_replicas}) cannot exceed "
                f"world_size ({world_size})"
            )
        
        # Generate target ranks using round-robin
        # First replica is always self, then (num_replicas - 1) subsequent ranks
        targets = []
        for i in range(self.num_replicas):
            target_rank = (my_rank + i) % world_size
            targets.append(target_rank)
        
        logger.info(
            f"Gemini Replicas: [Rank {my_rank}] Calculated target ranks: {targets} "
            f"({self.num_replicas} replicas)"
        )
        
        return targets
    
    def _get_gemini_replicas_network_config(self, rank: int, world_size: int) -> dict:
        """
        Get network configuration for Gemini Replicas ASIO connections.
        
        Each rank needs to:
        1. Send data to (num_replicas - 1) target ranks
        2. Receive data from ranks that have this rank as target
        
        Port allocation strategy (for single-machine multi-GPU):
        - Each rank allocates unique ports for each connection pair
        - Port for rank_i -> rank_j: base_port + rank_i * world_size + rank_j
        - This ensures no port conflicts on the same machine
        
        Args:
            rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            dict: Network configuration with keys:
                - 'my_ip': str - This rank's IP address
                - 'base_port': int - Base port number
                - 'target_ranks': List[int] - Ranks to send data to (excluding self)
                - 'source_ranks': List[int] - Ranks to receive data from
                - 'rank_ips': Dict[int, str] - IP addresses of all ranks
                - 'target_ips': List[str] - IP addresses for each target rank
                - 'target_ports': List[int] - Ports for sending to each target rank
                - 'recv_ports': Dict[int, int] - Ports for receiving from each source rank
        """
        import socket
        
        # Step 1: Get base IP address
        base_ip = os.environ.get('GEMINI_REPLICAS_BASE_IP')
        
        if not base_ip:
            # Check if specific network interface is requested
            interface_name = os.environ.get('GEMINI_REPLICAS_INTERFACE')
            
            if interface_name:
                # Try to get IP from specific interface using netifaces
                try:
                    import netifaces
                    addrs = netifaces.ifaddresses(interface_name)
                    if netifaces.AF_INET in addrs:
                        base_ip = addrs[netifaces.AF_INET][0]['addr']
                        logger.info(f"Gemini Replicas: Using IP from interface {interface_name}: {base_ip}")
                    else:
                        logger.warning(f"Gemini Replicas: Interface {interface_name} has no IPv4 address")
                        base_ip = None
                except ImportError:
                    logger.warning(
                        "Gemini Replicas: netifaces module not installed. "
                        "Install via 'pip install netifaces' to use GEMINI_REPLICAS_INTERFACE. "
                        "Falling back to auto-detection."
                    )
                    base_ip = None
                except Exception as e:
                    logger.warning(f"Gemini Replicas: Failed to get IP from interface {interface_name}: {e}")
                    base_ip = None
            
            if not base_ip:
                # Fallback to auto-detection
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(('8.8.8.8', 80))
                    base_ip = s.getsockname()[0]
                    s.close()
                    logger.info(f"Gemini Replicas: Auto-detected IP address: {base_ip}")
                except Exception as e:
                    logger.warning(f"Gemini Replicas: Failed to auto-detect IP: {e}")
                    base_ip = os.environ.get('MASTER_ADDR', '127.0.0.1')
                    logger.warning(f"Gemini Replicas: Using fallback IP: {base_ip}")
        
        # Step 2: Get base port
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('GEMINI_REPLICAS_BASE_PORT', master_port + 30000))
        
        # Step 3: Calculate target and source ranks
        target_ranks_all = self._calculate_target_ranks(rank, world_size)
        # Remove self from targets (we don't send to ourselves)
        target_ranks = [r for r in target_ranks_all if r != rank]
        
        # Calculate source ranks (ranks that have this rank as target)
        source_ranks = []
        for src_rank in range(world_size):
            if src_rank == rank:
                continue
            src_targets = self._calculate_target_ranks(src_rank, world_size)
            if rank in src_targets:
                source_ranks.append(src_rank)
        
        # Step 4: Calculate ports (unique per connection pair for single-machine)
        # Strategy: Each rank uses a unique base port range
        # Rank i uses ports: base_port + i * 100 to base_port + i * 100 + 99
        # This ensures no conflicts on single machine with reasonable world_size
        
        # My port range starts at: base_port + rank * 100
        my_port_base = base_port + rank * 100
        
        # Receive port: my_port_base (only one recv port for now, accepts connections sequentially)
        recv_port = my_port_base
        recv_ports = {src: recv_port for src in source_ranks}  # All sources connect to same port
        
        # Send ports: connect to each target's recv port
        target_ports = []
        for target in target_ranks:
            # Target's recv port is at: base_port + target * 100
            port = base_port + target * 100
            target_ports.append(port)
        
        logger.info(
            f"Gemini Replicas: [Rank {rank}] Port allocation:\n"
            f"  My port range: {my_port_base} - {my_port_base + 99}\n"
            f"  My recv port: {recv_port}\n"
            f"  Target ports: {target_ports}"
        )
        
        # Step 5: Exchange IP addresses via torch.distributed.all_gather
        rank_ips = {}
        
        if torch.distributed.is_initialized():
            try:
                ip_list = [None] * world_size
                torch.distributed.all_gather_object(ip_list, base_ip)
                
                for r, ip in enumerate(ip_list):
                    rank_ips[r] = ip
                
                logger.info(
                    f"Gemini Replicas: [Rank {rank}] IP exchange completed - "
                    f"All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"Gemini Replicas: Failed to exchange IPs via all_gather, using local IP: {e}"
                )
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            logger.info("Gemini Replicas: Distributed not initialized, using local IP for all ranks")
            rank_ips[rank] = base_ip
        
        # Prepare target IPs for C++ module
        target_ips = [rank_ips[r] for r in target_ranks]
        
        config = {
            'my_ip': base_ip,
            'base_port': base_port,
            'target_ranks': target_ranks,
            'source_ranks': source_ranks,
            'rank_ips': rank_ips,
            'target_ips': target_ips,
            'target_ports': target_ports,
            'recv_ports': recv_ports,
        }
        
        logger.info(
            f"Gemini Replicas: [Rank {rank}] Network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  Target ranks (send to): {config['target_ranks']}\n"
            f"  Source ranks (recv from): {config['source_ranks']}\n"
            f"  Target ports (send): {config['target_ports']}\n"
            f"  Recv ports (from sources): {config['recv_ports']}\n"
            f"  All rank IPs: {config['rank_ips']}"
        )
        
        return config
    
    def init_gemini_replicas_if_enabled(self):
        """Initialize Gemini Replicas C++ module if enabled and distributed environment is ready."""
        if self._gemini_replicas_native is not None:
            logger.debug("Gemini Replicas: Already initialized, skipping")
            return
        
        try:
            from megatron.training import get_args as input_args
            args = input_args()
            self.use_gemini_replicas = getattr(args, 'use_gemini_replicas', False)
            self.use_gemini_replicas_optimized = getattr(args, 'use_gemini_replicas_optimized', False)
            self.num_replicas = getattr(args, 'gemini_replicas_num', 3)
            
            if not self.use_gemini_replicas or not self.use_gemini_replicas_optimized:
                return
            
            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning("Gemini Replicas: Distributed environment not initialized, skipping initialization")
                return
            
            # Initialize Gemini Replicas C++ module
            self._init_gemini_replicas_native()
            
        except Exception as e:
            logger.warning(f"Gemini Replicas: Failed to initialize: {e}")
            self._gemini_replicas_native = None
    
    def _init_gemini_replicas_native(self):
        """Initialize Gemini Replicas C++ native module with ASIO."""
        gemini_replicas_native = None
        try:
            # Load .so file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            import glob as _glob_module
            so_files = _glob_module.glob(os.path.join(current_dir, "gemini_replicas_native*.so"))
            
            if not so_files:
                logger.warning(f"Gemini Replicas: No gemini_replicas_native.so file found in {current_dir}, will use fallback")
                return
            
            # Load .so file directly
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("gemini_replicas_native", so_path)
            gemini_replicas_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(gemini_replicas_native)
            logger.debug(f"Gemini Replicas: Loaded .so file from {so_path}")
            
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # Get network configuration
            net_config = self._get_gemini_replicas_network_config(rank, world_size)
            
            # Store target ranks for later use
            self.target_ranks = net_config['target_ranks']
            
            # Synchronize all ranks before creating C++ instances
            logger.info(f"Gemini Replicas: [Rank {rank}] Synchronizing all ranks before creating C++ native module...")
            torch.distributed.barrier()
            logger.info(f"Gemini Replicas: [Rank {rank}] All ranks synchronized, creating C++ native module with ASIO...")
            
            # Use the prepared target_ips and target_ports from config
            target_ips = net_config['target_ips']
            target_ports = net_config['target_ports']
            
            # Get my recv port (all sources will connect to this port)
            recv_ports_list = list(net_config['recv_ports'].values())
            my_recv_port = recv_ports_list[0] if recv_ports_list else net_config['base_port'] + rank * 100
            
            # Calculate number of source ranks
            num_source_ranks = len(net_config['source_ranks'])
            
            # Create C++ instance with ASIO parameters (Phase 1: start acceptor only)
            logger.info(f"Gemini Replicas: Creating C++ native module with ASIO (Phase 1: acceptor)...")
            print(f"Gemini Replicas: [Rank {rank}] Creating C++ native module (Phase 1: starting acceptor)...")
            print(f"Gemini Replicas: [Rank {rank}] Target ranks: {net_config['target_ranks']}")
            print(f"Gemini Replicas: [Rank {rank}] Target IPs: {target_ips}")
            print(f"Gemini Replicas: [Rank {rank}] Target ports for sending: {target_ports}")
            print(f"Gemini Replicas: [Rank {rank}] My recv port: {my_recv_port} (accepting from {num_source_ranks} sources: {net_config['source_ranks']})")
            
            self._gemini_replicas_native = gemini_replicas_native.GeminiReplicasNative(
                rank, world_size,
                net_config['target_ranks'],  # List of target ranks
                target_ips,  # List of target IPs
                target_ports,  # List of target ports
                net_config['my_ip'],  # My IP for acceptor
                my_recv_port,  # My recv port
                num_source_ranks  # Number of expected incoming connections
            )
            
            logger.info(f"Gemini Replicas: C++ native module created (acceptor ready) for rank {rank}")
            print(f"Gemini Replicas: [Rank {rank}] Acceptor ready, waiting for all ranks...")
            
            # Synchronize all ranks before connecting (Phase 2)
            torch.distributed.barrier()
            logger.info(f"Gemini Replicas: [Rank {rank}] All ranks ready, starting Phase 2 (connecting)...")
            print(f"Gemini Replicas: [Rank {rank}] Phase 2: Connecting to target ranks {net_config['target_ranks']}...")
            
            # Phase 2: Connect to all targets
            self._gemini_replicas_native.finalize_connections()
            
            logger.info(f"Gemini Replicas: C++ native module fully initialized (rank={rank}, targets={net_config['target_ranks']})")
            print(f"Gemini Replicas: [Rank {rank}] C++ native module fully initialized - ASIO connections ready")
            
        except Exception as e:
            logger.error(f"Gemini Replicas: Failed to initialize C++ native module: {e}")
            import traceback
            traceback.print_exc()
            self._gemini_replicas_native = None
    
    def prepare_decomposed_state_dict(self, plan, planner):
        """
        Prepare decomposed state dict from SavePlan for efficient GPU-to-CPU transfer.
        
        Similar to Gemini's prepare_decomposed_state_dict.
        
        Args:
            plan: SavePlan from PyTorch distributed checkpoint
            planner: SavePlanner instance
        """
        from time import time
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        start = time()
        
        logger.info(f"Gemini Replicas: [Rank {rank}] Preparing decomposed state dict from plan...")
        
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
                if hasattr(data, 'getvalue'):  # BytesIO object
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
            f"Gemini Replicas: [Rank {rank}] Processed {byte_io_count} BytesIO items, {tensor_count} tensor items"
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
            f"Gemini Replicas: [Rank {rank}] Prepared decomposed state dict in {process_time:.2f}s\n"
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
                logger.info(f"Gemini Replicas: [Rank {rank}] Reusing existing preallocated buffer")
                return
        
        logger.info(f"Gemini Replicas: [Rank {rank}] Allocating preallocated buffer: {size_bytes / (1024**3):.2f} GB")
        
        if self.gemini_replicas_pin_memory and torch.cuda.is_available():
            self.preallocated_cpu_buffer = torch.empty(size_bytes, dtype=torch.uint8).pin_memory()
            logger.info(f"Gemini Replicas: [Rank {rank}] Allocated pinned memory buffer")
        else:
            self.preallocated_cpu_buffer = torch.empty(size_bytes, dtype=torch.uint8)
            logger.info(f"Gemini Replicas: [Rank {rank}] Allocated regular CPU buffer")
    
    def get_native_module(self):
        """Get the C++ native module instance."""
        return self._gemini_replicas_native
    
    def is_initialized(self) -> bool:
        """Check if Gemini Replicas native module is initialized."""
        return self._gemini_replicas_native is not None
    
    def cleanup(self):
        """Cleanup resources."""
        if self._gemini_replicas_native is not None:
            logger.info("Gemini Replicas: Cleaning up native module")
            self._gemini_replicas_native = None
        
        self.preallocated_cpu_buffer = None
        self.decomposed_state_dict = None
        self.replica_buffers = []
        self.replica_metadata = []

