# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""ECLATIN manager for shared eclatin_native initialization and buffer management."""

import os
import queue
import threading
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
from dataclasses import replace

from .state_dict_decomposer import GlobalMetadataRegistry, TensorMetadata

logger = getLogger(__name__)


class ECLATINManager:
    """Shared manager for ECLATIN C++ module initialization and buffer management.
    
    This class provides a singleton instance that manages:
    - ECLATIN C++ native module (_eclatin_native)
    - Buffer allocation and management (data and recv buffers, pooled)
    - Buffer poller thread for releasing buffers
    
    Note: The 4 persistent blocks (data_block_1/2, parity_block_1/2) are allocated
    in strategy after metadata exchange, not in manager.
    
    Both TorchDistSaveShardedStrategy and TorchDistLoadShardedStrategy
    can share the same manager instance to reuse initialized resources.
    """
    
    _instance: Optional['ECLATINManager'] = None
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
        
        self._eclatin_native = None
        self.use_eclatin = False
        
        # Buffer configuration
        self.eclatin_data_buffers_count = 12
        self.eclatin_recv_buffers_count = 12  # Pooled recv buffers
        self.eclatin_buffer_size = 64 * 1024 * 1024  # 64MB
        self.eclatin_pin_memory = True
        
        # Buffers (simplified: only data and recv pools)
        self.eclatin_data_buffers: Optional[List[torch.Tensor]] = None
        self.eclatin_recv_buffers: Optional[List[torch.Tensor]] = None  # Pooled recv buffers
        # Note: The 4 persistent blocks (data_block_1/2, parity_block_1/2) are allocated in strategy
        
        # Free buffer queues (simplified)
        self._free_data_buffer_queue: Optional[queue.Queue] = None
        self._free_recv_buffer_queue: Optional[queue.Queue] = None
        
        # Buffer poller thread
        self._buffer_poller_thread: Optional[threading.Thread] = None
        self._buffer_poller_stop_event: Optional[threading.Event] = None
        self._buffer_poller_active_event: Optional[threading.Event] = None
        
        self._initialized = True


    def _get_xor_paired_rank(self, my_rank: int, world_size: int) -> int:
        """Get the paired rank for parity exchange."""
        if world_size % 2 != 0:
            raise ValueError(f"ECLATIN: World size must be even for pairing, got {world_size}")
        
        if my_rank == 0:
            return 2
        if my_rank == 1:
            return 3
        if my_rank == 2:
            return 0
        if my_rank == 3:
            return 1
        
        # half_size = world_size // 2
        
        # if my_rank < half_size:
        #     # First half pairs with second half
        #     paired_rank = my_rank + half_size
        # else:
        #     # Second half pairs with first half
        #     paired_rank = my_rank - half_size
        
        # logger.debug(f"ECLATIN: Rank {my_rank} paired with Rank {paired_rank}")
        # return paired_rank

    def get_p2p_partner_rank(self, my_rank: int, world_size: int) -> int:
        """Get P2P partner rank for data/parity exchange.
        
        P2P pairing rules (different from XOR pairing):
        - Rank 0 ↔ Rank 1 (P2P)
        - Rank 2 ↔ Rank 3 (P2P)
        
        XOR pairing (for reference):
        - Rank 0 ↔ Rank 2 (XOR)
        - Rank 1 ↔ Rank 3 (XOR)
        
        Args:
            my_rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            int: P2P partner rank
        """
        if world_size % 2 != 0:
            raise ValueError(f"ECLATIN: World size must be even for P2P pairing, got {world_size}")
        
        # P2P pairing: adjacent ranks in pairs
        # For 4-rank setup: (0,1) and (2,3)
        if my_rank % 2 == 0:
            # Even rank: pair with next rank
            p2p_partner_rank = my_rank + 1
        else:
            # Odd rank: pair with previous rank
            p2p_partner_rank = my_rank - 1
        
        # Ensure partner rank is valid
        if p2p_partner_rank < 0 or p2p_partner_rank >= world_size:
            raise ValueError(f"ECLATIN: Invalid P2P partner rank {p2p_partner_rank} for rank {my_rank}")
        
        logger.debug(f"ECLATIN: Rank {my_rank} P2P partner is Rank {p2p_partner_rank}")
        return p2p_partner_rank
    
    def get_parity2_send1_partner_rank(self, my_rank: int, world_size: int) -> int:
        """Get Parity2 send1 partner rank.
        
        Parity2 send1 pairing rules:
        - Rank 0 → Rank 3
        - Rank 1 → Rank 2
        - Rank 2 → Rank 1
        - Rank 3 → Rank 0
        
        Formula: (world_size - my_rank) % world_size
        
        Args:
            my_rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            int: Parity2 send1 partner rank
        """
        partner_rank = (world_size - my_rank) % world_size
        logger.debug(f"ECLATIN: Rank {my_rank} Parity2 send1 partner is Rank {partner_rank}")
        return partner_rank
    
    def get_parity2_send2_partner_rank(self, my_rank: int, world_size: int) -> int:
        """Get Parity2 send2 partner rank.
        
        Parity2 send2 pairing rules:
        - Rank 0 → Rank 2
        - Rank 1 → Rank 3
        - Rank 2 → Rank 0
        - Rank 3 → Rank 1
        
        Formula: (my_rank + world_size // 2) % world_size
        
        Args:
            my_rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            int: Parity2 send2 partner rank
        """
        partner_rank = (my_rank + world_size // 2) % world_size
        logger.debug(f"ECLATIN: Rank {my_rank} Parity2 send2 partner is Rank {partner_rank}")
        return partner_rank
    
    def _get_eclatin_network_config(self, rank: int, world_size: int) -> dict:
        """
        Get network configuration for ECLATIN ASIO connections.
        
        This function:
        1. Gets base IP address (from ECLATIN_BASE_IP env var, MASTER_ADDR, or auto-detect)
        2. Calculates ports for this rank (base_port + rank * 4 + offset)
        3. Exchanges IP addresses with all ranks via torch.distributed.all_gather
        4. Returns configuration dictionary
        
        Args:
            rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            dict: Network configuration with keys:
                - 'my_ip': str - This rank's IP address
                - 'base_port': int - Base port number
                - 'ports': dict - Port numbers for each connection type
                    - 'send1': int
                    - 'send2': int
                    - 'recv1': int
                    - 'recv2': int
        """
        import socket
        
        # Step 1: Get base IP address
        # Priority: ECLATIN_BASE_IP > auto-detect > MASTER_ADDR (fallback)
        base_ip = os.environ.get('ECLATIN_BASE_IP')
        
        # If ECLATIN_BASE_IP is not set, try to auto-detect actual IP
        if not base_ip:
            try:
                # Get IP of the interface used for distributed training
                # Connect to a remote address (doesn't actually send data)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # Use a public DNS server IP to determine the default route interface
                s.connect(('8.8.8.8', 80))
                base_ip = s.getsockname()[0]
                s.close()
                logger.info(f"ECLATIN: Auto-detected IP address: {base_ip}")
            except Exception as e:
                logger.warning(f"ECLATIN: Failed to auto-detect IP: {e}")
                # Fallback to MASTER_ADDR or localhost
                base_ip = os.environ.get('MASTER_ADDR', '127.0.0.1')
                logger.warning(f"ECLATIN: Using fallback IP: {base_ip}")
        
        # Step 2: Get base port
        # Priority: ECLATIN_BASE_PORT > MASTER_PORT + 10000 > default 16000
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('ECLATIN_BASE_PORT', master_port + 10000))
        
        # Step 3: Calculate ports for this rank
        # Port allocation: base_port + rank * 8 + offset
        # Parity 1: offset 0-3 (send1, send2, recv1, recv2)
        # Parity 2: offset 4-7 (send1, send2, recv1, recv2)
        ports = {
            # Parity 1 ports
            'parity1_send1': base_port + rank * 8 + 0,
            'parity1_send2': base_port + rank * 8 + 1,
            'parity1_recv1': base_port + rank * 8 + 2,
            'parity1_recv2': base_port + rank * 8 + 3,
            # Parity 2 ports
            'parity2_send1': base_port + rank * 8 + 4,
            'parity2_send2': base_port + rank * 8 + 5,
            'parity2_recv1': base_port + rank * 8 + 6,
            'parity2_recv2': base_port + rank * 8 + 7,
        }
        
        # Load mode ports (for rank2 recovery)
        # rank2 needs 6 recv sockets (from rank0/1/3)
        # rank0/1/3 need 2 send sockets each (to rank2)
        # Port allocation: base_port + 1000 + offset (to avoid conflict with save mode)
        load_base_port = base_port + 1000
        if rank == 2:
            # rank2: 6 recv ports
            ports.update({
                'load_recv_rank0_data2': load_base_port + 0,
                'load_recv_rank0_parity2': load_base_port + 1,
                'load_recv_rank1_data1': load_base_port + 2,
                'load_recv_rank1_parity1': load_base_port + 3,
                'load_recv_rank3_data1': load_base_port + 4,
                'load_recv_rank3_data2': load_base_port + 5,
            })
        else:
            # rank0/1/3: 2 send ports each
            if rank == 0:
                ports.update({
                    'load_send_rank0_data2': load_base_port + 0,  # connects to rank2's load_recv_rank0_data2
                    'load_send_rank0_parity2': load_base_port + 1,  # connects to rank2's load_recv_rank0_parity2
                })
            elif rank == 1:
                ports.update({
                    'load_send_rank1_data1': load_base_port + 2,  # connects to rank2's load_recv_rank1_data1
                    'load_send_rank1_parity1': load_base_port + 3,  # connects to rank2's load_recv_rank1_parity1
                })
            elif rank == 3:
                ports.update({
                    'load_send_rank3_data1': load_base_port + 4,  # connects to rank2's load_recv_rank3_data1
                    'load_send_rank3_data2': load_base_port + 5,  # connects to rank2's load_recv_rank3_data2
                })
        
        # Step 4: Exchange IP addresses via torch.distributed.all_gather
        # TODO: Determine partner ranks for send1/send2/recv1/recv2 based on ECLATIN pairing logic
        rank_ips = {}
        
        if torch.distributed.is_initialized():
            try:
                # Convert IP to bytes, then to int list for tensor
                my_ip_bytes = socket.inet_aton(base_ip)
                my_ip_tensor = torch.tensor(
                    [int(b) for b in my_ip_bytes], 
                    dtype=torch.uint8
                )
                
                # Move to CUDA if available (for NCCL backend compatibility)
                if torch.cuda.is_available():
                    my_ip_tensor = my_ip_tensor.cuda()
                
                # Gather all IPs
                ip_list = [torch.zeros_like(my_ip_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(ip_list, my_ip_tensor)
                
                # Convert back to IP strings
                for r, ip_tensor in enumerate(ip_list):
                    ip_bytes = bytes(ip_tensor.cpu().tolist())
                    rank_ips[r] = socket.inet_ntoa(ip_bytes)
                
                logger.info(
                    f"ECLATIN: [Rank {rank}] IP exchange completed - "
                    f"All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"ECLATIN: Failed to exchange IPs via all_gather, using local IP: {e}"
                )
                # Fallback to using local IP for all ranks
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            # Single rank mode - use local IP
            logger.info("ECLATIN: Distributed not initialized, using local IP for all ranks")
            rank_ips[0] = base_ip
        
        config = {
            'my_ip': base_ip,
            'base_port': base_port,
            'rank_ips': rank_ips,
            'ports': ports,
        }
        
        logger.info(
            f"ECLATIN: [Rank {rank}] Network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  Ports: {config['ports']}\n"
            f"  All rank IPs: {config['rank_ips']}"
        )
        
        return config
    
    def init_eclatin_if_enabled(self):
        """Initialize ECLATIN C++ module if enabled and distributed environment is ready."""
        if self._eclatin_native is not None:
            logger.debug("ECLATIN: Already initialized, skipping")
            return
        
        try:
            from megatron.training import get_args as input_args
            args = input_args()
            self.use_eclatin = args.use_eclatin
            if not getattr(args, 'use_eclatin', False):
                return
                
            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning("ECLATIN: Distributed environment not initialized, skipping ECLATIN initialization")
                return
                
            # Initialize ECLATIN C++ module
            self._init_eclatin_native()
            
            # Start persistent buffer poller thread
            self._start_buffer_poller_thread()
            
        except Exception as e:
            logger.warning(f"ECLATIN: Failed to initialize during manager initialization: {e}")
            self._eclatin_native = None
    
    def _init_eclatin_native(self):
        """Initialize ECLATIN C++ native module."""
        eclatin_native = None
        try:
            # Direct import .so file without modifying sys.path or affecting other packages
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Find .so file
            import glob as _glob_module
            so_files = _glob_module.glob(os.path.join(current_dir, "eclatin_native*.so"))
            
            if not so_files:
                raise ImportError(f"No eclatin_native.so file found in {current_dir}")
            
            # Load .so file directly using importlib
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("eclatin_native", so_path)
            eclatin_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(eclatin_native)
            logger.debug(f"ECLATIN: Loaded .so file from {so_path}")
            
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # ECLATIN only uses ASIO (no NCCL support)
            # Create instance with error handling
            try:
                # ===== ASIO Initialization Path =====
                logger.info(f"ECLATIN: [Rank {rank}] Using ASIO for communication")
                
                # Get network configuration
                net_config = self._get_eclatin_network_config(rank, world_size)
                
                # Synchronize all ranks before creating C++ instances
                logger.info(f"ECLATIN: [Rank {rank}] Synchronizing all ranks before creating C++ native module (ASIO)...")
                torch.distributed.barrier()
                logger.info(f"ECLATIN: [Rank {rank}] All ranks synchronized, creating C++ native module with ASIO...")
                
                # Create C++ instance with ASIO parameters
                logger.info(f"ECLATIN: Creating C++ native module with ASIO (this will block until ASIO connections are established)...")
                print(f"ECLATIN: [Rank {rank}] Creating C++ native module with ASIO (blocking until ASIO initialization completes)...")
                
                # ECLATIN requires 16 parameters (8 per parity):
                # Parity 1: send1_ip, send1_port, send2_ip, send2_port, recv1_ip, recv1_port, recv2_ip, recv2_port
                # Parity 2: send1_ip, send1_port, send2_ip, send2_port, recv1_ip, recv1_port, recv2_ip, recv2_port
                
                # Parity 1 pairing rules:
                # - send1 pairs with partner's recv2 (XOR pairing: 0↔2, 1↔3)
                # - send2 pairs with partner's recv1 (P2P pairing: 0↔1, 2↔3)
                parity1_send1_partner_rank = self._get_xor_paired_rank(rank, world_size)  # send1 → partner's recv2 (XOR pairing)
                parity1_send2_partner_rank = self.get_p2p_partner_rank(rank, world_size)  # send2 → partner's recv1 (P2P pairing)
                
                # Parity 2 pairing rules:
                # - send1 pairs with partner's recv2 (reverse pairing: 0↔3, 1↔2)
                # - send2 pairs with partner's recv1 (cross-half pairing: 0↔2, 1↔3)
                parity2_send1_partner_rank = self.get_parity2_send1_partner_rank(rank, world_size)  # send1 → partner's recv2
                parity2_send2_partner_rank = self.get_parity2_send2_partner_rank(rank, world_size)  # send2 → partner's recv1
                
                base_port = net_config['base_port']
                rank_ips = net_config['rank_ips']
                
                # Calculate partner ports (send connects to partner's recv port)
                # Parity 1: port offset = rank * 8 + offset
                parity1_send1_partner_recv_port = base_port + parity1_send1_partner_rank * 8 + 3  # recv2 offset
                parity1_send2_partner_recv_port = base_port + parity1_send2_partner_rank * 8 + 2  # recv1 offset
                
                # Parity 2: port offset = rank * 8 + 4 + offset
                parity2_send1_partner_recv_port = base_port + parity2_send1_partner_rank * 8 + 7  # recv2 offset (4+3)
                parity2_send2_partner_recv_port = base_port + parity2_send2_partner_rank * 8 + 6  # recv1 offset (4+2)
                
                self._eclatin_native = eclatin_native.ECLATINNative(
                    # Parity 1: send1, send2, recv1, recv2
                    rank_ips.get(parity1_send1_partner_rank, net_config['my_ip']), parity1_send1_partner_recv_port,
                    rank_ips.get(parity1_send2_partner_rank, net_config['my_ip']), parity1_send2_partner_recv_port,
                    net_config['my_ip'], net_config['ports']['parity1_recv1'],
                    net_config['my_ip'], net_config['ports']['parity1_recv2'],
                    # Parity 2: send1, send2, recv1, recv2
                    rank_ips.get(parity2_send1_partner_rank, net_config['my_ip']), parity2_send1_partner_recv_port,
                    rank_ips.get(parity2_send2_partner_rank, net_config['my_ip']), parity2_send2_partner_recv_port,
                    net_config['my_ip'], net_config['ports']['parity2_recv1'],
                    net_config['my_ip'], net_config['ports']['parity2_recv2']
                )
                
                # If we reach here, ASIO connections are ready and threads are running
                logger.info(f"ECLATIN: C++ native module initialized successfully with ASIO (rank={rank}, world_size={world_size})")
                print(f"ECLATIN: [Rank {rank}] C++ native module initialized - ASIO connections ready for data exchange")
                
                # Initialize ECLATIN buffers
                self._init_eclatin_buffers()
        
            except Exception as e:
                logger.warning(f"ECLATIN: Failed to create C++ native module instance: {e}")
                # Try to stop the pipeline if it was partially created
                try:
                    if hasattr(self, '_eclatin_native') and self._eclatin_native is not None:
                        self._eclatin_native.stop()
                except:
                    pass
                self._eclatin_native = None
                raise e
            
        except ImportError as e:
            logger.warning(f"ECLATIN: C++ native module not available: {e}, ECLATIN functionality will not work")
            self._eclatin_native = None
        except Exception as e:
            logger.warning(f"ECLATIN: Failed to initialize C++ native module: {e}, ECLATIN functionality will not work")
            self._eclatin_native = None
    
    def _init_eclatin_buffers(self):
        """Initialize ECLATIN buffers during C++ module initialization.
        
        Note: Only allocates data and recv buffers (pooled) at initialization.
        The 4 persistent blocks (data_block_1/2, parity_block_1/2) will be allocated
        in strategy after metadata exchange.
        """
        rank = torch.distributed.get_rank()
        logger.info("ECLATIN: Initializing buffers for ECLATIN (data and recv pools only)")
        print(f"ECLATIN: Initializing buffers for ECLATIN (rank={rank}, data and recv pools only)")
        
        # Allocate data buffers for storing original tensor data
        self.eclatin_data_buffers = self._allocate_data_buffers()
        
        # Allocate recv buffers (pooled) for receiving data
        self.eclatin_recv_buffers = self._allocate_recv_buffers()
        
        # Initialize free buffer queues
        self._free_data_buffer_queue = queue.Queue()
        for buffer in self.eclatin_data_buffers:
            self._free_data_buffer_queue.put(int(buffer.data_ptr()))
        
        self._free_recv_buffer_queue = queue.Queue()
        for buffer in self.eclatin_recv_buffers:
            self._free_recv_buffer_queue.put(int(buffer.data_ptr()))

        logger.info(f"ECLATIN: Buffer initialization completed - "
                   f"Data buffers: {len(self.eclatin_data_buffers)}, "
                   f"Recv buffers: {len(self.eclatin_recv_buffers)}")
        print(f"ECLATIN: Buffer initialization completed (rank={rank}) - "
              f"Data buffers: {len(self.eclatin_data_buffers)}, "
              f"Recv buffers: {len(self.eclatin_recv_buffers)}")
    
    def _allocate_data_buffers(self):
        """Allocate data buffers for storing original tensor data."""
        logger.info(f"ECLATIN: Allocating data buffers ({self.eclatin_data_buffers_count} buffers, {self.eclatin_buffer_size // (1024*1024)}MB each)")
        
        data_buffers = []
        for i in range(self.eclatin_data_buffers_count):
            buffer = torch.empty(self.eclatin_buffer_size, dtype=torch.uint8, pin_memory=self.eclatin_pin_memory)
            data_buffers.append(buffer)
            logger.debug(f"ECLATIN: Allocated data buffer {i}: {self.eclatin_buffer_size} bytes")
        
        logger.info(f"ECLATIN: Allocated {len(data_buffers)} data buffers")
        return data_buffers
    
    def _allocate_recv_buffers(self):
        """Allocate recv buffers (pooled) for receiving data."""
        logger.info(f"ECLATIN: Allocating recv buffers ({self.eclatin_recv_buffers_count} buffers, {self.eclatin_buffer_size // (1024*1024)}MB each)")
        
        recv_buffers = []
        for i in range(self.eclatin_recv_buffers_count):
            buffer = torch.empty(self.eclatin_buffer_size, dtype=torch.uint8, pin_memory=self.eclatin_pin_memory)
            recv_buffers.append(buffer)
            logger.debug(f"ECLATIN: Allocated recv buffer {i}: {self.eclatin_buffer_size} bytes")
        
        logger.info(f"ECLATIN: Allocated {len(recv_buffers)} recv buffers")
        return recv_buffers
    
    def allocate_eclatin_load_recv_buffers(self, global_registry: GlobalMetadataRegistry) -> Dict[str, torch.Tensor]:
        """
        Allocate 6 recv buffers for rank0 to receive blocks from other ranks.
        
        Rank0 needs to receive:
        - rank1_data1, rank1_data2 (from rank1)
        - rank2_data2, rank2_parity2 (from rank2)
        - rank3_data1, rank3_parity1 (from rank3)
        
        Each buffer size is aligned_half_block_size (half of max_total_bytes).
        
        Args:
            global_registry (GlobalMetadataRegistry): Complete metadata from all ranks
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 6 recv buffers
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        if rank != 2:
            logger.warning("ECLATIN: allocate_eclatin_load_recv_buffers called on non-rank2, returning empty dict")
            return {}
        
        # Calculate maximum data size across all ranks
        max_total_bytes = 0
        for r in range(world_size):
            rank_metadata = global_registry.rank_metadata.get(r, [])
            rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
            if rank_total_size > max_total_bytes:
                max_total_bytes = rank_total_size
        
        # Calculate aligned half block size (same as save phase)
        half_max_total_bytes = max_total_bytes // 2
        aligned_half_block_size = ((half_max_total_bytes + self.eclatin_buffer_size - 1) // self.eclatin_buffer_size) * self.eclatin_buffer_size
        
        logger.info(
            f"ECLATIN: Allocating 6 recv buffers for rank2 load recovery\n"
            f"  Pipeline max size: {max_total_bytes / (1024**3):.2f} GB\n"
            f"  Aligned half block size (per buffer): {aligned_half_block_size / (1024**3):.2f} GB\n"
            f"  Total recv memory: {6 * aligned_half_block_size / (1024**3):.2f} GB"
        )
        
        # Allocate 6 recv buffers
        recv_buffers = {
            'rank0_data2': torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=self.eclatin_pin_memory),
            'rank0_parity2': torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=self.eclatin_pin_memory),
            'rank1_data1': torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=self.eclatin_pin_memory),
            'rank1_parity1': torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=self.eclatin_pin_memory),
            'rank3_data1': torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=self.eclatin_pin_memory),
            'rank3_data2': torch.empty(aligned_half_block_size, dtype=torch.uint8, pin_memory=self.eclatin_pin_memory),
        }
        
        logger.info(
            f"ECLATIN: Allocated 6 recv buffers for rank2: "
            f"{aligned_half_block_size / (1024**3):.2f} GB each"
        )
        
        return recv_buffers
    
    def _poll_and_release_buffers(self):
        """Poll C++ for buffers ready to be released and put them back to queues."""
        if self._eclatin_native is None:
            return
        
        # Get data buffers ready for release
        data_buffers = self._eclatin_native.get_data_buffers_to_release()
        for data_addr in data_buffers:
            try:
                self._free_data_buffer_queue.put_nowait(data_addr)
                # logger.debug(f"ECLATIN: Released data buffer at address {data_addr}")
            except Exception:
                logger.error(f"ECLATIN: Data buffer queue is full, cannot release buffer {data_addr}")
        
        # Get recv buffers ready for release
        recv_buffers = self._eclatin_native.get_recv_buffers_to_release()
        for recv_addr in recv_buffers:
            try:
                self._free_recv_buffer_queue.put_nowait(recv_addr)
                # logger.debug(f"ECLATIN: Released recv buffer at address {recv_addr}")
            except Exception:
                logger.error(f"ECLATIN: Recv buffer queue is full, cannot release buffer {recv_addr}")
    
    def _start_buffer_poller_thread(self):
        """Start a persistent background thread to poll and release buffers."""
        if hasattr(self, '_buffer_poller_thread') and self._buffer_poller_thread is not None:
            logger.warning("ECLATIN: Buffer poller thread already started")
            return
        
        # Create control events
        self._buffer_poller_stop_event = threading.Event()
        self._buffer_poller_active_event = threading.Event()
        
        def buffer_poller_worker():
            """Persistent background thread that polls for buffer releases."""
            logger.info("ECLATIN: Buffer poller thread started")
            poll_count = 0
            
            while not self._buffer_poller_stop_event.is_set():
                # Only poll when active
                if self._buffer_poller_active_event.is_set():
                    self._poll_and_release_buffers()
                    poll_count += 1
                    if poll_count % 1000 == 0:
                        logger.debug(f"ECLATIN: Buffer poller running (polled {poll_count} times)")
                
                # Sleep briefly to avoid busy waiting
                from time import sleep
                sleep(0.001)  # 1ms
            
            logger.info("ECLATIN: Buffer poller thread stopping")
        
        # Start the daemon thread
        self._buffer_poller_thread = threading.Thread(target=buffer_poller_worker, daemon=True)
        self._buffer_poller_thread.start()
        logger.info("ECLATIN: Buffer poller thread created and started")
    
    def _stop_buffer_poller_thread(self):
        """Stop the persistent buffer poller thread."""
        if not hasattr(self, '_buffer_poller_thread') or self._buffer_poller_thread is None:
            return
        
        logger.info("ECLATIN: Stopping buffer poller thread...")
        
        # Signal the thread to stop
        if self._buffer_poller_stop_event:
            self._buffer_poller_stop_event.set()
        
        # Wait for thread to finish
        if self._buffer_poller_thread.is_alive():
            self._buffer_poller_thread.join(timeout=2.0)
            if self._buffer_poller_thread.is_alive():
                logger.warning("ECLATIN: Buffer poller thread did not stop in time")
            else:
                logger.info("ECLATIN: Buffer poller thread stopped successfully")
        
        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None
    
    def get_eclatin_buffers(self):
        """Get ECLATIN buffers for FileSystemWriterAsync.
        
        Note: Returns data and recv buffers (pooled).
        The 4 persistent blocks (data_block_1/2, parity_block_1/2) will be allocated
        in strategy after metadata exchange.
        
        Returns:
            Dict containing all buffer information, or None if not initialized
        """
        if self.eclatin_data_buffers is None:
            return None
        
        return {
            'data_buffers': self.eclatin_data_buffers,
            'recv_buffers': self.eclatin_recv_buffers,
            'free_data_buffer_queue': self._free_data_buffer_queue,
            'free_recv_buffer_queue': self._free_recv_buffer_queue,
            # Pass buffer poller control objects
            'buffer_poller_active_event': self._buffer_poller_active_event,
            'poll_and_release_buffers': self._poll_and_release_buffers,
            # Note: The 4 persistent blocks (data_block_1/2, parity_block_1/2) 
            # will be allocated in strategy after metadata exchange
        }
    
    def cleanup(self):
        """Cleanup ECLATIN resources when manager is destroyed."""
        try:
            # Stop buffer poller thread
            self._stop_buffer_poller_thread()
            
            # Stop the C++ pipeline
            if hasattr(self, '_eclatin_native') and self._eclatin_native is not None:
                self._eclatin_native.stop()
                logger.info("ECLATIN: C++ native module stopped in manager cleanup")
                
        except Exception as e:
            logger.warning(f"ECLATIN: Error during manager cleanup: {e}")
    
    def __del__(self):
        """Cleanup ECLATIN resources when manager is destroyed."""
        self.cleanup()

