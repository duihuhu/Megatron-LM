# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""EC-CHECK manager for shared eccheck_native initialization and buffer management."""

import os
import queue
import threading
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
from dataclasses import replace

from .state_dict_decomposer import GlobalMetadataRegistry, TensorMetadata

logger = getLogger(__name__)

# Number of ranks per EC group (each group behaves like the original 4-rank setup)
RANKS_PER_GROUP = 4


class ECCHECKManager:
    """Shared manager for EC-CHECK C++ module initialization and buffer management.
    
    This class provides a singleton instance that manages:
    - EC-CHECK C++ native module (_eccheck_native)
    - Buffer allocation and management (data, encoding, parity buffers)
    - Buffer poller thread for releasing buffers
    
    Both TorchDistSaveShardedStrategy and TorchDistLoadShardedStrategy
    can share the same manager instance to reuse initialized resources.
    """
    
    _instance: Optional['ECCHECKManager'] = None
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
        
        self._eccheck_native = None
        self.use_eccheck = False
        self.use_rdma = False
        
        # Buffer configuration
        self.eccheck_data_buffers_count = 12
        self.eccheck_encoding_buffers_count = 24  # data_count * m (12 * 2)
        self.eccheck_buffer_size = 64 * 1024 * 1024  # 64MB
        self.eccheck_pin_memory = True
        
        # Buffers
        self.eccheck_data_buffers: Optional[List[torch.Tensor]] = None
        self.eccheck_encoding_buffers: Optional[List[torch.Tensor]] = None
        self.eccheck_recv_encoding_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.eccheck_parity_buffers: Optional[List[torch.Tensor]] = None
        self.eccheck_p2p_buffers: Optional[Dict[str, torch.Tensor]] = None
        
        # Free buffer queues
        self._free_data_buffer_queue: Optional[queue.Queue] = None
        self._free_encoding_buffer_queue: Optional[queue.Queue] = None
        self._free_parity_buffer_queue: Optional[queue.Queue] = None
        
        # Buffer poller thread
        self._buffer_poller_thread: Optional[threading.Thread] = None
        self._buffer_poller_stop_event: Optional[threading.Event] = None
        self._buffer_poller_active_event: Optional[threading.Event] = None
        
        # RDMA buffer tracking (similar to Gemini)
        self.registered_buffers: Dict[int, Tuple[int, int]] = {}  # {buffer_addr: (size, iteration)}
        self.current_iteration: int = 0
        
        self._initialized = True

    @staticmethod
    def _get_group_id(rank: int, world_size: int) -> int:
        """Get group id for multi-rank. Groups: 0,2,4,6 -> group 0; 1,3,5,7 -> group 1 (8 ranks)."""
        num_groups = world_size // RANKS_PER_GROUP
        return rank % num_groups

    @staticmethod
    def _get_rank_in_group(rank: int, world_size: int) -> int:
        """Get rank index within group (0..3). Same group logic as _get_group_id."""
        num_groups = world_size // RANKS_PER_GROUP
        return rank // num_groups

    def _get_xor_paired_rank(self, my_rank: int, world_size: int) -> int:
        """Get the paired rank for parity exchange (XOR pairing within group).

        Multi-rank: world_size must be divisible by 4. Each group of 4 ranks uses
        same pairing as original 4-rank: in-group 0<->2, 1<->3 (by position).
        E.g. 8 ranks: group0={0,2,4,6} -> 0<->4, 2<->6; group1={1,3,5,7} -> 1<->5, 3<->7.
        """
        if world_size % RANKS_PER_GROUP != 0:
            raise ValueError(
                f"EC-CHECK: World size must be divisible by {RANKS_PER_GROUP} for multi-rank, got {world_size}"
            )
        num_groups = world_size // RANKS_PER_GROUP
        group_id = self._get_group_id(my_rank, world_size)
        rank_in_group = self._get_rank_in_group(my_rank, world_size)
        # In-group XOR pairing: position 0<->2, 1<->3
        paired_rank_in_group = (rank_in_group + 2) % RANKS_PER_GROUP
        paired_rank = group_id + num_groups * paired_rank_in_group
        logger.debug(
            f"EC-CHECK: Rank {my_rank} (group_id={group_id}, rank_in_group={rank_in_group}) "
            f"XOR paired with Rank {paired_rank}"
        )
        return paired_rank

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
            raise ValueError(f"EC-CHECK: World size must be even for P2P pairing, got {world_size}")
        
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
            raise ValueError(f"EC-CHECK: Invalid P2P partner rank {p2p_partner_rank} for rank {my_rank}")
        
        logger.debug(f"EC-CHECK: Rank {my_rank} P2P partner is Rank {p2p_partner_rank}")
        return p2p_partner_rank
    
    def _get_eccheck_network_config(self, rank: int, world_size: int) -> dict:
        """
        Get network configuration for EC-CHECK ASIO connections.
        
        This function:
        1. Gets base IP address (from ECCHECK_BASE_IP env var, MASTER_ADDR, or auto-detect)
        2. Calculates ports for this rank (base_port + rank * 6 + offset)
        3. Exchanges IP addresses with all ranks via torch.distributed.all_gather
        4. Returns configuration dictionary
        
        Args:
            rank (int): Current rank
            world_size (int): Total number of ranks
            
        Returns:
            dict: Network configuration with keys:
                - 'my_ip': str - This rank's IP address
                - 'base_port': int - Base port number
                - 'xor_partner_ip': str - XOR partner's IP address
                - 'p2p_partner_ip': str - P2P partner's IP address
                - 'ports': dict - Port numbers for each connection type
                    - 'xor_send': int
                    - 'xor_recv': int
                    - 'p2p_send': int
                    - 'p2p_recv': int
        """
        import socket
        
        # Step 1: Get base IP address
        # Priority: ECCHECK_BASE_IP > MASTER_ADDR > auto-detect
        base_ip = os.environ.get('ECCHECK_BASE_IP')
        if not base_ip:
            base_ip = os.environ.get('MASTER_ADDR', '127.0.0.1')
        
        # If using localhost, try to get actual IP
        if base_ip == '127.0.0.1' or base_ip == 'localhost':
            # Check if specific network interface is requested
            interface_name = os.environ.get('ECCHECK_INTERFACE')
            
            if interface_name:
                try:
                    import netifaces
                    addrs = netifaces.ifaddresses(interface_name)
                    if netifaces.AF_INET in addrs:
                        base_ip = addrs[netifaces.AF_INET][0]['addr']
                        logger.info(f"EC-CHECK: Using IP from interface {interface_name}: {base_ip}")
                    else:
                        logger.warning(f"EC-CHECK: Interface {interface_name} has no IPv4 address")
                        base_ip = '127.0.0.1'
                except ImportError:
                    logger.warning(
                        "EC-CHECK: netifaces module not installed. "
                        "Install via 'pip install netifaces' to use ECCHECK_INTERFACE. "
                        "Falling back to auto-detection."
                    )
                    base_ip = '127.0.0.1'
                except Exception as e:
                    logger.warning(f"EC-CHECK: Failed to get IP from interface {interface_name}: {e}")
                    base_ip = '127.0.0.1'
            
            if base_ip == '127.0.0.1':
                try:
                    # Get IP of the interface used for distributed training
                    # Connect to a remote address (doesn't actually send data)
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(('8.8.8.8', 80))
                    base_ip = s.getsockname()[0]
                    s.close()
                    logger.info(f"EC-CHECK: Auto-detected IP address: {base_ip}")
                except Exception as e:
                    # Fallback to localhost
                    logger.warning(f"EC-CHECK: Failed to auto-detect IP, using localhost: {e}")
                    base_ip = '127.0.0.1'
        
        # Step 2: Get base port
        # Priority: ECCHECK_BASE_PORT > MASTER_PORT + 10000 > default 16000
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('ECCHECK_BASE_PORT', master_port + 10000))
        
        # Step 3: Calculate ports for this rank
        # Port allocation: base_port + rank * 6 + offset
        # offset: 0=xor_send, 1=xor_recv, 2=p2p_send, 3=p2p_recv, 4=step6_p2p_send, 5=step6_p2p_recv
        ports = {
            'xor_send': base_port + rank * 6 + 0,
            'xor_recv': base_port + rank * 6 + 1,
            'p2p_send': base_port + rank * 6 + 2,
            'p2p_recv': base_port + rank * 6 + 3,
            'step6_p2p_send': base_port + rank * 6 + 4,  # rank3 uses this to send to rank2
            'step6_p2p_recv': base_port + rank * 6 + 5,   # rank2 uses this to receive from rank3
        }
        
        # Step 4: Get partner ranks
        xor_partner = self._get_xor_paired_rank(rank, world_size)
        p2p_partner = self.get_p2p_partner_rank(rank, world_size)
        
        # Step 5: Exchange IP addresses via torch.distributed.all_gather
        # Initialize with fallback values
        xor_partner_ip = base_ip
        p2p_partner_ip = base_ip
        
        if torch.distributed.is_initialized():
            try:
                # Get actual IP address for each rank
                # Priority: ECCHECK_RANK_IP_<rank> > ECCHECK_INTERFACE > auto-detect from network interface
                my_actual_ip = os.environ.get(f'ECCHECK_RANK_IP_{rank}')
                if not my_actual_ip:
                    # Check if specific network interface is requested
                    interface_name = os.environ.get('ECCHECK_INTERFACE')
                    
                    if interface_name:
                        try:
                            import netifaces
                            addrs = netifaces.ifaddresses(interface_name)
                            if netifaces.AF_INET in addrs:
                                my_actual_ip = addrs[netifaces.AF_INET][0]['addr']
                                logger.info(f"EC-CHECK: [Rank {rank}] Using IP from interface {interface_name}: {my_actual_ip}")
                            else:
                                logger.warning(f"EC-CHECK: [Rank {rank}] Interface {interface_name} has no IPv4 address")
                                my_actual_ip = None
                        except ImportError:
                            logger.warning(
                                f"EC-CHECK: [Rank {rank}] netifaces module not installed. "
                                "Install via 'pip install netifaces' to use ECCHECK_INTERFACE. "
                                "Falling back to auto-detection."
                            )
                            my_actual_ip = None
                        except Exception as e:
                            logger.warning(f"EC-CHECK: [Rank {rank}] Failed to get IP from interface {interface_name}: {e}")
                            my_actual_ip = None
                    
                    if not my_actual_ip:
                        # Try to get actual IP from network interface
                        try:
                            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                            s.connect(('8.8.8.8', 80))
                            my_actual_ip = s.getsockname()[0]
                            s.close()
                            logger.info(f"EC-CHECK: [Rank {rank}] Auto-detected my IP: {my_actual_ip}")
                        except Exception as e:
                            logger.warning(f"EC-CHECK: [Rank {rank}] Failed to auto-detect IP, using base_ip: {e}")
                            my_actual_ip = base_ip
                else:
                    logger.info(f"EC-CHECK: [Rank {rank}] Using IP from ECCHECK_RANK_IP_{rank}: {my_actual_ip}")
                
                # Convert IP to bytes, then to int list for tensor
                my_ip_bytes = socket.inet_aton(my_actual_ip)
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
                rank_ips = {}
                for r, ip_tensor in enumerate(ip_list):
                    ip_bytes = bytes(ip_tensor.cpu().tolist())
                    rank_ips[r] = socket.inet_ntoa(ip_bytes)
                
                logger.info(f"EC-CHECK: [Rank {rank}] All ranks IPs: {rank_ips}")
                
                # Get partner IPs from gathered results
                xor_partner_ip = rank_ips.get(xor_partner, base_ip)
                p2p_partner_ip = rank_ips.get(p2p_partner, base_ip)
                
                # Update my_ip to use the actual detected IP
                base_ip = my_actual_ip
                
                logger.info(
                    f"EC-CHECK: [Rank {rank}] IP exchange completed - "
                    f"XOR partner ({xor_partner}): {xor_partner_ip}, "
                    f"P2P partner ({p2p_partner}): {p2p_partner_ip}"
                )
            except Exception as e:
                logger.warning(
                    f"EC-CHECK: Failed to exchange IPs via all_gather, using local IP: {e}"
                )
                # Fallback to using local IP for all partners
                xor_partner_ip = base_ip
                p2p_partner_ip = base_ip
                rank_ips = {r: base_ip for r in range(world_size)}
        else:
            # Single rank mode - use local IP
            logger.info("EC-CHECK: Distributed not initialized, using local IP for all partners")
            rank_ips = {r: base_ip for r in range(world_size)}

        config = {
            'my_ip': base_ip,
            'base_port': base_port,
            'xor_partner_ip': xor_partner_ip,
            'p2p_partner_ip': p2p_partner_ip,
            'ports': ports,
            'rank_ips': rank_ips,
        }
        
        logger.info(
            f"EC-CHECK: [Rank {rank}] Network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  XOR partner IP: {config['xor_partner_ip']}\n"
            f"  P2P partner IP: {config['p2p_partner_ip']}\n"
            f"  Ports: {config['ports']}"
        )
        
        return config
    
    def init_eccheck_if_enabled(self):
        """Initialize EC-CHECK C++ module if enabled and distributed environment is ready."""
        if self._eccheck_native is not None:
            logger.debug("EC-CHECK: Already initialized, skipping")
            return
        
        try:
            from megatron.training import get_args as input_args
            args = input_args()
            self.use_eccheck = args.use_eccheck
            self.use_rdma = getattr(args, 'use_rdma', False)
            
            if not getattr(args, 'use_eccheck', False):
                return
                
            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning("EC-CHECK: Distributed environment not initialized, skipping EC-CHECK initialization")
                return
                
            # Initialize EC-CHECK C++ module
            self._init_eccheck_native()
            
            # Start persistent buffer poller thread
            self._start_buffer_poller_thread()
            
        except Exception as e:
            logger.warning(f"EC-CHECK: Failed to initialize during manager initialization: {e}")
            self._eccheck_native = None
    
    def _init_eccheck_native(self):
        """Initialize EC-CHECK C++ native module."""
        eccheck_native = None
        try:
            # Direct import .so file without modifying sys.path or affecting other packages
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Find .so file
            import glob as _glob_module
            so_files = _glob_module.glob(os.path.join(current_dir, "eccheck_native*.so"))
            
            if not so_files:
                raise ImportError(f"No eccheck_native.so file found in {current_dir}")
            
            # Load .so file directly using importlib
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("eccheck_native", so_path)
            eccheck_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(eccheck_native)
            logger.debug(f"EC-CHECK: Loaded .so file from {so_path}")
            
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            paired_rank = self._get_xor_paired_rank(rank, world_size)
            
            # Check if using ASIO or RDMA (via environment variable or args)
            use_asio = os.environ.get('ECCHECK_USE_ASIO', 'false').lower() in ('true', '1', 'yes')
            
            # Create instance with error handling
            try:
                if use_asio or self.use_rdma:
                    # ===== ASIO/RDMA Initialization Path =====
                    transport_mode = "RDMA" if self.use_rdma else "ASIO"
                    logger.info(f"EC-CHECK: [Rank {rank}] Using {transport_mode} for communication")
                    
                    # Check RDMA availability if RDMA mode is requested
                    if self.use_rdma:
                        try:
                            rdma_available = eccheck_native.is_rdma_available()
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
                            logger.warning(f"EC-CHECK: Could not check RDMA availability: {check_err}")
                            logger.warning("EC-CHECK: Will attempt to initialize RDMA anyway...")
                    
                    # Get network configuration
                    net_config = self._get_eccheck_network_config(rank, world_size)
                    
                    # Synchronize all ranks before creating C++ instances
                    logger.info(f"EC-CHECK: [Rank {rank}] Synchronizing all ranks before creating C++ native module ({transport_mode})...")
                    torch.distributed.barrier()
                    logger.info(f"EC-CHECK: [Rank {rank}] All ranks synchronized, creating C++ native module with {transport_mode}...")
                    
                    # Create C++ instance with ASIO/RDMA parameters
                    logger.info(f"EC-CHECK: Creating C++ native module with {transport_mode} (this will block until connections are established)...")
                    print(f"EC-CHECK: [Rank {rank}] Creating C++ native module with {transport_mode} (blocking until initialization completes)...")
                    
                    # Calculate partner ports (send connects to partner's recv port)
                    # For XOR: rank 0 sends to rank 2's recv port, rank 2 sends to rank 0's recv port
                    # For P2P: rank 0 sends to rank 1's recv port, rank 1 sends to rank 0's recv port
                    xor_partner = self._get_xor_paired_rank(rank, world_size)
                    p2p_partner = self.get_p2p_partner_rank(rank, world_size)
                    
                    # Partner's recv ports (where we send to)
                    base_port = net_config['base_port']
                    xor_partner_recv_port = base_port + xor_partner * 6 + 1  # partner's xor_recv port
                    p2p_partner_recv_port = base_port + p2p_partner * 6 + 3  # partner's p2p_recv port
                    
                    # Step6 P2P: rank_in_group 3 sends to rank_in_group 2 in same group
                    # rank_in_group 2 listens on step6_p2p_recv port (multi-rank: per-group)
                    num_groups = world_size // RANKS_PER_GROUP
                    group_id = self._get_group_id(rank, world_size)
                    rank_in_group = self._get_rank_in_group(rank, world_size)
                    rank_ips = net_config.get('rank_ips', {})
                    if rank_in_group == 3:
                        step6_p2p_partner_rank = group_id + num_groups * 2
                        step6_p2p_partner_ip = rank_ips.get(step6_p2p_partner_rank, net_config['my_ip'])
                        step6_p2p_send_port = base_port + step6_p2p_partner_rank * 6 + 5
                        step6_p2p_listen_ip = ""
                        step6_p2p_recv_port = 0
                    elif rank_in_group == 2:
                        step6_p2p_partner_rank = -1
                        step6_p2p_partner_ip = ""
                        step6_p2p_send_port = 0
                        step6_p2p_listen_ip = net_config['my_ip']
                        step6_p2p_recv_port = net_config['ports']['step6_p2p_recv']
                    else:
                        step6_p2p_partner_rank = -1
                        step6_p2p_partner_ip = ""
                        step6_p2p_send_port = 0
                        step6_p2p_listen_ip = ""
                        step6_p2p_recv_port = 0
                    
                    self._eccheck_native = eccheck_native.ECCHECKNative(
                        rank, world_size, paired_rank,
                        # XOR connections: (partner_ip, partner_recv_port, my_ip, my_recv_port, use_rdma)
                        net_config['xor_partner_ip'], xor_partner_recv_port,
                        net_config['my_ip'], net_config['ports']['xor_recv'],
                        # P2P connections: (partner_ip, partner_recv_port, my_ip, my_recv_port, use_rdma)
                        net_config['p2p_partner_ip'], p2p_partner_recv_port,
                        net_config['my_ip'], net_config['ports']['p2p_recv'],
                        # Step6 P2P connections: (partner_ip, partner_recv_port, my_ip, my_recv_port)
                        # Only rank2/3 use these (rank2 recv, rank3 send)
                        step6_p2p_partner_ip, step6_p2p_send_port,
                        step6_p2p_listen_ip, step6_p2p_recv_port,
                        self.use_rdma,
                        rank_in_group,
                    )
                    
                    # If we reach here, ASIO/RDMA connections are ready and threads are running
                    logger.info(f"EC-CHECK: C++ native module initialized successfully with {transport_mode} (rank={rank}, world_size={world_size}, paired_rank={paired_rank})")
                    print(f"EC-CHECK: [Rank {rank}] C++ native module initialized - {transport_mode} connections ready for data exchange")
                    
                    # Initialize EC-CHECK buffers (same for both ASIO and NCCL)
                    self._init_eccheck_buffers()
                    
                else:
                    # ===== NCCL Initialization Path (original) =====
                    logger.info(f"EC-CHECK: [Rank {rank}] Using NCCL for communication")
                    
                    # ===== Step 1: Rank 0 generates four NCCL IDs =====
                    # thread1: for rank0↔rank2 XOR communication
                    # thread2: for rank1↔rank3 XOR communication
                    # p2p_0_1: for rank0↔rank1 P2P communication
                    # p2p_2_3: for rank2↔rank3 P2P communication
                    if rank == 0:
                        # Generate NCCL IDs using module-level function (no instance needed)
                        nccl_id_thread1 = eccheck_native.generate_nccl_id()  # rank0↔rank2
                        nccl_id_thread2 = eccheck_native.generate_nccl_id()  # rank1↔rank3
                        nccl_id_p2p_0_1 = eccheck_native.generate_nccl_id()  # rank0↔rank1
                        nccl_id_p2p_2_3 = eccheck_native.generate_nccl_id()  # rank2↔rank3
                        logger.info(f"EC-CHECK: [Rank 0] Generated four NCCL IDs (size: {len(nccl_id_thread1)} bytes each)")
                    else:
                        # Other ranks prepare empty lists (will be filled by broadcast)
                        nccl_id_thread1 = [0] * 128  # NCCL ID is typically 128 bytes
                        nccl_id_thread2 = [0] * 128
                        nccl_id_p2p_0_1 = [0] * 128
                        nccl_id_p2p_2_3 = [0] * 128
                    
                    # ===== Step 2: Broadcast NCCL IDs to all ranks =====
                    # Convert lists to torch tensors for broadcasting
                    # NOTE: NCCL backend requires tensors to be on CUDA device
                    nccl_id_size = 128  # sizeof(ncclUniqueId)
                    
                    # Convert to tensors and move to CUDA (NCCL requires CUDA tensors)
                    if rank == 0:
                        id1_tensor = torch.tensor(nccl_id_thread1, dtype=torch.uint8, device=torch.cuda.current_device())
                        id2_tensor = torch.tensor(nccl_id_thread2, dtype=torch.uint8, device=torch.cuda.current_device())
                        id_p2p_0_1_tensor = torch.tensor(nccl_id_p2p_0_1, dtype=torch.uint8, device=torch.cuda.current_device())
                        id_p2p_2_3_tensor = torch.tensor(nccl_id_p2p_2_3, dtype=torch.uint8, device=torch.cuda.current_device())
                    else:
                        id1_tensor = torch.zeros(nccl_id_size, dtype=torch.uint8, device=torch.cuda.current_device())
                        id2_tensor = torch.zeros(nccl_id_size, dtype=torch.uint8, device=torch.cuda.current_device())
                        id_p2p_0_1_tensor = torch.zeros(nccl_id_size, dtype=torch.uint8, device=torch.cuda.current_device())
                        id_p2p_2_3_tensor = torch.zeros(nccl_id_size, dtype=torch.uint8, device=torch.cuda.current_device())
                    
                    # Broadcast all four IDs (synchronous operation - all ranks wait)
                    # NCCL backend requires tensors to be on CUDA device
                    torch.distributed.broadcast(id1_tensor, src=0)
                    torch.distributed.broadcast(id2_tensor, src=0)
                    torch.distributed.broadcast(id_p2p_0_1_tensor, src=0)
                    torch.distributed.broadcast(id_p2p_2_3_tensor, src=0)
                    
                    # Convert back to lists (move to CPU first, then tolist)
                    nccl_id_thread1 = id1_tensor.cpu().tolist()
                    nccl_id_thread2 = id2_tensor.cpu().tolist()
                    nccl_id_p2p_0_1 = id_p2p_0_1_tensor.cpu().tolist()
                    nccl_id_p2p_2_3 = id_p2p_2_3_tensor.cpu().tolist()
                    
                    logger.info(f"EC-CHECK: [Rank {rank}] Received four NCCL IDs via broadcast")
                    
                    # ===== Step 3: Synchronize all ranks before creating C++ instances =====
                    # This barrier ensures all ranks start creating C++ instances at roughly the same time,
                    # which helps synchronize the NCCL communicator initialization calls.
                    logger.info(f"EC-CHECK: [Rank {rank}] Synchronizing all ranks before creating C++ native module...")
                    torch.distributed.barrier()
                    logger.info(f"EC-CHECK: [Rank {rank}] All ranks synchronized, creating C++ native module...")
                    
                    # ===== Step 4: Create C++ instance with broadcasted IDs =====
                    # IMPORTANT: This constructor call will BLOCK until:
                    # 1. Send and recv threads are started
                    # 2. All NCCL communicators are fully initialized using the broadcasted IDs
                    # 3. All threads are ready for data exchange
                    # Only after all initialization is complete will this call return.
                    logger.info(f"EC-CHECK: Creating C++ native module (this will block until NCCL is initialized)...")
                    print(f"EC-CHECK: [Rank {rank}] Creating C++ native module (blocking until NCCL initialization completes)...")
                    
                    rank_in_group = self._get_rank_in_group(rank, world_size)
                    self._eccheck_native = eccheck_native.ECCHECKNative(
                        rank, world_size, paired_rank,
                        nccl_id_thread1,    # rank0↔rank2 XOR
                        nccl_id_thread2,    # rank1↔rank3 XOR
                        nccl_id_p2p_0_1,   # rank0↔rank1 P2P
                        nccl_id_p2p_2_3,   # rank2↔rank3 P2P
                        rank_in_group,
                    )
                    
                    # If we reach here, NCCL communicators are ready and threads are running
                    logger.info(f"EC-CHECK: C++ native module initialized successfully (rank={rank}, world_size={world_size}, paired_rank={paired_rank})")
                    print(f"EC-CHECK: [Rank {rank}] C++ native module initialized - NCCL communicators ready for data exchange")
                    
                    # Initialize EC-CHECK buffers
                    self._init_eccheck_buffers()
        
            except Exception as e:
                logger.warning(f"EC-CHECK: Failed to create C++ native module instance: {e}")
                # Try to stop the pipeline if it was partially created
                try:
                    if hasattr(self, '_eccheck_native') and self._eccheck_native is not None:
                        self._eccheck_native.stop_pipeline()
                except:
                    pass
                self._eccheck_native = None
                raise e
            
        except ImportError as e:
            logger.warning(f"EC-CHECK: C++ native module not available: {e}, EC-CHECK functionality will not work")
            self._eccheck_native = None
        except Exception as e:
            logger.warning(f"EC-CHECK: Failed to initialize C++ native module: {e}, EC-CHECK functionality will not work")
            self._eccheck_native = None
    
    def _init_eccheck_buffers(self):
        """Initialize EC-CHECK buffers during C++ module initialization.
        
        Note: Only allocates data and encoding buffers at initialization.
        Receive and parity buffers will be allocated after metadata exchange,
        when peer data sizes are known.
        """
        rank = torch.distributed.get_rank()
        logger.info("EC-CHECK: Initializing buffers for EC-CHECK (data and encoding only)")
        print(f"EC-CHECK: Initializing buffers for EC-CHECK (rank={rank}, data and encoding only)")
        
        # Allocate data buffers for storing original tensor data
        self.eccheck_data_buffers = self._allocate_data_buffers()
        
        # Allocate encoding buffers for encoded packets
        self.eccheck_encoding_buffers = self._allocate_encoding_buffers()
        
        # Allocate receive buffers for peer encoded packets (will be allocated later)
        self.eccheck_recv_encoding_buffers = None
        
        # Allocate parity buffers for XOR computation results
        self.eccheck_parity_buffers = self._allocate_parity_buffers()
        
        # Allocate P2P buffers (will be allocated after metadata exchange)
        self.eccheck_p2p_buffers = None
        
        # Initialize free buffer queues
        self._free_data_buffer_queue = queue.Queue()
        for buffer in self.eccheck_data_buffers:
            self._free_data_buffer_queue.put(int(buffer.data_ptr()))
        
        self._free_encoding_buffer_queue = queue.Queue()
        for buffer in self.eccheck_encoding_buffers:
            self._free_encoding_buffer_queue.put(int(buffer.data_ptr()))
        
        self._free_parity_buffer_queue = queue.Queue()
        for buffer in self.eccheck_parity_buffers:
            self._free_parity_buffer_queue.put(int(buffer.data_ptr()))

        logger.info(f"EC-CHECK: Buffer initialization completed - "
                   f"Data buffers: {len(self.eccheck_data_buffers)}, "
                   f"Encoding buffers: {len(self.eccheck_encoding_buffers)}, "
                   f"Parity buffers: {len(self.eccheck_parity_buffers)}")
        print(f"EC-CHECK: Buffer initialization completed (rank={rank}) - "
              f"Data buffers: {len(self.eccheck_data_buffers)}, "
              f"Encoding buffers: {len(self.eccheck_encoding_buffers)}, "
              f"Parity buffers: {len(self.eccheck_parity_buffers)}")
        
        # Register all buffers for RDMA if RDMA is enabled
        if self.use_rdma:
            self.register_all_buffers_for_rdma()
    
    def _allocate_data_buffers(self):
        """Allocate data buffers for storing original tensor data."""
        logger.info(f"EC-CHECK: Allocating data buffers ({self.eccheck_data_buffers_count} buffers, {self.eccheck_buffer_size // (1024*1024)}MB each)")
        
        data_buffers = []
        for i in range(self.eccheck_data_buffers_count):
            buffer = torch.empty(self.eccheck_buffer_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
            data_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated data buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.info(f"EC-CHECK: Allocated {len(data_buffers)} data buffers")
        return data_buffers
    
    def _allocate_encoding_buffers(self):
        """Allocate encoding buffers for encoded packets."""
        logger.info(f"EC-CHECK: Allocating encoding buffers ({self.eccheck_encoding_buffers_count} buffers, {self.eccheck_buffer_size // (1024*1024)}MB each)")
        
        encoding_buffers = []
        for i in range(self.eccheck_encoding_buffers_count):
            buffer = torch.empty(self.eccheck_buffer_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
            encoding_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated encoding buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.info(f"EC-CHECK: Allocated {len(encoding_buffers)} encoding buffers")
        return encoding_buffers
    
    def _allocate_parity_buffers(self):
        """Allocate parity buffers for XOR computation results.
        
        Note: Parity buffer count should match encoding buffer count (24) to support
        pipelined operations where each data chunk needs 2 parity buffers (one per thread).
        """
        # Use encoding buffer count instead of data buffer count
        # Each data chunk needs 2 parity buffers (thread1 and thread2)
        parity_buffer_count = self.eccheck_encoding_buffers_count
        logger.info(f"EC-CHECK: Allocating parity buffers ({parity_buffer_count} buffers)")
        
        parity_buffers = []
        for i in range(parity_buffer_count):
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
                # logger.info(f"EC-CHECK: Released data buffer at address {data_addr}")
            except Exception:
                logger.error(f"EC-CHECK: Data buffer queue is full, cannot release buffer {data_addr}")
        
        # Get encoding buffers ready for release
        encoding_buffers = self._eccheck_native.get_encoding_buffers_to_release()
        for encoding_addr in encoding_buffers:
            try:
                self._free_encoding_buffer_queue.put_nowait(encoding_addr)
                # logger.info(f"EC-CHECK: Released encoding buffer at address {encoding_addr}")
            except Exception:
                logger.error(f"EC-CHECK: Encoding buffer queue is full, cannot release buffer {encoding_addr}")
        
        # Get parity buffers ready for release
        parity_buffers = self._eccheck_native.get_parity_buffers_to_release()
        for parity_addr in parity_buffers:
            try:
                self._free_parity_buffer_queue.put_nowait(parity_addr)
                logger.debug(f"EC-CHECK: Released parity buffer at address {parity_addr}")
            except Exception:
                logger.error(f"EC-CHECK: Parity buffer queue is full, cannot release buffer {parity_addr}")
    
    def _start_buffer_poller_thread(self):
        """Start a persistent background thread to poll and release buffers."""
        if hasattr(self, '_buffer_poller_thread') and self._buffer_poller_thread is not None:
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
                from time import sleep
                sleep(0.001)  # 1ms
            
            logger.info("EC-CHECK: Buffer poller thread stopping")
        
        # Start the daemon thread
        self._buffer_poller_thread = threading.Thread(target=buffer_poller_worker, daemon=True)
        self._buffer_poller_thread.start()
        logger.info("EC-CHECK: Buffer poller thread created and started")
    
    def _stop_buffer_poller_thread(self):
        """Stop the persistent buffer poller thread."""
        if not hasattr(self, '_buffer_poller_thread') or self._buffer_poller_thread is None:
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
    
    def get_eccheck_buffers(self):
        """Get EC-CHECK buffers for FileSystemWriterAsync.
        
        Note: Returns data, encoding, and parity buffers.
        Receive buffers will be allocated by FileSystemWriterAsync
        after metadata exchange.
        
        Returns:
            Dict containing all buffer information, or None if not initialized
        """
        if self.eccheck_data_buffers is None:
            return None
        
        return {
            'data_buffers': self.eccheck_data_buffers,
            'encoding_buffers': self.eccheck_encoding_buffers,
            'parity_buffers': self.eccheck_parity_buffers,
            'free_data_buffer_queue': self._free_data_buffer_queue,
            'free_encoding_buffer_queue': self._free_encoding_buffer_queue,
            'free_parity_buffer_queue': self._free_parity_buffer_queue,
            # Pass buffer poller control objects
            'buffer_poller_active_event': self._buffer_poller_active_event,
            'poll_and_release_buffers': self._poll_and_release_buffers,
            # Note: recv_encoding_buffers will be allocated by FileSystemWriterAsync after metadata exchange
        }
    
    def allocate_recv_encoding_buffers_phase2(self, global_registry: GlobalMetadataRegistry):
        """
        Allocate TWO large receive buffers for peer encoded packets (one per encoding thread).
        
        Each buffer is equal to peer's total data size, aligned to buffer_size (64MB).
        This is called after metadata exchange when peer data sizes are known.
        
        Args:
            global_registry (GlobalMetadataRegistry): Complete metadata from all ranks
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two receive buffers (one for thread1, one for thread2)
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        paired_rank = self._get_xor_paired_rank(rank, world_size)
        
        # Get peer's total data size from global registry (actual reference)
        peer_metadata = global_registry.rank_metadata.get(paired_rank, [])
        peer_total_size = sum(meta.size_bytes for meta in peer_metadata)
        
        # Calculate maximum total size across all ranks for pipeline synchronization
        max_total_size = 0
        for r in range(world_size):
            rank_metadata = global_registry.rank_metadata.get(r, [])
            rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
            if rank_total_size > max_total_size:
                max_total_size = rank_total_size
        
        # Align maximum size to buffer_size (64MB) so recv buffers match pipeline iterations
        aligned_size = ((max_total_size + self.eccheck_buffer_size - 1) // self.eccheck_buffer_size) * self.eccheck_buffer_size
        
        logger.info(
            f"EC-CHECK: Allocating TWO receive buffers using global maximum size\n"
            f"  Paired rank: {paired_rank}\n"
            f"  Peer data size: {peer_total_size / (1024**3):.2f} GB\n"
            f"  Pipeline max size: {max_total_size / (1024**3):.2f} GB\n"
            f"  Aligned buffer size (per buffer): {aligned_size / (1024**3):.2f} GB\n"
            f"  Total receive memory: {2 * aligned_size / (1024**3):.2f} GB"
        )
        
        # Allocate two large continuous buffers (one for each encoding thread)
        recv_buffer_thread1 = torch.empty(aligned_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
        recv_buffer_thread2 = torch.empty(aligned_size, dtype=torch.uint8, pin_memory=self.eccheck_pin_memory)
        
        logger.info(
            f"EC-CHECK: Allocated TWO receive buffers: {aligned_size / (1024**3):.2f} GB each "
            f"({aligned_size / (1024**2):.0f} MB each)"
        )
        
        self.eccheck_recv_encoding_buffers = (recv_buffer_thread1, recv_buffer_thread2)
        
        # Register receive buffers for RDMA if RDMA is enabled
        if self.use_rdma:
            logger.info(f"EC-CHECK: [Rank {rank}] Registering receive buffers for RDMA...")
            self.register_buffer(recv_buffer_thread1)
            self.register_buffer(recv_buffer_thread2)
        
        return (recv_buffer_thread1, recv_buffer_thread2)
    
    def register_buffer(self, buffer: torch.Tensor):
        """Register buffer for RDMA operations (called during buffer allocation).
        
        Args:
            buffer: PyTorch tensor to register
        """
        if not self.use_rdma or self._eccheck_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        buffer_size = buffer.numel() * buffer.element_size()
        
        # Check if already registered
        if buffer_addr in self.registered_buffers:
            logger.debug(f"EC-CHECK: [Rank {rank}] Buffer already registered at 0x{buffer_addr:x} (size: {buffer_size / (1024**2):.2f} MB)")
            return
        
        try:
            logger.info(f"EC-CHECK: [Rank {rank}] Registering buffer at 0x{buffer_addr:x}, size: {buffer_size / (1024**3):.2f} GB, numel: {buffer.numel()}, dtype: {buffer.dtype} (iteration {self.current_iteration})")
            self._eccheck_native.register_buffer(buffer_addr, buffer_size)
            self.registered_buffers[buffer_addr] = (buffer_size, self.current_iteration)
            logger.info(f"EC-CHECK: [Rank {rank}] Buffer registered successfully (total registered: {len(self.registered_buffers)})")
            
            # Print all registered buffers
            logger.info(f"EC-CHECK: [Rank {rank}] All registered buffers:")
            for addr, (size, iteration) in self.registered_buffers.items():
                logger.info(f"  - 0x{addr:x}: {size / (1024**2):.2f} MB (iteration {iteration})")
        except Exception as e:
            logger.error(f"EC-CHECK: [Rank {rank}] Failed to register buffer: {e}")
            raise
    
    def unregister_buffer(self, buffer: torch.Tensor):
        """Unregister buffer for RDMA operations.
        
        Args:
            buffer: PyTorch tensor to unregister
        """
        if not self.use_rdma or self._eccheck_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        
        if buffer_addr not in self.registered_buffers:
            logger.debug(f"EC-CHECK: [Rank {rank}] Buffer not registered at 0x{buffer_addr:x}")
            return
        
        try:
            logger.info(f"EC-CHECK: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x}")
            self._eccheck_native.unregister_buffer(buffer_addr)
            del self.registered_buffers[buffer_addr]
            logger.info(f"EC-CHECK: [Rank {rank}] Buffer unregistered successfully")
        except Exception as e:
            logger.error(f"EC-CHECK: [Rank {rank}] Failed to unregister buffer: {e}")
    
    def register_all_buffers_for_rdma(self):
        """Register all allocated buffers for RDMA operations.
        
        This should be called during initialization after all buffers are allocated.
        """
        if not self.use_rdma or self._eccheck_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(f"EC-CHECK: [Rank {rank}] Registering all buffers for RDMA...")
        
        # Register data buffers
        if self.eccheck_data_buffers:
            for i, buffer in enumerate(self.eccheck_data_buffers):
                self.register_buffer(buffer)
        
        # Register encoding buffers
        if self.eccheck_encoding_buffers:
            for i, buffer in enumerate(self.eccheck_encoding_buffers):
                self.register_buffer(buffer)
        
        # Register parity buffers
        if self.eccheck_parity_buffers:
            for i, buffer in enumerate(self.eccheck_parity_buffers):
                self.register_buffer(buffer)
        
        # Register receive encoding buffers (if already allocated)
        if self.eccheck_recv_encoding_buffers:
            recv_buffer_thread1, recv_buffer_thread2 = self.eccheck_recv_encoding_buffers
            self.register_buffer(recv_buffer_thread1)
            self.register_buffer(recv_buffer_thread2)
        
        logger.info(f"EC-CHECK: [Rank {rank}] All buffers registered for RDMA (total: {len(self.registered_buffers)})")
    
    
    def cleanup(self):
        """Cleanup EC-CHECK resources when manager is destroyed."""
        try:
            # Unregister all RDMA buffers
            if self.use_rdma and self._eccheck_native is not None:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                logger.info(f"EC-CHECK: [Rank {rank}] Unregistering all RDMA buffers...")
                for buffer_addr in list(self.registered_buffers.keys()):
                    try:
                        logger.info(f"EC-CHECK: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x} during cleanup")
                        self._eccheck_native.unregister_buffer(buffer_addr)
                    except Exception as e:
                        logger.warning(f"EC-CHECK: [Rank {rank}] Failed to unregister buffer during cleanup: {e}")
                self.registered_buffers.clear()
            
            # Stop buffer poller thread
            self._stop_buffer_poller_thread()
            
            # Stop the C++ pipeline
            if hasattr(self, '_eccheck_native') and self._eccheck_native is not None:
                self._eccheck_native.stop_pipeline()
                logger.info("EC-CHECK: C++ native module stopped in manager cleanup")
                
        except Exception as e:
            logger.warning(f"EC-CHECK: Error during manager cleanup: {e}")
    
    def __del__(self):
        """Cleanup EC-CHECK resources when manager is destroyed."""
        self.cleanup()

