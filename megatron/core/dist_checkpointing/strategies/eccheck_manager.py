# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""EC-CHECK manager for shared eccheck_native initialization and buffer management."""

import os
import queue
import socket
import threading
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import torch
from dataclasses import replace

from .hugepage_alloc import allocate_hugepage_slices, allocate_hugepage_tensor
from .state_dict_decomposer import GlobalMetadataRegistry, TensorMetadata
from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip

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
        data_buffers_count_value = os.environ.get("ECCHECK_DATA_BUFFERS_COUNT", "12")
        try:
            self.eccheck_data_buffers_count = int(data_buffers_count_value)
        except ValueError as exc:
            raise RuntimeError(
                "ECCHECK_DATA_BUFFERS_COUNT must be a positive integer, "
                f"got {data_buffers_count_value!r}"
            ) from exc
        if self.eccheck_data_buffers_count <= 0:
            raise RuntimeError(
                "ECCHECK_DATA_BUFFERS_COUNT must be a positive integer, "
                f"got {data_buffers_count_value!r}"
            )
        self.eccheck_encoding_buffers_count = self.eccheck_data_buffers_count * 2
        self.eccheck_buffer_size = 64 * 1024 * 1024  # 64MB
        self.eccheck_pin_memory = True
        
        # Buffers
        self.eccheck_data_buffers: Optional[List[torch.Tensor]] = None
        self.eccheck_encoding_buffers: Optional[List[torch.Tensor]] = None
        self.eccheck_recv_encoding_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.eccheck_hw1_recv_encoding_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.eccheck_hw2_recv_encoding_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.eccheck_hw2_recv_ring_depth: Optional[int] = None
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
        self.preallocated_cpu_buffer: Optional[torch.Tensor] = None

        # Legacy in-process recovery workspace. It is populated only while the
        # benchmark recovery cycle is active and intentionally lives until exit.
        self._legacy_inprocess_workspace_key: Optional[tuple] = None
        self._legacy_inprocess_workspace: Optional[Dict[str, Any]] = None

        self._initialized = True

    @property
    def is_two_failures_mode(self) -> bool:
        """Whether the current load is in two-failure hardware recovery mode."""
        try:
            from megatron.training import get_args
            return bool(getattr(get_args(), "use_eccheck_two_failures", False))
        except Exception:
            return False

    def allocate_preallocated_buffer(self, size_bytes: int):
        """Allocate or reuse cached CPU buffer.  Grows only when needed."""
        if self.preallocated_cpu_buffer is not None:
            if self.preallocated_cpu_buffer.numel() >= size_bytes:
                return
        pin = self.eccheck_pin_memory and torch.cuda.is_available()
        logger.debug(
            f"ECCHECK: Allocating preallocated buffer: {size_bytes / (1024**3):.2f} GB (pin={pin})"
        )
        self.preallocated_cpu_buffer = allocate_hugepage_tensor(
            size_bytes, fallback_pin_memory=pin, touch_pages=False,
        )

    # cache for persistent P2P blocks (own_buffer, partner_buffer)
    _cached_block_count: int = 0
    _cached_block_size: int = 0
    _cached_blocks: Optional[List[torch.Tensor]] = None

    def allocate_preallocated_blocks(self, count: int, aligned_size: int):
        """Allocate or reuse cached persistent blocks (e.g. own_buffer, partner_buffer)."""
        if (self._cached_blocks is not None and self._cached_block_count == count
                and self._cached_block_size >= aligned_size):
            return self._cached_blocks
        pin = self.eccheck_pin_memory and torch.cuda.is_available()
        logger.debug(
            f"ECCHECK: Allocating {count} blocks: {aligned_size / (1024**3):.2f} GB each "
            f"({count * aligned_size / (1024**3):.2f} GB total, pin={pin})"
        )
        self._cached_blocks = list(allocate_hugepage_slices(
            aligned_size, count, fallback_pin_memory=pin, touch_pages=True,
        ))
        self._cached_block_count = count
        self._cached_block_size = aligned_size
        if self.use_rdma:
            for b in self._cached_blocks:
                self.register_buffer(b)
        return self._cached_blocks

    @classmethod
    def _get_ranks_per_node(cls) -> int:
        """Detect ranks per node from CUDA_VISIBLE_DEVICES or torch.cuda."""
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible_devices:
            return len(cuda_visible_devices.split(','))
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return int(os.environ.get('LOCAL_WORLD_SIZE', '1'))

    @classmethod
    def _get_group_layout(cls, world_size: int) -> Dict[str, int]:
        """Return EC grouping layout: mode 1 (node-aware) or mode 0 (simple)."""
        n = RANKS_PER_GROUP
        if world_size <= 0:
            return {"mode": 0, "num_groups": 1, "ranks_per_node": 1,
                    "num_nodes": 1, "clusters": 1}
        num_groups = max(1, world_size // n)
        ranks_per_node = cls._get_ranks_per_node()
        if (world_size >= n and world_size % n == 0
                and ranks_per_node > 0
                and world_size % ranks_per_node == 0):
            num_nodes = world_size // ranks_per_node
            if num_nodes >= n and num_nodes % n == 0:
                clusters = num_nodes // n
                node_aware_groups = ranks_per_node * clusters
                if node_aware_groups == num_groups:
                    return {"mode": 1, "num_groups": num_groups,
                            "ranks_per_node": ranks_per_node,
                            "num_nodes": num_nodes, "clusters": clusters}
        return {"mode": 0, "num_groups": num_groups,
                "ranks_per_node": max(1, ranks_per_node),
                "num_nodes": max(1, world_size // max(1, ranks_per_node)),
                "clusters": 1}

    @classmethod
    def _get_cluster_id(cls, rank: int, world_size: int) -> int:
        """Get the four-node recovery cluster containing this rank."""
        layout = cls._get_group_layout(world_size)
        if layout["mode"] == 1:
            node_id = rank // layout["ranks_per_node"]
            return node_id % layout["clusters"]
        return cls._get_group_id(rank, world_size)

    @classmethod
    def _get_group_id(cls, rank: int, world_size: int) -> int:
        """Get group id for multi-rank (mode 0 or mode 1)."""
        layout = cls._get_group_layout(world_size)
        num_groups = layout["num_groups"]
        if layout["mode"] == 1:
            ranks_per_node = layout["ranks_per_node"]
            clusters = layout["clusters"]
            node_id = rank // ranks_per_node
            local_rank = rank % ranks_per_node
            cluster_id = node_id % clusters
            return local_rank * clusters + cluster_id
        return rank % num_groups

    @classmethod
    def _get_rig_remap_offset(cls) -> int:
        """Return the process-wide physical-to-logical rig rotation."""
        try:
            from megatron.training import get_args

            offset = int(getattr(get_args(), "eccheck_rig_remap_offset", 0))
        except (ImportError, RuntimeError, AssertionError):
            offset = int(os.environ.get("ECCHECK_RIG_REMAP_OFFSET", "0"))
        if offset < 0 or offset >= RANKS_PER_GROUP:
            raise ValueError(
                f"ECCHECK rig remap offset must be in [0, {RANKS_PER_GROUP - 1}], "
                f"got {offset}"
            )
        return offset

    @classmethod
    def _get_physical_rank_in_group(cls, rank: int, world_size: int) -> int:
        """Get the unrotated physical position within a four-rank group."""
        layout = cls._get_group_layout(world_size)
        num_groups = layout["num_groups"]
        if layout["mode"] == 1:
            node_id = rank // layout["ranks_per_node"]
            return node_id // layout["clusters"]
        return rank // num_groups

    @classmethod
    def _get_rank_in_group(cls, rank: int, world_size: int) -> int:
        """Map a global rank to its rotated logical ECCHECK rig position."""
        physical_rig = cls._get_physical_rank_in_group(rank, world_size)
        return (physical_rig + cls._get_rig_remap_offset()) % RANKS_PER_GROUP

    @classmethod
    def _get_rank_by_group_position(
        cls, group_id: int, rank_in_group: int, world_size: int
    ) -> int:
        """Map a logical group position back to its global rank."""
        layout = cls._get_group_layout(world_size)
        num_groups = layout["num_groups"]
        physical_rig = (
            rank_in_group - cls._get_rig_remap_offset()
        ) % RANKS_PER_GROUP
        if layout["mode"] == 1:
            ranks_per_node = layout["ranks_per_node"]
            clusters = layout["clusters"]
            local_rank = group_id // clusters
            cluster_id = group_id % clusters
            node_id = physical_rig * clusters + cluster_id
            return node_id * ranks_per_node + local_rank
        return group_id + num_groups * physical_rig

    def _get_xor_paired_rank(self, my_rank: int, world_size: int) -> int:
        """Get the paired rank for parity exchange (XOR pairing within group).

        Multi-rank: world_size must be divisible by 4. Each group of 4 ranks uses
        same pairing as original 4-rank: in-group 0<->2, 1<->3 (by position).
        """
        if world_size % RANKS_PER_GROUP != 0:
            raise ValueError(
                f"EC-CHECK: World size must be divisible by {RANKS_PER_GROUP} for multi-rank, got {world_size}"
            )
        group_id = self._get_group_id(my_rank, world_size)
        rank_in_group = self._get_rank_in_group(my_rank, world_size)
        # In-group XOR pairing: position 0<->2, 1<->3
        paired_rank_in_group = (rank_in_group + 2) % RANKS_PER_GROUP
        paired_rank = self._get_rank_by_group_position(
            group_id, paired_rank_in_group, world_size
        )
        logger.debug(
            f"EC-CHECK: Rank {my_rank} (group_id={group_id}, rank_in_group={rank_in_group}) "
            f"XOR paired with Rank {paired_rank}"
        )
        return paired_rank

    def get_p2p_partner_rank(self, my_rank: int, world_size: int) -> int:
        """Get P2P partner rank for data/parity exchange.

        Group-based pairing, same grouping as XOR:
        - rank_in_group 0 <-> 1 (P2P pair)
        - rank_in_group 2 <-> 3 (P2P pair)
        """
        if world_size % 2 != 0:
            raise ValueError(f"EC-CHECK: World size must be even for P2P pairing, got {world_size}")
        group_id = self._get_group_id(my_rank, world_size)
        rank_in_group = self._get_rank_in_group(my_rank, world_size)
        if rank_in_group % 2 == 0:
            partner_rig = rank_in_group + 1
        else:
            partner_rig = rank_in_group - 1
        p2p_partner_rank = self._get_rank_by_group_position(
            group_id, partner_rig, world_size
        )
        logger.debug(
            f"EC-CHECK: Rank {my_rank} (group_id={group_id}, rank_in_group={rank_in_group}) "
            f"P2P partner is Rank {p2p_partner_rank}"
        )
        return p2p_partner_rank

    def get_recovery_partner_rank_for_rank1_software(self, my_rank: int, world_size: int) -> int:
        """Get recovery partner for rank_in_group=1 software failure.

        Within the same EC group, rank_in_group 0 sends to rank_in_group 1.
        Returns -1 for rank_in_group 2/3 (not participating).
        """
        if world_size < 4 or world_size % 4 != 0:
            raise ValueError(
                f"EC-CHECK: world_size must be >=4 and divisible by 4 for recovery, got {world_size}"
            )
        group_id = self._get_group_id(my_rank, world_size)
        rank_in_group = self._get_rank_in_group(my_rank, world_size)
        if rank_in_group == 0:
            return self._get_rank_by_group_position(group_id, 1, world_size)
        if rank_in_group == 1:
            return self._get_rank_by_group_position(group_id, 0, world_size)
        return -1

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
        # Step 1: Get base IP address (with multi-NIC per-rank support)
        base_ip = resolve_ip("ECCHECK", rank=rank)

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
                # base_ip already resolved via resolve_ip() which handles
                # ECCHECK_RANK_IP_{rank}, ECCHECK_BASE_IP, ECCHECK_INTERFACE, etc.
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
                rank_ips = {}
                for r, ip_tensor in enumerate(ip_list):
                    ip_bytes = bytes(ip_tensor.cpu().tolist())
                    rank_ips[r] = socket.inet_ntoa(ip_bytes)
                
                logger.debug(f"EC-CHECK: [Rank {rank}] All ranks IPs: {rank_ips}")

                # Get partner IPs from gathered results
                xor_partner_ip = rank_ips.get(xor_partner, base_ip)
                p2p_partner_ip = rank_ips.get(p2p_partner, base_ip)

                logger.debug(
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
            logger.debug("EC-CHECK: Distributed not initialized, using local IP for all partners")
            rank_ips = {r: base_ip for r in range(world_size)}

        group_id = self._get_group_id(rank, world_size)
        rank_in_group = self._get_rank_in_group(rank, world_size)

        config = {
            'my_ip': base_ip,
            'base_port': base_port,
            'xor_partner_ip': xor_partner_ip,
            'p2p_partner_ip': p2p_partner_ip,
            'ports': ports,
            'rank_ips': rank_ips,
            'rank_in_group': rank_in_group,
            'group_id': group_id,
        }

        logger.debug(
            f"EC-CHECK: [Rank {rank}] Network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  XOR partner IP: {config['xor_partner_ip']}\n"
            f"  P2P partner IP: {config['p2p_partner_ip']}\n"
            f"  Group: {group_id}, Rank in group: {rank_in_group}\n"
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
                    logger.debug(f"EC-CHECK: [Rank {rank}] Using {transport_mode} for communication")
                    
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
                    logger.debug(f"EC-CHECK: [Rank {rank}] Synchronizing all ranks before creating C++ native module ({transport_mode})...")
                    torch.distributed.barrier()
                    logger.debug(f"EC-CHECK: [Rank {rank}] All ranks synchronized, creating C++ native module with {transport_mode}...")
                    
                    # Create C++ instance with ASIO/RDMA parameters
                    logger.debug(f"EC-CHECK: Creating C++ native module with {transport_mode} (this will block until connections are established)...")
                    
                    # Calculate partner ports (send connects to partner's recv port)
                    # For XOR: rank 0 sends to rank 2's recv port, rank 2 sends to rank 0's recv port
                    # For P2P: rank 0 sends to rank 1's recv port, rank 1 sends to rank 0's recv port
                    xor_partner = self._get_xor_paired_rank(rank, world_size)
                    p2p_partner = self.get_p2p_partner_rank(rank, world_size)
                    # rank_in_group=1 software failure: use recovery partner (same EC group 0<->1) so 8-rank gets (0->2),(1->3); 4-rank unchanged
                    try:
                        from megatron.training import get_args as _get_args
                        _args = _get_args()
                        if getattr(_args, 'use_eccheck_software_failure', False):
                            recovery_partner = self.get_recovery_partner_rank_for_rank1_software(
                                rank, world_size
                            )
                            if recovery_partner >= 0:
                                p2p_partner = recovery_partner
                                rank_ips = net_config.get('rank_ips', {})
                                net_config['p2p_partner_ip'] = rank_ips.get(
                                    p2p_partner, net_config['my_ip']
                                )
                    except Exception:
                        pass

                    # Partner's recv ports (where we send to)
                    base_port = net_config['base_port']
                    xor_partner_recv_port = base_port + xor_partner * 6 + 1  # partner's xor_recv port
                    p2p_partner_recv_port = base_port + p2p_partner * 6 + 3  # partner's p2p_recv port
                    
                    # Step6 P2P: rank_in_group 3 sends to rank_in_group 2 in same group
                    # rank_in_group 2 listens on step6_p2p_recv port (multi-rank: per-group)
                    group_id = self._get_group_id(rank, world_size)
                    rank_in_group = self._get_rank_in_group(rank, world_size)
                    rank_ips = net_config.get('rank_ips', {})
                    if rank_in_group == 3:
                        step6_p2p_partner_rank = self._get_rank_by_group_position(
                            group_id, 2, world_size
                        )
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
                    
                    p2p_partner_rank = self.get_p2p_partner_rank(rank, world_size)
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
                        p2p_partner_rank,
                    )
                    
                    # If we reach here, ASIO/RDMA connections are ready and threads are running
                    logger.debug(f"EC-CHECK: C++ native module initialized successfully with {transport_mode} (rank={rank}, world_size={world_size}, paired_rank={paired_rank})")
                    
                    # Initialize EC-CHECK buffers (same for both ASIO and NCCL)
                    self._init_eccheck_buffers()
                    
                else:
                    # ===== NCCL Initialization Path (original) =====
                    logger.debug(f"EC-CHECK: [Rank {rank}] Using NCCL for communication")
                    
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
                        logger.debug(f"EC-CHECK: [Rank 0] Generated four NCCL IDs (size: {len(nccl_id_thread1)} bytes each)")
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
                    
                    logger.debug(f"EC-CHECK: [Rank {rank}] Received four NCCL IDs via broadcast")
                    
                    # ===== Step 3: Synchronize all ranks before creating C++ instances =====
                    # This barrier ensures all ranks start creating C++ instances at roughly the same time,
                    # which helps synchronize the NCCL communicator initialization calls.
                    logger.debug(f"EC-CHECK: [Rank {rank}] Synchronizing all ranks before creating C++ native module...")
                    torch.distributed.barrier()
                    logger.debug(f"EC-CHECK: [Rank {rank}] All ranks synchronized, creating C++ native module...")
                    
                    # ===== Step 4: Create C++ instance with broadcasted IDs =====
                    # IMPORTANT: This constructor call will BLOCK until:
                    # 1. Send and recv threads are started
                    # 2. All NCCL communicators are fully initialized using the broadcasted IDs
                    # 3. All threads are ready for data exchange
                    # Only after all initialization is complete will this call return.
                    logger.debug(f"EC-CHECK: Creating C++ native module (this will block until NCCL is initialized)...")
                    
                    rank_in_group = self._get_rank_in_group(rank, world_size)
                    p2p_partner_rank = self.get_p2p_partner_rank(rank, world_size)
                    self._eccheck_native = eccheck_native.ECCHECKNative(
                        rank, world_size, paired_rank,
                        nccl_id_thread1,    # rank0↔rank2 XOR
                        nccl_id_thread2,    # rank1↔rank3 XOR
                        nccl_id_p2p_0_1,   # rank0↔rank1 P2P
                        nccl_id_p2p_2_3,   # rank2↔rank3 P2P
                        rank_in_group,
                        p2p_partner_rank,
                    )
                    
                    # If we reach here, NCCL communicators are ready and threads are running
                    logger.debug(f"EC-CHECK: C++ native module initialized successfully (rank={rank}, world_size={world_size}, paired_rank={paired_rank})")
                    
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
        logger.debug("EC-CHECK: Initializing buffers for EC-CHECK (data and encoding only)")
        
        # Allocate data buffers for storing original tensor data
        self.eccheck_data_buffers = self._allocate_data_buffers()
        
        # Allocate encoding buffers for encoded packets
        self.eccheck_encoding_buffers = self._allocate_encoding_buffers()
        
        # Allocate receive buffers for peer encoded packets (will be allocated later)
        self.eccheck_recv_encoding_buffers = None
        self.eccheck_hw1_recv_encoding_buffers = None
        self.eccheck_hw2_recv_encoding_buffers = None
        self.eccheck_hw2_recv_ring_depth = None
        
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

        logger.debug(f"EC-CHECK: Buffer initialization completed - "
                   f"Data buffers: {len(self.eccheck_data_buffers)}, "
                   f"Encoding buffers: {len(self.eccheck_encoding_buffers)}, "
                   f"Parity buffers: {len(self.eccheck_parity_buffers)}")
        
        # Register all buffers for RDMA if RDMA is enabled
        if self.use_rdma:
            self.register_all_buffers_for_rdma()
    
    def _allocate_data_buffers(self):
        """Allocate data buffers for storing original tensor data."""
        logger.debug(f"EC-CHECK: Allocating data buffers ({self.eccheck_data_buffers_count} buffers, {self.eccheck_buffer_size // (1024*1024)}MB each)")
        
        data_buffers = []
        for i in range(self.eccheck_data_buffers_count):
            buffer = allocate_hugepage_tensor(
                self.eccheck_buffer_size,
                fallback_pin_memory=self.eccheck_pin_memory,
                touch_pages=True,
            )
            data_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated data buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.debug(f"EC-CHECK: Allocated {len(data_buffers)} data buffers")
        return data_buffers
    
    def _allocate_encoding_buffers(self):
        """Allocate encoding buffers for encoded packets."""
        logger.debug(f"EC-CHECK: Allocating encoding buffers ({self.eccheck_encoding_buffers_count} buffers, {self.eccheck_buffer_size // (1024*1024)}MB each)")
        
        encoding_buffers = []
        for i in range(self.eccheck_encoding_buffers_count):
            buffer = allocate_hugepage_tensor(
                self.eccheck_buffer_size,
                fallback_pin_memory=self.eccheck_pin_memory,
                touch_pages=True,
            )
            encoding_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated encoding buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.debug(f"EC-CHECK: Allocated {len(encoding_buffers)} encoding buffers")
        return encoding_buffers
    
    def _allocate_parity_buffers(self):
        """Allocate parity buffers for XOR computation results.
        
        Note: Parity buffer count should match encoding buffer count to support
        pipelined operations where each data chunk needs 2 parity buffers (one per thread).
        """
        # Use encoding buffer count instead of data buffer count
        # Each data chunk needs 2 parity buffers (thread1 and thread2)
        parity_buffer_count = self.eccheck_encoding_buffers_count
        logger.debug(f"EC-CHECK: Allocating parity buffers ({parity_buffer_count} buffers)")
        
        parity_buffers = []
        for i in range(parity_buffer_count):
            buffer = allocate_hugepage_tensor(
                self.eccheck_buffer_size,
                fallback_pin_memory=self.eccheck_pin_memory,
                touch_pages=True,
            )
            parity_buffers.append(buffer)
            logger.debug(f"EC-CHECK: Allocated parity buffer {i}: {self.eccheck_buffer_size} bytes")
        
        logger.debug(f"EC-CHECK: Allocated {len(parity_buffers)} parity buffers")
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
            logger.debug("EC-CHECK: Buffer poller thread started")
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
            
            logger.debug("EC-CHECK: Buffer poller thread stopping")
        
        # Start the daemon thread
        self._buffer_poller_thread = threading.Thread(target=buffer_poller_worker, daemon=True)
        self._buffer_poller_thread.start()
        logger.debug("EC-CHECK: Buffer poller thread created and started")
    
    def _stop_buffer_poller_thread(self):
        """Stop the persistent buffer poller thread."""
        if not hasattr(self, '_buffer_poller_thread') or self._buffer_poller_thread is None:
            return
        
        logger.debug("EC-CHECK: Stopping buffer poller thread...")
        
        # Signal the thread to stop
        if self._buffer_poller_stop_event:
            self._buffer_poller_stop_event.set()
        
        # Wait for thread to finish
        if self._buffer_poller_thread.is_alive():
            self._buffer_poller_thread.join(timeout=2.0)
            if self._buffer_poller_thread.is_alive():
                logger.warning("EC-CHECK: Buffer poller thread did not stop in time")
            else:
                logger.debug("EC-CHECK: Buffer poller thread stopped successfully")
        
        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None
    
    def find_legacy_inprocess_workspace(
        self, bootstrap_key: tuple
    ) -> Optional[Dict[str, Any]]:
        """Find a workspace before metadata-derived key dimensions are available."""
        if self._legacy_inprocess_workspace_key is None:
            return None
        if self._legacy_inprocess_workspace_key[0] != bootstrap_key:
            if self._eccheck_native is not None:
                raise RuntimeError(
                    "ECCHECK in-process recovery workspace inputs changed while native "
                    "buffers are registered; restart the process before using the new "
                    "checkpoint or recovery configuration"
                )
            self._legacy_inprocess_workspace_key = None
            self._legacy_inprocess_workspace = None
            return None
        return self._legacy_inprocess_workspace

    def get_legacy_inprocess_workspace(self, key: tuple) -> Optional[Dict[str, Any]]:
        """Return the exact-match legacy recovery workspace, if installed."""
        if self._legacy_inprocess_workspace_key is None:
            return None
        if self._legacy_inprocess_workspace_key != key:
            if self._eccheck_native is not None:
                raise RuntimeError(
                    "ECCHECK in-process recovery workspace inputs changed while native "
                    "buffers are registered; restart the process before using the new "
                    "checkpoint or recovery configuration"
                )
            self._legacy_inprocess_workspace_key = None
            self._legacy_inprocess_workspace = None
            return None
        return self._legacy_inprocess_workspace

    def install_legacy_inprocess_workspace(
        self, key: tuple, workspace: Dict[str, Any]
    ) -> None:
        """Install the manager-owned legacy recovery workspace once."""
        existing = self.get_legacy_inprocess_workspace(key)
        if existing is not None and existing is not workspace:
            raise RuntimeError("ECCHECK in-process recovery workspace was replaced unexpectedly")
        self._legacy_inprocess_workspace_key = key
        self._legacy_inprocess_workspace = workspace

    def reset_native_for_inprocess_recovery(self, failed_rank: int) -> None:
        """Restart load workers, then clear all native and Python cycle state."""
        if self._eccheck_native is None:
            raise RuntimeError("ECCHECK native module is not initialized")
        # set_load_mode joins workers completed by the previous sentinel batch and
        # starts a fresh worker set. Resetting before this call would clear the
        # completion flags needed to join those workers.
        self._eccheck_native.set_load_mode(True, failed_rank)
        self._eccheck_native.reset_encoding_completion_flags()
        self.reset_free_buffer_queues_for_inprocess_recovery()

    def reset_free_buffer_queues_for_inprocess_recovery(self):
        """Reset reusable free-buffer queues between in-process recovery cycles."""
        if self.eccheck_data_buffers is None:
            return

        # Drain stale release notifications from the previous recovery cycle.
        if self._eccheck_native is not None:
            try:
                self._eccheck_native.get_data_buffers_to_release()
                self._eccheck_native.get_encoding_buffers_to_release()
                self._eccheck_native.get_parity_buffers_to_release()
            except Exception as exc:
                logger.debug("EC-CHECK: ignored stale release queue drain failure: %s", exc)

        self._free_data_buffer_queue = queue.Queue()
        for buffer in self.eccheck_data_buffers or []:
            self._free_data_buffer_queue.put(int(buffer.data_ptr()))

        self._free_encoding_buffer_queue = queue.Queue()
        for buffer in self.eccheck_encoding_buffers or []:
            self._free_encoding_buffer_queue.put(int(buffer.data_ptr()))

        self._free_parity_buffer_queue = queue.Queue()
        for buffer in self.eccheck_parity_buffers or []:
            self._free_parity_buffer_queue.put(int(buffer.data_ptr()))

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
    
    def allocate_recv_encoding_buffers_phase2(
        self,
        global_registry: GlobalMetadataRegistry,
        *,
        hw1_single_physical_buffer: bool = False,
        hw2_single_physical_buffer: bool = False,
        hw2_ring_depth: Optional[int] = None,
    ):
        """
        Allocate logical receive buffers for peer encoded packets.

        HW1 and HW2 each have one receiving encoding lane per role, so their two
        logical lanes can safely alias one physical buffer. HW2 uses a bounded chunk
        ring; HW1 uses full aligned storage. Save and other modes retain two full
        physical buffers. The modes use independent caches.
        This is called after metadata exchange when peer data sizes are known.
        
        Args:
            global_registry (GlobalMetadataRegistry): Complete metadata from all ranks
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two receive buffers (one for thread1, one for thread2)
        """
        if hw1_single_physical_buffer and hw2_single_physical_buffer:
            raise ValueError("HW1 and HW2 receive-buffer modes are mutually exclusive")
        if hw2_ring_depth is not None and not hw2_single_physical_buffer:
            raise ValueError("hw2_ring_depth requires hw2_single_physical_buffer=True")
        if hw2_single_physical_buffer:
            if isinstance(hw2_ring_depth, bool) or not isinstance(hw2_ring_depth, int) or hw2_ring_depth <= 0:
                raise ValueError("hw2_ring_depth must be a positive integer for HW2")
            if self.eccheck_hw2_recv_encoding_buffers is not None:
                if self.eccheck_hw2_recv_ring_depth != hw2_ring_depth:
                    raise RuntimeError(
                        "ECCHECK HW2 receive ring depth changed after allocation: "
                        f"allocated={self.eccheck_hw2_recv_ring_depth}, requested={hw2_ring_depth}"
                    )
                return self.eccheck_hw2_recv_encoding_buffers
        elif hw1_single_physical_buffer and self.eccheck_hw1_recv_encoding_buffers is not None:
            return self.eccheck_hw1_recv_encoding_buffers
        elif not hw1_single_physical_buffer and self.eccheck_recv_encoding_buffers is not None:
            return self.eccheck_recv_encoding_buffers

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        paired_rank = self._get_xor_paired_rank(rank, world_size)
        
        # Get peer's total data size from global registry (actual reference)
        peer_metadata = global_registry.rank_metadata.get(paired_rank, [])
        peer_total_size = sum(meta.size_bytes for meta in peer_metadata)
        
        # HW2 uses a bounded chunk ring; HW1/save retain full aligned receive storage.
        max_total_size = 0
        for r in range(world_size):
            rank_metadata = global_registry.rank_metadata.get(r, [])
            rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
            if rank_total_size > max_total_size:
                max_total_size = rank_total_size
        full_aligned_size = max(
            1, (max_total_size + self.eccheck_buffer_size - 1) // self.eccheck_buffer_size
        ) * self.eccheck_buffer_size
        aligned_size = (
            hw2_ring_depth * self.eccheck_buffer_size
            if hw2_single_physical_buffer else full_aligned_size
        )

        single_physical_buffer = hw1_single_physical_buffer or hw2_single_physical_buffer
        physical_buffer_count = 1 if single_physical_buffer else 2
        logger.debug(
            "EC-CHECK: Allocating receive buffers using global maximum size\n"
            "  Paired rank: %d\n"
            "  Peer data size: %.2f GB\n"
            "  Pipeline max size: %.2f GB\n"
            "  Aligned buffer size (per buffer): %.2f GB\n"
            "  Physical buffers: %d\n"
            "  Total receive memory: %.2f GB",
            paired_rank,
            peer_total_size / (1024**3),
            max_total_size / (1024**3),
            aligned_size / (1024**3),
            physical_buffer_count,
            physical_buffer_count * aligned_size / (1024**3),
        )

        physical_buffers = allocate_hugepage_slices(
            aligned_size,
            physical_buffer_count,
            fallback_pin_memory=self.eccheck_pin_memory,
            touch_pages=True,
        )
        recv_buffer_thread1 = physical_buffers[0]
        recv_buffer_thread2 = physical_buffers[-1]

        if single_physical_buffer:
            mode = "HW1" if hw1_single_physical_buffer else "HW2"
            logger.debug(
                "EC-CHECK receive layout: mode=%s physical=1 logical=2 "
                "aligned_size=%d alias=lane1:lane2",
                mode,
                aligned_size,
            )
        else:
            logger.debug(
                f"EC-CHECK: Allocated TWO receive buffers: {aligned_size / (1024**3):.2f} GB each "
                f"({aligned_size / (1024**2):.0f} MB each)"
            )

        recv_buffers = (recv_buffer_thread1, recv_buffer_thread2)
        if hw1_single_physical_buffer:
            self.eccheck_hw1_recv_encoding_buffers = recv_buffers
        elif hw2_single_physical_buffer:
            self.eccheck_hw2_recv_encoding_buffers = recv_buffers
            self.eccheck_hw2_recv_ring_depth = hw2_ring_depth
        else:
            self.eccheck_recv_encoding_buffers = recv_buffers

        # Register each physical receive buffer once when RDMA is enabled.
        if self.use_rdma:
            logger.debug(f"EC-CHECK: [Rank {rank}] Registering receive buffers for RDMA...")
            for buffer in physical_buffers:
                self.register_buffer(buffer)

        return recv_buffers
    
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
        
        # Check if already registered. Re-register when the same address needs a larger coverage.
        if buffer_addr in self.registered_buffers:
            registered_size, _ = self.registered_buffers[buffer_addr]
            if registered_size >= buffer_size:
                logger.debug(
                    f"EC-CHECK: [Rank {rank}] Buffer already registered at 0x{buffer_addr:x} "
                    f"(registered={registered_size / (1024**2):.2f} MB, requested={buffer_size / (1024**2):.2f} MB)"
                )
                return
            logger.debug(
                f"EC-CHECK: [Rank {rank}] Re-registering buffer at 0x{buffer_addr:x} "
                f"to expand coverage from {registered_size / (1024**2):.2f} MB "
                f"to {buffer_size / (1024**2):.2f} MB"
            )
            try:
                self._eccheck_native.unregister_buffer(buffer_addr)
            except Exception as e:
                logger.warning(
                    f"EC-CHECK: [Rank {rank}] Failed to unregister old MR at 0x{buffer_addr:x} "
                    f"before re-register: {e}"
                )
            self.registered_buffers.pop(buffer_addr, None)
        
        try:
            logger.debug(f"EC-CHECK: [Rank {rank}] Registering buffer at 0x{buffer_addr:x}, size: {buffer_size / (1024**3):.2f} GB, numel: {buffer.numel()}, dtype: {buffer.dtype} (iteration {self.current_iteration})")
            self._eccheck_native.register_buffer(buffer_addr, buffer_size)
            self.registered_buffers[buffer_addr] = (buffer_size, self.current_iteration)
            logger.debug(f"EC-CHECK: [Rank {rank}] Buffer registered successfully (total registered: {len(self.registered_buffers)})")
            
            # Print all registered buffers
            logger.debug(f"EC-CHECK: [Rank {rank}] All registered buffers:")
            # for addr, (size, iteration) in self.registered_buffers.items():
            #     logger.info(f"  - 0x{addr:x}: {size / (1024**2):.2f} MB (iteration {iteration})")
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
            # logger.info(f"EC-CHECK: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x}")
            self._eccheck_native.unregister_buffer(buffer_addr)
            del self.registered_buffers[buffer_addr]
            # logger.info(f"EC-CHECK: [Rank {rank}] Buffer unregistered successfully")
        except Exception as e:
            logger.error(f"EC-CHECK: [Rank {rank}] Failed to unregister buffer: {e}")
    
    def register_all_buffers_for_rdma(self):
        """Register all allocated buffers for RDMA operations.
        
        This should be called during initialization after all buffers are allocated.
        """
        if not self.use_rdma or self._eccheck_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.debug(f"EC-CHECK: [Rank {rank}] Registering all buffers for RDMA...")
        
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
        
        # Register each physical receive buffer once across all mode caches.
        registered_recv_ptrs = set()
        for recv_buffers in (
            self.eccheck_recv_encoding_buffers,
            self.eccheck_hw1_recv_encoding_buffers,
            self.eccheck_hw2_recv_encoding_buffers,
        ):
            if not recv_buffers:
                continue
            for buffer in recv_buffers:
                buffer_ptr = int(buffer.data_ptr())
                if buffer_ptr in registered_recv_ptrs:
                    continue
                registered_recv_ptrs.add(buffer_ptr)
                self.register_buffer(buffer)
        
        logger.debug(f"EC-CHECK: [Rank {rank}] All buffers registered for RDMA (total: {len(self.registered_buffers)})")
    
    
    def cleanup(self):
        """Cleanup EC-CHECK resources when manager is destroyed."""
        try:
            from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
                release_hugepage_host_registration,
            )

            released_host_ptrs = set()

            def _release_host_registrations_for_buffers(buffers) -> None:
                if not buffers:
                    return
                for buffer in buffers:
                    if not torch.is_tensor(buffer):
                        continue
                    buffer_ptr = int(buffer.data_ptr())
                    if buffer_ptr in released_host_ptrs:
                        continue
                    released_host_ptrs.add(buffer_ptr)
                    release_hugepage_host_registration(buffer)

            # Unregister all RDMA buffers
            if self.use_rdma and self._eccheck_native is not None:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                logger.debug(f"EC-CHECK: [Rank {rank}] Unregistering all RDMA buffers...")
                for buffer_addr in list(self.registered_buffers.keys()):
                    try:
                        # logger.info(f"EC-CHECK: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x} during cleanup")
                        self._eccheck_native.unregister_buffer(buffer_addr)
                    except Exception as e:
                        logger.warning(f"EC-CHECK: [Rank {rank}] Failed to unregister buffer during cleanup: {e}")
                self.registered_buffers.clear()

            _release_host_registrations_for_buffers(self._cached_blocks)
            if self.preallocated_cpu_buffer is not None:
                _release_host_registrations_for_buffers([self.preallocated_cpu_buffer])
            _release_host_registrations_for_buffers(self.eccheck_data_buffers)
            _release_host_registrations_for_buffers(self.eccheck_encoding_buffers)
            _release_host_registrations_for_buffers(self.eccheck_parity_buffers)
            if self.eccheck_recv_encoding_buffers:
                _release_host_registrations_for_buffers(self.eccheck_recv_encoding_buffers)
            if self.eccheck_hw1_recv_encoding_buffers:
                _release_host_registrations_for_buffers(self.eccheck_hw1_recv_encoding_buffers)
            if self.eccheck_hw2_recv_encoding_buffers:
                _release_host_registrations_for_buffers(self.eccheck_hw2_recv_encoding_buffers)
            
            # Stop buffer poller thread
            self._stop_buffer_poller_thread()
            
            # Stop the C++ pipeline
            if hasattr(self, '_eccheck_native') and self._eccheck_native is not None:
                self._eccheck_native.stop_pipeline()
                logger.debug("EC-CHECK: C++ native module stopped in manager cleanup")

            # Release cached allocations
            self.preallocated_cpu_buffer = None
            self._legacy_inprocess_workspace_key = None
            self._legacy_inprocess_workspace = None
            self._cached_blocks = None
            self._cached_block_count = 0
            self._cached_block_size = 0
            self.eccheck_data_buffers = None
            self.eccheck_encoding_buffers = None
            self.eccheck_parity_buffers = None
            self.eccheck_recv_encoding_buffers = None
            self.eccheck_hw1_recv_encoding_buffers = None
            self.eccheck_hw2_recv_encoding_buffers = None
            self.eccheck_hw2_recv_ring_depth = None
            self._free_data_buffer_queue = None
            self._free_encoding_buffer_queue = None
            self._free_parity_buffer_queue = None

        except Exception as e:
            logger.warning(f"EC-CHECK: Error during manager cleanup: {e}")
    
    def __del__(self):
        """Cleanup EC-CHECK resources when manager is destroyed."""
        self.cleanup()

