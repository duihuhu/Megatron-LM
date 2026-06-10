# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""CheckCode manager (file renamed from eclatin_manager.py).

Partial rename: loads checkcode_native*.so (falls back to eclatin_native*.so).
Internal class/attr names (ECLATINManager, _eclatin_native, ECLATIN_* env vars)
are unchanged until a full on-disk format migration.
"""

import os
import queue
import threading
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
from dataclasses import replace

from .hugepage_alloc import allocate_hugepage_slices, allocate_hugepage_tensor
from .state_dict_decomposer import GlobalMetadataRegistry, TensorMetadata
from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip

logger = getLogger(__name__)

# Number of ranks per EC group (each group behaves like the original 4-rank setup)
RANKS_PER_GROUP = 4


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
        self.use_rdma = False  # RDMA support flag

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

        # Track registered buffers (for RDMA)
        self.registered_buffers: Dict[int, Tuple[int, int]] = {}  # {buffer_addr: (size, iteration)}
        self.current_iteration: int = 0
        self.preallocated_cpu_buffer: Optional[torch.Tensor] = None

        self._initialized = True

    def allocate_preallocated_buffer(self, size_bytes: int):
        """Allocate or reuse cached CPU buffer.  Grows only when needed."""
        if self.preallocated_cpu_buffer is not None:
            if self.preallocated_cpu_buffer.numel() >= size_bytes:
                return
        pin = self.eclatin_pin_memory and torch.cuda.is_available()
        logger.info(
            f"ECLATIN: Allocating preallocated buffer: {size_bytes / (1024**3):.2f} GB (pin={pin})"
        )
        self.preallocated_cpu_buffer = allocate_hugepage_tensor(
            size_bytes, fallback_pin_memory=pin, touch_pages=False,
        )

    _cached_block_count: int = 0
    _cached_block_size: int = 0
    _cached_blocks: Optional[List[torch.Tensor]] = None

    def allocate_preallocated_blocks(self, count: int, aligned_size: int):
        """Allocate or reuse cached persistent blocks (4 blocks)."""
        if (self._cached_blocks is not None and self._cached_block_count == count
                and self._cached_block_size >= aligned_size):
            return self._cached_blocks
        pin = self.eclatin_pin_memory and torch.cuda.is_available()
        logger.info(
            f"ECLATIN: Allocating {count} blocks: {aligned_size / (1024**3):.2f} GB each "
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

    # ---- Group layout methods (identical pattern to ECNAIVE) ----

    @staticmethod
    def _get_ranks_per_node() -> int:
        """Best-effort infer per-node rank count from environment."""
        env_keys = (
            "LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE",
            "MPI_LOCALNRANKS", "MV2_COMM_WORLD_LOCAL_SIZE",
        )
        for key in env_keys:
            val = os.environ.get(key)
            if not val:
                continue
            try:
                parsed = int(val.strip())
                if parsed > 0:
                    return parsed
            except Exception:
                continue
        slurm_val = os.environ.get("SLURM_NTASKS_PER_NODE")
        if slurm_val:
            token = slurm_val.split(",")[0].strip()
            token = token.split("(")[0].strip()
            if token:
                try:
                    parsed = int(token)
                    if parsed > 0:
                        return parsed
                except Exception:
                    pass
        cuda_count = torch.cuda.device_count()
        return max(1, cuda_count)

    @classmethod
    def _get_group_layout(cls, world_size: int) -> Dict[str, int]:
        """Return EC grouping layout: mode 1 (node-aware) or mode 0 (simple)."""
        n = RANKS_PER_GROUP
        if world_size <= 0:
            return {
                "mode": 0, "num_groups": 1, "ranks_per_node": 1,
                "num_nodes": 1, "clusters": 1,
            }
        num_groups = max(1, world_size // n)
        ranks_per_node = cls._get_ranks_per_node()
        if (
            world_size >= n
            and world_size % n == 0
            and ranks_per_node > 0
            and world_size % ranks_per_node == 0
        ):
            num_nodes = world_size // ranks_per_node
            if num_nodes >= n and num_nodes % n == 0:
                clusters = num_nodes // n
                node_aware_groups = ranks_per_node * clusters
                if node_aware_groups == num_groups:
                    return {
                        "mode": 1, "num_groups": num_groups,
                        "ranks_per_node": ranks_per_node,
                        "num_nodes": num_nodes, "clusters": clusters,
                    }
        return {
            "mode": 0, "num_groups": num_groups,
            "ranks_per_node": max(1, ranks_per_node),
            "num_nodes": max(1, world_size // max(1, ranks_per_node)),
            "clusters": 1,
        }

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
    def _get_rank_in_group(cls, rank: int, world_size: int) -> int:
        """Get rank index within group (0..3), mode 0 or mode 1."""
        layout = cls._get_group_layout(world_size)
        num_groups = layout["num_groups"]
        if layout["mode"] == 1:
            ranks_per_node = layout["ranks_per_node"]
            clusters = layout["clusters"]
            node_id = rank // ranks_per_node
            return node_id // clusters
        return rank // num_groups

    @classmethod
    def _get_rank_by_group_position(
        cls, group_id: int, rank_in_group: int, world_size: int
    ) -> int:
        """Map (group_id, rank_in_group) -> global rank."""
        layout = cls._get_group_layout(world_size)
        num_groups = layout["num_groups"]
        if layout["mode"] == 1:
            ranks_per_node = layout["ranks_per_node"]
            clusters = layout["clusters"]
            local_rank = group_id // clusters
            cluster_id = group_id % clusters
            node_id = rank_in_group * clusters + cluster_id
            return node_id * ranks_per_node + local_rank
        return group_id + num_groups * rank_in_group

    def _get_xor_paired_rank(self, my_rank: int, world_size: int) -> int:
        """Get the paired rank for parity exchange (XOR pairing within group).

        Multi-rank: world_size must be divisible by 4. Each group of 4 ranks uses
        same pairing as original 4-rank: in-group 0<->2, 1<->3 (by position).
        E.g. 8 ranks: group0={0,2,4,6} -> 0<->4, 2<->6; group1={1,3,5,7} -> 1<->5, 3<->7.
        """
        if world_size % RANKS_PER_GROUP != 0:
            raise ValueError(
                f"ECLATIN: World size must be divisible by {RANKS_PER_GROUP} for multi-rank, got {world_size}"
            )
        group_id = self._get_group_id(my_rank, world_size)
        rank_in_group = self._get_rank_in_group(my_rank, world_size)
        # In-group XOR pairing: position 0<->2, 1<->3
        paired_rank_in_group = (rank_in_group + 2) % RANKS_PER_GROUP
        paired_rank = self._get_rank_by_group_position(group_id, paired_rank_in_group, world_size)
        logger.debug(
            f"ECLATIN: Rank {my_rank} (group_id={group_id}, rank_in_group={rank_in_group}) "
            f"XOR paired with Rank {paired_rank}"
        )
        return paired_rank

    def get_p2p_partner_rank(self, my_rank: int, world_size: int) -> int:
        """Get P2P partner rank for data/parity exchange.

        Multi-rank: within each group, rank_in_group 0<->1, 2<->3 (same as 4-rank).
        """
        if world_size % RANKS_PER_GROUP != 0:
            raise ValueError(
                f"ECLATIN: World size must be divisible by {RANKS_PER_GROUP} for multi-rank, got {world_size}"
            )
        group_id = self._get_group_id(my_rank, world_size)
        rank_in_group = self._get_rank_in_group(my_rank, world_size)
        # In-group P2P: 0<->1, 2<->3
        partner_rank_in_group = rank_in_group ^ 1
        p2p_partner_rank = self._get_rank_by_group_position(group_id, partner_rank_in_group, world_size)
        logger.debug(f"ECLATIN: Rank {my_rank} P2P partner is Rank {p2p_partner_rank}")
        return p2p_partner_rank

    def get_parity2_send1_partner_rank(self, my_rank: int, world_size: int) -> int:
        """Get Parity2 send1 partner rank.

        Multi-rank: within each group, rank_in_group 0<->3, 1<->2 (same as 4-rank).
        """
        if world_size % RANKS_PER_GROUP != 0:
            raise ValueError(
                f"ECLATIN: World size must be divisible by {RANKS_PER_GROUP} for multi-rank, got {world_size}"
            )
        group_id = self._get_group_id(my_rank, world_size)
        rank_in_group = self._get_rank_in_group(my_rank, world_size)
        # In-group: 0<->3, 1<->2
        partner_rank_in_group = (3 - rank_in_group) % RANKS_PER_GROUP
        partner_rank = self._get_rank_by_group_position(group_id, partner_rank_in_group, world_size)
        logger.debug(f"ECLATIN: Rank {my_rank} Parity2 send1 partner is Rank {partner_rank}")
        return partner_rank

    def get_parity2_send2_partner_rank(self, my_rank: int, world_size: int) -> int:
        """Get Parity2 send2 partner rank.

        Multi-rank: within each group, rank_in_group 0<->2, 1<->3 (same as 4-rank).
        """
        if world_size % RANKS_PER_GROUP != 0:
            raise ValueError(
                f"ECLATIN: World size must be divisible by {RANKS_PER_GROUP} for multi-rank, got {world_size}"
            )
        group_id = self._get_group_id(my_rank, world_size)
        rank_in_group = self._get_rank_in_group(my_rank, world_size)
        # In-group: 0<->2, 1<->3
        partner_rank_in_group = (rank_in_group + 2) % RANKS_PER_GROUP
        partner_rank = self._get_rank_by_group_position(group_id, partner_rank_in_group, world_size)
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
        # Step 1: Get base IP address (with multi-NIC per-rank support)
        base_ip = resolve_ip("ECLATIN", rank=rank)

        # Step 2: Get base port
        # Priority: ECLATIN_BASE_PORT > MASTER_PORT + 10000 > default 16000
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('ECLATIN_BASE_PORT', master_port + 10000))

        # Step 3: Calculate ports for this rank
        # Per-group port allocation to avoid cross-group conflicts (same pattern as ECNAIVE)
        n = RANKS_PER_GROUP
        ports_per_rank = 8  # 4 per parity × 2 parities
        if world_size >= n and world_size % n == 0:
            group_id = self._get_group_id(rank, world_size)
            rank_in_group = self._get_rank_in_group(rank, world_size)
            port_base = base_port + group_id * (n * ports_per_rank) + rank_in_group * ports_per_rank
        else:
            port_base = base_port + rank * ports_per_rank

        ports = {
            # Parity 1 ports (offsets 0-3)
            'parity1_send1': port_base + 0,
            'parity1_send2': port_base + 1,
            'parity1_recv1': port_base + 2,
            'parity1_recv2': port_base + 3,
            # Parity 2 ports (offsets 4-7)
            'parity2_send1': port_base + 4,
            'parity2_send2': port_base + 5,
            'parity2_recv1': port_base + 6,
            'parity2_recv2': port_base + 7,
        }

        # Multi-rank: group_id and rank_in_group for load mode ports
        group_id = self._get_group_id(rank, world_size)
        rank_in_group = self._get_rank_in_group(rank, world_size)
        # Load mode ports (for rank_in_group 2 recovery): per-group base to avoid port conflict
        load_base_port = base_port + 1000 + group_id * 100
        from megatron.training import get_args as _get_args
        _args = _get_args()
        _no_shared = getattr(_args, "no_shared_block", False)

        if _no_shared:
            ports.update({
                'ns_recv_n1_d1': load_base_port + 0,
                'ns_recv_n3_p1': load_base_port + 1,
                'ns_recv_n0_d0': load_base_port + 2,
                'ns_recv_n3_p0': load_base_port + 3,
                'ns_recv_n1_d0': load_base_port + 4,
                'ns_recv_n3_d1': load_base_port + 5,
                'ns_recv_n0_d1': load_base_port + 6,
                'ns_recv_n3_d0': load_base_port + 7,
            })
            if rank_in_group != 2:
                if rank_in_group == 0:
                    ports.update({
                        'ns_send_n0_d0': load_base_port + 2,
                        'ns_send_n0_d1': load_base_port + 6,
                    })
                elif rank_in_group == 1:
                    ports.update({
                        'ns_send_n1_d1': load_base_port + 0,
                        'ns_send_n1_d0': load_base_port + 4,
                    })
                elif rank_in_group == 3:
                    ports.update({
                        'ns_send_n3_p1': load_base_port + 1,
                        'ns_send_n3_p0': load_base_port + 3,
                        'ns_send_n3_d1': load_base_port + 5,
                        'ns_send_n3_d0': load_base_port + 7,
                    })
        else:
            # Always add all 6 load_recv_* port keys (for rank_in_group 2 receiver)
            # so all ranks can look them up unconditionally.
            ports.update({
                'load_recv_rank0_data2': load_base_port + 0,
                'load_recv_rank0_parity2': load_base_port + 1,
                'load_recv_rank1_data1': load_base_port + 2,
                'load_recv_rank1_parity1': load_base_port + 3,
                'load_recv_rank3_data1': load_base_port + 4,
                'load_recv_rank3_data2': load_base_port + 5,
            })
            # Add per-rank send port keys for non-receiver ranks
            if rank_in_group != 2:
                if rank_in_group == 0:
                    ports.update({
                        'load_send_rank0_data2': load_base_port + 0,
                        'load_send_rank0_parity2': load_base_port + 1,
                    })
                elif rank_in_group == 1:
                    ports.update({
                        'load_send_rank1_data1': load_base_port + 2,
                        'load_send_rank1_parity1': load_base_port + 3,
                    })
                elif rank_in_group == 3:
                    ports.update({
                        'load_send_rank3_data1': load_base_port + 4,
                        'load_send_rank3_data2': load_base_port + 5,
                    })

        # Two-failures load mode ports (always added for all ranks)
        # 8 ports per group:
        #   surv_exch base (+0..+3): 4 parallel survivor↔survivor channels
        #   n1_from_n3/4, n2_from_n3/4: failed nodes accept from survivors
        two_fail_base_port = load_base_port + 50
        ports.update({
            'twf_surv_exch':   two_fail_base_port + 0,
            'twf_n1_from_n3':  two_fail_base_port + 4,
            'twf_n1_from_n4':  two_fail_base_port + 5,
            'twf_n2_from_n3':  two_fail_base_port + 6,
            'twf_n2_from_n4':  two_fail_base_port + 7,
        })

        # Step 4: Exchange IP addresses via broadcast (more reliable than all_gather_object with NCCL)
        # Use sequential broadcast to avoid NCCL issues with Python objects
        rank_ips = {}

        if torch.distributed.is_initialized():
            try:
                # Method: Each rank broadcasts its IP to all other ranks sequentially
                # This avoids NCCL backend issues with all_gather_object
                for src_rank in range(world_size):
                    if src_rank == rank:
                        # Broadcast my IP to all ranks
                        ip_to_broadcast = base_ip
                    else:
                        # Prepare to receive IP from src_rank
                        ip_to_broadcast = ""

                    # Create a list with single element for broadcast_object_list
                    ip_list = [ip_to_broadcast]
                    torch.distributed.broadcast_object_list(ip_list, src=src_rank)

                    # Store the received IP
                    rank_ips[src_rank] = ip_list[0]

                logger.info(
                    f"ECLATIN: [Rank {rank}] IP exchange completed - "
                    f"All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"ECLATIN: Failed to exchange IPs via broadcast, using local IP: {e}"
                )
                # Fallback to using local IP for all ranks
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            # Single rank mode - use local IP
            logger.info("ECLATIN: Distributed not initialized, using local IP for all ranks")
            rank_ips[0] = base_ip

        # load_receiver_rank: global rank of rank_in_group 2 in this group (for init_load_connections)
        load_receiver_rank = self._get_rank_by_group_position(group_id, 2, world_size) if world_size >= RANKS_PER_GROUP else 2
        config = {
            'my_ip': base_ip,
            'base_port': base_port,
            'rank_ips': rank_ips,
            'ports': ports,
            'rank_in_group': rank_in_group,
            'group_id': group_id,
            'load_receiver_rank': load_receiver_rank,
        }

        logger.info(
            f"ECLATIN: [Rank {rank}] Network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  rank_in_group: {config['rank_in_group']}, load_receiver_rank: {config['load_receiver_rank']}\n"
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
            self.use_rdma = getattr(args, 'use_rdma', False)
            if not getattr(args, 'use_eclatin', False):
                return

            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning("ECLATIN: Distributed environment not initialized, skipping ECLATIN initialization")
                return

            # Initialize ECLATIN C++ module
            self._init_eclatin_native()

            # Start persistent buffer poller thread (only for non-layerwise mode)
            # In layerwise mode, we don't use buffer pools, so no need for poller thread
            use_eclatin_layerwise = getattr(args, 'use_eclatin_layerwise', False)
            if not use_eclatin_layerwise:
                self._start_buffer_poller_thread()
            else:
                logger.info("ECLATIN: Layerwise mode - skipping buffer poller thread (no buffer pools)")

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
            so_files = _glob_module.glob(os.path.join(current_dir, "checkcode_native*.so"))
            if not so_files:
                # Backward compat: pre-rename builds still install eclatin_native*.so
                so_files = _glob_module.glob(os.path.join(current_dir, "eclatin_native*.so"))

            if not so_files:
                raise ImportError(
                    f"No checkcode_native*.so or eclatin_native*.so found in {current_dir}"
                )

            # Load .so file directly using importlib
            import importlib.util as _importlib_util
            so_path = so_files[0]
            mod_name = (
                "checkcode_native"
                if os.path.basename(so_path).startswith("checkcode_native")
                else "eclatin_native"
            )
            spec = _importlib_util.spec_from_file_location(mod_name, so_path)
            eclatin_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(eclatin_native)
            logger.debug(f"ECLATIN: Loaded .so file from {so_path}")

            # Check RDMA availability if RDMA mode is requested (strict mode: fail if not available)
            if self.use_rdma:
                try:
                    rdma_available = eclatin_native.is_rdma_available()
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
                    logger.warning(f"ECLATIN: Could not check RDMA availability: {check_err}")
                    logger.warning("ECLATIN: Will attempt to initialize RDMA anyway...")

            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            # ECLATIN uses ASIO or RDMA (based on use_rdma flag)
            # Create instance with error handling
            try:
                # ===== Network Initialization Path (ASIO or RDMA) =====
                transport_mode = "RDMA" if self.use_rdma else "ASIO"
                logger.info(f"ECLATIN: [Rank {rank}] Using {transport_mode} for communication")

                # Get network configuration
                net_config = self._get_eclatin_network_config(rank, world_size)

                # Synchronize all ranks before creating C++ instances
                logger.info(f"ECLATIN: [Rank {rank}] Synchronizing all ranks before creating C++ native module ({transport_mode})...")
                torch.distributed.barrier()
                logger.info(f"ECLATIN: [Rank {rank}] All ranks synchronized, creating C++ native module with {transport_mode}...")

                # Create C++ instance with network parameters
                logger.info(f"ECLATIN: Creating C++ native module with {transport_mode} (this will block until connections are established)...")
                print(f"ECLATIN: [Rank {rank}] Creating C++ native module with {transport_mode} (blocking until initialization completes)...")

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

                rank_ips = net_config['rank_ips']

                # Calculate partner ports using per-group allocation
                n = RANKS_PER_GROUP
                ports_per_rank = 8

                def _partner_recv_port(partner_global_rank: int, offset: int) -> int:
                    """Compute partner's recv port from group-aware port layout."""
                    master_port = int(os.environ.get('MASTER_PORT', '6000'))
                    base_port = int(os.environ.get('ECLATIN_BASE_PORT', master_port + 10000))
                    if world_size >= n and world_size % n == 0:
                        partner_gid = self._get_group_id(partner_global_rank, world_size)
                        partner_rig = self._get_rank_in_group(partner_global_rank, world_size)
                        partner_pb = base_port + partner_gid * (n * ports_per_rank) + partner_rig * ports_per_rank
                    else:
                        partner_pb = base_port + partner_global_rank * ports_per_rank
                    return partner_pb + offset

                # Parity 1 partner recv ports
                parity1_send1_partner_recv_port = _partner_recv_port(parity1_send1_partner_rank, 3)  # recv2 offset
                parity1_send2_partner_recv_port = _partner_recv_port(parity1_send2_partner_rank, 2)  # recv1 offset

                # Parity 2 partner recv ports
                parity2_send1_partner_recv_port = _partner_recv_port(parity2_send1_partner_rank, 7)  # recv2 offset (4+3)
                parity2_send2_partner_recv_port = _partner_recv_port(parity2_send2_partner_rank, 6)  # recv1 offset (4+2)

                # Get CUDA stream configuration from environment or args
                from megatron.training import get_args
                args = get_args()
                num_cuda_streams = int(os.environ.get('ECLATIN_NUM_CUDA_STREAMS',
                                                      getattr(args, 'eclatin_num_cuda_streams', 4)))

                logger.info(f"ECLATIN: [Rank {rank}] Using {num_cuda_streams} CUDA streams for async transfers")

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
                    net_config['my_ip'], net_config['ports']['parity2_recv2'],
                    # CUDA streams configuration
                    num_cuda_streams,
                    # RDMA flag
                    self.use_rdma,
                    # Multi-rank support: rank, world_size, rank_in_group
                    rank,
                    world_size,
                    net_config['rank_in_group'],
                )

                # If we reach here, connections are ready and threads are running
                logger.info(f"ECLATIN: C++ native module initialized successfully with {transport_mode} (rank={rank}, world_size={world_size})")
                print(f"ECLATIN: [Rank {rank}] C++ native module initialized - {transport_mode} connections ready for data exchange")

                # Initialize ECLATIN buffers (skip buffer pool for layerwise mode)
                # In layerwise mode, we use continuous recv buffers allocated in strategy, not pooled buffers
                from megatron.training import get_args
                args = get_args()
                use_eclatin_layerwise = getattr(args, 'use_eclatin_layerwise', False)

                if not use_eclatin_layerwise:
                    # Only allocate buffer pools for non-layerwise mode
                    self._init_eclatin_buffers()
                else:
                    # Layerwise mode: skip buffer pool allocation
                    # Data buffers: not needed (directly send from layer_cpu_buffer)
                    # Recv buffers: allocated in strategy as continuous buffers
                    self.eclatin_data_buffers = []
                    self.eclatin_recv_buffers = []
                    self._free_data_buffer_queue = queue.Queue()
                    self._free_recv_buffer_queue = queue.Queue()
                    logger.info("ECLATIN: Layerwise mode - skipping buffer pool allocation (using continuous buffers from strategy)")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"ECLATIN: Failed to create C++ native module instance: {error_msg}")

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
        from megatron.training import get_args
        args = get_args()
        if getattr(args, "no_shared_block", False):
            self.eclatin_recv_buffers_count = 16

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

        # Register buffers for RDMA if enabled
        if self.use_rdma:
            logger.info(f"ECLATIN: [Rank {rank}] Registering buffer pools for RDMA...")
            for buffer in self.eclatin_data_buffers:
                self.register_buffer(buffer)
            for buffer in self.eclatin_recv_buffers:
                self.register_buffer(buffer)
            logger.info(f"ECLATIN: [Rank {rank}] Buffer pools registered for RDMA")

    def _allocate_data_buffers(self):
        """Allocate data buffers for storing original tensor data."""
        logger.info(f"ECLATIN: Allocating data buffers ({self.eclatin_data_buffers_count} buffers, {self.eclatin_buffer_size // (1024*1024)}MB each)")

        data_buffers = []
        for i in range(self.eclatin_data_buffers_count):
            buffer = allocate_hugepage_tensor(
                self.eclatin_buffer_size,
                fallback_pin_memory=self.eclatin_pin_memory,
                touch_pages=True,
            )
            data_buffers.append(buffer)
            logger.debug(f"ECLATIN: Allocated data buffer {i}: {self.eclatin_buffer_size} bytes")

        logger.info(f"ECLATIN: Allocated {len(data_buffers)} data buffers")
        return data_buffers

    def _allocate_recv_buffers(self):
        """Allocate recv buffers (pooled) for receiving data."""
        logger.info(f"ECLATIN: Allocating recv buffers ({self.eclatin_recv_buffers_count} buffers, {self.eclatin_buffer_size // (1024*1024)}MB each)")

        recv_buffers = []
        for i in range(self.eclatin_recv_buffers_count):
            buffer = allocate_hugepage_tensor(
                self.eclatin_buffer_size,
                fallback_pin_memory=self.eclatin_pin_memory,
                touch_pages=True,
            )
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
        rank_in_group = self._get_rank_in_group(rank, world_size)

        if rank_in_group != 2:
            logger.warning(
                "ECLATIN: allocate_eclatin_load_recv_buffers called on non rank_in_group 2, returning empty dict"
            )
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
            f"ECLATIN: Allocating 6 recv buffers for rank_in_group 2 load recovery\n"
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
            f"ECLATIN: Allocated 6 recv buffers for rank_in_group 2: "
            f"{aligned_half_block_size / (1024**3):.2f} GB each"
        )

        return recv_buffers

    def allocate_eclatin_load_recv_buffers_two_fail(
        self, global_registry: GlobalMetadataRegistry
    ) -> Dict[str, torch.Tensor]:
        """
        Allocate 4 recv buffers for two-failures recovery on failed ranks (0 or 1).

        Each failed rank receives 2 blocks from surviving rank 2 and 2 blocks from
        surviving rank 3, totalling 4 recv buffers (down from 8 in the old scheme).

        Returns:
            rig0: {'n3_b12', 'n3_b14', 'n4_b11', 'n4_b13'}
            rig1: {'n3_b21', 'n3_b23', 'n4_b22', 'n4_b24'}
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank_in_group = self._get_rank_in_group(rank, world_size)

        if rank_in_group not in (0, 1):
            logger.warning(
                "ECLATIN: allocate_eclatin_load_recv_buffers_two_fail called on "
                f"rank_in_group {rank_in_group}, expecting 0 or 1; returning empty dict"
            )
            return {}

        max_total_bytes = 0
        for r in range(world_size):
            rank_metadata = global_registry.rank_metadata.get(r, [])
            rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
            if rank_total_size > max_total_bytes:
                max_total_bytes = rank_total_size

        half_max_total_bytes = max_total_bytes // 2
        aligned_half_block_size = (
            (half_max_total_bytes + self.eclatin_buffer_size - 1) // self.eclatin_buffer_size
        ) * self.eclatin_buffer_size

        logger.info(
            f"ECLATIN: Allocating 4 recv buffers for two-failures recovery\n"
            f"  Pipeline max size: {max_total_bytes / (1024**3):.2f} GB\n"
            f"  Aligned half block size: {aligned_half_block_size / (1024**3):.2f} GB\n"
            f"  Total recv memory: {4 * aligned_half_block_size / (1024**3):.2f} GB"
        )

        if rank_in_group == 0:
            keys = ['n3_b12', 'n3_b14', 'n4_b11', 'n4_b13']
        else:
            keys = ['n3_b21', 'n3_b23', 'n4_b22', 'n4_b24']

        recv_buffers = {}
        for key in keys:
            recv_buffers[key] = torch.empty(
                aligned_half_block_size, dtype=torch.uint8,
                pin_memory=self.eclatin_pin_memory,
            )

        logger.info(
            f"ECLATIN: Allocated 4 recv buffers for two-failures recovery (rig{rank_in_group}): "
            f"{aligned_half_block_size / (1024**3):.2f} GB each"
        )
        return recv_buffers

    def allocate_twf_survivor_buffers(
        self, global_registry: GlobalMetadataRegistry
    ) -> Dict[str, torch.Tensor]:
        """
        Allocate temporary buffers for survivors (rig2/rig3) during two-failure recovery.

        Each survivor needs:
        - 2 recv buffers for peer's data blocks (from survivor exchange step)
        - 4 output buffers for XOR decode results

        Returns:
            Dict with keys: 'peer_d1', 'peer_d2', 'out1', 'out2', 'out3', 'out4'
            (rig2: peer=Node4's b41,b42; rig3: peer=Node3's b31,b32)
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank_in_group = self._get_rank_in_group(rank, world_size)

        if rank_in_group not in (2, 3):
            logger.warning(
                "ECLATIN: allocate_twf_survivor_buffers called on "
                f"rank_in_group {rank_in_group}, expecting 2 or 3; returning empty dict"
            )
            return {}

        max_total_bytes = 0
        for r in range(world_size):
            rank_metadata = global_registry.rank_metadata.get(r, [])
            rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
            if rank_total_size > max_total_bytes:
                max_total_bytes = rank_total_size

        half_max_total_bytes = max_total_bytes // 2
        aligned_half_block_size = (
            (half_max_total_bytes + self.eclatin_buffer_size - 1) // self.eclatin_buffer_size
        ) * self.eclatin_buffer_size

        logger.info(
            f"ECLATIN: Allocating 6 temp buffers for survivor two-failures (rig{rank_in_group})\n"
            f"  Aligned half block size: {aligned_half_block_size / (1024**3):.2f} GB each\n"
            f"  Total temp memory: {6 * aligned_half_block_size / (1024**3):.2f} GB"
        )

        buffers = {}
        # 2 recv buffers for peer data blocks
        for key in ['peer_d1', 'peer_d2']:
            buffers[key] = allocate_hugepage_tensor(
                aligned_half_block_size,
                fallback_pin_memory=self.eclatin_pin_memory,
                touch_pages=True,
            )
        # 4 output buffers for XOR decode results
        for key in ['out1', 'out2', 'out3', 'out4']:
            buffers[key] = allocate_hugepage_tensor(
                aligned_half_block_size,
                fallback_pin_memory=self.eclatin_pin_memory,
                touch_pages=True,
            )

        logger.info(
            f"ECLATIN: Allocated 6 temp buffers for survivor two-failures "
            f"(rig{rank_in_group}): {aligned_half_block_size / (1024**3):.2f} GB each"
        )
        return buffers

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

        # Get twofail pipeline pool buffers ready for release
        twofail_buffers = self._eclatin_native.get_twofail_buffers_to_release()
        for data_addr in twofail_buffers:
            try:
                self._free_data_buffer_queue.put_nowait(data_addr)
            except Exception:
                logger.error(
                    f"ECLATIN: Data buffer queue is full, cannot release twofail buffer {data_addr}"
                )

        # Get onefail pipeline recv pool buffers ready for release
        onefail_buffers = self._eclatin_native.get_onefail_buffers_to_release()
        for recv_addr in onefail_buffers:
            try:
                self._free_recv_buffer_queue.put_nowait(recv_addr)
            except Exception:
                logger.error(
                    f"ECLATIN: Recv buffer queue is full, cannot release onefail buffer {recv_addr}"
                )

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

    def register_buffer(self, buffer: torch.Tensor):
        """Register buffer for RDMA operations (called on first allocation in save phase).

        Args:
            buffer: PyTorch tensor to register
        """
        if not self.use_rdma or self._eclatin_native is None:
            return

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        buffer_size = buffer.numel() * buffer.element_size()

        # Check if already registered
        if buffer_addr in self.registered_buffers:
            logger.debug(f"ECLATIN: [Rank {rank}] Buffer already registered at 0x{buffer_addr:x} (size: {buffer_size / (1024**2):.2f} MB)")
            return

        try:
            logger.info(f"ECLATIN: [Rank {rank}] Registering buffer at 0x{buffer_addr:x}, size: {buffer_size / (1024**3):.2f} GB, numel: {buffer.numel()}, dtype: {buffer.dtype} (iteration {self.current_iteration})")
            self._eclatin_native.register_buffer(buffer_addr, buffer_size)
            self.registered_buffers[buffer_addr] = (buffer_size, self.current_iteration)
            logger.info(f"ECLATIN: [Rank {rank}] Buffer registered successfully (total registered: {len(self.registered_buffers)})")

            # Print all registered buffers
            logger.info(f"ECLATIN: [Rank {rank}] All registered buffers:")
            # for addr, (size, iteration) in self.registered_buffers.items():
            #     logger.info(f"  - 0x{addr:x}: {size / (1024**2):.2f} MB (iteration {iteration})")
        except Exception as e:
            logger.error(f"ECLATIN: [Rank {rank}] Failed to register buffer: {e}")
            raise

    def unregister_buffer(self, buffer: torch.Tensor):
        """Unregister buffer for RDMA operations.

        Args:
            buffer: PyTorch tensor to unregister
        """
        if not self.use_rdma or self._eclatin_native is None:
            return

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()

        if buffer_addr not in self.registered_buffers:
            logger.debug(f"ECLATIN: [Rank {rank}] Buffer not registered at 0x{buffer_addr:x}")
            return

        try:
            logger.info(f"ECLATIN: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x}")
            self._eclatin_native.unregister_buffer(buffer_addr)
            del self.registered_buffers[buffer_addr]
            logger.info(f"ECLATIN: [Rank {rank}] Buffer unregistered successfully")
        except Exception as e:
            logger.error(f"ECLATIN: [Rank {rank}] Failed to unregister buffer: {e}")

    def cleanup(self):
        """Cleanup ECLATIN resources when manager is destroyed."""
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

            # Unregister all buffers for RDMA
            if self.use_rdma and self._eclatin_native is not None:
                for buffer_addr in list(self.registered_buffers.keys()):
                    try:
                        logger.info(f"ECLATIN: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x} during cleanup")
                        self._eclatin_native.unregister_buffer(buffer_addr)
                    except Exception as e:
                        logger.warning(f"ECLATIN: [Rank {rank}] Failed to unregister buffer during cleanup: {e}")
                self.registered_buffers.clear()

            # Stop buffer poller thread
            self._stop_buffer_poller_thread()

            # Stop the C++ pipeline
            if hasattr(self, '_eclatin_native') and self._eclatin_native is not None:
                self._eclatin_native.stop()
                logger.info("ECLATIN: C++ native module stopped in manager cleanup")

            # Release cached allocations
            self.preallocated_cpu_buffer = None
            self._cached_blocks = None
            self._cached_block_count = 0
            self._cached_block_size = 0
            self.eclatin_data_buffers = None
            self.eclatin_recv_buffers = None
            self._free_data_buffer_queue = None
            self._free_recv_buffer_queue = None

        except Exception as e:
            logger.warning(f"ECLATIN: Error during manager cleanup: {e}")

    def __del__(self):
        """Cleanup ECLATIN resources when manager is destroyed."""
        self.cleanup()
