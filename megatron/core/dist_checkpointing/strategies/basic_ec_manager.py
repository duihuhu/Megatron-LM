# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""BasicEC manager for shared basic_ec_native initialization and buffer management."""

import os
import queue
import socket
import threading
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip

from .hugepage_alloc import allocate_hugepage_slices, allocate_hugepage_tensor

logger = getLogger(__name__)

# Legacy RS(4, 2) defaults: n = k + 2 ranks and 3 * (n - 1) ports per rank.
DEFAULT_RANKS_PER_GROUP = 4
DEFAULT_PORTS_PER_RANK = 9


class BasicECManager:
    """Shared manager for BasicEC C++ module initialization and buffer management.

    This class provides a singleton instance that manages:
    - BasicEC C++ native module (_basic_ec_native)
    - Buffer allocation and management (data and parity buffers, pooled)
    - Buffer poller thread for releasing buffers

    Persistent storage consists of one local data block and n - 1 received
    blocks. The strategy allocates these blocks after metadata exchange.

    Both TorchDistSaveShardedStrategy and TorchDistLoadShardedStrategy
    can share the same manager instance to reuse initialized resources.
    """

    _instance: Optional['BasicECManager'] = None
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

        self._basic_ec_native = None
        self.use_basic_ec = False
        self.use_rdma = False

        # BasicEC configuration
        self.basic_ec_k = 2  # Number of RS data blocks.
        self.basic_ec_n = DEFAULT_RANKS_PER_GROUP  # RS codeword size, n = k + 2.
        self.basic_ec_ports_per_rank = DEFAULT_PORTS_PER_RANK

        # Buffer configuration
        self.basic_ec_data_buffers_count = 12
        self.basic_ec_parity_buffers_count = 12  # Pooled parity buffers
        self.basic_ec_buffer_size = 64 * 1024 * 1024  # 64MB
        self.basic_ec_pin_memory = True

        # Buffers (simplified: only data and parity pools)
        self.basic_ec_data_buffers: Optional[List[torch.Tensor]] = None
        self.basic_ec_parity_buffers: Optional[List[torch.Tensor]] = None  # Pooled parity buffers
        # The strategy allocates one local and n - 1 received persistent blocks.

        # Free buffer queues (simplified)
        self._free_data_buffer_queue: Optional[queue.Queue] = None
        self._free_parity_buffer_queue: Optional[queue.Queue] = None

        # Buffer poller thread
        self._buffer_poller_thread: Optional[threading.Thread] = None
        self._buffer_poller_stop_event: Optional[threading.Event] = None
        self._buffer_poller_active_event: Optional[threading.Event] = None

        # RDMA buffer registry (similar to Gemini)
        self.registered_buffers = {}  # {addr: (size, iteration)}
        self.current_iteration = 0
        self.preallocated_cpu_buffer: Optional[torch.Tensor] = None

        # HW recovery: store recovered checkpoint blocks for cascading failure tolerance
        # {global_rank: {"own_data0": tensor, "recv_0": tensor, ...}}
        self._recovered_blocks: Dict[int, Dict[str, torch.Tensor]] = {}

        # The manager owns recovery buffers whose addresses may remain registered
        # with the native transport. Their lifetime must cover the native instance,
        # so an active workspace cannot be replaced in process. cleanup() releases
        # registrations before dropping buffers, but topology changes still require
        # a process restart because persistent recovery channels cannot be rebuilt.
        self._recovery_workspace_key: Optional[tuple] = None
        self._recovery_workspace: Dict[str, Any] = {}
        self._sw_recovery_connection_key: Optional[tuple] = None

        self._initialized = True

    def allocate_preallocated_buffer(self, size_bytes: int):
        """Allocate or reuse cached CPU buffer.  Grows only when needed."""
        if self.preallocated_cpu_buffer is not None:
            if self.preallocated_cpu_buffer.numel() >= size_bytes:
                return
        pin = self.basic_ec_pin_memory and torch.cuda.is_available()
        logger.debug(
            f"BasicEC: Allocating preallocated buffer: {size_bytes / (1024**3):.2f} GB (pin={pin})"
        )
        self.preallocated_cpu_buffer = allocate_hugepage_tensor(
            size_bytes, fallback_pin_memory=pin, touch_pages=False
        )

    _cached_block_count: int = 0
    _cached_block_size: int = 0
    _cached_blocks: Optional[List[torch.Tensor]] = None

    def allocate_preallocated_blocks(
        self, count: int, aligned_size: int, pin: Optional[bool] = None
    ):
        """Allocate or reuse cached persistent blocks (n = k+2 blocks).

        Args:
            count: Number of blocks.
            aligned_size: Size of each block in bytes.
            pin: Whether to pin memory. None = use manager default.
                 Set False during load to avoid exhausting CUDA lockable memory.
        """
        if (
            self._cached_blocks is not None
            and self._cached_block_count == count
            and self._cached_block_size >= aligned_size
        ):
            return self._cached_blocks
        if pin is None:
            pin = self.basic_ec_pin_memory and torch.cuda.is_available()
        logger.debug(
            f"BasicEC: Allocating {count} blocks: {aligned_size / (1024**3):.2f} GB each "
            f"({count * aligned_size / (1024**3):.2f} GB total, pin={pin})"
        )
        self._cached_blocks = list(
            allocate_hugepage_slices(aligned_size, count, fallback_pin_memory=pin, touch_pages=True)
        )
        self._cached_block_count = count
        self._cached_block_size = aligned_size
        if self.use_rdma:
            for b in self._cached_blocks:
                self.register_buffer(b)
        return self._cached_blocks

    @staticmethod
    def _get_ranks_per_node() -> int:
        """Best-effort infer per-node rank count from environment."""
        env_keys = (
            "LOCAL_WORLD_SIZE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
            "MPI_LOCALNRANKS",
            "MV2_COMM_WORLD_LOCAL_SIZE",
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
            # Common formats: "8" or "8(x2)"
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

    def _get_group_layout(self, world_size: int) -> Dict[str, int]:
        """Return EC grouping layout and mapping mode.

        node-aware mode is used when:
        - world_size is divisible by n (k+2)
        - inferred num_nodes is divisible by n
        - ranks_per_node divides world_size
        """
        n = self.basic_ec_n
        if world_size <= 0:
            return {"mode": 0, "num_groups": 1, "ranks_per_node": 1, "num_nodes": 1, "clusters": 1}
        num_groups = max(1, world_size // n)
        ranks_per_node = self._get_ranks_per_node()
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
                        "mode": 1,
                        "num_groups": num_groups,
                        "ranks_per_node": ranks_per_node,
                        "num_nodes": num_nodes,
                        "clusters": clusters,
                    }
        return {
            "mode": 0,
            "num_groups": num_groups,
            "ranks_per_node": max(1, ranks_per_node),
            "num_nodes": max(1, world_size // max(1, ranks_per_node)),
            "clusters": 1,
        }

    def _get_group_id(self, rank: int, world_size: int) -> int:
        """Get group id for BasicEC multi-rank setup."""
        layout = self._get_group_layout(world_size)
        num_groups = layout["num_groups"]
        if layout["mode"] == 1:
            ranks_per_node = layout["ranks_per_node"]
            clusters = layout["clusters"]
            node_id = rank // ranks_per_node
            local_rank = rank % ranks_per_node
            cluster_id = node_id % clusters
            return local_rank * clusters + cluster_id
        return rank % num_groups

    def _get_rank_in_group(self, rank: int, world_size: int) -> int:
        """Get rank index within group (0..n-1)."""
        layout = self._get_group_layout(world_size)
        num_groups = layout["num_groups"]
        if layout["mode"] == 1:
            ranks_per_node = layout["ranks_per_node"]
            clusters = layout["clusters"]
            node_id = rank // ranks_per_node
            return node_id // clusters
        return rank // num_groups

    def _get_rank_by_group_position(
        self, group_id: int, rank_in_group: int, world_size: int
    ) -> int:
        """Map (group_id, rank_in_group) -> global rank."""
        layout = self._get_group_layout(world_size)
        num_groups = layout["num_groups"]
        if layout["mode"] == 1:
            ranks_per_node = layout["ranks_per_node"]
            clusters = layout["clusters"]
            local_rank = group_id // clusters
            cluster_id = group_id % clusters
            node_id = rank_in_group * clusters + cluster_id
            return node_id * ranks_per_node + local_rank
        return group_id + num_groups * rank_in_group

    def _get_round_robin_ranks(self, rank: int, world_size: int) -> dict:
        """Calculate round-robin partner ranks for BasicEC.

        Generalized for k+2 scheme:
        - Rank i splits data into k blocks: d_{i,0}, ..., d_{i,k-1}
        - Rank i encodes: (d_{i,0}, ..., d_{i,k-1}) -> p_{i,0}, p_{i,1}
        - Rank i keeps d_{i,0}
        - Sends: d_{i,j} to rank i+j for j in [1, k-1]
                 p_{i,0} to rank i+k
                 p_{i,1} to rank i+k+1
        - Recvs: mirror of sends (n-1 blocks from other ranks in group)

        Returns dict with:
            'send_partners': list of (global_rank, label) for each send channel
            'recv_partners': list of (global_rank, label) for each recv channel
            'send_block_types': list of block type names for each send
            'recv_block_types': list of block type names for each recv
        For the legacy RS(4, 2) layout, the result also includes compatibility keys.
        """
        k = self.basic_ec_k
        n = self.basic_ec_n

        if world_size >= n and world_size % n == 0:
            group_id = self._get_group_id(rank, world_size)
            rank_in_group = self._get_rank_in_group(rank, world_size)

            send_partners = []
            recv_partners = []
            send_block_types = []
            recv_block_types = []

            # Data blocks d_{i,1} ... d_{i,k-1}: send to rank_in_group + j
            for j in range(1, k):
                target_ig = (rank_in_group + j) % n
                src_ig = (rank_in_group - j) % n
                send_partners.append(
                    self._get_rank_by_group_position(group_id, target_ig, world_size)
                )
                recv_partners.append(self._get_rank_by_group_position(group_id, src_ig, world_size))
                send_block_types.append(f"data_{j}")
                recv_block_types.append("data")  # received block is a data block from another rank

            # Parity 0: send to rank_in_group + k
            p0_target_ig = (rank_in_group + k) % n
            p0_src_ig = (rank_in_group - k) % n
            send_partners.append(
                self._get_rank_by_group_position(group_id, p0_target_ig, world_size)
            )
            recv_partners.append(self._get_rank_by_group_position(group_id, p0_src_ig, world_size))
            send_block_types.append("parity0")
            recv_block_types.append("parity0")

            # Parity 1: send to rank_in_group + k + 1
            p1_target_ig = (rank_in_group + k + 1) % n
            p1_src_ig = (rank_in_group - k - 1) % n
            send_partners.append(
                self._get_rank_by_group_position(group_id, p1_target_ig, world_size)
            )
            recv_partners.append(self._get_rank_by_group_position(group_id, p1_src_ig, world_size))
            send_block_types.append("parity1")
            recv_block_types.append("parity1")

            result = {
                'send_partners': send_partners,
                'recv_partners': recv_partners,
                'send_block_types': send_block_types,
                'recv_block_types': recv_block_types,
            }

            # Compatibility aliases for the legacy RS(4, 2) layout.
            if k == 2:
                result.update(
                    {
                        'send_data1_to': send_partners[0],
                        'send_parity0_to': send_partners[1],
                        'send_parity1_to': send_partners[2],
                        'recv_parity1_from': recv_partners[0],
                        'recv_parity0_from': recv_partners[1],
                        'recv_data1_from': recv_partners[2],
                    }
                )

            return result

        # Single-ring fallback (world_size not divisible by n)
        send_partners = []
        recv_partners = []
        send_block_types = []
        recv_block_types = []
        for j in range(1, k):
            send_partners.append((rank + j) % world_size)
            recv_partners.append((rank - j) % world_size)
            send_block_types.append(f"data_{j}")
            recv_block_types.append("data")
        send_partners.append((rank + k) % world_size)
        recv_partners.append((rank - k) % world_size)
        send_block_types.append("parity0")
        recv_block_types.append("parity0")
        send_partners.append((rank + k + 1) % world_size)
        recv_partners.append((rank - k - 1) % world_size)
        send_block_types.append("parity1")
        recv_block_types.append("parity1")

        result = {
            'send_partners': send_partners,
            'recv_partners': recv_partners,
            'send_block_types': send_block_types,
            'recv_block_types': recv_block_types,
        }
        if k == 2:
            result.update(
                {
                    'send_data1_to': send_partners[0],
                    'send_parity0_to': send_partners[1],
                    'send_parity1_to': send_partners[2],
                    'recv_parity1_from': recv_partners[0],
                    'recv_parity0_from': recv_partners[1],
                    'recv_data1_from': recv_partners[2],
                }
            )
        return result

    def _get_basic_ec_network_config(self, rank: int, world_size: int) -> dict:
        """
        Get network configuration for BasicEC ASIO connections.

        This function:
        1. Gets base IP address (from BASIC_EC_BASE_IP env var, MASTER_ADDR, or auto-detect)
        2. Assigns 3 * (n - 1) ports to each rank or group position
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
                    - 'send_data1': int
                    - 'send_parity0': int
                    - 'send_parity1': int
                    - 'recv_parity1': int
                    - 'recv_parity0': int
                    - 'recv_data1': int
        """
        # Step 1: Get base IP address (with multi-NIC per-rank support)
        base_ip = resolve_ip("BASIC_EC", rank=rank, fallback_prefixes=["ECNAIVE"])

        # Step 2: Get base port
        # Priority: BASIC_EC_BASE_PORT > MASTER_PORT + 10000 > default 16000
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(
            os.environ.get(
                'BASIC_EC_BASE_PORT', os.environ.get('ECNAIVE_BASE_PORT', master_port + 10000)
            )
        )

        # Step 3: assign each rank n - 1 ASIO send ports, n - 1 ASIO
        # receive ports, and n - 1 RDMA exchange receive ports. Group-based
        # allocation keeps concurrently initialized groups on distinct ports.
        n = self.basic_ec_n
        ports_per_rank = self.basic_ec_ports_per_rank
        if world_size >= n and world_size % n == 0:
            group_id = self._get_group_id(rank, world_size)
            rank_in_group = self._get_rank_in_group(rank, world_size)
            port_base = base_port + group_id * (n * ports_per_rank) + rank_in_group * ports_per_rank
        else:
            port_base = base_port + rank * ports_per_rank

        # Build general channel lists plus legacy RS(4, 2) compatibility keys.
        send_ports = [port_base + i for i in range(n - 1)]
        recv_ports = [port_base + (n - 1) + i for i in range(n - 1)]
        rdma_recv_ports = [port_base + 2 * (n - 1) + i for i in range(n - 1)]

        k = self.basic_ec_k
        ports = {
            'send_ports': send_ports,
            'recv_ports': recv_ports,
            'rdma_recv_ports': rdma_recv_ports,
        }
        # Preserve the named port keys used by the legacy RS(4, 2) path.
        if k == 2:
            ports.update(
                {
                    'send_data1': send_ports[0],
                    'send_parity0': send_ports[1],
                    'send_parity1': send_ports[2] if len(send_ports) > 2 else None,
                    'recv_parity1': recv_ports[0],
                    'recv_parity0': recv_ports[1],
                    'recv_data1': recv_ports[2] if len(recv_ports) > 2 else None,
                    'rdma_recv_parity1': rdma_recv_ports[0],
                    'rdma_recv_parity0': rdma_recv_ports[1],
                    'rdma_recv_data1': rdma_recv_ports[2] if len(rdma_recv_ports) > 2 else None,
                }
            )

        # Step 4: Exchange IP addresses via torch.distributed.all_gather
        rank_ips = {}

        if torch.distributed.is_initialized():
            try:
                # Convert IP to bytes, then to int list for tensor
                my_ip_bytes = socket.inet_aton(base_ip)
                my_ip_tensor = torch.tensor([int(b) for b in my_ip_bytes], dtype=torch.uint8)

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

                logger.debug(
                    f"BasicEC: [Rank {rank}] IP exchange completed - " f"All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"BasicEC: Failed to exchange IPs via all_gather, using local IP: {e}"
                )
                # Fallback to using local IP for all ranks
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            # Single rank mode - use local IP
            logger.debug("BasicEC: Distributed not initialized, using local IP for all ranks")
            rank_ips[0] = base_ip

        config = {'my_ip': base_ip, 'base_port': base_port, 'rank_ips': rank_ips, 'ports': ports}

        logger.debug(
            f"BasicEC: [Rank {rank}] Network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  Ports: {config['ports']}\n"
            f"  All rank IPs: {config['rank_ips']}"
        )

        return config

    def _get_basic_ec_load_network_config(self, rank: int, world_size: int) -> dict:
        """
        Get network configuration for the legacy k=2 hardware-recovery path.

        This fixed RS(4, 2) path uses eight ports with group rank 2 as receiver:
        - load_recv_rank3_data1: rank2 listens, rank3 connects (for d_{3,1})
        - load_recv_rank0_parity0: rank2 listens, rank0 connects (for p_{2,0})
        - load_recv_rank0_data0: rank2 listens, rank0 connects (for d_{0,0})
        - load_recv_rank1_data1: rank2 listens, rank1 connects (for d_{0,1})
        - load_recv_rank1_data0: rank2 listens, rank1 connects (for d_{1,0})
        - load_recv_rank1_parity1: rank2 listens, rank1 connects (for p_{1,1})
        - load_recv_rank3_data0: rank2 listens, rank3 connects (for d_{3,0})
        - load_recv_rank0_data1: rank2 listens, rank0 connects (for d_{3,1})

        Args:
            rank (int): Current global rank; group rank 2 receives and the other
                RS(4, 2) group ranks send.
            world_size (int): Total number of ranks

        Returns:
            dict: Network configuration with keys:
                - 'my_ip': str - This rank's IP address
                - 'base_port': int - Base port number
                - 'rank_ips': dict - IP addresses for all ranks
                - 'ports': dict - Eight legacy RS(4, 2) recovery ports
        """
        # Step 1: Get base IP address (with multi-NIC per-rank support)
        base_ip = resolve_ip("BASIC_EC", rank=rank, fallback_prefixes=["ECNAIVE"])

        # Get base port (same as save mode)
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(
            os.environ.get(
                'BASIC_EC_BASE_PORT', os.environ.get('ECNAIVE_BASE_PORT', master_port + 10000)
            )
        )

        # Multi-rank: per-group load ports to avoid conflict
        n = self.basic_ec_n
        group_id = self._get_group_id(rank, world_size)
        rank_in_group = self._get_rank_in_group(rank, world_size)
        # load_receiver_rank: global rank of rank_in_group 2 in this group (for init_basic_ec_load)
        load_receiver_rank = (
            self._get_rank_by_group_position(group_id, 2, world_size) if world_size >= n else 2
        )
        load_base_port = base_port + 1000 + group_id * 100
        # All ranks in group get same 8 ports (receiver binds, others connect)
        ports = {
            'load_recv_rank3_data1': load_base_port + 0,
            'load_recv_rank0_parity0': load_base_port + 1,
            'load_recv_rank0_data0': load_base_port + 2,
            'load_recv_rank1_data1': load_base_port + 3,
            'load_recv_rank1_data0': load_base_port + 4,
            'load_recv_rank1_parity1': load_base_port + 5,
            'load_recv_rank3_data0': load_base_port + 6,
            'load_recv_rank0_data1': load_base_port + 7,
        }

        # Exchange IP addresses via torch.distributed.all_gather
        rank_ips = {}

        if torch.distributed.is_initialized():
            try:
                my_ip_bytes = socket.inet_aton(base_ip)
                my_ip_tensor = torch.tensor([int(b) for b in my_ip_bytes], dtype=torch.uint8)

                if torch.cuda.is_available():
                    my_ip_tensor = my_ip_tensor.cuda()

                ip_list = [torch.zeros_like(my_ip_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(ip_list, my_ip_tensor)

                for r, ip_tensor in enumerate(ip_list):
                    ip_bytes = bytes(ip_tensor.cpu().tolist())
                    rank_ips[r] = socket.inet_ntoa(ip_bytes)

                logger.debug(
                    f"BasicEC: [Rank {rank}] Load mode IP exchange completed - "
                    f"All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"BasicEC: Failed to exchange IPs via all_gather, using local IP: {e}"
                )
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            logger.debug("BasicEC: Distributed not initialized, using local IP for all ranks")
            rank_ips[0] = base_ip

        config = {
            'my_ip': base_ip,
            'base_port': base_port,
            'rank_ips': rank_ips,
            'ports': ports,
            'rank_in_group': rank_in_group,
            'group_id': group_id,
            'load_receiver_rank': load_receiver_rank,
        }

        logger.debug(
            f"BasicEC: [Rank {rank}] Load mode network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  rank_in_group: {rank_in_group}, group_id: {group_id}, "
            f"load_receiver_rank: {load_receiver_rank}\n"
            f"  Load mode ports: {config['ports']}\n"
            f"  All rank IPs: {config['rank_ips']}"
        )

        return config

    def init_basic_ec_load_software_only(
        self, rank: int, world_size: int, net_config: Optional[dict] = None
    ) -> None:
        """Initialize the legacy k=2 software-recovery channel.

        Group rank 3 sends d_{2,1} to group rank 2 over one persistent channel.
        This fixed path avoids the eight-port hardware-recovery setup. If the
        caller supplies ``net_config``, reuse it to avoid another IP exchange.
        """
        if self._basic_ec_native is None:
            logger.error("BasicEC: Native module not initialized, cannot initialize load mode")
            return
        if not self.use_basic_ec:
            logger.warning("BasicEC: Manager not enabled, skipping load initialization")
            return
        failed_rank = 2
        self._basic_ec_native.set_load_mode(True, failed_rank, rank, is_software_only=True)
        logger.debug(
            f"BasicEC: [Rank {rank}] Set load mode (failed_rank={failed_rank}) for software-only"
        )
        if net_config is None:
            net_config = self._get_basic_ec_load_network_config(rank, world_size)
        rank_in_group = net_config['rank_in_group']
        load_receiver_rank = net_config['load_receiver_rank']
        receiver_ip = net_config['rank_ips'].get(load_receiver_rank, net_config['my_ip'])
        port = net_config['ports'].get('load_recv_rank3_data1', 0)
        torch.distributed.barrier()
        self._basic_ec_native.init_basic_ec_load_connections_software_only(
            rank_in_group, receiver_ip, port
        )
        logger.debug(f"BasicEC: [Rank {rank}] Software-only load connection initialized (1 port)")

    # ---- Generalized SW recovery (k-1 ports, any k >= 2) ----

    def init_basic_ec_sw_recovery(
        self, rank: int, world_size: int, failed_rank_in_group: int = 2
    ) -> None:
        """Initialize generalized SW recovery with k-1 ports (one per non-local data block).

        Replaces init_basic_ec_load_software_only for k > 2.  Sets up dedicated
        ASIO/RDMA connections from each sender rank (holding one of the failed
        rank's data blocks) to the receiver rank.
        """
        native = self._basic_ec_native
        if native is None:
            raise RuntimeError("BasicEC native module not initialized for SW recovery")
        k = self.basic_ec_k
        num_blocks = k - 1  # d_{f,1} .. d_{f,k-1} from network, d_{f,0} local

        # Network config: k-1 consecutive ports after the save ports
        base_ip = resolve_ip("BASIC_EC", rank=rank, fallback_prefixes=["ECNAIVE"])
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(
            os.environ.get(
                'BASIC_EC_BASE_PORT', os.environ.get('ECNAIVE_BASE_PORT', master_port + 10000)
            )
        )
        n = self.basic_ec_n
        group_id = self._get_group_id(rank, world_size)
        rank_in_group = self._get_rank_in_group(rank, world_size)
        failed_global_rank = (
            self._get_rank_by_group_position(group_id, failed_rank_in_group, world_size)
            if world_size >= n
            else failed_rank_in_group
        )
        native.set_load_mode(True, failed_global_rank, rank_in_group, is_software_only=True)
        load_receiver_rank = (
            self._get_rank_by_group_position(group_id, failed_rank_in_group, world_size)
            if world_size >= n
            else failed_rank_in_group
        )
        # Place SW recovery ports after load ports: base + 1000 + group*100 + 100 (offset from load)
        sw_base = base_port + 1200 + group_id * 200
        sw_ports = [sw_base + j for j in range(num_blocks)]

        # Exchange IPs
        rank_ips = {}
        if torch.distributed.is_initialized():
            try:
                ip_list = [None] * world_size
                torch.distributed.all_gather_object(ip_list, base_ip)
                for r, ip in enumerate(ip_list):
                    rank_ips[r] = ip
            except Exception:
                for r in range(world_size):
                    rank_ips[r] = base_ip
        receiver_ip = rank_ips.get(load_receiver_rank, base_ip)

        connection_key = (
            rank,
            world_size,
            self.basic_ec_k,
            self.basic_ec_n,
            failed_rank_in_group,
            group_id,
            rank_in_group,
            receiver_ip,
            tuple(sw_ports),
            self.use_rdma,
            os.environ.get("BASIC_EC_INTERFACE", os.environ.get("ECNAIVE_INTERFACE")),
        )
        if self._sw_recovery_connection_key is not None:
            if self._sw_recovery_connection_key != connection_key:
                raise RuntimeError(
                    "BasicEC software recovery connection topology changed while "
                    "persistent channels are live; restart the process"
                )
            native.reset_load_timing_stats()
            logger.info("BasicEC software in-process connection cache=hit rank=%d", rank)
            torch.distributed.barrier()
            return

        logger.info("BasicEC software in-process connection cache=miss rank=%d", rank)
        # Phase 1: bind/listen (receiver) + RDMA CQs (all ranks)
        logger.debug(f"BasicEC: [Rank {rank}] SW recovery phase 1: {num_blocks} ports")
        native.init_basic_ec_load_sw_bind_listen(rank_in_group, receiver_ip, num_blocks, sw_ports)
        torch.distributed.barrier()

        # Phase 2: receiver accepts (blocking), senders each connect to exactly one port.
        # Non-participating ranks skip entirely — they have no data block for the failed rank.
        if rank_in_group == failed_rank_in_group:
            logger.debug(
                "BasicEC: [Rank %d] SW recovery phase 2: accepting %d connections", rank, num_blocks
            )
            native.init_basic_ec_load_sw_accept(rank_in_group, num_blocks)
        else:
            block_idx = self.get_sw_recovery_block_idx_for_sender(
                rank_in_group, failed_rank_in_group=failed_rank_in_group
            )
            if block_idx >= 0:
                logger.debug(
                    "BasicEC: [Rank %d] SW recovery phase 2: connecting block_idx=%d " "port=%d",
                    rank,
                    block_idx,
                    sw_ports[block_idx],
                )
                native.init_basic_ec_load_sw_connect_one(
                    rank_in_group, receiver_ip, block_idx, sw_ports[block_idx]
                )
            else:
                logger.debug("BasicEC: [Rank %d] SW recovery phase 2: no block, skipping", rank)
        torch.distributed.barrier()
        self._sw_recovery_connection_key = connection_key
        native.reset_load_timing_stats()
        logger.debug(
            "BasicEC: [Rank %d] SW recovery connections ready (%d blocks)", rank, num_blocks
        )

    def get_sw_recovery_block_idx_for_sender(
        self, sender_rank_in_group: int, failed_rank_in_group: int = 2
    ) -> int:
        """Map sender's rank_in_group to the block_idx they hold for the failed rank.

        During save, rank f's data block j is sent to rank (f+j)%n.
        So rank s holds block j if s = (f+j)%n, i.e. j = (s-f)%n.
        Returns block_idx = j-1 (0-indexed for d_{f,1}..d_{f,k-1}).
        Returns -1 if this sender doesn't hold any data block.
        """
        n = self.basic_ec_n
        k = self.basic_ec_k
        j = (sender_rank_in_group - failed_rank_in_group + n) % n
        if 1 <= j < k:
            return j - 1  # block_idx 0 = d_{f,1}, ..., k-2 = d_{f,k-1}
        return -1

    def get_sw_recovery_sender_rank_for_block(
        self, block_idx: int, failed_rank_in_group: int = 2, world_size: int = 8
    ) -> int:
        """Get the global rank of the sender holding data block j = block_idx+1."""
        j = block_idx + 1
        n = self.basic_ec_n
        group_id = self._get_group_id(
            self._get_rank_by_group_position(0, failed_rank_in_group, world_size), world_size
        )
        sender_rig = (failed_rank_in_group + j) % n
        return self._get_rank_by_group_position(group_id, sender_rig, world_size)

    def get_software_recovery_source_ranks(
        self, failed_global_rank: int, world_size: int
    ) -> List[Tuple[int, int, int]]:
        """Compute block source mapping for software recovery.

        For a failed rank f with rank_in_group = r_f:
        - d_{f,0} is on rank f itself (own_data0, readable from local disk)
        - d_{f,j} for j in [1, k-1] is stored on rank (f + j) mod n

        Returns list of (source_global_rank, data_chunk_index_j, recv_slot_index)
        for each data block that needs to be fetched from a peer rank.
        recv_slot_index is the index into the recv_blocks list on the source rank.

        Example for k=6, n=8, failed rank f with r_f=3:
          d_{f,1}: source = rank (f+1), recv_slot = 0 (data from r_f-1 = r_f-1)
          Wait: recv slot 0 on rank r has data from rank (r-1) mod n.
          For source rank s = (f+j) mod n: which recv slot has data from rank f?
          On rank s, recv slot (s-f-1) mod (n-1) has data from rank f.
          Since s-f = j, recv_slot = j-1.
          So: d_{f,j} is in recv slot j-1 on rank s = (f+j) mod n.
        """
        k = self.basic_ec_k
        n = self.basic_ec_n
        group_id = self._get_group_id(failed_global_rank, world_size)
        failed_rig = self._get_rank_in_group(failed_global_rank, world_size)

        sources: List[Tuple[int, int, int]] = []
        for j in range(1, k):
            source_rig = (failed_rig + j) % n
            source_global = self._get_rank_by_group_position(group_id, source_rig, world_size)
            recv_slot = j - 1  # recv slot j-1 on source rank contains d_{f,j}
            sources.append((source_global, j, recv_slot))

        return sources

    def get_multi_failure_recovery_plan(
        self, failed_global_ranks: List[int], world_size: int
    ) -> Dict[int, Dict]:
        """Compute recovery plan for 1-2 failed ranks with RS decoding.

        For each failed rank f:
          lost_positions: data block indices j where d_{f,j} is on a failed rank
          surviving: list of (source_rank, block_label, recv_slot, data_chunk_or_parity)

        Returns dict keyed by failed_global_rank.
        """
        k = self.basic_ec_k
        n = self.basic_ec_n
        if k < 2 or n != k + 2 or world_size <= 0 or world_size % n != 0:
            raise ValueError(
                f"BasicEC recovery requires n=k+2 dividing world_size; "
                f"got k={k}, n={n}, world_size={world_size}"
            )
        try:
            failed_global_ranks = sorted({int(rank) for rank in failed_global_ranks})
        except (TypeError, ValueError) as exc:
            raise ValueError("BasicEC failed ranks must be integers") from exc
        if not failed_global_ranks:
            raise ValueError("BasicEC recovery requires at least one failed rank")
        invalid = [rank for rank in failed_global_ranks if rank < 0 or rank >= world_size]
        if invalid:
            raise ValueError(f"BasicEC failed ranks {invalid} are outside [0, {world_size - 1}]")
        failed_set = set(failed_global_ranks)
        group_counts: Dict[int, int] = {}
        for failed in failed_global_ranks:
            gid = self._get_group_id(failed, world_size)
            group_counts[gid] = group_counts.get(gid, 0) + 1
        excessive = {gid: count for gid, count in group_counts.items() if count > 2}
        if excessive:
            raise ValueError(
                f"BasicEC RS({n},{k}) supports at most two failed ranks per group; "
                f"observed {excessive}"
            )

        # Map each failed rank to group/rig
        failed_info = {}
        for fr in failed_global_ranks:
            failed_info[fr] = {
                'group_id': self._get_group_id(fr, world_size),
                'rank_in_group': self._get_rank_in_group(fr, world_size),
            }

        result: Dict[int, Dict] = {}
        for fr in failed_global_ranks:
            gid = failed_info[fr]['group_id']
            rig = failed_info[fr]['rank_in_group']

            # Which data blocks are lost?
            lost_positions = []
            for j in range(k):
                source_rig = (rig + j) % n
                source_global = self._get_rank_by_group_position(gid, source_rig, world_size)
                if source_global in failed_set:
                    lost_positions.append(j)

            # Which blocks survive and where are they?
            surviving: List[Tuple[int, str, int, Any]] = []
            for j in range(k):
                if j in lost_positions:
                    continue
                source_rig = (rig + j) % n
                source_global = self._get_rank_by_group_position(gid, source_rig, world_size)
                recv_slot = j - 1 if j > 0 else -1  # own_data0 has no recv slot
                surviving.append((source_global, f'data_{j}', recv_slot, j))

            # Parity blocks (may also be on failed ranks!)
            for pi, pname in enumerate(['parity0', 'parity1']):
                source_rig = (rig + k + pi) % n
                source_global = self._get_rank_by_group_position(gid, source_rig, world_size)
                if source_global not in failed_set:
                    surviving.append((source_global, pname, -1, pname))

            recv_block_recovery = self._compute_recv_block_recovery(fr, failed_set, world_size)

            result[fr] = {
                'lost_positions': lost_positions,
                'surviving': surviving,
                'recv_block_recovery': recv_block_recovery,
                'rank_in_group': rig,
                'group_id': gid,
            }

        return result

    def _compute_recv_block_recovery(
        self, failed_rank: int, failed_set: set, world_size: int
    ) -> List[Dict[str, Any]]:
        """Compute recovery plan for the recv blocks on a failed rank.

        Each recv block on the failed rank belongs to a different rank's codeword.
        This method determines:
        - Which rank's codeword the block belongs to (owner_rank)
        - Whether the owner is also failed (owner_is_failed)
        - The block type (data_j, parity0, parity1)
        - The recovery method (decode for data blocks, encode for parity blocks)

        For k=2 with n=4, recv slot layout on rank_in_group f:
          recv_0: d_{(f-1),1}       → owner = f-1, type = data_1,  method = decode
          recv_1: p_{(f-2),0}       → owner = f-2, type = parity0, method = encode
          recv_2: p_{(f-3),1}       → owner = f-3, type = parity1, method = encode

        Args:
            failed_rank (int): Global rank that failed.
            failed_set (set): Set of all failed global ranks.
            world_size (int): Total world size.

        Returns:
            List[Dict]: One dict per recv block with keys:
                recv_idx, owner_rank, owner_is_failed, block_type, recovery_method
        """
        n = self.basic_ec_n
        k = self.basic_ec_k
        rig = self._get_rank_in_group(failed_rank, world_size)
        group_id = self._get_group_id(failed_rank, world_size)

        recv_info: List[Dict[str, Any]] = []
        for recv_idx in range(n - 1):
            if recv_idx < k - 1:
                # Data block d_{(f - recv_idx - 1), recv_idx + 1}
                owner_rig = (rig - recv_idx - 1 + n) % n
                block_type = f'data_{recv_idx + 1}'
                recovery_method = 'decode'
            elif recv_idx == k - 1:
                # Parity0: p_{(f - k), 0}
                owner_rig = (rig - k + n) % n
                block_type = 'parity0'
                recovery_method = 'encode'
            else:  # recv_idx == k
                # Parity1: p_{(f - k - 1), 1}
                owner_rig = (rig - k - 1 + n) % n
                block_type = 'parity1'
                recovery_method = 'encode'

            owner_rank = self._get_rank_by_group_position(group_id, owner_rig, world_size)
            recv_info.append(
                {
                    'recv_idx': recv_idx,
                    'owner_rank': owner_rank,
                    'owner_rig': owner_rig,
                    'owner_is_failed': owner_rank in failed_set,
                    'block_type': block_type,
                    'recovery_method': recovery_method,
                }
            )

        return recv_info

    def get_send_channel_for_target(
        self, source_global_rank: int, target_global_rank: int, world_size: int
    ) -> int:
        """Return the send channel index on source_rank that connects to target_rank.

        During save, rank i's send channel j connects to rank (i + j + 1) mod n.
        So to send from source to target: channel = (target_rig - source_rig - 1 + n) % n.
        """
        n = self.basic_ec_n
        src_rig = self._get_rank_in_group(source_global_rank, world_size)
        tgt_rig = self._get_rank_in_group(target_global_rank, world_size)
        ch = (tgt_rig - src_rig - 1 + n) % n
        if ch >= n - 1:
            raise ValueError(
                f"No send channel from rank {source_global_rank} (rig={src_rig}) "
                f"to rank {target_global_rank} (rig={tgt_rig})"
            )
        return ch

    def get_recv_channel_from_source(
        self, my_global_rank: int, source_global_rank: int, world_size: int
    ) -> int:
        """Return the recv channel index on my_rank that receives from source_rank.

        During save, recv channel j on rank r receives from rank (r - j - 1) mod n.
        So to receive from source on my rank: channel = (my_rig - src_rig - 1 + n) % n.
        """
        n = self.basic_ec_n
        my_rig = self._get_rank_in_group(my_global_rank, world_size)
        src_rig = self._get_rank_in_group(source_global_rank, world_size)
        ch = (my_rig - src_rig - 1 + n) % n
        if ch >= n - 1:
            raise ValueError(
                f"No recv channel on rank {my_global_rank} (rig={my_rig}) "
                f"from rank {source_global_rank} (rig={src_rig})"
            )
        return ch

    def init_basic_ec_if_enabled(self):
        """Initialize BasicEC C++ module if enabled and distributed environment is ready."""
        if self._basic_ec_native is not None:
            logger.debug("BasicEC: Already initialized, skipping")
            return

        try:
            from megatron.training import get_args as input_args

            args = input_args()
            self.use_basic_ec = args.use_basic_ec
            if not getattr(args, 'use_basic_ec', False):
                return

            # Read RS k parameter (number of data blocks)
            self.basic_ec_k = getattr(args, 'basic_ec_rs_k', 2)
            if self.basic_ec_k < 2:
                raise RuntimeError(f"BasicEC: basic_ec_rs_k must be >= 2, got {self.basic_ec_k}")
            self.basic_ec_n = self.basic_ec_k + 2
            # Ports per rank: (n-1) ASIO send + (n-1) ASIO recv + (n-1) RDMA recv
            self.basic_ec_ports_per_rank = 3 * (self.basic_ec_n - 1)

            logger.debug(
                f"BasicEC: RS scheme {self.basic_ec_k}+2 → {self.basic_ec_n} ranks/group, "
                f"{self.basic_ec_ports_per_rank} ports/rank"
            )

            # Check RDMA flag
            self.use_rdma = getattr(args, 'use_rdma', False)
            logger.debug(f"BasicEC: RDMA support {'enabled' if self.use_rdma else 'disabled'}")

            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning(
                    "BasicEC: Distributed environment not initialized, "
                    "skipping BasicEC initialization"
                )
                return

            # Initialize BasicEC C++ module
            self._init_basic_ec_native()

            # Start persistent buffer poller thread
            # BasicEC does not use layerwise mode
            self._start_buffer_poller_thread()

        except Exception as e:
            logger.warning(f"BasicEC: Failed to initialize during manager initialization: {e}")
            self._basic_ec_native = None

    def _init_basic_ec_native(self):
        """Initialize BasicEC C++ native module."""
        basic_ec_native = None
        try:
            # Direct import .so file without modifying sys.path or affecting other packages
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Find .so file
            import glob as _glob_module

            so_files = _glob_module.glob(os.path.join(current_dir, "basic_ec_native*.so"))

            if not so_files:
                raise ImportError(f"No basic_ec_native.so file found in {current_dir}")

            # Load .so file directly using importlib
            import importlib.util as _importlib_util

            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("basic_ec_native", so_path)
            basic_ec_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(basic_ec_native)
            logger.debug(f"BasicEC: Loaded .so file from {so_path}")

            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            # BasicEC only uses ASIO (no NCCL support)
            # Create instance with error handling
            try:
                # ===== ASIO Initialization Path =====
                logger.debug(f"BasicEC: [Rank {rank}] Using ASIO for communication")

                # Get network configuration
                net_config = self._get_basic_ec_network_config(rank, world_size)

                # Synchronize all ranks before creating C++ instances
                logger.debug(
                    f"BasicEC: [Rank {rank}] Synchronizing all ranks before creating "
                    "C++ native module (ASIO)..."
                )
                torch.distributed.barrier()
                logger.debug(
                    f"BasicEC: [Rank {rank}] All ranks synchronized, creating C++ "
                    "native module with ASIO..."
                )

                # Create C++ instance with ASIO parameters
                logger.debug(
                    "BasicEC: Creating C++ native module with ASIO (this will block "
                    "until ASIO connections are established)..."
                )

                # The native constructor receives variable-length ASIO endpoint
                # lists, RDMA exchange port lists, RS k, transport mode, and the
                # rank's position in its n = k + 2 group.

                # Calculate round-robin partner ranks
                partner_ranks = self._get_round_robin_ranks(rank, world_size)

                base_port = net_config['base_port']
                rank_ips = net_config['rank_ips']
                n = self.basic_ec_n
                ports_per_rank = self.basic_ec_ports_per_rank

                # Per-rank port base helper (generalized)
                def _port_base_for_rank(r: int) -> int:
                    if world_size >= n and world_size % n == 0:
                        gid = self._get_group_id(r, world_size)
                        rig = self._get_rank_in_group(r, world_size)
                        return base_port + gid * (n * ports_per_rank) + rig * ports_per_rank
                    return base_port + r * ports_per_rank

                # Generalized connection setup:
                # For each send channel j in [0, n-2]:
                #   - Connect to send_partners[j]'s recv port: offset (n-1) + j
                #   - RDMA: connect to send_partners[j]'s RDMA recv port: offset 2*(n-1) + j
                # For each recv channel j:
                #   - Listen on local recv port: offset (n-1) + j
                #   - RDMA listen on local RDMA recv port: offset 2*(n-1) + j

                send_partners = partner_ranks['send_partners']
                num_channels = n - 1  # = k + 1 send channels = k + 1 recv channels

                # Build send connection params: (partner_ip, partner_recv_port) for each channel
                send_ips = []
                send_ports_to = []
                for j, target_r in enumerate(send_partners):
                    partner_port_base = _port_base_for_rank(target_r)
                    send_ips.append(rank_ips.get(target_r, net_config['my_ip']))
                    send_ports_to.append(partner_port_base + (n - 1) + j)  # partner's recv_ports[j]

                # Build recv connection params: (local_ip, local_recv_port) for each channel
                recv_ips = [net_config['my_ip']] * num_channels
                recv_ports_local = net_config['ports'][
                    'recv_ports'
                ]  # list of local recv listen ports

                # Build partner and local RDMA exchange ports for each channel.
                rdma_send_ports_to = []
                rdma_recv_ports_local = net_config['ports']['rdma_recv_ports']
                for j, target_r in enumerate(send_partners):
                    partner_port_base = _port_base_for_rank(target_r)
                    rdma_send_ports_to.append(
                        partner_port_base + 2 * (n - 1) + j
                    )  # partner's rdma_recv_ports[j]

                # Get rank_in_group for RDMA connection ordering
                rank_in_group = self._get_rank_in_group(rank, world_size)

                # Create C++ instance with generalized connection lists
                self._basic_ec_native = basic_ec_native.BasicECNative(
                    send_ips,
                    send_ports_to,
                    recv_ips,
                    recv_ports_local,
                    rdma_send_ports_to,
                    rdma_recv_ports_local,
                    self.basic_ec_k,
                    self.use_rdma,
                    rank_in_group,
                )

                # If we reach here, ASIO connections are ready and threads are running
                logger.debug(
                    "BasicEC: C++ native module initialized successfully with ASIO "
                    f"(rank={rank}, world_size={world_size})"
                )

                # Initialize BasicEC buffers
                # BasicEC does not use layerwise mode
                self._init_basic_ec_buffers()

                # Synchronize all ranks after RDMA/ASIO and buffers are ready
                logger.debug(
                    f"BasicEC: [Rank {rank}] Synchronizing all ranks after native module init..."
                )
                torch.distributed.barrier()
                logger.debug(f"BasicEC: [Rank {rank}] All ranks synchronized after BasicEC init")

            except Exception as e:
                logger.warning(f"BasicEC: Failed to create C++ native module instance: {e}")
                # Try to stop the pipeline if it was partially created
                try:
                    if hasattr(self, '_basic_ec_native') and self._basic_ec_native is not None:
                        self._basic_ec_native.stop()
                except Exception:
                    logger.debug(
                        "BasicEC: Failed to stop partially initialized native module", exc_info=True
                    )
                self._basic_ec_native = None
                raise e

        except ImportError as e:
            logger.warning(
                f"BasicEC: C++ native module not available: {e}; "
                "BasicEC functionality will not work"
            )
            self._basic_ec_native = None
        except Exception as e:
            logger.warning(
                f"BasicEC: Failed to initialize C++ native module: {e}; "
                "BasicEC functionality will not work"
            )
            self._basic_ec_native = None

    def _init_basic_ec_buffers(self):
        """Initialize BasicEC buffers during C++ module initialization.

        Only pooled data and parity buffers are allocated here. The strategy
        allocates one local and n - 1 received persistent blocks after metadata exchange.
        """
        rank = torch.distributed.get_rank()
        logger.debug("BasicEC: Initializing buffers for BasicEC (data and parity pools only)")

        # Allocate data buffers for storing original tensor data
        self.basic_ec_data_buffers = self._allocate_data_buffers()

        # Allocate parity buffers (pooled) for parity blocks
        self.basic_ec_parity_buffers = self._allocate_parity_buffers()

        # Register buffers for RDMA if enabled
        if self.use_rdma and self._basic_ec_native is not None:
            logger.debug(f"BasicEC: [Rank {rank}] Registering data and parity buffers for RDMA...")
            for buffer in self.basic_ec_data_buffers:
                self.register_buffer(buffer)
            for buffer in self.basic_ec_parity_buffers:
                self.register_buffer(buffer)
            logger.debug(f"BasicEC: [Rank {rank}] All pooled buffers registered for RDMA")

        # Initialize free buffer queues
        self._free_data_buffer_queue = queue.Queue()
        for buffer in self.basic_ec_data_buffers:
            self._free_data_buffer_queue.put(int(buffer.data_ptr()))

        self._free_parity_buffer_queue = queue.Queue()
        for buffer in self.basic_ec_parity_buffers:
            self._free_parity_buffer_queue.put(int(buffer.data_ptr()))

        logger.debug(
            f"BasicEC: Buffer initialization completed - "
            f"Data buffers: {len(self.basic_ec_data_buffers)}, "
            f"Parity buffers: {len(self.basic_ec_parity_buffers)}"
        )

    def _allocate_data_buffers(self):
        """Allocate data buffers for storing original tensor data."""
        logger.debug(
            "BasicEC: Allocating data buffers (%d buffers, %dMB each)",
            self.basic_ec_data_buffers_count,
            self.basic_ec_buffer_size // (1024 * 1024),
        )

        data_buffers = []
        for i in range(self.basic_ec_data_buffers_count):
            buffer = allocate_hugepage_tensor(
                self.basic_ec_buffer_size,
                fallback_pin_memory=self.basic_ec_pin_memory,
                touch_pages=True,
            )
            data_buffers.append(buffer)
            logger.debug(f"BasicEC: Allocated data buffer {i}: {self.basic_ec_buffer_size} bytes")

        logger.debug(f"BasicEC: Allocated {len(data_buffers)} data buffers")
        return data_buffers

    def _allocate_parity_buffers(self):
        """Allocate parity buffers (pooled) for parity blocks."""
        logger.debug(
            "BasicEC: Allocating parity buffers (%d buffers, %dMB each)",
            self.basic_ec_parity_buffers_count,
            self.basic_ec_buffer_size // (1024 * 1024),
        )

        parity_buffers = []
        for i in range(self.basic_ec_parity_buffers_count):
            buffer = allocate_hugepage_tensor(
                self.basic_ec_buffer_size,
                fallback_pin_memory=self.basic_ec_pin_memory,
                touch_pages=True,
            )
            parity_buffers.append(buffer)
            logger.debug(f"BasicEC: Allocated parity buffer {i}: {self.basic_ec_buffer_size} bytes")

        logger.debug(f"BasicEC: Allocated {len(parity_buffers)} parity buffers")
        return parity_buffers

    def _poll_and_release_buffers(self):
        """Poll C++ for buffers ready to be released and put them back to queues."""
        if self._basic_ec_native is None:
            return

        # Get data buffers ready for release
        data_buffers = self._basic_ec_native.get_data_buffers_to_release()
        for data_addr in data_buffers:
            try:
                self._free_data_buffer_queue.put_nowait(data_addr)
                # logger.debug(f"BasicEC: Released data buffer at address {data_addr}")
            except Exception:
                logger.error(
                    f"BasicEC: Data buffer queue is full, cannot release buffer {data_addr}"
                )

        # Get parity buffers ready for release
        parity_buffers = self._basic_ec_native.get_parity_buffers_to_release()
        for parity_addr in parity_buffers:
            try:
                self._free_parity_buffer_queue.put_nowait(parity_addr)
                # logger.debug(f"BasicEC: Released parity buffer at address {parity_addr}")
            except Exception:
                logger.error(
                    f"BasicEC: Parity buffer queue is full, cannot release buffer {parity_addr}"
                )

    def _start_buffer_poller_thread(self):
        """Start a persistent background thread to poll and release buffers."""
        if hasattr(self, '_buffer_poller_thread') and self._buffer_poller_thread is not None:
            logger.warning("BasicEC: Buffer poller thread already started")
            return

        # Create control events
        self._buffer_poller_stop_event = threading.Event()
        self._buffer_poller_active_event = threading.Event()

        def buffer_poller_worker():
            """Persistent background thread that polls for buffer releases."""
            logger.debug("BasicEC: Buffer poller thread started")
            poll_count = 0

            while not self._buffer_poller_stop_event.is_set():
                # Only poll when active
                if self._buffer_poller_active_event.is_set():
                    self._poll_and_release_buffers()
                    poll_count += 1
                    if poll_count % 1000 == 0:
                        logger.debug(f"BasicEC: Buffer poller running (polled {poll_count} times)")

                # Sleep briefly to avoid busy waiting
                from time import sleep

                sleep(0.001)  # 1ms

            logger.debug("BasicEC: Buffer poller thread stopping")

        # Start the daemon thread
        self._buffer_poller_thread = threading.Thread(target=buffer_poller_worker, daemon=True)
        self._buffer_poller_thread.start()
        logger.debug("BasicEC: Buffer poller thread created and started")

    def _stop_buffer_poller_thread(self):
        """Stop the persistent buffer poller thread."""
        if not hasattr(self, '_buffer_poller_thread') or self._buffer_poller_thread is None:
            return

        logger.debug("BasicEC: Stopping buffer poller thread...")

        # Signal the thread to stop
        if self._buffer_poller_stop_event:
            self._buffer_poller_stop_event.set()

        # Wait for thread to finish
        if self._buffer_poller_thread.is_alive():
            self._buffer_poller_thread.join(timeout=2.0)
            if self._buffer_poller_thread.is_alive():
                logger.warning("BasicEC: Buffer poller thread did not stop in time")
            else:
                logger.debug("BasicEC: Buffer poller thread stopped successfully")

        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None

    def get_basic_ec_buffers(self):
        """Get BasicEC buffers for FileSystemWriterAsync.

        Returns pooled data and parity buffers. The strategy allocates one local
        and n - 1 received persistent blocks after metadata exchange.

        Returns:
            Dict containing all buffer information, or None if not initialized
        """
        if self.basic_ec_data_buffers is None:
            return None

        return {
            'data_buffers': self.basic_ec_data_buffers,
            'parity_buffers': self.basic_ec_parity_buffers,
            'free_data_buffer_queue': self._free_data_buffer_queue,
            'free_parity_buffer_queue': self._free_parity_buffer_queue,
            # Pass buffer poller control objects
            'buffer_poller_active_event': self._buffer_poller_active_event,
            'poll_and_release_buffers': self._poll_and_release_buffers,
            # Persistent local/received blocks are allocated by the strategy.
        }

    def register_buffer(self, buffer: torch.Tensor):
        """Register buffer for RDMA operations (similar to Gemini).

        Args:
            buffer: PyTorch tensor to register for RDMA
        """
        if not self.use_rdma or self._basic_ec_native is None:
            return

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        buffer_size = buffer.numel() * buffer.element_size()

        # Check if already registered
        if buffer_addr in self.registered_buffers:
            logger.debug(
                "BasicEC: [Rank %d] Buffer already registered at 0x%x (size: %.2f MB)",
                rank,
                buffer_addr,
                buffer_size / (1024**2),
            )
            return

        try:
            logger.debug(
                "BasicEC: [Rank %d] Registering buffer at 0x%x, size: %.2f GB, "
                "numel: %d, dtype: %s (iteration %d)",
                rank,
                buffer_addr,
                buffer_size / (1024**3),
                buffer.numel(),
                buffer.dtype,
                self.current_iteration,
            )
            self._basic_ec_native.register_buffer(buffer_addr, buffer_size)
            self.registered_buffers[buffer_addr] = (buffer_size, self.current_iteration)
            logger.debug(
                "BasicEC: [Rank %d] Buffer registered successfully (total registered: %d)",
                rank,
                len(self.registered_buffers),
            )

            # Print all registered buffers
            logger.debug(f"BasicEC: [Rank {rank}] All registered buffers:")
            # for addr, (size, iteration) in self.registered_buffers.items():
            #     logger.debug(f"  - 0x{addr:x}: {size / (1024**2):.2f} MB (iteration {iteration})")
        except Exception as e:
            logger.error(f"BasicEC: [Rank {rank}] Failed to register buffer: {e}")
            raise

    def unregister_buffer(self, buffer: torch.Tensor):
        """Unregister buffer from RDMA.

        Args:
            buffer: PyTorch tensor to unregister
        """
        if not self.use_rdma or self._basic_ec_native is None:
            return

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()

        if buffer_addr not in self.registered_buffers:
            return

        try:
            logger.debug(f"BasicEC: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x}")
            self._basic_ec_native.unregister_buffer(buffer_addr)
            del self.registered_buffers[buffer_addr]
            logger.debug(
                "BasicEC: [Rank %d] Buffer unregistered successfully (remaining: %d)",
                rank,
                len(self.registered_buffers),
            )
        except Exception as e:
            logger.error(f"BasicEC: [Rank {rank}] Failed to unregister buffer: {e}")
            raise

    # ===== In-process HW recovery workspace =====

    def recovery_workspace_matches(self, key: tuple) -> bool:
        """Return whether the active recovery workspace has this identity."""
        return self._recovery_workspace_key == key

    def begin_recovery_workspace(self, key: tuple) -> bool:
        """Select a stable workspace, rejecting unsafe in-process replacement."""
        if self._recovery_workspace_key is None:
            self._recovery_workspace_key = key
            return False
        if self._recovery_workspace_key != key:
            raise RuntimeError(
                "BasicEC in-process recovery workspace identity changed while "
                "native/RDMA resources are live. Restart the process for a different "
                "checkpoint, failure set, topology, or transport configuration."
            )
        return True

    def get_recovery_workspace_value(self, name: str, default=None):
        """Read an object owned by the active recovery workspace."""
        return self._recovery_workspace.get(name, default)

    def set_recovery_workspace_value(self, name: str, value: Any) -> None:
        """Keep an object alive for subsequent in-process recovery rounds."""
        self._recovery_workspace[name] = value

    def get_recovery_slices(
        self, name: str, slice_bytes: int, count: int, *, register: bool = False
    ) -> List[torch.Tensor]:
        """Return page-backed slices with stable addresses across recovery rounds."""
        cached = self._recovery_workspace.get(name)
        if cached is not None:
            if len(cached) != count or any(buf.numel() < slice_bytes for buf in cached):
                raise RuntimeError(
                    f"BasicEC cached recovery slices {name!r} no longer match "
                    f"count={count}, bytes={slice_bytes}"
                )
            return [buf[:slice_bytes] for buf in cached]
        slices = list(
            allocate_hugepage_slices(slice_bytes, count, fallback_pin_memory=True, touch_pages=True)
        )
        if register and self.use_rdma:
            for buffer in slices:
                self.register_buffer(buffer)
        self._recovery_workspace[name] = slices
        return slices

    def get_recovery_buffer(
        self, name: str, size_bytes: int, *, pin: bool = False, register: bool = False
    ) -> torch.Tensor:
        """Return one stable recovery buffer without reallocating or retouching it."""
        cached = self._recovery_workspace.get(name)
        if cached is not None:
            if cached.numel() < size_bytes:
                raise RuntimeError(
                    f"BasicEC cached recovery buffer {name!r} is too small: "
                    f"{cached.numel()} < {size_bytes}"
                )
            return cached[:size_bytes]
        if pin and torch.cuda.is_available():
            try:
                buffer = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
            except Exception:
                buffer = allocate_hugepage_tensor(
                    size_bytes, fallback_pin_memory=True, touch_pages=True
                )
        else:
            buffer = allocate_hugepage_tensor(size_bytes, fallback_pin_memory=pin, touch_pages=True)
        if register and self.use_rdma:
            self.register_buffer(buffer)
        self._recovery_workspace[name] = buffer
        return buffer

    # ===== HW recovery: recovered checkpoint block storage =====

    def store_recovered_blocks(self, rank: int, blocks: Dict[str, torch.Tensor]) -> None:
        """Store recovered checkpoint blocks for cascading failure tolerance.

        After HW recovery, the failed rank's n blocks (own_data0 + recv blocks)
        are kept in memory so they can serve as source blocks if another rank
        in the same group fails before the next checkpoint save.

        Args:
            rank (int): Global rank whose blocks were recovered.
            blocks (Dict[str, torch.Tensor]): Dict with keys like
                'own_data0', 'recv_0', ..., 'recv_{n-2}'.
        """
        self._recovered_blocks[rank] = blocks
        logger.debug(f"BasicEC: Stored {len(blocks)} recovered blocks for rank {rank}")

    def get_recovered_blocks(self, rank: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get recovered checkpoint blocks for a rank, or None.

        Args:
            rank (int): Global rank.

        Returns:
            Optional[Dict[str, torch.Tensor]]: Recovered blocks dict or None.
        """
        return self._recovered_blocks.get(rank)

    def clear_recovered_blocks(self, rank: Optional[int] = None) -> None:
        """Clear recovered blocks. If rank is None, clear all.

        Args:
            rank (Optional[int]): Specific rank to clear, or None for all.
        """
        if rank is None:
            self._recovered_blocks.clear()
        else:
            self._recovered_blocks.pop(rank, None)

    def cleanup(self):
        """Cleanup BasicEC resources when manager is destroyed."""
        try:
            from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
                release_hugepage_host_registration,
            )

            def _release_host_registrations_for_buffers(buffers) -> None:
                if not buffers:
                    return
                for buffer in buffers:
                    if torch.is_tensor(buffer):
                        release_hugepage_host_registration(buffer)

            # Stop buffer poller thread
            self._stop_buffer_poller_thread()

            if self.use_rdma and self._basic_ec_native is not None:
                for buffer_addr in list(self.registered_buffers.keys()):
                    try:
                        self._basic_ec_native.unregister_buffer(buffer_addr)
                    except Exception as e:
                        logger.warning(
                            f"BasicEC: Failed to unregister buffer at 0x{buffer_addr:x}: {e}"
                        )

            _release_host_registrations_for_buffers(self._cached_blocks)
            if self.preallocated_cpu_buffer is not None:
                release_hugepage_host_registration(self.preallocated_cpu_buffer)
            _release_host_registrations_for_buffers(self.basic_ec_data_buffers)
            _release_host_registrations_for_buffers(self.basic_ec_parity_buffers)

            # Stop the C++ pipeline and null it out so re-init works
            if hasattr(self, '_basic_ec_native') and self._basic_ec_native is not None:
                self._basic_ec_native.stop()
                self._basic_ec_native = None
                logger.debug("BasicEC: C++ native module stopped in manager cleanup")

            # Clear registered buffers and cached allocations
            self.registered_buffers.clear()
            self.preallocated_cpu_buffer = None
            self._cached_blocks = None
            self._cached_block_count = 0
            self._cached_block_size = 0
            self.basic_ec_data_buffers = None
            self.basic_ec_parity_buffers = None
            self._free_data_buffer_queue = None
            self._free_parity_buffer_queue = None
            self._recovery_workspace.clear()
            self._recovery_workspace_key = None
            self._sw_recovery_connection_key = None

        except Exception as e:
            logger.warning(f"BasicEC: Error during manager cleanup: {e}")

    def __del__(self):
        """Cleanup BasicEC resources when manager is destroyed."""
        self.cleanup()
