# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""EC-NAIVE manager for shared ecnaive_native initialization and buffer management."""

import os
import queue
import socket
import threading
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
from dataclasses import replace

from .hugepage_alloc import allocate_hugepage_slices, allocate_hugepage_tensor
from .state_dict_decomposer import GlobalMetadataRegistry, TensorMetadata
from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip

logger = getLogger(__name__)

# Default number of ranks per EC-NAIVE group (backward compatible 2+2 scheme)
DEFAULT_RANKS_PER_GROUP = 4
# Ports per rank for default 2+2 scheme:
# 6 ASIO (send_data1, send_parity0, send_parity1, recv_parity1, recv_parity0, recv_data1)
# + 3 RDMA exchange (rdma_recv_parity1, rdma_recv_parity0, rdma_recv_data1)
DEFAULT_PORTS_PER_RANK = 9


class ECNAIVEManager:
    """Shared manager for EC-NAIVE C++ module initialization and buffer management.
    
    This class provides a singleton instance that manages:
    - EC-NAIVE C++ native module (_ecnaive_native)
    - Buffer allocation and management (data and parity buffers, pooled)
    - Buffer poller thread for releasing buffers
    
    Note: The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) are allocated
    in strategy after metadata exchange, not in manager.
    
    Both TorchDistSaveShardedStrategy and TorchDistLoadShardedStrategy
    can share the same manager instance to reuse initialized resources.
    """
    
    _instance: Optional['ECNAIVEManager'] = None
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

        self._ecnaive_native = None
        self.use_ecnaive = False
        self.use_rdma = False  # New: RDMA support flag

        # EC-NAIVE configuration
        self.ecnaive_k = 2        # number of data blocks (default 2 → 2+2 scheme)
        self.ecnaive_n = 4        # ranks per group = k + 2
        self.ecnaive_ports_per_rank = DEFAULT_PORTS_PER_RANK  # computed from n

        # Buffer configuration
        self.ecnaive_data_buffers_count = 12
        self.ecnaive_parity_buffers_count = 12  # Pooled parity buffers
        self.ecnaive_buffer_size = 64 * 1024 * 1024  # 64MB
        self.ecnaive_pin_memory = True
        
        # Buffers (simplified: only data and parity pools)
        self.ecnaive_data_buffers: Optional[List[torch.Tensor]] = None
        self.ecnaive_parity_buffers: Optional[List[torch.Tensor]] = None  # Pooled parity buffers
        # Note: The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) are allocated in strategy
        
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

        self._initialized = True

    def allocate_preallocated_buffer(self, size_bytes: int):
        """Allocate or reuse cached CPU buffer.  Grows only when needed."""
        if self.preallocated_cpu_buffer is not None:
            if self.preallocated_cpu_buffer.numel() >= size_bytes:
                return
        pin = self.ecnaive_pin_memory and torch.cuda.is_available()
        logger.info(
            f"ECNAIVE: Allocating preallocated buffer: {size_bytes / (1024**3):.2f} GB (pin={pin})"
        )
        self.preallocated_cpu_buffer = allocate_hugepage_tensor(
            size_bytes, fallback_pin_memory=pin, touch_pages=False,
        )

    _cached_block_count: int = 0
    _cached_block_size: int = 0
    _cached_blocks: Optional[List[torch.Tensor]] = None

    def allocate_preallocated_blocks(self, count: int, aligned_size: int):
        """Allocate or reuse cached persistent blocks (n = k+2 blocks)."""
        if (self._cached_blocks is not None and self._cached_block_count == count
                and self._cached_block_size >= aligned_size):
            return self._cached_blocks
        pin = self.ecnaive_pin_memory and torch.cuda.is_available()
        logger.info(
            f"ECNAIVE: Allocating {count} blocks: {aligned_size / (1024**3):.2f} GB each "
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
        n = self.ecnaive_n
        if world_size <= 0:
            return {
                "mode": 0,
                "num_groups": 1,
                "ranks_per_node": 1,
                "num_nodes": 1,
                "clusters": 1,
            }
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
        """Get group id for EC-NAIVE multi-rank setup."""
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
        """Calculate round-robin partner ranks for EC-NAIVE.

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
        For backward compat with 2+2, also includes legacy keys.
        """
        k = self.ecnaive_k
        n = self.ecnaive_n

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
                send_partners.append(self._get_rank_by_group_position(group_id, target_ig, world_size))
                recv_partners.append(self._get_rank_by_group_position(group_id, src_ig, world_size))
                send_block_types.append(f"data_{j}")
                recv_block_types.append(f"data")  # received block is a data block from another rank

            # Parity 0: send to rank_in_group + k
            p0_target_ig = (rank_in_group + k) % n
            p0_src_ig = (rank_in_group - k) % n
            send_partners.append(self._get_rank_by_group_position(group_id, p0_target_ig, world_size))
            recv_partners.append(self._get_rank_by_group_position(group_id, p0_src_ig, world_size))
            send_block_types.append("parity0")
            recv_block_types.append("parity0")

            # Parity 1: send to rank_in_group + k + 1
            p1_target_ig = (rank_in_group + k + 1) % n
            p1_src_ig = (rank_in_group - k - 1) % n
            send_partners.append(self._get_rank_by_group_position(group_id, p1_target_ig, world_size))
            recv_partners.append(self._get_rank_by_group_position(group_id, p1_src_ig, world_size))
            send_block_types.append("parity1")
            recv_block_types.append("parity1")

            result = {
                'send_partners': send_partners,
                'recv_partners': recv_partners,
                'send_block_types': send_block_types,
                'recv_block_types': recv_block_types,
            }

            # Backward compatibility aliases for k=2
            if k == 2:
                result.update({
                    'send_data1_to': send_partners[0],
                    'send_parity0_to': send_partners[1],
                    'send_parity1_to': send_partners[2],
                    'recv_parity1_from': recv_partners[0],
                    'recv_parity0_from': recv_partners[1],
                    'recv_data1_from': recv_partners[2],
                })

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
            result.update({
                'send_data1_to': send_partners[0],
                'send_parity0_to': send_partners[1],
                'send_parity1_to': send_partners[2],
                'recv_parity1_from': recv_partners[0],
                'recv_parity0_from': recv_partners[1],
                'recv_data1_from': recv_partners[2],
            })
        return result
    
    def _get_ecnaive_network_config(self, rank: int, world_size: int) -> dict:
        """
        Get network configuration for EC-NAIVE ASIO connections.
        
        This function:
        1. Gets base IP address (from ECNAIVE_BASE_IP env var, MASTER_ADDR, or auto-detect)
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
                - 'ports': dict - Port numbers for each connection type
                    - 'send_data1': int
                    - 'send_parity0': int
                    - 'send_parity1': int
                    - 'recv_parity1': int
                    - 'recv_parity0': int
                    - 'recv_data1': int
        """
        # Step 1: Get base IP address (with multi-NIC per-rank support)
        base_ip = resolve_ip("ECNAIVE", rank=rank)

        # Step 2: Get base port
        # Priority: ECNAIVE_BASE_PORT > MASTER_PORT + 10000 > default 16000
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('ECNAIVE_BASE_PORT', master_port + 10000))
        
        # Step 3: Calculate ports for this rank
        # Each rank gets 3*(n-1) ports: (n-1) ASIO send + (n-1) ASIO recv + (n-1) RDMA recv
        # Per-group allocation to avoid port conflicts across groups
        n = self.ecnaive_n
        ports_per_rank = self.ecnaive_ports_per_rank
        if world_size >= n and world_size % n == 0:
            group_id = self._get_group_id(rank, world_size)
            rank_in_group = self._get_rank_in_group(rank, world_size)
            port_base = base_port + group_id * (n * ports_per_rank) + rank_in_group * ports_per_rank
        else:
            port_base = base_port + rank * ports_per_rank

        # Build port dicts: generalized lists + backward-compat named keys for k=2
        send_ports = [port_base + i for i in range(n - 1)]
        recv_ports = [port_base + (n - 1) + i for i in range(n - 1)]
        rdma_recv_ports = [port_base + 2 * (n - 1) + i for i in range(n - 1)]

        k = self.ecnaive_k
        ports = {
            'send_ports': send_ports,
            'recv_ports': recv_ports,
            'rdma_recv_ports': rdma_recv_ports,
        }
        # Backward compatibility: named keys for k=2 (original 2+2 scheme)
        if k == 2:
            ports.update({
                'send_data1': send_ports[0],
                'send_parity0': send_ports[1],
                'send_parity1': send_ports[2] if len(send_ports) > 2 else None,
                'recv_parity1': recv_ports[0],
                'recv_parity0': recv_ports[1],
                'recv_data1': recv_ports[2] if len(recv_ports) > 2 else None,
                'rdma_recv_parity1': rdma_recv_ports[0],
                'rdma_recv_parity0': rdma_recv_ports[1],
                'rdma_recv_data1': rdma_recv_ports[2] if len(rdma_recv_ports) > 2 else None,
            })
        
        # Step 4: Exchange IP addresses via torch.distributed.all_gather
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
                    f"EC-NAIVE: [Rank {rank}] IP exchange completed - "
                    f"All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"EC-NAIVE: Failed to exchange IPs via all_gather, using local IP: {e}"
                )
                # Fallback to using local IP for all ranks
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            # Single rank mode - use local IP
            logger.info("EC-NAIVE: Distributed not initialized, using local IP for all ranks")
            rank_ips[0] = base_ip
        
        config = {
            'my_ip': base_ip,
            'base_port': base_port,
            'rank_ips': rank_ips,
            'ports': ports,
        }
        
        logger.info(
            f"EC-NAIVE: [Rank {rank}] Network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  Ports: {config['ports']}\n"
            f"  All rank IPs: {config['rank_ips']}"
        )
        
        return config
    
    def _get_ecnaive_load_network_config(self, rank: int, world_size: int) -> dict:
        """
        Get network configuration for EC-NAIVE load mode (rank2 recovery).
        
        EC-NAIVE load mode needs 8 ports for full recovery:
        - load_recv_rank3_data1: rank2 listens, rank3 connects (for d_{3,1})
        - load_recv_rank0_parity0: rank2 listens, rank0 connects (for p_{2,0})
        - load_recv_rank0_data0: rank2 listens, rank0 connects (for d_{0,0})
        - load_recv_rank1_data1: rank2 listens, rank1 connects (for d_{0,1})
        - load_recv_rank1_data0: rank2 listens, rank1 connects (for d_{1,0})
        - load_recv_rank1_parity1: rank2 listens, rank1 connects (for p_{1,1})
        - load_recv_rank3_data0: rank2 listens, rank3 connects (for d_{3,0})
        - load_recv_rank0_data1: rank2 listens, rank0 connects (for d_{3,1})
        
        Args:
            rank (int): Current rank (should be 2 for receiver, 0/1/3 for senders)
            world_size (int): Total number of ranks
            
        Returns:
            dict: Network configuration with keys:
                - 'my_ip': str - This rank's IP address
                - 'base_port': int - Base port number
                - 'rank_ips': dict - IP addresses for all ranks
                - 'ports': dict - Port numbers for load mode connections (8 ports for rank2)
        """
        # Step 1: Get base IP address (with multi-NIC per-rank support)
        base_ip = resolve_ip("ECNAIVE", rank=rank)

        # Get base port (same as save mode)
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('ECNAIVE_BASE_PORT', master_port + 10000))
        
        # Multi-rank: per-group load ports to avoid conflict
        n = self.ecnaive_n
        num_groups = max(1, world_size // n) if world_size >= n else 1
        group_id = self._get_group_id(rank, world_size)
        rank_in_group = self._get_rank_in_group(rank, world_size)
        # load_receiver_rank: global rank of rank_in_group 2 in this group (for init_ecnaive_load)
        load_receiver_rank = (
            self._get_rank_by_group_position(group_id, 2, world_size)
            if world_size >= n
            else 2
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
                my_ip_tensor = torch.tensor(
                    [int(b) for b in my_ip_bytes], 
                    dtype=torch.uint8
                )
                
                if torch.cuda.is_available():
                    my_ip_tensor = my_ip_tensor.cuda()
                
                ip_list = [torch.zeros_like(my_ip_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(ip_list, my_ip_tensor)
                
                for r, ip_tensor in enumerate(ip_list):
                    ip_bytes = bytes(ip_tensor.cpu().tolist())
                    rank_ips[r] = socket.inet_ntoa(ip_bytes)
                
                logger.info(
                    f"EC-NAIVE: [Rank {rank}] Load mode IP exchange completed - "
                    f"All rank IPs: {rank_ips}"
                )
            except Exception as e:
                logger.warning(
                    f"EC-NAIVE: Failed to exchange IPs via all_gather, using local IP: {e}"
                )
                for r in range(world_size):
                    rank_ips[r] = base_ip
        else:
            logger.info("EC-NAIVE: Distributed not initialized, using local IP for all ranks")
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
        
        logger.info(
            f"EC-NAIVE: [Rank {rank}] Load mode network config:\n"
            f"  My IP: {config['my_ip']}\n"
            f"  Base port: {config['base_port']}\n"
            f"  rank_in_group: {rank_in_group}, group_id: {group_id}, load_receiver_rank: {load_receiver_rank}\n"
            f"  Load mode ports: {config['ports']}\n"
            f"  All rank IPs: {config['rank_ips']}"
        )
        
        return config
    
    def init_ecnaive_load(self, rank: int, world_size: int) -> None:
        """Initialize EC-NAIVE load mode for rank2 recovery.
        
        This method:
        1. Sets load mode in C++ native module
        2. Gets network configuration for load mode (8 ports for full recovery)
        3. Phase 1 (all ranks): RDMA load CQs + rank2 bind/listen on 8 ports
        4. torch.distributed.barrier() so clients never connect before listen
        5. Phase 2 (all ranks): rank2 accepts 8 TCP; rank0/1/3 connect; RDMA load channels
        6. Final barrier after TCP+RDMA setup
        
        Args:
            rank: Current rank (0, 1, 2, or 3)
            world_size: Total number of ranks
        """
        if self._ecnaive_native is None:
            logger.error("EC-NAIVE: Native module not initialized, cannot initialize load mode")
            return
        
        if not self.use_ecnaive:
            logger.warning("EC-NAIVE: Manager not enabled, skipping load initialization")
            return
        
        # Step 1: Set load mode in C++ native module (full recovery: start worker threads)
        failed_rank = 2  # EC-NAIVE recovers rank2
        self._ecnaive_native.set_load_mode(True, failed_rank, rank, is_software_only=False)
        logger.info(f"EC-NAIVE: [Rank {rank}] Set load mode (failed_rank={failed_rank})")
        
        # Step 2: Get network config for current rank (per-group: rank_in_group 2 is receiver)
        net_config = self._get_ecnaive_load_network_config(rank, world_size)
        rank_in_group = net_config['rank_in_group']
        load_receiver_rank = net_config['load_receiver_rank']
        rank2_ip = net_config['rank_ips'].get(load_receiver_rank, net_config['my_ip'])
        
        ports = net_config['ports']
        load_ports = {
            'recv_rank3_data1': ports.get('load_recv_rank3_data1', 0),
            'recv_rank0_parity0': ports.get('load_recv_rank0_parity0', 0),
            'recv_rank0_data0': ports.get('load_recv_rank0_data0', 0),
            'recv_rank1_data1': ports.get('load_recv_rank1_data1', 0),
            'recv_rank1_data0': ports.get('load_recv_rank1_data0', 0),
            'recv_rank1_parity1': ports.get('load_recv_rank1_parity1', 0),
            'recv_rank3_data0': ports.get('load_recv_rank3_data0', 0),
            'recv_rank0_data1': ports.get('load_recv_rank0_data1', 0),
        }
        
        # Step 3–4: Phase 1 on all ranks, then barrier (rank2 must listen before any connect).
        logger.info(f"EC-NAIVE: [Rank {rank}] Load phase 1 (bind/listen on receiver, RDMA CQs on all ranks)...")
        self._ecnaive_native.init_ecnaive_load_bind_listen_only(
            rank_in_group,
            rank2_ip,
            load_ports['recv_rank3_data1'],
            load_ports['recv_rank0_parity0'],
            load_ports['recv_rank0_data0'],
            load_ports['recv_rank1_data1'],
            load_ports['recv_rank1_data0'],
            load_ports['recv_rank1_parity1'],
            load_ports['recv_rank3_data0'],
            load_ports['recv_rank0_data1'],
        )
        torch.distributed.barrier()

        # Step 5: Phase 2 — rank2 blocks on accept; clients connect concurrently (no barrier between them).
        logger.info(f"EC-NAIVE: [Rank {rank}] Load phase 2 (TCP handshake + RDMA load channels)...")
        self._ecnaive_native.init_ecnaive_load_tcp_handshake_and_rdma(
            rank_in_group,
            rank2_ip,
            load_ports['recv_rank3_data1'],
            load_ports['recv_rank0_parity0'],
            load_ports['recv_rank0_data0'],
            load_ports['recv_rank1_data1'],
            load_ports['recv_rank1_data0'],
            load_ports['recv_rank1_parity1'],
            load_ports['recv_rank3_data0'],
            load_ports['recv_rank0_data1'],
        )
        torch.distributed.barrier()
        logger.info(f"EC-NAIVE: [Rank {rank}] Load connections initialized")

    def init_ecnaive_load_software_only(
        self, rank: int, world_size: int, net_config: Optional[dict] = None
    ) -> None:
        """Initialize EC-NAIVE load for software failure only: 1 port (rank3_data1), 1 barrier.
        Use instead of init_ecnaive_load when use_ecnaive_software_failure to avoid 8-port + 2-barrier overhead.
        If net_config is provided (e.g. from caller who already called _get_ecnaive_load_network_config),
        reuse it to avoid duplicate IP all_gather.
        """
        if self._ecnaive_native is None:
            logger.error("EC-NAIVE: Native module not initialized, cannot initialize load mode")
            return
        if not self.use_ecnaive:
            logger.warning("EC-NAIVE: Manager not enabled, skipping load initialization")
            return
        failed_rank = 2
        self._ecnaive_native.set_load_mode(True, failed_rank, rank, is_software_only=True)
        logger.info(f"EC-NAIVE: [Rank {rank}] Set load mode (failed_rank={failed_rank}) for software-only")
        if net_config is None:
            net_config = self._get_ecnaive_load_network_config(rank, world_size)
        rank_in_group = net_config['rank_in_group']
        load_receiver_rank = net_config['load_receiver_rank']
        rank2_ip = net_config['rank_ips'].get(load_receiver_rank, net_config['my_ip'])
        port = net_config['ports'].get('load_recv_rank3_data1', 0)
        torch.distributed.barrier()
        self._ecnaive_native.init_ecnaive_load_connections_software_only(
            rank_in_group, rank2_ip, port
        )
        logger.info(f"EC-NAIVE: [Rank {rank}] Software-only load connection initialized (1 port)")

    # ---- Generalized SW recovery (k-1 ports, any k >= 2) ----

    def init_ecnaive_sw_recovery(self, rank: int, world_size: int,
                                  failed_rank_in_group: int = 2) -> None:
        """Initialize generalized SW recovery with k-1 ports (one per non-local data block).

        Replaces init_ecnaive_load_software_only for k > 2.  Sets up dedicated
        ASIO/RDMA connections from each sender rank (holding one of the failed
        rank's data blocks) to the receiver rank.
        """
        native = self._ecnaive_native
        if native is None:
            raise RuntimeError("EC-NAIVE native module not initialized for SW recovery")
        k = self.ecnaive_k
        num_blocks = k - 1  # d_{f,1} .. d_{f,k-1} from network, d_{f,0} local

        # Network config: k-1 consecutive ports after the save ports
        base_ip = resolve_ip("ECNAIVE", rank=rank)
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        base_port = int(os.environ.get('ECNAIVE_BASE_PORT', master_port + 10000))
        n = self.ecnaive_n
        group_id = self._get_group_id(rank, world_size)
        rank_in_group = self._get_rank_in_group(rank, world_size)
        load_receiver_rank = (
            self._get_rank_by_group_position(group_id, failed_rank_in_group, world_size)
            if world_size >= n else failed_rank_in_group
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

        # Phase 1: bind/listen (receiver) + RDMA CQs (all ranks)
        logger.info(f"EC-NAIVE: [Rank {rank}] SW recovery phase 1: {num_blocks} ports")
        native.init_ecnaive_load_sw_bind_listen(
            rank_in_group, receiver_ip, num_blocks, sw_ports)
        torch.distributed.barrier()

        # Phase 2: accept (receiver) or connect (senders) + RDMA channels
        logger.info(f"EC-NAIVE: [Rank {rank}] SW recovery phase 2: connect")
        native.init_ecnaive_load_sw_connect(
            rank_in_group, receiver_ip, num_blocks, sw_ports)
        torch.distributed.barrier()
        logger.info(f"EC-NAIVE: [Rank {rank}] SW recovery connections ready ({num_blocks} blocks)")

    def get_sw_recovery_block_idx_for_sender(
        self, sender_rank_in_group: int, failed_rank_in_group: int = 2
    ) -> int:
        """Map sender's rank_in_group to the block_idx they hold for the failed rank.

        During save, rank f's data block j is sent to rank (f+j)%n.
        So rank s holds block j if s = (f+j)%n, i.e. j = (s-f)%n.
        Returns block_idx = j-1 (0-indexed for d_{f,1}..d_{f,k-1}).
        Returns -1 if this sender doesn't hold any data block.
        """
        n = self.ecnaive_n
        k = self.ecnaive_k
        j = (sender_rank_in_group - failed_rank_in_group + n) % n
        if 1 <= j < k:
            return j - 1  # block_idx 0 = d_{f,1}, ..., k-2 = d_{f,k-1}
        return -1

    def get_sw_recovery_sender_rank_for_block(
        self, block_idx: int, failed_rank_in_group: int = 2, world_size: int = 8
    ) -> int:
        """Get the global rank of the sender holding data block j = block_idx+1."""
        j = block_idx + 1
        n = self.ecnaive_n
        group_id = self._get_group_id(
            self._get_rank_by_group_position(0, failed_rank_in_group, world_size), world_size)
        sender_rig = (failed_rank_in_group + j) % n
        return self._get_rank_by_group_position(group_id, sender_rig, world_size)

    def allocate_ecnaive_load_recv_buffers(self, global_registry: GlobalMetadataRegistry) -> Dict[str, torch.Tensor]:
        """
        Allocate recv buffers for rank2 load recovery.
        
        EC-NAIVE needs 8 recv buffers for full recovery:
        - p20_from_rank0: for p_{2,0} from rank0 (for recovering data0)
        - d21_from_rank3: for d_{2,1} from rank3 (for recovering data0)
        - d00_from_rank0: for d_{0,0} from rank0 (for recovering recv_parity0)
        - d01_from_rank1: for d_{0,1} from rank1 (for recovering recv_parity0)
        - d10_from_rank1: for d_{1,0} from rank1 (for recovering recv_data1)
        - p11_from_rank1: for p_{1,1} from rank1 (for recovering recv_data1)
        - d30_from_rank3: for d_{3,0} from rank3 (for recovering recv_parity1)
        - d31_from_rank0: for d_{3,1} from rank0 (for recovering recv_parity1)
        
        Each buffer size is aligned_block_size (max_total_bytes).
        
        Args:
            global_registry (GlobalMetadataRegistry): Complete metadata from all ranks
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 8 recv buffers for rank2
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank_in_group = self._get_rank_in_group(rank, world_size)
        
        if rank_in_group != 2:
            logger.warning(
                "EC-NAIVE: allocate_ecnaive_load_recv_buffers called on non rank_in_group 2, returning empty dict"
            )
            return {}
        
        # Calculate maximum data size across all ranks
        max_total_bytes = 0
        for r in range(world_size):
            rank_metadata = global_registry.rank_metadata.get(r, [])
            rank_total_size = sum(meta.size_bytes for meta in rank_metadata)
            if rank_total_size > max_total_bytes:
                max_total_bytes = rank_total_size
        
        # Calculate aligned block size (same as save phase)
        # EC-NAIVE uses full block size (not half like ECLATIN)
        aligned_block_size = ((max_total_bytes + self.ecnaive_buffer_size - 1) // self.ecnaive_buffer_size) * self.ecnaive_buffer_size
        
        logger.info(
            f"EC-NAIVE: Allocating 8 recv buffers for rank2 load recovery\n"
            f"  Pipeline max size: {max_total_bytes / (1024**3):.2f} GB\n"
            f"  Aligned block size (per buffer): {aligned_block_size / (1024**3):.2f} GB\n"
            f"  Total recv memory: {8 * aligned_block_size / (1024**3):.2f} GB"
        )
        
        # Allocate 8 recv buffers for full recovery
        recv_chunks = allocate_hugepage_slices(
            aligned_block_size,
            8,
            fallback_pin_memory=self.ecnaive_pin_memory,
            touch_pages=True,
        )
        recv_buffers = {
            # For recovering data0: d_{2,0} = p_{2,0} ⊕ d_{2,1}
            'p20_from_rank0': recv_chunks[0],  # p_{2,0}
            'd21_from_rank3': recv_chunks[1],  # d_{2,1}
            
            # For recovering recv_parity0: p_{0,0} = d_{0,0} ⊕ d_{0,1}
            'd00_from_rank0': recv_chunks[2],  # d_{0,0}
            'd01_from_rank1': recv_chunks[3],  # d_{0,1}
            
            # For recovering recv_data1: d_{1,1} = d_{1,0} ⊕ p_{1,1}
            'd10_from_rank1': recv_chunks[4],  # d_{1,0}
            'p11_from_rank1': recv_chunks[5],  # p_{1,1}
            
            # For recovering recv_parity1: p_{3,1} = d_{3,0} ⊕ d_{3,1}
            'd30_from_rank3': recv_chunks[6],  # d_{3,0}
            'd31_from_rank0': recv_chunks[7],  # d_{3,1}
        }
        
        # Register buffers for RDMA if enabled
        if self.use_rdma and self._ecnaive_native is not None:
            logger.info(f"EC-NAIVE: [Rank {rank}] Registering load recv buffers for RDMA...")
            for buffer_name, buffer in recv_buffers.items():
                self.register_buffer(buffer)
            logger.info(f"EC-NAIVE: [Rank {rank}] Load recv buffers registered for RDMA")
        
        logger.info(
            f"EC-NAIVE: Allocated 8 recv buffers for rank2: "
            f"{aligned_block_size / (1024**3):.2f} GB each"
        )
        
        return recv_buffers

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
        k = self.ecnaive_k
        n = self.ecnaive_n
        group_id = self._get_group_id(failed_global_rank, world_size)
        failed_rig = self._get_rank_in_group(failed_global_rank, world_size)

        sources: List[Tuple[int, int, int]] = []
        for j in range(1, k):
            source_rig = (failed_rig + j) % n
            source_global = self._get_rank_by_group_position(
                group_id, source_rig, world_size
            )
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
        k = self.ecnaive_k
        n = self.ecnaive_n
        failed_set = set(failed_global_ranks)

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

            result[fr] = {
                'lost_positions': lost_positions,
                'surviving': surviving,
                'rank_in_group': rig,
                'group_id': gid,
            }

        return result

    def get_send_channel_for_target(
        self, source_global_rank: int, target_global_rank: int, world_size: int
    ) -> int:
        """Return the send channel index on source_rank that connects to target_rank.

        During save, rank i's send channel j connects to rank (i + j + 1) mod n.
        So to send from source to target: channel = (target_rig - source_rig - 1 + n) % n.
        """
        n = self.ecnaive_n
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
        n = self.ecnaive_n
        my_rig = self._get_rank_in_group(my_global_rank, world_size)
        src_rig = self._get_rank_in_group(source_global_rank, world_size)
        ch = (my_rig - src_rig - 1 + n) % n
        if ch >= n - 1:
            raise ValueError(
                f"No recv channel on rank {my_global_rank} (rig={my_rig}) "
                f"from rank {source_global_rank} (rig={src_rig})"
            )
        return ch

    def init_ecnaive_if_enabled(self):
        """Initialize EC-NAIVE C++ module if enabled and distributed environment is ready."""
        if self._ecnaive_native is not None:
            logger.debug("EC-NAIVE: Already initialized, skipping")
            return

        try:
            from megatron.training import get_args as input_args
            args = input_args()
            self.use_ecnaive = args.use_ecnaive
            if not getattr(args, 'use_ecnaive', False):
                return

            # Read RS k parameter (number of data blocks)
            self.ecnaive_k = getattr(args, 'ecnaive_rs_k', 2)
            if self.ecnaive_k < 2:
                raise RuntimeError(f"EC-NAIVE: ecnaive_rs_k must be >= 2, got {self.ecnaive_k}")
            self.ecnaive_n = self.ecnaive_k + 2
            # Ports per rank: (n-1) ASIO send + (n-1) ASIO recv + (n-1) RDMA recv
            self.ecnaive_ports_per_rank = 3 * (self.ecnaive_n - 1)

            logger.info(
                f"EC-NAIVE: RS scheme {self.ecnaive_k}+2 → {self.ecnaive_n} ranks/group, "
                f"{self.ecnaive_ports_per_rank} ports/rank"
            )

            # Check RDMA flag
            self.use_rdma = getattr(args, 'use_rdma', False)
            logger.info(f"EC-NAIVE: RDMA support {'enabled' if self.use_rdma else 'disabled'}")
                
            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning("EC-NAIVE: Distributed environment not initialized, skipping EC-NAIVE initialization")
                return
                
            # Initialize EC-NAIVE C++ module
            self._init_ecnaive_native()
            
            # Start persistent buffer poller thread
            # EC-NAIVE does not use layerwise mode
            self._start_buffer_poller_thread()
            
        except Exception as e:
            logger.warning(f"EC-NAIVE: Failed to initialize during manager initialization: {e}")
            self._ecnaive_native = None
    
    def _init_ecnaive_native(self):
        """Initialize EC-NAIVE C++ native module."""
        ecnaive_native = None
        try:
            # Direct import .so file without modifying sys.path or affecting other packages
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Find .so file
            import glob as _glob_module
            so_files = _glob_module.glob(os.path.join(current_dir, "ecnaive_native*.so"))
            
            if not so_files:
                raise ImportError(f"No ecnaive_native.so file found in {current_dir}")
            
            # Load .so file directly using importlib
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("ecnaive_native", so_path)
            ecnaive_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(ecnaive_native)
            logger.debug(f"EC-NAIVE: Loaded .so file from {so_path}")
            
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # EC-NAIVE only uses ASIO (no NCCL support)
            # Create instance with error handling
            try:
                # ===== ASIO Initialization Path =====
                logger.info(f"EC-NAIVE: [Rank {rank}] Using ASIO for communication")
                
                # Get network configuration
                net_config = self._get_ecnaive_network_config(rank, world_size)
                
                # Synchronize all ranks before creating C++ instances
                logger.info(f"EC-NAIVE: [Rank {rank}] Synchronizing all ranks before creating C++ native module (ASIO)...")
                torch.distributed.barrier()
                logger.info(f"EC-NAIVE: [Rank {rank}] All ranks synchronized, creating C++ native module with ASIO...")
                
                # Create C++ instance with ASIO parameters
                logger.info(f"EC-NAIVE: Creating C++ native module with ASIO (this will block until ASIO connections are established)...")
                print(f"EC-NAIVE: [Rank {rank}] Creating C++ native module with ASIO (blocking until ASIO initialization completes)...")
                
                # EC-NAIVE requires 12 parameters (6 pairs of ip:port):
                # send_data1_ip, send_data1_port, send_parity0_ip, send_parity0_port, 
                # send_parity1_ip, send_parity1_port, recv_parity1_ip, recv_parity1_port,
                # recv_parity0_ip, recv_parity0_port, recv_data1_ip, recv_data1_port
                
                # Calculate round-robin partner ranks
                partner_ranks = self._get_round_robin_ranks(rank, world_size)

                base_port = net_config['base_port']
                rank_ips = net_config['rank_ips']
                n = self.ecnaive_n
                ports_per_rank = self.ecnaive_ports_per_rank

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
                recv_partners = partner_ranks['recv_partners']
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
                recv_ports_local = net_config['ports']['recv_ports']  # list of local recv listen ports

                # Build RDMA exchange params: (partner_rdma_recv_port, local_rdma_recv_port) for each channel
                rdma_send_ports_to = []
                rdma_recv_ports_local = net_config['ports']['rdma_recv_ports']
                for j, target_r in enumerate(send_partners):
                    partner_port_base = _port_base_for_rank(target_r)
                    rdma_send_ports_to.append(partner_port_base + 2 * (n - 1) + j)  # partner's rdma_recv_ports[j]

                # Get rank_in_group for RDMA connection ordering
                rank_in_group = self._get_rank_in_group(rank, world_size)

                # Create C++ instance with generalized connection lists
                self._ecnaive_native = ecnaive_native.ECNaiveNative(
                    send_ips, send_ports_to,
                    recv_ips, recv_ports_local,
                    rdma_send_ports_to, rdma_recv_ports_local,
                    self.ecnaive_k,
                    self.use_rdma,
                    rank_in_group
                )
                
                # If we reach here, ASIO connections are ready and threads are running
                logger.info(f"EC-NAIVE: C++ native module initialized successfully with ASIO (rank={rank}, world_size={world_size})")
                print(f"EC-NAIVE: [Rank {rank}] C++ native module initialized - ASIO connections ready for data exchange")
                
                # Initialize EC-NAIVE buffers
                # EC-NAIVE does not use layerwise mode
                self._init_ecnaive_buffers()
                
                # Synchronize all ranks after RDMA/ASIO and buffers are ready
                logger.info(f"EC-NAIVE: [Rank {rank}] Synchronizing all ranks after native module init...")
                torch.distributed.barrier()
                logger.info(f"EC-NAIVE: [Rank {rank}] All ranks synchronized after EC-NAIVE init")
        
            except Exception as e:
                logger.warning(f"EC-NAIVE: Failed to create C++ native module instance: {e}")
                # Try to stop the pipeline if it was partially created
                try:
                    if hasattr(self, '_ecnaive_native') and self._ecnaive_native is not None:
                        self._ecnaive_native.stop()
                except:
                    pass
                self._ecnaive_native = None
                raise e
            
        except ImportError as e:
            logger.warning(f"EC-NAIVE: C++ native module not available: {e}, EC-NAIVE functionality will not work")
            self._ecnaive_native = None
        except Exception as e:
            logger.warning(f"EC-NAIVE: Failed to initialize C++ native module: {e}, EC-NAIVE functionality will not work")
            self._ecnaive_native = None
    
    def _init_ecnaive_buffers(self):
        """Initialize EC-NAIVE buffers during C++ module initialization.
        
        Note: Only allocates data and parity buffers (pooled) at initialization.
        The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) will be allocated
        in strategy after metadata exchange.
        """
        rank = torch.distributed.get_rank()
        logger.info("EC-NAIVE: Initializing buffers for EC-NAIVE (data and parity pools only)")
        print(f"EC-NAIVE: Initializing buffers for EC-NAIVE (rank={rank}, data and parity pools only)")
        
        # Allocate data buffers for storing original tensor data
        self.ecnaive_data_buffers = self._allocate_data_buffers()
        
        # Allocate parity buffers (pooled) for parity blocks
        self.ecnaive_parity_buffers = self._allocate_parity_buffers()
        
        # Register buffers for RDMA if enabled
        if self.use_rdma and self._ecnaive_native is not None:
            logger.info(f"EC-NAIVE: [Rank {rank}] Registering data and parity buffers for RDMA...")
            for buffer in self.ecnaive_data_buffers:
                self.register_buffer(buffer)
            for buffer in self.ecnaive_parity_buffers:
                self.register_buffer(buffer)
            logger.info(f"EC-NAIVE: [Rank {rank}] All pooled buffers registered for RDMA")
        
        # Initialize free buffer queues
        self._free_data_buffer_queue = queue.Queue()
        for buffer in self.ecnaive_data_buffers:
            self._free_data_buffer_queue.put(int(buffer.data_ptr()))
        
        self._free_parity_buffer_queue = queue.Queue()
        for buffer in self.ecnaive_parity_buffers:
            self._free_parity_buffer_queue.put(int(buffer.data_ptr()))

        logger.info(f"EC-NAIVE: Buffer initialization completed - "
                   f"Data buffers: {len(self.ecnaive_data_buffers)}, "
                   f"Parity buffers: {len(self.ecnaive_parity_buffers)}")
        print(f"EC-NAIVE: Buffer initialization completed (rank={rank}) - "
              f"Data buffers: {len(self.ecnaive_data_buffers)}, "
              f"Parity buffers: {len(self.ecnaive_parity_buffers)}")
    
    def _allocate_data_buffers(self):
        """Allocate data buffers for storing original tensor data."""
        logger.info(f"EC-NAIVE: Allocating data buffers ({self.ecnaive_data_buffers_count} buffers, {self.ecnaive_buffer_size // (1024*1024)}MB each)")
        
        data_buffers = []
        for i in range(self.ecnaive_data_buffers_count):
            buffer = allocate_hugepage_tensor(
                self.ecnaive_buffer_size,
                fallback_pin_memory=self.ecnaive_pin_memory,
                touch_pages=True,
            )
            data_buffers.append(buffer)
            logger.debug(f"EC-NAIVE: Allocated data buffer {i}: {self.ecnaive_buffer_size} bytes")
        
        logger.info(f"EC-NAIVE: Allocated {len(data_buffers)} data buffers")
        return data_buffers
    
    def _allocate_parity_buffers(self):
        """Allocate parity buffers (pooled) for parity blocks."""
        logger.info(f"EC-NAIVE: Allocating parity buffers ({self.ecnaive_parity_buffers_count} buffers, {self.ecnaive_buffer_size // (1024*1024)}MB each)")
        
        parity_buffers = []
        for i in range(self.ecnaive_parity_buffers_count):
            buffer = allocate_hugepage_tensor(
                self.ecnaive_buffer_size,
                fallback_pin_memory=self.ecnaive_pin_memory,
                touch_pages=True,
            )
            parity_buffers.append(buffer)
            logger.debug(f"EC-NAIVE: Allocated parity buffer {i}: {self.ecnaive_buffer_size} bytes")
        
        logger.info(f"EC-NAIVE: Allocated {len(parity_buffers)} parity buffers")
        return parity_buffers
    
    
    def _poll_and_release_buffers(self):
        """Poll C++ for buffers ready to be released and put them back to queues."""
        if self._ecnaive_native is None:
            return
        
        # Get data buffers ready for release
        data_buffers = self._ecnaive_native.get_data_buffers_to_release()
        for data_addr in data_buffers:
            try:
                self._free_data_buffer_queue.put_nowait(data_addr)
                # logger.debug(f"EC-NAIVE: Released data buffer at address {data_addr}")
            except Exception:
                logger.error(f"EC-NAIVE: Data buffer queue is full, cannot release buffer {data_addr}")
        
        # Get parity buffers ready for release
        parity_buffers = self._ecnaive_native.get_parity_buffers_to_release()
        for parity_addr in parity_buffers:
            try:
                self._free_parity_buffer_queue.put_nowait(parity_addr)
                # logger.debug(f"EC-NAIVE: Released parity buffer at address {parity_addr}")
            except Exception:
                logger.error(f"EC-NAIVE: Parity buffer queue is full, cannot release buffer {parity_addr}")
    
    def _start_buffer_poller_thread(self):
        """Start a persistent background thread to poll and release buffers."""
        if hasattr(self, '_buffer_poller_thread') and self._buffer_poller_thread is not None:
            logger.warning("EC-NAIVE: Buffer poller thread already started")
            return
        
        # Create control events
        self._buffer_poller_stop_event = threading.Event()
        self._buffer_poller_active_event = threading.Event()
        
        def buffer_poller_worker():
            """Persistent background thread that polls for buffer releases."""
            logger.info("EC-NAIVE: Buffer poller thread started")
            poll_count = 0
            
            while not self._buffer_poller_stop_event.is_set():
                # Only poll when active
                if self._buffer_poller_active_event.is_set():
                    self._poll_and_release_buffers()
                    poll_count += 1
                    if poll_count % 1000 == 0:
                        logger.debug(f"EC-NAIVE: Buffer poller running (polled {poll_count} times)")
                
                # Sleep briefly to avoid busy waiting
                from time import sleep
                sleep(0.001)  # 1ms
            
            logger.info("EC-NAIVE: Buffer poller thread stopping")
        
        # Start the daemon thread
        self._buffer_poller_thread = threading.Thread(target=buffer_poller_worker, daemon=True)
        self._buffer_poller_thread.start()
        logger.info("EC-NAIVE: Buffer poller thread created and started")
    
    def _stop_buffer_poller_thread(self):
        """Stop the persistent buffer poller thread."""
        if not hasattr(self, '_buffer_poller_thread') or self._buffer_poller_thread is None:
            return
        
        logger.info("EC-NAIVE: Stopping buffer poller thread...")
        
        # Signal the thread to stop
        if self._buffer_poller_stop_event:
            self._buffer_poller_stop_event.set()
        
        # Wait for thread to finish
        if self._buffer_poller_thread.is_alive():
            self._buffer_poller_thread.join(timeout=2.0)
            if self._buffer_poller_thread.is_alive():
                logger.warning("EC-NAIVE: Buffer poller thread did not stop in time")
            else:
                logger.info("EC-NAIVE: Buffer poller thread stopped successfully")
        
        self._buffer_poller_thread = None
        self._buffer_poller_stop_event = None
        self._buffer_poller_active_event = None
    
    def get_ecnaive_buffers(self):
        """Get EC-NAIVE buffers for FileSystemWriterAsync.
        
        Note: Returns data and parity buffers (pooled).
        The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) will be allocated
        in strategy after metadata exchange.
        
        Returns:
            Dict containing all buffer information, or None if not initialized
        """
        if self.ecnaive_data_buffers is None:
            return None
        
        return {
            'data_buffers': self.ecnaive_data_buffers,
            'parity_buffers': self.ecnaive_parity_buffers,
            'free_data_buffer_queue': self._free_data_buffer_queue,
            'free_parity_buffer_queue': self._free_parity_buffer_queue,
            # Pass buffer poller control objects
            'buffer_poller_active_event': self._buffer_poller_active_event,
            'poll_and_release_buffers': self._poll_and_release_buffers,
            # Note: The 4 persistent blocks (data0, recv_parity1, recv_parity0, recv_data1) 
            # will be allocated in strategy after metadata exchange
        }
    
    def register_buffer(self, buffer: torch.Tensor):
        """Register buffer for RDMA operations (similar to Gemini).
        
        Args:
            buffer: PyTorch tensor to register for RDMA
        """
        if not self.use_rdma or self._ecnaive_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        buffer_size = buffer.numel() * buffer.element_size()
        
        # Check if already registered
        if buffer_addr in self.registered_buffers:
            logger.debug(f"EC-NAIVE: [Rank {rank}] Buffer already registered at 0x{buffer_addr:x} (size: {buffer_size / (1024**2):.2f} MB)")
            return
        
        try:
            logger.info(f"EC-NAIVE: [Rank {rank}] Registering buffer at 0x{buffer_addr:x}, size: {buffer_size / (1024**3):.2f} GB, numel: {buffer.numel()}, dtype: {buffer.dtype} (iteration {self.current_iteration})")
            self._ecnaive_native.register_buffer(buffer_addr, buffer_size)
            self.registered_buffers[buffer_addr] = (buffer_size, self.current_iteration)
            logger.info(f"EC-NAIVE: [Rank {rank}] Buffer registered successfully (total registered: {len(self.registered_buffers)})")
            
            # Print all registered buffers
            logger.info(f"EC-NAIVE: [Rank {rank}] All registered buffers:")
            for addr, (size, iteration) in self.registered_buffers.items():
                logger.info(f"  - 0x{addr:x}: {size / (1024**2):.2f} MB (iteration {iteration})")
        except Exception as e:
            logger.error(f"EC-NAIVE: [Rank {rank}] Failed to register buffer: {e}")
            raise
    
    def unregister_buffer(self, buffer: torch.Tensor):
        """Unregister buffer from RDMA.
        
        Args:
            buffer: PyTorch tensor to unregister
        """
        if not self.use_rdma or self._ecnaive_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        
        if buffer_addr not in self.registered_buffers:
            return
        
        try:
            logger.info(f"EC-NAIVE: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x}")
            self._ecnaive_native.unregister_buffer(buffer_addr)
            del self.registered_buffers[buffer_addr]
            logger.info(f"EC-NAIVE: [Rank {rank}] Buffer unregistered successfully (remaining: {len(self.registered_buffers)})")
        except Exception as e:
            logger.error(f"EC-NAIVE: [Rank {rank}] Failed to unregister buffer: {e}")
            raise
    
    def cleanup(self):
        """Cleanup EC-NAIVE resources when manager is destroyed."""
        try:
            # Stop buffer poller thread
            self._stop_buffer_poller_thread()

            # Stop the C++ pipeline and null it out so re-init works
            if hasattr(self, '_ecnaive_native') and self._ecnaive_native is not None:
                self._ecnaive_native.stop()
                self._ecnaive_native = None
                logger.info("EC-NAIVE: C++ native module stopped in manager cleanup")

            # Clear registered buffers and cached allocations
            self.registered_buffers.clear()
            self.preallocated_cpu_buffer = None
            self._cached_blocks = None
            self._cached_block_count = 0
            self._cached_block_size = 0
            self.ecnaive_data_buffers = None
            self.ecnaive_parity_buffers = None
            self._free_data_buffer_queue = None
            self._free_parity_buffer_queue = None

        except Exception as e:
            logger.warning(f"EC-NAIVE: Error during manager cleanup: {e}")
    
    def __del__(self):
        """Cleanup EC-NAIVE resources when manager is destroyed."""
        self.cleanup()

