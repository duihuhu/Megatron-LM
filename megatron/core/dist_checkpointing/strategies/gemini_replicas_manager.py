# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Gemini Replicas manager for multi-replica data transfer with C++ ASIO or RDMA implementation."""

import os
import queue
import threading
from logging import getLogger
from typing import Dict, List, Optional, Set, Tuple

import torch

from .state_dict_decomposer import DecomposedStateDict, TensorInfo
from .hugepage_alloc import allocate_hugepage_tensor
from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip

logger = getLogger(__name__)


def _gemini_replicas_debug_enabled() -> bool:
    try:
        from megatron.training import get_args
        return bool(getattr(get_args(), "gemini_replicas_debug", False))
    except Exception:
        return False


class GeminiReplicasManager:
    """Shared manager for Gemini Replicas multi-replica data transfer.
    
    This class provides a singleton instance that manages:
    - Gemini Replicas C++ native module (_gemini_replicas_native) for ASIO or RDMA communication
    - Buffer allocation and management for multi-replica data exchange
    - Decomposed state dict for efficient GPU-to-CPU transfer
    - Buffer registration for RDMA (when use_rdma is enabled)
    
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
        self.use_rdma = False
        self.use_gdr = False
        
        # Replica configuration
        self.num_replicas = 3  # Default: 3 replicas (including local)
        self.group_size: Optional[int] = None  # None = global, int = independent groups of this size
        
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
        
        # Track registered buffers (for RDMA)
        self.registered_buffers: Dict[int, Tuple[int, int]] = {}  # {buffer_addr: (size, iteration)}
        self.current_iteration: int = 0
        
        self._initialized = True
    
    @staticmethod
    def _get_ranks_per_node() -> int:
        """Detect how many ranks run per node from environment variables.

        Priority: LOCAL_WORLD_SIZE > OMPI_COMM_WORLD_LOCAL_SIZE >
        MPI_LOCALNRANKS > MV2_COMM_WORLD_LOCAL_SIZE > SLURM_NTASKS_PER_NODE >
        torch.cuda.device_count().
        """
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

    def _calculate_target_ranks(self, my_rank: int, world_size: int) -> List[int]:
        """Calculate target ranks for replicas using round-robin strategy.

        When group_size is set, ranks are partitioned into node-interleaved groups
        (one rank per node per group, following FRCheck's layout), and round-robin
        placement is confined within each group.  This ensures replicas of the same
        data land on different physical nodes.

        Round-robin placement (global, no grouping):
        - rank0 (3 replicas): [0, 1, 2]
        - rank1 (3 replicas): [1, 2, 3]

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

        # --- global round-robin (no grouping) ---
        if self.group_size is None or self.group_size >= world_size:
            targets = []
            for i in range(self.num_replicas):
                targets.append((my_rank + i) % world_size)
            logger.info(
                f"Gemini Replicas: [Rank {my_rank}] Calculated target ranks: {targets} "
                f"({self.num_replicas} replicas, global)"
            )
            return targets

        # --- node-interleaved grouping (FRCheck layout) ---
        gs = self.group_size
        ranks_per_node = self._get_ranks_per_node()

        use_node_aware = (
            world_size >= gs
            and world_size % gs == 0
            and ranks_per_node > 0
            and world_size % ranks_per_node == 0
        )
        if use_node_aware:
            num_nodes = world_size // ranks_per_node
            if num_nodes >= gs and num_nodes % gs == 0:
                clusters = num_nodes // gs
                node_id = my_rank // ranks_per_node
                local_rank = my_rank % ranks_per_node
                cluster_id = node_id % clusters

                # group_id follows FRCheck: local_rank * clusters + cluster_id
                group_id = local_rank * clusters + cluster_id

                # Build ordered list of members in this group (rank_in_group 0..gs-1)
                group_members = []
                for pos in range(gs):
                    member_node = pos * clusters + cluster_id
                    group_members.append(member_node * ranks_per_node + local_rank)

                if self.num_replicas > len(group_members):
                    raise ValueError(
                        f"Gemini Replicas: num_replicas ({self.num_replicas}) exceeds "
                        f"group member count ({len(group_members)}) for rank {my_rank}"
                    )

                my_pos = group_members.index(my_rank)
                targets = []
                for i in range(self.num_replicas):
                    targets.append(group_members[(my_pos + i) % len(group_members)])

                # logger.info(
                #     f"Gemini Replicas: [Rank {my_rank}] node-aware targets: {targets} "
                #     f"({self.num_replicas} replicas, group_id={group_id}, "
                #     f"members={group_members}, nodes/group={gs})"
                # )
                return targets
            else:
                logger.warning(
                    f"Gemini Replicas: num_nodes ({num_nodes}) not divisible by "
                    f"group_size ({gs}), falling back to consecutive grouping for rank {my_rank}"
                )

        # --- fallback: simple consecutive grouping ---
        group_id = my_rank // gs
        group_start = group_id * gs
        group_end = min(group_start + gs, world_size)
        group_world = group_end - group_start
        local_rank = my_rank - group_start

        if self.num_replicas > group_world:
            raise ValueError(
                f"Gemini Replicas: num_replicas ({self.num_replicas}) cannot exceed "
                f"group size ({group_world}) for rank {my_rank}"
            )

        targets = []
        for i in range(self.num_replicas):
            target_local = (local_rank + i) % group_world
            targets.append(group_start + target_local)

        logger.info(
            f"Gemini Replicas: [Rank {my_rank}] consecutive-group targets: {targets} "
            f"({self.num_replicas} replicas, group=[{group_start}, {group_end}))"
        )

        return targets

    def get_group_members(self, my_rank: int, world_size: int) -> List[int]:
        """Return all ranks that belong to the same group as my_rank.

        Uses the same node-interleaved (or consecutive) grouping as
        _calculate_target_ranks so recovery can determine group boundaries.
        """
        gs = self.group_size
        if gs is None or gs >= world_size:
            return list(range(world_size))

        ranks_per_node = self._get_ranks_per_node()
        use_node_aware = (
            world_size >= gs
            and world_size % gs == 0
            and ranks_per_node > 0
            and world_size % ranks_per_node == 0
        )
        if use_node_aware:
            num_nodes = world_size // ranks_per_node
            if num_nodes >= gs and num_nodes % gs == 0:
                clusters = num_nodes // gs
                node_id = my_rank // ranks_per_node
                local_rank = my_rank % ranks_per_node
                cluster_id = node_id % clusters
                members = []
                for pos in range(gs):
                    member_node = pos * clusters + cluster_id
                    members.append(member_node * ranks_per_node + local_rank)
                return members

        # Fallback: consecutive grouping
        group_id = my_rank // gs
        group_start = group_id * gs
        group_end = min(group_start + gs, world_size)
        return list(range(group_start, group_end))

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
        # Step 1: Get base IP address (with multi-NIC per-rank support)
        base_ip = resolve_ip("GEMINI_REPLICAS", rank=rank)

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
            self.group_size = getattr(args, 'gemini_replicas_group_size', None)
            self.use_rdma = getattr(args, 'use_rdma', False)
            
            if not self.use_gemini_replicas or not self.use_gemini_replicas_optimized:
                return
            
            # Check if distributed environment is initialized
            if not torch.distributed.is_initialized():
                logger.warning("Gemini Replicas: Distributed environment not initialized, skipping initialization")
                return
            
            self._prepare_native_reinit()

            # Initialize Gemini Replicas C++ module
            self._init_gemini_replicas_native()
            
        except Exception as e:
            logger.error(f"Gemini Replicas: Failed to initialize: {e}")
            self._gemini_replicas_native = None
            raise

    def _prepare_native_reinit(self):
        """Clear stale native state before binding a fresh listener."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.debug(f"Gemini Replicas: [Rank {rank}] Preparing native reinit")
        self._stop_native_gracefully()
        self.registered_buffers.clear()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    def _init_gemini_replicas_native(self):
        """Initialize Gemini Replicas C++ native module with ASIO or RDMA."""
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
            mode_str = "RDMA" if self.use_rdma else "ASIO"
            logger.info(f"Gemini Replicas: [Rank {rank}] Synchronizing all ranks before creating C++ native module...")
            torch.distributed.barrier()
            logger.info(f"Gemini Replicas: [Rank {rank}] All ranks synchronized, creating C++ native module with {mode_str}...")
            
            # Use the prepared target_ips and target_ports from config
            target_ips = net_config['target_ips']
            target_ports = net_config['target_ports']
            
            # Get my recv port (all sources will connect to this port)
            recv_ports_list = list(net_config['recv_ports'].values())
            my_recv_port = recv_ports_list[0] if recv_ports_list else net_config['base_port'] + rank * 100
            
            # Calculate number of source ranks
            num_source_ranks = len(net_config['source_ranks'])
            
            # Create C++ instance (Phase 1: start acceptor only)
            logger.info(f"Gemini Replicas: Creating C++ native module with {mode_str} (Phase 1: acceptor)...")
            print(f"Gemini Replicas: [Rank {rank}] Creating C++ native module (Phase 1: starting acceptor, mode: {mode_str})...")
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
                num_source_ranks,  # Number of expected incoming connections
                self.use_rdma  # Use RDMA or ASIO
            )
            if hasattr(self._gemini_replicas_native, "set_debug"):
                self._gemini_replicas_native.set_debug(_gemini_replicas_debug_enabled())
            
            logger.info(f"Gemini Replicas: C++ native module created (acceptor ready) for rank {rank}")
            print(f"Gemini Replicas: [Rank {rank}] Acceptor ready, waiting for all ranks...")
            
            # Synchronize all ranks before connecting (Phase 2)
            torch.distributed.barrier()
            logger.info(f"Gemini Replicas: [Rank {rank}] All ranks ready, starting Phase 2 (connecting)...")
            print(f"Gemini Replicas: [Rank {rank}] Phase 2: Connecting to target ranks {net_config['target_ranks']}...")
            
            # Phase 2: Connect to all targets
            self._gemini_replicas_native.finalize_connections()

            # Post-finalize barrier (aligned with EC-NAIVE load: all TCP+RDMA ready)
            torch.distributed.barrier()
            logger.info(
                f"Gemini Replicas: [Rank {rank}] All ranks finished Phase 2 connections"
            )

            # Start persistent send/recv worker threads (like ecnaive)
            self._gemini_replicas_native.start_workers(net_config['source_ranks'])

            logger.info(f"Gemini Replicas: C++ native module fully initialized (rank={rank}, targets={net_config['target_ranks']}, mode={mode_str})")
            print(f"Gemini Replicas: [Rank {rank}] C++ native module fully initialized - {mode_str} connections ready")

            # GDR setup: check availability and start mirror worker for async D2H
            if self.use_rdma:
                try:
                    self.use_gdr = gemini_replicas_native.GeminiReplicasNative.gdr_available()
                    if not self.use_gdr:
                        # Module not found in /proc/modules — try actual GPU MR registration
                        self.use_gdr = self._gemini_replicas_native.probe_gdr()
                except AttributeError:
                    self.use_gdr = False  # old .so without GDR support
                if self.use_gdr:
                    logger.info(f"Gemini Replicas: [Rank {rank}] GDR (GPU Direct RDMA) available, starting mirror worker")
                    self._gemini_replicas_native.set_require_registered_mr(True)
                    self._gemini_replicas_native.start_mirror_worker()
                    print(f"Gemini Replicas: [Rank {rank}] GDR mirror worker started")
                else:
                    logger.warning(f"Gemini Replicas: [Rank {rank}] GDR not available (nvidia-peermem missing)")
            
        except Exception as e:
            logger.error(f"Gemini Replicas: Failed to initialize C++ native module: {e}")
            import traceback
            traceback.print_exc()
            self._stop_native_gracefully()
            raise

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
                if _gemini_replicas_debug_enabled():
                    logger.info(f"Gemini Replicas: [Rank {rank}] Reusing existing preallocated buffer")
                return
        
        if _gemini_replicas_debug_enabled():
            logger.info(f"Gemini Replicas: [Rank {rank}] Allocating preallocated buffer: {size_bytes / (1024**3):.2f} GB")
        
        pin = self.gemini_replicas_pin_memory and torch.cuda.is_available()
        self.preallocated_cpu_buffer = allocate_hugepage_tensor(
            size_bytes, fallback_pin_memory=pin, touch_pages=False,
        )
        if _gemini_replicas_debug_enabled():
            logger.info(f"Gemini Replicas: [Rank {rank}] Allocated preallocated buffer: "
                        f"{size_bytes / (1024**3):.2f} GB (hugepage, pin={pin})")

    _cached_recv_buffers: Dict[int, torch.Tensor] = {}

    def allocate_recv_buffer(self, src_rank: int, size_bytes: int):
        """Allocate or reuse a cached receive buffer for *src_rank*.

        Uses hugepage memory so that ``ibv_reg_mr`` (RDMA registration) can
        pin the buffer — plain ``torch.empty`` pages routinely fail pinning
        at the multi-GB scale required for large models.
        """
        cached = self._cached_recv_buffers.get(src_rank)
        if cached is not None and cached.numel() >= size_bytes:
            return cached
        pin = self.gemini_replicas_pin_memory and torch.cuda.is_available()
        buf = allocate_hugepage_tensor(
            size_bytes, fallback_pin_memory=pin, touch_pages=False,
        )
        self._cached_recv_buffers[src_rank] = buf
        return buf

    def register_buffer(self, buffer: torch.Tensor):
        """Register buffer for RDMA operations (called on first allocation in save phase).
        
        Args:
            buffer: PyTorch tensor to register
        """
        if not self.use_rdma or self._gemini_replicas_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        buffer_size = buffer.numel() * buffer.element_size()
        
        # Check if already registered
        if buffer_addr in self.registered_buffers:
            if _gemini_replicas_debug_enabled():
                logger.info(f"Gemini Replicas: [Rank {rank}] Buffer already registered at 0x{buffer_addr:x} (size: {buffer_size / (1024**2):.2f} MB)")
            return
        
        try:
            if _gemini_replicas_debug_enabled():
                logger.info(f"Gemini Replicas: [Rank {rank}] Registering buffer at 0x{buffer_addr:x}, size: {buffer_size / (1024**3):.2f} GB, numel: {buffer.numel()}, dtype: {buffer.dtype} (iteration {self.current_iteration})")
            self._gemini_replicas_native.register_buffer(buffer_addr, buffer_size)
            self.registered_buffers[buffer_addr] = (buffer_size, self.current_iteration)
            if _gemini_replicas_debug_enabled():
                logger.info(f"Gemini Replicas: [Rank {rank}] Buffer registered successfully (total registered: {len(self.registered_buffers)})")
            
            # Print all registered buffers
            if _gemini_replicas_debug_enabled():
                logger.info(f"Gemini Replicas: [Rank {rank}] All registered buffers:")
            # for addr, (size, iteration) in self.registered_buffers.items():
            #     logger.info(f"  - 0x{addr:x}: {size / (1024**2):.2f} MB (iteration {iteration})")
        except Exception as e:
            logger.error(f"Gemini Replicas: [Rank {rank}] Failed to register buffer: {e}")
            raise
    
    def unregister_buffer(self, buffer: torch.Tensor):
        """Unregister buffer for RDMA operations.
        
        Args:
            buffer: PyTorch tensor to unregister
        """
        if not self.use_rdma or self._gemini_replicas_native is None:
            return
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        buffer_addr = buffer.data_ptr()
        
        if buffer_addr not in self.registered_buffers:
            logger.debug(f"Gemini Replicas: [Rank {rank}] Buffer not registered at 0x{buffer_addr:x}")
            return
        
        try:
            logger.info(f"Gemini Replicas: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x}")
            self._gemini_replicas_native.unregister_buffer(buffer_addr)
            del self.registered_buffers[buffer_addr]
            logger.info(f"Gemini Replicas: [Rank {rank}] Buffer unregistered successfully")
        except Exception as e:
            logger.error(f"Gemini Replicas: [Rank {rank}] Failed to unregister buffer: {e}")
    
    def get_native_module(self):
        """Get the C++ native module instance."""
        return self._gemini_replicas_native

    def is_initialized(self) -> bool:
        """Check if Gemini Replicas native module is initialized."""
        return self._gemini_replicas_native is not None

    def send_to_rank(
        self,
        target_rank: int,
        buffer: torch.Tensor,
        *,
        register: bool = True,
    ) -> None:
        """Directed P2P send to a single target rank for hardware recovery.

        Uses the existing ASIO/RDMA connection (established during init), not
        the broadcast worker threads.  Synchronous and blocking.

        Args:
            target_rank: Destination rank.
            buffer: uint8 contiguous CPU tensor to send.
            register: When False, skip RDMA registration (buffer pre-registered).
        """
        native = self._gemini_replicas_native
        if native is None:
            raise RuntimeError(
                "Gemini Replicas native module not initialized "
                "— call init_gemini_replicas_if_enabled() first"
            )
        if (
            buffer.dtype == torch.uint8
            and buffer.is_contiguous()
            and buffer.dim() == 1
        ):
            buf = buffer
        else:
            buf = buffer.detach().contiguous().view(torch.uint8).reshape(-1)
        if self.use_rdma and register:
            self.register_buffer(buf)
        native.send_to_rank(target_rank, int(buf.data_ptr()), buf.numel())

    def recv_from_rank(
        self,
        source_rank: int,
        expected_size: int,
        buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Directed P2P recv from a specific source rank for hardware recovery.

        Uses the existing ASIO/RDMA connection.  Synchronous and blocking.

        Args:
            source_rank: Rank to receive from.
            expected_size: Exact number of bytes expected.
            buffer: Optional pre-allocated receive buffer (already registered).

        Returns:
            uint8 contiguous CPU tensor with the received data.
        """
        native = self._gemini_replicas_native
        if native is None:
            raise RuntimeError(
                "Gemini Replicas native module not initialized "
                "— call init_gemini_replicas_if_enabled() first"
            )
        if buffer is None:
            pin = self.gemini_replicas_pin_memory and torch.cuda.is_available()
            buf = allocate_hugepage_tensor(
                expected_size, fallback_pin_memory=pin, touch_pages=False,
            )
            if self.use_rdma:
                self.register_buffer(buf)
        else:
            if buffer.numel() < expected_size:
                raise RuntimeError(
                    f"Pre-allocated recv buffer too small: "
                    f"need {expected_size}, got {buffer.numel()}"
                )
            buf = buffer[:expected_size]
        native.recv_from_rank(source_rank, int(buf.data_ptr()), expected_size)
        return buf

    def _stop_native_gracefully(self):
        """Stop native workers and drop the C++ instance without crashing.

        Workers are torn down via the destructor to avoid double-free of RDMA
        resources that the C++ destructor also cleans up.
        """
        native = self._gemini_replicas_native
        if native is not None:
            try:
                native.stop_workers()
            except Exception:
                pass
            try:
                native.stop()
            except Exception:
                pass
            self._gemini_replicas_native = None
            # Native teardown invalidates prior RDMA registrations.
            self.registered_buffers.clear()

    def _compute_recovery_connection_ranks(
        self,
        rank: int,
        recovery_ranks: Set[int],
        main_assignments: Dict[int, int],
        replica_needed: Dict[int, int],
        replica_failed_sources: Dict[int, List[int]],
        world_size: int,
    ) -> Tuple[List[int], List[int]]:
        """Build sparse sender↔receiver connection lists for recovery.

        Main recovery uses (failed → sender) from *main_assignments*.
        Replica recovery adds (failed → sender) edges for each source replica
        that a failed rank still needs.
        """
        target_set: Set[int] = set()
        source_set: Set[int] = set()

        for failed_rank, sender in main_assignments.items():
            if failed_rank not in recovery_ranks:
                continue
            if sender == rank:
                target_set.add(failed_rank)
            if failed_rank == rank:
                source_set.add(sender)

        for failed_rank in recovery_ranks:
            for src in replica_failed_sources.get(failed_rank, []):
                sender = replica_needed.get(src)
                if sender is None:
                    continue
                targets = self._calculate_target_ranks(src, world_size)
                if failed_rank not in targets:
                    continue
                if sender == rank:
                    target_set.add(failed_rank)
                if failed_rank == rank:
                    source_set.add(sender)

        return sorted(target_set), sorted(source_set)

    def reinit_for_recovery(
        self,
        recovery_ranks: set,
        main_assignments: Optional[Dict[int, int]] = None,
        replica_needed: Optional[Dict[int, int]] = None,
        replica_failed_sources: Optional[Dict[int, List[int]]] = None,
    ) -> None:
        """Rebuild sparse P2P connections for HW recovery.

        Only ranks that actually send or receive during recovery get new
        connections.  Senders connect to their assigned failed ranks; failed
        ranks accept from their assigned senders (main + replica recovery).

        Args:
            recovery_ranks: Global ranks that need recovery (disk data lost).
            main_assignments: failed_rank → sender for main data recovery.
            replica_needed: source_rank → sender for replica recovery.
            replica_failed_sources: failed_rank → source ranks still needed.
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        if not recovery_ranks:
            return

        main_assignments = main_assignments or {}
        replica_needed = replica_needed or {}
        replica_failed_sources = replica_failed_sources or {}

        # Compute group memberships so we only connect within the same group
        all_members = self.get_group_members(rank, world_size)
        failed_in_group = recovery_ranks & set(all_members)
        healthy_in_group = set(all_members) - failed_in_group

        target_ranks, source_ranks = self._compute_recovery_connection_ranks(
            rank,
            recovery_ranks,
            main_assignments,
            replica_needed,
            replica_failed_sources,
            world_size,
        )

        # Ranks in a group with both failed and healthy members rebuild sparse P2P.
        # Ranks in unaffected groups keep their save-time mesh but must still join
        # the global sync below so all_gather/barriers do not deadlock.
        participates = bool(failed_in_group and healthy_in_group)
        has_p2p_role = participates and bool(target_ranks or source_ranks)

        if participates:
            logger.info(
                f"Gemini Replicas: [Rank {rank}] Reinitializing for recovery: "
                f"failed_in_group={sorted(failed_in_group)}, "
                f"healthy_in_group={sorted(healthy_in_group)}, "
                f"targets={target_ranks}, sources={source_ranks}"
            )
            self._stop_native_gracefully()
        else:
            logger.info(
                f"Gemini Replicas: [Rank {rank}] No recovery in local group "
                f"(failed_in_group={sorted(failed_in_group)}), "
                f"keeping save-time connections"
            )

        # Sync teardown across all ranks before any reconnect/IP exchange.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # ---- build recovery topology ----
        base_ip = resolve_ip("GEMINI_REPLICAS", rank=rank)
        master_port = int(os.environ.get('MASTER_PORT', '6000'))
        # Use a different base port to avoid conflicts with the save connections
        reco_base_port = int(os.environ.get(
            'GEMINI_REPLICAS_RECOVERY_BASE_PORT',
            master_port + 40000
        ))

        # Exchange IPs — all ranks in this reinit wave must participate.
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

        # Each rank listens on its own port: reco_base_port + rank * 100
        # Senders connect to the TARGET's listen port
        PORT_STEP = 100
        recv_port = reco_base_port + rank * PORT_STEP
        target_ips = [rank_ips.get(t, base_ip) for t in target_ranks]
        target_ports = [reco_base_port + t * PORT_STEP for t in target_ranks]
        num_sources = len(source_ranks)

        # ---- create new native module (or skip if this rank has no P2P role) ----
        try:
            if not has_p2p_role:
                if participates:
                    logger.info(
                        f"Gemini Replicas recovery: [Rank {rank}] no P2P role, "
                        f"skipping native module creation"
                    )
                # Match acceptor-ready and post-finalize barriers on P2P ranks.
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    torch.distributed.barrier()
                return

            current_dir = os.path.dirname(os.path.abspath(__file__))
            import glob as _glob_module
            so_files = _glob_module.glob(
                os.path.join(current_dir, "gemini_replicas_native*.so")
            )
            if not so_files:
                logger.error("Gemini Replicas: No .so found for recovery reinit")
                return
            import importlib.util as _importlib_util
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location(
                "gemini_replicas_native", so_path
            )
            gemini_replicas_native = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(gemini_replicas_native)

            mode_str = "RDMA" if self.use_rdma else "ASIO"
            logger.info(
                f"Gemini Replicas recovery: [Rank {rank}] creating native module: "
                f"targets={target_ranks}, sources={source_ranks} ({mode_str})"
            )

            self._gemini_replicas_native = gemini_replicas_native.GeminiReplicasNative(
                rank, world_size,
                target_ranks, target_ips, target_ports,
                base_ip, recv_port, num_sources,
                self.use_rdma,
            )

            # All acceptors must be listening before any rank connects.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            self._gemini_replicas_native.finalize_connections()

            # Post-finalize barrier (aligned with save-time init)
            torch.distributed.barrier()
            logger.info(
                f"Gemini Replicas recovery: [Rank {rank}] "
                f"All ranks finished recovery Phase 2 connections"
            )

            self._gemini_replicas_native.start_workers(source_ranks)

            logger.info(
                f"Gemini Replicas recovery: [Rank {rank}] reinit complete, "
                f"targets={target_ranks}, sources={source_ranks}"
            )
        except Exception as e:
            logger.error(
                f"Gemini Replicas: [Rank {rank}] recovery reinit failed: {e}"
            )
            import traceback
            traceback.print_exc()
            self._gemini_replicas_native = None
            raise

    def cleanup(self):
        """Cleanup resources."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Unregister all buffers for RDMA
        if self.use_rdma and self._gemini_replicas_native is not None:
            for buffer_addr in list(self.registered_buffers.keys()):
                try:
                    logger.info(f"Gemini Replicas: [Rank {rank}] Unregistering buffer at 0x{buffer_addr:x} during cleanup")
                    self._gemini_replicas_native.unregister_buffer(buffer_addr)
                except Exception as e:
                    logger.warning(f"Gemini Replicas: [Rank {rank}] Failed to unregister buffer during cleanup: {e}")
            self.registered_buffers.clear()

        # Delegate to the thorough shutdown path that stops workers before
        # dropping the C++ instance — ordering matters for port release.
        self._stop_native_gracefully()

        self.preallocated_cpu_buffer = None
        self.decomposed_state_dict = None
        self.replica_buffers = []
        self.replica_metadata = []
        self._cached_recv_buffers = {}

