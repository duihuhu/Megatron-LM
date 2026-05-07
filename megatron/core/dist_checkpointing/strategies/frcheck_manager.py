# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""FRCheck manager: POA table + RDMA topology + stripe encode pipeline."""

import glob as _glob_module
import importlib.util as _importlib_util
import os
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
    allocate_hugepage_tensor,
)

logger = getLogger(__name__)


class StripeRole(IntEnum):
    """Role of this rank within a single POA stripe."""
    SOURCE = 0
    ENCODER = 1
    PARITY_TARGET = 2


@dataclass
class StripePlan:
    """Pre-compiled action plan for a single stripe."""
    stripe_id: int
    row: List[int]
    role: StripeRole
    source_node_ids: List[int] = field(default_factory=list)
    encoder_node_id: int = -1
    parity_target_node_id: int = -1


class FRCheckManager:
    """Singleton for FRCheck C++ module, RDMA topology, and stripe plan compilation."""

    _instance: Optional["FRCheckManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._frcheck_native: Any = None
        self.use_frcheck = False
        self.frcheck_n: Optional[int] = None
        self.frcheck_table_path: Optional[str] = None
        self.group_layout: Optional[Dict[str, int]] = None
        self.group_id: Optional[int] = None
        self.rank_in_group: Optional[int] = None
        self.group_member_ranks: Optional[List[int]] = None
        self.node_slot_to_global_rank: Optional[Dict[int, int]] = None

        # RDMA / stripe encode
        self.stripe_plans: List[StripePlan] = []
        self.block_size: int = 0
        self.gdr_available: bool = False
        self.num_stripes: int = 0
        self.data_buffer: Optional[torch.Tensor] = None
        self.recv_buffer: Optional[torch.Tensor] = None
        self.parity1_buffer: Optional[torch.Tensor] = None
        self.parity2_buffer: Optional[torch.Tensor] = None
        self.parity1_accum: Optional[torch.Tensor] = None
        self.parity2_accum: Optional[torch.Tensor] = None
        self._initialized = True

    def init_frcheck_if_enabled(self) -> None:
        """Load native .so, init RDMA, compile stripe plans."""
        from megatron.training import get_args

        args = get_args()
        self.use_frcheck = bool(getattr(args, "use_frcheck", False))
        if not self.use_frcheck:
            return
        if self._frcheck_native is not None and self.stripe_plans:
            return

        n = self._resolve_frcheck_n(args)
        path = self._resolve_frcheck_table_path(args, n)
        self._validate_and_build_grouping(n)
        self._init_frcheck_native(path)

        # Detect GDR capability and num_stripes (available from POA, no RDMA needed)
        native = self._frcheck_native
        if native is not None:
            self.gdr_available = native.gdr_available()
            self.num_stripes = native.num_stripes()
        logger.info(
            "FRCheck: GDR %s, n=%d num_stripes=%d",
            "available" if self.gdr_available else "not available",
            self.frcheck_n, self.num_stripes,
        )

        # Init RDMA connections within group (allocates buffers using num_stripes)
        self._init_rdma(args)

        # Pre-compile stripe plans (reads from native after init_rdma)
        self._compile_stripe_plans()

    @staticmethod
    def _get_ranks_per_node() -> int:
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

    @classmethod
    def _get_group_layout(cls, world_size: int, n: int) -> Dict[str, int]:
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
    def _get_group_id(cls, rank: int, world_size: int, n: int) -> int:
        layout = cls._get_group_layout(world_size, n)
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
    def _get_rank_in_group(cls, rank: int, world_size: int, n: int) -> int:
        layout = cls._get_group_layout(world_size, n)
        num_groups = layout["num_groups"]
        if layout["mode"] == 1:
            ranks_per_node = layout["ranks_per_node"]
            clusters = layout["clusters"]
            node_id = rank // ranks_per_node
            return node_id // clusters
        return rank // num_groups

    @classmethod
    def _get_rank_by_group_position(
        cls, group_id: int, rank_in_group: int, world_size: int, n: int
    ) -> int:
        layout = cls._get_group_layout(world_size, n)
        num_groups = layout["num_groups"]
        if layout["mode"] == 1:
            ranks_per_node = layout["ranks_per_node"]
            clusters = layout["clusters"]
            local_rank = group_id // clusters
            cluster_id = group_id % clusters
            node_id = rank_in_group * clusters + cluster_id
            return node_id * ranks_per_node + local_rank
        return group_id + num_groups * rank_in_group

    def _resolve_frcheck_n(self, args) -> int:
        n = getattr(args, "frcheck_n", None)
        if n is None:
            raise RuntimeError("FRCheck: --frcheck-n is required when --use-frcheck is set.")
        if n < 3:
            raise RuntimeError("FRCheck: --frcheck-n must be >= 3.")
        self.frcheck_n = int(n)
        return self.frcheck_n

    def _resolve_frcheck_table_path(self, args, n: int) -> str:
        explicit_path = getattr(args, "frcheck_table_path", None)
        if explicit_path:
            if not os.path.isfile(explicit_path):
                raise RuntimeError(f"FRCheck: --frcheck-table-path not found: {explicit_path}")
            self.frcheck_table_path = str(Path(explicit_path).resolve())
            return self.frcheck_table_path

        table_dir = getattr(args, "frcheck_table_dir", None)
        if not table_dir:
            raise RuntimeError(
                "FRCheck: --frcheck-table-dir is required when --frcheck-table-path is not set."
            )
        d = Path(table_dir).resolve()
        candidates = [
            d / f"frcheck_poa_n{n}.txt",
            d / f"poa_n{n}.txt",
            d / f"n{n}.poa",
        ]
        for c in candidates:
            if c.is_file():
                self.frcheck_table_path = str(c)
                return self.frcheck_table_path
        pretty = ", ".join(str(c) for c in candidates)
        raise RuntimeError(f"FRCheck: no POA table found for n={n}. Tried: {pretty}")

    def _validate_and_build_grouping(self, n: int) -> None:
        if not torch.distributed.is_initialized():
            self.group_layout = {
                "mode": 0, "num_groups": 1, "ranks_per_node": 1,
                "num_nodes": 1, "clusters": 1,
            }
            self.group_id = 0
            self.rank_in_group = 0
            self.group_member_ranks = [0]
            self.node_slot_to_global_rank = {1: 0}
            return

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        ranks_per_node = self._get_ranks_per_node()
        if world_size % ranks_per_node != 0:
            raise RuntimeError(
                "FRCheck: invalid topology: world_size must be divisible by ranks_per_node "
                f"(world_size={world_size}, ranks_per_node={ranks_per_node})"
            )
        num_nodes = world_size // ranks_per_node
        if num_nodes % n != 0:
            raise RuntimeError(
                "FRCheck: invalid topology: num_nodes must be divisible by n "
                f"(world_size={world_size}, ranks_per_node={ranks_per_node}, "
                f"num_nodes={num_nodes}, n={n})"
            )
        self.group_layout = self._get_group_layout(world_size, n)
        self.group_id = self._get_group_id(rank, world_size, n)
        self.rank_in_group = self._get_rank_in_group(rank, world_size, n)
        self.group_member_ranks = [
            self._get_rank_by_group_position(self.group_id, i, world_size, n)
            for i in range(n)
        ]
        self.node_slot_to_global_rank = {
            i + 1: self.group_member_ranks[i] for i in range(len(self.group_member_ranks))
        }

    def _init_frcheck_native(self, poa_path: str) -> None:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            so_files = _glob_module.glob(os.path.join(current_dir, "frcheck_native*.so"))
            if not so_files:
                raise ImportError(f"No frcheck_native*.so found in {current_dir}")
            so_path = so_files[0]
            spec = _importlib_util.spec_from_file_location("frcheck_native", so_path)
            mod = _importlib_util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            logger.info("FRCheck: loaded native module from %s", so_path)
            self._frcheck_native = mod.FRCheckNative(poa_path)
            self.frcheck_table_path = poa_path
            logger.info(
                "FRCheck: native initialized n=%s num_stripes=%s",
                self._frcheck_native.n(),
                self._frcheck_native.num_stripes(),
            )
        except Exception as e:
            logger.warning("FRCheck: failed to initialize native module: %s", e)
            self._frcheck_native = None
            raise

    def _init_rdma(self, args) -> None:
        """Initialize RDMA full-mesh connections within the group."""
        native = self._frcheck_native
        if native is None:
            raise RuntimeError("FRCheck: native module not available for RDMA init")

        if not torch.distributed.is_initialized():
            logger.warning("FRCheck: distributed not initialized, skipping RDMA init")
            return

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        n = self.frcheck_n

        # Determine my IP
        my_ip = self._resolve_my_ip()

        # Exchange IPs across all ranks via all_gather
        rank_ip_list = [""] * world_size
        my_ip_tensor = torch.tensor(
            [int(b) for b in my_ip.encode("utf-8").ljust(64, b"\x00")[:64]],
            dtype=torch.uint8,
        )
        all_ips_tensor = [torch.zeros(64, dtype=torch.uint8) for _ in range(world_size)]
        torch.distributed.all_gather(all_ips_tensor, my_ip_tensor)
        for i in range(world_size):
            raw = bytes(all_ips_tensor[i].tolist()).rstrip(b"\x00")
            rank_ip_list[i] = raw.decode("utf-8") if raw else "127.0.0.1"

        # Build peer IPs for my group
        peer_ips = []
        for i in range(n):
            peer_global_rank = self._get_rank_by_group_position(self.group_id, i, world_size, n)
            peer_ips.append(rank_ip_list[peer_global_rank])

        # Compute base port: same scheme as ecnaive
        # Use MASTER_PORT + 20000 as base, offset by group_id * (n * 100)
        master_port = int(os.environ.get("MASTER_PORT", "6000"))
        base_port = int(os.environ.get("FRCHECK_BASE_PORT", str(master_port + 20000)))
        group_offset = self.group_id * (n * 100)
        base_port += group_offset

        rg = self.rank_in_group

        logger.info(
            "FRCheck RDMA: rank_in_group=%d/%d base_port=%d my_ip=%s peers=%s",
            rg, n, base_port, my_ip, peer_ips,
        )

        native.init_rdma(
            group_size=n,
            rank_in_group=rg,
            base_port=base_port,
            my_ip=my_ip,
            peer_ips=peer_ips,
            use_rdma=True,
        )

        # Register default buffers
        self._allocate_default_buffers(native, n)

        # Barrier after RDMA init
        torch.distributed.barrier()
        logger.info("FRCheck RDMA: group initialized (rank_in_group=%d/%d)", rg, n)

    def _resolve_my_ip(self) -> str:
        """Determine my IP for listen socket."""
        # Priority: env var, MASTER_ADDR, or auto-detect
        import socket

        base_ip = os.environ.get("FRCHECK_BASE_IP") or os.environ.get("ECLATIN_BASE_IP")
        if base_ip:
            return base_ip

        interface_name = os.environ.get("FRCHECK_INTERFACE") or os.environ.get("ECLATIN_INTERFACE")
        if interface_name:
            try:
                import netifaces
                addrs = netifaces.ifaddresses(interface_name)
                if netifaces.AF_INET in addrs:
                    return addrs[netifaces.AF_INET][0]["addr"]
            except ImportError:
                pass

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return os.environ.get("MASTER_ADDR", "127.0.0.1")

    def _allocate_default_buffers(self, native, n: int) -> None:
        """Allocate data/recv/parity buffers and register for RDMA."""
        from megatron.training import get_args
        args = get_args()
        ecnaive_buf_size = getattr(args, "ecnaive_buffer_size", 65536)
        default_block_size = ecnaive_buf_size * 1024  # 64 MB default

        self.block_size = default_block_size

        # Data buffer: holds this rank's data for each stripe
        #   GDR mode → GPU to avoid GPU→CPU A1 copy
        #   CPU mode → hugepage
        # Recv buffer: encoder receives n-2 blocks (CPU — ISAL encode in CPU)
        # Parity buffers: encoder outputs parity1 (CPU) + parity2 (CPU → sent to target)
        recv_total = (n - 2) * default_block_size

        if self.gdr_available:
            self.data_buffer = torch.cuda.ByteTensor(default_block_size)
            self.data_buffer.zero_()
        else:
            self.data_buffer = allocate_hugepage_tensor(
                default_block_size, fallback_pin_memory=True
            )
        self.recv_buffer = allocate_hugepage_tensor(
            recv_total, fallback_pin_memory=True
        )
        self.parity1_buffer = allocate_hugepage_tensor(
            default_block_size, fallback_pin_memory=True
        )
        self.parity2_buffer = allocate_hugepage_tensor(
            default_block_size, fallback_pin_memory=True
        )

        # Accumulation buffers — collect parity results across all stripes
        # Each stripe produces block_size worth of parity data
        accum_total = default_block_size * self.num_stripes
        self.parity1_accum = allocate_hugepage_tensor(
            accum_total, fallback_pin_memory=True
        )
        self.parity1_accum.zero_()
        self.parity2_accum = allocate_hugepage_tensor(
            accum_total, fallback_pin_memory=True
        )
        self.parity2_accum.zero_()

        # Register for RDMA
        native.register_buffer(self.data_buffer.data_ptr(), self.data_buffer.numel())
        native.register_buffer(self.recv_buffer.data_ptr(), self.recv_buffer.numel())
        native.register_buffer(self.parity1_buffer.data_ptr(), self.parity1_buffer.numel())
        native.register_buffer(self.parity2_buffer.data_ptr(), self.parity2_buffer.numel())

        buf_type = "GPU" if self.gdr_available else "CPU"
        logger.info(
            "FRCheck: allocated %s buffers block_size=%s recv_total=%s accum_total=%s",
            buf_type, default_block_size, recv_total, accum_total,
        )

    def _compile_stripe_plans(self) -> None:
        """Pre-compile StripePlan for each POA row."""
        native = self._frcheck_native
        if native is None:
            return
        self.stripe_plans.clear()
        ns = native.num_stripes()
        for sid in range(ns):
            role_val = native.get_role_for_stripe(sid)
            plan = StripePlan(
                stripe_id=sid,
                row=list(native.row(sid)),
                role=StripeRole(role_val),
                source_node_ids=list(native.get_source_node_ids(sid)),
                encoder_node_id=native.get_encoder_node_id(sid),
                parity_target_node_id=native.get_parity_target_node_id(sid),
            )
            self.stripe_plans.append(plan)
        logger.info(
            "FRCheck: compiled %d stripe plans, role_counts=%s",
            ns,
            {r.name: sum(1 for p in self.stripe_plans if p.role == r)
             for r in StripeRole},
        )

    def submit_stripe(self, stripe_id: int) -> None:
        """Encode a single stripe using pre-allocated buffers."""
        native = self._frcheck_native
        if native is None:
            raise RuntimeError("FRCheck: native module not available")
        native.submit_stripe_encode(
            stripe_id=stripe_id,
            my_data_addr=self.data_buffer.data_ptr(),
            block_size=self.block_size,
            recv_buf_addr=self.recv_buffer.data_ptr(),
            recv_buf_size=self.recv_buffer.numel(),
            parity1_out_addr=self.parity1_buffer.data_ptr(),
            parity2_out_addr=self.parity2_buffer.data_ptr(),
            parity2_in_addr=self.parity2_buffer.data_ptr(),
        )

        # Accumulate parity results (ENCODER / PARITY_TARGET only)
        plan = self.stripe_plans[stripe_id]
        off = stripe_id * self.block_size
        if plan.role == StripeRole.ENCODER:
            self.parity1_accum[off : off + self.block_size].copy_(
                self.parity1_buffer[:].clone().view(torch.uint8)
            )
        elif plan.role == StripeRole.PARITY_TARGET:
            self.parity2_accum[off : off + self.block_size].copy_(
                self.parity2_buffer[:].clone().view(torch.uint8)
            )

    def get_native(self) -> Any:
        return self._frcheck_native

    def get_resolved_table_path(self) -> Optional[str]:
        return self.frcheck_table_path

    def get_runtime_layout(self) -> Dict[str, Any]:
        return {
            "n": self.frcheck_n,
            "table_path": self.frcheck_table_path,
            "group_layout": self.group_layout,
            "group_id": self.group_id,
            "rank_in_group": self.rank_in_group,
            "group_member_ranks": self.group_member_ranks,
            "node_slot_to_global_rank": self.node_slot_to_global_rank,
            "num_stripes": len(self.stripe_plans),
        }

    def stop(self) -> None:
        if self._frcheck_native is not None and hasattr(self._frcheck_native, "stop"):
            try:
                self._frcheck_native.stop()
            except Exception as e:
                logger.warning("FRCheck: stop() failed: %s", e)

    def cleanup(self) -> None:
        self.stop()
        self._frcheck_native = None
        self.stripe_plans.clear()
