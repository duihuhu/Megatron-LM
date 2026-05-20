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
    allocate_hugepage_slices,
    allocate_hugepage_tensor,
)
from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip

logger = getLogger(__name__)


class StripeRole(IntEnum):
    """Role of this rank within a single POA stripe."""
    SOURCE = 0
    ENCODER = 1
    PARITY_TARGET = 2
    # Recovery roles (mirror save roles)
    HELPER = 3        # Sends stripe block to decoder
    DECODER = 4       # Receives, RS-decodes, sends to failed rank
    FAILED_RANK = 5   # Receives decoded blocks, assembles per-layer


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
        self.num_source_stripes: int = 0  # My SOURCE stripe count
        # Per-stripe buffers for source data (async pipeline)
        self.stripe_data_bufs: List[Optional[torch.Tensor]] = []
        # Per-stripe recv/parity buffers (avoids slot reuse race with RDMA)
        self.recv_bufs: List[Optional[torch.Tensor]] = []
        self.parity1_bufs: List[Optional[torch.Tensor]] = []
        self.parity2_bufs: List[Optional[torch.Tensor]] = []
        # Keep old single-buffer names for backward compat (tests, legacy code)
        self.data_buffer: Optional[torch.Tensor] = None
        self.recv_buffer: Optional[torch.Tensor] = None
        self.parity1_buffer: Optional[torch.Tensor] = None
        self.parity2_buffer: Optional[torch.Tensor] = None
        # Recovery state
        self.is_recovery_mode: bool = False
        self.failed_global_ranks: List[int] = []
        self.recovery_stripe_plans: List[Dict] = []  # Per-stripe recovery plan
        # Per-stripe recovery buffers
        self.recovery_helper_bufs: List[Optional[torch.Tensor]] = []
        self.recovery_decoder_bufs: List[Optional[torch.Tensor]] = []
        self.recovery_failed_bufs: List[Optional[torch.Tensor]] = []

        self._initialized = True

    _cached_layer_buffers: Dict[int, torch.Tensor] = {}
    _rdma_registered_addrs: set = set()
    _layer_block_sizes: Optional[Dict[int, int]] = None  # layer_idx → block_size

    def allocate_layer_buffer(self, layer_idx: int, size_bytes: int, gdr: bool):
        """Allocate or reuse a cached per-layer tensor_buffer."""
        cached = self._cached_layer_buffers.get(layer_idx)
        if cached is not None and cached.numel() >= size_bytes:
            return cached
        if gdr:
            buf = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        else:
            buf = allocate_hugepage_tensor(size_bytes, fallback_pin_memory=True)
        self._cached_layer_buffers[layer_idx] = buf
        return buf

    def compute_layer_block_sizes(self, layer_groups) -> None:
        """All_gather per-layer sizes across world, keep max in my group."""
        if self._layer_block_sizes is not None:
            return
        if not torch.distributed.is_initialized():
            self._layer_block_sizes = {-1: self.block_size}
            return
        ws = torch.distributed.get_world_size()
        n_src = (self.frcheck_n - 1) * (self.frcheck_n - 2)
        self._layer_block_sizes = {}

        # Collect layer_idx → total_bytes for all groups
        my_sizes: Dict[int, int] = {}
        for g in layer_groups:
            my_sizes[g.layer_idx] = g.total_bytes

        # Exchange across all ranks
        all_group_sizes = [None] * ws
        torch.distributed.all_gather_object(all_group_sizes, my_sizes)

        # For each of my layers, compute max across group members
        if self.group_member_ranks is None:
            for g in layer_groups:
                sz = g.total_bytes
                bs = int(((sz + n_src - 1) // n_src + 4095) & ~4095)
                self._layer_block_sizes[g.layer_idx] = max(bs, 4096)
        else:
            for g in layer_groups:
                lidx = g.layer_idx
                vals = [g.total_bytes]
                for r in self.group_member_ranks:
                    if r != torch.distributed.get_rank():
                        d = all_group_sizes[r]
                        if d is not None and lidx in d:
                            vals.append(d[lidx])
                sz = max(vals)
                bs = int(((sz + n_src - 1) // n_src + 4095) & ~4095)
                self._layer_block_sizes[lidx] = max(bs, 4096)

        # Allocate per-stripe buffers sized for the max block across all layers
        max_blk = max(self._layer_block_sizes.values()) if self._layer_block_sizes else self.block_size
        self._allocate_per_stripe_bufs(self._frcheck_native, max_blk)

        logger.info(
            "FRCheck: computed per-layer block sizes (max=%dMB): %s",
            max_blk // (1024*1024),
            [(f"layer_{k}", f"{v//(1024*1024)}MB") for k, v
             in sorted(self._layer_block_sizes.items())])

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

        # Per-stripe buffers allocated lazily on first save (after adaptive block_size)

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

        # Exchange IPs across all ranks (GPU tensors required for NCCL backend)
        ip_bytes = my_ip.encode("utf-8").ljust(64, b"\x00")[:64]
        ip_tensor = torch.tensor([b for b in ip_bytes], dtype=torch.uint8, device="cuda")
        ip_list_tensors = [torch.zeros(64, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
        torch.distributed.all_gather(ip_list_tensors, ip_tensor)
        ip_list = []
        for t in ip_list_tensors:
            raw = bytes(t.cpu().tolist()).rstrip(b"\x00")
            ip_list.append(raw.decode("utf-8") if raw else "127.0.0.1")

        # Build peer IPs for my group
        peer_ips = []
        for i in range(n):
            peer_global_rank = self._get_rank_by_group_position(self.group_id, i, world_size, n)
            peer_ips.append(ip_list[peer_global_rank])

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
        """Determine my IP for listen socket, with multi-NIC per-rank support."""
        return resolve_ip("FRCHECK", fallback_prefixes=["ECLATIN"])

    def _allocate_default_buffers(self, native, n: int) -> None:
        """Allocate back-compat data/recv/parity buffers."""
        default_block_size = 64 * 1024 * 1024  # 64 MiB fallback (per-layer block_size overrides)

        self.block_size = default_block_size
        recv_total = (n - 2) * default_block_size

        # Placeholder data_buffer (per-stripe bufs allocated after stripe plans compiled)
        if self.gdr_available:
            self.data_buffer = torch.cuda.ByteTensor(default_block_size)
        else:
            self.data_buffer = allocate_hugepage_tensor(
                default_block_size, fallback_pin_memory=True
            )
        native.register_buffer(self.data_buffer.data_ptr(), self.data_buffer.numel())

        # Backward-compat single buffers for sync path / tests
        self.recv_buffer = allocate_hugepage_tensor(recv_total, fallback_pin_memory=True)
        self.parity1_buffer = allocate_hugepage_tensor(default_block_size, fallback_pin_memory=True)
        self.parity2_buffer = allocate_hugepage_tensor(default_block_size, fallback_pin_memory=True)

        buf_type = "GPU" if self.gdr_available else "CPU"
        logger.info(
            "FRCheck: allocated %s buffers block_size=%s recv_total=%s",
            buf_type, default_block_size, recv_total,
        )

    def _allocate_per_stripe_bufs(self, native, block_sz: int) -> None:
        """Allocate per-stripe buffers with given block size."""
        if hasattr(self, '_per_stripe_alloc_size') and self._per_stripe_alloc_size >= block_sz:
            return  # already allocated with sufficient size
        self._per_stripe_alloc_size = block_sz
        recv_total = (self.frcheck_n - 2) * block_sz

        src_indices = [sid for sid in range(self.num_stripes)
                       if self.stripe_plans[sid].role == StripeRole.SOURCE]
        enc_indices = [sid for sid in range(self.num_stripes)
                       if self.stripe_plans[sid].role == StripeRole.ENCODER]
        par_indices = [sid for sid in range(self.num_stripes)
                       if self.stripe_plans[sid].role == StripeRole.PARITY_TARGET]

        self.stripe_data_bufs = [None] * self.num_stripes
        self.recv_bufs = [None] * self.num_stripes
        self.parity1_bufs = [None] * self.num_stripes
        self.parity2_bufs = [None] * self.num_stripes

        if src_indices:
            if self.gdr_available:
                base = torch.cuda.ByteTensor(block_sz * len(src_indices))
                for slot, sid in enumerate(src_indices):
                    self.stripe_data_bufs[sid] = base[slot * block_sz : (slot + 1) * block_sz]
                    native.register_buffer(self.stripe_data_bufs[sid].data_ptr(), self.stripe_data_bufs[sid].numel())
            else:
                slices = allocate_hugepage_slices(block_sz, len(src_indices), fallback_pin_memory=True)
                for slot, sid in enumerate(src_indices):
                    self.stripe_data_bufs[sid] = slices[slot]
                    native.register_buffer(self.stripe_data_bufs[sid].data_ptr(), self.stripe_data_bufs[sid].numel())

        if enc_indices:
            r_slices = allocate_hugepage_slices(recv_total, len(enc_indices), fallback_pin_memory=True)
            p1_slices = allocate_hugepage_slices(block_sz, len(enc_indices), fallback_pin_memory=True)
            p2_slices = allocate_hugepage_slices(block_sz, len(enc_indices), fallback_pin_memory=True)
            for slot, sid in enumerate(enc_indices):
                self.recv_bufs[sid] = r_slices[slot]
                self.parity1_bufs[sid] = p1_slices[slot]
                self.parity2_bufs[sid] = p2_slices[slot]
                native.register_buffer(self.recv_bufs[sid].data_ptr(), self.recv_bufs[sid].numel())
                native.register_buffer(self.parity1_bufs[sid].data_ptr(), self.parity1_bufs[sid].numel())
                native.register_buffer(self.parity2_bufs[sid].data_ptr(), self.parity2_bufs[sid].numel())

        if par_indices:
            p2_slices = allocate_hugepage_slices(block_sz, len(par_indices), fallback_pin_memory=True)
            for slot, sid in enumerate(par_indices):
                self.parity2_bufs[sid] = p2_slices[slot]
                native.register_buffer(self.parity2_bufs[sid].data_ptr(), self.parity2_bufs[sid].numel())

        logger.info(
            "FRCheck: allocated per-stripe bufs blk=%dMB: %d source, %d enc, %d par",
            block_sz // (1024*1024), len(src_indices), len(enc_indices), len(par_indices),
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
                self.parity1_buffer[:].view(torch.uint8)
            )
        elif plan.role == StripeRole.PARITY_TARGET:
            self.parity2_accum[off : off + self.block_size].copy_(
                self.parity2_buffer[:].view(torch.uint8)
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

    # ---- Hardware recovery ----

    def _compile_recovery_plans(self, failed_rank_node: int) -> List[Dict]:
        """For each stripe, compute recovery roles (DECODER/HELPER/FAILED_RANK)
        given the failed rank's node id (1-based in the POA table).

        The n-2 ranks cyclically to the right of the failed rank in each POA row
        collaborate: 1st right = DECODER, remaining n-3 = HELPERS.
        """
        n = self.frcheck_n
        plans = []
        for sp in self.stripe_plans:
            row = sp.row  # list of 1-based node IDs
            sid = sp.stripe_id
            try:
                failed_pos = row.index(failed_rank_node)
            except ValueError:
                continue  # Failed rank not in this stripe (should not happen)

            # n-2 right-side positions (cyclic)
            decoder_pos = (failed_pos + 1) % n
            helper_positions = [(failed_pos + 2 + i) % n for i in range(n - 3)]

            decoder_node = row[decoder_pos]
            helper_nodes = [row[p] for p in helper_positions]

            plans.append({
                'stripe_id': sid,
                'failed_node': failed_rank_node,
                'failed_pos': failed_pos,
                'decoder_node': decoder_node,
                'decoder_pos': decoder_pos,
                'helper_nodes': helper_nodes,
                'helper_positions': helper_positions,
                'original_role': int(sp.role),
            })
        return plans

    def _allocate_recovery_bufs(self, native, block_sz: int) -> None:
        """Allocate per-stripe buffers for hardware recovery.

        HELPER: one block-sized buffer to read from disk → send to decoder.
        DECODER: recv buffer for (n-3) helper blocks, + one decode output buffer.
        FAILED_RANK: recv buffer for each stripe's recovered block.
        """
        n = self.frcheck_n
        num_helper = n - 3
        ns = self.num_stripes

        self.recovery_helper_bufs = [None] * ns
        self.recovery_decoder_bufs = [None] * ns
        self.recovery_failed_bufs = [None] * ns

        for plan in self.recovery_stripe_plans:
            sid = plan['stripe_id']
            my_node = self.rank_in_group + 1

            if my_node == plan['decoder_node']:
                # Allocate recv buffer for helpers' blocks + decode output
                recv_sz = num_helper * block_sz
                from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
                    allocate_hugepage_slices, allocate_hugepage_tensor,
                )
                if self.gdr_available:
                    self.recovery_decoder_bufs[sid] = torch.cuda.ByteTensor(recv_sz + block_sz)
                else:
                    self.recovery_decoder_bufs[sid] = allocate_hugepage_tensor(
                        recv_sz + block_sz, fallback_pin_memory=True)
                native.register_buffer(
                    self.recovery_decoder_bufs[sid].data_ptr(),
                    self.recovery_decoder_bufs[sid].numel())

            elif my_node in plan['helper_nodes']:
                # Allocate block-sized buffer for reading from disk
                if self.gdr_available:
                    self.recovery_helper_bufs[sid] = torch.cuda.ByteTensor(block_sz)
                else:
                    from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
                        allocate_hugepage_tensor,
                    )
                    self.recovery_helper_bufs[sid] = allocate_hugepage_tensor(
                        block_sz, fallback_pin_memory=True)
                native.register_buffer(
                    self.recovery_helper_bufs[sid].data_ptr(),
                    self.recovery_helper_bufs[sid].numel())

            elif my_node == plan['failed_node']:
                # Allocate recv buffer for decoded block
                from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
                    allocate_hugepage_tensor,
                )
                self.recovery_failed_bufs[sid] = allocate_hugepage_tensor(
                    block_sz, fallback_pin_memory=True)
                native.register_buffer(
                    self.recovery_failed_bufs[sid].data_ptr(),
                    self.recovery_failed_bufs[sid].numel())

    def init_frcheck_hardware_recovery(self, failed_global_ranks: List[int]) -> Dict[int, Dict]:
        """Initialize hardware recovery mode for a list of failed global ranks.

        Each failed rank is in a different POA group (same-node failure pattern).
        Returns a dict mapping failed_global_rank → recovery context with:
          - 'group_id': group that contains this failed rank
          - 'failed_rig': rank_in_group of the failed rank
          - 'failed_node': 1-based node id in the POA table
          - 'recovery_plans': per-stripe recovery plan
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError("FRCheck hardware recovery requires torch.distributed")

        world_size = torch.distributed.get_world_size()
        my_rank = torch.distributed.get_rank()

        self.is_recovery_mode = True
        self.failed_global_ranks = list(failed_global_ranks)

        recovery_contexts: Dict[int, Dict] = {}

        for failed_rank in failed_global_ranks:
            # Determine which group this failed rank belongs to
            group_id = self._get_group_id(failed_rank, world_size, self.frcheck_n)
            failed_rig = self._get_rank_in_group(failed_rank, world_size, self.frcheck_n)
            failed_node = failed_rig + 1  # 1-based in POA

            ctx = {
                'group_id': group_id,
                'failed_rig': failed_rig,
                'failed_node': failed_node,
                'recovery_plans': self._compile_recovery_plans(failed_node),
            }
            recovery_contexts[failed_rank] = ctx

            if my_rank == failed_rank:
                logger.info(
                    "FRCheck hardware recovery: I am the failed rank %d (group %d, rig %d)",
                    failed_rank, group_id, failed_rig)
            elif group_id == self.group_id:
                logger.info(
                    "FRCheck hardware recovery: I am in group %d with failed rank %d (rig %d)",
                    group_id, failed_rank, failed_rig)

        # If I'm in a group with a failed rank, compile my per-stripe recovery role
        my_recovery_ctx = None
        for fr, ctx in recovery_contexts.items():
            if ctx['group_id'] == self.group_id:
                my_recovery_ctx = ctx
                break

        if my_recovery_ctx is not None:
            self.recovery_stripe_plans = my_recovery_ctx['recovery_plans']
            # Determine my role for each stripe and summary
            my_node = self.rank_in_group + 1
            n_decoder = n_helper = n_failed = 0
            for plan in self.recovery_stripe_plans:
                if my_node == plan['decoder_node']:
                    n_decoder += 1
                elif my_node in plan['helper_nodes']:
                    n_helper += 1
                elif my_node == plan['failed_node']:
                    n_failed += 1
            logger.info(
                "FRCheck recovery: my roles — %d decoder, %d helper, %d failed "
                "(total %d relevant stripes)",
                n_decoder, n_helper, n_failed, len(self.recovery_stripe_plans))

        return recovery_contexts

    def stop(self, timeout: float = 10.0) -> None:
        """Stop C++ encode workers with a timeout to prevent hangs on exit."""
        if self._frcheck_native is None or not hasattr(self._frcheck_native, "stop"):
            return
        import threading
        exc = []
        def _do_stop():
            try:
                self._frcheck_native.stop()
            except Exception as e:
                exc.append(e)
        t = threading.Thread(target=_do_stop, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            logger.warning(
                "FRCheck: C++ stop() timed out after %.1fs — encode pool threads "
                "will be reclaimed by the OS on exit", timeout,
            )
        elif exc:
            logger.warning("FRCheck: stop() failed: %s", exc[0])

    def cleanup(self) -> None:
        self.stop(timeout=5.0)  # shorter timeout at explicit cleanup
        self._frcheck_native = None
        self.stripe_plans.clear()
        self._cached_layer_buffers = {}
        self._rdma_registered_addrs = set()
