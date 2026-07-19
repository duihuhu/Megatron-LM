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


def _frcheck_debug_enabled() -> bool:
    if os.environ.get("FRCHECK_TRACE_INIT", "0") == "1":
        return True
    try:
        from megatron.training import get_args
        return bool(getattr(get_args(), "frcheck_debug", False))
    except Exception:
        return False


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


@dataclass
class LayerStripeBufs:
    """Per-layer encode buffers: source region + CPU mirror + stripe role bufs."""
    layer_buf_gpu: torch.Tensor
    layer_mirror_cpu: torch.Tensor
    recv_bufs: List[Optional[torch.Tensor]]
    parity1_bufs: List[Optional[torch.Tensor]]
    parity2_bufs: List[Optional[torch.Tensor]]
    remote_layer_bufs: Dict[int, torch.Tensor] = field(default_factory=dict)
    send_layer_bufs: Dict[int, torch.Tensor] = field(default_factory=dict)
    zero_block: Optional[torch.Tensor] = None
    source_on_cpu: bool = False


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
        self.num_stripes: int = 0
        self.num_source_stripes: int = 0  # My SOURCE stripe count
        # Per-layer encode buffers (keyed by layer_idx)
        self.layer_stripe_bufs: Dict[int, LayerStripeBufs] = {}
        self._layer_stripe_alloc_sizes: Dict[int, int] = {}
        # Legacy single-layer aliases (first allocated layer, for backward compat)
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
        self.recovery_dual_failure: bool = False
        # Per-stripe recovery buffers
        self.recovery_helper_bufs: List[Optional[torch.Tensor]] = []
        self.recovery_decoder_bufs: List[Optional[torch.Tensor]] = []
        self.recovery_failed_bufs: List[Optional[torch.Tensor]] = []
        self._retired_runtime_buffers: List[Any] = []

        self._frcheck_recovery_native_cleaned: bool = False
        self._initialized = True

    _rdma_registered_addrs: set = set()
    _layer_block_sizes: Optional[Dict[int, int]] = None  # layer_idx → block_size
    _layer_per_rank_bytes: Optional[Dict[int, Dict[int, int]]] = None  # layer_idx → {rank_in_group(0-based): total_bytes}
    _full_buf: Optional[torch.Tensor] = None
    def allocate_full_buf(self, size_bytes: int):
        """Allocate or reuse cached full tensor buffer (grows-only)."""
        if self._full_buf is not None:
            if self._full_buf.numel() >= size_bytes:
                return self._full_buf
        self._full_buf = allocate_hugepage_tensor(
            size_bytes, fallback_pin_memory=torch.cuda.is_available(),
        )
        return self._full_buf

    def get_layer_capacity_bytes(self, layer_idx: int) -> int:
        """Total GPU/CPU mirror bytes for one layer (n_src * block_size)."""
        bs = self._layer_block_sizes.get(layer_idx) if self._layer_block_sizes else None
        if bs is None:
            raise RuntimeError(
                f"FRCheck: block size not computed for layer_idx={layer_idx}"
            )
        n_src = (self.frcheck_n - 1) * (self.frcheck_n - 2)
        return n_src * bs

    def get_n_filled_for_node(self, layer_idx: int, node_id: int) -> int:
        """Number of non-zero source blocks for POA node (1-based) in this layer."""
        if self._layer_per_rank_bytes is None:
            return 0
        per_rank = self._layer_per_rank_bytes.get(layer_idx, {})
        rank_in_group = node_id - 1
        total_bytes = per_rank.get(rank_in_group, 0)
        if total_bytes <= 0:
            return 0
        bs = self._layer_block_sizes.get(layer_idx, 0) if self._layer_block_sizes else 0
        if bs <= 0:
            return 0
        return (total_bytes + bs - 1) // bs

    def compute_layer_block_sizes(self, layer_groups) -> None:
        """All_gather per-layer sizes across world, keep max in my group."""
        if self._layer_block_sizes is not None:
            return
        if not torch.distributed.is_initialized():
            n_src = (self.frcheck_n - 1) * (self.frcheck_n - 2)
            self._layer_block_sizes = {}
            for g in layer_groups:
                sz = g.total_bytes
                bs = int(((sz + n_src - 1) // n_src + 4095) & ~4095)
                self._layer_block_sizes[g.layer_idx] = max(bs, 4096)
            native = self._frcheck_native
            for g in layer_groups:
                if native is not None:
                    self._allocate_layer_stripe_bufs(
                        native, g.layer_idx, self._layer_block_sizes[g.layer_idx],
                    )
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

        # Persist per-rank per-layer sizes for encoder active-mask computation
        if self.group_member_ranks is not None and all_group_sizes is not None:
            self._layer_per_rank_bytes = {}
            for g in layer_groups:
                lidx = g.layer_idx
                per_rank: Dict[int, int] = {}
                for rig, global_r in enumerate(self.group_member_ranks):
                    d = all_group_sizes[global_r]
                    if d is not None and lidx in d:
                        per_rank[rig] = d[lidx]
                self._layer_per_rank_bytes[lidx] = per_rank

        native = self._frcheck_native
        max_blk = max(self._layer_block_sizes.values()) if self._layer_block_sizes else self.block_size
        for g in layer_groups:
            lidx = g.layer_idx
            bs = self._layer_block_sizes[lidx]
            if native is not None:
                self._allocate_layer_stripe_bufs(native, lidx, bs)

        if _frcheck_debug_enabled():
            logger.info(
                "FRCheck: computed per-layer block sizes (max=%dMB): %s",
                max_blk // (1024*1024),
                [(f"layer_{k}", f"{v//(1024*1024)}MB") for k, v
                 in sorted(self._layer_block_sizes.items())])

    def get_layer_stripe_bufs(self, layer_idx: int) -> LayerStripeBufs:
        bufs = self.layer_stripe_bufs.get(layer_idx)
        if bufs is None:
            raise RuntimeError(
                f"FRCheck: stripe buffers not allocated for layer_idx={layer_idx}"
            )
        return bufs

    def init_frcheck_if_enabled(self) -> None:
        """Load native .so, init RDMA, compile stripe plans."""
        from megatron.training import get_args

        args = get_args()
        self.use_frcheck = bool(getattr(args, "use_frcheck", False))
        if not self.use_frcheck:
            return
        if self._frcheck_native is not None and self.stripe_plans and not self._frcheck_recovery_native_cleaned:
            return
        if self._frcheck_recovery_native_cleaned:
            # Drop the cleaned native handle only when reinitializing in the main
            # process, not at the pre-dataloader safe point where worker fork
            # stability matters most.
            self._frcheck_native = None
            self._frcheck_recovery_native_cleaned = False

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: begin init_frcheck_if_enabled", rank)
        n = self._resolve_frcheck_n(args)
        path = self._resolve_frcheck_table_path(args, n)
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: resolved n=%d table=%s", rank, n, path)
        self._validate_and_build_grouping(n)
        if _frcheck_debug_enabled():
            logger.info(
                "FRCHECK init trace rank=%d: grouping group_id=%s rank_in_group=%s members=%s",
                rank, self.group_id, self.rank_in_group, self.group_member_ranks,
            )
        self._init_frcheck_native(path)
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: native module loaded", rank)

        native = self._frcheck_native
        if native is not None:
            if not native.gdr_available():
                raise RuntimeError(
                    "FRCheck requires GDR (nvidia-peermem module). "
                    "Load nvidia-peermem and ensure IB GPU Direct RDMA is available."
                )
            native.set_require_registered_mr(True)
            self.num_stripes = native.num_stripes()
            if _frcheck_debug_enabled():
                native.set_debug(True)
        if _frcheck_debug_enabled():
            logger.info(
                "FRCheck: GDR required (enabled), n=%d num_stripes=%d",
                self.frcheck_n, self.num_stripes,
            )

        # Init RDMA connections within group (allocates buffers using num_stripes)
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: before _init_rdma", rank)
        self._init_rdma(args)
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: after _init_rdma", rank)

        # Pre-compile stripe plans (reads from native after init_rdma)
        self._compile_stripe_plans()
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: compiled stripe plans", rank)

        # Per-stripe buffers allocated lazily on first save (after adaptive block_size)

    @staticmethod
    def _get_ranks_per_node() -> int:
        env_keys = (
            "FRCHECK_RANKS_PER_NODE",
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
            if _frcheck_debug_enabled():
                logger.info("FRCheck: loaded native module from %s", so_path)
            self._frcheck_native = mod.FRCheckNative(poa_path)
            self.frcheck_table_path = poa_path
            if _frcheck_debug_enabled():
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
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: resolved my_ip=%s", rank, my_ip)

        # Exchange IPs across all ranks (GPU tensors required for NCCL backend)
        ip_bytes = my_ip.encode("utf-8").ljust(64, b"\x00")[:64]
        ip_tensor = torch.tensor([b for b in ip_bytes], dtype=torch.uint8, device="cuda")
        ip_list_tensors = [torch.zeros(64, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: before ip all_gather world_size=%d", rank, world_size)
        torch.distributed.all_gather(ip_list_tensors, ip_tensor)
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: after ip all_gather", rank)
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

        if _frcheck_debug_enabled():
            logger.info(
                "FRCHECK init trace rank=%d: before native.init_rdma rg=%d/%d base_port=%d peers=%s",
                rank, rg, n, base_port, peer_ips,
            )
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
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: after native.init_rdma", rank)

        # Register default buffers
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: before default buffers", rank)
        self._allocate_default_buffers(native, n)
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: after default buffers", rank)

        # Barrier after RDMA init
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: before RDMA barrier", rank)
        torch.distributed.barrier()
        if _frcheck_debug_enabled():
            logger.info("FRCHECK init trace rank=%d: after RDMA barrier", rank)
        if _frcheck_debug_enabled():
            logger.info("FRCheck RDMA: group initialized (rank_in_group=%d/%d)", rg, n)

    def _resolve_my_ip(self) -> str:
        """Determine my IP for listen socket, with multi-NIC per-rank support."""
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else None
        )
        return resolve_ip("FRCHECK", rank=rank, fallback_prefixes=["ECLATIN"])

    def _allocate_default_buffers(self, native, n: int) -> None:
        """Keep legacy default sizes; per-layer save/recovery allocates real buffers."""
        default_block_size = 64 * 1024 * 1024  # 64 MiB fallback (per-layer block_size overrides)

        self.block_size = default_block_size
        native.set_require_registered_mr(True)

        # The current FRCheck paths allocate and register per-layer buffers after
        # layer sizes are known. Eagerly registering this legacy GPU fallback can
        # stall RDMA/GDR init before the save path starts, so leave it disabled
        # unless an old path explicitly opts in for debugging.
        alloc_default = os.environ.get("FRCHECK_ALLOC_DEFAULT_BUFFERS", "0") == "1"
        if not alloc_default:
            self.data_buffer = None
            self.recv_buffer = None
            self.parity1_buffer = None
            self.parity2_buffer = None
            if _frcheck_debug_enabled():
                logger.info(
                    "FRCheck: skipped legacy default buffers block_size=%s",
                    default_block_size,
                )
            return

        recv_total = (n - 2) * default_block_size
        self.data_buffer = torch.empty(default_block_size, dtype=torch.uint8, device="cuda")
        native.register_buffer(self.data_buffer.data_ptr(), self.data_buffer.numel())

        self.recv_buffer = allocate_hugepage_tensor(recv_total, fallback_pin_memory=True)
        self.parity1_buffer = allocate_hugepage_tensor(default_block_size, fallback_pin_memory=True)
        self.parity2_buffer = allocate_hugepage_tensor(default_block_size, fallback_pin_memory=True)

        if _frcheck_debug_enabled():
            logger.info(
                "FRCheck: allocated legacy default buffer block_size=%s recv_total=%s",
                default_block_size, recv_total,
            )

    def _allocate_layer_stripe_bufs(
        self, native, layer_idx: int, block_sz: int, source_on_cpu: bool = False
    ) -> None:
        """Allocate per-stripe buffers for one layer (grows-only per layer_idx)."""
        prev = self._layer_stripe_alloc_sizes.get(layer_idx, 0)
        prev_bufs = self.layer_stripe_bufs.get(layer_idx)
        if (
            prev >= block_sz
            and prev_bufs is not None
            and bool(getattr(prev_bufs, "source_on_cpu", False)) == bool(source_on_cpu)
        ):
            return

        n_src = (self.frcheck_n - 1) * (self.frcheck_n - 2)
        layer_capacity = n_src * block_sz
        self._layer_stripe_alloc_sizes[layer_idx] = block_sz
        recv_total = (self.frcheck_n - 2) * block_sz

        native.set_require_registered_mr(True)

        enc_indices = [sid for sid in range(self.num_stripes)
                       if self.stripe_plans[sid].role == StripeRole.ENCODER]
        par_indices = [sid for sid in range(self.num_stripes)
                       if self.stripe_plans[sid].role == StripeRole.PARITY_TARGET]

        layer_mirror_cpu = allocate_hugepage_tensor(
            layer_capacity, fallback_pin_memory=True,
        )
        layer_mirror_cpu.zero_()
        if source_on_cpu:
            layer_buf_gpu = layer_mirror_cpu
        else:
            layer_buf_gpu = torch.zeros(layer_capacity, dtype=torch.uint8, device="cuda")

        buf_addr = layer_buf_gpu.data_ptr()
        if buf_addr not in self._rdma_registered_addrs:
            native.register_buffer(buf_addr, layer_buf_gpu.numel())
            self._rdma_registered_addrs.add(buf_addr)

        recv_bufs: List[Optional[torch.Tensor]] = [None] * self.num_stripes
        parity1_bufs: List[Optional[torch.Tensor]] = [None] * self.num_stripes
        parity2_bufs: List[Optional[torch.Tensor]] = [None] * self.num_stripes
        remote_layer_bufs: Dict[int, torch.Tensor] = {}
        send_layer_bufs: Dict[int, torch.Tensor] = {}
        my_node = self.rank_in_group + 1 if self.rank_in_group is not None else None
        recv_counts: Dict[int, int] = {}
        send_counts: Dict[int, int] = {}
        if my_node is not None:
            for plan in self.stripe_plans:
                if plan.role == StripeRole.ENCODER:
                    for src_node in plan.source_node_ids:
                        if src_node != my_node:
                            recv_counts[src_node] = recv_counts.get(src_node, 0) + 1
                if my_node in plan.source_node_ids and plan.encoder_node_id != my_node:
                    peer = plan.encoder_node_id
                    send_counts[peer] = send_counts.get(peer, 0) + 1
        for node, count in recv_counts.items():
            if count <= 0:
                continue
            buf = allocate_hugepage_tensor(count * block_sz, fallback_pin_memory=True)
            buf.zero_()
            remote_layer_bufs[node] = buf
            addr = buf.data_ptr()
            if addr not in self._rdma_registered_addrs:
                native.register_buffer(addr, buf.numel())
                self._rdma_registered_addrs.add(addr)
        # CPU staging buffers to gather a layer's scattered source blocks into a
        # contiguous per-peer region, enabling few large aggregated RDMA sends.
        for node, count in send_counts.items():
            if count <= 0:
                continue
            buf = allocate_hugepage_tensor(count * block_sz, fallback_pin_memory=True)
            send_layer_bufs[node] = buf
            addr = buf.data_ptr()
            if addr not in self._rdma_registered_addrs:
                native.register_buffer(addr, buf.numel())
                self._rdma_registered_addrs.add(addr)
        zero_block = allocate_hugepage_tensor(block_sz, fallback_pin_memory=True)
        zero_block.zero_()
        zero_addr = zero_block.data_ptr()
        if zero_addr not in self._rdma_registered_addrs:
            native.register_buffer(zero_addr, zero_block.numel())
            self._rdma_registered_addrs.add(zero_addr)

        if enc_indices:
            r_slices = allocate_hugepage_slices(
                recv_total, len(enc_indices), fallback_pin_memory=True
            )
            p1_slices = allocate_hugepage_slices(
                block_sz, len(enc_indices), fallback_pin_memory=True
            )
            p2_slices = allocate_hugepage_slices(
                block_sz, len(enc_indices), fallback_pin_memory=True
            )
            for slot, sid in enumerate(enc_indices):
                recv_bufs[sid] = r_slices[slot]
                parity1_bufs[sid] = p1_slices[slot]
                parity2_bufs[sid] = p2_slices[slot]
                for buf in (recv_bufs[sid], parity1_bufs[sid], parity2_bufs[sid]):
                    addr = buf.data_ptr()
                    if addr not in self._rdma_registered_addrs:
                        native.register_buffer(addr, buf.numel())
                        self._rdma_registered_addrs.add(addr)

        if par_indices:
            p1_slices = allocate_hugepage_slices(
                block_sz, len(par_indices), fallback_pin_memory=True
            )
            p2_slices = allocate_hugepage_slices(
                block_sz, len(par_indices), fallback_pin_memory=True
            )
            for slot, sid in enumerate(par_indices):
                parity1_bufs[sid] = p1_slices[slot]
                parity2_bufs[sid] = p2_slices[slot]
                for buf in (parity1_bufs[sid], parity2_bufs[sid]):
                    addr = buf.data_ptr()
                    if addr not in self._rdma_registered_addrs:
                        native.register_buffer(addr, buf.numel())
                        self._rdma_registered_addrs.add(addr)

        layer_bufs = LayerStripeBufs(
            layer_buf_gpu=layer_buf_gpu,
            layer_mirror_cpu=layer_mirror_cpu,
            recv_bufs=recv_bufs,
            parity1_bufs=parity1_bufs,
            parity2_bufs=parity2_bufs,
            remote_layer_bufs=remote_layer_bufs,
            send_layer_bufs=send_layer_bufs,
            zero_block=zero_block,
            source_on_cpu=source_on_cpu,
        )
        self.layer_stripe_bufs[layer_idx] = layer_bufs

        if not self.recv_bufs:
            self.recv_bufs = recv_bufs
            self.parity1_bufs = parity1_bufs
            self.parity2_bufs = parity2_bufs

        if _frcheck_debug_enabled():
            lname = f"layer_{layer_idx}" if layer_idx >= 0 else "layer_common"
            logger.info(
                "FRCheck: allocated %s bufs blk=%dMB cap=%dMB: %d enc, %d par",
                lname,
                block_sz // (1024 * 1024),
                layer_capacity // (1024 * 1024),
                len(enc_indices), len(par_indices),
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
        if _frcheck_debug_enabled():
            logger.info(
                "FRCheck: compiled %d stripe plans, role_counts=%s",
                ns,
                {r.name: sum(1 for p in self.stripe_plans if p.role == r)
                 for r in StripeRole},
            )

    def allocate_registered_save_buffer(self, size_bytes: int) -> torch.Tensor:
        """Allocate and register a save-scoped CPU transfer buffer."""
        if size_bytes <= 0:
            raise ValueError("FRCheck save buffer size must be positive")
        native = self._frcheck_native
        if native is None:
            raise RuntimeError("FRCheck native module is not initialized")
        buffer = allocate_hugepage_tensor(
            size_bytes, fallback_pin_memory=torch.cuda.is_available(),
        )
        addr = int(buffer.data_ptr())
        native.register_buffer(addr, int(buffer.numel()))
        self._rdma_registered_addrs.add(addr)
        return buffer

    def release_registered_save_buffers(self, buffers: List[torch.Tensor]) -> None:
        """Unregister save-scoped transfer buffers after native completion."""
        native = self._frcheck_native
        for buffer in buffers:
            addr = int(buffer.data_ptr())
            if addr not in self._rdma_registered_addrs:
                continue
            if native is not None:
                native.unregister_buffer(addr)
            self._rdma_registered_addrs.discard(addr)

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

    @staticmethod
    def _role_for_node_in_row(row: List[int], node_id: int, n: int) -> StripeRole:
        """Stripe role for a node id (1-based) in a POA row."""
        pos = row.index(node_id)
        if pos < n - 2:
            return StripeRole.SOURCE
        if pos == n - 2:
            return StripeRole.ENCODER
        return StripeRole.PARITY_TARGET

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
                'dual_failure': False,
                'failed_node': failed_rank_node,
                'failed_pos': failed_pos,
                'recovery_kind': 'data' if failed_pos < n - 2 else 'parity',
                'round_id': failed_pos,
                'decoder_node': decoder_node,
                'decoder_pos': decoder_pos,
                'helper_nodes': helper_nodes,
                'helper_positions': helper_positions,
                'original_role': int(sp.role),
            })
        return plans

    def _compile_recovery_plans_dual(self, failed_nodes: List[int]) -> List[Dict]:
        """Per-stripe dual-failure plan: helpers send once, decoder recovers both erasures."""
        if len(failed_nodes) != 2:
            raise RuntimeError(
                f"FRCheck dual recovery requires exactly 2 failed nodes, got {failed_nodes}"
            )
        n = self.frcheck_n
        failed_set = set(failed_nodes)
        plans = []
        for sp in self.stripe_plans:
            row = sp.row
            sid = sp.stripe_id
            failed_targets = []
            for fn in failed_nodes:
                try:
                    fp = row.index(fn)
                except ValueError:
                    raise RuntimeError(
                        f"FRCheck dual recovery: failed node {fn} missing in stripe {sid}"
                    )
                failed_targets.append({
                    'failed_node': fn,
                    'failed_pos': fp,
                    'original_role': int(
                        self._role_for_node_in_row(row, fn, n)
                    ),
                })
            survivor_positions = [i for i in range(n) if row[i] not in failed_set]
            if len(survivor_positions) != n - 2:
                raise RuntimeError(
                    f"FRCheck dual recovery: stripe {sid} expected {n - 2} survivors, "
                    f"got {len(survivor_positions)}"
                )
            decoder_pos = survivor_positions[0]
            helper_positions = survivor_positions[1:]
            plans.append({
                'stripe_id': sid,
                'dual_failure': True,
                'failed_nodes': list(failed_nodes),
                'failed_targets': failed_targets,
                'decoder_node': row[decoder_pos],
                'decoder_pos': decoder_pos,
                'helper_nodes': [row[p] for p in helper_positions],
                'helper_positions': helper_positions,
                'survivor_positions': survivor_positions,
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
                self.recovery_decoder_bufs[sid] = torch.cuda.ByteTensor(recv_sz + block_sz)
                native.register_buffer(
                    self.recovery_decoder_bufs[sid].data_ptr(),
                    self.recovery_decoder_bufs[sid].numel())

            elif my_node in plan['helper_nodes']:
                # Allocate block-sized buffer for reading from disk
                self.recovery_helper_bufs[sid] = torch.cuda.ByteTensor(block_sz)
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
        """Initialize hardware recovery mode for up to 2 failed ranks per POA group.

        In node-aware mode (mode 1), each POA group contains one rank from each of
        n physical nodes.  A full-node failure manifests as one failed rank per
        group, so ``--frcheck-failed-ranks`` can list all GPU ranks on the failed
        node(s) — the per-group RS(2) recovery handles up to 2 node failures.

        Single failure: per-failed-rank cyclic decoder/helper plan (HW1).
        Dual failure: one plan per stripe; helpers send once, decoder recovers both
        erasures and sends to both failed ranks (HW2).

        Returns a dict mapping failed_global_rank → recovery context.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError("FRCheck hardware recovery requires torch.distributed")

        world_size = torch.distributed.get_world_size()
        my_rank = torch.distributed.get_rank()

        self.is_recovery_mode = True
        self.failed_global_ranks = list(failed_global_ranks)
        self.recovery_dual_failure = False

        recovery_contexts: Dict[int, Dict] = {}
        failed_in_my_group: List[int] = []
        all_groups_failed: Dict[int, List[int]] = {}  # group_id → failed ranks in that group

        for failed_rank in failed_global_ranks:
            group_id = self._get_group_id(failed_rank, world_size, self.frcheck_n)
            failed_rig = self._get_rank_in_group(failed_rank, world_size, self.frcheck_n)
            failed_node = failed_rig + 1  # 1-based in POA

            ctx = {
                'group_id': group_id,
                'failed_rig': failed_rig,
                'failed_node': failed_node,
            }
            recovery_contexts[failed_rank] = ctx

            if group_id not in all_groups_failed:
                all_groups_failed[group_id] = []
            all_groups_failed[group_id].append(failed_rank)

            if group_id == self.group_id:
                failed_in_my_group.append(failed_rank)

        # Validate per-group limits across all groups (not just my group).
        # RS(2) can recover at most 2 erasures per stripe → at most 2 failed
        # node-slots per POA group.  In node-aware mode a full-node failure
        # produces one failed rank per group, so many global ranks are legal
        # as long as no single group exceeds 2.
        for gid, ranks in all_groups_failed.items():
            if len(ranks) > 2:
                raise RuntimeError(
                    f"FRCheck: at most 2 failed ranks per POA group, "
                    f"group {gid} has {len(ranks)} failed ranks: {ranks}"
                )
            rigs = [recovery_contexts[r]['failed_rig'] for r in ranks]
            if len(rigs) != len(set(rigs)):
                raise RuntimeError(
                    f"FRCheck: duplicate node slots in group {gid}: "
                    f"failed ranks {ranks} map to rigs {rigs} — "
                    f"two ranks from the same node in one group"
                )

        if failed_in_my_group:
            if len(failed_in_my_group) == 2:
                failed_nodes = sorted(
                    recovery_contexts[fr]['failed_node'] for fr in failed_in_my_group
                )
                self.recovery_stripe_plans = self._compile_recovery_plans_dual(
                    failed_nodes
                )
                self.recovery_dual_failure = True
                for fr in failed_in_my_group:
                    recovery_contexts[fr]['recovery_plans'] = self.recovery_stripe_plans
            else:
                fr = failed_in_my_group[0]
                failed_node = recovery_contexts[fr]['failed_node']
                self.recovery_stripe_plans = self._compile_recovery_plans(failed_node)
                recovery_contexts[fr]['recovery_plans'] = self.recovery_stripe_plans

            native = self._frcheck_native
            if native is not None:
                failed_nodes = sorted(
                    recovery_contexts[fr]['failed_node'] for fr in failed_in_my_group
                )
                native.init_recovery_plans(failed_nodes)

        return recovery_contexts

    def _unregister_all_buffers(self) -> None:
        """Unregister all RDMA-registered buffers (ECLATIN-style cleanup)."""
        native = self._frcheck_native
        if native is None:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        for addr in list(self._rdma_registered_addrs):
            try:
                native.unregister_buffer(addr)
            except Exception as e:
                logger.warning(
                    "FRCheck: rank %d failed to unregister buffer 0x%x: %s",
                    rank, addr, e,
                )
        self._rdma_registered_addrs.clear()

    def release_save_layer_buffers(self, empty_cuda_cache: bool = True) -> None:
        """Release per-layer save buffers that can be rebuilt at the next checkpoint."""
        if not self.layer_stripe_bufs:
            return
        self._unregister_all_buffers()
        self.layer_stripe_bufs.clear()
        self._layer_stripe_alloc_sizes.clear()
        self._layer_block_sizes = None
        self._layer_per_rank_bytes = None
        self.recv_bufs = []
        self.parity1_bufs = []
        self.parity2_bufs = []
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stop(self) -> None:
        """Stop C++ encode workers and RS pool (synchronous, ECLATIN-style)."""
        if self._frcheck_native is None or not hasattr(self._frcheck_native, "stop"):
            return
        try:
            self._frcheck_native.stop()
        except Exception as e:
            logger.warning("FRCheck: stop() failed: %s", e)

    def stop_recovery_runtime(self) -> None:
        """Stop only C++ recovery workers when supported by the native module."""
        if self._frcheck_native is None:
            return
        try:
            if hasattr(self._frcheck_native, "stop_recovery_runtime"):
                self._frcheck_native.stop_recovery_runtime()
            else:
                self.stop()
        except Exception as e:
            logger.warning("FRCheck: stop_recovery_runtime() failed: %s", e)

    def cleanup_recovery_runtime(self) -> None:
        """Release recovery RDMA/native resources before DataLoader workers fork."""
        if self._frcheck_native is None:
            return
        try:
            if hasattr(self._frcheck_native, "cleanup_recovery_runtime"):
                self._frcheck_native.cleanup_recovery_runtime()
            else:
                self.stop()
        except Exception as e:
            logger.warning("FRCheck: cleanup_recovery_runtime() failed: %s", e)

    def _clear_runtime_buffers_after_native_cleanup(self) -> None:
        self._frcheck_native = None
        self._frcheck_recovery_native_cleaned = False
        self._rdma_registered_addrs.clear()
        self.stripe_plans.clear()
        self.layer_stripe_bufs.clear()
        self._layer_stripe_alloc_sizes.clear()
        self._layer_block_sizes = None
        self.recv_bufs = []
        self.parity1_bufs = []
        self.parity2_bufs = []
        self.data_buffer = None
        self.recv_buffer = None
        self.parity1_buffer = None
        self.parity2_buffer = None

    def _clear_recovery_native_handle(self) -> None:
        # Keep old tensor/native objects alive at the pre-dataloader safe point,
        # but remove them from active maps so the next save reinitializes and
        # re-registers all RDMA buffers with the fresh native module.
        retired = [self._frcheck_native]
        retired.extend(self.layer_stripe_bufs.values())
        retired.extend([
            self.data_buffer, self.recv_buffer,
            self.parity1_buffer, self.parity2_buffer,
        ])
        self._retired_runtime_buffers.extend(x for x in retired if x is not None)
        self._frcheck_native = None
        self._frcheck_recovery_native_cleaned = False
        self._rdma_registered_addrs.clear()
        self.stripe_plans.clear()
        self.layer_stripe_bufs.clear()
        self._layer_stripe_alloc_sizes.clear()
        self._layer_block_sizes = None
        self.recv_bufs = []
        self.parity1_bufs = []
        self.parity2_bufs = []
        self.data_buffer = None
        self.recv_buffer = None
        self.parity1_buffer = None
        self.parity2_buffer = None

    def end_recovery(self) -> None:
        """Clear recovery-mode state without tearing down RDMA / RS pool."""
        self.is_recovery_mode = False
        self.failed_global_ranks = []
        self.recovery_stripe_plans = []
        self.recovery_dual_failure = False
        self.recovery_helper_bufs = []
        self.recovery_decoder_bufs = []
        self.recovery_failed_bufs = []

    def cleanup_recovery(self, teardown: bool = False, sync: bool = True) -> None:
        """Cleanup recovery-only state without full native teardown when possible."""
        if not teardown:
            return
        try:
            if sync and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                torch.distributed.barrier()
            self.cleanup_recovery_runtime()
            self._clear_recovery_native_handle()
            self.end_recovery()
        except Exception as e:
            logger.warning("FRCheck: error during recovery cleanup: %s", e)

    def cleanup(self, teardown: bool = False, sync: bool = True) -> None:
        """Cleanup FRCheck resources (ECLATIN-style).

        Save/train/eval: do not call — native module stays alive for reuse.
        Load path: call with teardown=True after recovery completes.
        """
        if not teardown:
            return
        try:
            if sync and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                torch.distributed.barrier()
            self._unregister_all_buffers()
            self.stop()
            self._clear_runtime_buffers_after_native_cleanup()
            self.end_recovery()
        except Exception as e:
            logger.warning("FRCheck: error during manager cleanup: %s", e)

    def __del__(self):
        # Do not stop native here — save/eval keep resources alive (ECLATIN save path).
        # Load path calls cleanup(teardown=True) explicitly before returning state_dict.
        pass
