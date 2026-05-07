# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""LEGACY checkpoint path for Gemini Replicas (multi-replica round-robin).
Mirrors ecnaive_legacy.py: decompose state dict, exchange via C++ native,
save/load .pt files with torch.save / torch.load.
"""

from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from megatron.core.dist_checkpointing.strategies.gemini_replicas_manager import (
    GeminiReplicasManager,
)
from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
    allocate_hugepage_tensor,
)
from megatron.core.dist_checkpointing.strategies.state_dict_decomposer import (
    DecomposedStateDict,
    GlobalMetadataRegistry,
    TensorMetadata,
    decompose_state_dict,
    extract_tensors_from_continuous_buffer,
    reconstruct_state_dict,
)

from megatron.core.dist_checkpointing.strategies.async_utils import (
    get_or_create_global_gloo_group,
)

logger = getLogger(__name__)


def _cpu_uint8_view(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    return t.contiguous().view(torch.uint8).reshape(-1)


def _checkpoint_dir_from_path(checkpoint_name: str) -> Path:
    checkpoint_path = Path(checkpoint_name)
    return checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent


def _build_global_registry(
    local_metadata: List[TensorMetadata],
    local_non_tensor: Dict[str, Any],
) -> Tuple[Dict[int, List[TensorMetadata]], Dict[int, Dict[str, Any]]]:
    if not torch.distributed.is_initialized():
        return {0: local_metadata}, {0: local_non_tensor}

    world_size = torch.distributed.get_world_size()
    gathered_meta: List[Any] = [None for _ in range(world_size)]
    gathered_non_tensor: List[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_meta, local_metadata)
    torch.distributed.all_gather_object(gathered_non_tensor, local_non_tensor)
    rank_metadata = {r: gathered_meta[r] for r in range(world_size)}
    rank_non_tensor = {r: gathered_non_tensor[r] for r in range(world_size)}
    return rank_metadata, rank_non_tensor


def _tensor_infos_to_local_metadata(
    rank: int, tensor_infos: List[Any]
) -> List[TensorMetadata]:
    out: List[TensorMetadata] = []
    for info in tensor_infos:
        out.append(
            TensorMetadata(
                key=info.key,
                shape=tuple(info.shape),
                dtype=str(info.dtype),
                size_bytes=info.size_bytes,
                global_offset=tuple(info.global_offset) if info.global_offset else tuple(),
                shard_index=info.shard_index if info.shard_index is not None else 0,
                chunk_type="data",
                target_rank=rank,
                source_rank=rank,
            )
        )
    return out


def _infer_flat_key_roots(main_payload: Dict[str, Any]) -> Set[str]:
    if "flat_key_roots" in main_payload:
        return set(main_payload["flat_key_roots"])
    flat_key_roots: Set[str] = set()
    for info in main_payload.get("tensor_infos", []):
        first_seg = info["key"].split(".")[0]
        if first_seg == "model" or (
            first_seg.startswith("model")
            and len(first_seg) > 5
            and first_seg[5:].isdigit()
        ):
            flat_key_roots.add(first_seg)
    return flat_key_roots


def state_dict_from_gemini_replicas_main_metadata_only(
    main_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Build state_dict from gemini_replicas main file payload (non-distributed)."""
    flat_key_roots = _infer_flat_key_roots(main_payload)
    if isinstance(main_payload.get("tensor_buffer"), torch.Tensor):
        tb = main_payload["tensor_buffer"]
        buf = tb.detach().contiguous().reshape(-1).view(torch.uint8)
        tensor_infos = main_payload["tensor_infos"]
        tensor_data = extract_tensors_from_continuous_buffer(buf, tensor_infos)
        decomposed = DecomposedStateDict(
            non_tensor_data=main_payload["non_tensor_data"],
            tensor_infos=tensor_infos,
            tensor_data=tensor_data,
            flat_key_roots=flat_key_roots,
        )
        return reconstruct_state_dict(decomposed)

    decomposed = DecomposedStateDict(
        non_tensor_data=main_payload["non_tensor_data"],
        tensor_infos=[],
        tensor_data=[],
        flat_key_roots=flat_key_roots,
    )
    return reconstruct_state_dict(decomposed)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_gemini_replicas_legacy_checkpoint(
    state_dict: Dict[str, Any], checkpoint_name: str
) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )

    manager = GeminiReplicasManager()
    manager.init_gemini_replicas_if_enabled()
    if manager._gemini_replicas_native is None:
        raise RuntimeError(
            "Gemini Replicas native module is not available in legacy save path"
        )

    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes

    safety_margin = max(int(total_tensor_size * 0.01), 1024 * 1024)
    tensor_buffer = allocate_hugepage_tensor(
        total_tensor_size + safety_margin,
        fallback_pin_memory=manager.gemini_replicas_pin_memory
        and torch.cuda.is_available(),
    )
    tensor_buffer.zero_()

    offset = 0
    local_tensor_metadata: List[TensorMetadata] = []
    for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
        tensor_bytes = info.size_bytes
        tensor_bytes_view = _cpu_uint8_view(tensor)
        if tensor_bytes_view.numel() != tensor_bytes:
            raise RuntimeError(
                f"Gemini Replicas legacy save: tensor bytes mismatch for {info.key}, "
                f"expected={tensor_bytes}, got={tensor_bytes_view.numel()}"
            )
        tensor_buffer[offset : offset + tensor_bytes].copy_(tensor_bytes_view)
        info.offset = offset
        local_tensor_metadata.append(
            TensorMetadata(
                key=info.key,
                shape=info.shape,
                dtype=str(info.dtype),
                size_bytes=info.size_bytes,
                global_offset=tuple(info.global_offset)
                if info.global_offset
                else tuple(),
                shard_index=info.shard_index if info.shard_index is not None else 0,
                chunk_type="data",
                target_rank=rank,
                source_rank=rank,
            )
        )
        offset += tensor_bytes

    # Build global metadata registry (for size exchange)
    rank_metadata, _ = _build_global_registry(
        local_tensor_metadata, decomposed.non_tensor_data
    )

    if manager.use_rdma:
        manager.register_buffer(tensor_buffer)

    # Exchange buffer sizes via gloo all_gather
    send_buffer_size = total_tensor_size
    global_gloo_group = get_or_create_global_gloo_group()
    size_tensor = torch.tensor([send_buffer_size], dtype=torch.long, device="cpu")
    all_sizes = [torch.zeros_like(size_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, size_tensor, group=global_gloo_group)
    rank_sizes = {r: int(all_sizes[r][0].item()) for r in range(world_size)}

    # Determine source ranks (ranks whose target list includes us)
    target_ranks = manager._calculate_target_ranks(rank, world_size)
    source_ranks = []
    for src_r in range(world_size):
        if src_r == rank:
            continue
        src_targets = manager._calculate_target_ranks(src_r, world_size)
        if rank in src_targets:
            source_ranks.append(src_r)

    logger.info(
        f"Gemini Replicas legacy save rank {rank}: "
        f"targets={target_ranks}, sources={source_ranks}, "
        f"buffer={send_buffer_size / (1024**2):.2f} MB"
    )

    # Allocate receive buffers
    receive_buffers: Dict[int, torch.Tensor] = {}
    for src_r in source_ranks:
        src_size = rank_sizes[src_r]
        recv_buf = torch.empty(src_size, dtype=torch.uint8)
        if manager.use_rdma:
            manager.register_buffer(recv_buf)
        receive_buffers[src_r] = recv_buf

    torch.distributed.barrier()

    # Submit to C++ native and execute exchange
    native = manager._gemini_replicas_native
    send_addr = tensor_buffer.data_ptr()
    native.submit_send_buffer(send_addr, send_buffer_size)

    for src_r, recv_buf in receive_buffers.items():
        native.submit_recv_buffer(src_r, recv_buf.data_ptr(), recv_buf.numel())

    logger.info(
        f"Gemini Replicas legacy save rank {rank}: executing C++ exchange "
        f"(send to {len(target_ranks) - 1} targets, recv from {len(source_ranks)} sources)..."
    )
    native.execute_exchange()
    logger.info(f"Gemini Replicas legacy save rank {rank}: C++ exchange done")

    # ===== Exchange metadata so replica files are self-contained =====
    # Each rank needs to know every source rank's tensor_infos + non_tensor_data
    # so that during hardware recovery the sender can provide full metadata.
    my_meta = {
        "tensor_infos": [
            {
                "key": info.key,
                "shape": list(info.shape),
                "dtype": str(info.dtype),
                "offset": info.offset,
                "size_bytes": info.size_bytes,
            }
            for info in decomposed.tensor_infos
        ],
        "non_tensor_data": decomposed.non_tensor_data,
        "tensor_buffer_size": total_tensor_size,
        "flat_key_roots": list(decomposed.flat_key_roots)
        if decomposed.flat_key_roots
        else [],
    }
    all_meta: List[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_meta, my_meta)
    rank_meta = {r: all_meta[r] for r in range(world_size)}

    # Save .pt files
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    full_tensor_buffer = tensor_buffer[:total_tensor_size].detach().clone()

    main_payload = {
        "version": 1,
        "format": "gemini_replicas_torch_legacy",
        "rank": rank,
        "non_tensor_data": decomposed.non_tensor_data,
        "tensor_infos": rank_meta[rank]["tensor_infos"],
        "tensor_buffer": full_tensor_buffer.contiguous().view(torch.uint8),
        "tensor_buffer_size": total_tensor_size,
        "flat_key_roots": rank_meta[rank]["flat_key_roots"],
        "world_size": world_size,
        "num_replicas": manager.num_replicas,
        "group_size": manager.group_size,
        "target_ranks": target_ranks,
        "source_ranks": source_ranks,
        # Store all ranks' metadata so any healthy main file can serve
        # as metadata source for any failed rank during recovery.
        "all_tensor_infos": {r: rank_meta[r]["tensor_infos"] for r in range(world_size)},
        "all_non_tensor_data": {r: rank_meta[r]["non_tensor_data"] for r in range(world_size)},
        "all_flat_key_roots": {r: rank_meta[r]["flat_key_roots"] for r in range(world_size)},
        "all_tensor_buffer_sizes": {r: rank_meta[r]["tensor_buffer_size"] for r in range(world_size)},
    }

    main_file = checkpoint_dir / f"gemini_replicas_main_rank{rank}.pt"
    torch.save(main_payload, main_file)
    logger.info(
        f"Gemini Replicas legacy save rank {rank}: saved main file {main_file}"
    )

    # Save replica files — include source rank's metadata so recovery is self-contained
    for src_r, recv_buf in receive_buffers.items():
        replica_file = (
            checkpoint_dir / f"gemini_replicas_replica_rank{rank}_from{src_r}.pt"
        )
        replica_payload = {
            "version": 1,
            "format": "gemini_replicas_torch_legacy",
            "rank": rank,
            "source_rank": src_r,
            "buffer_size": recv_buf.numel(),
            "tensor_buffer": recv_buf.detach().clone().contiguous().view(torch.uint8),
            # Include source rank's metadata for recovery
            "source_tensor_infos": rank_meta[src_r]["tensor_infos"],
            "source_non_tensor_data": rank_meta[src_r]["non_tensor_data"],
            "source_flat_key_roots": rank_meta[src_r]["flat_key_roots"],
            "source_tensor_buffer_size": rank_meta[src_r]["tensor_buffer_size"],
        }
        torch.save(replica_payload, replica_file)
        logger.info(
            f"Gemini Replicas legacy save rank {rank}: saved replica file {replica_file}"
        )

    if world_size > 1:
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _load_gemini_replicas_main_payload(
    checkpoint_dir: Path, rank: int, world_size: int
) -> Dict[str, Any]:
    """Load gemini_replicas_main_rank{rank}.pt with all_gather fallback."""
    main_path = checkpoint_dir / f"gemini_replicas_main_rank{rank}.pt"
    local_payload: Optional[Dict[str, Any]] = None
    if main_path.is_file():
        local_payload = torch.load(main_path, map_location="cpu", weights_only=False)

    if world_size <= 1 or not torch.distributed.is_initialized():
        if local_payload is None:
            raise FileNotFoundError(
                f"Gemini Replicas legacy: missing main file {main_path}"
            )
        return local_payload

    gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_payload)

    chosen = gathered[rank]
    if chosen is None:
        raise FileNotFoundError(
            f"Gemini Replicas legacy: gemini_replicas_main_rank{rank}.pt "
            f"missing on all ranks under {checkpoint_dir}"
        )
    return chosen


def _reconstruct_from_main_payload(
    main_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Reconstruct state_dict from a main payload that has tensor_buffer."""
    tb = main_payload["tensor_buffer"]
    buf = tb.detach().contiguous().reshape(-1).view(torch.uint8)
    tensor_infos = main_payload["tensor_infos"]
    tensor_data = extract_tensors_from_continuous_buffer(buf, tensor_infos)
    flat_key_roots = _infer_flat_key_roots(main_payload)
    decomposed = DecomposedStateDict(
        non_tensor_data=main_payload["non_tensor_data"],
        tensor_infos=tensor_infos,
        tensor_data=tensor_data,
        flat_key_roots=flat_key_roots,
    )
    return reconstruct_state_dict(decomposed)


def _run_hardware_recovery(
    manager: GeminiReplicasManager,
    checkpoint_dir: Path,
    rank: int,
    world_size: int,
    main_payload: Dict[str, Any],
) -> torch.Tensor:
    """Hardware failure recovery for Gemini Replicas legacy.

    Protocol:
      Phase 1 — Health check: exchange who still has their main file.
      Phase 2 — Role assignment (per group):
          For each failed rank f, pick the first healthy rank in the group
          that holds a replica of f's data.  That rank becomes the sender for f.
      Phase 3 — Data transfer: senders load replica .pt files and use
          torch.distributed.send to push data to the failed ranks (gloo backend).
      Phase 4 — Reconstruct: the recovering rank returns the recovered buffer.
    """
    gloo_group = get_or_create_global_gloo_group()

    # Build per-rank group membership (must match _calculate_target_ranks layout)
    rank_to_group: Dict[int, List[int]] = {}
    for r in range(world_size):
        if r not in rank_to_group:
            members = manager.get_group_members(r, world_size)
            for m in members:
                rank_to_group[m] = members

    my_group = rank_to_group[rank]

    # ---- Phase 1: health check ----
    main_file_exists = (checkpoint_dir / f"gemini_replicas_main_rank{rank}.pt").is_file()
    health_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(health_list, main_file_exists)
    healthy = {r for r, ok in enumerate(health_list) if ok}
    failed = {r for r in range(world_size) if r not in healthy}

    logger.info(
        f"Gemini Replicas recovery rank {rank}: group={my_group}, "
        f"healthy={sorted(healthy)}, failed={sorted(failed)}"
    )

    if rank in healthy:
        # ---- Phase 2: role assignment (sender side) ----
        assignments: Dict[int, int] = {}  # failed_rank -> sender_rank

        for f in sorted(failed):
            f_group = set(rank_to_group.get(f, []))
            # Find a healthy rank in the same group that holds f's replica
            sender = None
            for candidate in sorted(healthy):
                if candidate not in f_group:
                    continue
                replica_path = (
                    checkpoint_dir
                    / f"gemini_replicas_replica_rank{candidate}_from{f}.pt"
                )
                if replica_path.is_file():
                    sender = candidate
                    break
            if sender is not None:
                assignments[f] = sender
                logger.info(
                    f"Gemini Replicas recovery: failed rank {f} → sender {sender}"
                )
            else:
                logger.warning(
                    f"Gemini Replicas recovery: no replica found for failed rank {f}"
                )

        my_assignments = [
            (f, s) for f, s in assignments.items() if s == rank
        ]
        all_assignments: List[List[Tuple[int, int]]] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_assignments, my_assignments)

        global_assignments: Dict[int, int] = {}
        for r_assign in all_assignments:
            if r_assign:
                for f, s in r_assign:
                    global_assignments[f] = s

        # ---- Phase 3: send replica data ----
        for f, s in global_assignments.items():
            if s != rank:
                continue
            replica_path = (
                checkpoint_dir / f"gemini_replicas_replica_rank{rank}_from{f}.pt"
            )
            logger.info(
                f"Gemini Replicas recovery rank {rank}: loading replica for rank {f} "
                f"from {replica_path}"
            )
            replica_payload = torch.load(
                replica_path, map_location="cpu", weights_only=False
            )
            replica_buffer = replica_payload["tensor_buffer"]
            buf = (
                replica_buffer.detach().contiguous().reshape(-1).view(torch.uint8)
            )
            buf_size = torch.tensor([buf.numel()], dtype=torch.long)

            torch.distributed.send(buf_size, dst=f, group=gloo_group)
            torch.distributed.send(buf, dst=f, group=gloo_group)
            logger.info(
                f"Gemini Replicas recovery rank {rank}: sent "
                f"{buf.numel() / (1024**2):.2f} MB to rank {f}"
            )

    else:
        # ---- Phase 2: role assignment (receiver side) ----
        my_assignments: List[Tuple[int, int]] = []
        all_assignments = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_assignments, my_assignments)

        global_assignments: Dict[int, int] = {}
        for r_assign in all_assignments:
            if r_assign:
                for f, s in r_assign:
                    global_assignments[f] = s

        sender = global_assignments.get(rank)
        if sender is None:
            raise RuntimeError(
                f"Gemini Replicas recovery rank {rank}: no sender assigned — "
                f"not enough replicas to recover. global_assignments={global_assignments}"
            )

        # ---- Phase 3: receive replica data ----
        buf_size_tensor = torch.empty(1, dtype=torch.long)
        torch.distributed.recv(buf_size_tensor, src=sender, group=gloo_group)
        actual_size = int(buf_size_tensor[0].item())

        recovered_buffer = torch.empty(actual_size, dtype=torch.uint8)
        torch.distributed.recv(recovered_buffer, src=sender, group=gloo_group)

        logger.info(
            f"Gemini Replicas recovery rank {rank}: received "
            f"{actual_size / (1024**2):.2f} MB from sender rank {sender}"
        )
        return recovered_buffer

    return torch.zeros(0, dtype=torch.uint8)


def load_gemini_replicas_legacy_checkpoint(
    checkpoint_name: str,
) -> Dict[str, Any]:
    """Load Gemini Replicas torch legacy checkpoint.

    Software failure (all ranks alive, main file present on every rank):
      Each rank loads its main .pt file and reconstructs the state_dict directly.

    Hardware failure (one or more ranks' main files missing):
      Uses the replica files saved by other ranks during save.  For each failed
      rank, a healthy sender in the same group that holds a replica loads it from
      disk and pushes it to the failed rank via torch.distributed.send/recv.
    """
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )

    main_payload = _load_gemini_replicas_main_payload(checkpoint_dir, rank, world_size)

    from megatron.training import get_args

    args = get_args()
    if not getattr(args, "use_gemini_replicas", False):
        logger.warning(
            "Gemini Replicas legacy load: args.use_gemini_replicas is False; "
            "enabling for native module init"
        )
        args.use_gemini_replicas = True

    manager = GeminiReplicasManager()
    # Init C++ native only when we need it (save-time data exchange).
    # During load, init is needed only to get group_size/num_replicas for
    # the recovery topology calculation — the actual recovery transfer uses
    # torch.distributed, not C++ native.
    manager.init_gemini_replicas_if_enabled()

    needs_recovery = not isinstance(main_payload.get("tensor_buffer"), torch.Tensor)

    if needs_recovery:
        logger.info(
            f"Gemini Replicas legacy load rank {rank}: main payload missing "
            f"tensor_buffer, entering hardware recovery"
        )
        recovered_buffer = _run_hardware_recovery(
            manager, checkpoint_dir, rank, world_size, main_payload
        )
        main_payload["tensor_buffer"] = recovered_buffer.contiguous().view(
            torch.uint8
        )

    state_dict = _reconstruct_from_main_payload(main_payload)

    if manager._gemini_replicas_native is not None:
        logger.info(
            f"Gemini Replicas legacy load: cleaning up native module (rank {rank})"
        )
        manager.cleanup()

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return state_dict
