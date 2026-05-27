# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""LEGACY checkpoint path for Gemini Replicas (multi-replica round-robin).
Mirrors ecnaive_legacy.py: decompose state dict, exchange via C++ native,
save/load .pt files with torch.save / torch.load.
"""

import pickle
import struct
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
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
    flatten_optimizer_fp32_params,
    reconstruct_state_dict,
    unflatten_optimizer_fp32_params,
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


_BUILD_GLOBAL_REGISTRY_CACHE: Dict[tuple, tuple] = {}

def _build_global_registry(
    local_metadata: List[TensorMetadata],
    local_non_tensor: Dict[str, Any],
) -> Tuple[Dict[int, List[TensorMetadata]], Dict[int, Dict[str, Any]]]:
    if not torch.distributed.is_initialized():
        return {0: local_metadata}, {0: local_non_tensor}

    world_size = torch.distributed.get_world_size()
    total_bytes = sum(m.size_bytes for m in local_metadata)
    cache_key = (world_size, len(local_metadata), total_bytes)
    if cache_key in _BUILD_GLOBAL_REGISTRY_CACHE:
        return _BUILD_GLOBAL_REGISTRY_CACHE[cache_key]

    gathered_meta: List[Any] = [None for _ in range(world_size)]
    gathered_non_tensor: List[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_meta, local_metadata)
    torch.distributed.all_gather_object(gathered_non_tensor, local_non_tensor)
    rank_metadata = {r: gathered_meta[r] for r in range(world_size)}
    rank_non_tensor = {r: gathered_non_tensor[r] for r in range(world_size)}
    _BUILD_GLOBAL_REGISTRY_CACHE[cache_key] = (rank_metadata, rank_non_tensor)
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
        result = reconstruct_state_dict(decomposed)
    else:
        decomposed = DecomposedStateDict(
            non_tensor_data=main_payload["non_tensor_data"],
            tensor_infos=[],
            tensor_data=[],
            flat_key_roots=flat_key_roots,
        )
        result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

# Meta exchange cache: tensor_infos (shapes, keys, dtypes) and slimmed
# non_tensor_data are identical across iterations for a fixed model.
# Cache the first exchange to avoid repeated NCCL all_gather_object calls.
_cached_rank_metadata: Optional[Dict[int, List[TensorMetadata]]] = None
_cached_rank_non_tensor: Optional[Dict[int, Dict[str, Any]]] = None
_cached_rank_flat_key_roots: Optional[Dict[int, list]] = None
_cached_rank_tensor_infos: Optional[Dict[int, list]] = None


def save_gemini_replicas_legacy_checkpoint(
    state_dict: Dict[str, Any], checkpoint_name: str
) -> None:
    global _cached_rank_metadata, _cached_rank_non_tensor
    global _cached_rank_flat_key_roots, _cached_rank_tensor_infos

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

    flatten_optimizer_fp32_params(state_dict)
    t0 = time.time()
    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    logger.info(f"GEMINI save timing: decompose {time.time()-t0:.3f}s")

    start_time = t0 = time.time()
    safety_margin = max(int(total_tensor_size * 0.01), 1024 * 1024)
    manager.allocate_preallocated_buffer(total_tensor_size + safety_margin)
    tensor_buffer = manager.preallocated_cpu_buffer

    offset = 0
    local_tensor_metadata: List[TensorMetadata] = []
    local_tensor_infos: List[Dict[str, Any]] = []  # for replica file metadata
    for i, (info, tensor) in enumerate(zip(decomposed.tensor_infos, decomposed.tensor_data)):
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
        local_tensor_infos.append({
            "key": info.key,
            "shape": list(info.shape),
            "dtype": str(info.dtype),
            "offset": info.offset,
            "size_bytes": info.size_bytes,
        })
        offset += tensor_bytes
        decomposed.tensor_data[i] = None  # free GPU tensor ref immediately

    del decomposed.tensor_data  # drop remaining refs
    logger.info(f"GEMINI save timing: D2H+copy {time.time()-t0:.3f}s")

    t0 = time.time()
    if manager.use_rdma:
        manager.register_buffer(tensor_buffer)
    logger.info(f"GEMINI save timing: RDMA reg tensor {time.time()-t0:.3f}s")

    t0 = time.time()
    # ===== Metadata exchange via all_gather_object on NCCL (aligned with ecnaive/eccheck) =====
    # Tensor shapes, keys, and dtypes are identical across iterations for a fixed model.
    # Cache the results from the first exchange to skip expensive NCCL all_gather_object
    # on subsequent iterations (~3s → 0s for 32 ranks, 3GB models).
    if _cached_rank_tensor_infos is None:
        _NTD_SKIP_PREFIXES = ("optimizer", "rng_state", "rerun_state_machine", "args")
        slim_ntd = {
            k: v for k, v in decomposed.non_tensor_data.items()
            if not k.startswith(_NTD_SKIP_PREFIXES)
        }
        rank_metadata, rank_non_tensor = _build_global_registry(local_tensor_metadata, slim_ntd)

        my_flat_key_roots = list(decomposed.flat_key_roots) if decomposed.flat_key_roots else []
        all_meta: List[Any] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_meta, (my_flat_key_roots, local_tensor_infos))
        rank_flat_key_roots = {r: all_meta[r][0] for r in range(world_size)}
        rank_tensor_infos = {r: all_meta[r][1] for r in range(world_size)}

        _cached_rank_metadata = rank_metadata
        _cached_rank_non_tensor = rank_non_tensor
        _cached_rank_flat_key_roots = rank_flat_key_roots
        _cached_rank_tensor_infos = rank_tensor_infos
        logger.info(f"GEMINI save timing: meta exchange (first) {time.time()-t0:.3f}s")
    else:
        rank_metadata = _cached_rank_metadata
        rank_non_tensor = _cached_rank_non_tensor
        rank_flat_key_roots = _cached_rank_flat_key_roots
        rank_tensor_infos = _cached_rank_tensor_infos
        logger.info(f"GEMINI save timing: meta exchange (cached) {time.time()-t0:.3f}s")

    # Compute buffer sizes from metadata (replaces separate gloo size exchange)
    rank_sizes = {
        r: sum(m.size_bytes for m in rank_metadata[r])
        for r in range(world_size)
    }

    send_buffer_size = total_tensor_size

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

    t0 = time.time()
    # Allocate or reuse cached receive buffers
    receive_buffers: Dict[int, torch.Tensor] = {}
    for src_r in source_ranks:
        src_size = rank_sizes[src_r]
        recv_buf = manager.allocate_recv_buffer(src_r, src_size)
        if manager.use_rdma:
            manager.register_buffer(recv_buf)
        receive_buffers[src_r] = recv_buf

    torch.distributed.barrier()
    logger.info(f"GEMINI save timing: recv buf alloc + barrier {time.time()-t0:.3f}s")

    t0 = time.time()
    # Submit to C++ native workers (non-blocking, like ecnaive).
    # Workers were started by manager._init_gemini_replicas_native() right
    # after finalize_connections, so they are already waiting on CVs.
    native = manager._gemini_replicas_native
    native.reset_exchange_state()

    send_addr = tensor_buffer.data_ptr()
    native.submit_send_buffer(send_addr, send_buffer_size)

    for src_r, recv_buf in receive_buffers.items():
        native.submit_recv_buffer(src_r, recv_buf.data_ptr(), recv_buf.numel())

    logger.info(
        f"Gemini Replicas legacy save rank {rank}: executing C++ exchange "
        f"(send to {len(target_ranks) - 1} targets, recv from {len(source_ranks)} sources)..."
    )
    native.wait_for_exchange_completion()
    logger.info(f"Gemini Replicas legacy save rank {rank}: C++ exchange done ({time.time()-t0:.3f}s)")

    # Build rank_meta from pre-exchanged data (meta exchange already done before C++ transfer).
    # Format is compatible with the file writing code below.
    rank_meta = {}
    for r in range(world_size):
        rank_meta[r] = {
            "tensor_infos": rank_tensor_infos.get(r, []),
            "non_tensor_data": rank_non_tensor.get(r, {}),
            "tensor_buffer_size": rank_sizes.get(r, 0),
            "flat_key_roots": rank_flat_key_roots.get(r, []),
        }

    # Save .pt files
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from megatron.training.legacy_io_utils import write_main_prepared, write_block_prepared, MAGIC_GEMINI, MAGIC_GEMINI_REPLICA

    # ---- Pre-serialize main file ----
    main_meta1 = pickle.dumps(decomposed.non_tensor_data)
    main_meta2 = pickle.dumps(rank_meta[rank]["tensor_infos"])
    main_extra = pickle.dumps({
        "version": 1, "format": "gemini_replicas_torch_legacy", "rank": rank,
        "tensor_buffer_size": total_tensor_size,
        "flat_key_roots": rank_meta[rank]["flat_key_roots"],
        "world_size": world_size, "num_replicas": manager.num_replicas,
        "group_size": manager.group_size,
        "target_ranks": target_ranks, "source_ranks": source_ranks,
        "all_tensor_infos": {r: rank_meta[r]["tensor_infos"] for r in range(world_size)},
        "all_flat_key_roots": {r: rank_meta[r]["flat_key_roots"] for r in range(world_size)},
        "all_tensor_buffer_sizes": {r: rank_meta[r]["tensor_buffer_size"] for r in range(world_size)},
    })
    buf = tensor_buffer[:total_tensor_size]
    if not buf.is_contiguous():
        buf = buf.contiguous()
    main_mv = memoryview(buf.numpy())

    # ---- Pre-serialize replica files ----
    replica_tasks = []
    for src_r, recv_buf in receive_buffers.items():
        replica_file = checkpoint_dir / f"gemini_replicas_replica_rank{rank}_from{src_r}.pt"
        meta_bytes = pickle.dumps({
            "version": 1, "format": "gemini_replicas_torch_legacy",
            "rank": rank, "source_rank": src_r, "buffer_size": recv_buf.numel(),
            "source_tensor_infos": rank_meta[src_r]["tensor_infos"],
            "source_non_tensor_data": rank_meta[src_r]["non_tensor_data"],
            "source_flat_key_roots": rank_meta[src_r]["flat_key_roots"],
            "source_tensor_buffer_size": rank_meta[src_r]["tensor_buffer_size"],
        })
        b = recv_buf[: recv_buf.numel()]
        if not b.is_contiguous():
            b = b.contiguous()
        replica_tasks.append((str(replica_file), meta_bytes, memoryview(b.numpy())))

    torch.distributed.barrier()
    logger.info(f"GEMINI REPLICAS legacy save: done in {time.time() - start_time:.2f}s")
    

    # ---- Parallel writes ----
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1 + len(replica_tasks)) as ex:
        main_file = checkpoint_dir / f"gemini_replicas_main_rank{rank}.pt"
        futs = [ex.submit(write_main_prepared, str(main_file), MAGIC_GEMINI,
                          main_meta1, main_meta2, main_extra, main_mv, total_tensor_size)]
        for rep_path, rep_meta, rep_mv in replica_tasks:
            futs.append(ex.submit(_write_replica_file, rep_path, MAGIC_GEMINI_REPLICA,
                                  rep_meta, rep_mv))
        for f in futs:
            f.result()


def _write_replica_file(path, magic, meta_bytes, mv):
    import struct as _struct
    with open(path, "wb") as f:
        f.write(_struct.pack("<4sQ", magic, len(meta_bytes)))
        f.write(meta_bytes)
        f.write(mv)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _dict_to_tensor_info(info: Dict) -> Any:
    """Convert a serialized tensor info dict back to an object with the
    attributes expected by extract_tensors_from_continuous_buffer and
    reconstruct_state_dict (.offset, .size_bytes, .dtype, .shape, .key)."""
    from types import SimpleNamespace

    dtype_str = info.get("dtype", "torch.float32")
    dtype = getattr(torch, dtype_str.split(".")[-1]) if "." in dtype_str else torch.float32
    return SimpleNamespace(
        key=info.get("key", ""),
        offset=info["offset"],
        size_bytes=info["size_bytes"],
        dtype=dtype,
        shape=tuple(info["shape"]),
    )


def _reconstruct_from_payload(
    tensor_infos: List[Dict],
    non_tensor_data: Dict[str, Any],
    tensor_buffer: torch.Tensor,
    flat_key_roots: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Reconstruct state_dict from metadata + tensor buffer."""
    tb = tensor_buffer.detach().contiguous().reshape(-1).view(torch.uint8)
    ti_objs = [_dict_to_tensor_info(info) for info in tensor_infos]
    tensor_data = extract_tensors_from_continuous_buffer(tb, ti_objs)
    decomposed = DecomposedStateDict(
        non_tensor_data=non_tensor_data,
        tensor_infos=ti_objs,
        tensor_data=tensor_data,
        flat_key_roots=flat_key_roots or set(),
    )
    result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


def _collect_metadata_for_failed_rank(
    checkpoint_dir: Path,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    """Gather metadata (tensor_infos, non_tensor_data) for a failed rank.

    Every healthy rank loads its own main file which contains all-to-all
    metadata tables.  Failed ranks contribute None.  After all_gather,
    each failed rank picks its own metadata from any healthy source.
    """
    main_path = checkpoint_dir / f"gemini_replicas_main_rank{rank}.pt"

    my_all_meta: Optional[Dict[str, Any]] = None
    if main_path.is_file():
        from megatron.training.legacy_io_utils import is_raw_format, read_raw_checkpoint, MAGIC_GEMINI
        own = (read_raw_checkpoint(str(main_path), MAGIC_GEMINI)
               if is_raw_format(str(main_path), MAGIC_GEMINI)
               else torch.load(main_path, map_location="cpu", weights_only=False))
        my_all_meta = {
            "all_tensor_infos": own.get("all_tensor_infos", {}),
            "all_non_tensor_data": own.get("all_non_tensor_data", {}),
            "all_flat_key_roots": own.get("all_flat_key_roots", {}),
            "all_tensor_buffer_sizes": own.get("all_tensor_buffer_sizes", {}),
        }

    gathered: List[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, my_all_meta)

    # Find metadata for THIS specific rank from any healthy source
    for meta in gathered:
        if meta is None:
            continue
        ti = meta["all_tensor_infos"].get(rank)
        if ti is not None:
            return {
                "tensor_infos": ti,
                "non_tensor_data": meta["all_non_tensor_data"].get(rank, {}),
                "flat_key_roots": meta["all_flat_key_roots"].get(rank, []),
                "tensor_buffer_size": meta["all_tensor_buffer_sizes"].get(rank, 0),
            }

    raise RuntimeError(
        f"Gemini Replicas legacy load rank {rank}: "
        f"no metadata available for recovery. "
        f"At least one healthy rank must have a main file."
    )


# ---------------------------------------------------------------------------
# Replica file loading helpers (used by _run_hardware_recovery)
# ---------------------------------------------------------------------------

def _load_replica_metadata(replica_path: Path) -> Optional[Dict[str, Any]]:
    """Load only the metadata portion of a replica file (no tensor data).

    Used in Phase 2 to compute combined transfer size without loading the
    full multi-GB tensor buffer into memory.
    """
    from megatron.training.legacy_io_utils import is_raw_format, MAGIC_GEMINI_REPLICA
    try:
        if is_raw_format(str(replica_path), MAGIC_GEMINI_REPLICA):
            with open(str(replica_path), "rb") as _f:
                _f.read(4)  # skip magic
                meta_len = struct.unpack("<Q", _f.read(8))[0]
                return pickle.loads(_f.read(meta_len))
        else:
            rp = torch.load(replica_path, map_location="cpu", weights_only=False)
            return {
                "source_tensor_infos": rp.get("source_tensor_infos", []),
                "source_non_tensor_data": rp.get("source_non_tensor_data", {}),
                "source_flat_key_roots": rp.get("source_flat_key_roots", []),
                "source_tensor_buffer_size": rp.get("source_tensor_buffer_size", 0),
            }
    except Exception as e:
        logger.warning(
            f"Gemini Replicas: failed to load replica metadata from "
            f"{replica_path}: {e}"
        )
        return None


def _load_replica_full(replica_path: Path) -> Dict[str, Any]:
    """Load a replica file including its tensor buffer."""
    from megatron.training.legacy_io_utils import is_raw_format, MAGIC_GEMINI_REPLICA
    if is_raw_format(str(replica_path), MAGIC_GEMINI_REPLICA):
        with open(str(replica_path), "rb") as _f:
            _f.read(4)  # skip magic
            meta_len = struct.unpack("<Q", _f.read(8))[0]
            rp = pickle.loads(_f.read(meta_len))
            rp["tensor_buffer"] = torch.from_numpy(
                np.frombuffer(_f.read(), dtype=np.uint8))
            return rp
    else:
        return torch.load(replica_path, map_location="cpu", weights_only=False)


def _run_hardware_recovery(
    manager: GeminiReplicasManager,
    checkpoint_dir: Path,
    rank: int,
    world_size: int,
    failed_override: Optional[Set[int]] = None,
) -> torch.Tensor:
    """Hardware failure recovery for Gemini Replicas legacy.

    Protocol:
      Phase 1 — Health check: exchange who still has their main file.
      Phase 2 — Role assignment + size exchange:
          For each failed rank f, pick the first healthy rank in the group
          that holds a replica of f's data.  Senders compute combined_size
          (header + metadata + tensor) and all ranks exchange both assignments
          and sizes via all_gather_object.
      Phase 3 — Data transfer over C++ ASIO/RDMA connections:
          Senders pack [header(8B)][metadata(pickled)][tensor_data] into a
          single combined buffer and call manager.send_to_rank(f, combined).
          Receivers call manager.recv_from_rank(sender, combined_size), then
          parse the header to split metadata from tensor data.
    """

    # Build per-rank group membership
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

    if failed_override:
        healthy = {r for r in range(world_size) if r not in failed_override}
        failed = failed_override
    else:
        healthy = {r for r, ok in enumerate(health_list) if ok}
        failed = {r for r in range(world_size) if r not in healthy}

    logger.info(
        f"Gemini Replicas recovery rank {rank}: group={my_group}, "
        f"healthy={sorted(healthy)}, failed={sorted(failed)}"
    )

    if rank in healthy:
        # ---- Phase 2: role assignment (sender side) ----
        assignments: Dict[int, int] = {}

        for f in sorted(failed):
            f_group = set(rank_to_group.get(f, []))
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

        # Compute combined sizes for the assignments where this rank is sender
        my_assignments: List[Tuple[int, int]] = []
        combined_sizes: Dict[int, int] = {}  # failed_rank → combined_size
        for f, s in assignments.items():
            if s == rank:
                my_assignments.append((f, s))
                # Load just enough to compute sizes (don't load full tensor yet)
                replica_path = (
                    checkpoint_dir / f"gemini_replicas_replica_rank{rank}_from{f}.pt"
                )
                rp = _load_replica_metadata(replica_path)
                if rp is not None:
                    tensor_buf_size = rp.get("source_tensor_buffer_size", 0)
                    # Build metadata dict to measure its pickled size
                    meta = {
                        "tensor_infos": rp["source_tensor_infos"],
                        "non_tensor_data": rp["source_non_tensor_data"],
                        "flat_key_roots": rp["source_flat_key_roots"],
                        "tensor_buffer_size": tensor_buf_size,
                    }
                    meta_bytes = pickle.dumps(meta)
                    # combined = [header:8B][meta_bytes][tensor_data]
                    combined_sizes[f] = 8 + len(meta_bytes) + tensor_buf_size

        # Exchange assignments
        all_assignments: List[List[Tuple[int, int]]] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_assignments, my_assignments)

        global_assignments: Dict[int, int] = {}
        for r_assign in all_assignments:
            if r_assign:
                for f, s in r_assign:
                    global_assignments[f] = s

        # Exchange combined sizes so receivers know how much to recv
        all_combined_sizes: List[Dict[int, int]] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_combined_sizes, combined_sizes)

        global_combined_sizes: Dict[int, int] = {}
        for cs in all_combined_sizes:
            if cs:
                global_combined_sizes.update(cs)

        # ---- Phase 3: send via C++ ASIO/RDMA ----
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
            rp = _load_replica_full(replica_path)
            meta = {
                "tensor_infos": rp["source_tensor_infos"],
                "non_tensor_data": rp["source_non_tensor_data"],
                "flat_key_roots": rp["source_flat_key_roots"],
                "tensor_buffer_size": rp["source_tensor_buffer_size"],
            }
            meta_bytes = pickle.dumps(meta)
            header = struct.pack("<Q", len(meta_bytes))

            replica_buffer = rp["tensor_buffer"]
            buf = replica_buffer.detach().contiguous().reshape(-1).view(torch.uint8)

            # Pack: [header:8B][meta_bytes][tensor_data]
            combined = torch.cat([
                torch.frombuffer(bytearray(header), dtype=torch.uint8),
                torch.frombuffer(bytearray(meta_bytes), dtype=torch.uint8),
                buf,
            ])

            logger.info(
                f"Gemini Replicas recovery rank {rank}: sending combined buffer "
                f"({combined.numel() / (1024**2):.2f} MB) to rank {f} via "
                f"{'RDMA' if manager.use_rdma else 'ASIO'}"
            )
            manager.send_to_rank(f, combined)
            logger.info(
                f"Gemini Replicas recovery rank {rank}: sent to rank {f}"
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
                f"Gemini Replicas recovery rank {rank}: no sender assigned. "
                f"global_assignments={global_assignments}"
            )

        # Exchange combined sizes
        combined_sizes: Dict[int, int] = {}
        all_combined_sizes: List[Dict[int, int]] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_combined_sizes, combined_sizes)

        global_combined_sizes: Dict[int, int] = {}
        for cs in all_combined_sizes:
            if cs:
                global_combined_sizes.update(cs)

        combined_size = global_combined_sizes.get(rank)
        if combined_size is None:
            raise RuntimeError(
                f"Gemini Replicas recovery rank {rank}: no combined size info. "
                f"global_combined_sizes={global_combined_sizes}"
            )

        # ---- Phase 3: receive via C++ ASIO/RDMA ----
        logger.info(
            f"Gemini Replicas recovery rank {rank}: receiving "
            f"{combined_size / (1024**2):.2f} MB from sender rank {sender} via "
            f"{'RDMA' if manager.use_rdma else 'ASIO'}"
        )
        combined = manager.recv_from_rank(sender, combined_size)

        # Parse combined buffer: [header:8B][meta_bytes][tensor_data]
        header = combined[:8].numpy().tobytes()
        meta_size = struct.unpack("<Q", header)[0]
        meta_bytes = combined[8:8 + meta_size].numpy().tobytes()
        meta = pickle.loads(meta_bytes)

        # Store metadata for reconstruction
        _recovery_meta[rank] = meta

        recovered_buffer = combined[8 + meta_size:combined_size].clone()
        logger.info(
            f"Gemini Replicas recovery rank {rank}: received metadata "
            f"({meta_size} B) + tensor data "
            f"({recovered_buffer.numel() / (1024**2):.2f} MB) from sender {sender}"
        )
        return recovered_buffer

    return torch.zeros(0, dtype=torch.uint8)


# Module-level dict so the receiver side of _run_hardware_recovery can
# hand metadata back to load_gemini_replicas_legacy_checkpoint.
_recovery_meta: Dict[int, Dict[str, Any]] = {}


def load_gemini_replicas_legacy_checkpoint(
    checkpoint_name: str,
) -> Dict[str, Any]:
    """Load Gemini Replicas torch legacy checkpoint.

    Normal load (main file present):
      Each rank loads its own main .pt directly.  Replicas are not touched.

    Hardware recovery (main file missing, or --gemini-replicas-recovery-rank):
      Failed ranks recover from other ranks' replica files.  After recovery,
      the main .pt file is regenerated.
    """
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )

    from megatron.training import get_args

    args = get_args()
    if not getattr(args, "use_gemini_replicas", False):
        logger.warning(
            "Gemini Replicas legacy load: args.use_gemini_replicas is False; "
            "enabling for native module init"
        )
        args.use_gemini_replicas = True

    manager = GeminiReplicasManager()
    manager.init_gemini_replicas_if_enabled()

    # ---- Determine which ranks need recovery ----
    recovery_rank_str = getattr(args, "gemini_replicas_recovery_rank", None)
    main_path = checkpoint_dir / f"gemini_replicas_main_rank{rank}.pt"
    main_file_exists = main_path.is_file()
    health_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(health_list, main_file_exists)

    if recovery_rank_str:
        # Explicit ranks treated as failed (for testing)
        recovery_ranks = {int(x.strip()) for x in recovery_rank_str.split(",")}
        is_failed = rank in recovery_ranks
        logger.info(
            f"Gemini Replicas load rank {rank}: "
            f"recovery_ranks={recovery_ranks}, is_failed={is_failed}"
        )
    else:
        # Auto-detect from file existence
        is_failed = not main_file_exists

    main_payload: Optional[Dict[str, Any]] = None
    if main_file_exists:
        from megatron.training.legacy_io_utils import is_raw_format, read_raw_checkpoint, MAGIC_GEMINI
        main_payload = (read_raw_checkpoint(str(main_path), MAGIC_GEMINI)
                        if is_raw_format(str(main_path), MAGIC_GEMINI)
                        else torch.load(main_path, map_location="cpu", weights_only=False))

    # ---- Timing collection (excl disk IO) ----
    # network_encode: C++ ASIO/RDMA send/recv + block assembly (HW only; SW=0)
    # rebuild_sd: extract + reconstruct + unflatten
    # total: network_encode + rebuild_sd
    _t: Dict[str, float] = {}

    # ---- Software failure path ----
    sw_failure = bool(getattr(args, "use_gemini_replicas_software_failure", False))
    if sw_failure:
        if main_payload is None:
            raise FileNotFoundError(
                f"Gemini Replicas software failure: rank {rank} main file not found "
                f"at {main_path}. In software failure mode, main.pt must exist on disk."
            )
        _t['network_encode'] = 0.0
        _t0 = time.time()
        state_dict = _reconstruct_from_payload(
            tensor_infos=main_payload["tensor_infos"],
            non_tensor_data=main_payload["non_tensor_data"],
            tensor_buffer=main_payload["tensor_buffer"],
            flat_key_roots=_infer_flat_key_roots(main_payload),
        )
        _t['rebuild_sd'] = time.time() - _t0
        _t['total'] = _t['network_encode'] + _t['rebuild_sd']
        logger.info(
            "GEMINI REPLICAS legacy load timing (SW): "
            "total=%(total).2fs network_encode=%(network_encode).2fs "
            "rebuild_sd=%(rebuild_sd).2fs", _t
        )
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return state_dict

    if not is_failed and all(health_list):
        # ---- Normal load ----
        _t['network_encode'] = 0.0
        _t0 = time.time()
        state_dict = _reconstruct_from_payload(
            tensor_infos=main_payload["tensor_infos"],
            non_tensor_data=main_payload["non_tensor_data"],
            tensor_buffer=main_payload["tensor_buffer"],
            flat_key_roots=_infer_flat_key_roots(main_payload),
        )
        _t['rebuild_sd'] = time.time() - _t0
        _t['total'] = _t['network_encode'] + _t['rebuild_sd']
        logger.info(
            "GEMINI REPLICAS legacy load timing (normal): "
            "total=%(total).2fs network_encode=%(network_encode).2fs "
            "rebuild_sd=%(rebuild_sd).2fs", _t
        )
    else:
        # ---- Hardware recovery ----
        logger.info(
            f"Gemini Replicas legacy load rank {rank}: "
            f"{sum(health_list)}/{world_size} healthy, entering recovery"
        )

        if recovery_rank_str:
            failed_override = recovery_ranks
        else:
            failed_override = None

        # Setup (not timed): metadata collection
        meta = _collect_metadata_for_failed_rank(checkpoint_dir, rank, world_size)

        # === timing: network/encode (ASIO/RDMA send/recv) ===
        _t0 = time.time()
        recovered_buffer = _run_hardware_recovery(
            manager, checkpoint_dir, rank, world_size,
            failed_override=failed_override,
        )
        _t['network_encode'] = time.time() - _t0

        # Step 3: Reconstruct state dict
        _t0 = time.time()
        if is_failed:
            if rank in _recovery_meta:
                meta.update(_recovery_meta.pop(rank))
            state_dict = _reconstruct_from_payload(
                tensor_infos=meta["tensor_infos"],
                non_tensor_data=meta["non_tensor_data"],
                tensor_buffer=recovered_buffer,
                flat_key_roots=set(meta.get("flat_key_roots", [])),
            )
            # Regenerate main file
            from megatron.training.legacy_io_utils import write_raw_checkpoint, MAGIC_GEMINI
            regen_tensor = recovered_buffer.contiguous().view(torch.uint8)
            write_raw_checkpoint(
                str(main_path), MAGIC_GEMINI,
                meta["non_tensor_data"], meta["tensor_infos"],
                regen_tensor, meta["tensor_buffer_size"],
                version=1, format="gemini_replicas_torch_legacy",
                rank=rank, world_size=world_size,
                num_replicas=manager.num_replicas,
                group_size=manager.group_size,
                tensor_buffer_size=meta["tensor_buffer_size"],
                flat_key_roots=meta.get("flat_key_roots", []),
            )
            logger.info(
                f"Gemini Replicas hardware recovery rank {rank}: "
                f"regenerated main file {main_path}"
            )
        else:
            state_dict = _reconstruct_from_payload(
                tensor_infos=main_payload["tensor_infos"],
                non_tensor_data=main_payload["non_tensor_data"],
                tensor_buffer=main_payload["tensor_buffer"],
                flat_key_roots=_infer_flat_key_roots(main_payload),
            )
        _t['rebuild_sd'] = time.time() - _t0
        _t['total'] = _t['network_encode'] + _t['rebuild_sd']

        logger.info(
            "GEMINI REPLICAS legacy load timing (HW): "
            "total=%(total).2fs network_encode=%(network_encode).2fs "
            "rebuild_sd=%(rebuild_sd).2fs", _t
        )

    if manager._gemini_replicas_native is not None:
        logger.info(
            f"Gemini Replicas legacy load: cleaning up native module (rank {rank})"
        )
        manager.cleanup()
        manager._gemini_replicas_native = None

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return state_dict
