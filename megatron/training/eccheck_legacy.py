# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""LEGACY checkpoint path for ECCHECK (XOR-based erasure coding with 4-rank groups).

Saves/loads via torch.save / torch.load with .pt files (same pattern as
eclatin_legacy.py and ecnaive_legacy.py), reusing the shared ECCHECKManager
singleton and its C++ native module.
"""

import ctypes
import os
import queue
from collections import deque
import struct
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from megatron.core.dist_checkpointing.strategies.eccheck_manager import ECCHECKManager
from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
    allocate_hugepage_slices,
    allocate_hugepage_tensor,
)
from megatron.core.dist_checkpointing.strategies.state_dict_decomposer import (
    DecomposedStateDict,
    GlobalMetadataRegistry,
    TensorMetadata,
    decompose_state_dict,
    decompose_state_dict_for_save,
    extract_tensors_from_continuous_buffer,
    reconstruct_state_dict,
    unflatten_optimizer_fp32_params,
)

logger = getLogger(__name__)


def _eccheck_hw2_recv_ring_depth() -> int:
    raw_depth = os.getenv("ECCHECK_HW2_RECV_RING_DEPTH", "12")
    try:
        depth = int(raw_depth)
    except ValueError as exc:
        raise ValueError(
            "ECCHECK_HW2_RECV_RING_DEPTH must be an integer in [1, 64], "
            f"got {raw_depth!r}"
        ) from exc
    if depth < 1 or depth > 64:
        raise ValueError(
            "ECCHECK_HW2_RECV_RING_DEPTH must be in [1, 64], "
            f"got {depth}"
        )
    return depth


def _timing_max(value: float) -> float:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return float(value)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def _timing_max_dict(timings: Dict[str, float]) -> Dict[str, float]:
    return {key: _timing_max(value) for key, value in timings.items()}


def _native_ft_timing(native) -> Dict[str, float]:
    if native is None:
        return {"net_s": 0.0, "encode_s": 0.0, "decode_s": 0.0}
    try:
        stats = native.get_ft_timing_stats()
        return {
            "net_s": float(stats.get("net_s", 0.0)),
            "encode_s": float(stats.get("encode_s", 0.0)),
            "decode_s": float(stats.get("decode_s", stats.get("xor_s", 0.0))),
        }
    except AttributeError:
        return {"net_s": 0.0, "encode_s": 0.0, "decode_s": 0.0}


def _timed_barrier() -> float:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0.0
    start = time.time()
    torch.distributed.barrier()
    return time.time() - start

_FORMAT = "eccheck_torch_legacy"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


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
        chunk_type = getattr(info, "chunk_type", "data")
        target_rank = getattr(info, "target_rank", rank)
        source_rank = getattr(info, "source_rank", rank)
        out.append(
            TensorMetadata(
                key=info.key,
                shape=tuple(info.shape),
                dtype=str(info.dtype),
                size_bytes=info.size_bytes,
                global_offset=tuple(info.global_offset) if info.global_offset else tuple(),
                shard_index=info.shard_index if info.shard_index is not None else 0,
                chunk_type=chunk_type,
                target_rank=target_rank,
                source_rank=source_rank,
            )
        )
    return out


def _max_tensor_bytes_from_registry(registry: GlobalMetadataRegistry, world_size: int) -> int:
    max_bytes = 0
    for r in range(world_size):
        rank_metadata = registry.rank_metadata.get(r, [])
        rank_total = sum(meta.size_bytes for meta in rank_metadata)
        if rank_total > max_bytes:
            max_bytes = rank_total
    return max_bytes


def _eccheck_pipeline_transfer_bytes(registry: GlobalMetadataRegistry, world_size: int) -> int:
    """Return the logical EC pipeline length used by recovery transfers."""
    return _max_tensor_bytes_from_registry(registry, world_size)


def _rank_total_bytes(rank_metadata: Dict[int, List[TensorMetadata]], rank: int) -> int:
    return sum(meta.size_bytes for meta in rank_metadata.get(rank, []))


def _rank_data_transfer_bytes(
    registry: GlobalMetadataRegistry, rank: int, world_size: int
) -> int:
    """Return actual data bytes for one rank, without EC padding."""
    if rank < 0 or rank >= world_size:
        return 0
    return _rank_total_bytes(registry.rank_metadata, rank)


def _block_payload_sizes_for_save(
    rank: int,
    world_size: int,
    rank_metadata: Dict[int, List[TensorMetadata]],
    blocks: Dict[str, Any],
) -> Dict[str, int]:
    """Return logical bytes to persist for each ECCHECK side block."""
    pipeline_bytes = int(blocks["pipeline_size"])
    own_actual = _rank_total_bytes(rank_metadata, rank)
    if world_size > 1:
        partner_rank = ECCHECKManager().get_p2p_partner_rank(rank, world_size)
        partner_actual = _rank_total_bytes(rank_metadata, partner_rank)
        rank_in_group = ECCHECKManager._get_rank_in_group(rank, world_size)
    else:
        partner_actual = own_actual
        rank_in_group = 0

    # Even positions store parity locally and partner data remotely; odd positions
    # store data locally and partner parity remotely. Parity must keep pipeline size.
    if rank_in_group % 2 == 0:
        sizes = {"own_buffer": pipeline_bytes, "partner_buffer": partner_actual}
    else:
        sizes = {"own_buffer": own_actual, "partner_buffer": pipeline_bytes}

    block_write_sizes = blocks.get("block_write_sizes", {})
    return {
        name: min(int(size), int(block_write_sizes.get(name, blocks[name].numel())))
        for name, size in sizes.items()
    }


def _infer_flat_key_roots(main_payload: Dict[str, Any]) -> Set[str]:
    if "flat_key_roots" in main_payload:
        return set(main_payload["flat_key_roots"])
    flat_key_roots: Set[str] = set()
    for info in main_payload.get("tensor_infos", []):
        key = info.get("key", "") if isinstance(info, dict) else getattr(info, "key", "")
        first_seg = key.split(".")[0]
        if first_seg == "model" or (
            first_seg.startswith("model")
            and len(first_seg) > 5
            and first_seg[5:].isdigit()
        ):
            flat_key_roots.add(first_seg)
    return flat_key_roots


# ---------------------------------------------------------------------------
# Block allocation (save + load)
# ---------------------------------------------------------------------------

def _allocate_eccheck_blocks_legacy(
    manager: ECCHECKManager,
    rank_metadata: Dict[int, List[TensorMetadata]],
    block_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    own_total_size = sum(meta.size_bytes for meta in rank_metadata.get(rank, []))
    if block_names is None:
        block_names = ["own_buffer", "partner_buffer"]

    if world_size > 1:
        all_sizes = [
            sum(meta.size_bytes for meta in rank_metadata.get(r, []))
            for r in range(world_size)
        ]
        max_total_bytes = max(all_sizes)
    else:
        max_total_bytes = own_total_size

    aligned_size = (
        (max_total_bytes + manager.eccheck_buffer_size - 1)
        // manager.eccheck_buffer_size
    ) * manager.eccheck_buffer_size

    allocated_blocks: Dict[str, torch.Tensor] = {}
    if block_names:
        buffers = manager.allocate_preallocated_blocks(len(block_names), aligned_size)
        allocated_blocks = dict(zip(block_names, buffers))

    blocks = {
        "actual_size": own_total_size,
        "pipeline_size": max_total_bytes,
        "aligned_size": aligned_size,
        "block_names": block_names,
    }
    blocks.update(allocated_blocks)
    return blocks


def _allocate_recovered_buffer(size_bytes: int, pin: bool = True) -> torch.Tensor:
    """Pinned buffer for recovery output and fast H2D."""
    if torch.cuda.is_available() and pin:
        try:
            return torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
        except Exception:
            pass
    return allocate_hugepage_tensor(
        size_bytes,
        fallback_pin_memory=torch.cuda.is_available() and pin,
        touch_pages=True,
    )


# ---------------------------------------------------------------------------
# Encoding pipeline (save)
# ---------------------------------------------------------------------------

def _eccheck_chunk_take(
    src_pos: int,
    pipeline_total_bytes: int,
    buffer_size: int,
    recv_offset_1: int,
    recv_offset_2: int,
    own_offset: int,
    partner_offset: int,
    recv_buffer_thread1: torch.Tensor,
    recv_buffer_thread2: torch.Tensor,
    own_buffer: torch.Tensor,
    partner_buffer: torch.Tensor,
) -> int:
    """Return the byte count for one ECCHECK encode stripe at src_pos."""
    remaining = pipeline_total_bytes - src_pos
    take = min(buffer_size, remaining)

    recv_offset_1_aligned = ((recv_offset_1 + 63) // 64) * 64
    recv_offset_2_aligned = ((recv_offset_2 + 63) // 64) * 64
    recv_rem_1 = recv_buffer_thread1.numel() - recv_offset_1_aligned
    recv_rem_2 = recv_buffer_thread2.numel() - recv_offset_2_aligned
    max_recv_space = min(recv_rem_1, recv_rem_2)
    if take > max_recv_space:
        if max_recv_space < 64:
            return 0
        take = max_recv_space

    own_offset_aligned = ((own_offset + 63) // 64) * 64
    partner_offset_aligned = ((partner_offset + 63) // 64) * 64
    p2p_rem_own = own_buffer.numel() - own_offset_aligned
    p2p_rem_partner = partner_buffer.numel() - partner_offset_aligned
    max_p2p_space = min(p2p_rem_own, p2p_rem_partner)
    if take > max_p2p_space:
        if max_p2p_space < 64:
            return 0
        take = max_p2p_space
    return take


def _copy_buffer_range_from_tensors(
    tensor_buffer: torch.Tensor,
    range_start: int,
    range_end: int,
    tensor_infos: List[Any],
    tensor_data: List[Optional[torch.Tensor]],
    non_blocking: bool = True,
) -> None:
    """Copy bytes [range_start, range_end) from source tensors into tensor_buffer."""
    if range_end <= range_start:
        return
    for info, tensor in zip(tensor_infos, tensor_data):
        if tensor is None:
            continue
        tensor_start = info.offset
        tensor_end = tensor_start + info.size_bytes
        if tensor_end <= range_start:
            continue
        if tensor_start >= range_end:
            break
        copy_start = max(range_start, tensor_start)
        copy_end = min(range_end, tensor_end)
        local_off = copy_start - tensor_start
        nbytes = copy_end - copy_start
        dst_off = copy_start
        tensor_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
        if tensor_view.numel() != info.size_bytes:
            raise RuntimeError(
                f"ECCHECK legacy save: tensor bytes mismatch for {info.key}, "
                f"expected={info.size_bytes}, got={tensor_view.numel()}"
            )
        use_non_blocking = non_blocking and tensor.is_cuda
        tensor_buffer[dst_off : dst_off + nbytes].copy_(
            tensor_view[local_off : local_off + nbytes],
            non_blocking=use_non_blocking,
        )


def _submit_eccheck_d2h_chunk(
    tensor_buffer: torch.Tensor,
    range_start: int,
    range_end: int,
    actual_data_bytes: int,
    tensor_infos: List[Any],
    tensor_data: List[Optional[torch.Tensor]],
    d2h_stream: Optional[torch.cuda.Stream],
) -> Optional[Tuple[torch.cuda.Event, torch.cuda.Event]]:
    """Launch async D2H for one encode stripe; returns (start, end) CUDA events."""
    if range_start >= actual_data_bytes:
        return None
    d2h_end = min(range_end, actual_data_bytes)
    if d2h_end <= range_start:
        return None
    if d2h_stream is not None:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(d2h_stream):
            start_event.record(d2h_stream)
            _copy_buffer_range_from_tensors(
                tensor_buffer,
                range_start,
                d2h_end,
                tensor_infos,
                tensor_data,
                non_blocking=True,
            )
            end_event.record(d2h_stream)
        return start_event, end_event
    _copy_buffer_range_from_tensors(
        tensor_buffer,
        range_start,
        d2h_end,
        tensor_infos,
        tensor_data,
        non_blocking=False,
    )
    return None


def _encode_eccheck_with_native(
    manager: ECCHECKManager,
    tensor_buffer: torch.Tensor,
    actual_data_bytes: int,
    blocks: Dict[str, Any],
    tensor_infos: List[Any],
    tensor_data: List[Optional[torch.Tensor]],
    rank_metadata: Dict[int, List[TensorMetadata]],
) -> float:
    native = manager._eccheck_native
    if native is None:
        raise RuntimeError("ECCHECK native module is not initialized")

    buffers = manager.get_eccheck_buffers()
    if buffers is None:
        raise RuntimeError("ECCHECK legacy save: buffer pools are not initialized")

    free_data_queue = buffers["free_data_buffer_queue"]
    free_encoding_queue = buffers["free_encoding_buffer_queue"]
    free_parity_queue = buffers["free_parity_buffer_queue"]
    active_event = buffers.get("buffer_poller_active_event")
    poll_and_release = buffers.get("poll_and_release_buffers")

    def _get_free_buffer_with_poll(free_queue, label: str):
        # Keep polling native release queues while waiting; a blocking Queue.get()
        # can otherwise miss releases when the background poller is delayed.
        waited_s = 0.0
        while True:
            if poll_and_release is not None:
                poll_and_release()
            try:
                return free_queue.get(timeout=0.1)
            except queue.Empty:
                waited_s += 0.1
                if waited_s >= 5.0:
                    logger.warning(
                        f"ECCHECK legacy: still waiting for free {label} buffer "
                        f"after {waited_s:.1f}s"
                    )
                    waited_s = 0.0

    def _get_free_data_buffer():
        return _get_free_buffer_with_poll(free_data_queue, "data")

    def _get_free_encoding_buffer():
        return _get_free_buffer_with_poll(free_encoding_queue, "encoding")

    def _get_free_parity_buffer():
        return _get_free_buffer_with_poll(free_parity_queue, "parity")

    own_buffer = blocks["own_buffer"]
    partner_buffer = blocks["partner_buffer"]
    own_base = int(own_buffer.data_ptr())
    partner_base = int(partner_buffer.data_ptr())

    # Get recv encoding buffers (allocated during init)
    if manager.eccheck_recv_encoding_buffers is None:
        raise RuntimeError("ECCHECK legacy save: recv_encoding_buffers not allocated")
    recv_buffer_thread1, recv_buffer_thread2 = manager.eccheck_recv_encoding_buffers
    recv_base_1 = int(recv_buffer_thread1.data_ptr())
    recv_base_2 = int(recv_buffer_thread2.data_ptr())

    pipeline_total_bytes = blocks["pipeline_size"]
    buffer_size = manager.eccheck_buffer_size
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    xor_peer_rank = manager._get_xor_paired_rank(rank, world_size) if world_size > 1 else rank
    xor_peer_actual_bytes = _rank_total_bytes(rank_metadata, xor_peer_rank)
    if world_size > 1:
        rank_in_group = manager._get_rank_in_group(rank, world_size)
        if rank_in_group % 2 == 0:
            p2p_data_rank = manager.get_p2p_partner_rank(rank, world_size)
        else:
            p2p_data_rank = rank
    else:
        p2p_data_rank = rank
    p2p_data_actual_bytes = _rank_total_bytes(rank_metadata, p2p_data_rank)

    native.reset_encoding_completion_flags()
    native.set_load_mode(False, -1)

    if active_event is not None:
        active_event.set()

    src_pos = 0
    recv_offset_1 = 0
    recv_offset_2 = 0
    own_offset = 0
    partner_offset = 0
    src_base_ptr = tensor_buffer.data_ptr()
    d2h_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    d2h_s = 0.0
    pending_d2h: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None

    try:
        while src_pos < pipeline_total_bytes:
            take = _eccheck_chunk_take(
                src_pos,
                pipeline_total_bytes,
                buffer_size,
                recv_offset_1,
                recv_offset_2,
                own_offset,
                partner_offset,
                recv_buffer_thread1,
                recv_buffer_thread2,
                own_buffer,
                partner_buffer,
            )
            if take <= 0:
                logger.warning(
                    f"ECCHECK legacy: recv/P2P buffers exhausted, stopping at "
                    f"{src_pos / (1024**3):.2f} GB / {pipeline_total_bytes / (1024**3):.2f} GB"
                )
                break

            if pending_d2h is None:
                pending_d2h = _submit_eccheck_d2h_chunk(
                    tensor_buffer,
                    src_pos,
                    src_pos + take,
                    actual_data_bytes,
                    tensor_infos,
                    tensor_data,
                    d2h_stream,
                )

            if pending_d2h is not None:
                start_event, end_event = pending_d2h
                end_event.synchronize()
                d2h_s += start_event.elapsed_time(end_event) / 1000.0
                pending_d2h = None

            next_src = src_pos + take
            if next_src < pipeline_total_bytes:
                recv_offset_1_aligned = ((recv_offset_1 + 63) // 64) * 64
                recv_offset_2_aligned = ((recv_offset_2 + 63) // 64) * 64
                next_recv_offset_1 = recv_offset_1_aligned + take
                next_recv_offset_2 = recv_offset_2_aligned + take
                next_own_offset = ((own_offset + 63) // 64) * 64 + take
                next_partner_offset = ((partner_offset + 63) // 64) * 64 + take
                next_take = _eccheck_chunk_take(
                    next_src,
                    pipeline_total_bytes,
                    buffer_size,
                    next_recv_offset_1,
                    next_recv_offset_2,
                    next_own_offset,
                    next_partner_offset,
                    recv_buffer_thread1,
                    recv_buffer_thread2,
                    own_buffer,
                    partner_buffer,
                )
                if next_take > 0:
                    pending_d2h = _submit_eccheck_d2h_chunk(
                        tensor_buffer,
                        next_src,
                        next_src + next_take,
                        actual_data_bytes,
                        tensor_infos,
                        tensor_data,
                        d2h_stream,
                    )

            cur_buffer_addr = _get_free_data_buffer()

            # Copy from tensor_buffer to data buffer (with zero-padding)
            buffer_ptr = ctypes.cast(cur_buffer_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer_array = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_uint8 * take))

            if src_pos < actual_data_bytes:
                bytes_to_copy = min(take, actual_data_bytes - src_pos)
                ctypes.memmove(buffer_array.contents, src_base_ptr + src_pos, bytes_to_copy)
                if take > bytes_to_copy:
                    padding_ptr = ctypes.cast(
                        ctypes.addressof(buffer_array.contents) + bytes_to_copy,
                        ctypes.POINTER(ctypes.c_uint8),
                    )
                    ctypes.memset(padding_ptr, 0, take - bytes_to_copy)
            else:
                ctypes.memset(buffer_array.contents, 0, take)

            enc_addr1 = _get_free_encoding_buffer()
            enc_addr2 = _get_free_encoding_buffer()
            parity_addr1 = _get_free_parity_buffer()
            parity_addr2 = _get_free_parity_buffer()

            recv_chunk_size = take

            recv_offset_1_aligned = ((recv_offset_1 + 63) // 64) * 64
            recv_offset_2_aligned = ((recv_offset_2 + 63) // 64) * 64
            recv_addr_1 = recv_base_1 + recv_offset_1_aligned
            recv_addr_2 = recv_base_2 + recv_offset_2_aligned
            recv_offset_1 = recv_offset_1_aligned + recv_chunk_size
            recv_offset_2 = recv_offset_2_aligned + recv_chunk_size

            own_offset_aligned = ((own_offset + 63) // 64) * 64
            partner_offset_aligned = ((partner_offset + 63) // 64) * 64
            own_write_addr = own_base + own_offset_aligned
            partner_write_addr = partner_base + partner_offset_aligned
            own_offset = own_offset_aligned + take
            partner_offset = partner_offset_aligned + take

            local_is_zero_tail = src_pos >= actual_data_bytes
            remote_is_zero_tail = src_pos >= xor_peer_actual_bytes
            p2p_data_size = min(take, max(0, p2p_data_actual_bytes - src_pos))
            p2p_data_is_zero_tail = p2p_data_size == 0
            native.submit_data_for_encoding_thread1(
                cur_buffer_addr, take, enc_addr1, recv_addr_1,
                recv_chunk_size, parity_addr1, own_write_addr, partner_write_addr,
                local_is_zero_tail, remote_is_zero_tail, p2p_data_is_zero_tail,
                p2p_data_size,
            )
            native.submit_data_for_encoding_thread2(
                cur_buffer_addr, take, enc_addr2, recv_addr_2,
                recv_chunk_size, parity_addr2, own_write_addr, partner_write_addr,
                local_is_zero_tail, remote_is_zero_tail, p2p_data_is_zero_tail,
                p2p_data_size,
            )

            src_pos += take

        if pending_d2h is not None:
            start_event, end_event = pending_d2h
            end_event.synchronize()
            d2h_s += start_event.elapsed_time(end_event) / 1000.0
            pending_d2h = None

        # Sentinels
        native.submit_data_for_encoding_thread1(0, 0, 0, 0, 0, 0, 0, 0)
        native.submit_data_for_encoding_thread2(0, 0, 0, 0, 0, 0, 0, 0)

        native.wait_for_encoding_completion()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        blocks["block_write_sizes"] = {
            "own_buffer": min(own_offset, own_buffer.numel()),
            "partner_buffer": min(partner_offset, partner_buffer.numel()),
        }
    finally:
        if active_event is not None:
            active_event.clear()
    return d2h_s


# ---------------------------------------------------------------------------
# Save .pt files
# ---------------------------------------------------------------------------

def _save_eccheck_pt_files(
    checkpoint_name: str,
    rank: int,
    world_size: int,
    non_tensor_data: Dict[str, Any],
    tensor_infos: List[Any],
    blocks: Dict[str, Any],
    full_tensor_buffer: torch.Tensor,
    flat_key_roots: Optional[Set[str]] = None,
    all_tensor_infos: Optional[Dict[int, List[Any]]] = None,
) -> None:
    checkpoint_path = Path(checkpoint_name)
    checkpoint_dir = checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    block_files = {
        "own_buffer": f"eccheck_block_rank{rank}_own_buffer.pt",
        "partner_buffer": f"eccheck_block_rank{rank}_partner_buffer.pt",
    }

    main_file = checkpoint_dir / f"eccheck_main_rank{rank}.pt"
    from megatron.training.legacy_io_utils import write_raw_checkpoint, write_raw_block, MAGIC_ECCHECK, MAGIC_BLOCK

    # Pre-serialize metadata + prepare memoryview for main file
    import pickle as _pickle
    meta1 = _pickle.dumps(non_tensor_data)
    meta2 = _pickle.dumps(tensor_infos)
    block_payload_sizes = _block_payload_sizes_for_save(
        rank, world_size, all_tensor_infos or {}, blocks,
    )
    extra = _pickle.dumps({
        "version": 1, "format": _FORMAT, "rank": rank,
        "actual_tensor_size": blocks["actual_size"],
        "pipeline_total_bytes": blocks["pipeline_size"],
        "aligned_block_size": blocks["aligned_size"],
        "block_write_sizes": blocks.get("block_write_sizes", {}),
        "block_payload_sizes": block_payload_sizes,
        "flat_key_roots": list(flat_key_roots) if flat_key_roots else [],
        "block_files": block_files,
        "all_tensor_infos": all_tensor_infos if all_tensor_infos is not None else {},
    })
    buf = full_tensor_buffer[: blocks["actual_size"]]
    if not buf.is_contiguous():
        buf = buf.contiguous()
    if buf.device.type != "cpu":
        buf = buf.to("cpu")
    main_mv = memoryview(buf.numpy())

    # ECCHECK blocks keep the internal 64B gapped layout, but trim unused tail bytes.
    block_names = ("own_buffer", "partner_buffer")
    block_write_sizes = block_payload_sizes
    block_mvs = {}
    for name in block_names:
        b = blocks[name][: blocks[name].numel()]
        if not b.is_contiguous():
            b = b.contiguous()
        if b.device.type != "cpu":
            b = b.to("cpu")
        block_mvs[name] = memoryview(b.numpy())

    # Parallel writes (f.write releases GIL — truly concurrent I/O)
    import concurrent.futures
    from megatron.training.legacy_io_utils import write_main_prepared, write_block_prepared
    with concurrent.futures.ThreadPoolExecutor(max_workers=1 + len(block_names)) as ex:
        futs = [ex.submit(write_main_prepared, str(main_file), MAGIC_ECCHECK,
                          meta1, meta2, extra, main_mv, blocks["actual_size"])]
        for name in block_names:
            block_file = checkpoint_dir / f"eccheck_block_rank{rank}_{name}.pt"
            block_write_size = int(block_write_sizes.get(name, blocks[name].numel()))
            futs.append(ex.submit(write_block_prepared,
                                  str(block_file), MAGIC_BLOCK,
                                  block_mvs[name], block_write_size))
        for f in futs:
            f.result()




# ---------------------------------------------------------------------------
# Main save entry point
# ---------------------------------------------------------------------------

def save_eccheck_legacy_checkpoint(
    state_dict: Dict[str, Any], checkpoint_name: str, write_to_disk: bool = True
) -> None:
    t0 = time.time()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    manager = ECCHECKManager()
    manager.init_eccheck_if_enabled()
    if manager._eccheck_native is None:
        raise RuntimeError("ECCHECK native module is not available in legacy save path")

    t0 = time.time()
    decomposed, save_copy_s, save_flatten_s, decompose_s = decompose_state_dict_for_save(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    logger.debug(
        "ECCHECK save timing: copy %.3fs flatten %.3fs decompose %.3fs",
        save_copy_s, save_flatten_s, decompose_s,
    )

    t0 = time.time()
    safety_margin = max(int(total_tensor_size * 0.01), manager.eccheck_buffer_size)
    manager.allocate_preallocated_buffer(total_tensor_size + safety_margin)
    tensor_buffer = manager.preallocated_cpu_buffer

    offset = 0
    local_tensor_metadata: List[TensorMetadata] = []
    for info in decomposed.tensor_infos:
        info.offset = offset
        local_tensor_metadata.append(
            TensorMetadata(
                key=info.key,
                shape=info.shape,
                dtype=str(info.dtype),
                size_bytes=info.size_bytes,
                offset=offset,
                global_offset=tuple(info.global_offset) if info.global_offset else tuple(),
                shard_index=info.shard_index if info.shard_index is not None else 0,
                chunk_type="data",
                target_rank=rank,
                source_rank=rank,
            )
        )
        offset += info.size_bytes

    t0 = time.time()
    # Only tensor metadata needed for block sizing; non_tensor_data (~250MB)
    # is exchanged by all_gather_object but never consumed here.  Pass an empty
    # dict to avoid wasting 5+ seconds on unnecessary exchange.
    rank_metadata, _ = _build_global_registry(local_tensor_metadata, {})
    metadata_s = time.time() - t0

    t0 = time.time()
    blocks = _allocate_eccheck_blocks_legacy(manager, rank_metadata)
    block_alloc_s = time.time() - t0

    # Allocate recv encoding buffers now that we know peer data sizes
    registry = GlobalMetadataRegistry(
        rank_metadata=rank_metadata, rank_non_tensor_data={}
    )
    t0 = time.time()
    if manager.eccheck_recv_encoding_buffers is None:
        manager.eccheck_recv_encoding_buffers = (
            manager.allocate_recv_encoding_buffers_phase2(registry)
        )
    recv_alloc_s = time.time() - t0

    if manager.use_rdma:
        manager.register_buffer(tensor_buffer)

    logger.debug(
        f"ECCHECK legacy save: rank {rank} encoding "
        f"{blocks['pipeline_size'] / (1024**3):.2f} GB pipeline "
        f"(actual: {total_tensor_size / (1024**3):.2f} GB)"
    )

    if world_size > 1:
        torch.distributed.barrier()
    e2e_t0 = time.time()

    encode_t0 = time.time()
    d2h_s = _encode_eccheck_with_native(
        manager=manager,
        tensor_buffer=tensor_buffer,
        actual_data_bytes=total_tensor_size,
        blocks=blocks,
        tensor_infos=decomposed.tensor_infos,
        tensor_data=decomposed.tensor_data,
        rank_metadata=rank_metadata,
    )
    del decomposed.tensor_data
    network_encode_s = time.time() - encode_t0
    native_timing = _native_ft_timing(manager._eccheck_native)
    e2e_s = time.time() - e2e_t0
    if world_size > 1:
        torch.distributed.barrier()
    summary = _timing_max_dict({
        "e2e_s": e2e_s,
        "d2h_s": d2h_s,
        "network_encode_s": network_encode_s,
        "net_s": native_timing["net_s"],
        "encode_s": native_timing["encode_s"],
    })
    if rank == 0:
        logger.info(
            "ECCHECK save timing: e2e_s=%(e2e_s).2fs d2h_s=%(d2h_s).2fs "
            "network_encode_s=%(network_encode_s).2fs net_s=%(net_s).2fs encode_s=%(encode_s).2fs",
            summary,
        )

    if write_to_disk:
        _save_eccheck_pt_files(
            checkpoint_name=checkpoint_name,
            rank=rank,
            world_size=world_size,
            non_tensor_data=decomposed.non_tensor_data,
            tensor_infos=decomposed.tensor_infos,
            blocks=blocks,
            full_tensor_buffer=tensor_buffer[:total_tensor_size],
            flat_key_roots=decomposed.flat_key_roots,
            all_tensor_infos=rank_metadata,
        )
    if world_size > 1:
        _timed_barrier()


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_eccheck_main_payload_local(
    checkpoint_dir: Path,
    rank: int,
    load_tensor_buffer: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    main_path = checkpoint_dir / f"eccheck_main_rank{rank}.pt"
    local_payload: Optional[Dict[str, Any]] = None
    local_error: Optional[str] = None
    if main_path.is_file():
        try:
            from megatron.training.legacy_io_utils import (
                is_raw_format, read_raw_checkpoint, read_raw_checkpoint_metadata,
                MAGIC_ECCHECK, pin_payload_tensor_buffer_if_available,
            )
            if is_raw_format(str(main_path), MAGIC_ECCHECK):
                if load_tensor_buffer:
                    local_payload = read_raw_checkpoint(
                        str(main_path), MAGIC_ECCHECK, pin_tensor_buffer=True,
                    )
                else:
                    local_payload = read_raw_checkpoint_metadata(
                        str(main_path), MAGIC_ECCHECK,
                    )
            else:
                local_payload = torch.load(main_path, map_location="cpu", weights_only=False)
                if load_tensor_buffer:
                    pin_payload_tensor_buffer_if_available(local_payload)
                else:
                    logger.warning(
                        "ECCHECK legacy: metadata-only load requested for old torch "
                        "checkpoint format; tensor_buffer must still be deserialized"
                    )
                    local_payload["tensor_buffer"] = None
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    return local_payload, local_error


def _load_eccheck_main_payload(
    checkpoint_dir: Path,
    rank: int,
    world_size: int,
    load_tensor_buffer: bool = True,
) -> Dict[str, Any]:
    main_path = checkpoint_dir / f"eccheck_main_rank{rank}.pt"
    local_payload, local_error = _load_eccheck_main_payload_local(
        checkpoint_dir, rank, load_tensor_buffer=load_tensor_buffer,
    )

    if world_size <= 1 or not torch.distributed.is_initialized():
        if local_error is not None:
            raise RuntimeError(
                f"ECCHECK legacy: failed reading main file {main_path}: {local_error}"
            )
        if local_payload is None:
            raise FileNotFoundError(f"ECCHECK legacy: missing main file {main_path}")
        return local_payload

    # Strip tensor_buffer before all_gather — it's multiple GB for large models
    # and NCCL all_gather creates GPU staging buffers proportional to
    # world_size × serialized_size → OOM on 7B+.
    if local_payload is not None:
        stripped: Dict[str, Any] = {}
        for k, v in local_payload.items():
            if k == "tensor_buffer":
                continue
            stripped[k] = v
    else:
        stripped = None

    if local_error is not None:
        stripped = {
            "__eccheck_load_error__": local_error,
            "__eccheck_load_path__": str(main_path),
        }

    gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, stripped)

    load_errors = {
        r: (payload.get("__eccheck_load_path__"), payload.get("__eccheck_load_error__"))
        for r, payload in enumerate(gathered)
        if payload is not None and "__eccheck_load_error__" in payload
    }
    if load_errors:
        details = "; ".join(
            f"rank {r} path={path}: {err}" for r, (path, err) in load_errors.items()
        )
        raise RuntimeError(f"ECCHECK legacy: failed reading main payload(s): {details}")

    if local_payload is not None:
        return local_payload

    # HW failure: local disk lost — recover metadata from another rank.
    for r in range(world_size):
        if gathered[r] is not None:
            payload = dict(gathered[r])
            all_ti = payload.get("all_tensor_infos")
            if all_ti and rank in all_ti:
                logger.debug(
                    f"ECCHECK legacy: eccheck_main_rank{rank}.pt missing locally; "
                    f"recovered tensor_infos for rank {rank} from rank {r}"
                )
                payload["tensor_infos"] = all_ti[rank]
                payload["tensor_buffer"] = None
                return payload
            if "tensor_infos" in payload:
                logger.warning(
                    f"ECCHECK legacy: using rank {r}'s tensor_infos as fallback "
                    f"for rank {rank} — may be incorrect (old checkpoint format)"
                )
                payload["tensor_buffer"] = None
                return payload

    raise FileNotFoundError(
        f"ECCHECK legacy: eccheck_main_rank{rank}.pt missing on all ranks "
        f"under {checkpoint_dir}"
    )


def _copy_block_file_into_tensor(
    checkpoint_dir: Path, rank: int, block_name: str, dest: torch.Tensor
) -> None:
    block_path = checkpoint_dir / f"eccheck_block_rank{rank}_{block_name}.pt"
    if not block_path.is_file():
        raise FileNotFoundError(f"ECCHECK legacy load: missing block file {block_path}")
    from megatron.training.legacy_io_utils import (
        is_raw_format, read_raw_block_range, MAGIC_BLOCK, pin_uint8_tensor_if_available,
    )
    if dest.device.type != "cpu" or dest.dtype != torch.uint8 or not dest.is_contiguous():
        raise ValueError("ECCHECK raw block destination must be a contiguous CPU uint8 tensor")
    dst = dest.view(-1)
    if is_raw_format(str(block_path), MAGIC_BLOCK):
        with open(block_path, "rb") as block_file:
            header = block_file.read(12)
        if len(header) != 12:
            raise EOFError(f"Incomplete raw block header in {block_path}")
        magic, data_len = header[:4], struct.unpack("<Q", header[4:])[0]
        if magic != MAGIC_BLOCK:
            raise ValueError(
                f"Unexpected magic {magic!r} (expected {MAGIC_BLOCK!r}) in {block_path}"
            )
        n = min(data_len, dst.numel())
        read_raw_block_range(str(block_path), MAGIC_BLOCK, dst, 0, n)
        if n < dst.numel():
            dst[n:].zero_()
        return
    else:
        payload = torch.load(block_path, map_location="cpu", weights_only=False)
        src = pin_uint8_tensor_if_available(
            payload["tensor"].contiguous().view(torch.uint8).reshape(-1)
        )
    n = min(src.numel(), dst.numel())
    dst[:n].copy_(src[:n])
    if n < dst.numel():
        dst[n:].zero_()


def _load_eccheck_blocks_from_disk_into(
    blocks: Dict[str, Any],
    checkpoint_dir: Path,
    rank: int,
    rank_in_group: int,
    software_failure: bool = False,
    two_failures: bool = False,
) -> None:
    """Load block .pt files into pre-allocated P2P buffers.

    Normal recovery layout:
    - rank_in_group 2: loads nothing (data comes via network XOR recovery)
    - rank_in_group 0: loads own_buffer parity and partner_buffer data
    - rank_in_group 1: loads own_buffer (its own data)
    - rank_in_group 3: loads own_buffer + partner_buffer (its own data + received)

    Software failure layout (use_eccheck_software_failure=True):
    - rank_in_group 0: loads partner_buffer (d1, to send to rig=1 via C++ P2P)
    - rank_in_group 1: loads nothing (receives from rig=0 via C++ P2P)
    - rank_in_group 2/3: not participating in software failure

    Two-failures layout (use_eccheck_two_failures=True):
    - rank_in_group 0 (survivor): loads partner_buffer (d1, to send to rig1)
    - rank_in_group 1 (failed): loads nothing (receives d1 from rig0)
    - rank_in_group 2 (failed): loads nothing (receives p2 from rig3, XOR recovers d2)
    - rank_in_group 3 (survivor): loads own_buffer + partner_buffer (p3 + p2)
    """
    if software_failure:
        if rank_in_group == 0:
            _copy_block_file_into_tensor(
                checkpoint_dir, rank, "partner_buffer", blocks["partner_buffer"]
            )
        # rig=1: will receive via C++ P2P, no local disk load
        # rig=2/3: not participating
        return

    if two_failures:
        if rank_in_group == 0:
            # Survivor: load d1 (partner_buffer) for sending to rig1
            _copy_block_file_into_tensor(
                checkpoint_dir, rank, "partner_buffer", blocks["partner_buffer"]
            )
        # rig=1: failed, loads nothing
        # rig=2: failed, loads nothing
        elif rank_in_group == 3:
            # Survivor: load p3 (own_buffer = 2·d1⊕2·d3) and p2 (partner_buffer = d0⊕d2)
            _copy_block_file_into_tensor(
                checkpoint_dir, rank, "own_buffer", blocks["own_buffer"]
            )
            _copy_block_file_into_tensor(
                checkpoint_dir, rank, "partner_buffer", blocks["partner_buffer"]
            )
        elif rank_in_group not in (0, 3):
            pass  # rig1, rig2: no disk load
        return

    if rank_in_group == 2:
        # Hardware failure: this rank's data needs network recovery, skip disk load
        return

    if rank_in_group == 0:
        # Even save roles keep parity in own_buffer and partner data in partner_buffer.
        # HW1 encodes from own_buffer and sends partner_buffer in Step2.
        _copy_block_file_into_tensor(
            checkpoint_dir, rank, "own_buffer", blocks["own_buffer"]
        )
        _copy_block_file_into_tensor(
            checkpoint_dir, rank, "partner_buffer", blocks["partner_buffer"]
        )
    elif rank_in_group == 1:
        _copy_block_file_into_tensor(
            checkpoint_dir, rank, "own_buffer", blocks["own_buffer"]
        )
    elif rank_in_group == 3:
        _copy_block_file_into_tensor(
            checkpoint_dir, rank, "own_buffer", blocks["own_buffer"]
        )
        _copy_block_file_into_tensor(
            checkpoint_dir, rank, "partner_buffer", blocks["partner_buffer"]
        )
    else:
        raise RuntimeError(
            f"ECCHECK legacy load: unexpected rank_in_group={rank_in_group}"
        )


# ---------------------------------------------------------------------------
# Recovery helpers
# ---------------------------------------------------------------------------

def _extract_dense_from_gapped_buffer(
    gapped_buf: torch.Tensor,
    chunk_size: int,
    total_bytes: int,
) -> torch.Tensor:
    """Reconstruct a dense byte buffer from a gapped buffer written by C++.

    The C++ encoding writes chunks at 64-byte-aligned offsets.  Between
    chunks there may be 0--63 bytes of alignment padding.  This function
    extracts only the data bytes, producing a contiguous dense buffer.

    Mirrors ``_decode_data0_to_linear_first_half`` in ecnaive_legacy.py.
    """
    out = torch.zeros(total_bytes, dtype=torch.uint8, device=gapped_buf.device)
    src_pos = 0       # position in output (dense)
    buf_offset = 0    # position in gapped_buf
    buf_size = gapped_buf.numel()
    while src_pos < total_bytes:
        remaining = total_bytes - src_pos
        take = min(chunk_size, remaining)
        aligned_offset = ((buf_offset + 63) // 64) * 64
        if aligned_offset + take > buf_size:
            logger.warning(
                f"ECCHECK legacy: gapped buffer exhausted at src_pos={src_pos} "
                f"(buf_size={buf_size}, aligned={aligned_offset}, take={take})"
            )
            break
        out[src_pos : src_pos + take].copy_(
            gapped_buf[aligned_offset : aligned_offset + take]
        )
        buf_offset = aligned_offset + take
        src_pos += take
    return out


# ---------------------------------------------------------------------------
# Recovery pipeline (load)
# ---------------------------------------------------------------------------

def _run_eccheck_legacy_recovery(
    manager: ECCHECKManager,
    rank: int,
    world_size: int,
    blocks: Dict[str, Any],
    recv_buffers: Optional[Dict[str, torch.Tensor]],
    recovered_buffer: Optional[torch.Tensor],
    total_size: int,
    registry: GlobalMetadataRegistry,
    native_prepared: bool = False,
) -> float:
    """Drive C++ recovery for ECCHECK legacy load using submit_load_pipeline_chunk.

    Returns the network and encoding time in seconds.

    Uses the same chunked pipeline API as the modern path but sources data
    from pre-loaded .pt block buffers instead of mmap files.
    """
    from time import monotonic, sleep, time

    native = manager._eccheck_native
    if native is None:
        raise RuntimeError("ECCHECK native module is not initialized")

    from megatron.training import get_args as _get_args
    args = _get_args()
    software_failure = bool(getattr(args, "use_eccheck_software_failure", False))

    rank_in_group = manager._get_rank_in_group(rank, world_size)

    # ---- software failure path ----
    # rank_in_group 0 sends partner_buffer (d1) to rank_in_group 1 via C++ P2P.
    # rank_in_group 1 receives into recovered_buffer.
    # This exercises the network path for worst-case recovery time measurement.
    if software_failure:
        start_t = time()
        if rank_in_group == 0:
            partner_rank = manager.get_p2p_partner_rank(rank, world_size)
            transfer_bytes = _rank_data_transfer_bytes(registry, partner_rank, world_size)
            partner_buf = blocks["partner_buffer"].contiguous().view(torch.uint8).reshape(-1)
            send_size = min(transfer_bytes, partner_buf.numel())
            native.simple_p2p_send(int(partner_buf.data_ptr()), send_size)
        elif rank_in_group == 1:
            if recovered_buffer is None:
                raise RuntimeError("ECCHECK legacy: software failure needs recovered_buffer")
            transfer_bytes = _rank_data_transfer_bytes(registry, rank, world_size)
            recv_size = min(transfer_bytes, recovered_buffer.numel())
            native.simple_p2p_recv(int(recovered_buffer.data_ptr()), recv_size)
            if recv_size < recovered_buffer.numel():
                recovered_buffer[recv_size:].zero_()
        # rig=2/3: no-op
        return time() - start_t

    # ---- hardware failure path (rank_in_group 2) ----
    failed_rank = 2
    if not native_prepared:
        native.set_load_mode(True, failed_rank)
    logger.debug(f"ECCHECK legacy: set load mode (failed_rank={failed_rank})")

    # Compute pipeline size
    max_total_bytes = _max_tensor_bytes_from_registry(registry, world_size)
    if max_total_bytes == 0:
        return 0.0
    buffer_size = manager.eccheck_buffer_size

    # Get buffer pools (reuse save-time pools via manager)
    buffers = manager.get_eccheck_buffers()
    if buffers is None:
        raise RuntimeError("ECCHECK legacy: buffer pools not initialized")
    free_data_queue = buffers["free_data_buffer_queue"]
    free_encoding_queue = buffers["free_encoding_buffer_queue"]
    free_parity_queue = buffers["free_parity_buffer_queue"]
    poll_and_release = buffers.get("poll_and_release_buffers")
    active_event = buffers.get("buffer_poller_active_event")

    def _get_free_data():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_data_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECCHECK legacy: timeout waiting for free data buffer")
            return free_data_queue.get()

    def _get_free_encoding():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_encoding_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECCHECK legacy: timeout waiting for free encoding buffer")
            return free_encoding_queue.get()

    def _get_free_parity():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_parity_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECCHECK legacy: timeout waiting for free parity buffer")
            return free_parity_queue.get()

    # HW1 uses an isolated single-physical receive cache. Save may retain its
    # independent two-buffer cache in an in-process manager.
    if manager.eccheck_hw1_recv_encoding_buffers is None:
        manager.allocate_recv_encoding_buffers_phase2(
            registry, hw1_single_physical_buffer=True,
        )
    __, recv_buf2 = manager.eccheck_hw1_recv_encoding_buffers
    recv_base2 = int(recv_buf2.data_ptr())

    own_base = int(blocks["own_buffer"].data_ptr())
    partner_base = int(blocks["partner_buffer"].data_ptr())
    own_buf = blocks["own_buffer"]
    partner_buf = blocks["partner_buffer"]
    full_buffers = [own_buf, partner_buf]
    if recovered_buffer is not None:
        full_buffers.append(recovered_buffer)
    recovered_alias = (
        "own" if recovered_buffer is own_buf
        else "partner" if recovered_buffer is partner_buf
        else "none"
    )
    logger.debug(
        "ECCHECK HW1 memory layout: role=rig%d alias=recovered:%s "
        "unique_full_buffers=%d recv_physical=%d logical_recv=2",
        rank_in_group,
        recovered_alias,
        len({int(buffer.data_ptr()) for buffer in full_buffers}),
        len({int(buffer.data_ptr()) for buffer in manager.eccheck_hw1_recv_encoding_buffers}),
    )

    if active_event is not None:
        active_event.set()

    processed = 0
    p2p_partner_offset = 0
    recv_offset2 = 0

    t_pipeline_net_start: Optional[float] = None
    t_pipeline_net = 0.0
    try:
        while processed < max_total_bytes:
            # Track only submissions accepted by the native HW2 pipeline.
            remaining = max_total_bytes - processed
            take = min(buffer_size, remaining)

            cur_buffer_addr = _get_free_data()

            # Copy source data from pre-loaded P2P block buffers (not timed)
            buffer_ptr = ctypes.cast(cur_buffer_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer_array = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_uint8 * take))

            if rank_in_group == 0:
                src_off = processed
                bytes_to_copy = min(take, own_buf.numel() - src_off)
                if bytes_to_copy > 0:
                    ctypes.memmove(buffer_array.contents,
                                   own_base + src_off, bytes_to_copy)
                if take > bytes_to_copy:
                    ctypes.memset(buffer_array.contents + bytes_to_copy, 0,
                                  take - bytes_to_copy)
            elif rank_in_group == 1:
                src_off = processed
                bytes_to_copy = min(take, partner_buf.numel() - src_off)
                if bytes_to_copy > 0:
                    ctypes.memmove(buffer_array.contents,
                                   partner_base + src_off, bytes_to_copy)
                if take > bytes_to_copy:
                    ctypes.memset(buffer_array.contents + bytes_to_copy, 0,
                                  take - bytes_to_copy)
            elif rank_in_group == 3:
                src_off = processed
                bytes_to_copy = min(take, own_buf.numel() - src_off)
                if bytes_to_copy > 0:
                    ctypes.memmove(buffer_array.contents,
                                   own_base + src_off, bytes_to_copy)
                if take > bytes_to_copy:
                    ctypes.memset(buffer_array.contents + bytes_to_copy, 0,
                                  take - bytes_to_copy)
            else:
                # rank2: fill with zeros, will receive data via network
                ctypes.memset(buffer_array.contents, 0, take)

            enc_addr2 = _get_free_encoding()

            # Receivers need a network buffer. Rig2 writes XOR output directly
            # into dense recovered storage; rig3 retains pooled parity scratch.
            if rank_in_group in (2, 3):
                if rank_in_group == 2:
                    if recovered_buffer is None:
                        raise RuntimeError("ECCHECK legacy: rig2 needs recovered_buffer")
                    parity_addr2 = int(recovered_buffer.data_ptr()) + processed
                    parity_is_pooled = False
                else:
                    parity_addr2 = _get_free_parity()
                    parity_is_pooled = True
                recv_offset2_aligned = ((recv_offset2 + 63) // 64) * 64
                recv_addr2 = recv_base2 + recv_offset2_aligned
                recv_chunk_size = take
                recv_offset2 = recv_offset2_aligned + recv_chunk_size
            else:
                parity_addr2 = 0
                parity_is_pooled = False
                recv_addr2 = 0
                recv_chunk_size = 0

            # Step2 P2P: rank0/3 send partner data; rank1/2 recv
            if rank_in_group in (0, 3):
                step2_send_addr = partner_base + processed
                step2_recv_data_addr = 0
                step2_size = take
            elif rank_in_group in (1, 2):
                step2_send_addr = 0
                step2_recv_data_addr = cur_buffer_addr
                step2_size = take
            else:
                step2_send_addr = 0
                step2_recv_data_addr = 0
                step2_size = 0

            # Step6 P2P: rank2 receives d3 from rank3
            if rank_in_group == 2:
                p2p_partner_offset_aligned = ((p2p_partner_offset + 63) // 64) * 64
                p2p_partner_write = partner_base + p2p_partner_offset_aligned
                p2p_partner_offset = p2p_partner_offset_aligned + take
            else:
                p2p_partner_write = 0

            if t_pipeline_net_start is None:
                t_pipeline_net_start = time()
            native.submit_load_pipeline_chunk(
                step2_send_addr=step2_send_addr,
                step2_recv_data_addr=step2_recv_data_addr,
                step2_size=step2_size,
                data_addr=cur_buffer_addr,
                size=take,
                encoding_addr=enc_addr2,
                recv_addr=recv_addr2,
                recv_chunk_size=recv_chunk_size,
                parity_addr=parity_addr2,
                parity_is_pooled=parity_is_pooled,
                p2p_partner_write_addr=p2p_partner_write,
            )

            processed += take

        # Sentinels and completion
        if t_pipeline_net_start is None:
            t_pipeline_net_start = time()
        native.submit_load_encoding_sentinel()
        native.wait_for_xor_worker_completion()

        if rank_in_group in (2, 3):
            native.submit_load_step6_p2p_sentinel()

        native.wait_for_encoding_completion()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_pipeline_net = time() - t_pipeline_net_start

        logger.debug(
            f"ECCHECK legacy: hw recovery pipeline done in {t_pipeline_net:.2f}s"
        )

    finally:
        if active_event is not None:
            active_event.clear()

    return t_pipeline_net


# ---------------------------------------------------------------------------
# Two-failure recovery pipeline (load)
# ---------------------------------------------------------------------------

def _run_eccheck_two_failures_recovery(
    manager: ECCHECKManager,
    rank: int,
    world_size: int,
    blocks: Dict[str, Any],
    source_blocks: Dict[str, torch.Tensor],
    recovered_buffer: Optional[torch.Tensor],
    total_size: int,
    registry: GlobalMetadataRegistry,
    native_prepared: bool = False,
) -> Dict[str, float]:
    """Drive C++ two-failure recovery for ECCHECK legacy load.

    Returns the recovery wall-time breakdown in seconds.

    Two physical nodes lost → rig1 and rig2 in each 4-rank group are failed.
    Survivors rig0 and rig3 use bidirectional XOR exchange to recover
    d2 (on rig2) and verify d3 (on rig3), then forward results via P2P.

    Uses save-path 16-thread encode pool (ec_rs_encode_pool) and RDMA transport,
    aligned with the save encoding pipeline.
    """
    from time import monotonic, sleep, time

    native = manager._eccheck_native
    if native is None:
        raise RuntimeError("ECCHECK native module is not initialized")

    rank_in_group = manager._get_rank_in_group(rank, world_size)
    is_failed = rank_in_group in (1, 2)
    is_survivor = rank_in_group in (0, 3)

    # HW2 uses persistent save-path workers; reset the completed cycle before submission.
    if not native_prepared:
        manager.reset_native_for_inprocess_recovery(10)
    logger.debug(
        f"ECCHECK legacy two-failures: load mode set (failed_rank=10), "
        f"rank={rank}, rank_in_group={rank_in_group}"
    )

    # Compute pipeline size
    max_total_bytes = _max_tensor_bytes_from_registry(registry, world_size)
    if max_total_bytes == 0:
        return {"network_encode": 0.0, "phase1_p2p_s": 0.0, "pipeline_wall_s": 0.0}
    buffer_size = manager.eccheck_buffer_size

    # Get buffer pools
    buffers = manager.get_eccheck_buffers()
    if buffers is None:
        raise RuntimeError("ECCHECK legacy: buffer pools not initialized")
    free_data_queue = buffers["free_data_buffer_queue"]
    free_encoding_queue = buffers["free_encoding_buffer_queue"]
    free_parity_queue = buffers["free_parity_buffer_queue"]
    poll_and_release = buffers.get("poll_and_release_buffers")
    active_event = buffers.get("buffer_poller_active_event")

    def _get_free_data():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_data_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECCHECK two-failures: timeout waiting for free data buffer")
            return free_data_queue.get()

    def _get_free_encoding():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_encoding_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECCHECK two-failures: timeout waiting for free encoding buffer")
            return free_encoding_queue.get()

    def _get_free_parity():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_parity_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECCHECK two-failures: timeout waiting for free parity buffer")
            return free_parity_queue.get()

    # Allocate the bounded receive ring and preserve the two-lane tuple API.
    ring_depth = _eccheck_hw2_recv_ring_depth()
    if manager.eccheck_hw2_recv_encoding_buffers is None:
        manager.allocate_recv_encoding_buffers_phase2(
            registry, hw2_single_physical_buffer=True, hw2_ring_depth=ring_depth,
        )
    elif manager.eccheck_hw2_recv_ring_depth != ring_depth:
        raise RuntimeError(
            "ECCHECK HW2 cached receive ring depth mismatch: "
            f"allocated={manager.eccheck_hw2_recv_ring_depth}, requested={ring_depth}"
        )
    recv_buf1, recv_buf2 = manager.eccheck_hw2_recv_encoding_buffers
    recv_base_1 = int(recv_buf1.data_ptr())
    recv_base_2 = int(recv_buf2.data_ptr())

    own_base = int(blocks["own_buffer"].data_ptr())
    partner_base = int(blocks["partner_buffer"].data_ptr())
    own_buf = blocks["own_buffer"]
    partner_buf = blocks["partner_buffer"]
    full_buffers = [own_buf, partner_buf]
    if recovered_buffer is not None:
        full_buffers.append(recovered_buffer)
    recovered_alias = (
        "partner_buffer" if recovered_buffer is partner_buf
        else "own_buffer" if recovered_buffer is own_buf
        else "none"
    )
    logger.debug(
        "ECCHECK HW2 memory layout: role=rig%d unique_full_blocks=%d "
        "recv_physical_buffers=1 recv_ring_depth=%d recv_ring_bytes=%d "
        "logical_lanes=2 alias=recovered:%s",
        rank_in_group,
        len({int(buffer.data_ptr()) for buffer in full_buffers}),
        ring_depth,
        ring_depth * buffer_size,
        recovered_alias,
    )
    if rank_in_group == 0:
        input_buf = source_blocks.get("d0")
        phase1_buf = source_blocks.get("d1")
    elif rank_in_group == 3:
        input_buf = source_blocks.get("p3")
        phase1_buf = source_blocks.get("p2")
    elif rank_in_group == 1:
        input_buf = partner_buf
        phase1_buf = None
    else:
        input_buf = own_buf
        phase1_buf = None
    if input_buf is None or (is_survivor and phase1_buf is None):
        raise RuntimeError(f"ECCHECK HW2 immutable sources missing for rig{rank_in_group}")
    input_base = int(input_buf.data_ptr())

    if active_event is not None:
        active_event.set()

    t_phase1_p2p = 0.0
    t_pipeline_net = 0.0
    t_pipeline_net_start: Optional[float] = None

    # ---- Phase 1: P2P data distribution (timed as network) ----
    # Rig0 → Rig1: send d1 (partner_buffer)
    # Rig3 → Rig2: send p2 = d0⊕d2 (partner_buffer)
    phase1_parity_bytes = _eccheck_pipeline_transfer_bytes(registry, world_size)
    _t0 = time()
    if rank_in_group == 0:
        rig1_rank = manager._get_rank_by_group_position(
            manager._get_group_id(rank, world_size), 1, world_size,
        )
        send_size = min(
            _rank_data_transfer_bytes(registry, rig1_rank, world_size),
            phase1_buf.numel(),
        )
        native.simple_p2p_send(int(phase1_buf.data_ptr()), send_size)
        logger.debug(
            f"ECCHECK two-failures: rig0 sent d1 to rig1 "
            f"({send_size / (1024**3):.2f} GB)"
        )
    elif rank_in_group == 1:
        recv_size = min(
            _rank_data_transfer_bytes(registry, rank, world_size),
            partner_buf.numel(),
        )
        native.simple_p2p_recv(int(partner_buf.data_ptr()), recv_size)
        if recv_size < partner_buf.numel():
            partner_buf[recv_size:].zero_()
        logger.debug(
            f"ECCHECK two-failures: rig1 received d1 from rig0 "
            f"({recv_size / (1024**3):.2f} GB)"
        )
    elif rank_in_group == 2:
        recv_size = min(phase1_parity_bytes, own_buf.numel())
        native.simple_p2p_recv(int(own_buf.data_ptr()), recv_size)
        logger.debug(
            f"ECCHECK two-failures: rig2 received p2 from rig3 "
            f"({recv_size / (1024**3):.2f} GB)"
        )
    elif rank_in_group == 3:
        send_size = min(phase1_parity_bytes, phase1_buf.numel())
        native.simple_p2p_send(int(phase1_buf.data_ptr()), send_size)
        logger.debug(
            f"ECCHECK two-failures: rig3 sent p2 to rig2 "
            f"({send_size / (1024**3):.2f} GB)"
        )
    t_phase1_p2p = time() - _t0

    processed = 0
    own_offset = 0
    partner_offset = 0

    stale_releases = native.get_two_failure_recv_buffers_to_release()
    if stale_releases:
        raise RuntimeError(
            "ECCHECK HW2 typed receive release queue was not empty at cycle start: "
            f"count={len(stale_releases)}"
        )
    if recv_base_1 != recv_base_2:
        raise RuntimeError("ECCHECK HW2 logical receive lanes must alias one physical ring")
    ring_base = recv_base_1
    ring_bytes = ring_depth * buffer_size
    if recv_buf1.numel() != ring_bytes or recv_buf2.numel() != ring_bytes:
        raise RuntimeError(
            "ECCHECK HW2 receive ring size mismatch: "
            f"expected={ring_bytes}, lane1={recv_buf1.numel()}, lane2={recv_buf2.numel()}"
        )
    free_slots = deque(ring_base + index * buffer_size for index in range(ring_depth))
    inflight: Set[int] = set()
    logger.debug(
        "ECCHECK HW2 receive ring setup: ring_depth=%d ring_bytes=%d",
        ring_depth, ring_bytes,
    )
    ring_stall_s = 0.0
    ring_wait_count = 0
    max_inflight = 0
    submitted_chunks = 0

    def _poll_recv_slot_releases() -> int:
        released = native.get_two_failure_recv_buffers_to_release()
        for address_value in released:
            address = int(address_value)
            offset = address - ring_base
            if offset < 0 or offset >= ring_bytes:
                raise RuntimeError(
                    f"ECCHECK HW2 native released address outside receive ring: 0x{address:x}"
                )
            if offset % buffer_size != 0:
                raise RuntimeError(
                    f"ECCHECK HW2 native released unaligned receive slot: 0x{address:x}"
                )
            if address not in inflight:
                raise RuntimeError(
                    f"ECCHECK HW2 duplicate or non-inflight receive slot release: 0x{address:x}"
                )
            inflight.remove(address)
            free_slots.append(address)
        return len(released)

    def _get_recv_slot() -> int:
        nonlocal ring_stall_s, ring_wait_count
        _poll_recv_slot_releases()
        if free_slots:
            address = free_slots.popleft()
            inflight.add(address)
            return address
        wait_start = monotonic()
        deadline = wait_start + 120.0
        next_warning = wait_start + 5.0
        ring_wait_count += 1
        while not free_slots:
            _poll_recv_slot_releases()
            if free_slots:
                break
            now = monotonic()
            if now >= deadline:
                raise RuntimeError(
                    "ECCHECK HW2 timed out waiting for a free receive ring slot: "
                    f"inflight={len(inflight)}, free={len(free_slots)}, depth={ring_depth}"
                )
            if now >= next_warning:
                logger.warning(
                    "ECCHECK HW2 waiting for a free receive ring slot for %.1fs",
                    now - wait_start,
                )
                next_warning = now + 5.0
            sleep(0.001)
        ring_stall_s += monotonic() - wait_start
        address = free_slots.popleft()
        inflight.add(address)
        return address

    network_encode = 0.0
    try:
        while processed < max_total_bytes:
            remaining = max_total_bytes - processed
            take = min(buffer_size, remaining)

            # Bounds cover only role-owned, partner, and recovered destinations.
            own_offset_aligned = ((own_offset + 63) // 64) * 64
            partner_offset_aligned = ((partner_offset + 63) // 64) * 64
            own_remaining = own_buf.numel() - own_offset_aligned
            partner_remaining = partner_buf.numel() - partner_offset_aligned
            recovered_remaining = (
                recovered_buffer.numel() - processed
                if recovered_buffer is not None else take
            )
            if rank_in_group == 1:
                destination_remaining = min(own_remaining, recovered_remaining)
            elif rank_in_group == 2:
                destination_remaining = min(partner_remaining, recovered_remaining)
            else:
                destination_remaining = min(own_remaining, partner_remaining)
            take = min(take, destination_remaining)
            if take < 64:
                logger.warning(
                    "ECCHECK two-failures: destination buffers exhausted at %.2f GB",
                    processed / (1024**3),
                )
                break

            # Reserve the receive slot before taking pooled data/encoding resources.
            recv_slot_addr = _get_recv_slot()
            max_inflight = max(max_inflight, len(inflight))

            submitted = False
            try:
                cur_buffer_addr = _get_free_data()
                buffer_ptr = ctypes.cast(cur_buffer_addr, ctypes.POINTER(ctypes.c_uint8))
                buffer_array = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
                src_off = processed
                bytes_to_copy = min(take, max(0, input_buf.numel() - src_off))
                if bytes_to_copy > 0:
                    ctypes.memmove(buffer_array.contents, input_base + src_off, bytes_to_copy)
                if take > bytes_to_copy:
                    ctypes.memset(
                        ctypes.cast(ctypes.addressof(buffer_array.contents) + bytes_to_copy,
                                    ctypes.POINTER(ctypes.c_uint8)),
                        0, take - bytes_to_copy,
                    )

                enc_addr_0 = _get_free_encoding()
                enc_addr_1 = _get_free_encoding()
                own_write_addr = own_base + own_offset_aligned
                partner_write_addr = partner_base + partner_offset_aligned

                if rank_in_group in (0, 1):
                    own_offset = own_offset_aligned + take
                if rank_in_group in (2, 3):
                    partner_offset = partner_offset_aligned + take

                if t_pipeline_net_start is None:
                    t_pipeline_net_start = time()
                native.submit_two_failure_encoding_chunk(
                    data_addr=cur_buffer_addr,
                    size=take,
                    enc_addr_0=enc_addr_0,
                    enc_addr_1=enc_addr_1,
                    recv_addr_1=recv_slot_addr,
                    recv_addr_2=recv_slot_addr,
                    recv_chunk_size=take,
                    own_write_addr=own_write_addr,
                    partner_write_addr=partner_write_addr,
                    recovered_write_addr=(
                        int(recovered_buffer.data_ptr()) + processed
                        if is_failed and recovered_buffer is not None else 0
                    ),
                )
                submitted = True
                submitted_chunks += 1
            except Exception:
                if not submitted and recv_slot_addr in inflight:
                    inflight.remove(recv_slot_addr)
                    free_slots.appendleft(recv_slot_addr)
                raise
            processed += take

        # Sentinels and completion
        if t_pipeline_net_start is None:
            t_pipeline_net_start = time()
        native.submit_two_failure_encoding_sentinels()
        native.wait_for_two_failure_chunk_completion(
            expected_chunks=submitted_chunks, timeout_seconds=120.0
        )
        native.wait_for_encoding_completion()
        release_deadline = monotonic() + 120.0
        while inflight:
            _poll_recv_slot_releases()
            if not inflight:
                break
            if monotonic() >= release_deadline:
                raise RuntimeError(
                    "ECCHECK HW2 timed out waiting for receive ring slots after completion: "
                    f"inflight={len(inflight)}, free={len(free_slots)}, depth={ring_depth}"
                )
            sleep(0.001)
        if len(free_slots) != ring_depth:
            raise RuntimeError(
                "ECCHECK HW2 receive ring did not fully return at cycle end: "
                f"free={len(free_slots)}, depth={ring_depth}"
            )
        logger.debug(
            "ECCHECK HW2 pipeline ring_stall_s=%.3f ring_wait_count=%d max_inflight=%d",
            ring_stall_s, ring_wait_count, max_inflight,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_pipeline_net = time() - t_pipeline_net_start

        network_encode = t_phase1_p2p + t_pipeline_net
        logger.debug(
            f"ECCHECK legacy: two-failure recovery pipeline done in "
            f"{network_encode:.2f}s"
        )

    finally:
        if active_event is not None:
            active_event.clear()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return {
        "network_encode": network_encode,
        "phase1_p2p_s": t_phase1_p2p,
        "pipeline_wall_s": t_pipeline_net,
    }


# ---------------------------------------------------------------------------
# State dict reconstruction
# ---------------------------------------------------------------------------

def _tensor_buffer_as_uint8_view(buffer: torch.Tensor) -> torch.Tensor:
    """View checkpoint buffer as contiguous uint8 without copying pinned storage."""
    buf = buffer.detach()
    if buf.dtype == torch.uint8:
        buf = buf.reshape(-1)
    else:
        buf = buf.reshape(-1).view(torch.uint8)
    if not buf.is_contiguous():
        buf = buf.contiguous()
    return buf


def _dict_to_tensor_info(info: Dict[str, Any]) -> Any:
    """Convert serialized tensor info dict to extract-compatible objects."""
    from types import SimpleNamespace

    dtype_str = info.get("dtype", "torch.float32")
    dtype = (
        getattr(torch, dtype_str.split(".")[-1])
        if isinstance(dtype_str, str) and "." in dtype_str
        else dtype_str
    )
    return SimpleNamespace(
        key=info.get("key", ""),
        offset=info["offset"],
        size_bytes=info["size_bytes"],
        dtype=dtype,
        shape=tuple(info["shape"]),
    )


def _coerce_tensor_infos_for_extract(tensor_infos: List[Any]) -> List[Any]:
    """Normalize tensor_infos to objects with torch.dtype for buffer extract."""
    if not tensor_infos:
        return tensor_infos
    if isinstance(tensor_infos[0], dict):
        return [_dict_to_tensor_info(info) for info in tensor_infos]
    if hasattr(tensor_infos[0], "dtype") and isinstance(tensor_infos[0].dtype, str):
        coerced = []
        for info in tensor_infos:
            dtype_str = info.dtype
            dt = (
                getattr(torch, dtype_str.split(".")[-1])
                if "." in dtype_str
                else torch.float32
            )
            info.dtype = dt
            coerced.append(info)
        return coerced
    return tensor_infos


def _reconstruct_state_dict_from_eccheck_buffer(
    main_payload: Dict[str, Any],
    recovered_buffer: Optional[torch.Tensor],
) -> Dict[str, Any]:
    flat_key_roots = _infer_flat_key_roots(main_payload)

    if recovered_buffer is not None:
        buf = _tensor_buffer_as_uint8_view(recovered_buffer)
    else:
        tensor_buffer = main_payload.get("tensor_buffer")
        if tensor_buffer is None:
            raise RuntimeError(
                "ECCHECK legacy: tensor_buffer missing and no recovered_buffer provided"
            )
        buf = _tensor_buffer_as_uint8_view(tensor_buffer)
    tensor_infos = _coerce_tensor_infos_for_extract(main_payload["tensor_infos"])
    tensor_data = extract_tensors_from_continuous_buffer(buf, tensor_infos)
    decomposed = DecomposedStateDict(
        non_tensor_data=main_payload["non_tensor_data"],
        tensor_infos=tensor_infos,
        tensor_data=tensor_data,
        flat_key_roots=flat_key_roots,
    )
    result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


def _reconstruct_state_dict_from_main_tensor_buffer(
    main_payload: Dict[str, Any],
) -> Dict[str, Any]:
    flat_key_roots = _infer_flat_key_roots(main_payload)
    tb = main_payload["tensor_buffer"]
    buf = tb.detach().contiguous().reshape(-1).view(torch.uint8)
    tensor_infos = _coerce_tensor_infos_for_extract(main_payload["tensor_infos"])
    tensor_data = extract_tensors_from_continuous_buffer(buf, tensor_infos)
    decomposed = DecomposedStateDict(
        non_tensor_data=main_payload["non_tensor_data"],
        tensor_infos=tensor_infos,
        tensor_data=tensor_data,
        flat_key_roots=flat_key_roots,
    )
    result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


# ---------------------------------------------------------------------------
# Metadata-only reconstruction (no distributed)
# ---------------------------------------------------------------------------

def state_dict_from_eccheck_main_metadata_only(
    main_payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Build state_dict from eccheck main file payload without distributed.

    Used when torch.distributed is not initialized (e.g. load_args_from_checkpoint).
    """
    if isinstance(main_payload.get("tensor_buffer"), torch.Tensor):
        result = _reconstruct_state_dict_from_main_tensor_buffer(main_payload)
    else:
        decomposed = DecomposedStateDict(
            non_tensor_data=main_payload["non_tensor_data"],
            tensor_infos=[],
            tensor_data=[],
            flat_key_roots=_infer_flat_key_roots(main_payload),
        )
        result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


# ---------------------------------------------------------------------------
# In-process recovery workspace
# ---------------------------------------------------------------------------


def _eccheck_input_file_signature(checkpoint_dir: Path) -> Tuple[Tuple[str, int, int], ...]:
    files = []
    for path in sorted(checkpoint_dir.glob("eccheck*.pt")):
        stat = path.stat()
        files.append((path.name, int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(files)


def _eccheck_inprocess_bootstrap_key(
    checkpoint_dir: Path, rank: int, world_size: int, args: Any, mode: str,
    rank_in_group: int, cluster_id: int, layout: Dict[str, int],
) -> tuple:
    env_names = (
        "ECCHECK_USE_ASIO", "ECCHECK_BASE_IP", "ECCHECK_INTERFACE",
        "ECCHECK_BASE_PORT", "MASTER_ADDR", "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
    )
    rank_ip_env = tuple(
        sorted((name, value) for name, value in os.environ.items()
               if name.startswith("ECCHECK_RANK_IP_") or name.startswith("ECCHECK_LOCAL_RANK_NIC_"))
    )
    failed_roles = (1, 2) if mode == "HW2" else ((1,) if mode == "SW" else (2,))
    return (
        str(checkpoint_dir.resolve()), _eccheck_input_file_signature(checkpoint_dir),
        rank, world_size, tuple(sorted(layout.items())), cluster_id, rank_in_group,
        int(getattr(args, "eccheck_recovery_cluster", 0)),
        int(getattr(args, "eccheck_rig_remap_offset", 0)), failed_roles, mode,
        bool(getattr(args, "use_rdma", False)),
        (torch.distributed.get_backend() if torch.distributed.is_initialized() else None),
        tuple((name, os.environ.get(name)) for name in env_names), rank_ip_env,
    )


def _metadata_workspace_key(
    bootstrap_key: tuple, manager: ECCHECKManager, rank_metadata: Dict[int, List[TensorMetadata]],
    blocks: Dict[str, Any], recovered_capacity: int,
) -> tuple:
    rank_sizes = tuple(
        (rank, sum(meta.size_bytes for meta in metadata), len(metadata))
        for rank, metadata in sorted(rank_metadata.items())
    )
    return (
        bootstrap_key, rank_sizes, int(manager.eccheck_buffer_size), 64,
        int(blocks.get("pipeline_size", 0)), int(blocks.get("aligned_size", 0)),
        tuple(blocks.get("block_names", ())), int(recovered_capacity),
        manager.eccheck_data_buffers_count, manager.eccheck_encoding_buffers_count,
    )

# ---------------------------------------------------------------------------
# Main load entry point
# ---------------------------------------------------------------------------

def load_eccheck_legacy_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    from megatron.training import get_args

    args = get_args()
    rank_in_group = ECCHECKManager._get_rank_in_group(rank, world_size) if world_size > 1 else 0
    cluster_id = ECCHECKManager._get_cluster_id(rank, world_size) if world_size > 1 else 0
    sw_failure = bool(getattr(args, "use_eccheck_software_failure", False))
    two_failures = bool(getattr(args, "use_eccheck_two_failures", False))
    hw2_ring_depth = _eccheck_hw2_recv_ring_depth() if two_failures else None
    target_cluster = int(getattr(args, "eccheck_recovery_cluster", 0))
    layout = ECCHECKManager._get_group_layout(world_size)
    num_clusters = int(
        layout["clusters"] if layout["mode"] == 1 else layout["num_groups"]
    )
    if target_cluster < 0 or target_cluster >= num_clusters:
        raise ValueError(
            f"ECCHECK recovery cluster {target_cluster} is outside "
            f"[0, {num_clusters - 1}]"
        )
    # Software-failure behavior remains unchanged. HW/HW2 recovery is confined
    # to one four-node cluster; ranks in other clusters load their local state.
    recovery_cluster_active = sw_failure or cluster_id == target_cluster

    load_tensor_buffer = True
    if world_size > 1 and recovery_cluster_active:
        # Ranks rebuilt from EC/P2P data do not need the multi-GB main tensor
        # buffer. In SW mode only rig1 is rebuilt from recovered_buffer;
        # rig0 still needs its local main tensor and sends d1 from partner_buffer.
        if two_failures:
            load_tensor_buffer = rank_in_group not in (1, 2)
        elif sw_failure:
            load_tensor_buffer = rank_in_group != 1
        else:
            load_tensor_buffer = rank_in_group != 2

    inprocess_cache = bool(
        getattr(args, "ft_inprocess_recovery_benchmark", False)
        and getattr(args, "_ft_inprocess_recovery_active", False)
    )
    if two_failures:
        _mode = "HW2"
    elif sw_failure:
        _mode = "SW"
    else:
        _mode = "HW"

    setup_start = time.perf_counter()
    metadata_plan_s = 0.0
    disk_preload_s = 0.0
    alloc_touch_register_s = 0.0
    native_reset_s = 0.0

    if not getattr(args, "use_eccheck", False):
        logger.warning(
            "ECCHECK legacy load: args.use_eccheck is False; enabling for native module init"
        )
        args.use_eccheck = True

    alloc_start = time.perf_counter()
    manager = ECCHECKManager()
    manager.init_eccheck_if_enabled()
    if manager._eccheck_native is None:
        raise RuntimeError("ECCHECK native module is not available in legacy load path")
    alloc_touch_register_s += time.perf_counter() - alloc_start

    bootstrap_key = None
    workspace = None
    if inprocess_cache:
        metadata_start = time.perf_counter()
        bootstrap_key = _eccheck_inprocess_bootstrap_key(
            checkpoint_dir, rank, world_size, args, _mode, rank_in_group,
            cluster_id, layout,
        )
        workspace = manager.find_legacy_inprocess_workspace(bootstrap_key)
        metadata_plan_s += time.perf_counter() - metadata_start

    if workspace is not None:
        main_payload = workspace["main_payload"]
        rank_metadata = workspace["rank_metadata"]
        registry = workspace["registry"]
        total_size = workspace["total_size"]
        blocks = workspace["blocks"]
        source_blocks = workspace.get("source_blocks", {})
        recovered_buffer = workspace["recovered_buffer"]
        cached_recv_buffers = (
            manager.eccheck_hw2_recv_encoding_buffers
            if two_failures
            else manager.eccheck_recv_encoding_buffers
            if sw_failure
            else manager.eccheck_hw1_recv_encoding_buffers
        )
        if workspace.get("recv_buffers") is not cached_recv_buffers:
            raise RuntimeError("ECCHECK cached receive-buffer identity changed")
        if two_failures and manager.eccheck_hw2_recv_ring_depth != hw2_ring_depth:
            raise RuntimeError(
                "ECCHECK cached HW2 receive ring depth mismatch: "
                f"allocated={manager.eccheck_hw2_recv_ring_depth}, requested={hw2_ring_depth}"
            )
        cache_status = "hit"
    else:
        cache_status = "miss"
        metadata_start = time.perf_counter()
        main_payload = _load_eccheck_main_payload(
            checkpoint_dir, rank, world_size, load_tensor_buffer=load_tensor_buffer,
        )
        tensor_infos = main_payload["tensor_infos"]
        local_metadata = _tensor_infos_to_local_metadata(rank, tensor_infos)
        if world_size > 1 and torch.distributed.is_initialized():
            gathered_meta: List[Any] = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_meta, local_metadata)
            rank_metadata = {i: gathered_meta[i] for i in range(world_size)}
        else:
            rank_metadata = {0: local_metadata}
        registry = GlobalMetadataRegistry(
            rank_metadata=rank_metadata, rank_non_tensor_data={}
        )
        total_size = sum(meta.size_bytes for meta in rank_metadata.get(rank, []))
        metadata_plan_s += time.perf_counter() - metadata_start

        if world_size <= 1:
            return _reconstruct_state_dict_from_eccheck_buffer(main_payload, None)

        alloc_start = time.perf_counter()
        if recovery_cluster_active:
            if sw_failure:
                required_block_names = ["partner_buffer"] if rank_in_group == 0 else []
            else:
                required_block_names = ["own_buffer", "partner_buffer"]
            blocks = _allocate_eccheck_blocks_legacy(
                manager, rank_metadata, block_names=required_block_names,
            )
        else:
            blocks = {}
        recovered_buffer: Optional[torch.Tensor] = None
        source_blocks: Dict[str, torch.Tensor] = {}

        if not recovery_cluster_active:
            pass
        elif two_failures:
            pipeline_capacity_bytes = _max_tensor_bytes_from_registry(registry, world_size)
            if rank_in_group == 0:
                local_canonical_required_bytes = _rank_total_bytes(rank_metadata, rank)
                tensor_buffer = main_payload.get("tensor_buffer")
                if not isinstance(tensor_buffer, torch.Tensor):
                    raise RuntimeError("ECCHECK HW2 rig0 requires main tensor_buffer as authoritative d0")
                if tensor_buffer.device.type != "cpu" or not tensor_buffer.is_contiguous():
                    raise RuntimeError(
                        "ECCHECK HW2 rig0 canonical main tensor_buffer must be contiguous CPU storage"
                    )
                d0 = _tensor_buffer_as_uint8_view(tensor_buffer)
                if d0.numel() < local_canonical_required_bytes:
                    raise RuntimeError(
                        "ECCHECK HW2 rig0 canonical main buffer must cover local rank data: "
                        f"required={local_canonical_required_bytes}, have={d0.numel()}"
                    )
                source_tensors = allocate_hugepage_slices(
                    blocks["aligned_size"], 1,
                    fallback_pin_memory=torch.cuda.is_available(), touch_pages=True,
                )
                source_blocks = {"d0": d0, "d1": source_tensors[0]}
                logger.debug(
                    "ECCHECK HW2 rig0 sources: d0 kind=canonical_main, "
                    "allocated_source_blocks=1, local_bytes=%d, pipeline_bytes=%d",
                    local_canonical_required_bytes, pipeline_capacity_bytes,
                )
                if manager.use_rdma and inprocess_cache:
                    manager.register_buffer(source_blocks["d1"])
            elif rank_in_group == 3:
                source_tensors = allocate_hugepage_slices(
                    blocks["aligned_size"], 2,
                    fallback_pin_memory=torch.cuda.is_available(), touch_pages=True,
                )
                source_blocks = {"p3": source_tensors[0], "p2": source_tensors[1]}
                if manager.use_rdma and inprocess_cache:
                    for source in source_blocks.values():
                        manager.register_buffer(source)
            if rank_in_group == 1:
                recovered_buffer = blocks["partner_buffer"]
                total_size = pipeline_capacity_bytes
            elif rank_in_group == 2:
                recovered_buffer = blocks["own_buffer"]
                total_size = pipeline_capacity_bytes
        elif sw_failure:
            if rank_in_group == 0:
                blocks = _allocate_eccheck_blocks_legacy(
                    manager, rank_metadata, block_names=["partner_buffer"],
                )
            elif rank_in_group == 1:
                actual_tensor_bytes = _max_tensor_bytes_from_registry(registry, world_size)
                recovered_buffer = _allocate_recovered_buffer(actual_tensor_bytes, True)
                total_size = actual_tensor_bytes
        elif rank_in_group == 2:
            blocks = _allocate_eccheck_blocks_legacy(manager, rank_metadata)
            recovered_capacity = _max_tensor_bytes_from_registry(registry, world_size)
            recovered_buffer = blocks["own_buffer"]
            total_size = recovered_capacity
        else:
            blocks = _allocate_eccheck_blocks_legacy(manager, rank_metadata)

        if (
            inprocess_cache
            and not sw_failure
            and recovery_cluster_active
            and (
                manager.eccheck_hw2_recv_encoding_buffers is None
                if two_failures else manager.eccheck_hw1_recv_encoding_buffers is None
            )
        ):
            manager.allocate_recv_encoding_buffers_phase2(
                registry,
                hw1_single_physical_buffer=not two_failures,
                hw2_single_physical_buffer=two_failures,
                hw2_ring_depth=hw2_ring_depth if two_failures else None,
            )
        recovered_is_block_alias = recovered_buffer is not None and any(
            recovered_buffer is buffer
            for name, buffer in blocks.items()
            if name in ("own_buffer", "partner_buffer")
        )
        if (
            manager.use_rdma
            and recovered_buffer is not None
            and not two_failures
            and not recovered_is_block_alias
        ):
            manager.register_buffer(recovered_buffer)
        alloc_touch_register_s += time.perf_counter() - alloc_start

        disk_start = time.perf_counter()
        if recovery_cluster_active:
            if two_failures and rank_in_group == 0:
                _copy_block_file_into_tensor(
                    checkpoint_dir, rank, "partner_buffer", source_blocks["d1"]
                )
            elif two_failures and rank_in_group == 3:
                _copy_block_file_into_tensor(
                    checkpoint_dir, rank, "own_buffer", source_blocks["p3"]
                )
                _copy_block_file_into_tensor(
                    checkpoint_dir, rank, "partner_buffer", source_blocks["p2"]
                )
            elif sw_failure and rank_in_group == 0:
                _load_eccheck_blocks_from_disk_into(
                    blocks, checkpoint_dir, rank, rank_in_group, software_failure=True,
                )
            elif not sw_failure and not two_failures and rank_in_group != 2:
                _load_eccheck_blocks_from_disk_into(
                    blocks, checkpoint_dir, rank, rank_in_group, software_failure=False,
                )
        disk_preload_s = time.perf_counter() - disk_start

        if inprocess_cache:
            recovered_capacity = recovered_buffer.numel() if recovered_buffer is not None else 0
            workspace_key = _metadata_workspace_key(
                bootstrap_key, manager, rank_metadata, blocks, recovered_capacity,
            )
            workspace = {
                "main_payload": main_payload, "rank_metadata": rank_metadata,
                "registry": registry, "total_size": total_size, "blocks": blocks,
                "source_blocks": source_blocks, "recovered_buffer": recovered_buffer,
                "recv_buffers": (
                    manager.eccheck_hw2_recv_encoding_buffers
                    if two_failures
                    else manager.eccheck_recv_encoding_buffers
                    if sw_failure
                    else manager.eccheck_hw1_recv_encoding_buffers
                ),
                "workspace_key": workspace_key,
            }
            manager.install_legacy_inprocess_workspace(workspace_key, workspace)

    native_start = time.perf_counter()
    native_prepared = False
    if inprocess_cache and recovery_cluster_active and not sw_failure:
        manager.reset_native_for_inprocess_recovery(10 if two_failures else 2)
        native_prepared = True
    native_reset_s = time.perf_counter() - native_start

    setup_total_s = time.perf_counter() - setup_start
    if inprocess_cache:
        setup_summary = _timing_max_dict({
            "metadata_plan_s": metadata_plan_s,
            "disk_preload_s": disk_preload_s,
            "alloc_touch_register_s": alloc_touch_register_s,
            "native_reset_s": native_reset_s,
            "total_s": setup_total_s,
        })
        if rank == 0:
            logger.info(
                "ECCHECK %s in-process setup cache=%s metadata_plan_s=%.3f "
                "disk_preload_s=%.3f alloc_touch_register_s=%.3f "
                "native_reset_s=%.3f total_s=%.3f",
                "software" if sw_failure else "hardware", cache_status,
                setup_summary["metadata_plan_s"],
                setup_summary["disk_preload_s"],
                setup_summary["alloc_touch_register_s"],
                setup_summary["native_reset_s"], setup_summary["total_s"],
            )

    # Sync all ranks after setup so network timing excludes setup skew.
    barrier_s = _timed_barrier()

    should_time_recovery_to_forward = (
        _mode in ("HW", "HW2")
        and (
            not getattr(args, "ft_inprocess_recovery_benchmark", False)
            or bool(getattr(args, "_ft_inprocess_recovery_active", False))
        )
    )
    if should_time_recovery_to_forward:
        if not recovery_cluster_active:
            recovery_role = "uninvolved"
        elif two_failures:
            recovery_role = "failed" if rank_in_group in (1, 2) else "survivor"
        else:
            recovery_role = "failed" if rank_in_group == 2 else "survivor"
        try:
            from megatron.training.global_vars import start_recovery_to_forward_timer
            start_recovery_to_forward_timer(
                "ECCHECK", "network_recovery", role=recovery_role, rank0_only_max=True,
            )
        except Exception:
            pass

    # === timing: network/encode (C++ P2P or XOR pipeline, excluding setup/copy) ===
    phase1_p2p_s = 0.0
    pipeline_wall_s = 0.0
    if not recovery_cluster_active:
        network_encode = 0.0
    elif two_failures:
        hw2_breakdown = _run_eccheck_two_failures_recovery(
            manager=manager,
            rank=rank,
            world_size=world_size,
            blocks=blocks,
            source_blocks=source_blocks,
            recovered_buffer=recovered_buffer,
            total_size=total_size,
            registry=registry,
            native_prepared=native_prepared,
        )
        network_encode = hw2_breakdown["network_encode"]
        phase1_p2p_s = hw2_breakdown["phase1_p2p_s"]
        pipeline_wall_s = hw2_breakdown["pipeline_wall_s"]
    else:
        network_encode = _run_eccheck_legacy_recovery(
            manager=manager,
            rank=rank,
            world_size=world_size,
            blocks=blocks,
            recv_buffers=None,
            recovered_buffer=recovered_buffer,
            total_size=total_size,
            registry=registry,
            native_prepared=native_prepared,
        )
    if should_time_recovery_to_forward:
        try:
            from megatron.training.global_vars import mark_recovery_to_forward_timer
            mark_recovery_to_forward_timer("eccheck_network_done")
        except Exception:
            pass

    # SW rig1 rebuilds from recovered_buffer; rig0/2/3 use their local main tensor.

    if not recovery_cluster_active:
        rebuild_recovered_buffer = None
    elif sw_failure:
        rebuild_recovered_buffer = (
            recovered_buffer if rank_in_group == 1 else None
        )
    elif two_failures:
        rebuild_recovered_buffer = (
            recovered_buffer if rank_in_group in (1, 2) else None
        )
    else:
        rebuild_recovered_buffer = (
            recovered_buffer if rank_in_group == 2 else None
        )

    t_rebuild = time.time()
    state_dict = _reconstruct_state_dict_from_eccheck_buffer(
        main_payload,
        recovered_buffer=rebuild_recovered_buffer,
    )
    rebuild_sd = time.time() - t_rebuild
    native_timing = _native_ft_timing(manager._eccheck_native)

    timings = {
        "total": network_encode + rebuild_sd,
        "network_encode": network_encode,
        "net_s": native_timing["net_s"],
        "encode_s": native_timing["encode_s"],
        "decode_s": native_timing["decode_s"],
        "phase1_p2p_s": phase1_p2p_s,
        "pipeline_wall_s": pipeline_wall_s,
        "rebuild_sd": rebuild_sd,
        "barrier": barrier_s,
        "rebuild_from_recovered": float(rebuild_recovered_buffer is not None),
        "recovery_cluster_active": float(recovery_cluster_active),
        "recovery_cluster": float(target_cluster),
    }
    from megatron.training.global_vars import set_ft_load_timing_context
    set_ft_load_timing_context("ECCHECK", _mode, timings)

    load_log = dict(timings)
    load_log["mode"] = _mode
    logger.debug(
        "ECCHECK load timing (%(mode)s local): e2e_s=%(total).2fs "
        "network_encode_s=%(network_encode).2fs "
        "rebuild_sd_s=%(rebuild_sd).2fs barrier_s=%(barrier).2fs",
        load_log,
    )

    prebenchmark_software_load = bool(
        sw_failure
        and getattr(args, "ft_inprocess_recovery_benchmark", False)
        and not getattr(args, "_ft_inprocess_recovery_active", False)
    )
    if (
        prebenchmark_software_load
        and manager.use_rdma
        and recovered_buffer is not None
    ):
        manager.unregister_buffer(recovered_buffer)
        if rank == 0:
            logger.info(
                "ECCHECK software preload: retained native P2P topology and "
                "released temporary recovery-buffer registration"
            )

    if should_time_recovery_to_forward:
        try:
            from megatron.training.global_vars import mark_recovery_to_forward_timer
            mark_recovery_to_forward_timer("eccheck_rebuild_done")
            mark_recovery_to_forward_timer("eccheck_final_barrier_start")
        except Exception:
            pass

    skip_final_barrier = bool(getattr(args, "ft_inprocess_recovery_benchmark", False))
    # Stop C++ load workers so they don't interfere with subsequent training.
    # Synchronize teardown so early ranks do not release RDMA resources while peers
    # are still draining their final load P2P completions. In in-process benchmark
    # mode cleanup is deferred, so skip this tail barrier.
    if world_size > 1 and torch.distributed.is_initialized() and not skip_final_barrier:
        torch.distributed.barrier()
        if should_time_recovery_to_forward:
            try:
                from megatron.training.global_vars import mark_recovery_to_forward_timer
                mark_recovery_to_forward_timer("eccheck_final_barrier_done")
            except Exception:
                pass
    elif skip_final_barrier and should_time_recovery_to_forward:
        try:
            from megatron.training.global_vars import mark_recovery_to_forward_timer
            mark_recovery_to_forward_timer("eccheck_final_barrier_skipped_inprocess")
        except Exception:
            pass

    if should_time_recovery_to_forward:
        try:
            from megatron.training.global_vars import mark_recovery_to_forward_timer
            mark_recovery_to_forward_timer("eccheck_load_return")
        except Exception:
            pass

    # Defer manager.cleanup() until after load_state_dict H2D in checkpointing.py.
    # Early cudaHostUnregister makes rebuild_sd views non-pinned and under-reports h2d_s.
    return state_dict
