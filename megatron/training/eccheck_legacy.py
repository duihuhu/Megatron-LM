# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""LEGACY checkpoint path for ECCHECK (XOR-based erasure coding with 4-rank groups).

Saves/loads via torch.save / torch.load with .pt files (same pattern as
eclatin_legacy.py and ecnaive_legacy.py), reusing the shared ECCHECKManager
singleton and its C++ native module.
"""

import ctypes
import queue
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
    extract_tensors_from_continuous_buffer,
    reconstruct_state_dict,
)

logger = getLogger(__name__)

_FORMAT = "eccheck_torch_legacy"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


def _infer_flat_key_roots(main_payload: Dict[str, Any]) -> Set[str]:
    if "flat_key_roots" in main_payload:
        return set(main_payload["flat_key_roots"])
    flat_key_roots: Set[str] = set()
    for info in main_payload.get("tensor_infos", []):
        key = info.key if isinstance(info, dict) else getattr(info, "key", "")
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
) -> Dict[str, Any]:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    own_total_size = sum(meta.size_bytes for meta in rank_metadata.get(rank, []))

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

    own_buffer, partner_buffer = allocate_hugepage_slices(
        aligned_size, 2, touch_pages=True,
    )

    if manager.use_rdma:
        manager.register_buffer(own_buffer)
        manager.register_buffer(partner_buffer)

    return {
        "own_buffer": own_buffer,
        "partner_buffer": partner_buffer,
        "actual_size": own_total_size,
        "pipeline_size": max_total_bytes,
        "aligned_size": aligned_size,
        "block_names": ["own_buffer", "partner_buffer"],
    }


# ---------------------------------------------------------------------------
# Encoding pipeline (save)
# ---------------------------------------------------------------------------

def _encode_eccheck_with_native(
    manager: ECCHECKManager,
    tensor_buffer: torch.Tensor,
    actual_data_bytes: int,
    blocks: Dict[str, Any],
) -> None:
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

    def _get_free_data_buffer():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_data_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECCHECK legacy: timeout waiting for free data buffer")
            return free_data_queue.get()

    def _get_free_encoding_buffer():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_encoding_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECCHECK legacy: timeout waiting for free encoding buffer")
            return free_encoding_queue.get()

    def _get_free_parity_buffer():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_parity_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECCHECK legacy: timeout waiting for free parity buffer")
            return free_parity_queue.get()

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

    try:
        while src_pos < pipeline_total_bytes:
            remaining = pipeline_total_bytes - src_pos
            take = min(buffer_size, remaining)

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

            native.submit_data_for_encoding_thread1(
                cur_buffer_addr, take, enc_addr1, recv_addr_1,
                recv_chunk_size, parity_addr1, own_write_addr, partner_write_addr,
            )
            native.submit_data_for_encoding_thread2(
                cur_buffer_addr, take, enc_addr2, recv_addr_2,
                recv_chunk_size, parity_addr2, own_write_addr, partner_write_addr,
            )

            src_pos += take

        # Sentinels
        native.submit_data_for_encoding_thread1(0, 0, 0, 0, 0, 0, 0, 0)
        native.submit_data_for_encoding_thread2(0, 0, 0, 0, 0, 0, 0, 0)

        native.wait_for_encoding_completion()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        if active_event is not None:
            active_event.clear()


# ---------------------------------------------------------------------------
# Save .pt files
# ---------------------------------------------------------------------------

def _save_eccheck_pt_files(
    checkpoint_name: str,
    rank: int,
    non_tensor_data: Dict[str, Any],
    tensor_infos: List[Any],
    blocks: Dict[str, Any],
    full_tensor_buffer: torch.Tensor,
    flat_key_roots: Optional[Set[str]] = None,
) -> None:
    checkpoint_path = Path(checkpoint_name)
    checkpoint_dir = checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    block_files = {
        "own_buffer": f"eccheck_block_rank{rank}_own_buffer.pt",
        "partner_buffer": f"eccheck_block_rank{rank}_partner_buffer.pt",
    }

    main_file = checkpoint_dir / f"eccheck_main_rank{rank}.pt"
    torch.save(
        {
            "version": 1,
            "format": _FORMAT,
            "rank": rank,
            "non_tensor_data": non_tensor_data,
            "tensor_infos": tensor_infos,
            "tensor_buffer": full_tensor_buffer.contiguous().view(torch.uint8).clone(),
            "actual_tensor_size": blocks["actual_size"],
            "pipeline_total_bytes": blocks["pipeline_size"],
            "aligned_block_size": blocks["aligned_size"],
            "flat_key_roots": list(flat_key_roots) if flat_key_roots else [],
            "block_files": block_files,
        },
        main_file,
    )

    for name in ("own_buffer", "partner_buffer"):
        block_file = checkpoint_dir / f"eccheck_block_rank{rank}_{name}.pt"
        block_tensor = blocks[name].detach().contiguous().clone()
        torch.save(
            {
                "version": 1,
                "format": _FORMAT,
                "rank": rank,
                "block_name": name,
                "tensor": block_tensor,
            },
            block_file,
        )


# ---------------------------------------------------------------------------
# Main save entry point
# ---------------------------------------------------------------------------

def save_eccheck_legacy_checkpoint(
    state_dict: Dict[str, Any], checkpoint_name: str
) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    manager = ECCHECKManager()
    manager.init_eccheck_if_enabled()
    if manager._eccheck_native is None:
        raise RuntimeError("ECCHECK native module is not available in legacy save path")

    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes

    safety_margin = max(int(total_tensor_size * 0.01), manager.eccheck_buffer_size)
    tensor_buffer = allocate_hugepage_tensor(
        total_tensor_size + safety_margin,
        fallback_pin_memory=manager.eccheck_pin_memory and torch.cuda.is_available(),
    )
    tensor_buffer.zero_()

    offset = 0
    local_tensor_metadata: List[TensorMetadata] = []
    for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
        tensor_bytes = info.size_bytes
        tensor_bytes_view = _cpu_uint8_view(tensor)
        if tensor_bytes_view.numel() != tensor_bytes:
            raise RuntimeError(
                f"ECCHECK legacy save: tensor bytes mismatch for {info.key}, "
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
                global_offset=tuple(info.global_offset) if info.global_offset else tuple(),
                shard_index=info.shard_index if info.shard_index is not None else 0,
                chunk_type="data",
                target_rank=rank,
                source_rank=rank,
            )
        )
        offset += tensor_bytes

    rank_metadata, _ = _build_global_registry(
        local_tensor_metadata, decomposed.non_tensor_data
    )
    blocks = _allocate_eccheck_blocks_legacy(manager, rank_metadata)

    # Allocate recv encoding buffers now that we know peer data sizes
    registry = GlobalMetadataRegistry(
        rank_metadata=rank_metadata, rank_non_tensor_data={}
    )
    if manager.eccheck_recv_encoding_buffers is None:
        manager.eccheck_recv_encoding_buffers = (
            manager.allocate_recv_encoding_buffers_phase2(registry)
        )

    if manager.use_rdma:
        manager.register_buffer(tensor_buffer)

    logger.info(
        f"ECCHECK legacy save: rank {rank} encoding "
        f"{blocks['pipeline_size'] / (1024**3):.2f} GB pipeline "
        f"(actual: {total_tensor_size / (1024**3):.2f} GB)"
    )

    _encode_eccheck_with_native(
        manager=manager,
        tensor_buffer=tensor_buffer,
        actual_data_bytes=total_tensor_size,
        blocks=blocks,
    )

    full_tensor_buffer = tensor_buffer[:total_tensor_size].detach().clone()

    _save_eccheck_pt_files(
        checkpoint_name=checkpoint_name,
        rank=rank,
        non_tensor_data=decomposed.non_tensor_data,
        tensor_infos=decomposed.tensor_infos,
        blocks=blocks,
        full_tensor_buffer=full_tensor_buffer,
    )

    if world_size > 1:
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_eccheck_main_payload(
    checkpoint_dir: Path, rank: int, world_size: int
) -> Dict[str, Any]:
    main_path = checkpoint_dir / f"eccheck_main_rank{rank}.pt"
    local_payload: Optional[Dict[str, Any]] = None
    if main_path.is_file():
        local_payload = torch.load(main_path, map_location="cpu", weights_only=False)

    if world_size <= 1 or not torch.distributed.is_initialized():
        if local_payload is None:
            raise FileNotFoundError(f"ECCHECK legacy: missing main file {main_path}")
        return local_payload

    gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_payload)

    chosen = gathered[rank]
    if chosen is None:
        raise FileNotFoundError(
            f"ECCHECK legacy: eccheck_main_rank{rank}.pt missing on all ranks under {checkpoint_dir}"
        )
    return chosen


def _copy_block_file_into_tensor(
    checkpoint_dir: Path, rank: int, block_name: str, dest: torch.Tensor
) -> None:
    block_path = checkpoint_dir / f"eccheck_block_rank{rank}_{block_name}.pt"
    if not block_path.is_file():
        raise FileNotFoundError(f"ECCHECK legacy load: missing block file {block_path}")
    payload = torch.load(block_path, map_location="cpu", weights_only=False)
    src = payload["tensor"].contiguous().view(torch.uint8).reshape(-1)
    dst = dest.contiguous().view(-1)
    n = min(src.numel(), dst.numel())
    dst[:n].copy_(src[:n])


def _load_eccheck_blocks_from_disk_into(
    blocks: Dict[str, Any],
    checkpoint_dir: Path,
    rank: int,
    rank_in_group: int,
    software_failure: bool = False,
) -> None:
    """Load block .pt files into pre-allocated P2P buffers.

    Recovery layout:
    - rank_in_group 2 (failed): normal recovery path loads nothing here
      (blocks allocated empty, data comes via network). Software failure
      path: rank_in_group 2 loads own_buffer from disk (its own saved data).
    - rank_in_group 0: loads partner_buffer (received from rank1 during save)
    - rank_in_group 1: loads own_buffer (its own data)
    - rank_in_group 3: loads own_buffer + partner_buffer (its own data + received)
    """
    if software_failure and rank_in_group == 2:
        # Software failure: rank2 reads own data from disk (no network recovery needed)
        _copy_block_file_into_tensor(
            checkpoint_dir, rank, "own_buffer", blocks["own_buffer"]
        )
        return

    if rank_in_group == 2:
        # Hardware failure: this rank's data needs network recovery, skip disk load
        return

    if rank_in_group == 0:
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
# Recovery pipeline (load)
# ---------------------------------------------------------------------------

def _allocate_eccheck_load_recv_buffers(
    manager: ECCHECKManager,
    registry: GlobalMetadataRegistry,
) -> Dict[str, torch.Tensor]:
    """Allocate 6 recv buffers for rank_in_group=2 hardware failure recovery.

    Buffer names match the C++ load_recover layout:
    - rank0_data2, rank0_parity2  (from rank_in_group 0)
    - rank1_data1, rank1_parity1  (from rank_in_group 1)
    - rank3_data1, rank3_data2    (from rank_in_group 3)
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    max_total_bytes = _max_tensor_bytes_from_registry(registry, world_size)
    aligned_size = (
        (max_total_bytes + manager.eccheck_buffer_size - 1)
        // manager.eccheck_buffer_size
    ) * manager.eccheck_buffer_size

    recv_names = [
        "rank0_data2", "rank0_parity2",
        "rank1_data1", "rank1_parity1",
        "rank3_data1", "rank3_data2",
    ]
    slices = allocate_hugepage_slices(aligned_size, len(recv_names), touch_pages=True)
    recv_buffers = {name: slices[i] for i, name in enumerate(recv_names)}

    if manager.use_rdma:
        for t in slices:
            manager.register_buffer(t)

    logger.info(
        f"ECCHECK legacy load: [Rank {rank}] Allocated {len(recv_names)} recv buffers "
        f"({aligned_size / (1024**3):.2f} GB each, total={len(recv_names) * aligned_size / (1024**3):.2f} GB)"
    )
    return recv_buffers


def _run_eccheck_legacy_recovery(
    manager: ECCHECKManager,
    rank: int,
    world_size: int,
    blocks: Dict[str, Any],
    recv_buffers: Optional[Dict[str, torch.Tensor]],
    recovered_buffer: Optional[torch.Tensor],
    total_size: int,
    registry: GlobalMetadataRegistry,
) -> None:
    """Drive C++ recovery for ECCHECK legacy load.

    Mirrors the send/recv layout from the modern path but uses pre-loaded
    .pt block files instead of mmap'd .distcp files.
    """
    from time import time

    native = manager._eccheck_native
    if native is None:
        raise RuntimeError("ECCHECK native module is not initialized")

    from megatron.training import get_args as _get_args
    args = _get_args()
    software_failure = bool(getattr(args, "use_eccheck_software_failure", False))

    rank_in_group = manager._get_rank_in_group(rank, world_size)

    # ---- software failure path (rank_in_group 1 / 2) ----
    if software_failure:
        if rank_in_group == 2:
            if recovered_buffer is None:
                raise RuntimeError(
                    "ECCHECK legacy load: software failure path needs recovered_buffer"
                )
            start_t = time()
            actual_tensor_bytes = _max_tensor_bytes_from_registry(registry, world_size)
            own_buf = blocks["own_buffer"].contiguous().view(torch.uint8).reshape(-1)
            if recovered_buffer.numel() >= total_size:
                n_copy = min(actual_tensor_bytes, total_size)
                recovered_buffer[:n_copy].copy_(own_buf[:n_copy])
            logger.info(
                f"ECCHECK legacy load: rank_in_group 2 software recovery done "
                f"in {time() - start_t:.2f}s"
            )
        else:
            logger.info(
                f"ECCHECK legacy load: rank_in_group {rank_in_group} no-op for software failure"
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    # ---- hardware failure path (rank_in_group 2) ----
    failed_rank = 2
    native.set_load_mode(True, failed_rank)
    logger.info(f"ECCHECK legacy load: set load mode (failed_rank={failed_rank})")

    # Get network config and rank2 IP
    net_config = manager._get_eccheck_network_config(rank, world_size)
    rank_ips = net_config.get("rank_ips", {})

    # Find rank_in_group=2 in our EC group
    group_id = manager._get_group_id(rank, world_size)
    rank2_global_rank = None
    for r in range(world_size):
        if (manager._get_rank_in_group(r, world_size) == 2
                and manager._get_group_id(r, world_size) == group_id):
            rank2_global_rank = r
            break
    if rank2_global_rank is None:
        raise RuntimeError("ECCHECK legacy load: cannot locate rank_in_group=2 in our group")
    rank2_ip = rank_ips.get(rank2_global_rank, net_config["my_ip"])

    # Build load connection ports following ECLATIN's pattern
    base_port = net_config["base_port"]
    load_recv_rank0_data2_port = base_port + rank2_global_rank * 6 + 2
    load_recv_rank0_parity2_port = base_port + rank2_global_rank * 6 + 3
    load_recv_rank1_data1_port = base_port + rank2_global_rank * 6 + 0
    load_recv_rank1_parity1_port = base_port + rank2_global_rank * 6 + 1
    load_recv_rank3_data1_port = base_port + rank2_global_rank * 6 + 4
    load_recv_rank3_data2_port = base_port + rank2_global_rank * 6 + 5

    if rank_in_group == 2:
        logger.info("ECCHECK legacy load: rank_in_group 2 init load accept connections")
        native.init_load_connections(
            rank_in_group,
            rank2_ip,
            load_recv_rank0_data2_port,
            load_recv_rank0_parity2_port,
            load_recv_rank1_data1_port,
            load_recv_rank1_parity1_port,
            load_recv_rank3_data1_port,
            load_recv_rank3_data2_port,
        )

    torch.distributed.barrier()

    if rank_in_group != 2:
        logger.info(
            f"ECCHECK legacy load: rank_in_group {rank_in_group} connecting load send sockets"
        )
        native.init_load_connections(
            rank_in_group,
            rank2_ip,
            load_recv_rank0_data2_port,
            load_recv_rank0_parity2_port,
            load_recv_rank1_data1_port,
            load_recv_rank1_parity1_port,
            load_recv_rank3_data1_port,
            load_recv_rank3_data2_port,
        )

    native.wait_for_load_connections(timeout_seconds=30)
    torch.distributed.barrier()

    start_t = time()
    aligned_block_size = blocks["own_buffer"].numel()

    if rank_in_group == 2:
        if recv_buffers is None or recovered_buffer is None:
            raise RuntimeError(
                "ECCHECK legacy load: rank_in_group 2 needs recv_buffers and recovered_buffer"
            )
        required_keys = [
            "rank0_data2", "rank0_parity2",
            "rank1_data1", "rank1_parity1",
            "rank3_data1", "rank3_data2",
        ]
        missing = [k for k in required_keys if k not in recv_buffers]
        if missing:
            raise RuntimeError(f"ECCHECK legacy load: missing recv buffers {missing}")

        recv_addrs = {k: int(recv_buffers[k].data_ptr()) for k in required_keys}
        out_own = int(blocks["own_buffer"].data_ptr())
        out_partner = int(blocks["partner_buffer"].data_ptr())

        native.load_recover(
            recv_addrs["rank0_data2"],
            recv_addrs["rank0_parity2"],
            recv_addrs["rank1_data1"],
            recv_addrs["rank1_parity1"],
            recv_addrs["rank3_data1"],
            recv_addrs["rank3_data2"],
            out_own,
            out_partner,
            aligned_block_size,
        )

        actual_tensor_bytes = _max_tensor_bytes_from_registry(registry, world_size)
        own_flat = blocks["own_buffer"].contiguous().view(torch.uint8).reshape(-1)
        if recovered_buffer.numel() >= total_size:
            n_copy = min(actual_tensor_bytes, total_size)
            recovered_buffer[:n_copy].copy_(own_flat[:n_copy])
        logger.info(
            f"ECCHECK legacy load: rank_in_group 2 recovery done in {time() - start_t:.4f}s"
        )

    elif rank_in_group == 0:
        partner_base = int(blocks["partner_buffer"].data_ptr())
        native.load_send_blocks(
            "rank0_data2", partner_base,
            "rank0_parity2", partner_base,
            aligned_block_size,
        )
    elif rank_in_group == 1:
        own_base = int(blocks["own_buffer"].data_ptr())
        native.load_send_blocks(
            "rank1_data1", own_base,
            "rank1_parity1", own_base,
            aligned_block_size,
        )
    elif rank_in_group == 3:
        own_base = int(blocks["own_buffer"].data_ptr())
        partner_base = int(blocks["partner_buffer"].data_ptr())
        native.load_send_blocks(
            "rank3_data1", own_base,
            "rank3_data2", partner_base,
            aligned_block_size,
        )
    else:
        raise RuntimeError(
            f"ECCHECK legacy load: unexpected rank_in_group={rank_in_group}"
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# State dict reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_state_dict_from_eccheck_buffer(
    main_payload: Dict[str, Any],
    recovered_buffer: Optional[torch.Tensor],
) -> Dict[str, Any]:
    flat_key_roots = _infer_flat_key_roots(main_payload)

    if recovered_buffer is not None:
        buf = recovered_buffer.detach().contiguous().reshape(-1).view(torch.uint8)
    else:
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


def _reconstruct_state_dict_from_main_tensor_buffer(
    main_payload: Dict[str, Any],
) -> Dict[str, Any]:
    flat_key_roots = _infer_flat_key_roots(main_payload)
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
        return _reconstruct_state_dict_from_main_tensor_buffer(main_payload)
    decomposed = DecomposedStateDict(
        non_tensor_data=main_payload["non_tensor_data"],
        tensor_infos=[],
        tensor_data=[],
        flat_key_roots=_infer_flat_key_roots(main_payload),
    )
    return reconstruct_state_dict(decomposed)


# ---------------------------------------------------------------------------
# Main load entry point
# ---------------------------------------------------------------------------

def load_eccheck_legacy_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    main_payload = _load_eccheck_main_payload(checkpoint_dir, rank, world_size)

    from megatron.training import get_args

    args = get_args()
    if not getattr(args, "use_eccheck", False):
        logger.warning(
            "ECCHECK legacy load: args.use_eccheck is False; enabling for native module init"
        )
        args.use_eccheck = True

    manager = ECCHECKManager()
    manager.init_eccheck_if_enabled()
    if manager._eccheck_native is None:
        raise RuntimeError("ECCHECK native module is not available in legacy load path")

    tensor_infos = main_payload["tensor_infos"]
    local_metadata = _tensor_infos_to_local_metadata(rank, tensor_infos)

    if world_size > 1 and torch.distributed.is_initialized():
        gathered_meta: List[Any] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_meta, local_metadata)
        rank_metadata = {i: gathered_meta[i] for i in range(world_size)}
    else:
        rank_metadata = {0: local_metadata}

    registry = GlobalMetadataRegistry(rank_metadata=rank_metadata, rank_non_tensor_data={})

    if world_size <= 1:
        return _reconstruct_state_dict_from_eccheck_buffer(main_payload, None)

    blocks = _allocate_eccheck_blocks_legacy(manager, rank_metadata)

    rank_in_group = manager._get_rank_in_group(rank, world_size)
    sw_failure = bool(getattr(args, "use_eccheck_software_failure", False))

    recv_buffers: Optional[Dict[str, torch.Tensor]] = None
    recovered_buffer: Optional[torch.Tensor] = None
    total_size = sum(meta.size_bytes for meta in rank_metadata.get(rank, []))

    if rank_in_group == 2:
        if sw_failure:
            _load_eccheck_blocks_from_disk_into(
                blocks, checkpoint_dir, rank, rank_in_group, software_failure=True,
            )
        else:
            recv_buffers = _allocate_eccheck_load_recv_buffers(manager, registry)
        pin = torch.cuda.is_available() and getattr(manager, "eccheck_pin_memory", False)
        recovered_buffer = torch.empty(total_size, dtype=torch.uint8, pin_memory=pin)
    else:
        _load_eccheck_blocks_from_disk_into(
            blocks, checkpoint_dir, rank, rank_in_group, software_failure=False,
        )

    if manager.use_rdma:
        for t in [blocks["own_buffer"], blocks["partner_buffer"]]:
            manager.register_buffer(t)
        if recv_buffers is not None:
            for t in recv_buffers.values():
                manager.register_buffer(t)
        if recovered_buffer is not None:
            manager.register_buffer(recovered_buffer)

    _run_eccheck_legacy_recovery(
        manager=manager,
        rank=rank,
        world_size=world_size,
        blocks=blocks,
        recv_buffers=recv_buffers,
        recovered_buffer=recovered_buffer,
        total_size=total_size,
        registry=registry,
    )

    state_dict = _reconstruct_state_dict_from_eccheck_buffer(
        main_payload,
        recovered_buffer=recovered_buffer if rank_in_group == 2 else None,
    )

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return state_dict
