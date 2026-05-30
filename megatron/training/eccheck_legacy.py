# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""LEGACY checkpoint path for ECCHECK (XOR-based erasure coding with 4-rank groups).

Saves/loads via torch.save / torch.load with .pt files (same pattern as
eclatin_legacy.py and ecnaive_legacy.py), reusing the shared ECCHECKManager
singleton and its C++ native module.
"""

import ctypes
import queue
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
    extract_tensors_from_continuous_buffer,
    flatten_optimizer_fp32_params,
    reconstruct_state_dict,
    unflatten_optimizer_fp32_params,
)

logger = getLogger(__name__)

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
        // manager.eccheck_buffer_size + 1
    ) * manager.eccheck_buffer_size

    own_buffer, partner_buffer = manager.allocate_preallocated_blocks(2, aligned_size)

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

            # Bounds check: recv buffers (prevent alignment-padding overflow)
            recv_offset_1_aligned = ((recv_offset_1 + 63) // 64) * 64
            recv_offset_2_aligned = ((recv_offset_2 + 63) // 64) * 64
            recv_rem_1 = recv_buffer_thread1.numel() - recv_offset_1_aligned
            recv_rem_2 = recv_buffer_thread2.numel() - recv_offset_2_aligned
            max_recv_space = min(recv_rem_1, recv_rem_2)
            if take > max_recv_space:
                if max_recv_space < 64:
                    logger.warning(
                        f"ECCHECK legacy: recv buffers exhausted "
                        f"(rem1={recv_rem_1}, rem2={recv_rem_2}), stopping at "
                        f"{src_pos / (1024**3):.2f} GB / {pipeline_total_bytes / (1024**3):.2f} GB"
                    )
                    break
                take = max_recv_space
                recv_chunk_size = take

            # Bounds check: P2P buffers
            own_offset_aligned = ((own_offset + 63) // 64) * 64
            partner_offset_aligned = ((partner_offset + 63) // 64) * 64
            p2p_rem_own = own_buffer.numel() - own_offset_aligned
            p2p_rem_partner = partner_buffer.numel() - partner_offset_aligned
            max_p2p_space = min(p2p_rem_own, p2p_rem_partner)
            if take > max_p2p_space:
                if max_p2p_space < 64:
                    logger.warning(
                        f"ECCHECK legacy: P2P buffers exhausted "
                        f"(rem_own={p2p_rem_own}, rem_partner={p2p_rem_partner}), stopping at "
                        f"{src_pos / (1024**3):.2f} GB / {pipeline_total_bytes / (1024**3):.2f} GB"
                    )
                    break
                take = max_p2p_space
                recv_chunk_size = take

            recv_addr_1 = recv_base_1 + recv_offset_1_aligned
            recv_addr_2 = recv_base_2 + recv_offset_2_aligned
            recv_offset_1 = recv_offset_1_aligned + recv_chunk_size
            recv_offset_2 = recv_offset_2_aligned + recv_chunk_size

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
    from megatron.training.legacy_io_utils import write_raw_checkpoint, write_raw_block, MAGIC_ECCHECK, MAGIC_BLOCK

    # Pre-serialize metadata + prepare memoryview for main file
    import pickle as _pickle
    meta1 = _pickle.dumps(non_tensor_data)
    meta2 = _pickle.dumps(tensor_infos)
    extra = _pickle.dumps({
        "version": 1, "format": _FORMAT, "rank": rank,
        "actual_tensor_size": blocks["actual_size"],
        "pipeline_total_bytes": blocks["pipeline_size"],
        "aligned_block_size": blocks["aligned_size"],
        "flat_key_roots": list(flat_key_roots) if flat_key_roots else [],
        "block_files": block_files,
    })
    buf = full_tensor_buffer[: blocks["actual_size"]]
    if not buf.is_contiguous():
        buf = buf.contiguous()
    if buf.device.type != "cpu":
        buf = buf.to("cpu")
    main_mv = memoryview(buf.numpy())

    # Pre-prepare block memoryviews
    block_names = ("own_buffer", "partner_buffer")
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
            futs.append(ex.submit(write_block_prepared,
                                  str(block_file), MAGIC_BLOCK,
                                  block_mvs[name], blocks[name].numel()))
        for f in futs:
            f.result()




# ---------------------------------------------------------------------------
# Main save entry point
# ---------------------------------------------------------------------------

def save_eccheck_legacy_checkpoint(
    state_dict: Dict[str, Any], checkpoint_name: str
) -> None:
    t0 = time.time()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    manager = ECCHECKManager()
    manager.init_eccheck_if_enabled()
    if manager._eccheck_native is None:
        raise RuntimeError("ECCHECK native module is not available in legacy save path")

    flatten_optimizer_fp32_params(state_dict)
    t0 = time.time()
    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    logger.info(f"ECCHECK save timing: decompose {time.time()-t0:.3f}s")

    start_time = t0 = time.time()
    safety_margin = max(int(total_tensor_size * 0.01), manager.eccheck_buffer_size)
    manager.allocate_preallocated_buffer(total_tensor_size + safety_margin)
    tensor_buffer = manager.preallocated_cpu_buffer

    offset = 0
    local_tensor_metadata: List[TensorMetadata] = []
    d2h_stream = torch.cuda.Stream()
    with torch.cuda.stream(d2h_stream):
        for i, (info, tensor) in enumerate(zip(decomposed.tensor_infos, decomposed.tensor_data)):
            tensor_bytes = info.size_bytes
            tensor_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
            if tensor_view.numel() != tensor_bytes:
                raise RuntimeError(
                    f"ECCHECK legacy save: tensor bytes mismatch for {info.key}, "
                    f"expected={tensor_bytes}, got={tensor_view.numel()}"
                )
            tensor_buffer[offset : offset + tensor_bytes].copy_(tensor_view, non_blocking=True)
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
        decomposed.tensor_data[i] = None  # free GPU tensor ref immediately
    d2h_stream.synchronize()

    del decomposed.tensor_data  # drop remaining refs
    logger.info(f"ECCHECK save timing: D2H+copy {time.time()-t0:.3f}s")

    t0 = time.time()
    # Only tensor metadata needed for block sizing; non_tensor_data (~250MB)
    # is exchanged by all_gather_object but never consumed here.  Pass an empty
    # dict to avoid wasting 5+ seconds on unnecessary exchange.
    rank_metadata, _ = _build_global_registry(local_tensor_metadata, {})
    logger.info(f"ECCHECK save timing: metadata exchange {time.time()-t0:.3f}s")

    t0 = time.time()
    blocks = _allocate_eccheck_blocks_legacy(manager, rank_metadata)
    logger.info(f"ECCHECK save timing: block alloc {time.time()-t0:.3f}s")

    # Allocate recv encoding buffers now that we know peer data sizes
    registry = GlobalMetadataRegistry(
        rank_metadata=rank_metadata, rank_non_tensor_data={}
    )
    t0 = time.time()
    if manager.eccheck_recv_encoding_buffers is None:
        manager.eccheck_recv_encoding_buffers = (
            manager.allocate_recv_encoding_buffers_phase2(registry)
        )
    logger.info(f"ECCHECK save timing: recv buf alloc {time.time()-t0:.3f}s")

    t0 = time.time()
    if manager.use_rdma:
        manager.register_buffer(tensor_buffer)
    logger.info(f"ECCHECK save timing: RDMA reg {time.time()-t0:.3f}s")

    logger.info(
        f"ECCHECK legacy save: rank {rank} encoding "
        f"{blocks['pipeline_size'] / (1024**3):.2f} GB pipeline "
        f"(actual: {total_tensor_size / (1024**3):.2f} GB)"
    )

    t0 = time.time()
    _encode_eccheck_with_native(
        manager=manager,
        tensor_buffer=tensor_buffer,
        actual_data_bytes=total_tensor_size,
        blocks=blocks,
    )
    logger.info(f"ECCHECK save timing: encode {time.time()-t0:.3f}s")
    torch.distributed.barrier()
    logger.info(f"ECCHECK legacy save: done in {time.time() - start_time:.2f}s")

    _save_eccheck_pt_files(
        checkpoint_name=checkpoint_name,
        rank=rank,
        non_tensor_data=decomposed.non_tensor_data,
        tensor_infos=decomposed.tensor_infos,
        blocks=blocks,
        full_tensor_buffer=tensor_buffer[:total_tensor_size],
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
        from megatron.training.legacy_io_utils import is_raw_format, read_raw_checkpoint, MAGIC_ECCHECK
        if is_raw_format(str(main_path), MAGIC_ECCHECK):
            local_payload = read_raw_checkpoint(str(main_path), MAGIC_ECCHECK)
        else:
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
    from megatron.training.legacy_io_utils import is_raw_format, read_raw_block, MAGIC_BLOCK
    if is_raw_format(str(block_path), MAGIC_BLOCK):
        src = read_raw_block(str(block_path), MAGIC_BLOCK)
    else:
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
    two_failures: bool = False,
) -> None:
    """Load block .pt files into pre-allocated P2P buffers.

    Normal recovery layout:
    - rank_in_group 2: loads nothing (data comes via network XOR recovery)
    - rank_in_group 0: loads partner_buffer (received from rank1 during save)
    - rank_in_group 1: loads own_buffer (its own data)
    - rank_in_group 3: loads own_buffer + partner_buffer (its own data + received)

    Software failure layout (use_eccheck_software_failure=True):
    - rank_in_group 0: loads own_buffer (to send to rig=1 via C++ P2P)
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
                checkpoint_dir, rank, "own_buffer", blocks["own_buffer"]
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
) -> None:
    """Drive C++ recovery for ECCHECK legacy load using submit_load_pipeline_chunk.

    Uses the same chunked pipeline API as the modern path but sources data
    from pre-loaded .pt block buffers instead of mmap files.
    """
    from time import time

    native = manager._eccheck_native
    if native is None:
        raise RuntimeError("ECCHECK native module is not initialized")

    from megatron.training import get_args as _get_args
    args = _get_args()
    software_failure = bool(getattr(args, "use_eccheck_software_failure", False))

    rank_in_group = manager._get_rank_in_group(rank, world_size)

    # ---- software failure path ----
    # rank_in_group 0 sends own_buffer to rank_in_group 1 via C++ P2P.
    # rank_in_group 1 receives into recovered_buffer.
    # This exercises the network path for worst-case recovery time measurement.
    if software_failure:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if rank_in_group == 0:
            own_buf = blocks["own_buffer"].contiguous().view(torch.uint8).reshape(-1)
            actual_tensor_bytes = _max_tensor_bytes_from_registry(registry, world_size)
            send_size = min(actual_tensor_bytes, own_buf.numel())
            native.simple_p2p_send(int(own_buf.data_ptr()), send_size)
        elif rank_in_group == 1:
            if recovered_buffer is None:
                raise RuntimeError("ECCHECK legacy: software failure needs recovered_buffer")
            native.simple_p2p_recv(
                int(recovered_buffer.data_ptr()), recovered_buffer.numel()
            )
        # rig=2/3: no-op
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    # ---- hardware failure path (rank_in_group 2) ----
    failed_rank = 2
    native.set_load_mode(True, failed_rank)
    logger.info(f"ECCHECK legacy: set load mode (failed_rank={failed_rank})")

    # Compute pipeline size
    max_total_bytes = _max_tensor_bytes_from_registry(registry, world_size)
    if max_total_bytes == 0:
        return
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

    # Ensure recv encoding buffers are allocated
    if manager.eccheck_recv_encoding_buffers is None:
        manager.eccheck_recv_encoding_buffers = (
            manager.allocate_recv_encoding_buffers_phase2(registry)
        )
    __, recv_buf2 = manager.eccheck_recv_encoding_buffers
    recv_base2 = int(recv_buf2.data_ptr())

    own_base = int(blocks["own_buffer"].data_ptr())
    partner_base = int(blocks["partner_buffer"].data_ptr())
    own_buf = blocks["own_buffer"]
    partner_buf = blocks["partner_buffer"]

    native.reset_encoding_completion_flags()
    if active_event is not None:
        active_event.set()

    processed = 0
    p2p_partner_offset = 0
    recv_offset2 = 0

    start_t = time()
    try:
        while processed < max_total_bytes:
            remaining = max_total_bytes - processed
            take = min(buffer_size, remaining)

            cur_buffer_addr = _get_free_data()

            # Copy source data from pre-loaded P2P block buffers
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

            # Parity and recv: only rank2/3 need these
            if rank_in_group in (2, 3):
                parity_addr2 = _get_free_parity()
                recv_offset2_aligned = ((recv_offset2 + 63) // 64) * 64
                recv_addr2 = recv_base2 + recv_offset2_aligned
                recv_chunk_size = take
                recv_offset2 = recv_offset2_aligned + recv_chunk_size
            else:
                parity_addr2 = 0
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
                p2p_partner_write_addr=p2p_partner_write,
            )

            processed += take

        # Sentinels and completion
        native.submit_load_encoding_sentinel()
        native.wait_for_xor_worker_completion()

        if rank_in_group in (2, 3):
            native.submit_load_step6_p2p_sentinel()

        native.wait_for_encoding_completion()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Extract recovered data from own_buffer for rank2.
        # own_buf was written by C++ at 64B-aligned offsets with gaps between
        # chunks.  We iterate over the same chunk layout used during encoding
        # to reconstruct a dense buffer.
        if rank_in_group == 2 and recovered_buffer is not None:
            recovered_dense = _extract_dense_from_gapped_buffer(
                own_buf, buffer_size, max_total_bytes,
            )
            actual_tensor_bytes = _max_tensor_bytes_from_registry(registry, world_size)
            if recovered_buffer.numel() >= total_size:
                n_copy = min(actual_tensor_bytes, total_size,
                             recovered_dense.numel())
                recovered_buffer[:n_copy].copy_(recovered_dense[:n_copy])

        logger.info(f"ECCHECK legacy: hw recovery pipeline done in {time() - start_t:.2f}s")

    finally:
        if active_event is not None:
            active_event.clear()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Two-failure recovery pipeline (load)
# ---------------------------------------------------------------------------

def _run_eccheck_two_failures_recovery(
    manager: ECCHECKManager,
    rank: int,
    world_size: int,
    blocks: Dict[str, Any],
    recovered_buffer: Optional[torch.Tensor],
    total_size: int,
    registry: GlobalMetadataRegistry,
) -> None:
    """Drive C++ two-failure recovery for ECCHECK legacy load.

    Two physical nodes lost → rig1 and rig2 in each 4-rank group are failed.
    Survivors rig0 and rig3 use bidirectional XOR exchange to recover
    d2 (on rig2) and verify d3 (on rig3), then forward results via P2P.

    Uses save-path 16-thread encode pool (ec_rs_encode_pool) and RDMA transport,
    aligned with the save encoding pipeline.
    """
    from time import time

    native = manager._eccheck_native
    if native is None:
        raise RuntimeError("ECCHECK native module is not initialized")

    rank_in_group = manager._get_rank_in_group(rank, world_size)
    is_failed = rank_in_group in (1, 2)
    is_survivor = rank_in_group in (0, 3)

    # Set C++ to two-failure mode (failed_rank=10, following ECLATIN convention)
    native.set_load_mode(True, 10)
    logger.info(
        f"ECCHECK legacy two-failures: load mode set (failed_rank=10), "
        f"rank={rank}, rank_in_group={rank_in_group}"
    )

    # Compute pipeline size
    max_total_bytes = _max_tensor_bytes_from_registry(registry, world_size)
    if max_total_bytes == 0:
        return
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

    # Allocate recv encoding buffers
    if manager.eccheck_recv_encoding_buffers is None:
        manager.eccheck_recv_encoding_buffers = (
            manager.allocate_recv_encoding_buffers_phase2(registry)
        )
    recv_buf1, recv_buf2 = manager.eccheck_recv_encoding_buffers
    recv_base_1 = int(recv_buf1.data_ptr())
    recv_base_2 = int(recv_buf2.data_ptr())

    own_base = int(blocks["own_buffer"].data_ptr())
    partner_base = int(blocks["partner_buffer"].data_ptr())
    own_buf = blocks["own_buffer"]
    partner_buf = blocks["partner_buffer"]

    native.reset_encoding_completion_flags()
    if active_event is not None:
        active_event.set()

    # ---- Phase 1: P2P data distribution (before encoding pipeline) ----
    # Rig0 → Rig1: send d1 (partner_buffer)
    # Rig3 → Rig2: send p2 = d0⊕d2 (partner_buffer)
    if rank_in_group == 0:
        native.simple_p2p_send(int(partner_buf.data_ptr()), partner_buf.numel())
        logger.info(
            f"ECCHECK two-failures: rig0 sent d1 to rig1 "
            f"({partner_buf.numel() / (1024**3):.2f} GB)"
        )
    elif rank_in_group == 1:
        native.simple_p2p_recv(int(partner_buf.data_ptr()), partner_buf.numel())
        logger.info(
            f"ECCHECK two-failures: rig1 received d1 from rig0 "
            f"({partner_buf.numel() / (1024**3):.2f} GB)"
        )
    elif rank_in_group == 2:
        native.simple_p2p_recv(int(own_buf.data_ptr()), own_buf.numel())
        logger.info(
            f"ECCHECK two-failures: rig2 received p2 from rig3 "
            f"({own_buf.numel() / (1024**3):.2f} GB)"
        )
    elif rank_in_group == 3:
        native.simple_p2p_send(int(partner_buf.data_ptr()), partner_buf.numel())
        logger.info(
            f"ECCHECK two-failures: rig3 sent p2 to rig2 "
            f"({partner_buf.numel() / (1024**3):.2f} GB)"
        )

    processed = 0
    own_offset = 0
    partner_offset = 0
    recv_offset_1 = 0
    recv_offset_2 = 0

    start_t = time()
    try:
        while processed < max_total_bytes:
            remaining = max_total_bytes - processed
            take = min(buffer_size, remaining)

            # ---- Phase 2: copy data into chunk buffer ----
            cur_buffer_addr = _get_free_data()
            buffer_ptr = ctypes.cast(cur_buffer_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer_array = ctypes.cast(buffer_ptr, ctypes.POINTER(ctypes.c_uint8 * take))

            if rank_in_group == 0:
                # Survivor: use d0 from own_buffer (loaded from disk)
                src_off = processed
                bytes_to_copy = min(take, own_buf.numel() - src_off)
                if bytes_to_copy > 0:
                    ctypes.memmove(buffer_array.contents,
                                   own_base + src_off, bytes_to_copy)
                if take > bytes_to_copy:
                    ctypes.memset(
                        ctypes.cast(ctypes.addressof(buffer_array.contents) + bytes_to_copy,
                                    ctypes.POINTER(ctypes.c_uint8)),
                        0, take - bytes_to_copy)
            elif rank_in_group == 1:
                # Failed: use d1 from partner_buffer (received via Phase 1 P2P from rig0)
                src_off = processed
                bytes_to_copy = min(take, partner_buf.numel() - src_off)
                if bytes_to_copy > 0:
                    ctypes.memmove(buffer_array.contents,
                                   partner_base + src_off, bytes_to_copy)
                if take > bytes_to_copy:
                    ctypes.memset(
                        ctypes.cast(ctypes.addressof(buffer_array.contents) + bytes_to_copy,
                                    ctypes.POINTER(ctypes.c_uint8)),
                        0, take - bytes_to_copy)
            elif rank_in_group == 2:
                # Failed: use p2 from own_buffer (received via Phase 1 P2P from rig3)
                src_off = processed
                bytes_to_copy = min(take, own_buf.numel() - src_off)
                if bytes_to_copy > 0:
                    ctypes.memmove(buffer_array.contents,
                                   own_base + src_off, bytes_to_copy)
                if take > bytes_to_copy:
                    ctypes.memset(
                        ctypes.cast(ctypes.addressof(buffer_array.contents) + bytes_to_copy,
                                    ctypes.POINTER(ctypes.c_uint8)),
                        0, take - bytes_to_copy)
            elif rank_in_group == 3:
                # Survivor: use d3 from own_buffer (loaded from disk)
                src_off = processed
                bytes_to_copy = min(take, own_buf.numel() - src_off)
                if bytes_to_copy > 0:
                    ctypes.memmove(buffer_array.contents,
                                   own_base + src_off, bytes_to_copy)
                if take > bytes_to_copy:
                    ctypes.memset(
                        ctypes.cast(ctypes.addressof(buffer_array.contents) + bytes_to_copy,
                                    ctypes.POINTER(ctypes.c_uint8)),
                        0, take - bytes_to_copy)
            else:
                ctypes.memset(buffer_array.contents, 0, take)

            # ---- Phase 3: allocate encoding buffers (TWO per chunk) ----
            enc_addr_0 = _get_free_encoding()  # for parity row 0
            enc_addr_1 = _get_free_encoding()  # for parity row 1

            # ---- Phase 3: XOR exchange (bidirectional) ----
            # Each rank both sends and receives:
            #   rig0 ↔ rig2  (parity row 0: rig0 sends enc_0(d0), recvs enc_0(p2), XOR → p0)
            #   rig1 ↔ rig3  (parity row 1: rig1 sends enc_1(d1), recvs enc_1(p3), XOR → d2)

            # Bounds check
            own_offset_aligned = ((own_offset + 63) // 64) * 64
            partner_offset_aligned = ((partner_offset + 63) // 64) * 64
            p2p_rem_own = own_buf.numel() - own_offset_aligned
            p2p_rem_partner = partner_buf.numel() - partner_offset_aligned
            max_p2p = min(p2p_rem_own, p2p_rem_partner)

            recv_offset_1_aligned = ((recv_offset_1 + 63) // 64) * 64
            recv_offset_2_aligned = ((recv_offset_2 + 63) // 64) * 64
            recv_rem_1 = recv_buf1.numel() - recv_offset_1_aligned
            recv_rem_2 = recv_buf2.numel() - recv_offset_2_aligned
            max_recv = min(recv_rem_1, recv_rem_2)

            take_bounded = min(take, max_p2p, max_recv)
            if take_bounded < 64:
                logger.warning(
                    f"ECCHECK two-failures: buffers exhausted at "
                    f"{processed / (1024**3):.2f} GB"
                )
                break
            take = take_bounded

            # Addresses for encode output and recv
            own_write_addr = own_base + own_offset_aligned
            partner_write_addr = partner_base + partner_offset_aligned
            recv_addr_1 = recv_base_1 + recv_offset_1_aligned
            recv_addr_2 = recv_base_2 + recv_offset_2_aligned

            # Advance offsets
            if rank_in_group in (0, 1):
                own_offset = own_offset_aligned + take
            if rank_in_group in (2, 3):
                partner_offset = partner_offset_aligned + take
            recv_offset_1 = recv_offset_1_aligned + take
            recv_offset_2 = recv_offset_2_aligned + take

            # Phase 2b: dual-coefficient encoding → route to save-path 16-thread pool
            native.submit_two_failure_encoding_chunk(
                data_addr=cur_buffer_addr,
                size=take,
                enc_addr_0=enc_addr_0,
                enc_addr_1=enc_addr_1,
                recv_addr_1=recv_addr_1,
                recv_addr_2=recv_addr_2,
                recv_chunk_size=take,
                own_write_addr=own_write_addr,
                partner_write_addr=partner_write_addr,
            )

            processed += take

        # Sentinels and completion
        native.submit_two_failure_encoding_sentinels()
        native.wait_for_encoding_completion()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ---- Phase 4: Extract recovered data ----
        # rig2: own_buffer was written by C++ via XOR as d2 (gapped)
        if rank_in_group == 2 and recovered_buffer is not None:
            recovered_dense = _extract_dense_from_gapped_buffer(
                own_buf, buffer_size, max_total_bytes,
            )
            actual_tensor_bytes = _max_tensor_bytes_from_registry(registry, world_size)
            if recovered_buffer.numel() >= total_size:
                n_copy = min(actual_tensor_bytes, total_size,
                             recovered_dense.numel())
                recovered_buffer[:n_copy].copy_(recovered_dense[:n_copy])

        # rig1: partner_buffer was written by C++ via P2P as d1 (gapped)
        if rank_in_group == 1 and recovered_buffer is not None:
            recovered_dense = _extract_dense_from_gapped_buffer(
                partner_buf, buffer_size, max_total_bytes,
            )
            actual_tensor_bytes = _max_tensor_bytes_from_registry(registry, world_size)
            if recovered_buffer.numel() >= total_size:
                n_copy = min(actual_tensor_bytes, total_size,
                             recovered_dense.numel())
                recovered_buffer[:n_copy].copy_(recovered_dense[:n_copy])

        logger.info(
            f"ECCHECK legacy: two-failure recovery pipeline done in "
            f"{time() - start_t:.2f}s"
        )

    finally:
        if active_event is not None:
            active_event.clear()

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
    result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


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

    rank_in_group = manager._get_rank_in_group(rank, world_size)
    sw_failure = bool(getattr(args, "use_eccheck_software_failure", False))
    two_failures = bool(getattr(args, "use_eccheck_two_failures", False))
    total_size = sum(meta.size_bytes for meta in rank_metadata.get(rank, []))

    blocks = _allocate_eccheck_blocks_legacy(manager, rank_metadata)

    recovered_buffer: Optional[torch.Tensor] = None

    if two_failures:
        # Two physical nodes lost: rig1 and rig2 are failed.
        # Survivors: rig0 and rig3.
        pin = torch.cuda.is_available() and getattr(manager, "eccheck_pin_memory", False)
        actual_tensor_bytes = _max_tensor_bytes_from_registry(registry, world_size)

        if rank_in_group in (0, 3):
            # Survivors: load existing blocks from disk
            _load_eccheck_blocks_from_disk_into(
                blocks, checkpoint_dir, rank, rank_in_group,
                software_failure=False, two_failures=True,
            )
        elif rank_in_group in (1, 2):
            # Failed ranks: allocate recovered_buffer
            recovered_buffer = torch.empty(
                actual_tensor_bytes, dtype=torch.uint8, pin_memory=pin
            )
            total_size = actual_tensor_bytes
            logger.info(
                f"ECCHECK two-failures: rig{rank_in_group} (failed) allocated "
                f"recovered_buffer {actual_tensor_bytes / (1024**3):.2f} GB"
            )
        else:
            raise RuntimeError(
                f"ECCHECK two-failures: unexpected rank_in_group={rank_in_group}"
            )
    elif sw_failure:
        # Software failure: rig=0 sends own_buffer to rig=1 via C++ P2P
        if rank_in_group == 0:
            _load_eccheck_blocks_from_disk_into(
                blocks, checkpoint_dir, rank, rank_in_group, software_failure=True,
            )
        elif rank_in_group == 1:
            # Buffer size must match rig=0's send size (max across group),
            # not this rank's own tensor total, to avoid RDMA buffer overflow.
            actual_tensor_bytes = _max_tensor_bytes_from_registry(registry, world_size)
            pin = torch.cuda.is_available() and getattr(manager, "eccheck_pin_memory", False)
            recovered_buffer = torch.empty(actual_tensor_bytes, dtype=torch.uint8, pin_memory=pin)
            total_size = actual_tensor_bytes
        # rig=2/3: not participating in software failure
    elif rank_in_group == 2:
        # Hardware failure: rank_in_group 2 is the failed rank, data comes via network
        pin = torch.cuda.is_available() and getattr(manager, "eccheck_pin_memory", False)
        recovered_buffer = torch.empty(total_size, dtype=torch.uint8, pin_memory=pin)
    else:
        _load_eccheck_blocks_from_disk_into(
            blocks, checkpoint_dir, rank, rank_in_group, software_failure=False,
        )

    if manager.use_rdma:
        for t in [blocks["own_buffer"], blocks["partner_buffer"]]:
            manager.register_buffer(t)
        if recovered_buffer is not None:
            manager.register_buffer(recovered_buffer)

    # === timing: network/encode (C++ P2P or XOR pipeline) ===
    _t_ec: Dict[str, float] = {}
    _t0 = time.time()
    if two_failures:
        _run_eccheck_two_failures_recovery(
            manager=manager,
            rank=rank,
            world_size=world_size,
            blocks=blocks,
            recovered_buffer=recovered_buffer,
            total_size=total_size,
            registry=registry,
        )
    else:
        _run_eccheck_legacy_recovery(
            manager=manager,
            rank=rank,
            world_size=world_size,
            blocks=blocks,
            recv_buffers=None,
            recovered_buffer=recovered_buffer,
            total_size=total_size,
            registry=registry,
        )
    _t_ec['network_encode'] = time.time() - _t0

    # === timing: rebuild state_dict ===
    _t0 = time.time()
    if two_failures:
        use_recovered = rank_in_group in (1, 2)
    else:
        use_recovered = rank_in_group == 2 or (sw_failure and rank_in_group == 1)
    state_dict = _reconstruct_state_dict_from_eccheck_buffer(
        main_payload,
        recovered_buffer=recovered_buffer if use_recovered else None,
    )
    _t_ec['rebuild_sd'] = time.time() - _t0
    _t_ec['total'] = _t_ec['network_encode'] + _t_ec['rebuild_sd']

    if two_failures:
        _mode = "2F"
    elif sw_failure:
        _mode = "SW"
    else:
        _mode = "HW"
    logger.info(
        "ECCHECK legacy load timing (%s): "
        "total=%(total).2fs network_encode=%(network_encode).2fs "
        "rebuild_sd=%(rebuild_sd).2fs", _mode, _t_ec
    )

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Stop C++ load workers so they don't interfere with subsequent training.
    # The singleton manager will be reinitialized on the next save.
    logger.info(f"ECCHECK legacy: cleaning up C++ module after recovery (rank {rank})")
    manager.cleanup()
    manager._eccheck_native = None

    return state_dict
