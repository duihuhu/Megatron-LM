import ctypes
import queue
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from megatron.core.dist_checkpointing.strategies.ecnaive_manager import ECNAIVEManager
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



_BUILD_GLOBAL_REGISTRY_CACHE: Dict[tuple, tuple] = {}

def _build_global_registry(local_metadata: List[TensorMetadata], local_non_tensor: Dict[str, Any]) -> Tuple[Dict[int, List[TensorMetadata]], Dict[int, Dict[str, Any]]]:
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


def _allocate_ecnaive_blocks(
    manager: ECNAIVEManager,
    rank_metadata: Dict[int, List[TensorMetadata]],
) -> Dict[str, Any]:
    """Allocate n persistent blocks for EC-NAIVE save (generalized k+2 scheme).

    Each rank stores n blocks: 1 own data block + (n-1) blocks received from peers.
    Each block size = ceil(max_total_bytes / k / buffer_size) * buffer_size.
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    own_total_size = sum(meta.size_bytes for meta in rank_metadata.get(rank, []))
    k = manager.ecnaive_k
    n = manager.ecnaive_n

    if world_size > 1:
        all_sizes = [
            sum(meta.size_bytes for meta in rank_metadata.get(r, []))
            for r in range(world_size)
        ]
        max_total_bytes = max(all_sizes)
    else:
        max_total_bytes = own_total_size

    block_data_size = (max_total_bytes + k - 1) // k  # ceil division
    aligned_block_size = (
        (block_data_size + manager.ecnaive_buffer_size - 1)
        // manager.ecnaive_buffer_size
    ) * manager.ecnaive_buffer_size

    slices = manager.allocate_preallocated_blocks(n, aligned_block_size)

    # Name blocks: own_data0 + recv_0 ... recv_{n-2}
    blocks: Dict[str, torch.Tensor] = {"own_data0": slices[0]}
    blocks["block_names"] = ["own_data0"]
    for j in range(1, n):
        name = f"recv_{j - 1}"
        blocks[name] = slices[j]
        blocks["block_names"].append(name)

    # Backward-compat aliases for k=2 (n=4)
    if k == 2:
        blocks["data0"] = blocks["own_data0"]
        blocks["recv_parity1"] = blocks["recv_0"]
        blocks["recv_parity0"] = blocks["recv_1"]
        blocks["recv_data1"] = blocks["recv_2"]

    blocks["actual_size"] = own_total_size
    blocks["pipeline_size"] = max_total_bytes
    blocks["aligned_size"] = aligned_block_size
    blocks["block_data_size"] = block_data_size
    return blocks


def _encode_with_native(
    manager: ECNAIVEManager,
    tensor_buffer: torch.Tensor,
    actual_data_bytes: int,
    ecnaive_blocks: Dict[str, Any],
) -> None:
    """Encode tensor data with ISA-L Reed-Solomon and distribute via C++ engine.

    Generalized for k+2 scheme: splits tensor_buffer into k data blocks,
    encodes to 2 parity blocks, keeps d_{i,0} locally, sends/receives the
    remaining n-1 blocks via round-robin.
    """
    buffers = manager.get_ecnaive_buffers()
    if buffers is None:
        raise RuntimeError("EC-NAIVE buffers are not initialized")
    if manager._ecnaive_native is None:
        raise RuntimeError("EC-NAIVE native module is not initialized")

    k = manager.ecnaive_k
    n = manager.ecnaive_n
    free_data_queue = buffers["free_data_buffer_queue"]
    free_parity_queue = buffers["free_parity_buffer_queue"]
    active_event = buffers.get("buffer_poller_active_event")
    poll_and_release = buffers.get("poll_and_release_buffers")

    def get_free_data_buffer():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_data_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("EC-NAIVE legacy: timeout waiting for free data buffer")
            return free_data_queue.get()

    def get_free_parity_buffer():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_parity_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("EC-NAIVE legacy: timeout waiting for free parity buffer")
            return free_parity_queue.get()

    pipeline_total_bytes = ecnaive_blocks["pipeline_size"]
    aligned_block_size = ecnaive_blocks["aligned_size"]
    block_data_size = ecnaive_blocks.get("block_data_size", (pipeline_total_bytes + k - 1) // k)

    # Persistent block base addresses
    own_data0 = ecnaive_blocks["own_data0"]
    own_data0_base = int(own_data0.data_ptr())
    own_data0_offset = 0

    recv_blocks = [ecnaive_blocks[name] for name in ecnaive_blocks["block_names"] if name != "own_data0"]
    recv_bases = [int(t.data_ptr()) for t in recv_blocks]
    recv_offsets = [0] * len(recv_blocks)
    num_recv = len(recv_blocks)  # should be n-1

    ecnaive_buffer_size = manager.ecnaive_buffer_size
    native = manager._ecnaive_native
    native.reset_encoding_completion_flags()

    if active_event is not None:
        active_event.set()

    src_pos = 0
    src_base_ptr = tensor_buffer.data_ptr()

    try:
        while src_pos < block_data_size:
            remaining_in_block = block_data_size - src_pos
            take = min(ecnaive_buffer_size, remaining_in_block)

            # Stage k data blocks into temporary buffers
            data_block_addrs = []  # data addrs for C++ encoder
            own_data0_write_addr = own_data0_base + own_data0_offset

            # d_{i,0}: write directly to persistent own_data0 block
            data_block_addrs.append(own_data0_write_addr)
            src_offset_0 = 0 * block_data_size + src_pos
            if src_offset_0 < actual_data_bytes:
                bytes_to_copy = min(take, actual_data_bytes - src_offset_0)
                ctypes.memmove(own_data0_write_addr, src_base_ptr + src_offset_0, bytes_to_copy)
                if take > bytes_to_copy:
                    ctypes.memset(own_data0_write_addr + bytes_to_copy, 0, take - bytes_to_copy)
            else:
                ctypes.memset(own_data0_write_addr, 0, take)

            # d_{i,1}..d_{i,k-1}: copy to temp pool buffers (will be sent)
            temp_data_bufs = []
            for j in range(1, k):
                data_addr = get_free_data_buffer()
                temp_data_bufs.append(data_addr)
                data_block_addrs.append(data_addr)
                src_offset_j = j * block_data_size + src_pos
                if src_offset_j < actual_data_bytes:
                    bytes_to_copy = min(take, actual_data_bytes - src_offset_j)
                    ctypes.memmove(data_addr, src_base_ptr + src_offset_j, bytes_to_copy)
                    if take > bytes_to_copy:
                        ctypes.memset(data_addr + bytes_to_copy, 0, take - bytes_to_copy)
                else:
                    ctypes.memset(data_addr, 0, take)

            # Parity buffers from pool
            parity0_addr = get_free_parity_buffer()
            parity1_addr = get_free_parity_buffer()

            # Aligned recv block write addresses
            recv_write_addrs = []
            for i in range(num_recv):
                aligned = ((recv_offsets[i] + 63) // 64) * 64
                if aligned + take > aligned_block_size:
                    logger.warning("EC-NAIVE legacy: recv block %d exhausted", i)
                    # reset to 0 to continue
                    aligned = 0
                recv_write_addrs.append(recv_bases[i] + aligned)
                recv_offsets[i] = aligned + take

            # Submit to C++ native: encode k data → 2 parity, send/recv over network
            native.submit_ecnaive_save_general(
                data_block_addrs, parity0_addr, parity1_addr,
                recv_write_addrs, take,
            )

            own_data0_offset += take
            src_pos += take

        # Sentinels: n-1 send + n-1 recv
        native.submit_send_sentinels(num_sends=num_recv)
        native.submit_recv_sentinels(num_recvs=num_recv)
        native.wait_for_encoding_completion()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        if active_event is not None:
            active_event.clear()


def _save_ecnaive_pt_files(
    checkpoint_name: str,
    rank: int,
    non_tensor_data: Dict[str, Any],
    tensor_infos: List[Any],
    blocks: Dict[str, Any],
    full_tensor_buffer: torch.Tensor,
    flat_key_roots: Optional[Set[str]] = None,
    manager: Optional[ECNAIVEManager] = None,
) -> None:
    checkpoint_path = Path(checkpoint_name)
    checkpoint_dir = checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    k = manager.ecnaive_k if manager else 2
    n = manager.ecnaive_n if manager else 4

    # Build block_files dict: own_data0 + recv blocks
    block_files: Dict[str, str] = {}
    block_names = blocks.get("block_names", [])
    for name in block_names:
        block_files[name] = f"ecnaive_block_rank{rank}_{name}.pt"

    # Backward-compat block_files keys for k=2
    if k == 2 and len(block_names) >= 4:
        block_files_legacy = {
            "data0": block_files.get("own_data0", ""),
            "recv_parity1": block_files.get("recv_0", ""),
            "recv_parity0": block_files.get("recv_1", ""),
            "recv_data1": block_files.get("recv_2", ""),
        }
    else:
        block_files_legacy = None

    main_file = checkpoint_dir / f"ecnaive_main_rank{rank}.pt"
    from megatron.training.legacy_io_utils import MAGIC_ECNAIVE, MAGIC_BLOCK

    # Pre-serialize metadata + prepare memoryview for main file
    import pickle as _pickle
    meta1 = _pickle.dumps(non_tensor_data)
    meta2 = _pickle.dumps(tensor_infos)
    extra = _pickle.dumps({
        "version": 2, "format": "ecnaive_torch_legacy", "rank": rank,
        "ecnaive_k": k, "ecnaive_n": n,
        "actual_tensor_size": blocks["actual_size"],
        "pipeline_total_bytes": blocks["pipeline_size"],
        "aligned_block_size": blocks["aligned_size"],
        "block_data_size": blocks.get("block_data_size", blocks["pipeline_size"] // k),
        "flat_key_roots": list(flat_key_roots) if flat_key_roots else [],
        "block_files": block_files,
        "_block_files_legacy": block_files_legacy,
    })
    buf = full_tensor_buffer[: blocks["actual_size"]]
    if not buf.is_contiguous():
        buf = buf.contiguous()
    if buf.device.type != "cpu":
        buf = buf.to("cpu")
    main_mv = memoryview(buf.numpy())

    # Pre-prepare block memoryviews
    block_mvs = {}
    for name in block_names:
        b = blocks[name][: blocks[name].numel()]
        if not b.is_contiguous():
            b = b.contiguous()
        if b.device.type != "cpu":
            b = b.to("cpu")
        block_mvs[name] = memoryview(b.numpy())

    # Parallel writes
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1 + len(block_names)) as ex:
        from megatron.training.legacy_io_utils import write_main_prepared, write_block_prepared
        futs = [ex.submit(write_main_prepared, str(main_file), MAGIC_ECNAIVE,
                          meta1, meta2, extra, main_mv, blocks["actual_size"])]
        for name in block_names:
            block_file = checkpoint_dir / f"ecnaive_block_rank{rank}_{name}.pt"
            futs.append(ex.submit(write_block_prepared,
                                  str(block_file), MAGIC_BLOCK,
                                  block_mvs[name], blocks[name].numel()))
        for f in futs:
            f.result()


def _checkpoint_dir_from_path(checkpoint_name: str) -> Path:
    checkpoint_path = Path(checkpoint_name)
    return checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent


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


def _load_ecnaive_main_payload(
    checkpoint_dir: Path, rank: int, world_size: int
) -> Dict[str, Any]:
    """
    Load ecnaive_main_rank{rank}.pt. If missing on this rank, recover payload via all_gather_object
    using other ranks' copies (rank r uses gathered[r] when present).
    """
    main_path = checkpoint_dir / f"ecnaive_main_rank{rank}.pt"
    local_payload: Optional[Dict[str, Any]] = None
    if main_path.is_file():
        from megatron.training.legacy_io_utils import is_raw_format, read_raw_checkpoint, MAGIC_ECNAIVE
        if is_raw_format(str(main_path), MAGIC_ECNAIVE):
            local_payload = read_raw_checkpoint(str(main_path), MAGIC_ECNAIVE)
        else:
            local_payload = torch.load(main_path, map_location="cpu", weights_only=False)

    if world_size <= 1 or not torch.distributed.is_initialized():
        if local_payload is None:
            raise FileNotFoundError(f"EC-NAIVE legacy: missing main file {main_path}")
        return local_payload

    gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_payload)

    chosen = gathered[rank]
    if chosen is None:
        raise FileNotFoundError(
            f"EC-NAIVE legacy: ecnaive_main_rank{rank}.pt missing on all ranks under {checkpoint_dir}"
        )
    return chosen


def _load_blocks_from_disk(checkpoint_dir: Path, rank: int) -> Dict[str, torch.Tensor]:
    from megatron.training.legacy_io_utils import is_raw_format, read_raw_block, MAGIC_BLOCK
    blocks: Dict[str, torch.Tensor] = {}
    for block_name in ("data0", "recv_parity1", "recv_parity0", "recv_data1"):
        block_path = checkpoint_dir / f"ecnaive_block_rank{rank}_{block_name}.pt"
        if not block_path.is_file():
            raise FileNotFoundError(f"EC-NAIVE legacy: missing block file {block_path}")
        if is_raw_format(str(block_path), MAGIC_BLOCK):
            blocks[block_name] = read_raw_block(str(block_path), MAGIC_BLOCK)
        else:
            payload = torch.load(block_path, map_location="cpu", weights_only=False)
            blocks[block_name] = payload["tensor"].contiguous().view(torch.uint8)
    return blocks


def _decode_data0_to_linear_first_half(
    data0: torch.Tensor,
    pipeline_total_bytes: int,
    ecnaive_buffer_size: int,
) -> torch.Tensor:
    """Invert legacy encode layout: recover tensor_buffer[0:half_total) from data0."""
    half_total = pipeline_total_bytes // 2
    out = torch.zeros(half_total, dtype=torch.uint8, device=data0.device)
    src_pos = 0
    data0_offset = 0
    block_elems = data0.numel()
    while src_pos < half_total:
        remaining_in_out = half_total - src_pos
        take = min(ecnaive_buffer_size, remaining_in_out)
        aligned = ((data0_offset + 63) // 64) * 64
        if aligned + take > block_elems:
            logger.warning("EC-NAIVE legacy load: data0 exhausted during decode")
            break
        out[src_pos : src_pos + take].copy_(data0[aligned : aligned + take])
        data0_offset = aligned + take
        src_pos += take
    return out


def _reconstruct_state_dict_from_main_and_data0(
    main_payload: Dict[str, Any],
    data0_uint8: torch.Tensor,
    manager: ECNAIVEManager,
    flat_key_roots: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    tensor_infos = main_payload["tensor_infos"]
    non_tensor_data = main_payload["non_tensor_data"]
    pipeline_total_bytes = int(main_payload["pipeline_total_bytes"])
    actual_tensor_size = int(main_payload["actual_tensor_size"])

    half_linear = _decode_data0_to_linear_first_half(
        data0_uint8,
        pipeline_total_bytes=pipeline_total_bytes,
        ecnaive_buffer_size=manager.ecnaive_buffer_size,
    )
    buf_len = max(pipeline_total_bytes, actual_tensor_size)
    full_buf = torch.zeros(buf_len, dtype=torch.uint8, device=half_linear.device)
    n = min(half_linear.numel(), buf_len)
    full_buf[:n].copy_(half_linear[:n])

    tensor_data = extract_tensors_from_continuous_buffer(full_buf, tensor_infos)
    decomposed = DecomposedStateDict(
        non_tensor_data=non_tensor_data,
        tensor_infos=tensor_infos,
        tensor_data=tensor_data,
        flat_key_roots=flat_key_roots or set(),
    )
    result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


def _reconstruct_full_state_dict_from_main_tensor_buffer(
    main_payload: Dict[str, Any],
    flat_key_roots: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Rebuild full state_dict from main payload when tensor_buffer is present (aligned with torch_dist)."""
    tb = main_payload["tensor_buffer"]
    buf = tb.detach().contiguous().reshape(-1).view(torch.uint8)
    tensor_infos = main_payload["tensor_infos"]
    tensor_data = extract_tensors_from_continuous_buffer(buf, tensor_infos)
    decomposed = DecomposedStateDict(
        non_tensor_data=main_payload["non_tensor_data"],
        tensor_infos=tensor_infos,
        tensor_data=tensor_data,
        flat_key_roots=flat_key_roots or set(),
    )
    result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


def _load_ecnaive_block_file(
    checkpoint_dir: Path,
    rank: int,
    canonical_name: str,
    legacy_name: str,
    block_files_legacy: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """Load a single EC-NAIVE block file with canonical/legacy name fallback.

    Args:
        checkpoint_dir: Checkpoint directory.
        rank: Global rank.
        canonical_name: Canonical block name (e.g. "own_data0", "recv_2").
        legacy_name: Legacy block name (e.g. "data0", "recv_data1").
        block_files_legacy: Optional legacy->canonical mapping from main_payload extra metadata.

    Returns:
        torch.Tensor of dtype uint8 with the block data.
    """
    from megatron.training.legacy_io_utils import is_raw_format, read_raw_block, MAGIC_BLOCK

    candidates: List[Path] = []

    if block_files_legacy and legacy_name in block_files_legacy:
        candidates.append(checkpoint_dir / block_files_legacy[legacy_name])

    candidates.append(checkpoint_dir / f"ecnaive_block_rank{rank}_{canonical_name}.pt")
    candidates.append(checkpoint_dir / f"ecnaive_block_rank{rank}_{legacy_name}.pt")

    block_path = None
    for p in candidates:
        if p.is_file():
            block_path = p
            break

    if block_path is None:
        raise FileNotFoundError(
            f"EC-NAIVE legacy: missing block {canonical_name}/{legacy_name} "
            f"for rank {rank} under {checkpoint_dir}"
        )

    if is_raw_format(str(block_path), MAGIC_BLOCK):
        return read_raw_block(str(block_path), MAGIC_BLOCK)
    payload = torch.load(str(block_path), map_location="cpu", weights_only=False)
    return payload["tensor"].contiguous().view(torch.uint8).reshape(-1)


def _decode_data_block(
    block: torch.Tensor,
    pipeline_total_bytes: int,
    block_idx: int,
    block_data_size: int,
    ecnaive_buffer_size: int,
) -> torch.Tensor:
    """Decode one padded data block into its linear segment of tensor_buffer.

    During save, data block j covers bytes [j*block_data_size, (j+1)*block_data_size)
    of the original tensor.  The block is written with 64-byte alignment between
    ecnaive_buffer_size chunks.
    """
    start_byte = block_idx * block_data_size
    end_byte = min(start_byte + block_data_size, pipeline_total_bytes)
    actual = max(0, end_byte - start_byte)
    out = torch.zeros(actual, dtype=torch.uint8, device=block.device)
    src_pos = 0
    block_offset = 0
    while src_pos < actual:
        take = min(ecnaive_buffer_size, actual - src_pos)
        aligned = ((block_offset + 63) // 64) * 64
        if aligned + take > block.numel():
            logger.warning("EC-NAIVE: data block %d exhausted during decode", block_idx)
            break
        out[src_pos : src_pos + take].copy_(block[aligned : aligned + take])
        block_offset = aligned + take
        src_pos += take
    return out


def _load_ecnaive_legacy_software_failure(
    checkpoint_dir: Path,
    rank: int,
    world_size: int,
    manager: ECNAIVEManager,
    main_payload: Dict[str, Any],
    global_registry: GlobalMetadataRegistry,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """EC-NAIVE legacy load software failure path (generalized for any k >= 2).

    The failed rank (rig=2 by convention) reads d_{2,0} from local disk and
    receives d_{2,1}..d_{2,k-1} from k-1 sender ranks via C++ ASIO/RDMA
    (k-1 ports).  Sender ranks load their block files and send via C++.
    Non-participating ranks just reconstruct from their own main.pt.

    If *timings* dict is provided, it will be populated with:
      init, xfer, rebuild_sd (all in seconds).
    """
    from time import time as _time
    _t = timings if timings is not None else {}
    _t0_func = _time()

    native = manager._ecnaive_native
    if native is None:
        raise RuntimeError("EC-NAIVE native module not initialized for sw recovery")

    k = manager.ecnaive_k
    n = manager.ecnaive_n
    rank_in_group = manager._get_rank_in_group(rank, world_size)
    group_id = manager._get_group_id(rank, world_size)
    failed_rig = 2
    num_network_blocks = k - 1  # blocks d_{2,1} .. d_{2,k-1} come from network

    flat_key_roots = _infer_flat_key_roots(main_payload)
    pipeline_total_bytes = int(main_payload["pipeline_total_bytes"])
    actual_tensor_size = int(main_payload["actual_tensor_size"])
    aligned_block_size = int(main_payload["aligned_block_size"])
    block_data_size = int(main_payload.get("block_data_size",
                         (pipeline_total_bytes + k - 1) // k))
    tensor_infos = main_payload["tensor_infos"]
    non_tensor_data = main_payload["non_tensor_data"]

    block_files_legacy = main_payload.get("_block_files_legacy", None)

    # Phase 1: setup (not timed) — RDMA connection setup + barrier
    manager.init_ecnaive_sw_recovery(rank, world_size, failed_rank_in_group=failed_rig)

    # === timing: network/encode (C++ send/recv + decode + assemble) ===
    _t0_net = _time()
    if rank_in_group == failed_rig:
        # ---- FAILED RANK ----
        # Phase 2: local d_{2,0} + network d_{2,1}..d_{2,k-1}
        own_data0 = _load_ecnaive_block_file(
            checkpoint_dir, rank,
            canonical_name="own_data0",
            legacy_name="data0",
            block_files_legacy=block_files_legacy,
        )
        recv_blocks = []
        for j in range(1, k):
            buf = torch.zeros(aligned_block_size, dtype=torch.uint8)
            if manager.use_rdma:
                manager.register_buffer(buf)
            native.sw_recv_data(j - 1, int(buf.data_ptr()), buf.numel())
            recv_blocks.append(buf)
            logger.info(
                "EC-NAIVE legacy sw: rank2 received d_2,%d (%d bytes)",
                j, buf.numel(),
            )

        # Decode gapped blocks → linear tensor_buffer
        decoded = []
        for idx, blk in enumerate([own_data0] + recv_blocks):
            decoded.append(_decode_data_block(
                blk, pipeline_total_bytes, idx,
                block_data_size, manager.ecnaive_buffer_size,
            ))
        full_buf = torch.cat(decoded, dim=0)
        if actual_tensor_size > 0:
            full_buf = full_buf[:actual_tensor_size]
        _t['network_encode'] = _time() - _t0_net

        _t0_rebuild = _time()
        tensor_data = extract_tensors_from_continuous_buffer(full_buf, tensor_infos)
        decomposed = DecomposedStateDict(
            non_tensor_data=non_tensor_data,
            tensor_infos=tensor_infos,
            tensor_data=tensor_data,
            flat_key_roots=flat_key_roots,
        )
        state_dict = reconstruct_state_dict(decomposed)
        unflatten_optimizer_fp32_params(state_dict)
        _t['rebuild_sd'] = _time() - _t0_rebuild
        logger.info(
            "EC-NAIVE legacy sw: rank_in_group=%d recovered state_dict in %.2fs",
            failed_rig, _time() - _t0_func,
        )

    else:
        # ---- SENDER RANK (if I hold a data block for the failed rank) ----
        sender_rig = rank_in_group
        j = (sender_rig - failed_rig + n) % n  # data block index j for this sender
        if 1 <= j < k:
            # init_ecnaive_sw_recovery already called above for all ranks
            block_idx = j - 1  # 0-indexed for sw_send_data

            # recv file index on this sender: recv_{(sender_rig - failed_rig - 1 + n) % n}
            recv_idx = (sender_rig - failed_rig - 1 + n) % n
            canonical_name = f"recv_{recv_idx}"

            # Legacy names (k=2 backward compat)
            if k == 2:
                legacy_map_2 = {0: "recv_parity1", 1: "recv_parity0", 2: "recv_data1"}
                legacy_name = legacy_map_2.get(recv_idx, canonical_name)
            else:
                legacy_name = canonical_name

            block = _load_ecnaive_block_file(
                checkpoint_dir, rank,
                canonical_name=canonical_name,
                legacy_name=legacy_name,
                block_files_legacy=block_files_legacy,
            )
            if manager.use_rdma:
                manager.register_buffer(block)
            native.sw_send_data(block_idx, int(block.data_ptr()), block.numel())
            logger.info(
                "EC-NAIVE legacy sw: rig=%d sent d_2,%d (block_idx=%d, %d bytes)",
                sender_rig, j, block_idx, block.numel(),
            )
            _t['network_encode'] = _time() - _t0_net
            _t0_rebuild = _time()
            state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
                main_payload, flat_key_roots=flat_key_roots,
            )
            _t['rebuild_sd'] = _time() - _t0_rebuild
        else:
            # This rank is in group but doesn't hold a data block for the failed rank
            # (parity channels only — not participating in SW recovery)
            logger.info(
                "EC-NAIVE legacy sw: righ=%d no data block, no-op", rank_in_group,
            )
            _t['network_encode'] = _time() - _t0_net
            _t0_rebuild = _time()
            state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
                main_payload, flat_key_roots=flat_key_roots,
            )
            _t['rebuild_sd'] = _time() - _t0_rebuild

    # NOTE: the old standalone else clause for ranks 0,1 (k=2) is absorbed into
    # the generalized else branch above.

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(
        f"EC-NAIVE legacy sw: done in {_time() - _t0_func:.2f}s (rank {rank})"
    )
    # NOTE: do not call manager.cleanup() here in the software failure path.
    # cleanup() calls native.stop() which tears down C++ resources, and the
    # subsequent reference drop triggers the C++ destructor (double-free on the
    # software-only RDMA channel).  The process exits shortly after load,
    # so leaving cleanup to __del__ is safe.

    return state_dict


def _run_ecnaive_full_recovery(
    manager: ECNAIVEManager,
    rank: int,
    world_size: int,
    ecnaive_blocks: Dict[str, torch.Tensor],
    recv_buffers: Optional[Dict[str, torch.Tensor]],
) -> None:
    """Mirror megatron.core.dist_checkpointing.strategies.torch recovery send/recv layout."""
    native = manager._ecnaive_native
    if native is None:
        raise RuntimeError("EC-NAIVE native module is not initialized")

    net_config = manager._get_ecnaive_load_network_config(rank, world_size)
    rank_in_group = net_config["rank_in_group"]

    from time import time

    start_time = time()

    if rank_in_group == 2:
        if recv_buffers is None:
            raise RuntimeError("EC-NAIVE legacy load: recv_buffers required on rank_in_group 2")
        required_keys = [
            "p20_from_rank0",
            "d21_from_rank3",
            "d00_from_rank0",
            "d01_from_rank1",
            "d10_from_rank1",
            "p11_from_rank1",
            "d30_from_rank3",
            "d31_from_rank0",
        ]
        missing = [k for k in required_keys if k not in recv_buffers]
        if missing:
            raise RuntimeError(f"EC-NAIVE legacy load: missing recv buffers {missing}")

        required_blocks = ["data0", "recv_parity0", "recv_data1", "recv_parity1"]
        missing_b = [k for k in required_blocks if k not in ecnaive_blocks]
        if missing_b:
            raise RuntimeError(f"EC-NAIVE legacy load: missing output blocks {missing_b}")

        recv_addrs = {
            "p20": int(recv_buffers["p20_from_rank0"].data_ptr()),
            "d21": int(recv_buffers["d21_from_rank3"].data_ptr()),
            "d00": int(recv_buffers["d00_from_rank0"].data_ptr()),
            "d01": int(recv_buffers["d01_from_rank1"].data_ptr()),
            "d10": int(recv_buffers["d10_from_rank1"].data_ptr()),
            "p11": int(recv_buffers["p11_from_rank1"].data_ptr()),
            "d30": int(recv_buffers["d30_from_rank3"].data_ptr()),
            "d31": int(recv_buffers["d31_from_rank0"].data_ptr()),
        }
        output_addrs = {
            "data0": int(ecnaive_blocks["data0"].data_ptr()),
            "recv_parity0": int(ecnaive_blocks["recv_parity0"].data_ptr()),
            "recv_data1": int(ecnaive_blocks["recv_data1"].data_ptr()),
            "recv_parity1": int(ecnaive_blocks["recv_parity1"].data_ptr()),
        }
        aligned_block_size = ecnaive_blocks["data0"].numel()

        native.submit_ecnaive_load_recovery_full(
            recv_p20_addr=recv_addrs["p20"],
            recv_d21_addr=recv_addrs["d21"],
            recv_d00_addr=recv_addrs["d00"],
            recv_d01_addr=recv_addrs["d01"],
            recv_d10_addr=recv_addrs["d10"],
            recv_p11_addr=recv_addrs["p11"],
            recv_d30_addr=recv_addrs["d30"],
            recv_d31_addr=recv_addrs["d31"],
            output_data0_addr=output_addrs["data0"],
            output_recv_parity0_addr=output_addrs["recv_parity0"],
            output_recv_data1_addr=output_addrs["recv_data1"],
            output_recv_parity1_addr=output_addrs["recv_parity1"],
            size=aligned_block_size,
        )
        native.submit_load_recv_sentinel()
        logger.info(
            "EC-NAIVE legacy load: rank_in_group 2 waiting for load completion "
            f"(aligned_block_size={aligned_block_size})"
        )
        native.wait_for_load_completion()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(
            f"EC-NAIVE legacy load: hw recovery done in {time() - start_time:.2f}s (rank_in_group=2)"
        )
    else:
        aligned_block_size = ecnaive_blocks["data0"].numel()
        if rank_in_group == 0:
            send_addrs = {
                "p20": int(ecnaive_blocks["recv_parity0"].data_ptr()),
                "d00": int(ecnaive_blocks["data0"].data_ptr()),
                "d31": int(ecnaive_blocks["recv_data1"].data_ptr()),
            }
            native.submit_load_send_rank0_parity0(
                send_addr=send_addrs["p20"], size=aligned_block_size
            )
            native.submit_load_send_rank0_data0(
                send_addr=send_addrs["d00"], size=aligned_block_size
            )
            native.submit_load_send_rank0_data1(
                send_addr=send_addrs["d31"], size=aligned_block_size
            )
            native.submit_load_send_sentinel()
        elif rank_in_group == 1:
            send_addrs = {
                "d01": int(ecnaive_blocks["recv_data1"].data_ptr()),
                "d10": int(ecnaive_blocks["data0"].data_ptr()),
                "p11": int(ecnaive_blocks["recv_parity1"].data_ptr()),
            }
            native.submit_load_send_rank1_data1(
                send_addr=send_addrs["d01"], size=aligned_block_size
            )
            native.submit_load_send_rank1_data0(
                send_addr=send_addrs["d10"], size=aligned_block_size
            )
            native.submit_load_send_rank1_parity1(
                send_addr=send_addrs["p11"], size=aligned_block_size
            )
            native.submit_load_send_sentinel()
        elif rank_in_group == 3:
            send_addrs = {
                "d21": int(ecnaive_blocks["recv_data1"].data_ptr()),
                "d30": int(ecnaive_blocks["data0"].data_ptr()),
            }
            native.submit_load_send_rank3_data1(
                send_addr=send_addrs["d21"], size=aligned_block_size
            )
            native.submit_load_send_rank3_data0(
                send_addr=send_addrs["d30"], size=aligned_block_size
            )
            native.submit_load_send_sentinel()
        else:
            raise RuntimeError(
                f"EC-NAIVE legacy load: unexpected rank_in_group={rank_in_group}"
            )
        logger.info(
            f"EC-NAIVE legacy load: submitted load sends in {time() - start_time:.2f}s "
            f"(rank_in_group={rank_in_group})"
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _infer_flat_key_roots(main_payload: Dict[str, Any]) -> Set[str]:
    """Infer flat key roots from checkpoint payload (for backward compatibility)."""
    if "flat_key_roots" in main_payload:
        return set(main_payload["flat_key_roots"])
    flat_key_roots: Set[str] = set()
    for info in main_payload.get("tensor_infos", []):
        first_seg = info.key.split('.')[0]
        if first_seg == "model" or (
            first_seg.startswith("model")
            and len(first_seg) > 5
            and first_seg[5:].isdigit()
        ):
            flat_key_roots.add(first_seg)
    return flat_key_roots


def state_dict_from_ecnaive_main_metadata_only(main_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build state_dict from ecnaive main file payload.
    When tensor_buffer is present, reconstruct full tensors; otherwise only non-tensor keys.
    Used when torch.distributed is not initialized yet (e.g. load_args_from_checkpoint).
    """
    flat_key_roots = _infer_flat_key_roots(main_payload)
    if isinstance(main_payload.get("tensor_buffer"), torch.Tensor):
        result = _reconstruct_full_state_dict_from_main_tensor_buffer(
            main_payload, flat_key_roots=flat_key_roots,
        )
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


def load_ecnaive_legacy_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    """
    Load EC-NAIVE torch legacy checkpoint.

    Normal load: always run 8-port recovery (aligned with torch_dist), then
    reconstruct state_dict from main tensor_buffer when present, else from
    decoded data0.

    Software failure (use_ecnaive_software_failure=True): rank_in_group=2
    reads d_{2,0} locally + receives d_{2,1} from rank_in_group=3 via C++
    (1 port), concatenates, and reconstructs.  No XOR decode or parity needed.
    """
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    main_payload = _load_ecnaive_main_payload(checkpoint_dir, rank, world_size)

    from megatron.training import get_args

    args = get_args()
    if not getattr(args, "use_ecnaive", False):
        logger.warning(
            "EC-NAIVE legacy load: args.use_ecnaive is False; enabling for native module init"
        )
        args.use_ecnaive = True

    manager = ECNAIVEManager()
    manager.init_ecnaive_if_enabled()
    if manager._ecnaive_native is None:
        raise RuntimeError("EC-NAIVE native module is not available in legacy load path")

    tensor_infos = main_payload["tensor_infos"]

    local_metadata = _tensor_infos_to_local_metadata(rank, tensor_infos)
    gathered_meta: List[Any]
    if world_size > 1 and torch.distributed.is_initialized():
        gathered_meta = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_meta, local_metadata)
        rank_metadata = {i: gathered_meta[i] for i in range(world_size)}
    else:
        rank_metadata = {0: local_metadata}

    global_registry = GlobalMetadataRegistry(rank_metadata=rank_metadata, rank_non_tensor_data={})

    # ---- Software failure fast path ----
    if bool(getattr(args, "use_ecnaive_software_failure", False)):
        logger.info("EC-NAIVE legacy: software failure recovery path")
        _t: Dict[str, float] = {}
        state_dict = _load_ecnaive_legacy_software_failure(
            checkpoint_dir=checkpoint_dir,
            rank=rank,
            world_size=world_size,
            manager=manager,
            main_payload=main_payload,
            global_registry=global_registry,
            timings=_t,
        )
        _t['total'] = _t.get('network_encode', 0) + _t.get('rebuild_sd', 0)
        logger.info(
            "EC-NAIVE legacy load timing (SW): "
            "total=%(total).2fs network_encode=%(network_encode).2fs "
            "rebuild_sd=%(rebuild_sd).2fs", _t
        )
        return state_dict

    # ---- Hardware recovery ----
    # Setup (not timed): init + buffer_alloc
    manager.init_ecnaive_load(rank, world_size)

    rank_in_group = manager._get_rank_in_group(rank, world_size)
    recv_buffers: Optional[Dict[str, torch.Tensor]] = None
    ecnaive_blocks: Dict[str, torch.Tensor]

    aligned_block_size = int(main_payload["aligned_block_size"])

    if rank_in_group == 2:
        recv_buffers = manager.allocate_ecnaive_load_recv_buffers(global_registry)
        data0, recv_parity1, recv_parity0, recv_data1 = allocate_hugepage_slices(
            aligned_block_size,
            4,
            touch_pages=True,
        )
        ecnaive_blocks = {
            "data0": data0,
            "recv_parity1": recv_parity1,
            "recv_parity0": recv_parity0,
            "recv_data1": recv_data1,
        }
        if manager.use_rdma:
            for _n, t in ecnaive_blocks.items():
                manager.register_buffer(t)
    else:
        ecnaive_blocks = _load_blocks_from_disk(checkpoint_dir, rank)
        if manager.use_rdma:
            for _n, t in ecnaive_blocks.items():
                manager.register_buffer(t)

    # === timing: network/encode (C++ pipeline) ===
    _t_hw: Dict[str, float] = {}
    _t0 = time.time()
    _run_ecnaive_full_recovery(
        manager=manager,
        rank=rank,
        world_size=world_size,
        ecnaive_blocks=ecnaive_blocks,
        recv_buffers=recv_buffers,
    )
    _t_hw['network_encode'] = time.time() - _t0

    # Backward compatibility: checkpoints saved before flat_key_roots existed.
    flat_key_roots = _infer_flat_key_roots(main_payload)

    _t0 = time.time()
    if isinstance(main_payload.get("tensor_buffer"), torch.Tensor):
        state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
            main_payload, flat_key_roots=flat_key_roots,
        )
    else:
        state_dict = _reconstruct_state_dict_from_main_and_data0(
            main_payload=main_payload,
            data0_uint8=ecnaive_blocks["data0"],
            manager=manager,
            flat_key_roots=flat_key_roots,
        )
    _t_hw['rebuild_sd'] = time.time() - _t0
    _t_hw['total'] = _t_hw['network_encode'] + _t_hw['rebuild_sd']

    logger.info(
        "EC-NAIVE legacy load timing (HW): "
        "total=%(total).2fs network_encode=%(network_encode).2fs "
        "rebuild_sd=%(rebuild_sd).2fs", _t_hw
    )

    # Clean up EC-NAIVE native module after load to prevent segfaults:
    # C++ worker threads and RDMA connections remain alive after recovery and
    # could access freed memory once local tensors (ecnaive_blocks, recv_buffers)
    # go out of scope. Also reset _ecnaive_native so the next save will
    # reinitialize the C++ module from scratch.
    logger.info(f"EC-NAIVE legacy load: cleaning up native module (rank {rank})")
    manager.cleanup()
    manager._ecnaive_native = None

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return state_dict


def load_ecnaive_legacy_checkpoint_hardware_recovery(
    checkpoint_name: str, failed_global_ranks: List[int]
) -> Dict[str, Any]:
    """Hardware recovery for 1-2 failed ranks using C++ ASIO send/recv + RS decode.

    Source ranks read surviving blocks from disk and send via existing C++ ASIO
    send channels. Failed ranks receive via C++ ASIO recv channels, then use the
    16-worker RS decode pool to recover lost data blocks.
    """
    start_time = time.time()
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    if not torch.distributed.is_initialized():
        raise RuntimeError("EC-NAIVE hardware recovery requires torch.distributed")

    from megatron.training import get_args
    args = get_args()
    ecnaive_k = getattr(args, "ecnaive_rs_k", 2)
    ecnaive_n = ecnaive_k + 2
    failed_set = set(failed_global_ranks)

    manager = ECNAIVEManager()
    manager.ecnaive_k = ecnaive_k
    manager.ecnaive_n = ecnaive_n
    if not getattr(args, "use_ecnaive", False):
        args.use_ecnaive = True

    # Init C++ native module for ALL ranks (ASIO connections needed for send/recv)
    manager.init_ecnaive_if_enabled()
    native = manager._ecnaive_native
    if native is None:
        raise RuntimeError("EC-NAIVE native module unavailable for hardware recovery")

    # Step 1: Load main payload + exchange metadata
    main_payload = _load_ecnaive_main_payload(checkpoint_dir, rank, world_size)
    tensor_infos = main_payload.get("tensor_infos", [])
    local_metadata = _tensor_infos_to_local_metadata(rank, tensor_infos)
    gathered_meta: List[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_meta, local_metadata)

    # Step 2: Compute recovery plan
    recovery_plan = manager.get_multi_failure_recovery_plan(
        failed_global_ranks, world_size
    )

    block_files = main_payload.get("block_files", {})
    pipeline_total_bytes = int(main_payload.get("pipeline_total_bytes", 0))
    block_data_size = int(main_payload.get("block_data_size",
                          (pipeline_total_bytes + ecnaive_k - 1) // ecnaive_k))
    flat_key_roots = _infer_flat_key_roots(main_payload)
    actual_tensor_size = int(main_payload.get("actual_tensor_size", 0))

    is_failed = (rank in failed_set)

    # Determine contributions: list of (failed_rank, label, recv_slot, j_or_parity)
    my_contributions: List[Tuple[int, str, int, Any]] = []
    for fr, plan in recovery_plan.items():
        for src_rank, label, recv_slot, j_or_parity in plan['surviving']:
            if src_rank == rank:
                my_contributions.append((fr, label, recv_slot, j_or_parity))

    torch.distributed.barrier()

    # === timing: network/encode (C++ send/recv + RS decode + assemble) ===
    _t: Dict[str, float] = {}
    _t0_net = time.time()

    # Helper: resolve block file name from label and recv_slot
    legacy_recv_names = {0: "recv_parity1", 1: "recv_parity0", 2: "recv_data1"}

    def _resolve_block_filename(label: str, recv_slot: int) -> str:
        if label.startswith('data_'):
            if recv_slot >= 0:
                key = f"recv_{recv_slot}"
                return block_files.get(key) or block_files.get(legacy_recv_names.get(recv_slot, "")) or f"ecnaive_block_rank{rank}_{key}.pt"
            else:
                return block_files.get("own_data0") or f"ecnaive_block_rank{rank}_own_data0.pt"
        elif label == 'parity0':
            key = f"recv_{ecnaive_k - 1}"
            return block_files.get(key) or f"ecnaive_block_rank{rank}_{key}.pt"
        elif label == 'parity1':
            key = f"recv_{ecnaive_k}"
            return block_files.get(key) or f"ecnaive_block_rank{rank}_{key}.pt"
        return ""

    num_channels = ecnaive_n - 1  # = k + 1

    if is_failed:
        plan = recovery_plan[rank]
        lost_positions = plan['lost_positions']
        m = len(lost_positions)
        if m == 0:
            logger.warning(f"EC-NAIVE hw recovery: rank {rank} has no lost blocks")
            return _reconstruct_full_state_dict_from_main_tensor_buffer(
                main_payload, flat_key_roots=flat_key_roots,
            )

        # Allocate recv buffers for all surviving blocks
        surviving_tensors: Dict[str, torch.Tensor] = {}
        surviving_addrs_ordered: List[str] = []  # ordered: data by pos, then parity

        data_labels = []
        parity_labels = []
        for _, label, _, _ in plan['surviving']:
            buf = allocate_hugepage_tensor(block_data_size, fallback_pin_memory=True)
            if manager.use_rdma:
                manager.register_buffer(buf)
            surviving_tensors[label] = buf
            if label.startswith('data_'):
                data_labels.append(label)
            else:
                parity_labels.append(label)
        data_labels.sort(key=lambda x: int(x.split('_')[1]))
        surviving_addrs_ordered = data_labels + sorted(parity_labels)

        # Allocate recovery buffers for lost blocks
        recovered_blocks = [torch.zeros(block_data_size, dtype=torch.uint8) for _ in range(m)]

        # Submit recv tasks to C++ recv workers
        native.reset_encoding_completion_flags()
        for src_rank, label, recv_slot, _ in plan['surviving']:
            recv_ch = manager.get_recv_channel_from_source(rank, src_rank, world_size)
            buf_addr = int(surviving_tensors[label].data_ptr())
            logger.info(
                f"EC-NAIVE hw recovery: failed rank {rank} recv {label} "
                f"from rank {src_rank} via recv channel {recv_ch} ({block_data_size} bytes)"
            )
            native.submit_recv_task(recv_ch, buf_addr, block_data_size)

        # Send sentinels for all channels (unused channels complete immediately)
        native.submit_send_sentinels(num_channels)
        native.submit_recv_sentinels(num_channels)

        # Wait for all recv workers to finish
        native.wait_for_encoding_completion()

        # RS decode per 64MB chunk using the decode pool
        surviving_block_data = [surviving_tensors[label] for label in surviving_addrs_ordered]
        ecnaive_buffer_size = manager.ecnaive_buffer_size
        src_pos = 0
        while src_pos < block_data_size:
            take = min(ecnaive_buffer_size, block_data_size - src_pos)
            surviving_chunk_addrs = [int(sb.data_ptr()) + src_pos for sb in surviving_block_data]
            recovered_chunk_addrs = [int(rb.data_ptr()) + src_pos for rb in recovered_blocks]
            native.submit_ecnaive_decode_recovery(
                ecnaive_k, m, lost_positions,
                surviving_chunk_addrs, recovered_chunk_addrs, take,
            )
            src_pos += take

        # Reassemble tensor_buffer
        all_data_blocks: Dict[int, torch.Tensor] = {}
        for label, tensor in surviving_tensors.items():
            if label.startswith('data_'):
                j = int(label.split('_')[1])
                all_data_blocks[j] = tensor
        for i, pos in enumerate(lost_positions):
            all_data_blocks[pos] = recovered_blocks[i]

        ordered_blocks = [all_data_blocks[j] for j in range(ecnaive_k)]
        tensor_buffer = torch.cat(ordered_blocks, dim=0)
        if actual_tensor_size > 0:
            tensor_buffer = tensor_buffer[:actual_tensor_size]
        _t['network_encode'] = time.time() - _t0_net

        _t0_rebuild = time.time()
        state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
            {"tensor_buffer": tensor_buffer,
             "tensor_infos": tensor_infos,
             "non_tensor_data": main_payload.get("non_tensor_data", {})},
            flat_key_roots=flat_key_roots,
        )
        _t['rebuild_sd'] = time.time() - _t0_rebuild
        logger.info(
            f"EC-NAIVE hw recovery: rank {rank} recovered via RS decode "
            f"(k={ecnaive_k}, m={m}, lost_pos={lost_positions})"
        )
        manager.cleanup()
        manager._ecnaive_native = None

    elif my_contributions:
        # Source rank: read blocks from disk and send via C++ ASIO send channels
        native.reset_encoding_completion_flags()

        # Keep references to block tensors until send workers finish (GC safety)
        block_tensors: List[torch.Tensor] = []
        for dest_fr, label, recv_slot, _ in my_contributions:
            filename = _resolve_block_filename(label, recv_slot)
            if not filename:
                continue
            block_path = checkpoint_dir / filename
            if not block_path.is_file():
                raise FileNotFoundError(
                    f"EC-NAIVE hw recovery: source rank {rank} missing block "
                    f"{label} for failed rank {dest_fr} at {block_path}"
                )
            from megatron.training.legacy_io_utils import is_raw_format, read_raw_block, MAGIC_BLOCK
            if is_raw_format(str(block_path), MAGIC_BLOCK):
                block_tensor = read_raw_block(str(block_path), MAGIC_BLOCK)[:block_data_size]
            else:
                payload = torch.load(block_path, map_location="cpu", weights_only=False)
                block_tensor = payload["tensor"].contiguous().view(torch.uint8)[:block_data_size]
            if manager.use_rdma:
                manager.register_buffer(block_tensor)
            block_tensors.append(block_tensor)  # keep alive
            send_ch = manager.get_send_channel_for_target(rank, dest_fr, world_size)
            logger.info(
                f"EC-NAIVE hw recovery: rank {rank} sending {label} "
                f"to failed rank {dest_fr} via send channel {send_ch} "
                f"({block_tensor.numel()} bytes)"
            )
            native.submit_send_task(send_ch, int(block_tensor.data_ptr()), block_tensor.numel())

        # Sentinels for all channels
        native.submit_send_sentinels(num_channels)
        native.submit_recv_sentinels(num_channels)
        native.wait_for_encoding_completion()
        _t['network_encode'] = time.time() - _t0_net

        # Load own state
        _t0_rebuild = time.time()
        state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
            main_payload, flat_key_roots=flat_key_roots,
        )
        _t['rebuild_sd'] = time.time() - _t0_rebuild

    else:
        _t0_rebuild = time.time()
        state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
            main_payload, flat_key_roots=flat_key_roots,
        )
        _t['rebuild_sd'] = time.time() - _t0_rebuild

    _t['total'] = _t['network_encode'] + _t['rebuild_sd']
    logger.info(
        "EC-NAIVE legacy load timing (HW recovery): "
        "total=%(total).2fs network_encode=%(network_encode).2fs "
        "rebuild_sd=%(rebuild_sd).2fs", _t
    )

    if world_size > 1:
        torch.distributed.barrier()

    return state_dict


def save_ecnaive_legacy_checkpoint(state_dict: Dict[str, Any], checkpoint_name: str) -> None:
    t0 = time.time()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    manager = ECNAIVEManager()
    manager.init_ecnaive_if_enabled()
    if manager._ecnaive_native is None:
        raise RuntimeError("EC-NAIVE native module is not available in legacy save path")

    flatten_optimizer_fp32_params(state_dict)
    t0 = time.time()
    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    logger.info(f"ECNAIVE save timing: decompose {time.time()-t0:.3f}s")

    start_time = t0 = time.time()
    safety_margin = max(int(total_tensor_size * 0.01), manager.ecnaive_buffer_size)
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
                    f"EC-NAIVE legacy save: tensor bytes mismatch for {info.key}, "
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
    logger.info(f"ECNAIVE save timing: D2H+copy {time.time()-t0:.3f}s")

    t0 = time.time()
    rank_metadata, _ = _build_global_registry(local_tensor_metadata, {})
    logger.info(f"ECNAIVE save timing: metadata exchange {time.time()-t0:.3f}s")

    t0 = time.time()
    blocks = _allocate_ecnaive_blocks(manager, rank_metadata)
    logger.info(f"ECNAIVE save timing: block alloc {time.time()-t0:.3f}s")

    t0 = time.time()
    if manager.use_rdma:
        manager.register_buffer(tensor_buffer)
    logger.info(f"ECNAIVE save timing: RDMA reg {time.time()-t0:.3f}s")

    t0 = time.time()
    _encode_with_native(
        manager=manager,
        tensor_buffer=tensor_buffer,
        actual_data_bytes=total_tensor_size,
        ecnaive_blocks=blocks,
    )
    logger.info(f"ECNAIVE save timing: encode {time.time()-t0:.3f}s")
    torch.distributed.barrier()
    logger.info(f"EC-NAIVE legacy save: done in {time.time() - start_time:.2f}s")

    _save_ecnaive_pt_files(
        checkpoint_name=checkpoint_name,
        rank=rank,
        non_tensor_data=decomposed.non_tensor_data,
        tensor_infos=decomposed.tensor_infos,
        blocks=blocks,
        full_tensor_buffer=tensor_buffer[:total_tensor_size],
        flat_key_roots=decomposed.flat_key_roots,
        manager=manager,
    )

    if world_size > 1:
        torch.distributed.barrier()
