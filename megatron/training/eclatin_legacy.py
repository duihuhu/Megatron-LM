import ctypes
import queue
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.dist_checkpointing.strategies.eclatin_manager import ECLATINManager
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


def _cpu_uint8_view(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    return t.contiguous().view(torch.uint8).reshape(-1)


def _build_global_registry(
    local_metadata: List[TensorMetadata], local_non_tensor: Dict[str, Any]
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


def _allocate_eclatin_blocks_legacy(
    manager: ECLATINManager,
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

    eclatin_buffer_size = manager.eclatin_buffer_size
    half_max_total_bytes = max_total_bytes // 2
    aligned_half_block_size = (
        (half_max_total_bytes + eclatin_buffer_size - 1) // eclatin_buffer_size
    ) * eclatin_buffer_size

    data_block_1, data_block_2, parity_block_1, parity_block_2 = \
        manager.allocate_preallocated_blocks(4, aligned_half_block_size)

    return {
        "data_block_1": data_block_1,
        "data_block_2": data_block_2,
        "parity_block_1": parity_block_1,
        "parity_block_2": parity_block_2,
        "actual_size": own_total_size,
        "pipeline_size": max_total_bytes,
        "aligned_size": aligned_half_block_size,
    }


def _encode_eclatin_with_native(
    manager: ECLATINManager,
    tensor_buffer: torch.Tensor,
    actual_data_bytes: int,
    pipeline_total_bytes: int,
    eclatin_blocks: Dict[str, Any],
) -> None:
    native = manager._eclatin_native
    if native is None:
        raise RuntimeError("ECLATIN native module is not initialized")

    buffers = manager.get_eclatin_buffers()
    if buffers is None:
        raise RuntimeError(
            "ECLATIN legacy save: buffer pools are not initialized "
            "(disable --use-eclatin-layerwise for legacy ECLATIN save)"
        )

    free_data_queue = buffers["free_data_buffer_queue"]
    free_recv_queue = buffers["free_recv_buffer_queue"]
    active_event = buffers.get("buffer_poller_active_event")
    poll_and_release = buffers.get("poll_and_release_buffers")

    def get_free_data_buffer():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_data_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECLATIN legacy: timeout waiting for free data buffer")
            return free_data_queue.get()

    def get_free_recv_buffer():
        if poll_and_release is not None:
            poll_and_release()
        try:
            return free_recv_queue.get(timeout=5.0)
        except queue.Empty:
            logger.error("ECLATIN legacy: timeout waiting for free recv buffer")
            return free_recv_queue.get()

    data_block_1 = eclatin_blocks["data_block_1"]
    data_block_2 = eclatin_blocks["data_block_2"]
    parity_block_1 = eclatin_blocks["parity_block_1"]
    parity_block_2 = eclatin_blocks["parity_block_2"]

    data_block_1_base = int(data_block_1.data_ptr())
    data_block_2_base = int(data_block_2.data_ptr())
    parity_block_1_base = int(parity_block_1.data_ptr())
    parity_block_2_base = int(parity_block_2.data_ptr())

    data_block_1_offset = 0
    data_block_2_offset = 0
    parity_block_1_offset = 0
    parity_block_2_offset = 0

    aligned_block_size = eclatin_blocks["aligned_size"]
    block_size = aligned_block_size
    eclatin_buffer_size = manager.eclatin_buffer_size

    total_bytes = pipeline_total_bytes
    half_total = total_bytes // 2
    src_pos = 0

    logger.info(
        f"ECLATIN legacy: processing {total_bytes / (1024**3):.2f} GB "
        f"(actual: {actual_data_bytes / (1024**3):.2f} GB) "
        f"in chunks of {eclatin_buffer_size / (1024**2):.0f} MB"
    )

    native.reset_encoding_completion_flags()

    if active_event is not None:
        active_event.set()

    try:
        while src_pos < half_total:
            remaining_in_source = total_bytes - src_pos
            take = min(eclatin_buffer_size, remaining_in_source)

            buffer1_addr = get_free_data_buffer()
            buffer2_addr = get_free_data_buffer()
            buffer3_addr = get_free_data_buffer()
            buffer4_addr = get_free_data_buffer()

            buffer1_ptr = ctypes.cast(buffer1_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer1_array = ctypes.cast(buffer1_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
            buffer3_ptr = ctypes.cast(buffer3_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer3_array = ctypes.cast(buffer3_ptr, ctypes.POINTER(ctypes.c_uint8 * take))

            src_pos_half1 = src_pos
            if src_pos_half1 < half_total:
                bytes_to_copy_half1 = min(take, half_total - src_pos_half1)
                if src_pos_half1 < actual_data_bytes:
                    actual_bytes_half1 = min(
                        bytes_to_copy_half1, actual_data_bytes - src_pos_half1
                    )
                    src_base_ptr = tensor_buffer.data_ptr()
                    src_addr_half1 = src_base_ptr + src_pos_half1
                    ctypes.memmove(buffer1_array.contents, src_addr_half1, actual_bytes_half1)
                    ctypes.memmove(buffer3_array.contents, src_addr_half1, actual_bytes_half1)

                    if bytes_to_copy_half1 > actual_bytes_half1:
                        padding_size = bytes_to_copy_half1 - actual_bytes_half1
                        padding_ptr1 = ctypes.cast(
                            ctypes.addressof(buffer1_array.contents) + actual_bytes_half1,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        padding_ptr3 = ctypes.cast(
                            ctypes.addressof(buffer3_array.contents) + actual_bytes_half1,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        ctypes.memset(padding_ptr1, 0, padding_size)
                        ctypes.memset(padding_ptr3, 0, padding_size)

                    if take > bytes_to_copy_half1:
                        remaining_padding = take - bytes_to_copy_half1
                        remaining_ptr1 = ctypes.cast(
                            ctypes.addressof(buffer1_array.contents) + bytes_to_copy_half1,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        remaining_ptr3 = ctypes.cast(
                            ctypes.addressof(buffer3_array.contents) + bytes_to_copy_half1,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        ctypes.memset(remaining_ptr1, 0, remaining_padding)
                        ctypes.memset(remaining_ptr3, 0, remaining_padding)
                else:
                    ctypes.memset(buffer1_array.contents, 0, take)
                    ctypes.memset(buffer3_array.contents, 0, take)
            else:
                ctypes.memset(buffer1_array.contents, 0, take)
                ctypes.memset(buffer3_array.contents, 0, take)

            buffer2_ptr = ctypes.cast(buffer2_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer2_array = ctypes.cast(buffer2_ptr, ctypes.POINTER(ctypes.c_uint8 * take))
            buffer4_ptr = ctypes.cast(buffer4_addr, ctypes.POINTER(ctypes.c_uint8))
            buffer4_array = ctypes.cast(buffer4_ptr, ctypes.POINTER(ctypes.c_uint8 * take))

            src_pos_in_second_half = half_total + src_pos

            if src_pos_in_second_half < total_bytes:
                bytes_to_copy_half2 = min(take, total_bytes - src_pos_in_second_half)
                if src_pos_in_second_half < actual_data_bytes:
                    actual_bytes_half2 = min(
                        bytes_to_copy_half2, actual_data_bytes - src_pos_in_second_half
                    )
                    src_base_ptr = tensor_buffer.data_ptr()
                    src_addr_half2 = src_base_ptr + src_pos_in_second_half
                    ctypes.memmove(buffer2_array.contents, src_addr_half2, actual_bytes_half2)
                    ctypes.memmove(buffer4_array.contents, src_addr_half2, actual_bytes_half2)

                    if bytes_to_copy_half2 > actual_bytes_half2:
                        padding_size = bytes_to_copy_half2 - actual_bytes_half2
                        padding_ptr2 = ctypes.cast(
                            ctypes.addressof(buffer2_array.contents) + actual_bytes_half2,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        padding_ptr4 = ctypes.cast(
                            ctypes.addressof(buffer4_array.contents) + actual_bytes_half2,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        ctypes.memset(padding_ptr2, 0, padding_size)
                        ctypes.memset(padding_ptr4, 0, padding_size)

                    if take > bytes_to_copy_half2:
                        remaining_padding = take - bytes_to_copy_half2
                        remaining_ptr2 = ctypes.cast(
                            ctypes.addressof(buffer2_array.contents) + bytes_to_copy_half2,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        remaining_ptr4 = ctypes.cast(
                            ctypes.addressof(buffer4_array.contents) + bytes_to_copy_half2,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        ctypes.memset(remaining_ptr2, 0, remaining_padding)
                        ctypes.memset(remaining_ptr4, 0, remaining_padding)
                else:
                    ctypes.memset(buffer2_array.contents, 0, take)
                    ctypes.memset(buffer4_array.contents, 0, take)
            else:
                ctypes.memset(buffer2_array.contents, 0, take)
                ctypes.memset(buffer4_array.contents, 0, take)

            data_block_1_offset_aligned = ((data_block_1_offset + 63) // 64) * 64
            data_block_2_offset_aligned = ((data_block_2_offset + 63) // 64) * 64

            if data_block_1_offset_aligned + take > block_size:
                logger.warning("ECLATIN legacy: data_block_1 exhausted")
                break
            if data_block_2_offset_aligned + take > block_size:
                logger.warning("ECLATIN legacy: data_block_2 exhausted")
                break

            data_block_1_write_addr = data_block_1_base + data_block_1_offset_aligned
            data_block_2_write_addr = data_block_2_base + data_block_2_offset_aligned

            data_block_1_ptr = ctypes.cast(data_block_1_write_addr, ctypes.POINTER(ctypes.c_uint8))
            data_block_2_ptr = ctypes.cast(data_block_2_write_addr, ctypes.POINTER(ctypes.c_uint8))

            half_actual_data = actual_data_bytes // 2

            if src_pos_half1 < half_actual_data:
                bytes_to_write_half1 = min(take, half_actual_data - src_pos_half1)
                if src_pos_half1 < actual_data_bytes:
                    actual_write_half1 = min(
                        bytes_to_write_half1, actual_data_bytes - src_pos_half1
                    )
                    ctypes.memmove(data_block_1_ptr, buffer1_array.contents, actual_write_half1)
                else:
                    ctypes.memmove(data_block_1_ptr, buffer1_array.contents, bytes_to_write_half1)
            else:
                ctypes.memset(data_block_1_ptr, 0, take)

            if src_pos_half1 >= half_actual_data and src_pos_half1 < actual_data_bytes:
                src_pos_in_second_half_mapped = src_pos_half1
                bytes_to_write_half2 = min(take, actual_data_bytes - src_pos_in_second_half_mapped)
                actual_write_half2 = min(
                    bytes_to_write_half2, actual_data_bytes - src_pos_in_second_half_mapped
                )
                ctypes.memmove(data_block_2_ptr, buffer2_array.contents, actual_write_half2)
            else:
                ctypes.memset(data_block_2_ptr, 0, take)

            data_block_1_offset = data_block_1_offset_aligned + take
            data_block_2_offset = data_block_2_offset_aligned + take

            recv1_addr_parity1 = get_free_recv_buffer()
            recv2_addr_parity1 = get_free_recv_buffer()
            recv1_addr_parity2 = get_free_recv_buffer()
            recv2_addr_parity2 = get_free_recv_buffer()

            parity_block_1_offset_aligned = ((parity_block_1_offset + 63) // 64) * 64
            parity_block_2_offset_aligned = ((parity_block_2_offset + 63) // 64) * 64

            if parity_block_1_offset_aligned + take > block_size:
                logger.warning("ECLATIN legacy: parity_block_1 exhausted")
                break
            if parity_block_2_offset_aligned + take > block_size:
                logger.warning("ECLATIN legacy: parity_block_2 exhausted")
                break

            parity_block_1_write_addr = parity_block_1_base + parity_block_1_offset_aligned
            parity_block_2_write_addr = parity_block_2_base + parity_block_2_offset_aligned

            parity_block_1_offset = parity_block_1_offset_aligned + take
            parity_block_2_offset = parity_block_2_offset_aligned + take

            assert parity_block_1_write_addr % 64 == 0
            assert parity_block_2_write_addr % 64 == 0

            native.submit_parity1_send1(buffer1_addr, take)
            native.submit_parity1_send2(buffer2_addr, take)
            native.submit_parity1_recv_xor(
                recv1_addr_parity1,
                recv2_addr_parity1,
                parity_block_1_write_addr,
                take,
            )

            native.submit_parity2_send1(buffer3_addr, take)
            native.submit_parity2_send2(buffer4_addr, take)
            native.submit_parity2_recv_xor(
                recv1_addr_parity2,
                recv2_addr_parity2,
                parity_block_2_write_addr,
                take,
            )

            src_pos += take

        native.submit_parity1_send1(0, 0)
        native.submit_parity1_send2(0, 0)
        native.submit_parity1_recv_xor(0, 0, 0, 0)
        native.submit_parity2_send1(0, 0)
        native.submit_parity2_send2(0, 0)
        native.submit_parity2_recv_xor(0, 0, 0, 0)

        native.wait_for_encoding_completion()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        if active_event is not None:
            active_event.clear()


def _save_eclatin_pt_files(
    checkpoint_name: str,
    rank: int,
    non_tensor_data: Dict[str, Any],
    tensor_infos: List[Any],
    blocks: Dict[str, Any],
    full_tensor_buffer: torch.Tensor,
) -> None:
    checkpoint_path = Path(checkpoint_name)
    checkpoint_dir = checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    main_file = checkpoint_dir / f"eclatin_main_rank{rank}.pt"
    from megatron.training.legacy_io_utils import MAGIC_ECLATIN, MAGIC_BLOCK

    block_file_map = {
        "data_block_1": f"eclatin_block_rank{rank}_data_block_1.pt",
        "data_block_2": f"eclatin_block_rank{rank}_data_block_2.pt",
        "parity_block_1": f"eclatin_block_rank{rank}_parity_block_1.pt",
        "parity_block_2": f"eclatin_block_rank{rank}_parity_block_2.pt",
    }

    # Pre-serialize metadata + prepare memoryview for main file
    import pickle as _pickle
    meta1 = _pickle.dumps(non_tensor_data)
    meta2 = _pickle.dumps(tensor_infos)
    extra = _pickle.dumps({
        "version": 1, "format": "eclatin_torch_legacy", "rank": rank,
        "actual_tensor_size": blocks["actual_size"],
        "pipeline_total_bytes": blocks["pipeline_size"],
        "aligned_block_size": blocks["aligned_size"],
        "block_files": block_file_map,
    })
    buf = full_tensor_buffer[: blocks["actual_size"]]
    if not buf.is_contiguous():
        buf = buf.contiguous()
    if buf.device.type != "cpu":
        buf = buf.to("cpu")
    main_mv = memoryview(buf.numpy())

    # Pre-prepare block memoryviews
    block_names = ("data_block_1", "data_block_2", "parity_block_1", "parity_block_2")
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
        futs = [ex.submit(write_main_prepared, str(main_file), MAGIC_ECLATIN,
                          meta1, meta2, extra, main_mv, blocks["actual_size"])]
        for name in block_names:
            block_file = checkpoint_dir / f"eclatin_block_rank{rank}_{name}.pt"
            futs.append(ex.submit(write_block_prepared,
                                  str(block_file), MAGIC_BLOCK,
                                  block_mvs[name], blocks[name].numel()))
        for f in futs:
            f.result()


def save_eclatin_legacy_checkpoint(state_dict: Dict[str, Any], checkpoint_name: str) -> None:
    t0 = time.time()
    from megatron.training import get_args

    args = get_args()
    if getattr(args, "use_eclatin_layerwise", False):
        raise RuntimeError(
            "ECLATIN legacy save does not support --use-eclatin-layerwise; "
            "disable layerwise or use torch_dist checkpoint format."
        )

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    manager = ECLATINManager()
    manager.init_eclatin_if_enabled()
    if manager._eclatin_native is None:
        raise RuntimeError("ECLATIN native module is not available in legacy save path")

    flatten_optimizer_fp32_params(state_dict)
    t0 = time.time()
    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    logger.info(f"ECLATIN save timing: decompose {time.time()-t0:.3f}s")

    start_time = t0 = time.time()
    safety_margin = max(int(total_tensor_size * 0.01), manager.eclatin_buffer_size)
    manager.allocate_preallocated_buffer(total_tensor_size + safety_margin)
    tensor_buffer = manager.preallocated_cpu_buffer

    offset = 0
    local_tensor_metadata: List[TensorMetadata] = []
    for i, (info, tensor) in enumerate(zip(decomposed.tensor_infos, decomposed.tensor_data)):
        tensor_bytes = info.size_bytes
        tensor_bytes_view = _cpu_uint8_view(tensor)
        if tensor_bytes_view.numel() != tensor_bytes:
            raise RuntimeError(
                f"ECLATIN legacy save: tensor bytes mismatch for {info.key}, "
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
        decomposed.tensor_data[i] = None  # free GPU tensor ref immediately

    del decomposed.tensor_data  # drop remaining refs
    logger.info(f"ECLATIN save timing: D2H+copy {time.time()-t0:.3f}s")

    t0 = time.time()
    rank_metadata, _ = _build_global_registry(local_tensor_metadata, {})
    logger.info(f"ECLATIN save timing: metadata exchange {time.time()-t0:.3f}s")

    t0 = time.time()
    blocks = _allocate_eclatin_blocks_legacy(manager, rank_metadata)
    logger.info(f"ECLATIN save timing: block alloc {time.time()-t0:.3f}s")

    t0 = time.time()
    if manager.use_rdma:
        manager.register_buffer(tensor_buffer)
    logger.info(f"ECLATIN save timing: RDMA reg {time.time()-t0:.3f}s")

    t0 = time.time()
    _encode_eclatin_with_native(
        manager=manager,
        tensor_buffer=tensor_buffer,
        actual_data_bytes=total_tensor_size,
        pipeline_total_bytes=blocks["pipeline_size"],
        eclatin_blocks=blocks,
    )
    logger.info(f"ECLATIN save timing: encode {time.time()-t0:.3f}s")

    torch.distributed.barrier()
    logger.info(f"ECLATIN legacy save: done in {time.time() - start_time:.2f}s")

    _save_eclatin_pt_files(
        checkpoint_name=checkpoint_name,
        rank=rank,
        non_tensor_data=decomposed.non_tensor_data,
        tensor_infos=decomposed.tensor_infos,
        blocks=blocks,
        full_tensor_buffer=tensor_buffer[:total_tensor_size],
    )

    if world_size > 1:
        torch.distributed.barrier()


def _checkpoint_dir_from_path(checkpoint_name: str) -> Path:
    checkpoint_path = Path(checkpoint_name)
    return checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent


def _tensor_infos_to_local_metadata(rank: int, tensor_infos: List[Any]) -> List[TensorMetadata]:
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


def _load_eclatin_main_payload(checkpoint_dir: Path, rank: int, world_size: int) -> Dict[str, Any]:
    """
    Load eclatin_main_rank{rank}.pt. If missing on this rank, recover payload via all_gather_object.
    """
    main_path = checkpoint_dir / f"eclatin_main_rank{rank}.pt"
    local_payload: Optional[Dict[str, Any]] = None
    if main_path.is_file():
        from megatron.training.legacy_io_utils import is_raw_format, read_raw_checkpoint, MAGIC_ECLATIN
        if is_raw_format(str(main_path), MAGIC_ECLATIN):
            local_payload = read_raw_checkpoint(str(main_path), MAGIC_ECLATIN)
        else:
            local_payload = torch.load(main_path, map_location="cpu", weights_only=False)

    if world_size <= 1 or not torch.distributed.is_initialized():
        if local_payload is None:
            raise FileNotFoundError(f"ECLATIN legacy: missing main file {main_path}")
        return local_payload

    gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_payload)

    chosen = gathered[rank]
    if chosen is None:
        raise FileNotFoundError(
            f"ECLATIN legacy: eclatin_main_rank{rank}.pt missing on all ranks under {checkpoint_dir}"
        )
    return chosen


def _copy_eclatin_block_file_into_tensor(
    checkpoint_dir: Path, rank: int, block_name: str, dest: torch.Tensor
) -> None:
    block_path = checkpoint_dir / f"eclatin_block_rank{rank}_{block_name}.pt"
    if not block_path.is_file():
        raise FileNotFoundError(f"ECLATIN legacy load: missing block file {block_path}")
    from megatron.training.legacy_io_utils import is_raw_format, read_raw_block, MAGIC_BLOCK
    if is_raw_format(str(block_path), MAGIC_BLOCK):
        src = read_raw_block(str(block_path), MAGIC_BLOCK)
    else:
        payload = torch.load(block_path, map_location="cpu", weights_only=False)
        src = payload["tensor"].contiguous().view(torch.uint8).reshape(-1)
    dst = dest.contiguous().view(-1)
    n = min(src.numel(), dst.numel())
    dst[:n].copy_(src[:n])


def _load_eclatin_blocks_from_disk_into(
    eclatin_blocks: Dict[str, Any],
    checkpoint_dir: Path,
    rank: int,
    rank_in_group: int,
    software_only: bool = False,
) -> None:
    """
    Load local block files into pre-allocated hugepage blocks (copy_, do not replace tensors).
    software_only + rank_in_group 2: load data_block_1 and data_block_2 for software-failure path.
    """
    if software_only and rank_in_group == 2:
        _copy_eclatin_block_file_into_tensor(
            checkpoint_dir, rank, "data_block_1", eclatin_blocks["data_block_1"]
        )
        _copy_eclatin_block_file_into_tensor(
            checkpoint_dir, rank, "data_block_2", eclatin_blocks["data_block_2"]
        )
        return

    if rank_in_group == 2:
        return

    if rank_in_group == 0:
        _copy_eclatin_block_file_into_tensor(
            checkpoint_dir, rank, "data_block_2", eclatin_blocks["data_block_2"]
        )
        _copy_eclatin_block_file_into_tensor(
            checkpoint_dir, rank, "parity_block_2", eclatin_blocks["parity_block_2"]
        )
    elif rank_in_group == 1:
        _copy_eclatin_block_file_into_tensor(
            checkpoint_dir, rank, "data_block_1", eclatin_blocks["data_block_1"]
        )
        _copy_eclatin_block_file_into_tensor(
            checkpoint_dir, rank, "parity_block_1", eclatin_blocks["parity_block_1"]
        )
    elif rank_in_group == 3:
        _copy_eclatin_block_file_into_tensor(
            checkpoint_dir, rank, "data_block_1", eclatin_blocks["data_block_1"]
        )
        _copy_eclatin_block_file_into_tensor(
            checkpoint_dir, rank, "data_block_2", eclatin_blocks["data_block_2"]
        )
    else:
        raise RuntimeError(
            f"ECLATIN legacy load: unexpected rank_in_group={rank_in_group}"
        )


def _reconstruct_state_dict_from_eclatin_buffer(
    main_payload: Dict[str, Any],
    recovered_buffer: Optional[torch.Tensor],
) -> Dict[str, Any]:
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
    )
    result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


def state_dict_from_eclatin_main_metadata_only(main_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build state_dict from eclatin main file; used when torch.distributed is not initialized."""
    if isinstance(main_payload.get("tensor_buffer"), torch.Tensor):
        result = _reconstruct_state_dict_from_eclatin_buffer(main_payload, None)
    else:
        decomposed = DecomposedStateDict(
            non_tensor_data=main_payload["non_tensor_data"],
            tensor_infos=[],
            tensor_data=[],
        )
        result = reconstruct_state_dict(decomposed)
    unflatten_optimizer_fp32_params(result)
    return result


def _max_tensor_bytes_from_registry(registry: GlobalMetadataRegistry, world_size: int) -> int:
    max_bytes = 0
    for r in range(world_size):
        rank_metadata = registry.rank_metadata.get(r, [])
        rank_total = sum(meta.size_bytes for meta in rank_metadata)
        if rank_total > max_bytes:
            max_bytes = rank_total
    return max_bytes


def _run_eclatin_full_recovery(
    manager: ECLATINManager,
    rank: int,
    world_size: int,
    eclatin_blocks: Dict[str, Any],
    recv_buffers: Optional[Dict[str, torch.Tensor]],
    recovered_buffer: Optional[torch.Tensor],
    total_size: int,
    registry: GlobalMetadataRegistry,
) -> None:
    from megatron.training import get_args as use_args
    from time import time

    native = manager._eclatin_native
    if native is None:
        raise RuntimeError("ECLATIN native module is not initialized")

    input_args = use_args()
    net_config = manager._get_eclatin_network_config(rank, world_size)
    rank_in_group = net_config["rank_in_group"]

    if input_args.use_eclatin_software_failure:
        if rank_in_group == 2:
            if recovered_buffer is None:
                raise RuntimeError("ECLATIN legacy load: software failure path needs recovered_buffer")
            start_time = time()
            actual_tensor_buffer_size = _max_tensor_bytes_from_registry(registry, world_size)
            half_actual_data = actual_tensor_buffer_size // 2
            if recovered_buffer.numel() >= total_size:
                first_half_actual = min(half_actual_data, total_size)
                recovered_buffer[:first_half_actual].copy_(
                    eclatin_blocks["data_block_1"][:first_half_actual]
                )
                if total_size > half_actual_data:
                    second_half_size = total_size - half_actual_data
                    recovered_buffer[first_half_actual:total_size].copy_(
                        eclatin_blocks["data_block_2"][:second_half_size]
                    )
            logger.info(
                f"ECLATIN legacy load: rank_in_group 2 software failure recovery done in "
                f"{time() - start_time:.2f}s"
            )
        else:
            logger.info(
                f"ECLATIN legacy load: rank_in_group {rank_in_group} no-op for software failure"
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    failed_rank = 2
    native.set_load_mode(True, failed_rank)
    logger.info(f"ECLATIN legacy load: set load mode (failed_rank={failed_rank})")

    rank2_ip = net_config["rank_ips"].get(net_config["load_receiver_rank"], net_config["my_ip"])
    load_recv_rank0_data2_port = net_config["ports"]["load_recv_rank0_data2"]
    load_recv_rank0_parity2_port = net_config["ports"]["load_recv_rank0_parity2"]
    load_recv_rank1_data1_port = net_config["ports"]["load_recv_rank1_data1"]
    load_recv_rank1_parity1_port = net_config["ports"]["load_recv_rank1_parity1"]
    load_recv_rank3_data1_port = net_config["ports"]["load_recv_rank3_data1"]
    load_recv_rank3_data2_port = net_config["ports"]["load_recv_rank3_data2"]

    if rank_in_group == 2:
        logger.info("ECLATIN legacy load: rank_in_group 2 init load accept connections")
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
            f"ECLATIN legacy load: rank_in_group {rank_in_group} connecting load send sockets"
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

    start_time = time()
    aligned_half_block_size = eclatin_blocks["data_block_1"].numel()

    if rank_in_group == 2:
        if recv_buffers is None or recovered_buffer is None:
            raise RuntimeError(
                "ECLATIN legacy load: rank_in_group 2 needs recv_buffers and recovered_buffer"
            )
        rank0_data2_addr = int(recv_buffers["rank0_data2"].data_ptr())
        rank0_parity2_addr = int(recv_buffers["rank0_parity2"].data_ptr())
        rank1_data1_addr = int(recv_buffers["rank1_data1"].data_ptr())
        rank1_parity1_addr = int(recv_buffers["rank1_parity1"].data_ptr())
        rank3_data1_addr = int(recv_buffers["rank3_data1"].data_ptr())
        rank3_data2_addr = int(recv_buffers["rank3_data2"].data_ptr())
        recovered_data1_addr = int(eclatin_blocks["data_block_1"].data_ptr())
        recovered_data2_addr = int(eclatin_blocks["data_block_2"].data_ptr())
        recovered_parity1_addr = int(eclatin_blocks["parity_block_1"].data_ptr())
        recovered_parity2_addr = int(eclatin_blocks["parity_block_2"].data_ptr())

        native.load_recover(
            rank0_data2_addr,
            rank0_parity2_addr,
            rank1_data1_addr,
            rank1_parity1_addr,
            rank3_data1_addr,
            rank3_data2_addr,
            recovered_data1_addr,
            recovered_data2_addr,
            recovered_parity1_addr,
            recovered_parity2_addr,
            aligned_half_block_size,
        )

        actual_tensor_buffer_size = _max_tensor_bytes_from_registry(registry, world_size)
        half_actual_data = actual_tensor_buffer_size // 2
        if recovered_buffer.numel() >= total_size:
            first_half_actual = min(half_actual_data, total_size)
            recovered_buffer[:first_half_actual].copy_(
                eclatin_blocks["data_block_1"][:first_half_actual]
            )
            if total_size > half_actual_data:
                second_half_size = total_size - half_actual_data
                recovered_buffer[first_half_actual:total_size].copy_(
                    eclatin_blocks["data_block_2"][:second_half_size]
                )
        logger.info(
            f"ECLATIN legacy load: hw recovery done in {time() - start_time:.2f}s (rank_in_group=2)"
        )
    elif rank_in_group == 0:
        data2_addr = int(eclatin_blocks["data_block_2"].data_ptr())
        parity2_addr = int(eclatin_blocks["parity_block_2"].data_ptr())
        native.load_send_blocks(
            "rank0_data2",
            data2_addr,
            "rank0_parity2",
            parity2_addr,
            aligned_half_block_size,
        )
    elif rank_in_group == 1:
        data1_addr = int(eclatin_blocks["data_block_1"].data_ptr())
        parity1_addr = int(eclatin_blocks["parity_block_1"].data_ptr())
        native.load_send_blocks(
            "rank1_data1",
            data1_addr,
            "rank1_parity1",
            parity1_addr,
            aligned_half_block_size,
        )
    elif rank_in_group == 3:
        data1_addr = int(eclatin_blocks["data_block_1"].data_ptr())
        data2_addr = int(eclatin_blocks["data_block_2"].data_ptr())
        native.load_send_blocks(
            "rank3_data1",
            data1_addr,
            "rank3_data2",
            data2_addr,
            aligned_half_block_size,
        )
    else:
        raise RuntimeError(
            f"ECLATIN legacy load: unexpected rank_in_group={rank_in_group}"
        )

    torch.distributed.barrier()


def load_eclatin_legacy_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    """
    Load ECLATIN torch legacy checkpoint: run recovery (aligned with torch_dist), then reconstruct
    state_dict from main tensor_buffer (rank_in_group 2 may use recovered_buffer).
    """
    start_time = time.time()
    from megatron.training import get_args

    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    main_payload = _load_eclatin_main_payload(checkpoint_dir, rank, world_size)

    args = get_args()
    if not getattr(args, "use_eclatin", False):
        logger.warning(
            "ECLATIN legacy load: args.use_eclatin is False; enabling for native module init"
        )
        args.use_eclatin = True

    manager = ECLATINManager()
    manager.init_eclatin_if_enabled()
    if manager._eclatin_native is None:
        raise RuntimeError("ECLATIN native module is not available in legacy load path")

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
        return _reconstruct_state_dict_from_eclatin_buffer(main_payload, None)

    eclatin_blocks = _allocate_eclatin_blocks_legacy(manager, rank_metadata)
    rank_in_group = manager._get_rank_in_group(rank, world_size)
    sw_failure = bool(getattr(args, "use_eclatin_software_failure", False))

    recv_buffers: Optional[Dict[str, torch.Tensor]] = None
    recovered_buffer: Optional[torch.Tensor] = None
    total_size = sum(meta.size_bytes for meta in rank_metadata.get(rank, []))

    if rank_in_group == 2:
        if sw_failure:
            _load_eclatin_blocks_from_disk_into(
                eclatin_blocks,
                checkpoint_dir,
                rank,
                rank_in_group,
                software_only=True,
            )
        else:
            recv_buffers = manager.allocate_eclatin_load_recv_buffers(registry)
        pin = torch.cuda.is_available() and getattr(manager, "eclatin_pin_memory", False)
        recovered_buffer = torch.empty(total_size, dtype=torch.uint8, pin_memory=pin)
    else:
        _load_eclatin_blocks_from_disk_into(
            eclatin_blocks,
            checkpoint_dir,
            rank,
            rank_in_group,
            software_only=False,
        )

    if manager.use_rdma:
        for key in ("data_block_1", "data_block_2", "parity_block_1", "parity_block_2"):
            manager.register_buffer(eclatin_blocks[key])
        if recv_buffers is not None:
            for t in recv_buffers.values():
                manager.register_buffer(t)
        if recovered_buffer is not None:
            manager.register_buffer(recovered_buffer)

    _run_eclatin_full_recovery(
        manager=manager,
        rank=rank,
        world_size=world_size,
        eclatin_blocks=eclatin_blocks,
        recv_buffers=recv_buffers,
        recovered_buffer=recovered_buffer,
        total_size=total_size,
        registry=registry,
    )

    state_dict = _reconstruct_state_dict_from_eclatin_buffer(
        main_payload,
        recovered_buffer=recovered_buffer if rank_in_group == 2 else None,
    )

    logger.info(f"ECLATIN legacy load: done in {time.time() - start_time:.2f}s")

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(f"ECLATIN legacy load: cleaning up native module (rank {rank})")
    manager.cleanup()
    manager._eclatin_native = None

    return state_dict
