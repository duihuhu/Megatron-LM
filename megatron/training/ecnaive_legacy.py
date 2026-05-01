import ctypes
import queue
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from megatron.core.dist_checkpointing.strategies.ecnaive_manager import ECNAIVEManager
from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
    allocate_hugepage_slices,
    allocate_hugepage_tensor,
)
from megatron.core.dist_checkpointing.strategies.state_dict_decomposer import (
    TensorMetadata,
    decompose_state_dict,
)

logger = getLogger(__name__)


def _cpu_uint8_view(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    return t.contiguous().view(torch.uint8)


def _build_global_registry(local_metadata: List[TensorMetadata], local_non_tensor: Dict[str, Any]) -> Tuple[Dict[int, List[TensorMetadata]], Dict[int, Dict[str, Any]]]:
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


def _allocate_ecnaive_blocks(
    manager: ECNAIVEManager,
    rank_metadata: Dict[int, List[TensorMetadata]],
) -> Dict[str, torch.Tensor]:
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

    half_max_total_bytes = max_total_bytes // 2
    aligned_half_block_size = (
        (half_max_total_bytes + manager.ecnaive_buffer_size - 1)
        // manager.ecnaive_buffer_size
    ) * manager.ecnaive_buffer_size

    data0, recv_parity1, recv_parity0, recv_data1 = allocate_hugepage_slices(
        aligned_half_block_size,
        4,
        touch_pages=True,
    )

    if manager.use_rdma:
        manager.register_buffer(data0)
        manager.register_buffer(recv_parity1)
        manager.register_buffer(recv_parity0)
        manager.register_buffer(recv_data1)

    return {
        "data0": data0,
        "recv_parity1": recv_parity1,
        "recv_parity0": recv_parity0,
        "recv_data1": recv_data1,
        "actual_size": own_total_size,
        "pipeline_size": max_total_bytes,
        "aligned_size": aligned_half_block_size,
    }


def _encode_with_native(
    manager: ECNAIVEManager,
    tensor_buffer: torch.Tensor,
    actual_data_bytes: int,
    pipeline_total_bytes: int,
    ecnaive_blocks: Dict[str, torch.Tensor],
) -> None:
    buffers = manager.get_ecnaive_buffers()
    if buffers is None:
        raise RuntimeError("EC-NAIVE buffers are not initialized")
    if manager._ecnaive_native is None:
        raise RuntimeError("EC-NAIVE native module is not initialized")

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

    data0 = ecnaive_blocks["data0"]
    recv_parity1 = ecnaive_blocks["recv_parity1"]
    recv_parity0 = ecnaive_blocks["recv_parity0"]
    recv_data1 = ecnaive_blocks["recv_data1"]

    data0_base = int(data0.data_ptr())
    recv_parity1_base = int(recv_parity1.data_ptr())
    recv_parity0_base = int(recv_parity0.data_ptr())
    recv_data1_base = int(recv_data1.data_ptr())

    data0_offset = 0
    recv_parity1_offset = 0
    recv_parity0_offset = 0
    recv_data1_offset = 0

    block_size = ecnaive_blocks["aligned_size"]
    ecnaive_buffer_size = manager.ecnaive_buffer_size
    half_total = pipeline_total_bytes // 2
    half_actual_data = actual_data_bytes // 2

    src_pos = 0
    native = manager._ecnaive_native
    native.reset_encoding_completion_flags()

    if active_event is not None:
        active_event.set()

    try:
        while src_pos < half_total:
            remaining_in_source = pipeline_total_bytes - src_pos
            take = min(ecnaive_buffer_size, remaining_in_source)

            data1_addr = get_free_data_buffer()
            parity0_addr = get_free_parity_buffer()
            parity1_addr = get_free_parity_buffer()

            data1_ptr = ctypes.cast(data1_addr, ctypes.POINTER(ctypes.c_uint8))
            data1_array = ctypes.cast(data1_ptr, ctypes.POINTER(ctypes.c_uint8 * take))

            src_pos_half1 = src_pos
            if src_pos_half1 < half_total:
                bytes_to_copy_half1 = min(take, half_total - src_pos_half1)
                if src_pos_half1 < actual_data_bytes:
                    actual_bytes_half1 = min(
                        bytes_to_copy_half1, actual_data_bytes - src_pos_half1
                    )
                    src_base_ptr = tensor_buffer.data_ptr()
                    src_addr_half1 = src_base_ptr + src_pos_half1
                    ctypes.memmove(data1_array.contents, src_addr_half1, actual_bytes_half1)

                    if bytes_to_copy_half1 > actual_bytes_half1:
                        padding_size = bytes_to_copy_half1 - actual_bytes_half1
                        padding_ptr = ctypes.cast(
                            ctypes.addressof(data1_array.contents) + actual_bytes_half1,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        ctypes.memset(padding_ptr, 0, padding_size)
                    if take > bytes_to_copy_half1:
                        remaining_padding = take - bytes_to_copy_half1
                        remaining_ptr = ctypes.cast(
                            ctypes.addressof(data1_array.contents) + bytes_to_copy_half1,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        ctypes.memset(remaining_ptr, 0, remaining_padding)
                else:
                    ctypes.memset(data1_array.contents, 0, take)
            else:
                ctypes.memset(data1_array.contents, 0, take)

            data0_offset_aligned = ((data0_offset + 63) // 64) * 64
            recv_parity1_offset_aligned = ((recv_parity1_offset + 63) // 64) * 64
            recv_parity0_offset_aligned = ((recv_parity0_offset + 63) // 64) * 64
            recv_data1_offset_aligned = ((recv_data1_offset + 63) // 64) * 64

            if (
                data0_offset_aligned + take > block_size
                or recv_parity1_offset_aligned + take > block_size
                or recv_parity0_offset_aligned + take > block_size
                or recv_data1_offset_aligned + take > block_size
            ):
                logger.warning("EC-NAIVE legacy: persistent block exhausted")
                break

            data0_write_addr = data0_base + data0_offset_aligned
            recv_parity1_write_addr = recv_parity1_base + recv_parity1_offset_aligned
            recv_parity0_write_addr = recv_parity0_base + recv_parity0_offset_aligned
            recv_data1_write_addr = recv_data1_base + recv_data1_offset_aligned

            data0_ptr = ctypes.cast(data0_write_addr, ctypes.POINTER(ctypes.c_uint8))
            if src_pos_half1 < half_actual_data:
                bytes_to_write_half1 = min(take, half_actual_data - src_pos_half1)
                if src_pos_half1 < actual_data_bytes:
                    actual_write_half1 = min(
                        bytes_to_write_half1, actual_data_bytes - src_pos_half1
                    )
                    src_base_ptr = tensor_buffer.data_ptr()
                    src_addr_half1 = src_base_ptr + src_pos_half1
                    ctypes.memmove(data0_ptr, src_addr_half1, actual_write_half1)
                    if take > actual_write_half1:
                        padding_ptr = ctypes.cast(
                            data0_write_addr + actual_write_half1,
                            ctypes.POINTER(ctypes.c_uint8),
                        )
                        ctypes.memset(padding_ptr, 0, take - actual_write_half1)
                else:
                    ctypes.memset(data0_ptr, 0, take)
            else:
                ctypes.memset(data0_ptr, 0, take)

            native.submit_ecnaive_save(
                data0_addr=data0_write_addr,
                data1_addr=data1_addr,
                parity0_addr=parity0_addr,
                parity1_addr=parity1_addr,
                recv_parity1_addr=recv_parity1_write_addr,
                recv_parity0_addr=recv_parity0_write_addr,
                recv_data1_addr=recv_data1_write_addr,
                size=take,
            )

            data0_offset = data0_offset_aligned + take
            recv_parity1_offset = recv_parity1_offset_aligned + take
            recv_parity0_offset = recv_parity0_offset_aligned + take
            recv_data1_offset = recv_data1_offset_aligned + take
            src_pos += take

        native.submit_send_data1_sentinel()
        native.submit_send_parity0_sentinel()
        native.submit_send_parity1_sentinel()
        native.submit_recv_parity1_sentinel()
        native.submit_recv_parity0_sentinel()
        native.submit_recv_data1_sentinel()
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
    blocks: Dict[str, torch.Tensor],
) -> None:
    checkpoint_path = Path(checkpoint_name)
    checkpoint_dir = checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    main_file = checkpoint_dir / f"ecnaive_main_rank{rank}.pt"
    torch.save(
        {
            "version": 1,
            "format": "ecnaive_torch_legacy",
            "rank": rank,
            "non_tensor_data": non_tensor_data,
            "tensor_infos": tensor_infos,
            "actual_tensor_size": blocks["actual_size"],
            "pipeline_total_bytes": blocks["pipeline_size"],
            "aligned_block_size": blocks["aligned_size"],
            "block_files": {
                "data0": f"ecnaive_block_rank{rank}_data0.pt",
                "recv_parity1": f"ecnaive_block_rank{rank}_recv_parity1.pt",
                "recv_parity0": f"ecnaive_block_rank{rank}_recv_parity0.pt",
                "recv_data1": f"ecnaive_block_rank{rank}_recv_data1.pt",
            },
        },
        main_file,
    )

    for block_name in ("data0", "recv_parity1", "recv_parity0", "recv_data1"):
        block_file = checkpoint_dir / f"ecnaive_block_rank{rank}_{block_name}.pt"
        torch.save(
            {
                "version": 1,
                "format": "ecnaive_torch_legacy",
                "rank": rank,
                "block_name": block_name,
                "tensor": blocks[block_name],
            },
            block_file,
        )


def save_ecnaive_legacy_checkpoint(state_dict: Dict[str, Any], checkpoint_name: str) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    manager = ECNAIVEManager()
    manager.init_ecnaive_if_enabled()
    if manager._ecnaive_native is None:
        raise RuntimeError("EC-NAIVE native module is not available in legacy save path")

    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes

    safety_margin = max(int(total_tensor_size * 0.01), manager.ecnaive_buffer_size)
    tensor_buffer = allocate_hugepage_tensor(
        total_tensor_size + safety_margin,
        fallback_pin_memory=manager.ecnaive_pin_memory and torch.cuda.is_available(),
    )
    tensor_buffer.zero_()

    offset = 0
    local_tensor_metadata: List[TensorMetadata] = []
    for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
        tensor_bytes = info.size_bytes
        tensor_buffer[offset : offset + tensor_bytes].copy_(_cpu_uint8_view(tensor))
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
    blocks = _allocate_ecnaive_blocks(manager, rank_metadata)

    if manager.use_rdma:
        manager.register_buffer(tensor_buffer)

    _encode_with_native(
        manager=manager,
        tensor_buffer=tensor_buffer,
        actual_data_bytes=total_tensor_size,
        pipeline_total_bytes=blocks["pipeline_size"],
        ecnaive_blocks=blocks,
    )

    _save_ecnaive_pt_files(
        checkpoint_name=checkpoint_name,
        rank=rank,
        non_tensor_data=decomposed.non_tensor_data,
        tensor_infos=decomposed.tensor_infos,
        blocks=blocks,
    )

    if world_size > 1:
        torch.distributed.barrier()
