import ctypes
import queue
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
    reconstruct_state_dict,
)

logger = getLogger(__name__)


def _cpu_uint8_view(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    return t.contiguous().view(torch.uint8).reshape(-1)


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
    full_tensor_buffer: torch.Tensor,
    flat_key_roots: Optional[Set[str]] = None,
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
            "tensor_buffer": full_tensor_buffer.contiguous().view(torch.uint8),
            "actual_tensor_size": blocks["actual_size"],
            "pipeline_total_bytes": blocks["pipeline_size"],
            "aligned_block_size": blocks["aligned_size"],
            "flat_key_roots": list(flat_key_roots) if flat_key_roots else [],
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
        # blocks are views from one shared base buffer; clone to avoid serializing the full base storage.
        block_tensor = blocks[block_name].contiguous().clone()
        torch.save(
            {
                "version": 1,
                "format": "ecnaive_torch_legacy",
                "rank": rank,
                "block_name": block_name,
                "tensor": block_tensor,
            },
            block_file,
        )


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
    blocks: Dict[str, torch.Tensor] = {}
    for block_name in ("data0", "recv_parity1", "recv_parity0", "recv_data1"):
        block_path = checkpoint_dir / f"ecnaive_block_rank{rank}_{block_name}.pt"
        if not block_path.is_file():
            raise FileNotFoundError(f"EC-NAIVE legacy: missing block file {block_path}")
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
        remaining_in_pipe = pipeline_total_bytes - src_pos
        take = min(ecnaive_buffer_size, remaining_in_pipe)
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
    return reconstruct_state_dict(decomposed)


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
    return reconstruct_state_dict(decomposed)


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
            f"EC-NAIVE legacy load: rank_in_group 2 recovery done in {time() - start_time:.4f}s"
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
            f"EC-NAIVE legacy load: rank_in_group {rank_in_group} submitted load sends "
            f"in {time() - start_time:.4f}s"
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
        return _reconstruct_full_state_dict_from_main_tensor_buffer(
            main_payload, flat_key_roots=flat_key_roots,
        )
    decomposed = DecomposedStateDict(
        non_tensor_data=main_payload["non_tensor_data"],
        tensor_infos=[],
        tensor_data=[],
        flat_key_roots=flat_key_roots,
    )
    return reconstruct_state_dict(decomposed)


def load_ecnaive_legacy_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    """
    Load EC-NAIVE torch legacy checkpoint: always run 8-port recovery (aligned with torch_dist),
    then reconstruct state_dict from main tensor_buffer when present, else from decoded data0.
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

    _run_ecnaive_full_recovery(
        manager=manager,
        rank=rank,
        world_size=world_size,
        ecnaive_blocks=ecnaive_blocks,
        recv_buffers=recv_buffers,
    )

    # Backward compatibility: checkpoints saved before flat_key_roots existed.
    # See _infer_flat_key_roots for the inference heuristic.
    flat_key_roots = _infer_flat_key_roots(main_payload)

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

    # Clean up EC-NAIVE native module after load to prevent segfaults:
    # C++ worker threads and RDMA connections remain alive after recovery and
    # could access freed memory once local tensors (ecnaive_blocks, recv_buffers)
    # go out of scope.
    logger.info(f"EC-NAIVE legacy load: cleaning up native module (rank {rank})")
    manager.cleanup()

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return state_dict


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
        tensor_bytes_view = _cpu_uint8_view(tensor)
        if tensor_bytes_view.numel() != tensor_bytes:
            raise RuntimeError(
                f"EC-NAIVE legacy save: tensor bytes mismatch for {info.key}, "
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

    full_tensor_buffer = tensor_buffer[:total_tensor_size].detach().clone()

    _save_ecnaive_pt_files(
        checkpoint_name=checkpoint_name,
        rank=rank,
        non_tensor_data=decomposed.non_tensor_data,
        tensor_infos=decomposed.tensor_infos,
        blocks=blocks,
        full_tensor_buffer=full_tensor_buffer,
        flat_key_roots=decomposed.flat_key_roots,
    )

    if world_size > 1:
        torch.distributed.barrier()
