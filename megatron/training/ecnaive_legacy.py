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
    decompose_state_dict_for_save,
    extract_tensors_from_continuous_buffer,
    reconstruct_state_dict,
    unflatten_optimizer_fp32_params,
)

logger = getLogger(__name__)


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
        return {"net_s": 0.0, "encode_s": 0.0}
    try:
        stats = native.get_ft_timing_stats()
        return {
            "net_s": float(stats.get("net_s", 0.0)),
            "encode_s": float(stats.get("encode_s", 0.0)),
        }
    except AttributeError:
        return {"net_s": 0.0, "encode_s": 0.0}


def _timed_barrier() -> float:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0.0
    start = time.time()
    torch.distributed.barrier()
    return time.time() - start



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
    pin: bool = True,
) -> Dict[str, Any]:
    """Allocate n persistent blocks for EC-NAIVE save/load (generalized k+2 scheme).

    Each rank stores n blocks: 1 own data block + (n-1) blocks received from peers.
    Each block size = ceil(max_total_bytes / k / buffer_size) * buffer_size.

    Args:
        pin: Whether to pin CPU memory. True for save (DMA), False for load
             (avoids exhausting CUDA lockable memory on large models).
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

    slices = manager.allocate_preallocated_blocks(n, aligned_block_size, pin=pin)

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


def _iter_ecnaive_load_stripes(total_bytes: int, stripe_bytes: int):
    """Yield (byte_offset, stripe_size) covering [0, total_bytes)."""
    if stripe_bytes <= 0:
        raise ValueError(f"EC-NAIVE load stripes: invalid stripe_bytes={stripe_bytes}")
    pos = 0
    while pos < total_bytes:
        take = min(stripe_bytes, total_bytes - pos)
        yield pos, take
        pos += take


def _submit_ecnaive_transfer_stripes(
    submit_fn,
    channel_idx: int,
    base_addr: int,
    total_bytes: int,
    stripe_bytes: int,
) -> int:
    """Submit byte-striped send/recv tasks on one channel (single buffer, block-major)."""
    n_stripes = 0
    for offset, take in _iter_ecnaive_load_stripes(total_bytes, stripe_bytes):
        submit_fn(channel_idx, base_addr + offset, take)
        n_stripes += 1
    return n_stripes


def _submit_hw_stripes_stripe_major(
    submit_fn,
    channel_idx: int,
    block_bases: List[int],
    total_bytes: int,
    stripe_bytes: int,
) -> int:
    """Submit HW recovery stripes: for each stripe, all blocks (matches save-side ordering).

    Send and recv must use the same stripe-major order:
      stripe0/block0, stripe0/block1, ..., stripe1/block0, ...
    Block-major per-buffer striping desyncs RDMA control-socket ACK handshakes.
    """
    n_tasks = 0
    for offset, take in _iter_ecnaive_load_stripes(total_bytes, stripe_bytes):
        for base in block_bases:
            submit_fn(channel_idx, base + offset, take)
            n_tasks += 1
    return n_tasks


def _submit_ecnaive_load_recovery_stripes(
    native,
    manager: ECNAIVEManager,
    rank_in_group: int,
    aligned_block_size: int,
    ecnaive_blocks: Dict[str, torch.Tensor],
    recv_buffers: Optional[Dict[str, torch.Tensor]] = None,
) -> int:
    """Submit chunked load recv/send tasks (mirrors save-side stripe pipeline).

    Each stripe is an independent recv/xor or send task. C++ recv and XOR workers
    drain their queues concurrently, so stripe N+1 network I/O can overlap XOR on
    stripe N.
    """
    stripe_bytes = manager.ecnaive_buffer_size
    n_stripes = sum(1 for _ in _iter_ecnaive_load_stripes(aligned_block_size, stripe_bytes))

    if rank_in_group == 2:
        if recv_buffers is None:
            raise RuntimeError("EC-NAIVE legacy load: recv_buffers required on rank_in_group 2")
        recv_base = {
            "p20": int(recv_buffers["p20_from_rank0"].data_ptr()),
            "d21": int(recv_buffers["d21_from_rank3"].data_ptr()),
            "d00": int(recv_buffers["d00_from_rank0"].data_ptr()),
            "d01": int(recv_buffers["d01_from_rank1"].data_ptr()),
            "d10": int(recv_buffers["d10_from_rank1"].data_ptr()),
            "p11": int(recv_buffers["p11_from_rank1"].data_ptr()),
            "d30": int(recv_buffers["d30_from_rank3"].data_ptr()),
            "d31": int(recv_buffers["d31_from_rank0"].data_ptr()),
        }
        output_base = {
            "data0": int(ecnaive_blocks["data0"].data_ptr()),
            "recv_parity0": int(ecnaive_blocks["recv_parity0"].data_ptr()),
            "recv_data1": int(ecnaive_blocks["recv_data1"].data_ptr()),
            "recv_parity1": int(ecnaive_blocks["recv_parity1"].data_ptr()),
        }
        for offset, take in _iter_ecnaive_load_stripes(aligned_block_size, stripe_bytes):
            native.submit_ecnaive_load_recovery_full(
                recv_p20_addr=recv_base["p20"] + offset,
                recv_d21_addr=recv_base["d21"] + offset,
                recv_d00_addr=recv_base["d00"] + offset,
                recv_d01_addr=recv_base["d01"] + offset,
                recv_d10_addr=recv_base["d10"] + offset,
                recv_p11_addr=recv_base["p11"] + offset,
                recv_d30_addr=recv_base["d30"] + offset,
                recv_d31_addr=recv_base["d31"] + offset,
                output_data0_addr=output_base["data0"] + offset,
                output_recv_parity0_addr=output_base["recv_parity0"] + offset,
                output_recv_data1_addr=output_base["recv_data1"] + offset,
                output_recv_parity1_addr=output_base["recv_parity1"] + offset,
                size=take,
            )
        native.submit_load_recv_sentinel()
    elif rank_in_group == 0:
        send_base = {
            "p20": int(ecnaive_blocks["recv_parity0"].data_ptr()),
            "d00": int(ecnaive_blocks["data0"].data_ptr()),
            "d31": int(ecnaive_blocks["recv_data1"].data_ptr()),
        }
        for offset, take in _iter_ecnaive_load_stripes(aligned_block_size, stripe_bytes):
            native.submit_load_send_rank0_parity0(
                send_addr=send_base["p20"] + offset, size=take,
            )
            native.submit_load_send_rank0_data0(
                send_addr=send_base["d00"] + offset, size=take,
            )
            native.submit_load_send_rank0_data1(
                send_addr=send_base["d31"] + offset, size=take,
            )
        native.submit_load_send_sentinel()
    elif rank_in_group == 1:
        send_base = {
            "d01": int(ecnaive_blocks["recv_data1"].data_ptr()),
            "d10": int(ecnaive_blocks["data0"].data_ptr()),
            "p11": int(ecnaive_blocks["recv_parity1"].data_ptr()),
        }
        for offset, take in _iter_ecnaive_load_stripes(aligned_block_size, stripe_bytes):
            native.submit_load_send_rank1_data1(
                send_addr=send_base["d01"] + offset, size=take,
            )
            native.submit_load_send_rank1_data0(
                send_addr=send_base["d10"] + offset, size=take,
            )
            native.submit_load_send_rank1_parity1(
                send_addr=send_base["p11"] + offset, size=take,
            )
        native.submit_load_send_sentinel()
    elif rank_in_group == 3:
        send_base = {
            "d21": int(ecnaive_blocks["recv_data1"].data_ptr()),
            "d30": int(ecnaive_blocks["data0"].data_ptr()),
        }
        for offset, take in _iter_ecnaive_load_stripes(aligned_block_size, stripe_bytes):
            native.submit_load_send_rank3_data1(
                send_addr=send_base["d21"] + offset, size=take,
            )
            native.submit_load_send_rank3_data0(
                send_addr=send_base["d30"] + offset, size=take,
            )
        native.submit_load_send_sentinel()
    else:
        raise RuntimeError(
            f"EC-NAIVE legacy load: unexpected rank_in_group={rank_in_group}"
        )
    return n_stripes


def _submit_hw_failed_recv_stripe(
    native,
    manager: ECNAIVEManager,
    rank: int,
    world_size: int,
    source_ranks: List[int],
    ecnaive_n: int,
    recv_pool: List[torch.Tensor],
    byte_offset: int,
    stripe_size: int,
) -> None:
    """Submit one recv stripe from all source ranks (failed rank HW recovery)."""
    for si, src_rank in enumerate(source_ranks):
        recv_ch = manager.get_recv_channel_from_source(rank, src_rank, world_size)
        base = si * ecnaive_n
        for bi in range(ecnaive_n):
            native.submit_recv_task(
                recv_ch,
                int(recv_pool[base + bi].data_ptr()) + byte_offset,
                stripe_size,
            )


def _build_hw_owner_codeword_plans(
    owner_rigs: List[int],
    my_rig: int,
    ecnaive_k: int,
    ecnaive_n: int,
    rig_to_si: Dict[int, int],
    recv_pool: List[torch.Tensor],
    block_data_size: int,
    store_bufs: Dict[str, torch.Tensor],
    recovered_slot_pool: List[torch.Tensor],
    parity_pool_0: List[torch.Tensor],
    parity_pool_1: List[torch.Tensor],
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Precompute per-codeword decode/encode metadata for striped HW recovery."""

    def _find_block_in_pool(owner_rig: int, role: str) -> Optional[torch.Tensor]:
        if role == "data0":
            si = rig_to_si.get(owner_rig)
            return recv_pool[si * ecnaive_n] if si is not None else None
        if role == "data1":
            si = rig_to_si.get((owner_rig + 1) % ecnaive_n)
            return recv_pool[si * ecnaive_n + 1] if si is not None else None
        if role == "parity0":
            si = rig_to_si.get((owner_rig + 2) % ecnaive_n)
            return recv_pool[si * ecnaive_n + 2] if si is not None else None
        if role == "parity1":
            si = rig_to_si.get((owner_rig + 3) % ecnaive_n)
            return recv_pool[si * ecnaive_n + 3] if si is not None else None
        return None

    owner_plans: List[Dict[str, Any]] = []
    recovered_refs: Dict[str, torch.Tensor] = {}
    owner_idx = 0

    for owner_rig in owner_rigs:
        raw_surviving: Dict[str, torch.Tensor] = {}
        lost: List[int] = []
        for role, label in [
            ("data0", "data_0"),
            ("data1", "data_1"),
            ("parity0", "parity0"),
            ("parity1", "parity1"),
        ]:
            block = _find_block_in_pool(owner_rig, role)
            if block is not None:
                raw_surviving[label] = block
        for pos, label in enumerate(["data_0", "data_1"]):
            if label not in raw_surviving:
                lost.append(pos)
        m_owner = len(lost)

        surviving_continuous = {
            label: tensor[:block_data_size] for label, tensor in raw_surviving.items()
        }

        surviving_bases: List[int] = []
        recovered_bases: List[int] = []
        encode_input_bases: List[int] = []

        if m_owner == 0:
            encode_input_bases = [
                int(surviving_continuous[f"data_{j}"].data_ptr())
                for j in range(ecnaive_k)
            ]
            recovered_data = [
                surviving_continuous[f"data_{j}"] for j in range(ecnaive_k)
            ]
        else:
            data_labels = sorted(
                [label for label in surviving_continuous if label.startswith("data_")],
                key=lambda x: int(x.split("_")[1]),
            )
            parity_labels = sorted(
                label for label in surviving_continuous if label.startswith("parity")
            )
            surviving_addrs_ordered = data_labels + parity_labels
            surviving_bases = [
                int(surviving_continuous[label].data_ptr())
                for label in surviving_addrs_ordered
            ]
            if owner_rig == my_rig:
                recovered_blocks = [
                    store_bufs["own_data0"],
                    store_bufs["my_data1"],
                ][:m_owner]
            else:
                base = owner_idx * ecnaive_k
                recovered_blocks = recovered_slot_pool[base : base + m_owner]
            recovered_bases = [int(block.data_ptr()) for block in recovered_blocks]

            recovered_data: List[Optional[torch.Tensor]] = [None] * ecnaive_k
            ri = 0
            for pos in range(ecnaive_k):
                label = f"data_{pos}"
                if label in surviving_continuous:
                    recovered_data[pos] = surviving_continuous[label]
                    encode_input_bases.append(int(surviving_continuous[label].data_ptr()))
                else:
                    recovered_data[pos] = recovered_blocks[ri]
                    encode_input_bases.append(recovered_bases[ri])
                    ri += 1

        need_parity0 = (owner_rig + 2) % ecnaive_n == my_rig
        need_parity1 = (owner_rig + 3) % ecnaive_n == my_rig
        parity0 = parity_pool_0[owner_idx]
        parity1 = parity_pool_1[owner_idx]

        if owner_rig == my_rig:
            recovered_refs["own_data0_ref"] = recovered_data[0]
            if ecnaive_k > 1:
                recovered_refs["my_data1_ref"] = recovered_data[1]
        if (owner_rig + 1) % ecnaive_n == my_rig:
            recovered_refs["recv_0_ref"] = recovered_data[1]
        if need_parity0:
            recovered_refs["recv_1_ref"] = parity0
        if need_parity1:
            recovered_refs["recv_2_ref"] = parity1

        owner_plans.append(
            {
                "owner_rig": owner_rig,
                "m_owner": m_owner,
                "lost": lost,
                "surviving_bases": surviving_bases,
                "recovered_bases": recovered_bases,
                "encode_input_bases": encode_input_bases,
                "need_parity0": need_parity0,
                "need_parity1": need_parity1,
                "parity0_base": int(parity0.data_ptr()),
                "parity1_base": int(parity1.data_ptr()),
            }
        )
        owner_idx += 1

    return owner_plans, recovered_refs


def _hw_decode_encode_all_owners_stripe(
    native,
    owner_plans: List[Dict[str, Any]],
    ecnaive_k: int,
    byte_offset: int,
    stripe_size: int,
) -> None:
    """RS-decode and parity-encode one byte stripe for every owner codeword."""
    for plan in owner_plans:
        m_owner = plan["m_owner"]
        if m_owner > 0:
            native.submit_ecnaive_decode_recovery(
                ecnaive_k,
                m_owner,
                plan["lost"],
                [addr + byte_offset for addr in plan["surviving_bases"]],
                [addr + byte_offset for addr in plan["recovered_bases"]],
                stripe_size,
            )
        if plan["need_parity0"] or plan["need_parity1"]:
            native.encode_ec_blocks(
                [addr + byte_offset for addr in plan["encode_input_bases"]],
                plan["parity0_base"] + byte_offset,
                plan["parity1_base"] + byte_offset,
                stripe_size,
            )


def _run_hw_failed_recv_decode_pipeline(
    native,
    manager: ECNAIVEManager,
    rank: int,
    world_size: int,
    source_ranks: List[int],
    ecnaive_k: int,
    ecnaive_n: int,
    recv_pool: List[torch.Tensor],
    block_data_size: int,
    owner_plans: List[Dict[str, Any]],
    stripe_bytes: int,
) -> int:
    """Recv/decode pipeline: recv stripe N+1 overlaps decode stripe N (mirrors save)."""
    stripes = list(_iter_ecnaive_load_stripes(block_data_size, stripe_bytes))
    if not stripes:
        return 0

    off0, take0 = stripes[0]
    _submit_hw_failed_recv_stripe(
        native, manager, rank, world_size, source_ranks, ecnaive_n,
        recv_pool, off0, take0,
    )
    native.wait_for_pending_network_tasks()

    for i in range(1, len(stripes)):
        off, take = stripes[i]
        prev_off, prev_take = stripes[i - 1]
        _submit_hw_failed_recv_stripe(
            native, manager, rank, world_size, source_ranks, ecnaive_n,
            recv_pool, off, take,
        )
        _hw_decode_encode_all_owners_stripe(
            native, owner_plans, ecnaive_k, prev_off, prev_take,
        )
        native.wait_for_pending_network_tasks()

    last_off, last_take = stripes[-1]
    _hw_decode_encode_all_owners_stripe(
        native, owner_plans, ecnaive_k, last_off, last_take,
    )
    return len(stripes)


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

            # Continuous recv block write addresses (no padding, aligned with ECLATIN)
            recv_write_addrs = []
            for i in range(num_recv):
                if recv_offsets[i] + take > aligned_block_size:
                    logger.warning("EC-NAIVE legacy: recv block %d exhausted", i)
                    recv_offsets[i] = 0
                recv_write_addrs.append(recv_bases[i] + recv_offsets[i])
                recv_offsets[i] += take

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
    all_tensor_infos: Optional[Dict[int, List[Any]]] = None,
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
        "version": 3, "format": "ecnaive_torch_legacy", "rank": rank,
        "ecnaive_k": k, "ecnaive_n": n,
        "actual_tensor_size": blocks["actual_size"],
        "pipeline_total_bytes": blocks["pipeline_size"],
        "aligned_block_size": blocks["aligned_size"],
        "block_data_size": blocks.get("block_data_size", blocks["pipeline_size"] // k),
        "flat_key_roots": list(flat_key_roots) if flat_key_roots else [],
        "block_files": block_files,
        "_block_files_legacy": block_files_legacy,
        "all_tensor_infos": all_tensor_infos if all_tensor_infos is not None else {},
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
        from megatron.training.legacy_io_utils import (
            is_raw_format, read_raw_checkpoint, MAGIC_ECNAIVE,
            pin_payload_tensor_buffer_if_available,
        )
        if is_raw_format(str(main_path), MAGIC_ECNAIVE):
            local_payload = read_raw_checkpoint(
                str(main_path), MAGIC_ECNAIVE, pin_tensor_buffer=True,
            )
        else:
            local_payload = torch.load(main_path, map_location="cpu", weights_only=False)
            pin_payload_tensor_buffer_if_available(local_payload)

    if world_size <= 1 or not torch.distributed.is_initialized():
        if local_payload is None:
            raise FileNotFoundError(f"EC-NAIVE legacy: missing main file {main_path}")
        return local_payload

    # all_gather_object pickles the entire payload including tensor_buffer
    # (multiple GB for large models).  NCCL all_gather creates GPU staging
    # buffers proportional to  world_size × serialized_size  → OOM on 7B+.
    # Strip tensor_buffer before the collective; only exchange metadata.
    if local_payload is not None:
        stripped: Dict[str, Any] = {}
        for k, v in local_payload.items():
            if k == "tensor_buffer":
                continue
            stripped[k] = v
    else:
        stripped = None

    gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, stripped)

    if local_payload is not None:
        return local_payload

    # HW failure: local disk lost — recover metadata from another rank.
    # main.pt stores all_tensor_infos (all ranks' metadata, like Gemini).
    # Extract the correct rank's tensor_infos from any healthy source.
    for r in range(world_size):
        if gathered[r] is not None:
            payload = dict(gathered[r])
            all_ti = payload.get("all_tensor_infos")
            if all_ti and rank in all_ti:
                logger.debug(
                    f"EC-NAIVE legacy: ecnaive_main_rank{rank}.pt missing locally; "
                    f"recovered tensor_infos for rank {rank} from rank {r}"
                )
                payload["tensor_infos"] = all_ti[rank]
                payload["tensor_buffer"] = None  # to be recovered via RS decode
                return payload
            # Backward compat: old checkpoints without all_tensor_infos
            if "tensor_infos" in payload:
                logger.warning(
                    f"EC-NAIVE legacy: using rank {r}'s tensor_infos as fallback "
                    f"for rank {rank} — may be incorrect (old checkpoint format)"
                )
                payload["tensor_buffer"] = None
                return payload

    raise FileNotFoundError(
        f"EC-NAIVE legacy: ecnaive_main_rank{rank}.pt missing on all ranks "
        f"under {checkpoint_dir}"
    )


def _load_blocks_from_disk(checkpoint_dir: Path, rank: int) -> Dict[str, torch.Tensor]:
    from megatron.training.legacy_io_utils import (
        is_raw_format, read_raw_block, MAGIC_BLOCK, pin_uint8_tensor_if_available,
    )
    blocks: Dict[str, torch.Tensor] = {}
    for block_name in ("data0", "recv_parity1", "recv_parity0", "recv_data1"):
        block_path = checkpoint_dir / f"ecnaive_block_rank{rank}_{block_name}.pt"
        if not block_path.is_file():
            raise FileNotFoundError(f"EC-NAIVE legacy: missing block file {block_path}")
        if is_raw_format(str(block_path), MAGIC_BLOCK):
            blocks[block_name] = read_raw_block(
                str(block_path), MAGIC_BLOCK, pin_tensor=True,
            )
        else:
            payload = torch.load(block_path, map_location="cpu", weights_only=False)
            blocks[block_name] = pin_uint8_tensor_if_available(
                payload["tensor"].contiguous().view(torch.uint8)
            )
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
    from megatron.training.legacy_io_utils import (
        is_raw_format, read_raw_block, MAGIC_BLOCK, pin_uint8_tensor_if_available,
    )

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
        return read_raw_block(str(block_path), MAGIC_BLOCK, pin_tensor=True)
    payload = torch.load(str(block_path), map_location="cpu", weights_only=False)
    return pin_uint8_tensor_if_available(
        payload["tensor"].contiguous().view(torch.uint8).reshape(-1)
    )


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


def _decode_block_direct(
    padded_block: torch.Tensor,
    output: torch.Tensor,
    dst_offset: int,
    block_data_size: int,
    pipeline_total_bytes: int,
    block_idx: int,
    ecnaive_buffer_size: int,
) -> None:
    """Decode a padded block directly into output buffer at dst_offset.

    Avoids the intermediate tensor allocation + torch.cat that
    _decode_data_block + concatenation would incur.
    """
    start_byte = block_idx * block_data_size
    end_byte = min(start_byte + block_data_size, pipeline_total_bytes)
    actual = max(0, end_byte - start_byte)
    src_pos = 0
    pe_offset = 0  # position in padded block
    while src_pos < actual:
        take = min(ecnaive_buffer_size, actual - src_pos)
        aligned = ((pe_offset + 63) // 64) * 64
        if aligned + take > padded_block.numel():
            logger.warning("EC-NAIVE: data block %d exhausted during decode", block_idx)
            break
        output[dst_offset + src_pos : dst_offset + src_pos + take].copy_(
            padded_block[aligned : aligned + take]
        )
        pe_offset = aligned + take
        src_pos += take


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

    # ---- prepare buffers / disk reads (not timed) ----
    tensor_buffer: Optional[torch.Tensor] = None
    recv_blocks: List[torch.Tensor] = []
    send_block: Optional[torch.Tensor] = None
    send_block_idx: int = -1

    ckpt_ver = int(main_payload.get("version", 2))
    has_padded_sw = ckpt_ver < 3

    if rank_in_group == failed_rig:
        # Pre-allocate final tensor_buffer once (pinned for fast CPU→GPU copy)
        buf_len = max(actual_tensor_size, pipeline_total_bytes)
        tensor_buffer = allocate_hugepage_tensor(buf_len, fallback_pin_memory=True)
        # Load local d_{f,0} into tensor_buffer
        own_data0 = _load_ecnaive_block_file(
            checkpoint_dir, rank,
            canonical_name="own_data0",
            legacy_name="data0",
            block_files_legacy=block_files_legacy,
        )
        if has_padded_sw:
            _decode_block_direct(
                own_data0, tensor_buffer, 0,
                block_data_size, pipeline_total_bytes, 0,
                manager.ecnaive_buffer_size,
            )
        else:
            n_copy = min(own_data0.numel(), block_data_size)
            tensor_buffer[:n_copy].copy_(own_data0[:n_copy])
        own_data0 = None  # free ref
        for _j in range(1, k):
            buf = torch.zeros(aligned_block_size, dtype=torch.uint8)
            if manager.use_rdma:
                manager.register_buffer(buf)
            recv_blocks.append(buf)
    else:
        sender_rig = rank_in_group
        j = (sender_rig - failed_rig + n) % n
        if 1 <= j < k:
            block_idx = j - 1
            recv_idx = (sender_rig - failed_rig - 1 + n) % n
            canonical_name = f"recv_{recv_idx}"
            if k == 2:
                legacy_map_2 = {0: "recv_parity1", 1: "recv_parity0", 2: "recv_data1"}
                legacy_name = legacy_map_2.get(recv_idx, canonical_name)
            else:
                legacy_name = canonical_name
            send_block = _load_ecnaive_block_file(
                checkpoint_dir, rank,
                canonical_name=canonical_name,
                legacy_name=legacy_name,
                block_files_legacy=block_files_legacy,
            )
            if manager.use_rdma:
                manager.register_buffer(send_block)
            send_block_idx = block_idx

    # Sync all ranks after setup so network timing excludes setup skew.
    _t['barrier'] = _timed_barrier() if world_size > 1 else 0.0

    # === timing: network/encode (ASIO send/recv only) ===
    _t0_net = _time()
    if rank_in_group == failed_rig:
        for idx, buf in enumerate(recv_blocks):
            native.sw_recv_data(idx, int(buf.data_ptr()), buf.numel())
    elif send_block is not None:
        native.sw_send_data(send_block_idx, int(send_block.data_ptr()), send_block.numel())
    _t['network_encode'] = _time() - _t0_net
    native_timing = _native_ft_timing(native)
    _t['net_s'] = native_timing['net_s']
    _t['encode_s'] = native_timing['encode_s']

    # Decode recv blocks → tensor_buffer (not timed)
    if rank_in_group == failed_rig:
        for idx, buf in enumerate(recv_blocks):
            dst_off = (idx + 1) * block_data_size
            if has_padded_sw:
                _decode_block_direct(
                    buf, tensor_buffer, dst_off,
                    block_data_size, pipeline_total_bytes, idx + 1,
                    manager.ecnaive_buffer_size,
                )
            else:
                n_copy = min(buf.numel(), block_data_size)
                tensor_buffer[dst_off:dst_off + n_copy].copy_(buf[:n_copy])
            logger.debug(
                "EC-NAIVE legacy sw: failed_rig=%d received data_block_idx=%d bytes=%d",
                failed_rig, idx + 1, buf.numel(),
            )
    if send_block is not None:
        logger.debug(
            "EC-NAIVE legacy sw: rig=%d sent (block_idx=%d, %d bytes)",
            rank_in_group, send_block_idx, send_block.numel(),
        )

    t_rebuild = _time()
    if rank_in_group == failed_rig:
        if actual_tensor_size > 0:
            tensor_buffer = tensor_buffer[:actual_tensor_size]
        tensor_data = extract_tensors_from_continuous_buffer(tensor_buffer, tensor_infos)
        decomposed = DecomposedStateDict(
            non_tensor_data=non_tensor_data,
            tensor_infos=tensor_infos,
            tensor_data=tensor_data,
            flat_key_roots=flat_key_roots,
        )
        state_dict = reconstruct_state_dict(decomposed)
        unflatten_optimizer_fp32_params(state_dict)
    else:
        state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
            main_payload, flat_key_roots=flat_key_roots,
        )
    _t['rebuild_sd'] = _time() - t_rebuild

    # NOTE: the old standalone else clause for ranks 0,1 (k=2) is absorbed into
    # the generalized else branch above.
    # barrier at end of rebuild_sd is handled by the caller (load_ecnaive_legacy_checkpoint).

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

        aligned_block_size = ecnaive_blocks["data0"].numel()
        n_stripes = _submit_ecnaive_load_recovery_stripes(
            native=native,
            manager=manager,
            rank_in_group=rank_in_group,
            aligned_block_size=aligned_block_size,
            ecnaive_blocks=ecnaive_blocks,
            recv_buffers=recv_buffers,
        )
        logger.debug(
            "EC-NAIVE legacy load: rank_in_group 2 submitted %d stripes "
            "(stripe_bytes=%d, block_bytes=%d), waiting for completion",
            n_stripes, manager.ecnaive_buffer_size, aligned_block_size,
        )
        native.wait_for_load_completion()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.debug(
            f"EC-NAIVE legacy load: hw recovery done in {time() - start_time:.2f}s (rank_in_group=2)"
        )
    else:
        aligned_block_size = ecnaive_blocks["data0"].numel()
        n_stripes = _submit_ecnaive_load_recovery_stripes(
            native=native,
            manager=manager,
            rank_in_group=rank_in_group,
            aligned_block_size=aligned_block_size,
            ecnaive_blocks=ecnaive_blocks,
        )
        logger.debug(
            f"EC-NAIVE legacy load: submitted {n_stripes} load send stripes in "
            f"{time() - start_time:.2f}s (rank_in_group={rank_in_group})"
        )

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
        logger.debug("EC-NAIVE legacy: software failure recovery path")
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
        _t['total'] = (
            _t.get('network_encode', 0.0)
            + _t.get('rebuild_sd', 0.0)
        )
        from megatron.training.global_vars import set_ft_load_timing_context
        set_ft_load_timing_context("EC-NAIVE", "SW", _t)
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

    # Sync all ranks after setup so network timing excludes setup skew.
    barrier_s = _timed_barrier() if world_size > 1 and torch.distributed.is_initialized() else 0.0

    # === timing: network/encode (C++ pipeline) ===
    _t0 = time.time()
    _run_ecnaive_full_recovery(
        manager=manager,
        rank=rank,
        world_size=world_size,
        ecnaive_blocks=ecnaive_blocks,
        recv_buffers=recv_buffers,
    )
    network_encode = time.time() - _t0

    # Backward compatibility: checkpoints saved before flat_key_roots existed.
    flat_key_roots = _infer_flat_key_roots(main_payload)

    t_rebuild = time.time()
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
    rebuild_sd = time.time() - t_rebuild
    native_timing = _native_ft_timing(manager._ecnaive_native)

    timings = {
        "total": network_encode + rebuild_sd,
        "network_encode": network_encode,
        "net_s": native_timing["net_s"],
        "encode_s": native_timing["encode_s"],
        "rebuild_sd": rebuild_sd,
        "barrier": barrier_s,
    }
    from megatron.training.global_vars import set_ft_load_timing_context
    set_ft_load_timing_context("EC-NAIVE", "HW", timings)

    return state_dict


def _load_all_blocks_from_disk(
    checkpoint_dir: Path,
    rank: int,
    ecnaive_k: int,
    ecnaive_n: int,
    block_files: Optional[Dict[str, str]] = None,
) -> List[torch.Tensor]:
    """Load all n checkpoint blocks from disk for a surviving rank.

    Block order: [own_data0, recv_0, recv_1, ..., recv_{n-2}]

    Tries canonical filenames first, falls back to legacy names for k=2.
    """
    from megatron.training.legacy_io_utils import (
        is_raw_format, read_raw_block, MAGIC_BLOCK, pin_uint8_tensor_if_available,
    )

    # Legacy name map for k=2 backward compatibility
    legacy_map: Dict[int, str] = {}
    if ecnaive_k == 2:
        legacy_map = {
            0: "data0",
            1: "recv_parity1",
            2: "recv_parity0",
            3: "recv_data1",
        }

    blocks: List[torch.Tensor] = []
    for i in range(ecnaive_n):
        canon_name = "own_data0" if i == 0 else f"recv_{i - 1}"
        candidates: List[Path] = []

        # 1. From block_files metadata (main_payload)
        if block_files and canon_name in block_files:
            candidates.append(checkpoint_dir / block_files[canon_name])
        # 2. Canonical filename
        candidates.append(
            checkpoint_dir / f"ecnaive_block_rank{rank}_{canon_name}.pt"
        )
        # 3. Legacy filename (k=2 only)
        if i in legacy_map:
            candidates.append(
                checkpoint_dir / f"ecnaive_block_rank{rank}_{legacy_map[i]}.pt"
            )

        tensor = None
        for path in candidates:
            if path.is_file():
                if is_raw_format(str(path), MAGIC_BLOCK):
                    tensor = read_raw_block(str(path), MAGIC_BLOCK, pin_tensor=True)
                else:
                    payload = torch.load(
                        str(path), map_location="cpu", weights_only=False
                    )
                    tensor = pin_uint8_tensor_if_available(
                        payload["tensor"].contiguous().view(torch.uint8)
                    )
                break

        if tensor is None:
            raise FileNotFoundError(
                f"EC-NAIVE hw recovery: survivor rank {rank} missing block "
                f"{i} ({canon_name}) — tried: {[str(p) for p in candidates]}"
            )
        blocks.append(tensor)

    return blocks
def load_ecnaive_legacy_checkpoint_hardware_recovery(
    checkpoint_name: str, failed_global_ranks: List[int]
) -> Dict[str, Any]:
    """Hardware recovery for 1-2 failed ranks using C++ ASIO send/recv + RS decode.

    Each surviving rank sends ALL its n checkpoint blocks to each failed rank
    in the same group. The failed rank receives from all survivors and recovers:
      1. Its own k data blocks → tensor_buffer → state_dict (RS decode)
      2. Its (n-1) recv blocks (for cascading failure tolerance)
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
    # version < 3: old checkpoints have 64B alignment gaps in recv blocks
    # version >= 3: continuous (no padding, aligned with ECLATIN)
    ckpt_version = int(main_payload.get("version", 3))
    has_padded_recv = ckpt_version < 3

    # old checkpoints: blocks have 64B gaps, use aligned_block_size for send/recv
    # new checkpoints: continuous, use block_data_size
    aligned_block_size = int(main_payload.get("aligned_block_size",
                             ((block_data_size + manager.ecnaive_buffer_size - 1)
                              // manager.ecnaive_buffer_size) * manager.ecnaive_buffer_size))
    recv_block_size = aligned_block_size if has_padded_recv else block_data_size

    flat_key_roots = _infer_flat_key_roots(main_payload)
    actual_tensor_size = int(main_payload.get("actual_tensor_size", 0))
    num_channels = ecnaive_n - 1  # = k + 1

    # Step 3: Determine group membership
    is_failed = (rank in failed_set)
    my_gid = manager._get_group_id(rank, world_size)
    my_rig = manager._get_rank_in_group(rank, world_size)

    failed_in_group: List[int] = [
        fr for fr in failed_global_ranks
        if manager._get_group_id(fr, world_size) == my_gid
    ]
    survivors_in_group: List[int] = [
        r for r in range(world_size)
        if manager._get_group_id(r, world_size) == my_gid and r not in failed_set
    ]

    # affected_group: my group has at least one failed rank
    affected_group = len(failed_in_group) > 0

    # Source ranks: first k survivors that send ALL blocks for recovery
    source_ranks = survivors_in_group[:ecnaive_k] if len(survivors_in_group) >= ecnaive_k else survivors_in_group
    is_source = rank in source_ranks
    source_set = set(source_ranks)

    barrier_s = 0.0

    # Pre-allocate recv pool for failed ranks, load blocks for source ranks (not timed)
    recv_pool_prealloc: List[torch.Tensor] = []
    source_blocks_prealloc: List[torch.Tensor] = []
    # Pre-computed decode layout and buffers for failed ranks (moved here to keep outside timing)
    _rig_to_si: Dict[int, int] = {}
    _owner_rigs: List[int] = []
    _num_owners: int = 0
    recovered_slot_pool_pre: List[torch.Tensor] = []
    parity_pool_0_pre: List[torch.Tensor] = []
    parity_pool_1_pre: List[torch.Tensor] = []
    store_bufs_pre: Dict[str, torch.Tensor] = {}
    tensor_buffer_pre: Optional[torch.Tensor] = None
    if affected_group and is_failed:
        total_pool_blocks = len(source_ranks) * ecnaive_n
        recv_pool_prealloc = list(allocate_hugepage_slices(
            recv_block_size, total_pool_blocks,
            fallback_pin_memory=True, touch_pages=True,
        ))
        if manager.use_rdma:
            for buf in recv_pool_prealloc:
                manager.register_buffer(buf)
        # Pre-compute block locator metadata
        _source_rig_map = {r: manager._get_rank_in_group(r, world_size) for r in source_ranks}
        _rig_to_si = {rig: si for si, (_, rig) in enumerate(
            sorted(((r, _source_rig_map[r]) for r in source_ranks), key=lambda x: x[1])
        )}
        _owner_rigs = [my_rig]
        for _recv_idx in range(ecnaive_n - 1):
            _ow = (my_rig - _recv_idx - 1) % ecnaive_n
            if _ow not in _owner_rigs:
                _owner_rigs.append(_ow)
        _num_owners = len(_owner_rigs)
        # Pre-allocate decode/encode/store/tensor buffers
        recovered_slot_pool_pre = [
            torch.empty(block_data_size, dtype=torch.uint8) for _ in range(_num_owners * ecnaive_k)
        ]
        parity_pool_0_pre = [
            torch.empty(block_data_size, dtype=torch.uint8) for _ in range(_num_owners)
        ]
        parity_pool_1_pre = [
            torch.empty(block_data_size, dtype=torch.uint8) for _ in range(_num_owners)
        ]
        store_bufs_pre = {
            _name: torch.empty(block_data_size, dtype=torch.uint8)
            for _name in ('own_data0', 'my_data1', 'recv_0', 'recv_1', 'recv_2')
        }
        tensor_buffer_pre = allocate_hugepage_tensor(
            max(block_data_size * ecnaive_k, actual_tensor_size), fallback_pin_memory=True,
        )
    elif affected_group and not is_failed and is_source:
        source_blocks_prealloc = _load_all_blocks_from_disk(
            checkpoint_dir, rank, ecnaive_k, ecnaive_n, block_files,
        )
        if manager.use_rdma:
            for b in source_blocks_prealloc:
                manager.register_buffer(b)

    # Sync after pre-alloc/load so network timing excludes setup skew.
    barrier_s += _timed_barrier()

    # === timing: network/encode (C++ send/recv + RS decode only) ===
    _t: Dict[str, float] = {}
    _t0_net = time.time()

    # ── Pipeline: SOURCE ranks send all n blocks to each failed rank ──
    if affected_group and not is_failed and is_source:
        native.reset_encoding_completion_flags()
        stripe_bytes = manager.ecnaive_buffer_size

        for dest_fr in failed_in_group:
            send_ch = manager.get_send_channel_for_target(rank, dest_fr, world_size)
            send_bases = [
                int(block_tensor.data_ptr()) for block_tensor in source_blocks_prealloc
            ]
            send_size = min(
                min(block_tensor.numel() for block_tensor in source_blocks_prealloc),
                recv_block_size,
            )
            _submit_hw_stripes_stripe_major(
                native.submit_send_task,
                send_ch,
                send_bases,
                send_size,
                stripe_bytes,
            )
            logger.debug(
                f"EC-NAIVE hw recovery: source rank {rank} sending all "
                f"{len(source_blocks_prealloc)} blocks (stripe-major, {stripe_bytes} B) "
                f"to failed rank {dest_fr}"
            )

        native.submit_send_sentinels(num_channels)
        native.submit_recv_sentinels(num_channels)
        native.wait_for_encoding_completion()
        _t['network_encode'] = time.time() - _t0_net

        t_rebuild = time.time()
        state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
            main_payload, flat_key_roots=flat_key_roots,
        )
        _t['rebuild_sd'] = time.time() - t_rebuild

    elif affected_group and not is_failed and not is_source:
        # Survivor but not selected as source: no-op on channels
        native.reset_encoding_completion_flags()
        native.submit_send_sentinels(num_channels)
        native.submit_recv_sentinels(num_channels)
        native.wait_for_encoding_completion()
        _t['network_encode'] = time.time() - _t0_net

        t_rebuild = time.time()
        state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
            main_payload, flat_key_roots=flat_key_roots,
        )
        _t['rebuild_sd'] = time.time() - t_rebuild

    elif affected_group and is_failed:
        # ═══════════════════════════════════════════════════════════════
        # FAILED RANK: receive k×n blocks from k source ranks, encode
        # ==========
        #   Unified decode: each owner codeword's k data blocks are
        #   recovered from k surviving blocks (data or parity) in the
        #   recv_pool using submit_ecnaive_decode_recovery.  No special
        #   casing — direct get / XOR / RS are all just the same decode.
        # ==========
        plan = recovery_plan[rank]
        my_rig = plan['rank_in_group']

        recv_pool = recv_pool_prealloc
        rig_to_si = _rig_to_si
        owner_rigs = _owner_rigs
        recovered_slot_pool = recovered_slot_pool_pre
        parity_pool_0 = parity_pool_0_pre
        parity_pool_1 = parity_pool_1_pre
        store_bufs = store_bufs_pre

        native.reset_encoding_completion_flags()
        stripe_bytes = manager.ecnaive_buffer_size
        _t0_decode = time.time()

        if has_padded_recv:
            # Old checkpoints: padded recv layout — batch recv then full-block decode.
            for si, src_rank in enumerate(source_ranks):
                recv_ch = manager.get_recv_channel_from_source(rank, src_rank, world_size)
                base = si * ecnaive_n
                recv_bases = [
                    int(recv_pool[base + bi].data_ptr()) for bi in range(ecnaive_n)
                ]
                _submit_hw_stripes_stripe_major(
                    native.submit_recv_task,
                    recv_ch,
                    recv_bases,
                    recv_block_size,
                    stripe_bytes,
                )
            native.submit_send_sentinels(num_channels)
            native.submit_recv_sentinels(num_channels)
            native.wait_for_encoding_completion()
            _t['network_recv'] = time.time() - _t0_net

            def _find_block_in_pool(owner_rig: int, role: str) -> Optional[torch.Tensor]:
                if role == 'data0':
                    si = rig_to_si.get(owner_rig)
                    return recv_pool[si * ecnaive_n] if si is not None else None
                if role == 'data1':
                    si = rig_to_si.get((owner_rig + 1) % ecnaive_n)
                    return recv_pool[si * ecnaive_n + 1] if si is not None else None
                if role == 'parity0':
                    si = rig_to_si.get((owner_rig + 2) % ecnaive_n)
                    return recv_pool[si * ecnaive_n + 2] if si is not None else None
                if role == 'parity1':
                    si = rig_to_si.get((owner_rig + 3) % ecnaive_n)
                    return recv_pool[si * ecnaive_n + 3] if si is not None else None
                return None

            recovered: Dict[str, torch.Tensor] = {}
            owner_idx = 0
            for owner_rig in owner_rigs:
                raw_surviving = {}
                lost = []
                for role, label in [
                    ('data0', 'data_0'), ('data1', 'data_1'),
                    ('parity0', 'parity0'), ('parity1', 'parity1'),
                ]:
                    block = _find_block_in_pool(owner_rig, role)
                    if block is not None:
                        raw_surviving[label] = block
                for pos, label in enumerate(['data_0', 'data_1']):
                    if label not in raw_surviving:
                        lost.append(pos)
                m_owner = len(lost)
                surviving: Dict[str, torch.Tensor] = {}
                is_padded: Dict[str, bool] = {}
                for label, padded in raw_surviving.items():
                    if label != 'data_0':
                        surviving[label] = padded
                        is_padded[label] = True
                    else:
                        surviving[label] = padded[:block_data_size]
                        is_padded[label] = False
                if m_owner == 0:
                    recovered_data = [surviving['data_0'], surviving['data_1']]
                else:
                    data_labels = sorted(
                        [l for l in surviving if l.startswith('data_')],
                        key=lambda x: int(x.split('_')[1]),
                    )
                    parity_labels = sorted([l for l in surviving if l.startswith('parity')])
                    surviving_addrs_ordered = data_labels + parity_labels
                    continuous_surviving = [
                        surviving[l][:block_data_size] for l in surviving_addrs_ordered
                    ]
                    if owner_rig == my_rig:
                        recovered_blocks = [
                            store_bufs['own_data0'], store_bufs['my_data1'],
                        ][:m_owner]
                    else:
                        base = owner_idx * ecnaive_k
                        recovered_blocks = recovered_slot_pool[base : base + m_owner]
                    native.submit_ecnaive_decode_recovery(
                        ecnaive_k, m_owner, lost,
                        [int(b.data_ptr()) for b in continuous_surviving],
                        [int(b.data_ptr()) for b in recovered_blocks],
                        block_data_size,
                    )
                    recovered_data = [None, None]
                    ri = 0
                    for pos in range(ecnaive_k):
                        label = f'data_{pos}'
                        if label in surviving:
                            recovered_data[pos] = surviving[label]
                        else:
                            recovered_data[pos] = recovered_blocks[ri]
                            ri += 1
                if owner_rig == my_rig:
                    recovered['own_data0_ref'] = recovered_data[0]
                    recovered['my_data1_ref'] = recovered_data[1]
                if (owner_rig + 1) % ecnaive_n == my_rig:
                    recovered['recv_0_ref'] = recovered_data[1]
                need_parity0 = (owner_rig + 2) % ecnaive_n == my_rig
                need_parity1 = (owner_rig + 3) % ecnaive_n == my_rig
                if need_parity0 or need_parity1:
                    encode_inputs = [
                        recovered_data[j][:block_data_size] for j in range(ecnaive_k)
                    ]
                    parity0 = parity_pool_0[owner_idx]
                    parity1 = parity_pool_1[owner_idx]
                    native.encode_ec_blocks(
                        [int(b.data_ptr()) for b in encode_inputs],
                        int(parity0.data_ptr()),
                        int(parity1.data_ptr()),
                        block_data_size,
                    )
                    if need_parity0:
                        recovered['recv_1_ref'] = parity0
                    if need_parity1:
                        recovered['recv_2_ref'] = parity1
                owner_idx += 1
        else:
            # Continuous blocks: recv stripe N+1 overlaps decode/encode stripe N (mirrors save).
            owner_plans, recovered = _build_hw_owner_codeword_plans(
                owner_rigs=owner_rigs,
                my_rig=my_rig,
                ecnaive_k=ecnaive_k,
                ecnaive_n=ecnaive_n,
                rig_to_si=rig_to_si,
                recv_pool=recv_pool,
                block_data_size=block_data_size,
                store_bufs=store_bufs,
                recovered_slot_pool=recovered_slot_pool,
                parity_pool_0=parity_pool_0,
                parity_pool_1=parity_pool_1,
            )
            n_stripes = _run_hw_failed_recv_decode_pipeline(
                native=native,
                manager=manager,
                rank=rank,
                world_size=world_size,
                source_ranks=source_ranks,
                ecnaive_k=ecnaive_k,
                ecnaive_n=ecnaive_n,
                recv_pool=recv_pool,
                block_data_size=block_data_size,
                owner_plans=owner_plans,
                stripe_bytes=stripe_bytes,
            )
            native.submit_send_sentinels(num_channels)
            native.submit_recv_sentinels(num_channels)
            native.wait_for_encoding_completion()
            _t['network_recv'] = time.time() - _t0_net
            logger.debug(
                "EC-NAIVE hw recovery: rank %d pipelined %d recv/decode stripes "
                "(stripe_bytes=%d, block_bytes=%d)",
                rank, n_stripes, stripe_bytes, block_data_size,
            )

        _t['network_encode'] = time.time() - _t0_decode

        # Copy recovered refs to pre-allocated store buffers (not timed)
        for _name in ('own_data0', 'my_data1', 'recv_0', 'recv_1', 'recv_2'):
            _ref = recovered.pop(f'{_name}_ref', None)
            if _ref is not None:
                store_bufs[_name].copy_(_ref[:store_bufs[_name].numel()])
                recovered[_name] = store_bufs[_name]

        # ── Store recovered blocks ──
        if recovered:
            manager.store_recovered_blocks(rank, recovered)
            logger.debug(
                f"EC-NAIVE hw recovery: stored {len(recovered)} recovered "
                f"blocks for rank {rank} via unified decode+encode"
            )

        # ── Assemble tensor_buffer from own k data blocks ──
        d0 = recovered.get('own_data0')
        d1 = recovered.get('my_data1')
        if d0 is None or d1 is None:
            raise RuntimeError(
                f"EC-NAIVE hw recovery: missing own data blocks for rank {rank}"
            )
        tensor_buffer = tensor_buffer_pre
        tensor_buffer[:d0.numel()].copy_(d0)
        if actual_tensor_size > 0:
            end = min(d1.numel(), max(0, actual_tensor_size - d0.numel()))
            tensor_buffer[d0.numel():d0.numel() + end].copy_(d1[:end])
        if actual_tensor_size > 0:
            tensor_buffer = tensor_buffer[:actual_tensor_size]

        t_rebuild = time.time()
        # Debug: compare RS-recovered tensor_buffer with main_payload
        if args.ecnaive_hw_debug and isinstance(main_payload.get("tensor_buffer"), torch.Tensor):
            orig_tb = main_payload["tensor_buffer"]
            orig = orig_tb.detach().reshape(-1).view(torch.uint8)
            recv = tensor_buffer.reshape(-1)
            if orig.numel() == recv.numel():
                mismatch = (orig[:recv.numel()] != recv).nonzero(as_tuple=False)
                nz_orig = orig.nonzero(as_tuple=False).numel()
                nz_recv = recv.nonzero(as_tuple=False).numel()
                zero_orig = orig.numel() - nz_orig
                zero_recv = recv.numel() - nz_recv
                if mismatch.numel() > 0:
                    first = mismatch[0].item()
                    logger.error(
                        f"EC-NAIVE hw debug: MISMATCH at byte {first}: "
                        f"orig=0x{orig[first].item():02x} "
                        f"recv=0x{recv[first].item():02x} "
                        f"(total mismatches: {mismatch.numel()}/{orig.numel()})"
                    )
                else:
                    logger.debug(
                        f"EC-NAIVE hw debug: tensor_buffer matches main_payload "
                        f"({orig.numel()} bytes)"
                    )
                logger.debug(
                    f"EC-NAIVE hw debug: zero-rate orig={zero_orig}/{orig.numel()} "
                    f"({100*zero_orig/orig.numel():.1f}%) "
                    f"recv={zero_recv}/{recv.numel()} "
                    f"({100*zero_recv/recv.numel():.1f}%)"
                )
            else:
                logger.error(
                    f"EC-NAIVE hw debug: SIZE MISMATCH "
                    f"orig={orig.numel()} vs recv={recv.numel()}"
                )
            # Use main_payload to continue training
            logger.debug(
                f"EC-NAIVE hw debug: rank {rank} rebuilding from main_payload "
                f"instead of RS-recovered tensor_buffer"
            )
            state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
                main_payload, flat_key_roots=flat_key_roots,
            )
        else:
            state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
                {"tensor_buffer": tensor_buffer,
                 "tensor_infos": tensor_infos,
                 "non_tensor_data": main_payload.get("non_tensor_data", {})},
                flat_key_roots=flat_key_roots,
            )

        _t['rebuild_sd'] = time.time() - t_rebuild
        logger.debug(
            f"EC-NAIVE hw recovery: rank {rank} recovered via encode_ec_blocks "
            f"(k={ecnaive_k}, n={ecnaive_n})"
        )

    else:
        # ═══════════════════════════════════════════════════════════════
        # UNAFFECTED rank (different group): no-op on ASIO channels
        # ═══════════════════════════════════════════════════════════════
        native.reset_encoding_completion_flags()
        native.submit_send_sentinels(num_channels)
        native.submit_recv_sentinels(num_channels)
        native.wait_for_encoding_completion()
        _t['network_encode'] = time.time() - _t0_net

        t_rebuild = time.time()
        state_dict = _reconstruct_full_state_dict_from_main_tensor_buffer(
            main_payload, flat_key_roots=flat_key_roots,
        )
        _t['rebuild_sd'] = time.time() - t_rebuild

    _t['barrier'] = barrier_s
    native_timing = _native_ft_timing(native)
    _t['net_s'] = native_timing['net_s']
    _t['encode_s'] = native_timing['encode_s']
    _t['total'] = (
        _t.get('network_encode', 0.0)
        + _t.get('rebuild_sd', 0.0)
    )
    from megatron.training.global_vars import set_ft_load_timing_context
    set_ft_load_timing_context("EC-NAIVE", "HW recovery", _t)

    # NOTE: do not call manager.cleanup() or native.stop() here.
    # The C++ destructor double-frees RDMA resources used during RS decode.
    # The process exits shortly after load, so leaving cleanup to __del__ is safe.

    return state_dict


def save_ecnaive_legacy_checkpoint(
    state_dict: Dict[str, Any], checkpoint_name: str, write_to_disk: bool = True
) -> None:
    t0 = time.time()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    manager = ECNAIVEManager()
    manager.init_ecnaive_if_enabled()
    if manager._ecnaive_native is None:
        raise RuntimeError("EC-NAIVE native module is not available in legacy save path")

    t0 = time.time()
    decomposed, save_copy_s, save_flatten_s, decompose_s = decompose_state_dict_for_save(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    logger.debug(
        "ECNAIVE save timing: copy %.3fs flatten %.3fs decompose %.3fs",
        save_copy_s, save_flatten_s, decompose_s,
    )

    t0 = time.time()
    safety_margin = max(int(total_tensor_size * 0.01), manager.ecnaive_buffer_size)
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
                global_offset=tuple(info.global_offset) if info.global_offset else tuple(),
                shard_index=info.shard_index if info.shard_index is not None else 0,
                chunk_type="data",
                target_rank=rank,
                source_rank=rank,
            )
        )
        offset += info.size_bytes

    t0 = time.time()
    rank_metadata, _ = _build_global_registry(local_tensor_metadata, {})
    metadata_s = time.time() - t0

    t0 = time.time()
    blocks = _allocate_ecnaive_blocks(manager, rank_metadata)
    buffer_alloc_s = time.time() - t0

    if manager.use_rdma:
        manager.register_buffer(tensor_buffer)

    if world_size > 1:
        torch.distributed.barrier()
    e2e_t0 = time.time()

    d2h_t0 = time.time()
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
            tensor_buffer[info.offset : info.offset + tensor_bytes].copy_(
                tensor_view, non_blocking=True
            )
            decomposed.tensor_data[i] = None
    d2h_stream.synchronize()
    del decomposed.tensor_data
    d2h_s = time.time() - d2h_t0

    encode_t0 = time.time()
    _encode_with_native(
        manager=manager,
        tensor_buffer=tensor_buffer,
        actual_data_bytes=total_tensor_size,
        ecnaive_blocks=blocks,
    )
    network_encode_s = time.time() - encode_t0
    native_timing = _native_ft_timing(manager._ecnaive_native)
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
    logger.info(
        "EC-NAIVE save timing: e2e_s=%(e2e_s).2fs d2h_s=%(d2h_s).2fs "
        "network_encode_s=%(network_encode_s).2fs net_s=%(net_s).2fs encode_s=%(encode_s).2fs",
        summary,
    )

    if write_to_disk:
        _save_ecnaive_pt_files(
            checkpoint_name=checkpoint_name,
            rank=rank,
            non_tensor_data=decomposed.non_tensor_data,
            tensor_infos=decomposed.tensor_infos,
            blocks=blocks,
            full_tensor_buffer=tensor_buffer[:total_tensor_size],
            flat_key_roots=decomposed.flat_key_roots,
            manager=manager,
            all_tensor_infos=rank_metadata,  # store all ranks' metadata for HW recovery
        )
    else:
        logger.info(
            "EC-NAIVE save: skipping checkpoint file writes for this iteration"
        )

    if world_size > 1:
        torch.distributed.barrier()
