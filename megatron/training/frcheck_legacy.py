# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""LEGACY checkpoint path for FRCheck (POA-driven stripe encode with RDMA).
Layerwise: groups tensors by transformer layer index, encodes each layer
independently so per-layer data fits within SOURCE stripe capacity.
"""

import copy
import pickle
import os
import re
import struct
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from dataclasses import dataclass

from megatron.core.dist_checkpointing.strategies.frcheck_manager import (
    FRCheckManager,
    LayerStripeBufs,
    StripeRole,
)
from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
    allocate_hugepage_tensor,
)
from megatron.core.dist_checkpointing.strategies.state_dict_decomposer import (
    decompose_state_dict,
    DecomposedStateDict,
    TensorInfo,
    TensorMetadata,
    flatten_optimizer_fp32_params,
    reconstruct_state_dict,
    extract_tensors_from_continuous_buffer,
)

logger = getLogger(__name__)

_LAYER_KEY_RE = re.compile(r"\.layers\.(\d+)\b")


def _extract_layer_idx(key: str) -> int:
    """Extract transformer layer index from a tensor FQN, or -1 if not a layer."""
    m = _LAYER_KEY_RE.search(key)
    return int(m.group(1)) if m else -1


@dataclass(frozen=False)
class _LayerGroup:
    layer_idx: int
    tensor_infos: List
    tensor_data: List
    total_bytes: int = 0


@dataclass
class _RecoveryBufPool:
    """Reusable RDMA buffers sized to max per-layer block_size for this rank."""
    decoder_recv_bufs: List[torch.Tensor]
    failed_recv_buf: Optional[torch.Tensor]
    decoder_recovered_buf: Optional[torch.Tensor]
    failed_layer_buf: Optional[torch.Tensor]
    max_block_size: int


@dataclass
class _LayerEncodeResult:
    layer_name: str
    layer_idx: int
    block_size: int
    actual_sizes: List[int]
    tensor_infos: List
    total_bytes: int


def _group_by_layer(decomposed) -> List[_LayerGroup]:
    """Split decomposed state_dict into per-layer groups based on FQN key."""
    groups: Dict[int, _LayerGroup] = {}
    for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
        lidx = _extract_layer_idx(info.key)
        if lidx not in groups:
            groups[lidx] = _LayerGroup(layer_idx=lidx, tensor_infos=[], tensor_data=[])
        g = groups[lidx]
        g.tensor_infos.append(info)
        g.tensor_data.append(tensor)
        g.total_bytes += info.size_bytes

    # Sort: non-layer (-1) first, then by layer index
    result = sorted(groups.values(), key=lambda g: (0 if g.layer_idx < 0 else 1, g.layer_idx))
    return result


def _exchange_frcheck_group_metadata(
    tensor_infos: List[Any],
    layer_order: List[str],
    layer_metadata: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[int, List[Any]], Dict[int, List[str]], Dict[int, Dict[str, Dict[str, Any]]]]:
    """All-gather per-rank tensor + per-layer metadata (keys are global ranks)."""
    local_pkg = {
        "tensor_infos": tensor_infos,
        "layer_order": layer_order,
        "layer_metadata": layer_metadata,
    }
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        gathered: List[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_pkg)
    else:
        gathered = [local_pkg]

    all_tensor_infos: Dict[int, List[Any]] = {}
    all_layer_order: Dict[int, List[str]] = {}
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r, pkg in enumerate(gathered):
        if pkg is None:
            continue
        all_tensor_infos[r] = pkg.get("tensor_infos", [])
        all_layer_order[r] = pkg.get("layer_order", [])
        all_layer_metadata[r] = pkg.get("layer_metadata", {})
    return all_tensor_infos, all_layer_order, all_layer_metadata


def _checkpoint_dir_from_path(checkpoint_name: str) -> Path:
    checkpoint_path = Path(checkpoint_name)
    return checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent


def _path_from_gathered_entry(entry: Any, context: str) -> Optional[Path]:
    """Convert an all_gather_object entry into a Path."""
    if entry is None:
        return None
    if isinstance(entry, Path):
        return entry
    if isinstance(entry, str):
        return Path(entry)
    raise TypeError(
        f"FRCheck: expected path string in {context}, got {type(entry).__name__}"
    )


def _gather_all_frcheck_dirs(
    checkpoint_dir: Path,
    world_size: int,
) -> List[Optional[Path]]:
    """All-gather each rank's checkpoint directory path."""
    my_dir = str(checkpoint_dir)
    if world_size > 1 and torch.distributed.is_initialized():
        gathered: List[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered, my_dir)
        return [_path_from_gathered_entry(p, "frcheck dir gather") for p in gathered]
    return [Path(my_dir)]


def _primary_failed_rank_in_group(
    failed_global_ranks: List[int],
    group_members: List[int],
) -> Optional[int]:
    """Return the failed global rank in this FRCheck group, if any."""
    failed_set = set(failed_global_ranks or [])
    for member in group_members or []:
        if member in failed_set:
            return member
    return None



def _stripes_per_role(stripe_plans: List, role: StripeRole) -> int:
    return sum(1 for sp in stripe_plans if sp.role == role)


def _clear_accum_buffers(manager) -> None:
    """Zero parity accumulation buffers for a fresh layer."""
    if manager.parity1_accum is not None:
        manager.parity1_accum.zero_()
    if manager.parity2_accum is not None:
        manager.parity2_accum.zero_()


def _build_local_src_addrs(
    plan,
    my_node: int,
    recv_bufs: List,
    stripe_id: int,
    block_size: int,
) -> List[int]:
    """Build per-source local buffer addresses for encoder self-source slots."""
    addrs = [0, 0, 0, 0]
    rb = recv_bufs[stripe_id]
    if rb is None:
        return addrs
    for i, node_id in enumerate(plan.source_node_ids):
        if node_id == my_node:
            addrs[i] = int(rb.data_ptr()) + i * block_size
    return addrs


def _encode_one_layer(
    manager,
    native,
    tensor_buffer: torch.Tensor,
    layer_tensor_size: int,
    n: int,
    num_stripes: int,
    block_size: int,
    my_node: int,
    layer_name: str,
    layer_bufs: LayerStripeBufs,
    world_size: int = 1,
    _dbg: bool = False,
):
    """Copy SOURCE data, then per-stripe lockstep encode (GDR send + D2H mirror)."""
    stripe_plans = manager.stripe_plans
    src_bufs = layer_bufs.stripe_data_bufs
    mirror_bufs = layer_bufs.source_mirror_bufs
    recv_bufs = layer_bufs.recv_bufs
    p1_bufs = layer_bufs.parity1_bufs
    p2_bufs = layer_bufs.parity2_bufs
    gdr = manager.gdr_available

    _t_start, _t_copy = time.time(), 0.0
    data_addrs = [0] * num_stripes
    actual_sizes = [0] * num_stripes
    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}
    has_src = has_enc = has_par = False

    for stripe_id in range(num_stripes):
        plan = stripe_plans[stripe_id]
        if plan.role == StripeRole.SOURCE:
            blk_idx = src_block_per_node[my_node]
            src_block_per_node[my_node] += 1
            src_offset = blk_idx * block_size
            ncopy = min(block_size, max(0, layer_tensor_size - src_offset))
            actual_sizes[stripe_id] = ncopy
            buf = src_bufs[stripe_id]
            if buf is not None:
                if ncopy > 0:
                    _t0 = time.time()
                    buf[:ncopy].copy_(tensor_buffer[src_offset : src_offset + ncopy])
                    _t_copy += time.time() - _t0
                if ncopy < block_size:
                    buf[ncopy:].zero_()
                data_addrs[stripe_id] = buf.data_ptr()
            has_src = True
        elif plan.role == StripeRole.ENCODER and my_node in plan.source_node_ids:
            blk_idx = src_block_per_node[my_node]
            src_block_per_node[my_node] += 1
            src_offset = blk_idx * block_size
            ncopy = min(block_size, max(0, layer_tensor_size - src_offset))
            rb = recv_bufs[stripe_id]
            if rb is not None:
                slot = plan.source_node_ids.index(my_node)
                dst = rb[slot * block_size : (slot + 1) * block_size]
                if ncopy > 0:
                    _t0 = time.time()
                    dst[:ncopy].copy_(tensor_buffer[src_offset : src_offset + ncopy])
                    _t_copy += time.time() - _t0
                if ncopy < block_size:
                    dst[ncopy:].zero_()

    if _dbg:
        _t_elapsed = time.time() - _t_start
        logger.info(
            "[FRCHECK-DEBUG] %s Phase1 copy: %.4fs (src_copy=%.4fs zero=%.4fs) "
            "total_bytes=%d",
            layer_name, _t_elapsed, _t_copy, _t_elapsed - _t_copy,
            sum(actual_sizes),
        )

    native.reset_encoding_completion()

    _t_encode = time.time()
    for stripe_id in range(num_stripes):
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()

        plan = stripe_plans[stripe_id]
        if plan.role == StripeRole.ENCODER:
            rb = recv_bufs[stripe_id]
            p1 = p1_bufs[stripe_id]
            p2 = p2_bufs[stripe_id]
            if rb is not None and p1 is not None and p2 is not None:
                if my_node not in plan.source_node_ids:
                    rb.zero_()
                p1.zero_()
                p2.zero_()
                local_src = _build_local_src_addrs(
                    plan, my_node, recv_bufs, stripe_id, block_size,
                )
                native.submit_encoder_encode(
                    stripe_id, rb.data_ptr(), p1.data_ptr(), p2.data_ptr(),
                    block_size, local_src,
                )
                has_enc = True
        elif plan.role == StripeRole.PARITY_TARGET:
            p2 = p2_bufs[stripe_id]
            if p2 is not None:
                p2.zero_()
                native.submit_parity_recv(stripe_id, p2.data_ptr(), block_size)
                has_par = True
        elif plan.role == StripeRole.SOURCE and data_addrs[stripe_id] != 0:
            cpu_mirror = 0
            if gdr and mirror_bufs[stripe_id] is not None:
                cpu_mirror = mirror_bufs[stripe_id].data_ptr()
            native.submit_source_send(
                stripe_id, data_addrs[stripe_id], cpu_mirror, block_size,
            )

        if _dbg:
            logger.info(
                "[FRCHECK-DEBUG] %s stripe sid=%d submit role=%s",
                layer_name, stripe_id, plan.role.name,
            )
        native.wait_stripe(stripe_id)
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if _dbg:
            logger.info(
                "[FRCHECK-DEBUG] %s stripe sid=%d done",
                layer_name, stripe_id,
            )

    if has_src:
        native.submit_source_send_sentinel()
    if has_enc:
        native.submit_encoder_sentinel()
    if has_par:
        native.submit_parity_recv_sentinel()
    native.wait_for_encoding_completion()
    _t_wait = time.time() - _t_encode

    if _dbg:
        _t_total = time.time() - _t_start
        logger.info(
            "[FRCHECK-DEBUG] %s encode lockstep: wait=%.4fs total=%.4fs",
            layer_name, _t_wait, _t_total,
        )

    return actual_sizes


def _save_frcheck_stripe_files(
    manager,
    output_dir: str,
    rank: int,
    num_stripes: int,
    encode_results: List[_LayerEncodeResult],
) -> None:
    """Write all layer stripe files after encode completes (aligned with ecnaive)."""
    import concurrent.futures

    _FRBK_MAGIC = b"FRBK"
    stripe_plans = manager.stripe_plans

    def _write_frbk_shard(
        layer_name: str,
        block_size: int,
        stripe_id: int,
        role: int,
        ncopy: int,
        buf,
        suffix: str,
    ) -> None:
        if buf is None:
            return
        data = buf[:ncopy]
        if data.device.type != "cpu":
            data = data.cpu()
        stripe_dir = Path(output_dir) / layer_name / f"stripe_{stripe_id}"
        stripe_dir.mkdir(parents=True, exist_ok=True)
        path = stripe_dir / f"frcheck_shard_rank{rank}{suffix}.pt"
        hdr = struct.pack("<4sIIQQ", _FRBK_MAGIC, stripe_id, role, ncopy, block_size)
        with open(path, "wb") as f:
            f.write(hdr)
            if ncopy > 0:
                f.write(memoryview(data.numpy()))

    jobs: List[Tuple[str, int, int, int, int, Any, str]] = []
    for result in encode_results:
        layer_bufs = manager.get_layer_stripe_bufs(result.layer_idx)
        block_size = result.block_size
        actual_sizes = result.actual_sizes
        layer_name = result.layer_name
        gdr = manager.gdr_available
        for sid in range(num_stripes):
            plan = stripe_plans[sid]
            if plan.role == StripeRole.SOURCE:
                if gdr and layer_bufs.source_mirror_bufs[sid] is not None:
                    src_buf = layer_bufs.source_mirror_bufs[sid]
                else:
                    src_buf = layer_bufs.stripe_data_bufs[sid]
                jobs.append((
                    layer_name, block_size, sid, 0, actual_sizes[sid],
                    src_buf, "",
                ))
            elif plan.role == StripeRole.ENCODER:
                jobs.append((
                    layer_name, block_size, sid, 1, block_size,
                    layer_bufs.parity1_bufs[sid], "_p1",
                ))
                jobs.append((
                    layer_name, block_size, sid, 2, block_size,
                    layer_bufs.parity2_bufs[sid], "_p2",
                ))
            elif plan.role == StripeRole.PARITY_TARGET:
                jobs.append((
                    layer_name, block_size, sid, 2, block_size,
                    layer_bufs.parity2_bufs[sid], "",
                ))

    if not jobs:
        return
    n_workers = min(len(jobs), 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(
                _write_frbk_shard, layer_name, block_size, sid, role, ncopy, buf, suffix,
            )
            for layer_name, block_size, sid, role, ncopy, buf, suffix in jobs
        ]
        for fut in futures:
            fut.result()


def _write_layer_shards(
    manager,
    checkpoint_dir: Path,
    layer_name: str,
    rank: int,
    rg: int,
    n_source_my: int,
    n_encoder_my: int,
    n_parity_my: int,
    block_size: int,
    gdr: bool,
    tensor_buffer: Optional[torch.Tensor],
) -> None:
    """Write aggregated source / parity1 / parity2 shards for one layer."""
    layer_dir = checkpoint_dir / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)

    from megatron.training.legacy_io_utils import write_raw_block, MAGIC_FRCHECK, MAGIC_FRCHECK_BLOCK
    import struct

    if n_source_my > 0 and tensor_buffer is not None:
        source_size = n_source_my * block_size
        source_data = tensor_buffer[:source_size]
        if gdr:
            source_data = source_data.cpu()
        meta = pickle.dumps({
            "version": 2, "format": "frcheck_torch_legacy", "rank": rank,
            "layer_name": layer_name, "rank_in_group": rg, "role": "SOURCE",
            "num_blocks": n_source_my, "block_size": block_size,
        })
        sp = str(layer_dir / f"frcheck_source_rank{rank}.pt")
        with open(sp, "wb") as _f:
            _f.write(struct.pack("<4sQ", MAGIC_FRCHECK_BLOCK, len(meta)))
            _f.write(meta)
            _f.write(memoryview(source_data[:source_size].numpy()))

    if n_encoder_my > 0:
        encoder_size = n_encoder_my * block_size
        meta = pickle.dumps({
            "version": 2, "format": "frcheck_torch_legacy", "rank": rank,
            "layer_name": layer_name, "rank_in_group": rg, "role": "ENCODER",
            "num_blocks": n_encoder_my, "block_size": block_size,
        })
        ep = str(layer_dir / f"frcheck_encoder_rank{rank}.pt")
        with open(ep, "wb") as _f:
            _f.write(struct.pack("<4sQ", MAGIC_FRCHECK_BLOCK, len(meta)))
            _f.write(meta)
            _f.write(memoryview(manager.parity1_accum[:encoder_size].numpy()))

    if n_parity_my > 0:
        par_size = n_parity_my * block_size
        meta = pickle.dumps({
            "version": 2, "format": "frcheck_torch_legacy", "rank": rank,
            "layer_name": layer_name, "rank_in_group": rg, "role": "PARITY_TARGET",
            "num_blocks": n_parity_my, "block_size": block_size,
        })
        pp = str(layer_dir / f"frcheck_parity2_rank{rank}.pt")
        with open(pp, "wb") as _f:
            _f.write(struct.pack("<4sQ", MAGIC_FRCHECK_BLOCK, len(meta)))
            _f.write(meta)
            _f.write(memoryview(manager.parity2_accum[:par_size].numpy()))


def save_frcheck_legacy_checkpoint(state_dict: Dict[str, Any], checkpoint_name: str) -> None:
    """Write frcheck_main_rank*.pt + layer-wise source/parity shards."""
    t0 = time.time()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    from megatron.training import get_args
    args = get_args()
    _dbg = getattr(args, "frcheck_debug", False)

    # 1. Decompose state_dict
    flatten_optimizer_fp32_params(state_dict)
    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    logger.info(
        "FRCheck save: rank=%d total_tensor_size=%d n_tensors=%d",
        rank, total_tensor_size, len(decomposed.tensor_data),
    )

    logger.info(f"FRCHECK save timing: decompose+group {time.time()-t0:.3f}s")

    # 1.5 Init manager early (cached after first call) and snapshot global offsets.
    #     Per-layer grouping overwrites info.offset, so snapshot first.
    start_time = t0 = time.time()
    _global_offsets = {id(info): info.offset for info in decomposed.tensor_infos}

    # 2. Init FRCheck manager + RDMA + stripe plans (once, shared across layers)
    manager = FRCheckManager()
    manager.init_frcheck_if_enabled()
    native = manager.get_native()
    if native is None:
        raise RuntimeError("FRCheck legacy save: native module not available")
    if native.group_size() != native.n():
        raise RuntimeError(
            f"FRCheck legacy save: group_size={native.group_size()} != n={native.n()}"
        )

    n = native.n()
    num_stripes = native.num_stripes()
    block_size = manager.block_size
    rg = manager.rank_in_group
    my_node = rg + 1
    stripe_plans = manager.stripe_plans
    gdr = manager.gdr_available

    buf_device = torch.device("cuda") if gdr else torch.device("cpu")

    # 3. Setup output directory (same layout as ecnaive: files live in mp_rank_* dir)
    checkpoint_path = Path(checkpoint_name)
    checkpoint_dir = checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 4. Allocate cached full_buf (grows-only, reused across saves; copy deferred to per-layer loop)
    full_buf = manager.allocate_full_buf(total_tensor_size)
    logger.info(f"FRCHECK save timing: full tensor buf build {time.time()-t0:.3f}s")

    # 5. Group tensors by layer index
    t0 = time.time()
    layer_groups = _group_by_layer(decomposed)
    n_tensors = len(decomposed.tensor_data)
    del decomposed.tensor_data  # GPU refs now held by per-layer groups
    num_layers = len(layer_groups)
    logger.info(
        "FRCheck save: grouped %d layers from %d tensors (layers: %s)",
        num_layers, n_tensors,
        [(g.layer_idx, g.total_bytes) for g in layer_groups],
    )

    flat_key_roots = decomposed.flat_key_roots

    logger.info(f"FRCHECK save timing: layer group+manager init {time.time()-t0:.3f}s")

    # 4.5 Compute per-layer adaptive block sizes (first save, cached thereafter)
    manager.compute_layer_block_sizes(layer_groups)

    if _dbg:
        n_src_total = (n - 1) * (n - 2)
        n_source_my = sum(1 for p in stripe_plans if p.role == StripeRole.SOURCE)
        n_encoder_my = sum(1 for p in stripe_plans if p.role == StripeRole.ENCODER)
        n_parity_my = sum(1 for p in stripe_plans if p.role == StripeRole.PARITY_TARGET)
        logger.info(
            "[FRCHECK-DEBUG] rank=%d n=%d stripes=%d roles(src=%d enc=%d par=%d) gdr=%s",
            rank, n, num_stripes, n_source_my, n_encoder_my, n_parity_my, gdr,
        )
        for g in layer_groups:
            lidx = g.layer_idx
            blk = manager._layer_block_sizes[lidx]
            cap = blk * n_source_my
            pad = max(0, cap - g.total_bytes)
            lname = f"layer_{lidx}" if lidx >= 0 else "layer_common"
            logger.info(
                "[FRCHECK-DEBUG] %s: total=%d blk=%d cap=%d nsrc=%d pad=%d (%.1f%%)",
                lname, g.total_bytes, blk, cap, n_source_my, pad,
                100.0 * pad / cap if cap > 0 else 0,
            )

    # 5. Encode each layer (per-layer stripe buffers), then unified disk write
    t0 = time.time()
    local_layer_order: List[str] = []
    local_layer_metadata: Dict[str, Dict[str, Any]] = {}
    encode_results: List[_LayerEncodeResult] = []
    for group in layer_groups:
        layer_name = f"layer_{group.layer_idx}" if group.layer_idx >= 0 else "layer_common"
        layer_idx = group.layer_idx
        layer_block_size = manager._layer_block_sizes[layer_idx]
        layer_bufs = manager.get_layer_stripe_bufs(layer_idx)

        # Allocate or reuse cached per-layer contiguous buffer
        safety_margin = max(int(group.total_bytes * 0.01), 4096)
        layer_buf_size = group.total_bytes + safety_margin
        tensor_buffer = manager.allocate_layer_buffer(layer_idx, layer_buf_size, gdr)

        # Copy layer tensors into layer buffer and full_buf (async D2H via dedicated stream)
        offset = 0
        copy_stream = torch.cuda.Stream()
        with torch.cuda.stream(copy_stream):
            for info, tensor in zip(group.tensor_infos, group.tensor_data):
                tensor_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
                nbytes = tensor_view.numel()
                tensor_buffer[offset : offset + nbytes].copy_(tensor_view, non_blocking=True)
                full_buf[_global_offsets[id(info)] : _global_offsets[id(info)] + nbytes].copy_(
                    tensor_view, non_blocking=True
                )
                info.offset = offset
                offset += nbytes
        copy_stream.synchronize()

        group.tensor_data = []  # free GPU refs for this layer
        addr = tensor_buffer.data_ptr()
        if addr not in manager._rdma_registered_addrs:
            native.register_buffer(addr, tensor_buffer.numel())
            manager._rdma_registered_addrs.add(addr)
        logger.info(
            "FRCheck save: %s tensor_buffer %s, size=%d layer_blk=%d",
            layer_name, "GPU" if gdr else "CPU", layer_buf_size, layer_block_size,
        )

        actual_sizes = _encode_one_layer(
            manager, native, tensor_buffer, group.total_bytes,
            n, num_stripes, layer_block_size, my_node, layer_name,
            layer_bufs, world_size=world_size, _dbg=_dbg,
        )

        if _dbg:
            total_sent = sum(actual_sizes)
            cap = layer_block_size * n_source_my
            logger.info(
                "[FRCHECK-DEBUG] %s encode: sent=%d cap=%d pad=%d (%.1f%%), "
                "stripe_sizes=%s",
                layer_name, total_sent, cap, cap - total_sent,
                100.0 * total_sent / cap if cap > 0 else 0,
                [actual_sizes[s] for s in range(num_stripes)
                 if stripe_plans[s].role == StripeRole.SOURCE],
            )

        encode_results.append(_LayerEncodeResult(
            layer_name=layer_name,
            layer_idx=layer_idx,
            block_size=layer_block_size,
            actual_sizes=actual_sizes,
            tensor_infos=group.tensor_infos,
            total_bytes=group.total_bytes,
        ))

    logger.info(f"FRCHECK save timing: all layers encode {time.time()-t0:.3f}s")

    if world_size > 1:
        torch.distributed.barrier()

    t0 = time.time()
    _save_frcheck_stripe_files(
        manager, str(checkpoint_dir), rank, num_stripes, encode_results,
    )
    logger.info(f"FRCHECK save timing: stripe file write {time.time()-t0:.3f}s")

    for result in encode_results:
        local_layer_order.append(result.layer_name)
        local_layer_metadata[result.layer_name] = {
            "block_size": result.block_size,
            "actual_tensor_size": result.total_bytes,
            "tensor_infos": copy.deepcopy(result.tensor_infos),
        }
        layer_main = checkpoint_dir / result.layer_name / f"frcheck_layer_main_rank{rank}.pt"
        torch.save(
            {
                "version": 2,
                "format": "frcheck_torch_legacy",
                "rank": rank,
                "layer_name": result.layer_name,
                "n": n,
                "num_stripes": num_stripes,
                "block_size": result.block_size,
                "rank_in_group": rg,
                "gdr": gdr,
                "tensor_infos": result.tensor_infos,
                "actual_tensor_size": result.total_bytes,
            },
            layer_main,
        )

    torch.distributed.barrier()
    logger.info(f"FRCHECK legacy save: done in {time.time() - start_time:.2f}s")

    # 6. Write top-level main file with full tensor_buffer + non_tensor_data,
    #    matching the pattern of eccheck / eclatin / ecnaive.
    main_file = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    from megatron.training.legacy_io_utils import write_raw_checkpoint, MAGIC_FRCHECK
    # restore global offsets (per-layer encode clobbered them with local offsets)
    for info in decomposed.tensor_infos:
        info.offset = _global_offsets[id(info)]

    t_meta = time.time()
    all_tensor_infos, all_layer_order, all_layer_metadata = _exchange_frcheck_group_metadata(
        decomposed.tensor_infos, local_layer_order, local_layer_metadata,
    )
    all_actual_tensor_sizes = {
        r: sum(getattr(info, "size_bytes", 0) for info in infos)
        for r, infos in all_tensor_infos.items()
    }
    logger.info(
        "FRCheck save: group metadata exchange %.3fs (ranks=%d)",
        time.time() - t_meta, len(all_tensor_infos),
    )

    write_raw_checkpoint(
        str(main_file), MAGIC_FRCHECK,
        decomposed.non_tensor_data, decomposed.tensor_infos,
        full_buf[:total_tensor_size], total_tensor_size,
        version=2, format="frcheck_torch_legacy", rank=rank,
        group_id=manager.group_id, rank_in_group=rg,
        poa_path=manager.get_resolved_table_path() or native.path(),
        n=n, num_stripes=num_stripes, num_layers=num_layers,
        block_size=block_size, gdr=gdr,
        actual_tensor_size=total_tensor_size,
        flat_key_roots=list(flat_key_roots) if flat_key_roots else [],
        state_dict_keys=list(state_dict.keys()),
        group_member_ranks=manager.group_member_ranks,
        node_slot_to_global_rank=manager.node_slot_to_global_rank,
        layer_names=local_layer_order,
        all_tensor_infos=all_tensor_infos,
        all_layer_order=all_layer_order,
        all_layer_metadata=all_layer_metadata,
        all_actual_tensor_sizes=all_actual_tensor_sizes,
    )

    logger.info(
        "FRCheck save: done rank=%d node=%d gdr=%s layers=%d file=%s",
        rank, my_node, gdr, num_layers, main_file,
    )

    del _global_offsets

    if world_size > 1:
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Hardware recovery helpers
# ---------------------------------------------------------------------------

def _read_frbk_block(filepath: str) -> Optional[torch.Tensor]:
    """Read a FRBK-format per-stripe block file. Returns the data tensor (CPU uint8)."""
    import struct as _struct
    path = Path(filepath)
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        hdr = f.read(28)  # <4sIIQQ: magic(4) + stripe_id(4) + role(4) + size(8) + block_size(8) = 28
        if len(hdr) < 28:
            return None
        magic, stripe_id, role, data_size, block_sz = _struct.unpack("<4sIIQQ", hdr)
        if magic != b"FRBK":
            return None
        data = torch.empty(data_size, dtype=torch.uint8)
        if data_size > 0:
            f.readinto(data.numpy())
    return data


def _stripe_block_path(
    checkpoint_dir: Path,
    layer_name: str,
    stripe_id: int,
    shard_owner_rank: int,
    stripe_role: int,
) -> Path:
    """Return the on-disk path for a participant's stripe shard."""
    stripe_dir = checkpoint_dir / layer_name / f"stripe_{stripe_id}"
    if stripe_role == int(StripeRole.ENCODER):
        return stripe_dir / f"frcheck_shard_rank{shard_owner_rank}_p1.pt"
    if stripe_role == int(StripeRole.PARITY_TARGET):
        return stripe_dir / f"frcheck_shard_rank{shard_owner_rank}.pt"
    return stripe_dir / f"frcheck_shard_rank{shard_owner_rank}.pt"


def _read_stripe_block(
    checkpoint_dir: Path,
    layer_name: str,
    stripe_id: int,
    shard_owner_rank: int,
    stripe_role: int,
) -> Optional[torch.Tensor]:
    """Read a stripe shard from a specific checkpoint directory."""
    filepath = _stripe_block_path(
        checkpoint_dir, layer_name, stripe_id, shard_owner_rank, stripe_role,
    )
    return _read_frbk_block(str(filepath))


def _read_local_frcheck_main_payload(
    checkpoint_dir: Path,
    rank: int,
    load_tensor_buffer: bool = True,
) -> Dict[str, Any]:
    """Read this rank's main.pt from local disk only (no collective)."""
    main_path = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    if not main_path.is_file():
        raise FileNotFoundError(f"FRCheck legacy: missing main file {main_path}")
    from megatron.training.legacy_io_utils import (
        is_raw_format,
        read_raw_checkpoint,
        read_raw_checkpoint_metadata,
        MAGIC_FRCHECK,
    )
    if is_raw_format(str(main_path), MAGIC_FRCHECK):
        if load_tensor_buffer:
            return read_raw_checkpoint(str(main_path), MAGIC_FRCHECK)
        return read_raw_checkpoint_metadata(str(main_path), MAGIC_FRCHECK)
    payload = torch.load(main_path, map_location="cpu", weights_only=False)
    if not load_tensor_buffer:
        payload["tensor_buffer"] = None
    return payload


def _load_frcheck_main_payload(
    checkpoint_dir: Path,
    rank: int,
    world_size: int,
    load_tensor_buffer: bool = True,
) -> Dict[str, Any]:
    """Load main payload; recover failed-rank metadata from group fields in main."""
    main_path = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    local_payload: Optional[Dict[str, Any]] = None
    local_error: Optional[str] = None
    if main_path.is_file():
        try:
            local_payload = _read_local_frcheck_main_payload(
                checkpoint_dir, rank, load_tensor_buffer=load_tensor_buffer,
            )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

    if local_error is not None:
        raise RuntimeError(
            f"FRCheck legacy: failed reading main file {main_path}: {local_error}"
        )

    if world_size <= 1 or not torch.distributed.is_initialized():
        if local_payload is None:
            raise FileNotFoundError(f"FRCheck legacy: missing main file {main_path}")
        return local_payload

    if local_payload is not None:
        stripped = {k: v for k, v in local_payload.items() if k != "tensor_buffer"}
    else:
        stripped = None

    gathered: List[Optional[Dict[str, Any]]] = [None] * world_size
    torch.distributed.all_gather_object(gathered, stripped)

    if local_payload is not None:
        return local_payload

    for src_rank, payload in enumerate(gathered):
        if payload is None:
            continue
        resolved = dict(payload)
        all_ti = resolved.get("all_tensor_infos")
        if all_ti and rank in all_ti:
            logger.info(
                "FRCheck legacy: frcheck_main_rank%d.pt missing locally; "
                "recovered tensor_infos from rank %d",
                rank, src_rank,
            )
            resolved["tensor_infos"] = all_ti[rank]
            all_sizes = resolved.get("all_actual_tensor_sizes", {})
            if rank in all_sizes:
                resolved["actual_tensor_size"] = all_sizes[rank]
            else:
                resolved["actual_tensor_size"] = sum(
                    getattr(info, "size_bytes", 0) for info in all_ti[rank]
                )
            resolved["tensor_buffer"] = None
            return resolved
        if "tensor_infos" in resolved:
            logger.warning(
                "FRCheck legacy: using rank %d tensor_infos as fallback for rank %d "
                "(old checkpoint without all_tensor_infos)",
                src_rank, rank,
            )
            resolved["tensor_buffer"] = None
            return resolved

    raise FileNotFoundError(
        f"FRCheck legacy: frcheck_main_rank{rank}.pt missing on all ranks under {checkpoint_dir}"
    )


def _frcheck_metadata_from_payload(
    main_payload: Dict[str, Any],
    failed_rank: Optional[int],
) -> Tuple[Dict[int, List[Any]], Dict[int, List[str]], Dict[int, Dict[str, Dict[str, Any]]]]:
    """Return group metadata dicts keyed by global rank (from main extra fields)."""
    all_tensor_infos = main_payload.get("all_tensor_infos") or {}
    all_layer_order = main_payload.get("all_layer_order") or {}
    all_layer_metadata = main_payload.get("all_layer_metadata") or {}

    if all_tensor_infos and all_layer_order and all_layer_metadata:
        return all_tensor_infos, all_layer_order, all_layer_metadata

    # Backward compat: single-rank fields only.
    rank = int(main_payload.get("rank", 0))
    layer_names = main_payload.get("layer_names", [])
    if not all_tensor_infos and main_payload.get("tensor_infos") is not None:
        all_tensor_infos = {rank: main_payload["tensor_infos"]}
    if not all_layer_order and layer_names:
        all_layer_order = {rank: list(layer_names)}
    if not all_layer_metadata and failed_rank is not None:
        logger.warning(
            "FRCheck recovery: checkpoint missing all_layer_metadata; "
            "per-layer metadata may be incomplete",
        )
    return all_tensor_infos, all_layer_order, all_layer_metadata


def _read_local_layer_metadata(
    checkpoint_dir: Path,
    layer_name: str,
    rank: int,
) -> Optional[Dict[str, Any]]:
    """Read per-layer metadata file for this rank, if present."""
    layer_main_path = checkpoint_dir / layer_name / f"frcheck_layer_main_rank{rank}.pt"
    if not layer_main_path.is_file():
        return None
    return torch.load(layer_main_path, map_location="cpu", weights_only=False)


def _resolve_layer_block_size(
    rank: int,
    layer_name: str,
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]],
    saved_block_size: int,
) -> int:
    """Per-layer adaptive block size from group metadata (matches save/preload)."""
    meta = all_layer_metadata.get(rank, {}).get(layer_name, {})
    return int(meta.get("block_size", saved_block_size) or saved_block_size)


def _max_layer_block_size(
    rank: int,
    all_layer_order: Dict[int, List[str]],
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]],
    saved_block_size: int,
) -> int:
    """Largest adaptive block_size across this rank's encode layers."""
    max_bs = saved_block_size
    for layer_name in all_layer_order.get(rank, []):
        max_bs = max(
            max_bs,
            _resolve_layer_block_size(
                rank, layer_name, all_layer_metadata, saved_block_size,
            ),
        )
    return max_bs


def _allocate_recovery_buf_pool(
    native,
    n: int,
    max_block_size: int,
    is_failed: bool,
    is_decoder: bool,
    is_helper: bool,
) -> Optional[_RecoveryBufPool]:
    """Pre-allocate stripe recovery buffers once before the per-layer network loop."""
    num_helper = max(n - 3, 0)
    num_source_stripes = (n - 1) * (n - 2)
    need_pool = is_failed or is_decoder or is_helper
    if not need_pool:
        return None

    decoder_recv_bufs: List[torch.Tensor] = []
    if is_decoder and num_helper > 0:
        decoder_recv_bufs = [
            allocate_hugepage_tensor(max_block_size, fallback_pin_memory=True)
            for _ in range(num_helper)
        ]

    failed_recv_buf = (
        allocate_hugepage_tensor(max_block_size, fallback_pin_memory=True)
        if is_failed else None
    )
    decoder_recovered_buf = (
        allocate_hugepage_tensor(max_block_size, fallback_pin_memory=True)
        if is_decoder else None
    )
    failed_layer_buf = (
        allocate_hugepage_tensor(
            num_source_stripes * max_block_size, fallback_pin_memory=True,
        )
        if is_failed else None
    )

    for buf in (
        decoder_recv_bufs
        + ([failed_recv_buf] if failed_recv_buf is not None else [])
        + ([decoder_recovered_buf] if decoder_recovered_buf is not None else [])
        + ([failed_layer_buf] if failed_layer_buf is not None else [])
    ):
        native.register_buffer(buf.data_ptr(), buf.numel())

    return _RecoveryBufPool(
        decoder_recv_bufs=decoder_recv_bufs,
        failed_recv_buf=failed_recv_buf,
        decoder_recovered_buf=decoder_recovered_buf,
        failed_layer_buf=failed_layer_buf,
        max_block_size=max_block_size,
    )


def _map_layer_buf_to_full_buf(
    layer_buf: torch.Tensor,
    layer_infos: List,
    global_tensor_infos: List,
    full_buf: torch.Tensor,
) -> int:
    """Copy recovered layer buffer into full_buf using key-based global offset mapping."""
    key_to_global_offset: Dict[str, int] = {}
    for info in global_tensor_infos:
        key = getattr(info, "key", "")
        if key and key not in key_to_global_offset:
            key_to_global_offset[key] = getattr(info, "offset", 0)

    copied = 0
    for info in layer_infos:
        key = getattr(info, "key", "")
        if not key or key not in key_to_global_offset:
            continue
        local_offset = getattr(info, "offset", 0)
        global_offset = key_to_global_offset[key]
        size = getattr(info, "size_bytes", 0)
        if (
            size > 0
            and global_offset + size <= full_buf.numel()
            and local_offset + size <= layer_buf.numel()
        ):
            full_buf[global_offset:global_offset + size].copy_(
                layer_buf[local_offset:local_offset + size]
            )
            copied += size
    return copied


def _normalize_stripe_block(
    blk: Optional[torch.Tensor],
    layer_block_size: int,
    rank: int,
    layer_name: str,
    stripe_id: int,
) -> torch.Tensor:
    """Pad/truncate a stripe block to layer_block_size."""
    if blk is None or blk.numel() == 0:
        return torch.zeros(layer_block_size, dtype=torch.uint8)
    if blk.numel() < layer_block_size:
        padded = torch.zeros(layer_block_size, dtype=torch.uint8)
        padded[:blk.numel()] = blk
        return padded
    if blk.numel() > layer_block_size:
        logger.warning(
            "FRCheck recovery: rank %d layer %s stripe %d block exceeds "
            "layer_block_size (%d > %d), truncating",
            rank, layer_name, stripe_id, blk.numel(), layer_block_size,
        )
        return blk[:layer_block_size].contiguous()
    return blk.contiguous()


def _preload_recovery_stripe_blocks(
    n_encode_iters: int,
    all_layer_order: Dict[int, List[str]],
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]],
    recovery_stripe_plans: List[Dict],
    rank: int,
    my_node: int,
    saved_block_size: int,
    all_frcheck_dirs: List[Optional[Path]],
) -> Dict[int, Dict[int, torch.Tensor]]:
    """Read decoder/helper stripe blocks keyed by encode iteration index."""
    import concurrent.futures

    result: Dict[int, Dict[int, torch.Tensor]] = {
        i: {} for i in range(n_encode_iters)
    }
    if not recovery_stripe_plans:
        return result

    participating = [
        p for p in recovery_stripe_plans
        if my_node == p['decoder_node'] or my_node in p['helper_nodes']
    ]
    if not participating:
        return result

    my_order = all_layer_order.get(rank, [])
    participant_frcheck_dir = (
        all_frcheck_dirs[rank] if rank < len(all_frcheck_dirs) else None
    )
    if participant_frcheck_dir is None:
        logger.warning("FRCheck recovery: missing frcheck dir for participant=%d", rank)
        return result

    def _load_one(job: Tuple[int, Dict]) -> Tuple[int, int, torch.Tensor]:
        encode_iter, plan = job
        if encode_iter >= len(my_order):
            return encode_iter, plan['stripe_id'], torch.zeros(1, dtype=torch.uint8)
        layer_name = my_order[encode_iter]
        meta = all_layer_metadata.get(rank, {}).get(layer_name, {})
        layer_block_size = int(
            meta.get("block_size", saved_block_size) or saved_block_size
        )
        sid = plan['stripe_id']
        blk = _read_stripe_block(
            participant_frcheck_dir,
            layer_name,
            sid,
            rank,
            plan['original_role'],
        )
        blk = _normalize_stripe_block(
            blk, layer_block_size, rank, layer_name, sid,
        )
        return encode_iter, sid, blk

    jobs = [
        (encode_iter, plan)
        for encode_iter in range(n_encode_iters)
        for plan in participating
    ]
    n_workers = min(len(jobs), 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        for encode_iter, sid, blk in executor.map(_load_one, jobs):
            result[encode_iter][sid] = blk
    return result


# ---------------------------------------------------------------------------
# Per-layer recovery pipeline (network + decode only)
# ---------------------------------------------------------------------------

def _recover_one_layer_network(
    manager,
    native,
    layer_name: str,
    layer_block_size: int,
    layer_total_bytes: int,
    n: int,
    rank: int,
    preloaded_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Run stripe-level RS decode recovery for one layer.

    Uses RDMA point-to-point for data transfer (send_to_peer / recv_from_peer).
    Three task types per stripe: HELPER, DECODER, FAILED_RANK.

    Processes stripes in a batched loop: per stripe, decoder posts recvs in threads
    (GIL released during recv_from_peer), helpers send, decoder decodes and sends
    to failed rank. No inter-stripe barriers — the TCP handshake in send_data/recv_data
    provides per-channel synchronization. For n≤8 and typical block sizes, the RS pool
    is the bottleneck, not stripe dispatch.
    """
    my_node = manager.rank_in_group + 1
    layer_timing: Dict[str, float] = {'rdma_xfer_s': 0.0, 'decode_s': 0.0}

    if not manager.recovery_stripe_plans:
        return None, layer_timing

    decoder_stripes = [p for p in manager.recovery_stripe_plans if my_node == p['decoder_node']]
    helper_stripes = [p for p in manager.recovery_stripe_plans if my_node in p['helper_nodes']]
    failed_stripes = [p for p in manager.recovery_stripe_plans if my_node == p['failed_node']]
    is_failed = len(failed_stripes) > 0

    logger.info(
        "FRCheck recovery layer %s: rank %d — %d decoder, %d helper, %d failed stripes",
        layer_name, rank, len(decoder_stripes), len(helper_stripes), len(failed_stripes),
    )

    my_blocks: Dict[int, torch.Tensor] = {}
    for plan in decoder_stripes + helper_stripes:
        sid = plan['stripe_id']
        blk = preloaded_blocks.get(sid)
        if blk is None:
            blk = torch.zeros(layer_block_size, dtype=torch.uint8)
            native.register_buffer(blk.data_ptr(), blk.numel())
        my_blocks[sid] = blk

    num_source_stripes = (n - 1) * (n - 2)
    layer_buf = None
    if is_failed:
        if buf_pool is not None and buf_pool.failed_layer_buf is not None:
            layer_buf = buf_pool.failed_layer_buf[
                : num_source_stripes * layer_block_size
            ]
        else:
            layer_buf = allocate_hugepage_tensor(
                num_source_stripes * layer_block_size, fallback_pin_memory=True,
            )
        layer_buf.zero_()

    # Node ID → rig mapping
    node_to_rig = {i + 1: i for i in range(n)}

    # Tracks SOURCE block indices (matches save order: increment only for SOURCE stripes)
    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}

    import threading

    for stri_plan in manager.recovery_stripe_plans:
        sid = stri_plan['stripe_id']
        decoder_node = stri_plan['decoder_node']
        helper_nodes = stri_plan['helper_nodes']
        failed_node = stri_plan['failed_node']
        failed_pos = stri_plan['failed_pos']
        decoder_pos = stri_plan['decoder_pos']
        helper_positions = stri_plan['helper_positions']
        original_role = stri_plan['original_role']

        decoder_rig = node_to_rig[decoder_node]
        failed_rig = node_to_rig[failed_node]

        # --- Decoder: post async recvs; helpers send after recvs are posted ---
        recv_threads: List[threading.Thread] = []
        recv_bufs: List[torch.Tensor] = []
        recv_ready = threading.Event()
        recv_ready_count = {'n': 0}
        recv_ready_lock = threading.Lock()
        n_helper_recvs = len(helper_nodes) if my_node == decoder_node else 0

        if my_node == decoder_node:
            for hi, helper_node in enumerate(helper_nodes):
                helper_rig = node_to_rig[helper_node]
                if (
                    buf_pool is not None
                    and hi < len(buf_pool.decoder_recv_bufs)
                ):
                    recv_buf = buf_pool.decoder_recv_bufs[hi][:layer_block_size]
                else:
                    recv_buf = allocate_hugepage_tensor(
                        layer_block_size, fallback_pin_memory=True,
                    )
                    native.register_buffer(recv_buf.data_ptr(), recv_buf.numel())
                recv_bufs.append(recv_buf)

                def _recv_thread(
                    rb=recv_buf, hrig=helper_rig, timing=layer_timing,
                    ready=recv_ready, ready_count=recv_ready_count,
                    ready_lock=recv_ready_lock, n_ready=n_helper_recvs,
                ):
                    with ready_lock:
                        ready_count['n'] += 1
                        if ready_count['n'] >= n_ready:
                            ready.set()
                    _t0 = time.time()
                    native.recv_from_peer(hrig, rb.data_ptr(), rb.numel())
                    timing['rdma_xfer_s'] += time.time() - _t0

                t = threading.Thread(target=_recv_thread, daemon=True)
                t.start()
                recv_threads.append(t)

            if n_helper_recvs > 0:
                recv_ready.wait(timeout=30.0)

        # --- Helpers: send blocks to decoder ---
        if my_node in helper_nodes:
            my_block = my_blocks.get(sid)
            if my_block is None:
                my_block = torch.zeros(layer_block_size, dtype=torch.uint8)
                native.register_buffer(my_block.data_ptr(), my_block.numel())
            _t0 = time.time()
            native.send_to_peer(decoder_rig, my_block.data_ptr(), my_block.numel())
            layer_timing['rdma_xfer_s'] += time.time() - _t0

        # --- Decoder: join, decode, send to failed ---
        if my_node == decoder_node:
            for t in recv_threads:
                t.join()

            k = n - 2
            survivor_positions = [decoder_pos] + helper_positions
            survivor_addrs = []

            my_block = my_blocks.get(sid)
            if my_block is None:
                my_block = torch.zeros(layer_block_size, dtype=torch.uint8)
            survivor_addrs.append(my_block.data_ptr())
            for recv_buf in recv_bufs:
                survivor_addrs.append(recv_buf.data_ptr())

            if buf_pool is not None and buf_pool.decoder_recovered_buf is not None:
                recovered = buf_pool.decoder_recovered_buf[:layer_block_size]
            else:
                recovered = torch.zeros(layer_block_size, dtype=torch.uint8)
                native.register_buffer(recovered.data_ptr(), recovered.numel())
            _t0 = time.time()
            native.submit_stripe_decode(
                k=k,
                survivor_positions=survivor_positions,
                lost_position=failed_pos,
                survivor_addrs=survivor_addrs,
                recovered_addr=recovered.data_ptr(),
                block_size=layer_block_size,
            )
            layer_timing['decode_s'] += time.time() - _t0

            _t0 = time.time()
            native.send_to_peer(failed_rig, recovered.data_ptr(), recovered.numel())
            layer_timing['rdma_xfer_s'] += time.time() - _t0

        # --- Failed rank: recv decoded block ---
        if my_node == failed_node:
            if buf_pool is not None and buf_pool.failed_recv_buf is not None:
                recv_buf = buf_pool.failed_recv_buf[:layer_block_size]
            else:
                recv_buf = allocate_hugepage_tensor(
                    layer_block_size, fallback_pin_memory=True,
                )
                native.register_buffer(recv_buf.data_ptr(), recv_buf.numel())
            _t0 = time.time()
            native.recv_from_peer(decoder_rig, recv_buf.data_ptr(), recv_buf.numel())
            layer_timing['rdma_xfer_s'] += time.time() - _t0

            if layer_buf is not None and original_role == int(StripeRole.SOURCE):
                blk_idx = src_block_per_node[my_node]
                src_block_per_node[my_node] += 1
                offset = blk_idx * layer_block_size
                ncopy = min(layer_block_size, max(0, layer_total_bytes - offset))
                if ncopy > 0:
                    layer_buf[offset:offset + ncopy].copy_(recv_buf[:ncopy])

    if is_failed:
        n_stored = src_block_per_node.get(my_node, 0)
        logger.info(
            "FRCheck recovery: %s — stored %d SOURCE blocks (expected %d)",
            layer_name, n_stored, (n - 1) * (n - 2),
        )

    return layer_buf, layer_timing


# ---------------------------------------------------------------------------
# Main recovery entry point
# ---------------------------------------------------------------------------

def recover_frcheck_legacy_hardware(
    checkpoint_name: str,
    failed_global_ranks: List[int],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Main entry point for FRCheck hardware recovery.

    Phases (timed separately):
    1. prep — RDMA init, main I/O, stripe preload, buffer pools (before network_encode)
    2. network_encode — per-layer RDMA RS recovery only
    3. rebuild_sd — reconstruct state_dict (survivor reuses cached main payload)
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    timings: Dict[str, float] = {
        'prep': 0.0,
        'main_io': 0.0,
        'disk_io': 0.0,
        'network_encode': 0.0,
        'rebuild_sd': 0.0,
        'rdma_xfer': 0.0,
        'decode': 0.0,
    }
    t_prep = time.time()

    manager = FRCheckManager()
    manager.init_frcheck_if_enabled()
    native = manager.get_native()
    if native is None:
        raise RuntimeError("FRCheck hardware recovery: native module not available")

    n = native.n()
    rg = manager.rank_in_group
    gdr = manager.gdr_available

    logger.info(
        "FRCheck hardware recovery: rank=%d group=%d rig=%d n=%d gdr=%s failed=%s",
        rank, manager.group_id, rg, n, gdr, failed_global_ranks,
    )

    manager.init_frcheck_hardware_recovery(failed_global_ranks)
    is_failed = rank in failed_global_ranks
    is_survivor = not is_failed
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    primary_failed = _primary_failed_rank_in_group(
        failed_global_ranks, manager.group_member_ranks or [],
    )

    # All ranks must participate in the same metadata all_gather (failed ranks may
    # lack local main.pt). Survivors load tensor_buffer locally afterward.
    survivor_main_cached: Optional[Dict[str, Any]] = None
    t_main = time.time()
    main_payload = _load_frcheck_main_payload(
        checkpoint_dir, rank, world_size, load_tensor_buffer=False,
    )
    if is_survivor:
        main_path = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
        if main_path.is_file():
            local_full = _read_local_frcheck_main_payload(
                checkpoint_dir, rank, load_tensor_buffer=True,
            )
            survivor_main_cached = dict(main_payload)
            survivor_main_cached["tensor_buffer"] = local_full.get("tensor_buffer")
        else:
            survivor_main_cached = main_payload
    timings['main_io'] = time.time() - t_main

    all_tensor_infos, all_layer_order, all_layer_metadata = _frcheck_metadata_from_payload(
        main_payload, primary_failed,
    )

    if is_failed and rank in all_tensor_infos:
        global_tensor_infos = all_tensor_infos[rank]
        main_payload["tensor_infos"] = global_tensor_infos
        all_sizes = main_payload.get("all_actual_tensor_sizes", {})
        if rank in all_sizes:
            total_tensor_size = int(all_sizes[rank])
        else:
            total_tensor_size = sum(
                getattr(info, "size_bytes", 0) for info in global_tensor_infos
            )
        main_payload["actual_tensor_size"] = total_tensor_size
    else:
        global_tensor_infos = main_payload.get("tensor_infos", [])
        total_tensor_size = int(main_payload.get("actual_tensor_size", 0))

    flat_key_roots = main_payload.get("flat_key_roots", [])
    saved_block_size = int(main_payload.get("block_size", 64 * 1024 * 1024))

    group_members = manager.group_member_ranks or [rank]
    if all_layer_order:
        n_encode_iters = max(len(all_layer_order.get(r, [])) for r in group_members)
    else:
        n_encode_iters = len(main_payload.get("layer_names", []))
    logger.info(
        "FRCheck recovery: main loaded encode_iters=%d group_layer_orders=%s",
        n_encode_iters,
        {r: all_layer_order.get(r, []) for r in group_members},
    )

    all_frcheck_dirs = _gather_all_frcheck_dirs(checkpoint_dir, world_size)
    involved = is_failed or bool(manager.recovery_stripe_plans)
    my_node = manager.rank_in_group + 1

    t_disk = time.time()
    if involved and not is_failed:
        preloaded = _preload_recovery_stripe_blocks(
            n_encode_iters, all_layer_order, all_layer_metadata,
            manager.recovery_stripe_plans, rank, my_node, saved_block_size,
            all_frcheck_dirs,
        )
        for blocks in preloaded.values():
            for blk in blocks.values():
                if blk.numel() > 1:
                    native.register_buffer(blk.data_ptr(), blk.numel())
    else:
        preloaded = {i: {} for i in range(n_encode_iters)}
    n_disk_blocks = sum(len(blocks) for blocks in preloaded.values())
    timings['disk_io'] = time.time() - t_disk
    if involved:
        logger.info(
            "FRCheck recovery: disk preload rank=%d blocks=%d time=%.2fs",
            rank, n_disk_blocks, timings['disk_io'],
        )

    if is_survivor and not involved:
        logger.info(
            "FRCheck recovery: rank %d not involved in recovery, loading directly",
            rank,
        )

    max_block_size = _max_layer_block_size(
        rank, all_layer_order, all_layer_metadata, saved_block_size,
    )
    is_decoder = any(
        my_node == p['decoder_node'] for p in manager.recovery_stripe_plans
    )
    is_helper = any(
        my_node in p['helper_nodes'] for p in manager.recovery_stripe_plans
    )
    buf_pool = _allocate_recovery_buf_pool(
        native, n, max_block_size, is_failed, is_decoder, is_helper,
    )

    full_buf = None
    if is_failed:
        safety_margin = max(int(total_tensor_size * 0.01), 4096)
        full_buf = manager.allocate_full_buf(total_tensor_size + safety_margin)
        full_buf.zero_()
        logger.info(
            "FRCheck recovery: rank %d allocated full_buf size=%d",
            rank, total_tensor_size,
        )

    timings['prep'] = time.time() - t_prep

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    t_net = time.time()
    for encode_iter in range(n_encode_iters):
        if not involved:
            continue

        my_order = all_layer_order.get(rank, main_payload.get("layer_names", []))
        if encode_iter >= len(my_order):
            if is_failed:
                continue
            layer_name = f"encode_iter_{encode_iter}"
            layer_block_size = saved_block_size
            actual_size = 0
            layer_infos: List[Any] = []
        else:
            layer_name = my_order[encode_iter]
            layer_block_size = _resolve_layer_block_size(
                rank, layer_name, all_layer_metadata, saved_block_size,
            )
            if is_failed:
                meta = all_layer_metadata.get(rank, {}).get(layer_name)
                if meta is None:
                    failed_dir = (
                        all_frcheck_dirs[rank]
                        if rank < len(all_frcheck_dirs)
                        else checkpoint_dir
                    )
                    meta = _read_local_layer_metadata(failed_dir, layer_name, rank)
                if meta is None:
                    logger.warning(
                        "FRCheck recovery: no metadata for rank %d layer %s (iter %d)",
                        rank, layer_name, encode_iter,
                    )
                    continue
                actual_size = int(meta.get("actual_tensor_size", 0))
                layer_infos = meta.get("tensor_infos", [])
                if actual_size == 0:
                    continue
            else:
                actual_size = 0
                layer_infos = []

        layer_buf, layer_timing = _recover_one_layer_network(
            manager, native, layer_name,
            layer_block_size, actual_size if is_failed else 0,
            n, rank, preloaded_blocks=preloaded.get(encode_iter, {}),
            buf_pool=buf_pool,
        )
        timings['rdma_xfer'] += layer_timing.get('rdma_xfer_s', 0.0)
        timings['decode'] += layer_timing.get('decode_s', 0.0)

        if is_failed and layer_buf is not None and full_buf is not None:
            copied = _map_layer_buf_to_full_buf(
                layer_buf, layer_infos, global_tensor_infos, full_buf,
            )
            logger.info(
                "FRCheck recovery: %s copied %d bytes into full_buf (expected %d)",
                layer_name, copied, actual_size,
            )

    timings['network_encode'] = time.time() - t_net

    t_rebuild = time.time()
    if is_survivor:
        own_payload = survivor_main_cached
        if own_payload is None:
            own_payload = _read_local_frcheck_main_payload(
                checkpoint_dir, rank, load_tensor_buffer=True,
            )
        result = _reconstruct_from_main_payload(own_payload, flat_key_roots)
    else:
        main_payload['tensor_buffer'] = full_buf[:total_tensor_size].clone()
        result = _reconstruct_from_main_payload(main_payload, flat_key_roots)
    timings['rebuild_sd'] = time.time() - t_rebuild

    manager.end_recovery()
    return result, timings


def _reconstruct_from_main_payload(main_payload: Dict, flat_key_roots: list = None) -> Dict[str, Any]:
    """Reconstruct state_dict from a main payload that contains tensor_infos + tensor_buffer."""
    tensor_infos = main_payload.get("tensor_infos", [])
    non_tensor_data = main_payload.get("non_tensor_data", {})
    flat_key_roots = flat_key_roots or main_payload.get("flat_key_roots", [])

    # Check if tensor_buffer is in the payload
    tensor_buffer = main_payload.get("tensor_buffer")
    if tensor_buffer is not None:
        buf = tensor_buffer.detach().contiguous().reshape(-1).view(torch.uint8)
        tensor_data = extract_tensors_from_continuous_buffer(buf, tensor_infos)
    else:
        tensor_data = []

    decomposed = DecomposedStateDict(
        non_tensor_data=non_tensor_data,
        tensor_infos=tensor_infos,
        tensor_data=tensor_data,
        flat_key_roots=set(flat_key_roots) if flat_key_roots else set(),
    )
    result = reconstruct_state_dict(decomposed)
    from megatron.core.dist_checkpointing.strategies.state_dict_decomposer import (
        unflatten_optimizer_fp32_params,
    )
    unflatten_optimizer_fp32_params(result)
    return result


# ---------------------------------------------------------------------------
# Software failure recovery helpers
# ---------------------------------------------------------------------------

def _parse_poa_table(poa_path: str, n: int) -> List[List[int]]:
    """Parse POA table text file.

    Format: lines starting with # are comments. Data lines have n
    space-separated integers (1-based node IDs).

    Returns list of rows, each a list of n ints.
    """
    rows: List[List[int]] = []
    with open(poa_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != n:
                continue
            rows.append([int(p) for p in parts])
    if len(rows) != n * (n - 1):
        logger.warning(
            f"FRCheck: POA table at {poa_path} has {len(rows)} rows, "
            f"expected {n * (n - 1)} for n={n}"
        )
    return rows


def _find_source_stripes(
    poa_rows: List[List[int]], my_node: int, n: int
) -> List[Tuple[int, int]]:
    """Find stripes where *my_node* (1-based) is a SOURCE.

    Returns list of (stripe_id, blk_idx) sorted by stripe_id.
    blk_idx is the 0-based accumulation index for SOURCE blocks of this node.
    """
    k = n - 2  # SOURCE columns are 0 .. k-1
    result: List[Tuple[int, int]] = []
    blk_idx = 0
    for stripe_id, row in enumerate(poa_rows):
        for col in range(k):
            if row[col] == my_node:
                result.append((stripe_id, blk_idx))
                blk_idx += 1
                break
    return result


def _recover_frcheck_legacy_software(
    checkpoint_name: str, failed_ranks: List[int]
) -> Dict[str, Any]:
    """FRCheck legacy software failure recovery.

    Failed ranks read their SOURCE stripe blocks from local disk and reassemble
    the tensor buffer. Non-failed ranks load from main.pt directly.
    No network transfer — source data is always stored locally.
    """
    start_time = time.time()
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # --- Load main payload for metadata ---
    main_path = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    if not main_path.is_file():
        raise FileNotFoundError(
            f"FRCheck SW recovery: missing main file {main_path}. "
            f"In software failure, main.pt must exist on disk."
        )

    from megatron.training.legacy_io_utils import is_raw_format, read_raw_checkpoint, MAGIC_FRCHECK
    if is_raw_format(str(main_path), MAGIC_FRCHECK):
        main_payload = read_raw_checkpoint(str(main_path), MAGIC_FRCHECK)
    else:
        main_payload = torch.load(main_path, map_location="cpu", weights_only=False)

    is_failed = rank in failed_ranks

    if not is_failed:
        # --- Non-failed rank: normal load from main.pt ---
        flat_key_roots = main_payload.get("flat_key_roots", [])
        result = _reconstruct_from_main_payload(main_payload, flat_key_roots)
        logger.info(
            f"FRCheck SW recovery: rank {rank} (non-failed) loaded normally"
        )
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return result

    # --- Failed rank: recover from local SOURCE stripe blocks ---
    n = int(main_payload.get("n", 0))
    if n < 3:
        raise RuntimeError(f"FRCheck SW recovery: invalid n={n} in main payload")

    poa_path = main_payload.get("poa_path", "")
    if not poa_path:
        raise RuntimeError("FRCheck SW recovery: poa_path not found in main payload")

    layer_names = main_payload.get("layer_names", [])
    if not layer_names:
        raise RuntimeError("FRCheck SW recovery: layer_names empty in main payload")

    total_tensor_size = int(main_payload.get("actual_tensor_size", 0))
    global_tensor_infos = main_payload.get("tensor_infos", [])
    non_tensor_data = main_payload.get("non_tensor_data", {})
    flat_key_roots = main_payload.get("flat_key_roots", [])

    my_node = main_payload.get("rank_in_group", 0) + 1  # 1-based node id

    logger.info(
        f"FRCheck SW recovery: rank {rank} (failed) — n={n}, my_node={my_node}, "
        f"layers={len(layer_names)}, total_tensor={total_tensor_size / (1024**2):.1f} MB"
    )

    # Parse POA table once
    poa_rows = _parse_poa_table(poa_path, n)
    source_stripes_global = _find_source_stripes(poa_rows, my_node, n)
    num_source = len(source_stripes_global)
    logger.info(
        f"FRCheck SW recovery: rank {rank} has {num_source} SOURCE stripes"
    )

    # Allocate full recovered buffer
    safety_margin = max(int(total_tensor_size * 0.01), 4096)
    full_buf = torch.zeros(total_tensor_size + safety_margin, dtype=torch.uint8)

    # Recover each layer: build per-layer buffer from source blocks,
    # then map to full_buf using global tensor_infos offsets.
    # Build a key→(layer_buf, local_offset) lookup from layer_infos.
    key_to_local: Dict[str, Tuple[torch.Tensor, int]] = {}

    for layer_name in layer_names:
        layer_dir = checkpoint_dir / layer_name
        layer_main_path = layer_dir / f"frcheck_layer_main_rank{rank}.pt"
        if not layer_main_path.is_file():
            logger.warning(
                f"FRCheck SW recovery: missing layer_main for {layer_name}, skipping"
            )
            continue

        layer_meta = torch.load(layer_main_path, map_location="cpu", weights_only=False)
        layer_block_size = layer_meta.get("block_size", 64 * 1024 * 1024)
        layer_infos = layer_meta.get("tensor_infos", [])
        layer_tensor_size = layer_meta.get("actual_tensor_size", 0)

        source_stripes = _find_source_stripes(poa_rows, my_node, n)
        layer_buf = torch.zeros(num_source * layer_block_size, dtype=torch.uint8)

        for stripe_id, blk_idx in source_stripes:
            block_path = (
                layer_dir / f"stripe_{stripe_id}" / f"frcheck_shard_rank{rank}.pt"
            )
            blk = _read_frbk_block(str(block_path))
            if blk is None:
                logger.warning(
                    f"FRCheck SW recovery: missing block {block_path}, skipping"
                )
                continue
            src_offset = blk_idx * layer_block_size
            ncopy = min(blk.numel(), max(0, layer_tensor_size - src_offset))
            if ncopy > 0:
                layer_buf[src_offset:src_offset + ncopy].copy_(blk[:ncopy])

        # Build lookup: tensor key -> (layer_buf, local_offset)
        for info in layer_infos:
            key = getattr(info, "key", "")
            local_offset = getattr(info, "offset", 0)
            key_to_local[key] = (layer_buf, local_offset)

    # Map global tensor_infos → full_buf using key→local lookup
    for global_info in global_tensor_infos:
        key = getattr(global_info, "key", "")
        if key in key_to_local:
            layer_buf, local_offset = key_to_local[key]
            global_offset = getattr(global_info, "offset", 0)
            size = getattr(global_info, "size_bytes", 0)
            if size > 0 and global_offset + size <= full_buf.numel():
                full_buf[global_offset:global_offset + size].copy_(
                    layer_buf[local_offset:local_offset + size]
                )

    # Reconstruct state dict from recovered full buffer
    main_payload["tensor_buffer"] = full_buf[:total_tensor_size].clone()
    result = _reconstruct_from_main_payload(main_payload, flat_key_roots)

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return result


def load_frcheck_legacy_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    """Load FRCheck legacy checkpoint.

    Supports three modes:
    - Software failure (--use-frcheck-software-failure): failed ranks read
      SOURCE stripe blocks from local disk and reassemble.
    - Hardware failure (--use-frcheck-hardware-failure): failed ranks recover
      via RDMA RS decode from surviving ranks.
    """
    from megatron.training import get_args
    args = get_args()

    sw_failure = getattr(args, "use_frcheck_software_failure", False)
    hw_failure = getattr(args, "use_frcheck_hardware_failure", False)
    failed_ranks_str = getattr(args, "frcheck_failed_ranks", None)
    failed_ranks_parsed = getattr(args, "frcheck_failed_ranks_parsed", None)
    logger.info(
        "FRCheck load: sw_failure=%s hw_failure=%s failed_ranks_raw=%r failed_ranks_parsed=%r",
        sw_failure, hw_failure, failed_ranks_str, failed_ranks_parsed,
    )

    # Resolve failed ranks list
    failed_ranks = failed_ranks_parsed
    if failed_ranks is None and failed_ranks_str:
        failed_ranks = [int(x.strip()) for x in failed_ranks_str.split(",")]

    # Check for software failure mode
    if sw_failure:
        if not failed_ranks:
            logger.warning(
                "FRCheck: --use-frcheck-software-failure set but no "
                "--frcheck-failed-ranks provided"
            )
            failed_ranks = []
        logger.info("FRCheck: software failure mode — failed ranks %s", failed_ranks)
        _t_fr: Dict[str, float] = {}
        _t0 = time.time()
        result = _recover_frcheck_legacy_software(checkpoint_name, failed_ranks)
        _t_fr['network_encode'] = time.time() - _t0
        _t_fr['rebuild_sd'] = 0.0
        _t_fr['total'] = _t_fr['network_encode']
        logger.info(
            "FRCheck legacy load timing (SW): "
            "total=%(total).2fs network_encode=%(network_encode).2fs "
            "rebuild_sd=%(rebuild_sd).2fs", _t_fr
        )
        return result

    # Check for hardware recovery mode
    if hw_failure:
        if failed_ranks:
            logger.info("FRCheck: hardware recovery mode — failed ranks %s", failed_ranks)
            result, _t_fr = recover_frcheck_legacy_hardware(checkpoint_name, failed_ranks)
            _t_fr['total'] = _t_fr['network_encode'] + _t_fr['rebuild_sd']
            logger.info(
                "FRCheck legacy load timing (HW): "
                "total=%(total).2fs prep=%(prep).2fs main_io=%(main_io).2fs "
                "disk_io=%(disk_io).2fs network_encode=%(network_encode).2fs "
                "rebuild_sd=%(rebuild_sd).2fs rdma_xfer=%(rdma_xfer).2fs "
                "decode=%(decode).2fs (prep excluded from total)",
                _t_fr,
            )
            return result

    raise NotImplementedError(
        "FRCheck legacy load (non-recovery) is not implemented; use a checkpoint saved "
        "with another format or extend load_frcheck_legacy_checkpoint."
    )
