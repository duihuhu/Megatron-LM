# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""LEGACY checkpoint path for FRCheck (POA-driven stripe encode with RDMA).
Layerwise: groups tensors by transformer layer index, encodes each layer
independently so per-layer data fits within SOURCE stripe capacity.
"""

import pickle
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


def _checkpoint_dir_from_path(checkpoint_name: str) -> Path:
    checkpoint_path = Path(checkpoint_name)
    return checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent


def _to_device_view(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return a contiguous uint8 view of tensor on the given device."""
    t = tensor.detach()
    if t.device != device:
        t = t.to(device, non_blocking=True)
    return t.contiguous().view(torch.uint8).reshape(-1)


def _stripes_per_role(stripe_plans: List, role: StripeRole) -> int:
    return sum(1 for sp in stripe_plans if sp.role == role)


def _clear_accum_buffers(manager) -> None:
    """Zero parity accumulation buffers for a fresh layer."""
    if manager.parity1_accum is not None:
        manager.parity1_accum.zero_()
    if manager.parity2_accum is not None:
        manager.parity2_accum.zero_()


def _encode_one_layer(
    manager,
    native,
    tensor_buffer: torch.Tensor,
    layer_tensor_size: int,
    n: int,
    num_stripes: int,
    block_size: int,
    my_node: int,
    gdr: bool,
    world_size: int,
) -> None:
    """Run the full stripe encode pipeline for one layer's data."""
    stripe_plans = manager.stripe_plans
    num_source_stripes = (n - 1) * (n - 2)

    required_data = block_size * num_source_stripes
    if layer_tensor_size > required_data:
        logger.warning(
            "FRCheck: layer tensor size %d > encoding capacity %d; truncating",
            layer_tensor_size, required_data,
        )

    src_block_per_node: Dict[int, int] = {}
    for node in range(1, n + 1):
        src_block_per_node[node] = 0

    for stripe_id in range(num_stripes):
        plan = stripe_plans[stripe_id]

        if plan.role == StripeRole.SOURCE:
            blk_idx = src_block_per_node[my_node]
            src_block_per_node[my_node] += 1
            src_offset = blk_idx * block_size
            ncopy = min(block_size, max(0, layer_tensor_size - src_offset))
            if ncopy > 0:
                manager.data_buffer[:ncopy].copy_(
                    tensor_buffer[src_offset : src_offset + ncopy]
                )
            if ncopy < block_size:
                manager.data_buffer[ncopy:].zero_()
        else:
            manager.data_buffer.zero_()

        manager.submit_stripe(stripe_id)

    if world_size > 1:
        torch.distributed.barrier()


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
    start_time = time.time()
    t0 = start_time
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # 1. Decompose state_dict
    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    logger.info(
        "FRCheck save: rank=%d total_tensor_size=%d n_tensors=%d",
        rank, total_tensor_size, len(decomposed.tensor_data),
    )

    logger.info(f"FRCHECK save timing: decompose+group {time.time()-t0:.3f}s")

    # 2. Group tensors by layer index
    t0 = time.time()
    layer_groups = _group_by_layer(decomposed)
    del decomposed.tensor_data  # GPU refs now held by per-layer groups
    num_layers = len(layer_groups)
    logger.info(
        "FRCheck save: grouped %d layers from %d tensors (layers: %s)",
        num_layers, len(decomposed.tensor_data),
        [(g.layer_idx, g.total_bytes) for g in layer_groups],
    )

    # 3. Init FRCheck manager + RDMA + stripe plans (once, shared across layers)
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

    n_source_my = _stripes_per_role(stripe_plans, StripeRole.SOURCE)
    n_encoder_my = _stripes_per_role(stripe_plans, StripeRole.ENCODER)
    n_parity_my = _stripes_per_role(stripe_plans, StripeRole.PARITY_TARGET)

    # 4. Setup output directory
    checkpoint_dir = Path(checkpoint_name)
    if checkpoint_name.endswith(".pt") or checkpoint_name.endswith(".ckpt"):
        checkpoint_dir = checkpoint_dir.parent / "frcheck"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    flat_key_roots = decomposed.flat_key_roots
    all_tensor_infos = decomposed.tensor_infos  # full list for metadata

    logger.info(f"FRCHECK save timing: layer group+manager init {time.time()-t0:.3f}s")

    # 5. Encode each layer independently
    t0 = time.time()
    for group in layer_groups:
        layer_name = f"layer_{group.layer_idx}" if group.layer_idx >= 0 else "layer_common"

        # Allocate or reuse cached per-layer contiguous buffer
        safety_margin = max(int(group.total_bytes * 0.01), 4096)
        layer_buf_size = group.total_bytes + safety_margin
        tensor_buffer = manager.allocate_layer_buffer(group.layer_idx, layer_buf_size, gdr)

        # Copy layer tensors into layer buffer
        offset = 0
        for info, tensor in zip(group.tensor_infos, group.tensor_data):
            tb = _to_device_view(tensor, buf_device)
            nbytes = tb.numel()
            tensor_buffer[offset : offset + nbytes].copy_(tb)
            info.offset = offset
            offset += nbytes

        group.tensor_data = []  # free GPU refs for this layer
        addr = tensor_buffer.data_ptr()
        if addr not in manager._rdma_registered_addrs:
            native.register_buffer(addr, tensor_buffer.numel())
            manager._rdma_registered_addrs.add(addr)
        logger.info(
            "FRCheck save: %s tensor_buffer %s, size=%d",
            layer_name, "GPU" if gdr else "CPU", layer_buf_size,
        )

        # Clear accum buffers before encoding this layer
        _clear_accum_buffers(manager)

        # Stripe encode (RDMA only)
        _encode_one_layer(
            manager, native, tensor_buffer, group.total_bytes,
            n, num_stripes, block_size, my_node, gdr, world_size,
        )

        # Write layer shards
        _write_layer_shards(
            manager, checkpoint_dir, layer_name, rank, rg,
            n_source_my, n_encoder_my, n_parity_my,
            block_size, gdr, tensor_buffer,
        )

        # Write layer metadata
        layer_main = checkpoint_dir / layer_name / f"frcheck_layer_main_rank{rank}.pt"
        torch.save(
            {
                "version": 2,
                "format": "frcheck_torch_legacy",
                "rank": rank,
                "layer_name": layer_name,
                "n": n,
                "num_stripes": num_stripes,
                "block_size": block_size,
                "rank_in_group": rg,
                "gdr": gdr,
                "tensor_infos": group.tensor_infos,
                "actual_tensor_size": group.total_bytes,
            },
            layer_main,
        )

        # Release per-layer GPU buffer (skip if cached — kept registered for next save)
        if addr not in manager._rdma_registered_addrs:
            native.unregister_buffer(addr)

    logger.info(f"FRCHECK save timing: all layers encode+write {time.time()-t0:.3f}s")

    # 6. Write top-level main file
    t0 = time.time()
    main_file = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    torch.save(
        {
            "version": 2,
            "format": "frcheck_torch_legacy",
            "rank": rank,
            "group_id": manager.group_id,
            "rank_in_group": rg,
            "poa_path": manager.get_resolved_table_path() or native.path(),
            "n": n,
            "num_stripes": num_stripes,
            "num_layers": num_layers,
            "block_size": block_size,
            "gdr": gdr,
            "non_tensor_data": decomposed.non_tensor_data,
            "tensor_infos": all_tensor_infos,
            "actual_tensor_size": total_tensor_size,
            "flat_key_roots": list(flat_key_roots) if flat_key_roots else [],
            "state_dict_keys": list(state_dict.keys()),
            "group_member_ranks": manager.group_member_ranks,
            "node_slot_to_global_rank": manager.node_slot_to_global_rank,
            "layer_names": [f"layer_{g.layer_idx}" if g.layer_idx >= 0 else "layer_common"
                           for g in layer_groups],
        },
        main_file,
    )

    logger.info(f"FRCHECK save timing: main file write {time.time()-t0:.3f}s")
    logger.info(
        "FRCheck save: done rank=%d node=%d gdr=%s layers=%d file=%s",
        rank, my_node, gdr, num_layers, main_file,
    )
    logger.info(f"FRCHECK legacy save: done in {time.time() - start_time:.2f}s")

    if world_size > 1:
        torch.distributed.barrier()


def load_frcheck_legacy_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    raise NotImplementedError(
        "FRCheck legacy load is not implemented; use a checkpoint saved "
        "with another format or extend load_frcheck_legacy_checkpoint."
    )
