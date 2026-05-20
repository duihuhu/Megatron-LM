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
    # If checkpoint_name has a file extension, use its parent directory
    base = checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent
    # FRCheck files may be in a "frcheck/" subdirectory (created during save).
    # Try the subdirectory first, fall back to base.
    frcheck_sub = base / "frcheck"
    if frcheck_sub.is_dir():
        return frcheck_sub
    return base


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
    output_dir: str,
    layer_name: str,
    rank: int,
) -> None:
    """Run async stripe encode pipeline. C++ poller writes per-stripe files."""
    stripe_plans = manager.stripe_plans
    num_source_stripes = (n - 1) * (n - 2)

    required_data = block_size * num_source_stripes
    if layer_tensor_size > required_data:
        logger.warning(
            "FRCheck: layer tensor size %d > encoding capacity %d; truncating",
            layer_tensor_size, required_data,
        )

    # Phase 1: Pre-copy all SOURCE stripe data to per-stripe buffers
    roles = [0] * num_stripes
    data_addrs = [0] * num_stripes
    actual_sizes = [0] * num_stripes

    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}

    for stripe_id in range(num_stripes):
        plan = stripe_plans[stripe_id]
        roles[stripe_id] = int(plan.role)

        if plan.role == StripeRole.SOURCE:
            blk_idx = src_block_per_node[my_node]
            src_block_per_node[my_node] += 1
            src_offset = blk_idx * block_size
            ncopy = min(block_size, max(0, layer_tensor_size - src_offset))
            actual_sizes[stripe_id] = ncopy
            buf = manager.stripe_data_bufs[stripe_id]
            if buf is not None:
                if ncopy > 0:
                    buf[:ncopy].copy_(tensor_buffer[src_offset : src_offset + ncopy])
                if ncopy < block_size:
                    buf[ncopy:].zero_()
                data_addrs[stripe_id] = buf.data_ptr()

    # Phase 2a: Post all recv WRs (before barrier, before any sends)
    recv_list = [(b.data_ptr() if b is not None else 0) for b in manager.recv_bufs]
    p2_list = [(b.data_ptr() if b is not None else 0) for b in manager.parity2_bufs]
    native.submit_stripes_post_recvs(
        num_stripes=num_stripes,
        roles=roles,
        block_size=block_size,
        recv_bufs=recv_list,
        parity2_addrs=p2_list,
    )

    if world_size > 1:
        torch.distributed.barrier()

    # Phase 2b: Post all send WRs + start poller (after barrier, all recvs ready)
    p1_list = [(b.data_ptr() if b is not None else 0) for b in manager.parity1_bufs]
    native.submit_stripes_post_sends(
        data_addrs=data_addrs,
        actual_sizes=actual_sizes,
        block_size=block_size,
        recv_bufs=recv_list,
        parity1_addrs=p1_list,
        parity2_addrs=p2_list,
        g_tbls=native._get_g_tbls_ptr(),
        output_dir=output_dir,
        layer_name=layer_name,
        rank=rank,
    )

    # Phase 3: Wait for all stripes to complete
    native.wait_stripes_async()

    # Phase 4: Write SOURCE files (GPU buffers, threaded for GPU→CPU + disk I/O)
    import struct
    import concurrent.futures
    _FRBK_MAGIC = b"FRBK"

    def _write_one_source(stripe_id, ncopy, buf):
        if buf is None:
            return
        data = buf[:ncopy]
        if gdr:
            data = data.cpu()  # GIL released during CUDA op
        stripe_dir = Path(output_dir) / layer_name / f"stripe_{stripe_id}"
        stripe_dir.mkdir(parents=True, exist_ok=True)
        path = stripe_dir / f"frcheck_shard_rank{rank}.pt"
        hdr = struct.pack("<4sIIQQ", _FRBK_MAGIC, stripe_id, 0, ncopy, block_size)
        with open(path, "wb") as f:
            f.write(hdr)
            if ncopy > 0:
                f.write(memoryview(data.numpy()))

    source_jobs = [(sid, actual_sizes[sid], manager.stripe_data_bufs[sid])
                   for sid in range(num_stripes)
                   if stripe_plans[sid].role == StripeRole.SOURCE]
    if source_jobs:
        n_workers = min(len(source_jobs), 6)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_write_one_source, sid, ncopy, buf)
                       for sid, ncopy, buf in source_jobs]
            for f in futures:
                f.result()  # propagate exceptions


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
    flatten_optimizer_fp32_params(state_dict)
    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    logger.info(
        "FRCheck save: rank=%d total_tensor_size=%d n_tensors=%d",
        rank, total_tensor_size, len(decomposed.tensor_data),
    )

    logger.info(f"FRCHECK save timing: decompose+group {time.time()-t0:.3f}s")

    # 1.5 Build contiguous tensor_buffer for main file (same as other EC strategies).
    #     Per-layer grouping overwrites info.offset → snapshot offsets first.
    t0 = time.time()
    safety_margin = max(int(total_tensor_size * 0.01), 4096)
    full_buf = allocate_hugepage_tensor(
        total_tensor_size + safety_margin, fallback_pin_memory=torch.cuda.is_available(),
    )
    full_buf.zero_()
    for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
        view = _to_device_view(tensor, torch.device("cpu"))
        full_buf[info.offset : info.offset + info.size_bytes].copy_(view)
    # save global offsets (decompose set them) before grouping clobbers them
    _global_offsets = {id(info): info.offset for info in decomposed.tensor_infos}
    logger.info(f"FRCHECK save timing: full tensor buf build {time.time()-t0:.3f}s")

    # 2. Group tensors by layer index
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

    logger.info(f"FRCHECK save timing: layer group+manager init {time.time()-t0:.3f}s")

    # 4.5 Compute per-layer adaptive block sizes (first save, cached thereafter)
    manager.compute_layer_block_sizes(layer_groups)

    # 5. Encode each layer independently
    t0 = time.time()
    for group in layer_groups:
        layer_name = f"layer_{group.layer_idx}" if group.layer_idx >= 0 else "layer_common"
        layer_block_size = manager._layer_block_sizes[group.layer_idx]

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
            "FRCheck save: %s tensor_buffer %s, size=%d layer_blk=%d",
            layer_name, "GPU" if gdr else "CPU", layer_buf_size, layer_block_size,
        )

        # Pre-create stripe directories so C++ poller's mkdir (single-level) succeeds
        layer_dir = checkpoint_dir / layer_name
        for sid in range(num_stripes):
            (layer_dir / f"stripe_{sid}").mkdir(parents=True, exist_ok=True)

        # Stripe encode (RDMA + per-stripe file writes by C++ poller)
        _encode_one_layer(
            manager, native, tensor_buffer, group.total_bytes,
            n, num_stripes, layer_block_size, my_node, gdr, world_size,
            str(checkpoint_dir), layer_name, rank,
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
                "block_size": layer_block_size,
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

    # 6. Write top-level main file with full tensor_buffer + non_tensor_data,
    #    matching the pattern of eccheck / eclatin / ecnaive.
    t0 = time.time()
    main_file = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    from megatron.training.legacy_io_utils import write_raw_checkpoint, MAGIC_FRCHECK
    # restore global offsets (per-layer encode clobbered them with local offsets)
    for info in decomposed.tensor_infos:
        info.offset = _global_offsets[id(info)]
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
        layer_names=[f"layer_{g.layer_idx}" if g.layer_idx >= 0 else "layer_common"
                     for g in layer_groups],
    )

    logger.info(f"FRCHECK save timing: main file write {time.time()-t0:.3f}s")
    logger.info(
        "FRCheck save: done rank=%d node=%d gdr=%s layers=%d file=%s",
        rank, my_node, gdr, num_layers, main_file,
    )
    logger.info(f"FRCHECK legacy save: done in {time.time() - start_time:.2f}s")

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


def _read_my_stripe_block(
    checkpoint_dir: Path,
    layer_name: str,
    stripe_id: int,
    my_global_rank: int,
    my_original_role: int,
) -> Optional[torch.Tensor]:
    """Read the stripe block for my rank from disk based on my original role in this stripe.

    SOURCE(0):     stripe_{sid}/frcheck_shard_rank{R}.pt
    ENCODER(1):    stripe_{sid}/frcheck_shard_rank{R}_p1.pt  (we need parity1)
    PARITY(2):     stripe_{sid}/frcheck_shard_rank{R}.pt     (parity2 is saved as this)
    """
    stripe_dir = checkpoint_dir / layer_name / f"stripe_{stripe_id}"
    if my_original_role == 1:  # ENCODER → read parity1
        filepath = stripe_dir / f"frcheck_shard_rank{my_global_rank}_p1.pt"
    else:
        filepath = stripe_dir / f"frcheck_shard_rank{my_global_rank}.pt"
    return _read_frbk_block(str(filepath))


# ---------------------------------------------------------------------------
# Per-layer recovery pipeline
# ---------------------------------------------------------------------------

def _recover_one_layer(
    manager,
    native,
    checkpoint_dir: Path,
    layer_name: str,
    layer_block_size: int,
    layer_total_bytes: int,
    n: int,
    gdr: bool,
    world_size: int,
    rank: int,
) -> Optional[torch.Tensor]:
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
    group_members = manager.group_member_ranks

    if not manager.recovery_stripe_plans:
        return None

    # Bin stripes by my role
    decoder_stripes = [p for p in manager.recovery_stripe_plans if my_node == p['decoder_node']]
    helper_stripes = [p for p in manager.recovery_stripe_plans if my_node in p['helper_nodes']]
    failed_stripes = [p for p in manager.recovery_stripe_plans if my_node == p['failed_node']]
    is_failed = len(failed_stripes) > 0

    logger.info(
        "FRCheck recovery layer %s: rank %d — %d decoder, %d helper, %d failed stripes",
        layer_name, rank, len(decoder_stripes), len(helper_stripes), len(failed_stripes),
    )

    # Phase 1: Survivors read ALL their blocks from disk (single batch)
    # Register every block for RDMA — send_to_peer requires registered buffers
    # (the 128MB temp buffer fallback is insufficient for layer_common's 132MB blocks)
    my_blocks: Dict[int, torch.Tensor] = {}
    for plan in decoder_stripes + helper_stripes:
        sid = plan['stripe_id']
        blk = _read_my_stripe_block(checkpoint_dir, layer_name, sid, rank, plan['original_role'])
        if blk is None or blk.numel() == 0:
            blk = torch.zeros(layer_block_size, dtype=torch.uint8)
        elif blk.numel() < layer_block_size:
            padded = torch.zeros(layer_block_size, dtype=torch.uint8)
            padded[:blk.numel()] = blk
            blk = padded
        blk = blk.contiguous()
        native.register_buffer(blk.data_ptr(), blk.numel())
        my_blocks[sid] = blk

    # Allocate layer buffer for failed rank
    layer_buf = None
    if is_failed:
        num_source_stripes = (n - 1) * (n - 2)
        from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
            allocate_hugepage_tensor,
        )
        layer_buf = allocate_hugepage_tensor(
            num_source_stripes * layer_block_size, fallback_pin_memory=True)
        layer_buf.zero_()

    # Node ID → rig mapping
    node_to_rig = {i + 1: i for i in range(n)}

    import threading
    import time as _time

    # Tracks SOURCE block indices (matches save order: increment only for SOURCE stripes)
    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}

    # Phase 2: Batch all stripes — post recvs (threaded), sends, decode, deliver
    # For each stripe, the decoder starts one recv thread per helper.
    # All recv threads block on TCP (GIL released), helpers send, threads unblock.
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

        # --- Decoder: start recv threads for each helper ---
        recv_threads = []
        recv_bufs = []
        if my_node == decoder_node:
            for helper_node in helper_nodes:
                helper_rig = node_to_rig[helper_node]
                from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
                    allocate_hugepage_tensor,
                )
                recv_buf = allocate_hugepage_tensor(layer_block_size, fallback_pin_memory=True)
                recv_bufs.append(recv_buf)
                native.register_buffer(recv_buf.data_ptr(), recv_buf.numel())

                def _recv_thread(rb=recv_buf, hrig=helper_rig):
                    native.recv_from_peer(hrig, rb.data_ptr(), rb.numel())

                t = threading.Thread(target=_recv_thread, daemon=True)
                t.start()
                recv_threads.append(t)

        # Brief yield so decoder recv threads post their RDMA WRs / TCP reads
        _time.sleep(0.01)

        # --- Helpers: send blocks to decoder ---
        if my_node in helper_nodes:
            my_block = my_blocks.get(sid)
            if my_block is None:
                my_block = torch.zeros(layer_block_size, dtype=torch.uint8)
                native.register_buffer(my_block.data_ptr(), my_block.numel())
            native.send_to_peer(decoder_rig, my_block.data_ptr(), my_block.numel())

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

            recovered = torch.zeros(layer_block_size, dtype=torch.uint8)
            native.register_buffer(recovered.data_ptr(), recovered.numel())
            native.submit_stripe_decode(
                k=k,
                survivor_positions=survivor_positions,
                lost_position=failed_pos,
                survivor_addrs=survivor_addrs,
                recovered_addr=recovered.data_ptr(),
                block_size=layer_block_size,
            )

            native.send_to_peer(failed_rig, recovered.data_ptr(), recovered.numel())

        # --- Failed rank: recv decoded block ---
        if my_node == failed_node:
            from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
                allocate_hugepage_tensor,
            )
            recv_buf = allocate_hugepage_tensor(layer_block_size, fallback_pin_memory=True)
            native.register_buffer(recv_buf.data_ptr(), recv_buf.numel())
            native.recv_from_peer(decoder_rig, recv_buf.data_ptr(), recv_buf.numel())

            # Only store SOURCE blocks; encoder/parity_target blocks are parity data
            if layer_buf is not None and original_role == int(StripeRole.SOURCE):
                blk_idx = src_block_per_node[failed_node]
                src_block_per_node[failed_node] += 1
                offset = blk_idx * layer_block_size
                ncopy = min(layer_block_size, max(0, layer_total_bytes - offset))
                if ncopy > 0:
                    layer_buf[offset:offset + ncopy].copy_(recv_buf[:ncopy])

            del recv_buf

        # Cleanup decoder recv bufs
        if my_node == decoder_node:
            for buf in recv_bufs:
                del buf

    if is_failed:
        n_stored = src_block_per_node.get(my_node, 0)
        logger.info(
            "FRCheck recovery: %s — stored %d SOURCE blocks (expected %d)",
            layer_name, n_stored, (n - 1) * (n - 2),
        )

    return layer_buf


# ---------------------------------------------------------------------------
# Main recovery entry point
# ---------------------------------------------------------------------------

def recover_frcheck_legacy_hardware(
    checkpoint_name: str,
    failed_global_ranks: List[int],
) -> Dict[str, Any]:
    """Main entry point for FRCheck hardware recovery.

    Steps:
    1. Initialize manager with recovery mode
    2. Read the main checkpoint payload (all_gather if local missing)
    3. Layer-by-layer recovery via _recover_one_layer()
    4. Reconstruct state_dict from recovered tensor buffer
    """
    start_time = time.time()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # 1. Init manager / RDMA
    manager = FRCheckManager()
    manager.init_frcheck_if_enabled()
    native = manager.get_native()
    if native is None:
        raise RuntimeError("FRCheck hardware recovery: native module not available")

    n = native.n()
    num_stripes = native.num_stripes()
    rg = manager.rank_in_group
    gdr = manager.gdr_available

    logger.info(
        "FRCheck hardware recovery: rank=%d group=%d rig=%d n=%d gdr=%s failed=%s",
        rank, manager.group_id, rg, n, gdr, failed_global_ranks,
    )

    # 2. Init recovery plans
    recovery_contexts = manager.init_frcheck_hardware_recovery(failed_global_ranks)
    is_failed = rank in failed_global_ranks
    is_survivor = not is_failed

    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)

    # 3. Read main payload (metadata + tensor info)
    main_path = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    main_payload = None
    if main_path.is_file():
        from megatron.training.legacy_io_utils import (
            is_raw_format, read_raw_checkpoint, MAGIC_FRCHECK,
        )
        try:
            if is_raw_format(str(main_path), MAGIC_FRCHECK):
                main_payload = read_raw_checkpoint(str(main_path), MAGIC_FRCHECK)
            else:
                main_payload = torch.load(main_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("FRCheck recovery: failed to read main file: %s", e)

    # Exchange payloads via all_gather so failed ranks get metadata
    if world_size > 1 and torch.distributed.is_initialized():
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, main_payload)
        if main_payload is None:
            # Find a valid payload from another rank
            for g in gathered:
                if g is not None:
                    main_payload = g
                    break

    if main_payload is None:
        raise FileNotFoundError(
            f"FRCheck: no main payload available at {checkpoint_dir}"
        )

    logger.info("FRCheck recovery: main payload loaded, layers=%s",
                main_payload.get("layer_names", []))

    tensor_infos = main_payload.get("tensor_infos", [])
    total_tensor_size = int(main_payload.get("actual_tensor_size", 0))
    non_tensor_data = main_payload.get("non_tensor_data", {})
    flat_key_roots = main_payload.get("flat_key_roots", [])
    layer_names = main_payload.get("layer_names", [])
    saved_block_size = int(main_payload.get("block_size", 64 * 1024 * 1024))

    # 4. Recovery simulation: if survivor, load own layer data from files
    #    If failed, recover via _recover_one_layer() for each layer

    if is_survivor and not manager.recovery_stripe_plans:
        # Not in a group with a failed rank — load normally
        logger.info("FRCheck recovery: rank %d not involved in recovery, loading directly", rank)
        from megatron.training.legacy_io_utils import read_raw_checkpoint, MAGIC_FRCHECK
        main_payload_direct = read_raw_checkpoint(str(main_path), MAGIC_FRCHECK)
        result = _reconstruct_from_main_payload(main_payload_direct, flat_key_roots)
        return result

    # For survivors in the recovery group: read own data normally from saved files
    # But they still participate in recovery for the failed rank's stripes
    if is_survivor:
        # Load our own tensor data from the main file
        from megatron.training.legacy_io_utils import read_raw_checkpoint, MAGIC_FRCHECK
        if main_path.is_file():
            own_payload = read_raw_checkpoint(str(main_path), MAGIC_FRCHECK)
        else:
            own_payload = main_payload

        # Read layer data: for each layer, read per-layer source/encoder/parity files
        for layer_name in layer_names:
            layer_dir = checkpoint_dir / layer_name
            layer_block_size = saved_block_size

            # Read layer metadata for block size
            layer_main_path = layer_dir / f"frcheck_layer_main_rank{rank}.pt"
            if layer_main_path.is_file():
                layer_meta = torch.load(layer_main_path, map_location="cpu", weights_only=False)
                layer_block_size = layer_meta.get("block_size", saved_block_size)

            # Recover failed rank's data for this layer
            _recover_one_layer(
                manager, native, checkpoint_dir, layer_name,
                layer_block_size, 0,  # layer_total_bytes not needed for survivor
                n, gdr, world_size, rank,
            )

        # Load own state dict normally
        result = _reconstruct_from_main_payload(own_payload, flat_key_roots)
        logger.info("FRCheck recovery: survivor rank %d done in %.2fs", rank, time.time() - start_time)
        return result

    # ---- Failed rank path ----
    # Allocate full tensor buffer for recovered data
    safety_margin = max(int(total_tensor_size * 0.01), 4096)
    from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
        allocate_hugepage_tensor,
    )
    full_buf = allocate_hugepage_tensor(
        total_tensor_size + safety_margin, fallback_pin_memory=True,
    )
    full_buf.zero_()

    # Recover each layer and copy into the full buffer at correct offsets
    # We need to reconstruct the layer-to-offset mapping
    # The layer names are in order: layer_common, layer_0, layer_1, ...
    t0 = time.time()
    total_recovered = 0
    for layer_name in layer_names:
        layer_dir = checkpoint_dir / layer_name
        layer_block_size = saved_block_size

        # Read layer metadata
        layer_main_path = layer_dir / f"frcheck_layer_main_rank{rank}.pt"
        if layer_main_path.is_file():
            layer_meta = torch.load(layer_main_path, map_location="cpu", weights_only=False)
            layer_block_size = layer_meta.get("block_size", saved_block_size)
            actual_size = layer_meta.get("actual_tensor_size", 0)
            layer_infos = layer_meta.get("tensor_infos", [])
        else:
            logger.warning("FRCheck recovery: no layer metadata for %s", layer_name)
            actual_size = 0
            layer_infos = []

        if actual_size == 0:
            continue

        layer_buf = _recover_one_layer(
            manager, native, checkpoint_dir, layer_name,
            layer_block_size, actual_size,
            n, gdr, world_size, rank,
        )

        if layer_buf is not None:
            # Copy recovered data into the full buffer at correct offsets
            for info in layer_infos:
                offset = info.offset
                size = info.size_bytes
                if offset + size <= full_buf.numel():
                    full_buf[offset:offset + size].copy_(layer_buf[offset:offset + size])
            total_recovered += actual_size

    logger.info(
        "FRCheck recovery: failed rank %d — recovered %d bytes in %.2fs",
        rank, total_recovered, time.time() - t0,
    )

    # 5. Reconstruct state_dict from recovered full buffer.
    #    Replace tensor_buffer with recovered data — main_payload's tensor_buffer
    #    is either the failed rank's old file data (simulated) or a peer's (real).
    main_payload['tensor_buffer'] = full_buf[:total_tensor_size].clone()
    result = _reconstruct_from_main_payload(main_payload, flat_key_roots)
    logger.info("FRCheck hardware recovery: done rank=%d in %.2fs", rank, time.time() - start_time)
    return result


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


def load_frcheck_legacy_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    """Load FRCheck legacy checkpoint, with optional hardware recovery."""
    from megatron.training import get_args
    args = get_args()

    hw_failure = getattr(args, "use_frcheck_hardware_failure", False)
    failed_ranks_str = getattr(args, "frcheck_failed_ranks", None)
    failed_ranks_parsed = getattr(args, "frcheck_failed_ranks_parsed", None)
    logger.info(
        "FRCheck load: hw_failure=%s failed_ranks_raw=%r failed_ranks_parsed=%r",
        hw_failure, failed_ranks_str, failed_ranks_parsed,
    )

    # Check for hardware recovery mode
    if hw_failure:
        failed_ranks = failed_ranks_parsed
        if failed_ranks is None and failed_ranks_str:
            failed_ranks = [int(x.strip()) for x in failed_ranks_str.split(",")]
        if failed_ranks:
            logger.info("FRCheck: hardware recovery mode — failed ranks %s", failed_ranks)
            return recover_frcheck_legacy_hardware(checkpoint_name, failed_ranks)

    raise NotImplementedError(
        "FRCheck legacy load (non-recovery) is not implemented; use a checkpoint saved "
        "with another format or extend load_frcheck_legacy_checkpoint."
    )
