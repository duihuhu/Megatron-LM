# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""LEGACY checkpoint path for FRCheck (POA-driven stripe encode with RDMA).
Layerwise: groups tensors by transformer layer index, encodes each layer
independently so per-layer data fits within SOURCE stripe capacity.
"""

import copy
import pickle
import queue
import os
import re
import struct
import threading
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from dataclasses import dataclass

from megatron.core.dist_checkpointing.strategies.frcheck_manager import (
    FRCheckManager,
    StripePlan,
    StripeRole,
)
from megatron.core.dist_checkpointing.strategies.hugepage_alloc import (
    allocate_hugepage_tensor,
)
from megatron.core.dist_checkpointing.strategies.state_dict_decomposer import (
    decompose_state_dict,
    decompose_state_dict_for_save,
    DecomposedStateDict,
    TensorInfo,
    TensorMetadata,
    reconstruct_state_dict,
    extract_tensors_from_continuous_buffer,
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


def _timed_barrier() -> float:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0.0
    start = time.time()
    torch.distributed.barrier()
    return time.time() - start


_async_p1_writer_thread: Optional[threading.Thread] = None
_async_p2_writer_thread: Optional[threading.Thread] = None
_async_writer_error: Optional[BaseException] = None
_p2_save_generation: int = 0
_recovery_async_parity_initialized: bool = False
_recovery_async_parity_submitted: bool = False
_recovery_async_parity_thread: Optional[threading.Thread] = None
_recovery_async_parity_error: Optional[BaseException] = None

_LAYER_KEY_RE = re.compile(r"\.layers\.(\d+)\b")

# Cached metadata exchange (identical across iterations for fixed model)
_cached_all_tensor_infos = None
_cached_all_layer_order = None
_cached_all_layer_metadata = None
_cached_all_actual_tensor_sizes = None
_cached_all_optimizer_layer_maps = None


def _frcheck_debug_enabled(default: bool = False) -> bool:
    try:
        from megatron.training import get_args
        return bool(getattr(get_args(), "frcheck_debug", default))
    except Exception:
        return bool(default)


@dataclass
class _TensorOwnership:
    layer_idx: int
    kind: str


def _extract_layer_idx(key: str) -> int:
    """Extract transformer layer index from a tensor FQN, or -1 if not a layer."""
    m = _LAYER_KEY_RE.search(key)
    return int(m.group(1)) if m else -1


def _normalize_model_state_key(key: str) -> str:
    if key.startswith("model."):
        key = key[len("model."):]
    if key.startswith("module."):
        key = key[len("module."):]
    return key


def _canonical_frcheck_model_key(key: str) -> str:
    """Normalize model tensor keys to match decomposed FQN format (model.* prefix)."""
    key = str(key)
    if key.startswith("module."):
        key = key[len("module."):]
    if not key.startswith("model."):
        key = f"model.{key}"
    return key


def _canonical_live_model_key(key: str) -> str:
    key = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("model.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _optimizer_key_category(key: str) -> str:
    if not key.startswith("optimizer."):
        return "non_optimizer"
    if ".fp32_params_flat." in key:
        return "fp32_params_flat"
    if key.endswith(".exp_avg"):
        return "exp_avg"
    if key.endswith(".exp_avg_sq"):
        return "exp_avg_sq"
    if key.endswith(".fp32_param"):
        return "fp32_param"
    if key.endswith(".step"):
        return "step"
    if "param_groups" in key:
        return "param_groups"
    return "other_optimizer"


def _summarize_optimizer_keys(keys) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for key in keys:
        category = _optimizer_key_category(str(key))
        summary[category] = summary.get(category, 0) + 1
    return summary


_FRCHECK_INFO_PROFILE_EVENTS = {
    "recovery_parity_repair_prepared",
    "recovery_parity_repair_start",
    "recovery_parity_repair_done",
    "recovery_parity_join_start",
    "recovery_parity_join_done",
}

_frcheck_first_layer_recovery_start_s: Optional[float] = None
_frcheck_first_layer_recovery_target_idx: Optional[int] = None
_frcheck_first_layer_milestones: Dict[str, Any] = {}
_frcheck_first_layer_milestones_lock = threading.Lock()
_frcheck_first_layer_cuda_start_events: Dict[int, Any] = {}
_frcheck_first_layer_pending_cuda_events: Dict[str, Dict[str, Any]] = {}
_frcheck_first_layer_generation: int = 0
_frcheck_first_layer_failed_rank: bool = False


_frcheck_async_runtime_timing_reported: bool = False


def _stash_frcheck_first_layer_milestones_locked() -> None:
    try:
        from megatron.training.global_vars import stash_recovery_timing_summary
        stash_recovery_timing_summary(
            "frcheck_first_layer_milestones", dict(_frcheck_first_layer_milestones)
        )
    except Exception:
        pass


def _start_frcheck_first_layer_recovery_timer(target_layer_idx: Optional[int] = None) -> None:
    global _frcheck_first_layer_recovery_start_s, _frcheck_first_layer_recovery_target_idx
    global _frcheck_first_layer_milestones, _frcheck_first_layer_generation
    global _frcheck_first_layer_failed_rank
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    failed_rank = rank in _get_frcheck_failed_ranks()
    with _frcheck_first_layer_milestones_lock:
        _frcheck_first_layer_generation += 1
        _frcheck_first_layer_recovery_start_s = None
        _frcheck_first_layer_recovery_target_idx = target_layer_idx
        _frcheck_first_layer_failed_rank = failed_rank
        _frcheck_first_layer_cuda_start_events.clear()
        _frcheck_first_layer_pending_cuda_events.clear()
        _frcheck_first_layer_milestones = {
            "layer_idx": int(target_layer_idx) if target_layer_idx is not None else -1,
            "rank": int(rank),
            "repair_s": 0.0,
            "repair_valid": False,
            "sent_to_failed_s": 0.0,
            "sent_to_failed_valid": False,
            "delivered_s": 0.0,
            "delivered_valid": False,
            "h2d_s": 0.0,
            "h2d_valid": False,
            "forward_done_s": 0.0,
            "forward_done_valid": False,
        }
        _stash_frcheck_first_layer_milestones_locked()


def _arm_frcheck_first_layer_cuda_timer() -> None:
    global _frcheck_first_layer_recovery_start_s
    try:
        from megatron.training.global_vars import get_recovery_to_forward_timer_start
        timer_start_s = get_recovery_to_forward_timer_start()
    except Exception:
        timer_start_s = 0.0
    if timer_start_s <= 0.0:
        return
    start_event = None
    device_idx = -1
    if torch.cuda.is_available():
        try:
            device_idx = torch.cuda.current_device()
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(device_idx))
        except Exception:
            start_event = None
            device_idx = -1
    with _frcheck_first_layer_milestones_lock:
        _frcheck_first_layer_recovery_start_s = timer_start_s
        if start_event is not None:
            _frcheck_first_layer_cuda_start_events[device_idx] = start_event


def _record_frcheck_first_layer_milestone(
    name: str, layer_idx: int, elapsed_s: float
) -> None:
    with _frcheck_first_layer_milestones_lock:
        if layer_idx != _frcheck_first_layer_recovery_target_idx or elapsed_s < 0.0:
            return
        valid_key = f"{name}_valid"
        if name not in _frcheck_first_layer_milestones:
            return
        if _frcheck_first_layer_milestones.get(valid_key, False):
            elapsed_s = max(float(_frcheck_first_layer_milestones[name]), elapsed_s)
        _frcheck_first_layer_milestones[name] = float(elapsed_s)
        _frcheck_first_layer_milestones[valid_key] = True
        _stash_frcheck_first_layer_milestones_locked()


def _record_frcheck_first_layer_cuda_event(name: str, layer_idx: int) -> None:
    if layer_idx != _frcheck_first_layer_recovery_target_idx:
        return
    if not _frcheck_first_layer_failed_rank:
        return
    if not torch.cuda.is_available():
        timer_start_s = _frcheck_first_layer_recovery_start_s
        if timer_start_s is not None:
            _record_frcheck_first_layer_milestone(
                name, layer_idx, max(0.0, time.time() - timer_start_s)
            )
        return
    with _frcheck_first_layer_milestones_lock:
        if (
            layer_idx != _frcheck_first_layer_recovery_target_idx
            or _frcheck_first_layer_milestones.get(f"{name}_valid", False)
            or name in _frcheck_first_layer_pending_cuda_events
        ):
            return
        generation = _frcheck_first_layer_generation
        timer_start_s = _frcheck_first_layer_recovery_start_s
    if timer_start_s is None:
        return
    try:
        device_idx = torch.cuda.current_device()
        stream = torch.cuda.current_stream(device_idx)
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
    except Exception:
        return
    with _frcheck_first_layer_milestones_lock:
        if generation != _frcheck_first_layer_generation:
            return
        _frcheck_first_layer_pending_cuda_events.setdefault(
            name,
            {
                "event": event,
                "device": device_idx,
                "generation": generation,
                "layer_idx": layer_idx,
            },
        )


def resolve_frcheck_first_layer_cuda_events() -> None:
    if _frcheck_first_layer_recovery_start_s is None:
        return
    with _frcheck_first_layer_milestones_lock:
        pending = dict(_frcheck_first_layer_pending_cuda_events)
        start_events = dict(_frcheck_first_layer_cuda_start_events)
        generation = _frcheck_first_layer_generation
    resolved = {}
    for name, item in pending.items():
        if item.get("generation") != generation:
            continue
        device_idx = int(item["device"])
        start_event = start_events.get(device_idx)
        if start_event is None:
            continue
        try:
            item["event"].synchronize()
            elapsed_s = float(start_event.elapsed_time(item["event"])) / 1000.0
        except Exception:
            continue
        resolved[name] = (int(item["layer_idx"]), elapsed_s)
    if not resolved:
        return
    with _frcheck_first_layer_milestones_lock:
        if generation != _frcheck_first_layer_generation:
            return
        for name, (layer_idx, elapsed_s) in resolved.items():
            if layer_idx != _frcheck_first_layer_recovery_target_idx:
                continue
            valid_key = f"{name}_valid"
            if name not in _frcheck_first_layer_milestones:
                continue
            _frcheck_first_layer_milestones[name] = float(elapsed_s)
            _frcheck_first_layer_milestones[valid_key] = True
            _frcheck_first_layer_pending_cuda_events.pop(name, None)
        _stash_frcheck_first_layer_milestones_locked()


def _record_frcheck_native_batch_milestones(
    native, batch_id: int, layer_idx: int
) -> None:
    if layer_idx != _frcheck_first_layer_recovery_target_idx:
        native.get_recovery_batch_milestones(batch_id)
        return
    milestones = dict(native.get_recovery_batch_milestones(batch_id))
    try:
        from megatron.training.global_vars import get_recovery_to_forward_timer_start
        timer_start_s = get_recovery_to_forward_timer_start()
    except Exception:
        timer_start_s = 0.0
    native_now_us = float(milestones.get("now_us", 0.0) or 0.0)
    if timer_start_s <= 0.0 or native_now_us <= 0.0:
        return
    wall_now_s = time.time()
    for event_key, metric_name in (
        ("decoder_decode_done_us", "repair_s"),
        ("decoder_send_done_us", "sent_to_failed_s"),
        ("failed_delivered_us", "delivered_s"),
    ):
        event_us = float(milestones.get(event_key, 0.0) or 0.0)
        if event_us <= 0.0:
            continue
        event_wall_s = wall_now_s - max(0.0, native_now_us - event_us) / 1.0e6
        _record_frcheck_first_layer_milestone(
            metric_name, layer_idx, max(0.0, event_wall_s - timer_start_s)
        )


def _get_frcheck_failed_ranks() -> Set[int]:
    try:
        from megatron.training import get_args
        args = get_args()
    except Exception:
        return set()
    configured = getattr(args, "frcheck_failed_ranks_parsed", None)
    if configured is None:
        configured = getattr(args, "frcheck_failed_ranks", None)
    if configured is None:
        return set()
    if isinstance(configured, str):
        configured = [value.strip() for value in configured.split(",") if value.strip()]
    try:
        return {int(value) for value in configured}
    except (TypeError, ValueError):
        return set()


def record_frcheck_first_layer_forward_done(layer_idx: int) -> None:
    _record_frcheck_first_layer_cuda_event("forward_done_s", layer_idx)

def _frcheck_recovery_profile(role: str, event: str, **fields) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    parts = [f"rank={rank}", f"role={role}", f"event={event}"]
    for key, value in fields.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    if event.endswith("_error"):
        log_fn = logger.warning
    elif event in _FRCHECK_INFO_PROFILE_EVENTS:
        log_fn = logger.info
    else:
        log_fn = logger.debug
    log_fn("FRCheck profile: %s", " ".join(parts))


def _frcheck_recovery_role(
    is_failed: bool, is_decoder: bool, is_helper: bool, involved: bool
) -> str:
    roles: List[str] = []
    if is_failed:
        roles.append("failed")
    if is_decoder:
        roles.append("decoder")
    if is_helper:
        roles.append("helper")
    if not roles:
        roles.append("survivor" if involved else "survivor_uninvolved")
    return "+".join(roles)


def _classify_frcheck_tensor(
    key: str,
    optimizer_layer_map: Optional[Dict[str, int]] = None,
) -> _TensorOwnership:
    layer_idx = _extract_layer_idx(key)
    if key.startswith("model.") or key == "model":
        return _TensorOwnership(
            layer_idx=layer_idx,
            kind="model_layer" if layer_idx >= 0 else "model_common",
        )
    if key.startswith("optimizer."):
        mapped_layer = -1
        if optimizer_layer_map:
            mapped_layer = int(optimizer_layer_map.get(key, -1))
        if mapped_layer < 0:
            mapped_layer = layer_idx
        return _TensorOwnership(
            layer_idx=mapped_layer,
            kind="optimizer_layer" if mapped_layer >= 0 else "optimizer_common",
        )
    return _TensorOwnership(layer_idx=-1, kind="common_state")


_PARAM_GROUP_ID_KEYS = ('wd_mult', 'lr_mult', 'is_expert_parallel', 'is_decoupled_lr')


def _model_tensor_param_group_key(
    key: str,
    tensor: torch.Tensor,
    default_skip_embedding_weight_decay: bool = False,
) -> Tuple[float, float, bool, bool]:
    """Approximate Megatron param-group bucket for a model tensor key."""
    no_wd = (
        key.endswith(".bias")
        or len(tensor.shape) == 1
        or (default_skip_embedding_weight_decay and "embedding" in key)
    )
    return (0.0 if no_wd else 1.0, 1.0, False, False)


def _resolve_optimizer_distribute_target(
    model_key: Optional[str],
    common_model_targets: Dict[str, "_LayerGroup"],
    others_by_layer: Dict[int, "_LayerGroup"],
) -> Optional["_LayerGroup"]:
    """Route optimizer tensors to the layer group that owns their model param."""
    if not model_key:
        return None
    canonical_key = _canonical_frcheck_model_key(model_key)
    target = common_model_targets.get(canonical_key)
    if target is not None:
        return target
    layer_idx = _extract_layer_idx(canonical_key)
    if layer_idx >= 0:
        return others_by_layer.get(layer_idx)
    return None


def _optimizer_keys_for_index(
    group_idx: int, param_idx: int, state_index: int,
) -> List[str]:
    """Build decomposed optimizer keys using global Adam state indices."""
    state_prefix = f"optimizer.optimizer.state.{state_index}"
    return [
        f"optimizer.fp32_params_flat._fp32_group{group_idx}_param{param_idx}",
        f"{state_prefix}.exp_avg",
        f"{state_prefix}.exp_avg_sq",
        f"{state_prefix}.fp32_param",
    ]


def _get_fp32_from_fp16_groups(optim_sd: Dict[str, Any]) -> Optional[List[List[torch.Tensor]]]:
    """Return fp32 master-weight groups aligned with optimizer param_groups."""
    fp32 = optim_sd.get("fp32_from_fp16_params")
    if fp32 is not None:
        return fp32

    flat = optim_sd.get("fp32_params_flat")
    structure = optim_sd.get("_fp32_structure")
    if not isinstance(flat, dict) or structure is None:
        return None

    groups: List[List[torch.Tensor]] = []
    for gi, group_size in enumerate(structure):
        group: List[torch.Tensor] = []
        for pi in range(int(group_size)):
            tensor = flat.get(f"_fp32_group{gi}_param{pi}")
            if tensor is None:
                return None
            group.append(tensor)
        groups.append(group)
    return groups


def _ordered_model_keys_for_fp32_params(
    model_param_entries: List[Tuple[str, torch.Tensor, int]],
    flat_fp32: List[Tuple[int, int, torch.Tensor]],
) -> Optional[List[Tuple[str, int]]]:
    """Align model keys to fp32 master weights via shape sequence in state_dict order."""
    pending_shapes = [tuple(tensor.shape) for _, _, tensor in flat_fp32]
    ordered: List[Tuple[str, int]] = []
    for key, tensor, layer_idx in model_param_entries:
        if not pending_shapes:
            break
        if tuple(tensor.shape) == pending_shapes[0]:
            ordered.append((key, layer_idx))
            pending_shapes.pop(0)
    if len(ordered) == len(flat_fp32):
        return ordered
    return None


def _iter_optimizer_model_param_entries(
    state_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Tuple[str, int, List[str]]]]:
    """Map optimizer tensors to model tensor keys via fp32 master weights."""
    model_sd = state_dict.get("model")
    optim_sd = state_dict.get("optimizer")
    if not isinstance(model_sd, dict) or not isinstance(optim_sd, dict):
        return {}, []

    model_param_entries: List[Tuple[str, torch.Tensor, int]] = []
    for key, tensor in model_sd.items():
        if torch.is_tensor(tensor):
            model_param_entries.append((str(key), tensor, _extract_layer_idx(str(key))))
    if not model_param_entries:
        return optim_sd, []

    torch_optim_sd = optim_sd.get("optimizer", optim_sd)
    if not isinstance(torch_optim_sd, dict):
        return optim_sd, []
    param_groups = torch_optim_sd.get("param_groups", [])
    fp32_groups = _get_fp32_from_fp16_groups(optim_sd)
    if not param_groups:
        return optim_sd, []
    if fp32_groups is None:
        logger.warning(
            "FRCheck optimizer map: missing fp32_from_fp16_params; "
            "optimizer-common distribution will be incomplete",
        )
        return optim_sd, []
    if len(fp32_groups) != len(param_groups):
        logger.warning(
            "FRCheck optimizer map: fp32 group count (%d) != param_groups (%d)",
            len(fp32_groups), len(param_groups),
        )
        return optim_sd, []

    flat_fp32: List[Tuple[int, int, torch.Tensor]] = []
    for group_idx, fp32_group in enumerate(fp32_groups):
        for param_idx, fp32_tensor in enumerate(fp32_group):
            flat_fp32.append((group_idx, param_idx, fp32_tensor))

    ordered_model_keys = _ordered_model_keys_for_fp32_params(
        model_param_entries, flat_fp32,
    )

    model_buckets: Dict[Tuple[float, float, bool, bool], List[Tuple[str, int, torch.Tensor]]] = {}
    for key, tensor, layer_idx in model_param_entries:
        bucket_key = _model_tensor_param_group_key(key, tensor)
        model_buckets.setdefault(bucket_key, []).append((key, layer_idx, tensor))

    bucket_cursor: Dict[Tuple[float, float, bool, bool], int] = {
        key: 0 for key in model_buckets
    }
    entries: List[Tuple[str, int, List[str]]] = []
    state_index = 0
    ordered_cursor = 0

    for group_idx, group in enumerate(param_groups):
        if not isinstance(group, dict):
            continue
        pg_key = tuple(
            group.get(key, False if key.startswith("is_") else 1.0)
            for key in _PARAM_GROUP_ID_KEYS
        )
        fp32_group = fp32_groups[group_idx]
        if len(fp32_group) != len(group.get("params", [])):
            logger.warning(
                "FRCheck optimizer map: group %d fp32 params (%d) != param_groups params (%d)",
                group_idx, len(fp32_group), len(group.get("params", [])),
            )

        for param_idx, fp32_tensor in enumerate(fp32_group):
            model_key: Optional[str] = None
            layer_idx = -1
            target_shape = tuple(fp32_tensor.shape)

            if ordered_model_keys is not None and ordered_cursor < len(ordered_model_keys):
                cand_key, cand_layer = ordered_model_keys[ordered_cursor]
                model_key = cand_key
                layer_idx = cand_layer
                ordered_cursor += 1

            if model_key is None:
                bucket = model_buckets.get(pg_key, [])
                cursor = bucket_cursor.get(pg_key, 0)
                for mi in range(cursor, len(bucket)):
                    cand_key, cand_layer, cand_tensor = bucket[mi]
                    if tuple(cand_tensor.shape) == target_shape:
                        model_key = cand_key
                        layer_idx = cand_layer
                        bucket_cursor[pg_key] = mi + 1
                        break

            if model_key is None:
                for bucket_key, candidates in model_buckets.items():
                    cur = bucket_cursor.get(bucket_key, 0)
                    for mi in range(cur, len(candidates)):
                        cand_key, cand_layer, cand_tensor = candidates[mi]
                        if tuple(cand_tensor.shape) == target_shape:
                            model_key = cand_key
                            layer_idx = cand_layer
                            bucket_cursor[bucket_key] = mi + 1
                            break
                    if model_key is not None:
                        break

            if model_key is not None:
                entries.append((
                    model_key,
                    layer_idx,
                    _optimizer_keys_for_index(group_idx, param_idx, state_index),
                ))
            state_index += 1

    return optim_sd, entries


def _build_optimizer_layer_map(state_dict: Dict[str, Any]) -> Dict[str, int]:
    """Map optimizer tensor keys to transformer layer indexes when possible."""
    _optim_sd, entries = _iter_optimizer_model_param_entries(state_dict)
    result: Dict[str, int] = {}
    for _model_key, layer_idx, optimizer_keys in entries:
        if layer_idx < 0:
            continue
        for opt_key in optimizer_keys:
            result[opt_key] = layer_idx
    return result


def _build_optimizer_model_key_map(state_dict: Dict[str, Any]) -> Dict[str, str]:
    """Map optimizer tensor keys back to their source model tensor key."""
    _optim_sd, entries = _iter_optimizer_model_param_entries(state_dict)
    result: Dict[str, str] = {}
    for model_key, _layer_idx, optimizer_keys in entries:
        canonical_model_key = _canonical_frcheck_model_key(model_key)
        for opt_key in optimizer_keys:
            result[opt_key] = canonical_model_key
    return result


@dataclass(frozen=False)
class _LayerGroup:
    layer_idx: int
    tensor_infos: List
    tensor_data: List
    total_bytes: int = 0
    distributed_common_keys: Optional[set] = None


@dataclass
class _RecoveryBufPool:
    """Reusable RDMA buffers sized to max per-layer block_size for this rank."""
    decoder_recv_bufs: List[torch.Tensor]
    decoder_recv_buf_slots: List[List[torch.Tensor]]
    failed_recv_buf: Optional[torch.Tensor]
    failed_recv_bufs: List[torch.Tensor]
    decoder_recovered_buf: Optional[torch.Tensor]
    decoder_recovered_bufs: List[torch.Tensor]
    decoder_recovered_buf2: Optional[torch.Tensor]
    decoder_recovered_buf2s: List[torch.Tensor]
    failed_layer_buf: Optional[torch.Tensor]
    max_block_size: int
    concurrency: int
    stable_failed_layer_bufs: Optional[Dict[int, torch.Tensor]] = None


@dataclass
class _LayerEncodeResult:
    layer_name: str
    layer_idx: int
    block_size: int
    actual_sizes: List[int]
    tensor_infos: List
    total_bytes: int
    n_filled_blocks: int = 0


@dataclass
class _LayerEncodeSubmitState:
    """Per-layer encode state after Phase1 pack; used for batch network submit."""
    layer_name: str
    layer_idx: int
    block_size: int
    layer_tensor_size: int
    actual_sizes: List[int]
    data_addrs: List[int]
    mirror_addrs: List[int]
    layer_buf_base: int
    n_filled_blocks: int = 0


@dataclass
class _FRCheckLayerReadyRecord:
    """Per-layer recovery readiness exported to the forward path."""
    layer_name: str
    layer_idx: int
    encode_iter: int
    tensor_keys: List[str]
    model_tensor_keys: List[str]
    optimizer_tensor_keys: List[str]
    nbytes: int
    contains_optimizer_state: bool = False
    materialize_s: float = 0.0
    ready: bool = True
    model_ready: bool = True
    optimizer_ready: bool = True
    error: str = ""
    tensors: Optional[Dict[str, torch.Tensor]] = None
    model_tensors: Optional[Dict[str, torch.Tensor]] = None
    optimizer_tensors: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class _FRCheckLayerRecoveryJob:
    encode_iter: int
    layer_name: str
    layer_idx: int
    layer_block_size: int
    actual_size: int
    layer_infos: List


@dataclass
class _FRCheckRecoveryWindow:
    job: _FRCheckLayerRecoveryJob
    plans: List[Dict[str, Any]]
    active_by_stripe: Dict[int, bool]
    wave_idx: int
    layer_buf: Optional[torch.Tensor]
    src_block_start: int
    slot_start: int
    source_block_indices: Dict[Tuple[int, int], int]
    decoder_stripes: int
    helper_stripes: int
    failed_stripes: int
    skipped_padding_stripes: int


@dataclass
class _FRCheckRecoveredWindow:
    window: _FRCheckRecoveryWindow
    timing: Dict[str, float]


def _split_recovered_tensors(
    tensors: Optional[Dict[str, torch.Tensor]],
    model_keys: List[str],
    optimizer_keys: List[str],
    optimizer_layer_map: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    if not tensors:
        return None, None
    model_key_set = set(model_keys)
    opt_key_set = set(optimizer_keys)
    model_tensors: Dict[str, torch.Tensor] = {}
    optimizer_tensors: Dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if key in model_key_set:
            model_tensors[key] = tensor
        elif key in opt_key_set:
            optimizer_tensors[key] = tensor
        else:
            ownership = _classify_frcheck_tensor(key, optimizer_layer_map)
            if ownership.kind == "model_layer":
                model_tensors[key] = tensor
            elif ownership.kind == "optimizer_layer":
                optimizer_tensors[key] = tensor
    return (model_tensors or None), (optimizer_tensors or None)


def _sort_recovery_jobs_by_forward_priority(
    jobs: List[_FRCheckLayerRecoveryJob],
) -> List[_FRCheckLayerRecoveryJob]:
    return sorted(
        jobs,
        key=lambda job: (
            0 if job.layer_idx < 0 else 1,
            job.layer_idx,
            job.encode_iter,
        ),
    )


class _FRCheckLayerwiseRuntime:
    """FRCheck layer-ready registry consumed by TransformerBlock."""

    def __init__(self, records: List[_FRCheckLayerReadyRecord]) -> None:
        self._records_by_layer: Dict[int, _FRCheckLayerReadyRecord] = {
            r.layer_idx: r for r in records if r.layer_idx >= 0
        }
        self._records_by_name: Dict[str, _FRCheckLayerReadyRecord] = {
            r.layer_name: r for r in records
        }
        self._model_keys_by_layer: Dict[int, Set[str]] = {}
        self._live_tensors: Dict[str, torch.Tensor] = {}
        self._model_events_by_layer: Dict[int, threading.Event] = {}
        self._optimizer_events_by_layer: Dict[int, threading.Event] = {}
        self._injected_layers: Set[int] = set()
        for layer_idx, record in self._records_by_layer.items():
            model_event = threading.Event()
            optimizer_event = threading.Event()
            if record.model_ready:
                model_event.set()
            if record.optimizer_ready or not record.contains_optimizer_state:
                optimizer_event.set()
            self._model_events_by_layer[layer_idx] = model_event
            self._optimizer_events_by_layer[layer_idx] = optimizer_event
        self.wait_s: float = 0.0
        self.forward_wait_model_s: float = 0.0
        self.optimizer_wait_s: float = 0.0
        self.optimizer_materialize_s: float = 0.0
        self.optimizer_load_s: float = 0.0
        self.optimizer_sync_s: float = 0.0
        self.materialized_layers: Set[int] = set()
        self.optimizer_materialized: bool = False
        self.missing_layers: Set[int] = set()
        self.first_wait_s: Optional[float] = None
        self.first_layer_ready_s: Optional[float] = None
        self.last_layer_ready_s: Optional[float] = None
        self.inject_s: float = 0.0

    def attach_model_state_keys(self, model_state_keys) -> None:
        keys_by_layer: Dict[int, Set[str]] = {}
        for key in model_state_keys:
            layer_idx = _extract_layer_idx(str(key))
            if layer_idx >= 0:
                keys_by_layer.setdefault(layer_idx, set()).add(str(key))
        self._model_keys_by_layer = keys_by_layer

    def attach_live_model(self, model) -> None:
        live_tensors: Dict[str, torch.Tensor] = {}
        canonical_live_tensors: Dict[str, torch.Tensor] = {}

        def register_live_tensor(name: str, tensor: torch.Tensor) -> None:
            variants = {
                str(name),
                _canonical_live_model_key(name),
            }
            for variant in variants:
                if variant:
                    live_tensors.setdefault(variant, tensor)
                    canonical_live_tensors.setdefault(
                        _canonical_live_model_key(variant), tensor
                    )

        modules = model if isinstance(model, (list, tuple)) else [model]
        for module in modules:
            for name, param in module.named_parameters():
                register_live_tensor(name, param.data)
            for name, buf in module.named_buffers():
                register_live_tensor(name, buf)
        live_tensors.update(canonical_live_tensors)
        self._live_tensors = live_tensors

    def _lookup_live_tensor(self, key: str) -> Optional[torch.Tensor]:
        candidates = (
            key,
            f"module.{key}",
            _normalize_model_state_key(key),
            _canonical_live_model_key(key),
        )
        for candidate in candidates:
            dst = self._live_tensors.get(candidate)
            if dst is not None:
                return dst
        canonical_key = _canonical_live_model_key(key)
        for live_key, dst in self._live_tensors.items():
            if live_key.endswith(canonical_key) or canonical_key.endswith(live_key):
                return dst
        return None

    def _inject_layer_tensors(self, layer_idx: int) -> None:
        if layer_idx in self._injected_layers:
            return
        record = self._records_by_layer.get(layer_idx)
        if record is None:
            return
        tensors = record.model_tensors
        if not tensors and record.tensors:
            tensors, _ = _split_recovered_tensors(
                record.tensors, record.model_tensor_keys, record.optimizer_tensor_keys,
            )
        if not tensors:
            return
        t0 = time.time()
        copied = 0
        matched = 0
        missing = 0
        with torch.no_grad():
            for key in record.model_tensor_keys:
                tensor = tensors.get(key)
                if tensor is None:
                    continue
                dst = self._lookup_live_tensor(key)
                if dst is None:
                    missing += 1
                    continue
                dst.copy_(tensor.to(device=dst.device, dtype=dst.dtype), non_blocking=True)
                copied += tensor.numel() * tensor.element_size()
                matched += 1
        if layer_idx == _frcheck_first_layer_recovery_target_idx and matched > 0:
            _record_frcheck_first_layer_cuda_event("h2d_s", layer_idx)
        self._injected_layers.add(layer_idx)
        inject_s = time.time() - t0
        self.inject_s += inject_s
        record.model_tensors = None
        if record.optimizer_tensors is None and record.tensors is not None:
            _, optimizer_tensors = _split_recovered_tensors(
                record.tensors,
                record.model_tensor_keys,
                record.optimizer_tensor_keys,
            )
            record.optimizer_tensors = optimizer_tensors
        record.tensors = record.optimizer_tensors if record.contains_optimizer_state else None

    def mark_model_ready(
        self,
        layer_idx: int,
        materialize_s: float = 0.0,
        nbytes: int = 0,
        model_tensors: Optional[Dict[str, torch.Tensor]] = None,
        error: str = "",
    ) -> None:
        record = self._records_by_layer.get(layer_idx)
        if record is None:
            return
        record.materialize_s = materialize_s
        record.nbytes = nbytes
        if model_tensors is not None:
            record.model_tensors = model_tensors
        record.error = error
        record.model_ready = not error
        record.ready = record.model_ready and (
            record.optimizer_ready or not record.contains_optimizer_state
        )
        if self.first_layer_ready_s is None:
            self.first_layer_ready_s = time.time()
        self.last_layer_ready_s = time.time()
        event = self._model_events_by_layer.setdefault(layer_idx, threading.Event())
        event.set()
        _frcheck_recovery_profile(
            "runtime", "model_ready_done", layer=record.layer_name,
            layer_idx=layer_idx, model_tensors=len(model_tensors or {}),
            error=error or "none",
        )

    def mark_optimizer_ready(
        self,
        layer_idx: int,
        optimizer_tensors: Optional[Dict[str, torch.Tensor]] = None,
        error: str = "",
    ) -> None:
        record = self._records_by_layer.get(layer_idx)
        if record is None:
            return
        if optimizer_tensors is not None:
            record.optimizer_tensors = optimizer_tensors
        record.error = error or record.error
        record.optimizer_ready = not error
        record.ready = record.model_ready and (
            record.optimizer_ready or not record.contains_optimizer_state
        )
        event = self._optimizer_events_by_layer.setdefault(layer_idx, threading.Event())
        event.set()
        _frcheck_recovery_profile(
            "runtime", "optimizer_ready_done", layer=record.layer_name,
            layer_idx=layer_idx,
            optimizer_tensors=len(optimizer_tensors or {}),
            error=error or "none",
        )

    def mark_layer_ready(
        self,
        layer_idx: int,
        materialize_s: float = 0.0,
        nbytes: int = 0,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        error: str = "",
    ) -> None:
        record = self._records_by_layer.get(layer_idx)
        if record is None:
            return
        model_tensors, optimizer_tensors = _split_recovered_tensors(
            tensors, record.model_tensor_keys, record.optimizer_tensor_keys,
        )
        self.mark_model_ready(
            layer_idx,
            materialize_s=materialize_s,
            nbytes=nbytes,
            model_tensors=model_tensors,
            error=error,
        )
        if record.contains_optimizer_state:
            self.mark_optimizer_ready(layer_idx, optimizer_tensors=optimizer_tensors, error=error)
        else:
            self.mark_optimizer_ready(layer_idx, error=error)

    def mark_layer_error(self, layer_idx: int, error: str) -> None:
        self.mark_model_ready(layer_idx, error=error)
        self.mark_optimizer_ready(layer_idx, error=error)

    def wait_and_materialize_layer(self, layer_idx: int) -> bool:
        t0 = time.time()
        try:
            record = self._records_by_layer.get(layer_idx)
            if record is None:
                self.missing_layers.add(layer_idx)
                return False
            event = self._model_events_by_layer.setdefault(layer_idx, threading.Event())
            event.wait()
            if not record.model_ready:
                raise RuntimeError(
                    f"FRCheck layer {record.layer_name} model is not ready: {record.error}"
                )
            self._inject_layer_tensors(layer_idx)
            first_touch = layer_idx not in self.materialized_layers
            self.materialized_layers.add(layer_idx)
            waited = time.time() - t0
            self.forward_wait_model_s += waited
            if self.first_wait_s is None:
                self.first_wait_s = waited
            if first_touch and _frcheck_debug_enabled():
                logger.debug(
                    "FRCheck layerwise forward: layer=%s idx=%d ready "
                    "recovery_materialize=%.4fs wait=%.4fs",
                    record.layer_name, layer_idx, record.materialize_s, waited,
                )
                try:
                    from megatron.training.global_vars import mark_recovery_to_forward_timer
                    mark_recovery_to_forward_timer(f"{record.layer_name}_forward_ready")
                except Exception:
                    pass
            return True
        finally:
            self.wait_s += time.time() - t0

    def wait_for_optimizer_layers(self) -> None:
        t0 = time.time()
        try:
            for layer_idx, record in sorted(self._records_by_layer.items()):
                if not record.contains_optimizer_state:
                    continue
                event = self._optimizer_events_by_layer.setdefault(layer_idx, threading.Event())
                event.wait()
                if not record.optimizer_ready:
                    raise RuntimeError(
                        f"FRCheck optimizer layer {record.layer_name} is not ready: "
                        f"{record.error}"
                    )
        finally:
            self.optimizer_wait_s += time.time() - t0

    def summary(self) -> Dict[str, Any]:
        pending_runtime_tensors = sum(
            len(record.model_tensors or {})
            + len(record.optimizer_tensors or {})
            + len(record.tensors or {})
            for record in self._records_by_layer.values()
        )
        return {
            "ready_layers": len(self._records_by_layer),
            "materialized_layers": len(self.materialized_layers),
            "missing_layers": sorted(self.missing_layers),
            "wait_s": self.wait_s,
            "forward_wait_model_s": self.forward_wait_model_s,
            "optimizer_wait_s": self.optimizer_wait_s,
            "optimizer_materialize_s": self.optimizer_materialize_s,
            "optimizer_load_s": self.optimizer_load_s,
            "optimizer_sync_s": self.optimizer_sync_s,
            "optimizer_h2d_s": self.optimizer_materialize_s + self.optimizer_load_s + self.optimizer_sync_s,
            "first_wait_s": self.first_wait_s,
            "first_layer_ready_s": self.first_layer_ready_s,
            "last_layer_ready_s": self.last_layer_ready_s,
            "inject_s": self.inject_s,
            "injected_layers": len(self._injected_layers),
            "optimizer_materialized": self.optimizer_materialized,
            "pending_runtime_tensors": pending_runtime_tensors,
        }


class _FRCheckRecoveryService:
    """Single owner for FRCheck recovery state across load and training safe points."""

    INIT = "INIT"
    COMMON_READY = "COMMON_READY"
    LAYERS_RUNNING = "LAYERS_RUNNING"
    MODEL_READY = "MODEL_READY"
    OPTIMIZER_READY = "OPTIMIZER_READY"
    DONE = "DONE"
    TORN_DOWN = "TORN_DOWN"
    ERROR = "ERROR"

    def __init__(self) -> None:
        self.role: str = "unknown"
        self.runtime: Optional[_FRCheckLayerwiseRuntime] = None
        self.worker: Optional[threading.Thread] = None
        self.state: str = self.INIT
        self.error: Optional[BaseException] = None
        self.safe_point_teardown_done: bool = False

    def reset_for_load(self, role: str) -> None:
        global _active_layerwise_runtime, _active_recovery_worker
        global _recovery_async_parity_initialized, _recovery_async_parity_submitted
        global _recovery_async_parity_thread, _recovery_async_parity_error
        global _pending_recovery_parity_repair, _frcheck_async_runtime_timing_reported
        self.role = role
        _frcheck_async_runtime_timing_reported = False
        self.runtime = None
        self.worker = None
        _active_layerwise_runtime = None
        _active_recovery_worker = None
        self.state = self.INIT
        self.error = None
        self.safe_point_teardown_done = False
        _recovery_async_parity_initialized = False
        _recovery_async_parity_submitted = False
        _recovery_async_parity_thread = None
        _recovery_async_parity_error = None
        _pending_recovery_parity_repair = None
        _frcheck_recovery_profile(self.role, "service_reset", state=self.state)

    def attach_runtime(self, runtime: Optional[_FRCheckLayerwiseRuntime]) -> None:
        self.runtime = runtime
        if runtime is not None and self.state == self.INIT:
            self.state = self.COMMON_READY
        _frcheck_recovery_profile(
            self.role, "service_attach_runtime",
            state=self.state, has_runtime=runtime is not None,
        )

    def attach_worker(self, worker: Optional[threading.Thread]) -> None:
        self.worker = worker
        if worker is not None:
            self.state = self.LAYERS_RUNNING
            self.role = getattr(worker, "_frcheck_recovery_role", self.role)
        _frcheck_recovery_profile(
            self.role, "service_attach_worker",
            state=self.state, has_worker=worker is not None,
        )

    def _check_worker_error(self) -> None:
        worker = self.worker
        if worker is None:
            return
        error_holder = getattr(worker, "_frcheck_error_holder", None)
        if isinstance(error_holder, dict) and error_holder.get("error") is not None:
            self.error = error_holder["error"]
            self.state = self.ERROR
            self.worker = None
            raise RuntimeError("FRCheck async recovery worker failed") from self.error

    def wait_all(self, reason: str = "explicit") -> None:
        worker = self.worker
        if worker is not None and worker.is_alive():
            role = getattr(worker, "_frcheck_recovery_role", self.role)
            jobs = getattr(worker, "_frcheck_job_count", -1)
            _frcheck_recovery_profile(role, "join_wait_start", jobs=jobs, reason=reason)
            t_join = time.time()
            if _frcheck_debug_enabled():
                logger.info("FRCheck: waiting for async recovery worker to finish (%s)", reason)
            worker.join()
            _frcheck_recovery_profile(
                role, "join_wait_done", jobs=jobs,
                elapsed_s=time.time() - t_join, reason=reason,
            )
        self._check_worker_error()
        self.worker = None
        if self.state != self.ERROR:
            self.state = self.DONE

    def wait_layer(self, layer_idx: int) -> bool:
        if self.runtime is None:
            return False
        return self.runtime.wait_and_materialize_layer(layer_idx)

    def wait_optimizer(self) -> bool:
        if self.runtime is None:
            return False
        self.runtime.wait_for_optimizer_layers()
        self.wait_all(reason="optimizer")
        if not self.safe_point_teardown_done:
            self.state = self.OPTIMIZER_READY
        return True

    def summary(self) -> Dict[str, Any]:
        runtime_summary = None if self.runtime is None else self.runtime.summary()
        worker_alive = self.worker is not None and self.worker.is_alive()
        return {
            "role": self.role,
            "state": self.state,
            "worker_alive": worker_alive,
            "has_runtime": self.runtime is not None,
            "safe_point_teardown_done": self.safe_point_teardown_done,
            "runtime": runtime_summary,
        }


_active_layerwise_runtime: Optional[_FRCheckLayerwiseRuntime] = None
_active_recovery_worker: Optional[threading.Thread] = None
_active_recovery_service: Optional[_FRCheckRecoveryService] = None
_pending_recovery_parity_repair: Optional[Dict[str, Any]] = None
_pending_optimizer_state: Optional[Dict[str, Any]] = None
_pending_optimizer_container: Optional[Dict[str, Any]] = None


def _get_active_frcheck_recovery_service() -> _FRCheckRecoveryService:
    global _active_recovery_service
    if _active_recovery_service is None:
        _active_recovery_service = _FRCheckRecoveryService()
    return _active_recovery_service


def _set_active_frcheck_layerwise_runtime(
    runtime: Optional[_FRCheckLayerwiseRuntime],
) -> None:
    global _active_layerwise_runtime
    _active_layerwise_runtime = runtime
    _get_active_frcheck_recovery_service().attach_runtime(runtime)


def _set_active_frcheck_recovery_worker(worker: Optional[threading.Thread]) -> None:
    global _active_recovery_worker
    _active_recovery_worker = worker
    _get_active_frcheck_recovery_service().attach_worker(worker)


def frcheck_wait_for_async_recovery() -> None:
    global _active_recovery_worker
    service = _get_active_frcheck_recovery_service()
    service.wait_all(reason="explicit")
    _active_recovery_worker = service.worker


def _layerwise_record_from_layer_buf(
    layer_name: str,
    encode_iter: int,
    layer_infos: List,
    nbytes: int,
    materialize_s: float,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
    optimizer_layer_map: Optional[Dict[str, int]] = None,
    model_tensor_keys: Optional[List[str]] = None,
    optimizer_tensor_keys: Optional[List[str]] = None,
) -> _FRCheckLayerReadyRecord:
    layer_idx = -1
    if layer_name.startswith("layer_"):
        try:
            layer_idx = int(layer_name.split("_", 1)[1])
        except (TypeError, ValueError):
            layer_idx = -1
    tensor_keys = [
        getattr(info, "key", "")
        for info in layer_infos
        if getattr(info, "key", "")
    ]
    model_tensor_keys = list(model_tensor_keys or [])
    optimizer_tensor_keys = list(optimizer_tensor_keys or [])
    if not model_tensor_keys or not optimizer_tensor_keys:
        for key in tensor_keys:
            ownership = _classify_frcheck_tensor(key, optimizer_layer_map)
            if (
                layer_idx >= 0
                and ownership.kind in ("model_layer", "model_common")
                and key not in model_tensor_keys
            ):
                model_tensor_keys.append(key)
            elif (
                layer_idx >= 0
                and ownership.kind in ("optimizer_layer", "optimizer_common")
                and key not in optimizer_tensor_keys
            ):
                optimizer_tensor_keys.append(key)
    model_tensors, optimizer_tensors = _split_recovered_tensors(
        tensors, model_tensor_keys, optimizer_tensor_keys, optimizer_layer_map,
    )
    return _FRCheckLayerReadyRecord(
        layer_name=layer_name,
        layer_idx=layer_idx,
        encode_iter=encode_iter,
        tensor_keys=tensor_keys,
        model_tensor_keys=model_tensor_keys,
        optimizer_tensor_keys=optimizer_tensor_keys,
        nbytes=nbytes,
        contains_optimizer_state=bool(optimizer_tensor_keys),
        materialize_s=materialize_s,
        tensors=tensors,
        model_tensors=model_tensors,
        optimizer_tensors=optimizer_tensors,
        model_ready=bool(model_tensors) or not model_tensor_keys,
        optimizer_ready=bool(optimizer_tensors) or not optimizer_tensor_keys,
    )


def _make_pending_layerwise_records(
    rank: int,
    is_failed: bool,
    main_payload: Dict[str, Any],
    all_layer_order: Dict[int, List[str]],
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]],
    all_frcheck_dirs: List[Optional[Path]],
    checkpoint_dir: Path,
    saved_block_size: int,
    n_encode_iters: int,
    recovery_rank: Optional[int] = None,
) -> List[_FRCheckLayerReadyRecord]:
    records: List[_FRCheckLayerReadyRecord] = []
    for encode_iter in range(n_encode_iters):
        job = _make_layer_recovery_job(
            encode_iter, rank, is_failed, main_payload, all_layer_order,
            all_layer_metadata, all_frcheck_dirs, checkpoint_dir, saved_block_size,
            recovery_rank=recovery_rank,
        )
        if job is None or job.layer_idx < 0:
            continue
        layer_meta = all_layer_metadata.get(rank, {}).get(job.layer_name, {})
        tensor_keys = [
            getattr(info, "key", "")
            for info in job.layer_infos
            if getattr(info, "key", "")
        ]
        model_tensor_keys = list(layer_meta.get("model_tensor_keys", []))
        optimizer_tensor_keys = list(layer_meta.get("optimizer_tensor_keys", []))
        if not model_tensor_keys:
            model_tensor_keys = [
                key for key in tensor_keys
                if _classify_frcheck_tensor(key).kind in ("model_layer", "model_common")
            ]
        if not optimizer_tensor_keys:
            optimizer_tensor_keys = [
                key for key in tensor_keys
                if _classify_frcheck_tensor(key).kind in ("optimizer_layer", "optimizer_common")
            ]
        records.append(
            _FRCheckLayerReadyRecord(
                layer_name=job.layer_name,
                layer_idx=job.layer_idx,
                encode_iter=job.encode_iter,
                tensor_keys=tensor_keys,
                model_tensor_keys=model_tensor_keys,
                optimizer_tensor_keys=optimizer_tensor_keys,
                nbytes=job.actual_size,
                contains_optimizer_state=bool(
                    layer_meta.get("contains_optimizer_state", bool(optimizer_tensor_keys))
                ),
                ready=False,
                model_ready=False,
                optimizer_ready=not bool(
                    layer_meta.get("contains_optimizer_state", bool(optimizer_tensor_keys))
                ),
            )
        )
    return records


def _extract_layer_tensors_from_buf(
    layer_buf: torch.Tensor,
    layer_infos: List,
    clone_storage: bool = True,
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    buf = layer_buf.detach().contiguous().reshape(-1).view(torch.uint8)
    for info in layer_infos:
        key = getattr(info, "key", "")
        size = int(getattr(info, "size_bytes", 0))
        offset = int(getattr(info, "offset", 0))
        dtype = getattr(info, "dtype", None)
        shape = tuple(getattr(info, "shape", ()))
        if not key or size <= 0 or dtype is None or not shape:
            continue
        if offset + size > buf.numel():
            continue
        n_elem = 1
        for dim in shape:
            n_elem *= int(dim)
        raw = buf[offset:offset + size]
        if clone_storage:
            raw = raw.clone()
        view = raw.view(dtype).reshape(shape)
        if view.numel() != n_elem:
            continue
        tensors[key] = view
    return tensors


def install_frcheck_layerwise_runtime_from_state_dict(
    state_dict: Dict[str, Any],
    model=None,
) -> None:
    """Install FRCheck layer-ready metadata for forward-time hooks."""
    global _active_layerwise_runtime
    metadata = state_dict.pop("__frcheck_layerwise_runtime__", None)
    if not metadata:
        _set_active_frcheck_layerwise_runtime(None)
        return

    if _active_layerwise_runtime is not None and metadata.get("active_runtime", False):
        runtime = _active_layerwise_runtime
        model_sd = state_dict.get("model")
        if isinstance(model_sd, dict):
            runtime.attach_model_state_keys(model_sd.keys())
        if model is not None:
            runtime.attach_live_model(model)
        optimizer_keys = [
            key
            for record in runtime._records_by_layer.values()
            for key in record.optimizer_tensor_keys
        ]
        if _frcheck_debug_enabled():
            logger.info(
                "FRCheck layerwise runtime attached to active recovery worker: "
                "ready_layers=%d live_tensors=%d optimizer_key_summary=%s",
                len(runtime._records_by_layer), len(runtime._live_tensors),
                _summarize_optimizer_keys(optimizer_keys),
            )
        return

    records = [
        _FRCheckLayerReadyRecord(
            layer_name=item.get("layer_name", ""),
            layer_idx=int(item.get("layer_idx", -1)),
            encode_iter=int(item.get("encode_iter", -1)),
            tensor_keys=list(item.get("tensor_keys", [])),
            model_tensor_keys=list(item.get("model_tensor_keys", [])),
            optimizer_tensor_keys=list(item.get("optimizer_tensor_keys", [])),
            nbytes=int(item.get("nbytes", 0)),
            contains_optimizer_state=bool(item.get("contains_optimizer_state", False)),
            materialize_s=float(item.get("materialize_s", 0.0)),
            ready=bool(item.get("ready", True)),
            model_ready=bool(item.get("model_ready", item.get("ready", True))),
            optimizer_ready=bool(
                item.get(
                    "optimizer_ready",
                    item.get("ready", True) or not item.get("contains_optimizer_state", False),
                )
            ),
            error=str(item.get("error", "")),
            tensors=item.get("tensors"),
            model_tensors=item.get("model_tensors"),
            optimizer_tensors=item.get("optimizer_tensors"),
        )
        for item in metadata.get("records", [])
    ]
    runtime = _FRCheckLayerwiseRuntime(records)
    model_sd = state_dict.get("model")
    if isinstance(model_sd, dict):
        runtime.attach_model_state_keys(model_sd.keys())
    if model is not None:
        runtime.attach_live_model(model)
    _set_active_frcheck_layerwise_runtime(runtime)
    optimizer_keys = [
        key
        for record in runtime._records_by_layer.values()
        for key in record.optimizer_tensor_keys
    ]
    if _frcheck_debug_enabled():
        logger.info(
            "FRCheck layerwise runtime installed: ready_layers=%d live_tensors=%d "
            "runtime_tensors=%d optimizer_key_summary=%s",
            len(runtime._records_by_layer),
            len(runtime._live_tensors),
            sum(len(r.tensors or {}) for r in runtime._records_by_layer.values()),
            _summarize_optimizer_keys(optimizer_keys),
        )


def frcheck_filter_layerwise_model_placeholders(
    state_dict: Dict[str, Any],
) -> Tuple[int, int, Set[str]]:
    """Drop layer-owned model placeholders that will be injected by runtime."""
    runtime = _active_layerwise_runtime
    if runtime is None:
        return 0, 0, set()

    skip_keys: Set[str] = set()
    for record in runtime._records_by_layer.values():
        for key in record.model_tensor_keys:
            canonical_key = _canonical_live_model_key(key)
            if canonical_key:
                skip_keys.add(canonical_key)
    if not skip_keys:
        return 0, 0, set()

    def should_skip(key: str) -> bool:
        canonical_key = _canonical_live_model_key(key)
        if canonical_key in skip_keys:
            return True
        for skip_key in skip_keys:
            if canonical_key.endswith(skip_key) or skip_key.endswith(canonical_key):
                return True
        return False

    removed = 0
    removed_bytes = 0
    for model_key, model_state in list(state_dict.items()):
        if model_key != "model" and not re.fullmatch(r"model\d+", str(model_key)):
            continue
        if not isinstance(model_state, dict):
            continue
        for key in list(model_state.keys()):
            if not should_skip(str(key)):
                continue
            tensor = model_state.pop(key)
            removed += 1
            if torch.is_tensor(tensor):
                removed_bytes += tensor.numel() * tensor.element_size()

    if removed and _frcheck_debug_enabled():
        logger.debug(
            "FRCheck layerwise model load: skipped %d placeholder tensors "
            "(%.2f MiB); runtime will inject them before layer forward",
            removed, removed_bytes / (1024 ** 2),
        )
    return removed, removed_bytes, skip_keys


def frcheck_wait_and_materialize_layer(layer_idx: int) -> bool:
    """Wait for a recovered FRCheck layer to be ready before forward."""
    return _get_active_frcheck_recovery_service().wait_layer(layer_idx)


def frcheck_async_layerwise_active() -> bool:
    """Return True when a recovery worker can overlap with training progress."""
    service = _get_active_frcheck_recovery_service()
    worker = service.worker
    return worker is not None and worker.is_alive()


def frcheck_materialize_all_layers() -> None:
    runtime = _active_layerwise_runtime
    if runtime is None:
        return
    for layer_idx in sorted(runtime._records_by_layer):
        runtime.wait_and_materialize_layer(layer_idx)
    frcheck_wait_for_async_recovery()


def frcheck_materialize_first_layer_for_timer() -> bool:
    """Inject the first recovered layer so timing ends before forward compute."""
    runtime = _active_layerwise_runtime
    if runtime is None:
        return False
    layer_indices = [
        idx for idx, record in runtime._records_by_layer.items()
        if record.model_tensor_keys
    ]
    if not layer_indices:
        return False
    layer_idx = min(layer_indices)
    t0 = time.time()
    record = runtime._records_by_layer.get(layer_idx)
    if record is None:
        return False
    event = runtime._model_events_by_layer.setdefault(layer_idx, threading.Event())
    event.wait()
    if not record.model_ready:
        raise RuntimeError(
            f"FRCheck layer {record.layer_name} model is not ready: {record.error}"
        )
    runtime._inject_layer_tensors(layer_idx)
    runtime.materialized_layers.add(layer_idx)
    waited = time.time() - t0
    runtime.wait_s += waited
    if runtime.first_wait_s is None:
        runtime.first_wait_s = waited
    if _frcheck_debug_enabled():
        logger.debug(
            "FRCheck layerwise timing: first layer=%s idx=%d injected wait=%.4fs",
            record.layer_name, layer_idx, waited,
        )
    try:
        from megatron.training.global_vars import mark_recovery_to_forward_timer
        mark_recovery_to_forward_timer(f"{record.layer_name}_injected")
    except Exception:
        pass
    return True


def frcheck_register_pending_optimizer_state(
    state_dict: Dict[str, Any], allow_without_runtime: bool = False,
) -> bool:
    """Keep optimizer state_dict alive until its first consumption point."""
    global _pending_optimizer_container, _pending_optimizer_state
    runtime = _active_layerwise_runtime
    if runtime is None and not allow_without_runtime:
        return False
    if runtime is not None and not any(
        r.contains_optimizer_state for r in runtime._records_by_layer.values()
    ):
        return False
    optim_state = state_dict.pop("optimizer", None)
    if optim_state is None:
        return False
    _pending_optimizer_container = {"optimizer": optim_state}
    _pending_optimizer_state = optim_state
    if _frcheck_debug_enabled():
        deferred_layers = 0 if runtime is None else sum(
            1 for r in runtime._records_by_layer.values() if r.contains_optimizer_state
        )
        logger.debug(
            "FRCheck optimizer recovery: deferred optimizer load for %d layers "
            "(runtime=%s)",
            deferred_layers,
            runtime is not None,
        )
    return True


def _frcheck_optimizer_state_dict(optim_state: Dict[str, Any]) -> Optional[Dict[Any, Any]]:
    if not isinstance(optim_state, dict):
        return None
    torch_optim_state = optim_state.get("optimizer", optim_state)
    if not isinstance(torch_optim_state, dict):
        return None
    state = torch_optim_state.get("state")
    return state if isinstance(state, dict) else None


def frcheck_normalize_optimizer_state_param_keys(optim_state: Dict[str, Any]) -> None:
    from megatron.core.optimizer.optimizer import normalize_optimizer_state_param_keys

    torch_optim = optim_state.get("optimizer")
    if isinstance(torch_optim, dict):
        normalize_optimizer_state_param_keys(torch_optim)
        return
    normalize_optimizer_state_param_keys(optim_state)


def _assign_deferred_optimizer_tensor(root: Dict[str, Any], flat_key: str, tensor: torch.Tensor) -> bool:
    """Materialize optimizer tensors without aliasing recovery buffers."""
    fp32_prefix = "optimizer.fp32_params_flat."
    if flat_key.startswith(fp32_prefix):
        suffix = flat_key[len(fp32_prefix):]
        if not suffix:
            return False
        flat = root.setdefault("fp32_params_flat", {})
        if not isinstance(flat, dict):
            return False
        flat[suffix] = tensor.detach().clone()
        return True

    if not flat_key.startswith("optimizer.optimizer.state."):
        return False

    parts = flat_key.split(".")
    if len(parts) != 5:
        return False
    state_id, leaf = parts[3], parts[4]
    if leaf not in ("exp_avg", "exp_avg_sq", "fp32_param", "step"):
        return False

    torch_optim = root.setdefault("optimizer", {})
    if not isinstance(torch_optim, dict):
        return False
    state = torch_optim.setdefault("state", {})
    if not isinstance(state, dict):
        return False

    try:
        state_key = int(state_id)
    except (TypeError, ValueError):
        state_key = state_id
    entry = state.setdefault(state_key, {})
    if not isinstance(entry, dict):
        return False
    # Optimizer state must not alias reusable FRCheck recovery buffers.
    entry[leaf] = tensor.detach().clone()
    return True


def _sync_frcheck_model_params_to_optimizer_main_params(optimizer) -> bool:
    """Copy layerwise-injected model params into optimizer fp32 master params once."""
    if optimizer is None:
        return False
    optimizers = getattr(optimizer, "chained_optimizers", None)
    if optimizers is None:
        optimizers = [optimizer]
    synced = False
    for optim_instance in optimizers:
        copy_fn = getattr(optim_instance, "_copy_model_params_to_main_params", None)
        if copy_fn is None:
            continue
        copy_fn()
        synced = True
    if synced and torch.cuda.is_available():
        torch.cuda.synchronize()
    return synced


def _install_current_fp32_params_for_optimizer_load(optimizer, optim_state: Dict[str, Any]) -> None:
    """Fallback to live fp32 master params when the checkpoint lacks them."""
    if "fp32_from_fp16_params" in optim_state:
        return

    source_optimizer = optimizer
    optimizers = getattr(optimizer, "chained_optimizers", None)
    if optimizers is not None:
        if len(optimizers) != 1:
            return
        source_optimizer = optimizers[0]

    fp32_groups = getattr(source_optimizer, "fp32_from_float16_groups", None)
    if fp32_groups is not None:
        optim_state["fp32_from_fp16_params"] = fp32_groups
        optim_state["fp32_from_fp16"] = fp32_groups


def _materialize_pending_optimizer_tensors(
    runtime: _FRCheckLayerwiseRuntime,
    optim_state: Dict[str, Any],
) -> Tuple[int, int, List[str]]:
    expected = 0
    updated = 0
    missing_keys: List[str] = []
    for record in runtime._records_by_layer.values():
        if not record.contains_optimizer_state:
            continue
        tensor_source = record.optimizer_tensors
        if not tensor_source and record.tensors:
            _, tensor_source = _split_recovered_tensors(
                record.tensors,
                record.model_tensor_keys,
                record.optimizer_tensor_keys,
                None,
            )
        if not tensor_source:
            continue
        for key in record.optimizer_tensor_keys:
            expected += 1
            tensor = tensor_source.get(key)
            if tensor is None:
                missing_keys.append(key)
                continue
            if _assign_deferred_optimizer_tensor(optim_state, key, tensor):
                updated += 1
            else:
                missing_keys.append(key)
    return updated, expected, missing_keys


def frcheck_wait_for_optimizer_state(optimizer=None) -> bool:
    """Wait for layerwise optimizer tensors and load deferred optimizer state once."""
    global _pending_optimizer_container, _pending_optimizer_state
    service = _get_active_frcheck_recovery_service()
    runtime = service.runtime
    if runtime is None:
        loaded = False
        if _pending_optimizer_state is not None and optimizer is not None:
            # Survivor state may reference reusable recovery buffers. Preserve the
            # original ownership guarantee, but move the clone off the first-forward
            # critical path together with optimizer loading.
            from megatron.training.checkpointing import _clone_inprocess_optimizer_tensors

            t_clone = time.time()
            optimizer_state = _clone_inprocess_optimizer_tensors(
                _pending_optimizer_state
            )
            clone_s = time.time() - t_clone
            t_load = time.time()
            optimizer.load_state_dict(optimizer_state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            load_s = time.time() - t_load
            logger.info(
                "FRCheck deferred survivor optimizer restore: clone_s=%.4fs "
                "load_s=%.4fs total_s=%.4fs",
                clone_s,
                load_s,
                clone_s + load_s,
            )
            _pending_optimizer_container = None
            _pending_optimizer_state = None
            loaded = True
        _finish_recovery_parity_repair_submissions("after_optimizer_state", service.role)
        if (
            _pending_recovery_parity_repair is None
            and _pending_optimizer_state is None
            and not service.safe_point_teardown_done
        ):
            _flush_recovery_async_parity("after_optimizer_state", service.role)
            _teardown_frcheck_native_after_load()
            service.safe_point_teardown_done = True
            service.state = service.TORN_DOWN
        return loaded
    t0 = time.time()
    if not runtime.optimizer_materialized and optimizer is not None:
        if _sync_frcheck_model_params_to_optimizer_main_params(optimizer):
            if _frcheck_debug_enabled():
                logger.debug(
                    "FRCheck optimizer recovery: synced injected model params to optimizer main params"
                )
    if _pending_optimizer_state is None:
        runtime.optimizer_materialized = True
        _finish_recovery_parity_repair_submissions("after_optimizer_state", service.role)
        if _pending_recovery_parity_repair is None and not service.safe_point_teardown_done:
            _flush_recovery_async_parity("after_optimizer_state", service.role)
            _teardown_frcheck_native_after_load()
            service.safe_point_teardown_done = True
            service.state = service.TORN_DOWN
        return False
    service.wait_optimizer()
    if optimizer is None:
        return False
    t_materialize = time.time()
    updated, expected, missing_keys = _materialize_pending_optimizer_tensors(
        runtime, _pending_optimizer_state,
    )
    runtime.optimizer_materialize_s += time.time() - t_materialize
    frcheck_normalize_optimizer_state_param_keys(_pending_optimizer_state)
    if updated != expected:
        logger.error(
            "FRCheck optimizer recovery: materialized %d/%d optimizer tensors; "
            "missing examples=%s",
            updated, expected, missing_keys[:8],
        )
        raise RuntimeError(
            "FRCheck optimizer recovery: incomplete deferred optimizer "
            f"materialization ({updated}/{expected})"
        )
    from megatron.core.dist_checkpointing.strategies.state_dict_decomposer import (
        unflatten_optimizer_fp32_params,
    )
    unflatten_optimizer_fp32_params({"optimizer": _pending_optimizer_state})
    _install_current_fp32_params_for_optimizer_load(optimizer, _pending_optimizer_state)
    t_load = time.time()
    optimizer.load_state_dict(_pending_optimizer_state)
    runtime.optimizer_load_s += time.time() - t_load
    for record in runtime._records_by_layer.values():
        if record.contains_optimizer_state:
            record.optimizer_tensors = None
            record.tensors = None
    _pending_optimizer_container = None
    _pending_optimizer_state = None
    runtime.optimizer_materialized = True
    if torch.cuda.is_available():
        t_sync = time.time()
        torch.cuda.synchronize()
        runtime.optimizer_sync_s += time.time() - t_sync
    _finish_recovery_parity_repair_submissions("after_optimizer_state", service.role)
    if _pending_recovery_parity_repair is None and not service.safe_point_teardown_done:
        _flush_recovery_async_parity("after_optimizer_state", service.role)
        _teardown_frcheck_native_after_load()
        service.safe_point_teardown_done = True
        service.state = service.TORN_DOWN
    if torch.cuda.is_available():
        t_sync = time.time()
        torch.cuda.synchronize()
        runtime.optimizer_sync_s += time.time() - t_sync
    _log_frcheck_async_runtime_timing_once("after_optimizer_state")
    if _frcheck_debug_enabled():
        logger.debug(
            "FRCheck optimizer recovery: loaded deferred optimizer state in %.4fs "
            "(updated_tensors=%d)",
            time.time() - t0, updated,
        )
    return True


def get_frcheck_layerwise_runtime_summary() -> Optional[Dict[str, Any]]:
    service = _get_active_frcheck_recovery_service()
    summary = service.summary()
    return None if not summary.get("has_runtime") else summary


def _log_frcheck_async_runtime_timing_once(context: str) -> None:
    global _frcheck_async_runtime_timing_reported
    if _frcheck_async_runtime_timing_reported:
        return
    summary = get_frcheck_layerwise_runtime_summary()
    runtime = (summary or {}).get("runtime", {}) or {}
    layer_inject_s = float(runtime.get("inject_s", 0.0) or 0.0)
    forward_wait_model_s = float(runtime.get("forward_wait_model_s", 0.0) or 0.0)
    optimizer_wait_s = float(runtime.get("optimizer_wait_s", 0.0) or 0.0)
    optimizer_materialize_s = float(runtime.get("optimizer_materialize_s", 0.0) or 0.0)
    optimizer_load_s = float(runtime.get("optimizer_load_s", 0.0) or 0.0)
    optimizer_sync_s = float(runtime.get("optimizer_sync_s", 0.0) or 0.0)
    optimizer_h2d_s = float(runtime.get("optimizer_h2d_s", 0.0) or 0.0)
    h2d_s = layer_inject_s + optimizer_h2d_s
    wait_s = float(runtime.get("wait_s", 0.0) or 0.0)
    materialized_layers = float(runtime.get("materialized_layers", 0.0) or 0.0)
    injected_layers = float(runtime.get("injected_layers", 0.0) or 0.0)
    first_wait_s = float(runtime.get("first_wait_s", 0.0) or 0.0)
    local_values = {
        "layer_inject_s": layer_inject_s,
        "forward_wait_model_s": forward_wait_model_s,
        "optimizer_wait_s": optimizer_wait_s,
        "optimizer_materialize_s": optimizer_materialize_s,
        "optimizer_load_s": optimizer_load_s,
        "optimizer_sync_s": optimizer_sync_s,
        "optimizer_h2d_s": optimizer_h2d_s,
        "h2d_s": h2d_s,
        "wait_s": wait_s,
        "materialized_layers": materialized_layers,
        "injected_layers": injected_layers,
        "first_wait_s": first_wait_s,
    }
    try:
        from megatron.training import get_args
        inprocess_recovery = bool(
            getattr(get_args(), "ft_inprocess_recovery_benchmark", False)
        )
    except Exception:
        inprocess_recovery = False
    # In-process recovery reaches the common forward-backward boundary on every rank.
    # Defer its report until then so failed-rank runtime values can be aggregated.
    if inprocess_recovery and context != "after_forward_backward":
        return
    values = _timing_max_dict(local_values)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _frcheck_async_runtime_timing_reported = True
    if rank == 0:
        logger.info(
            "FRCheck async runtime timing (%s): layer_inject_s=%.6fs "
            "optimizer_materialize_s=%.6fs optimizer_load_s=%.6fs "
            "optimizer_sync_s=%.6fs h2d_s=%.6fs forward_wait_model_s=%.6fs "
            "optimizer_wait_s=%.6fs wait_s=%.6fs first_wait_s=%.6fs "
            "materialized_layers=%d injected_layers=%d",
            context,
            values["layer_inject_s"],
            values["optimizer_materialize_s"],
            values["optimizer_load_s"],
            values["optimizer_sync_s"],
            values["h2d_s"],
            values["forward_wait_model_s"],
            values["optimizer_wait_s"],
            values["wait_s"],
            values["first_wait_s"],
            int(values["materialized_layers"]),
            int(values["injected_layers"]),
        )



def finalize_frcheck_recovery_timing() -> None:
    """Resolve deferred CUDA milestones and log one completed recovery cycle."""
    if _frcheck_first_layer_recovery_start_s is None:
        return
    resolve_frcheck_first_layer_cuda_events()
    _log_frcheck_async_runtime_timing_once("after_forward_backward")

def frcheck_recovery_safe_point(point: str) -> None:
    """Optional safe point for future async recovery wait/teardown orchestration."""
    try:
        from megatron.training import get_args
        args = get_args()
    except Exception:
        return
    if not getattr(args, "use_frcheck", False):
        return
    service = _get_active_frcheck_recovery_service()
    _frcheck_recovery_profile(service.role, "safe_point", point=point, state=service.state)
    if service.safe_point_teardown_done:
        return
    mode = getattr(args, "frcheck_recovery_safe_point", "load")
    async_layerwise = (
        bool(getattr(args, "frcheck_async_recovery_forward", False))
        and service.worker is not None
    )
    if (
        async_layerwise
        and mode != "after_load_checkpoint"
        and point in ("after_load_checkpoint", "train_step_start", "forward_step_start")
    ):
        _frcheck_recovery_profile(
            service.role, "safe_point_deferred_for_layerwise",
            point=point, mode=mode, state=service.state,
        )
        return
    should_wait = (
        (mode == "after_load_checkpoint" and point == "after_load_checkpoint")
        or (mode == "train_step_start" and point == "train_step_start")
        or (mode == "forward_step_start" and point == "forward_step_start")
        or (mode == "optimizer_step" and point == "before_optimizer_step")
        or (async_layerwise and point == "before_optimizer_step")
    )
    if should_wait:
        service.wait_all(reason=point)
        # At before_optimizer_step the deferred optimizer state may still be loaded
        # immediately after this safe-point check. Log after that load so optimizer
        # materialize/load/sync time is included.
        if not (point == "before_optimizer_step" and _pending_optimizer_state is not None):
            _log_frcheck_async_runtime_timing_once(point)
        _finish_recovery_parity_repair_submissions(point, service.role)
        if (
            _pending_recovery_parity_repair is None
            and _pending_optimizer_state is None
        ):
            _flush_recovery_async_parity(point, service.role)
            _teardown_frcheck_native_after_load()
            service.safe_point_teardown_done = True
            service.state = service.TORN_DOWN
        else:
            service.state = service.DONE


def frcheck_log_layerwise_runtime_summary(context: str) -> None:
    summary = get_frcheck_layerwise_runtime_summary()
    if summary is not None and _frcheck_debug_enabled():
        logger.info("FRCheck layerwise runtime summary (%s): %s", context, summary)


def _group_by_layer(
    decomposed,
    distribute_common: bool = False,
    debug: bool = False,
    optimizer_layer_map: Optional[Dict[str, int]] = None,
    optimizer_model_key_map: Optional[Dict[str, str]] = None,
) -> List[_LayerGroup]:
    """Split decomposed state_dict into per-layer groups based on FQN key."""
    groups: Dict[int, _LayerGroup] = {}
    moved_model_common_tensors = 0
    moved_model_common_bytes = 0
    for info, tensor in zip(decomposed.tensor_infos, decomposed.tensor_data):
        ownership = _classify_frcheck_tensor(info.key, optimizer_layer_map)
        lidx = ownership.layer_idx
        if lidx not in groups:
            groups[lidx] = _LayerGroup(layer_idx=lidx, tensor_infos=[], tensor_data=[])
        g = groups[lidx]
        g.tensor_infos.append(info)
        g.tensor_data.append(tensor)
        g.total_bytes += info.size_bytes

    if distribute_common and -1 in groups and len(groups) > 1:
        common = groups.pop(-1)
        others = [g for g in groups.values() if g.layer_idx >= 0]
        moved_infos = []
        moved_data = []
        kept_infos = []
        kept_data = []
        common_model_targets: Dict[str, _LayerGroup] = {}
        moved_optimizer_common_tensors = 0
        moved_optimizer_common_bytes = 0

        if others:
            common_model_idx = 0
            for info, tensor in zip(common.tensor_infos, common.tensor_data):
                ownership = _classify_frcheck_tensor(info.key, optimizer_layer_map)
                if ownership.kind == "model_common":
                    target = others[common_model_idx % len(others)]
                    common_model_idx += 1
                    target.tensor_infos.append(info)
                    target.tensor_data.append(tensor)
                    target.total_bytes += info.size_bytes
                    common_model_targets[info.key] = target
                    if target.distributed_common_keys is None:
                        target.distributed_common_keys = set()
                    target.distributed_common_keys.add(info.key)
                    moved_infos.append(info)
                    moved_data.append(tensor)
                else:
                    kept_infos.append(info)
                    kept_data.append(tensor)
        else:
            kept_infos = list(common.tensor_infos)
            kept_data = list(common.tensor_data)

        if common_model_targets:
            if not optimizer_model_key_map:
                logger.warning(
                    "FRCheck distribute_common: optimizer_model_key_map is empty; "
                    "optimizer-common tensors will remain in layer_common",
                )
            others_by_layer = {group.layer_idx: group for group in others}
            next_kept_infos = []
            next_kept_data = []
            for info, tensor in zip(kept_infos, kept_data):
                model_key = (
                    optimizer_model_key_map.get(info.key)
                    if optimizer_model_key_map else None
                )
                target = _resolve_optimizer_distribute_target(
                    model_key, common_model_targets, others_by_layer,
                )
                if target is not None:
                    target.tensor_infos.append(info)
                    target.tensor_data.append(tensor)
                    target.total_bytes += info.size_bytes
                    if target.distributed_common_keys is None:
                        target.distributed_common_keys = set()
                    target.distributed_common_keys.add(info.key)
                    moved_optimizer_common_tensors += 1
                    moved_optimizer_common_bytes += int(getattr(info, "size_bytes", 0))
                else:
                    next_kept_infos.append(info)
                    next_kept_data.append(tensor)
            kept_infos = next_kept_infos
            kept_data = next_kept_data

        moved_model_common_tensors = len(moved_infos)
        moved_model_common_bytes = sum(getattr(info, "size_bytes", 0) for info in moved_infos)
        kept_bytes = sum(getattr(info, "size_bytes", 0) for info in kept_infos)
        if kept_infos:
            common.tensor_infos = kept_infos
            common.tensor_data = kept_data
            common.total_bytes = kept_bytes
            groups[-1] = common
        if debug:
            mapped_optimizer = sum(
                1
                for group in groups.values()
                if group.layer_idx >= 0
                for info in group.tensor_infos
                if _classify_frcheck_tensor(info.key, optimizer_layer_map).kind == "optimizer_layer"
            )
            moved_bytes = sum(getattr(info, "size_bytes", 0) for info in moved_infos)
            logger.info(
                "FRCheck: semantic common distribution moved model_common=%d "
                "tensors (%d bytes), moved optimizer_common=%d tensors (%d bytes), "
                "grouped optimizer_layer=%d, kept common=%d tensors (%d bytes)",
                len(moved_infos), moved_bytes,
                moved_optimizer_common_tensors, moved_optimizer_common_bytes,
                mapped_optimizer, len(kept_infos), kept_bytes,
            )
        elif distribute_common and kept_bytes > 1_000_000:
            logger.warning(
                "FRCheck distribute_common: kept layer_common=%.1fMB (%d tensors) "
                "after moving model_common=%d optimizer_common=%d; "
                "optimizer_model_key_map entries=%d",
                kept_bytes / 1e6, len(kept_infos),
                moved_model_common_tensors, moved_optimizer_common_tensors,
                len(optimizer_model_key_map or {}),
            )

    # Sort: non-layer (-1) first, then by layer index
    result = sorted(groups.values(), key=lambda g: (0 if g.layer_idx < 0 else 1, g.layer_idx))
    ownership_counts: Dict[str, int] = {}
    ownership_bytes: Dict[str, int] = {}
    optimizer_layer_keys: List[str] = []
    optimizer_common_keys: List[str] = []
    for group in result:
        for info in group.tensor_infos:
            ownership = _classify_frcheck_tensor(info.key, optimizer_layer_map)
            ownership_counts[ownership.kind] = ownership_counts.get(ownership.kind, 0) + 1
            ownership_bytes[ownership.kind] = (
                ownership_bytes.get(ownership.kind, 0)
                + int(getattr(info, "size_bytes", 0))
            )
            if ownership.kind == "optimizer_layer":
                optimizer_layer_keys.append(info.key)
            elif ownership.kind == "optimizer_common":
                optimizer_common_keys.append(info.key)
    layer_sizes = [
        (
            "layer_common" if group.layer_idx < 0 else f"layer_{group.layer_idx}",
            len(group.tensor_infos),
            group.total_bytes,
        )
        for group in result
    ]
    if debug:
        logger.info(
            "FRCheck save grouping: distribute_common=%s moved_model_common=%d tensors "
            "(%.1fMB) groups=%s ownership_counts=%s ownership_mb=%s",
            distribute_common,
            moved_model_common_tensors,
            moved_model_common_bytes / 1e6,
            [(name, count, round(nbytes / 1e6, 1)) for name, count, nbytes in layer_sizes],
            ownership_counts,
            {kind: round(nbytes / 1e6, 1) for kind, nbytes in ownership_bytes.items()},
        )
        if optimizer_layer_keys or optimizer_common_keys:
            logger.info(
                "FRCheck save optimizer key summary: layer=%s common=%s",
                _summarize_optimizer_keys(optimizer_layer_keys),
                _summarize_optimizer_keys(optimizer_common_keys),
            )
    return result


def _synchronize_layer_groups(layer_groups: List[_LayerGroup]) -> List[_LayerGroup]:
    """Ensure all ranks in an FRCheck group iterate the same layer indexes."""
    if not torch.distributed.is_initialized():
        return layer_groups

    manager = FRCheckManager()
    group_members = manager.group_member_ranks
    if not group_members:
        return layer_groups

    rank = torch.distributed.get_rank()
    local_layer_idxs = [group.layer_idx for group in layer_groups]
    all_layer_idxs: List[Optional[List[int]]] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(all_layer_idxs, local_layer_idxs)

    union_layer_idxs: Set[int] = set(local_layer_idxs)
    for member_rank in group_members:
        member_layer_idxs = all_layer_idxs[member_rank]
        if member_layer_idxs:
            union_layer_idxs.update(int(layer_idx) for layer_idx in member_layer_idxs)

    groups_by_layer = {group.layer_idx: group for group in layer_groups}
    added_empty = 0
    for layer_idx in union_layer_idxs:
        if layer_idx not in groups_by_layer:
            groups_by_layer[layer_idx] = _LayerGroup(
                layer_idx=layer_idx,
                tensor_infos=[],
                tensor_data=[],
                total_bytes=0,
            )
            added_empty += 1

    result = sorted(
        groups_by_layer.values(),
        key=lambda group: (0 if group.layer_idx < 0 else 1, group.layer_idx),
    )
    if added_empty and _frcheck_debug_enabled():
        logger.info(
            "FRCheck save grouping: rank=%d added %d empty layer groups for "
            "group-synchronized layer order: %s",
            rank,
            added_empty,
            [group.layer_idx for group in result],
        )
    return result


def _exchange_frcheck_group_metadata(
    tensor_infos: List[Any],
    layer_order: List[str],
    layer_metadata: Dict[str, Dict[str, Any]],
    optimizer_layer_map: Optional[Dict[str, int]] = None,
) -> Tuple[
    Dict[int, List[Any]],
    Dict[int, List[str]],
    Dict[int, Dict[str, Dict[str, Any]]],
    Dict[int, Dict[str, int]],
]:
    """All-gather per-rank tensor + per-layer metadata (keys are global ranks)."""
    local_pkg = {
        "tensor_infos": tensor_infos,
        "layer_order": layer_order,
        "layer_metadata": layer_metadata,
        "optimizer_layer_map": optimizer_layer_map or {},
    }
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        gathered: List[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_pkg)
    else:
        gathered = [local_pkg]

    all_tensor_infos: Dict[int, List[Any]] = {}
    all_layer_order: Dict[int, List[str]] = {}
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
    all_optimizer_layer_maps: Dict[int, Dict[str, int]] = {}
    for r, pkg in enumerate(gathered):
        if pkg is None:
            continue
        all_tensor_infos[r] = pkg.get("tensor_infos", [])
        all_layer_order[r] = pkg.get("layer_order", [])
        all_layer_metadata[r] = pkg.get("layer_metadata", {})
        all_optimizer_layer_maps[r] = pkg.get("optimizer_layer_map", {})
    return all_tensor_infos, all_layer_order, all_layer_metadata, all_optimizer_layer_maps


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




def _prep_layer_phase1(
    manager,
    layer_buf: torch.Tensor,
    layer_mirror: torch.Tensor,
    layer_tensor_size: int,
    n: int,
    num_stripes: int,
    block_size: int,
    my_node: int,
    layer_name: str,
    layer_idx: int,
    _dbg: bool = False,
) -> _LayerEncodeSubmitState:
    """Compute GDR source/mirror pointers into the layer buffer (no scatter copy)."""
    stripe_plans = manager.stripe_plans
    layer_base = int(layer_buf.data_ptr())
    mirror_base = int(layer_mirror.data_ptr())

    data_addrs = [0] * num_stripes
    mirror_addrs = [0] * num_stripes
    actual_sizes = [0] * num_stripes
    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}
    n_filled_blocks = (layer_tensor_size + block_size - 1) // block_size if block_size > 0 else 0
    n_filled_blocks = min(n_filled_blocks, (n - 1) * (n - 2))  # cap at n_src

    for stripe_id in range(num_stripes):
        plan = stripe_plans[stripe_id]
        if plan.role == StripeRole.SOURCE:
            blk_idx = src_block_per_node[my_node]
            src_block_per_node[my_node] += 1
            if blk_idx < n_filled_blocks:
                src_offset = blk_idx * block_size
                data_addrs[stripe_id] = layer_base + src_offset
                mirror_addrs[stripe_id] = mirror_base + src_offset
                actual_sizes[stripe_id] = block_size
            # else: stays 0 → submit and disk write will skip

    if _dbg:
        logger.info(
            "[FRCHECK-DEBUG] %s Phase1 ptrs: layer_bytes=%d block_size=%d "
            "source_stripes=%d",
            layer_name, layer_tensor_size, block_size,
            sum(1 for s in actual_sizes if s > 0),
        )

    return _LayerEncodeSubmitState(
        layer_name=layer_name,
        layer_idx=layer_idx,
        block_size=block_size,
        layer_tensor_size=layer_tensor_size,
        actual_sizes=actual_sizes,
        data_addrs=data_addrs,
        mirror_addrs=mirror_addrs,
        layer_buf_base=layer_base,
        n_filled_blocks=n_filled_blocks,
    )


def _source_blk_idx_for_stripe(
    stripe_plans,
    stripe_id: int,
) -> int:
    """Block index within this rank's SOURCE stripes (matches prep/save order)."""
    if stripe_plans[stripe_id].role != StripeRole.SOURCE:
        return -1
    blk_idx = 0
    for sid in range(stripe_id):
        if stripe_plans[sid].role == StripeRole.SOURCE:
            blk_idx += 1
    return blk_idx


def _source_blk_idx_for_node_stripe(
    stripe_plans,
    stripe_id: int,
    node_id: int,
) -> int:
    """Block index for node_id within its SOURCE stripes up to stripe_id."""
    if stripe_id >= len(stripe_plans):
        return -1
    plan = stripe_plans[stripe_id]
    if node_id not in plan.source_node_ids:
        return -1
    blk_idx = 0
    for sid in range(stripe_id):
        if node_id in stripe_plans[sid].source_node_ids:
            blk_idx += 1
    return blk_idx


def _n_filled_blocks_for_layer(total_bytes: int, block_size: int, n_source: int) -> int:
    if total_bytes <= 0 or block_size <= 0:
        return 0
    return min((int(total_bytes) + int(block_size) - 1) // int(block_size), n_source)


def _build_layer_exchange_plan(
    manager,
    layer_idx: int,
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]], List[int]]:
    """Return packed send/recv block plans and local encoder stripes."""
    my_node = int(manager.rank_in_group) + 1
    send_blocks: Dict[int, List[Tuple[int, int]]] = {}
    recv_blocks: Dict[int, List[Tuple[int, int]]] = {}
    encoder_sids: List[int] = []
    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, manager.frcheck_n + 1)}
    for sid, plan in enumerate(manager.stripe_plans):
        block_indices = {
            src_node: src_block_per_node.get(src_node, 0)
            for src_node in plan.source_node_ids
        }
        if my_node in plan.source_node_ids and plan.encoder_node_id != my_node:
            blk_idx = block_indices[my_node]
            if blk_idx < manager.get_n_filled_for_node(layer_idx, my_node):
                send_blocks.setdefault(plan.encoder_node_id, []).append((sid, blk_idx))
        if plan.role == StripeRole.ENCODER:
            encoder_sids.append(sid)
            for src_node in plan.source_node_ids:
                if src_node == my_node:
                    continue
                blk_idx = block_indices[src_node]
                if blk_idx < manager.get_n_filled_for_node(layer_idx, src_node):
                    recv_blocks.setdefault(src_node, []).append((sid, blk_idx))
        for src_node in plan.source_node_ids:
            src_block_per_node[src_node] = src_block_per_node.get(src_node, 0) + 1
    return send_blocks, recv_blocks, encoder_sids


def _prepare_layer_exchange_network(
    manager,
    native,
    prepared: Dict[str, Any],
    n: int,
    pack_stream,
) -> Dict[str, Any]:
    """Pack one layer's source blocks and create non-started RDMA threads."""
    result = prepared["result"]
    state = prepared["state"]
    layer_bufs = prepared["layer_bufs"]
    layer_idx = int(result.layer_idx)
    block_size = int(result.block_size)
    my_node = int(manager.rank_in_group) + 1
    send_blocks, recv_blocks, encoder_sids = _build_layer_exchange_plan(manager, layer_idx)
    batch_base = int(prepared["batch_id"]) * 1000003
    lane_id = (int(prepared["batch_id"]) - 1) % max(1, int(getattr(manager, "num_stripes", 1)))

    payload_bytes = int(result.n_filled_blocks) * block_size
    if payload_bytes > 0:
        native.mirror_layer(state.layer_buf_base, int(layer_bufs.layer_mirror_cpu.data_ptr()), payload_bytes)

    # Gather each layer's scattered source blocks into a contiguous per-peer CPU
    # staging buffer so the exchange can issue a few large RDMA sends instead of
    # one send per block. The gather D2H copies are queued on `pack_stream` and
    # overlapped; the caller performs a single pack_stream.synchronize() before
    # any send is issued.
    layer_gpu = layer_bufs.layer_buf_gpu
    send_offsets: Dict[int, Dict[int, int]] = {}
    stage_pack_s = 0.0
    stage_pack_bytes = 0
    stage_pack_blocks = 0
    stage_pack_t0 = time.time()
    with torch.cuda.stream(pack_stream):
        for dst_node, blocks in send_blocks.items():
            pack_buf = layer_bufs.send_layer_bufs.get(dst_node)
            if pack_buf is None:
                raise RuntimeError(f"FRCheck layer exchange: missing send buffer for node {dst_node}")
            if int(pack_buf.numel()) < len(blocks) * block_size:
                raise RuntimeError(
                    f"FRCheck layer exchange: send buffer too small for node {dst_node}"
                )
            offsets: Dict[int, int] = {}
            for slot, (sid, blk_idx) in enumerate(blocks):
                dst_start = slot * block_size
                src_start = int(blk_idx) * block_size
                pack_buf[dst_start:dst_start + block_size].copy_(
                    layer_gpu[src_start:src_start + block_size], non_blocking=True
                )
                offsets[sid] = dst_start
                stage_pack_bytes += block_size
                stage_pack_blocks += 1
            send_offsets[dst_node] = offsets
        stage_event = torch.cuda.Event()
        stage_event.record(pack_stream)
    stage_pack_s = time.time() - stage_pack_t0

    recv_offsets: Dict[int, Dict[int, int]] = {}
    for src_node, blocks in recv_blocks.items():
        recv_buf = layer_bufs.remote_layer_bufs.get(src_node)
        if recv_buf is None:
            raise RuntimeError(f"FRCheck layer exchange: missing recv buffer for node {src_node}")
        required_bytes = len(blocks) * block_size
        if int(recv_buf.numel()) < required_bytes:
            raise RuntimeError(
                f"FRCheck layer exchange: recv buffer too small for node {src_node}"
            )
        recv_offsets[src_node] = {
            sid: slot * block_size for slot, (sid, _blk_idx) in enumerate(blocks)
        }

    num_stripes_local = max(1, int(getattr(manager, "num_stripes", 1)))
    # Segments per peer for the aggregated exchange. Staging makes each peer's
    # payload contiguous, so a small segment count yields few large transfers.
    # Keep it symmetric on send/recv so tagged transfers match across the pair.
    send_seg = max(1, int(os.environ.get("FRCHECK_LAYER_EXCHANGE_SEG", "1")))

    def _iter_staged_segments(block_count: int):
        if block_count <= 0:
            return
        nseg = min(send_seg, block_count)
        base = block_count // nseg
        rem = block_count % nseg
        start = 0
        for seg_idx in range(nseg):
            take = base + (1 if seg_idx < rem else 0)
            if take <= 0:
                continue
            yield seg_idx, start, take
            start += take

    send_tasks: List[Dict[str, Any]] = []
    recv_tasks: List[Dict[str, Any]] = []
    for src_node, blocks in sorted(recv_blocks.items()):
        if not blocks:
            continue
        recv_buf = layer_bufs.remote_layer_bufs[src_node]
        for seg_idx, start, take in _iter_staged_segments(len(blocks)):
            seg_lane = (lane_id + seg_idx) % num_stripes_local
            segment_blocks = list(blocks[start:start + take])
            recv_tasks.append({
                "peer_node": src_node,
                "peer_rig": src_node - 1,
                "addr": int(recv_buf.data_ptr()) + start * block_size,
                "size": take * block_size,
                "batch_id": batch_base + src_node * 1009 + my_node + seg_idx * 104729,
                "lane_id": seg_lane,
                "ready_sids": [int(sid) for sid, _blk_idx in segment_blocks],
            })
    for dst_node, blocks in sorted(send_blocks.items()):
        if not blocks:
            continue
        pack_buf = layer_bufs.send_layer_bufs[dst_node]
        for seg_idx, start, take in _iter_staged_segments(len(blocks)):
            seg_lane = (lane_id + seg_idx) % num_stripes_local
            send_tasks.append({
                "peer_node": dst_node,
                "peer_rig": dst_node - 1,
                "addr": int(pack_buf.data_ptr()) + start * block_size,
                "size": take * block_size,
                "batch_id": batch_base + my_node * 1009 + dst_node + seg_idx * 104729,
                "lane_id": seg_lane,
            })

    remaining = len(send_tasks) + len(recv_tasks)
    done_event = threading.Event()
    if remaining == 0:
        done_event.set()

    ctx = {
        "prepared": prepared,
        "send_blocks": send_blocks,
        "recv_blocks": recv_blocks,
        "encoder_sids": encoder_sids,
        "recv_offsets": recv_offsets,
        "send_tasks": send_tasks,
        "recv_tasks": recv_tasks,
        "remaining": remaining,
        "remaining_lock": threading.Lock(),
        "done_event": done_event,
        "errors": [],
        "start_time": time.time(),
        "stage_event": stage_event,
        "stage_pack_s": stage_pack_s,
        "stage_pack_bytes": stage_pack_bytes,
        "stage_pack_blocks": stage_pack_blocks,
    }
    encode_specs = _build_layer_exchange_encode_specs(manager, ctx, n)
    ready_queue: queue.Queue = queue.Queue()
    ready_pending_remote = {
        int(sid): set(spec["remote_deps"]) for sid, spec in encode_specs.items()
    }
    ready_queued_sids: Set[int] = set()
    for sid, pending in ready_pending_remote.items():
        if not pending:
            ready_queue.put(int(sid))
            ready_queued_sids.add(int(sid))
    ctx.update({
        "encode_specs": encode_specs,
        "ready_queue": ready_queue,
        "ready_lock": threading.Lock(),
        "ready_pending_remote": ready_pending_remote,
        "ready_queued_sids": ready_queued_sids,
        "encoded_sids": set(),
        "encode_done_event": threading.Event(),
        "ready_encode_batches": 0,
        "ready_first_s": 0.0,
        "ready_last_s": 0.0,
        "stream_encode_active_s": 0.0,
        "encode_batch_size": max(1, int(os.environ.get("FRCHECK_LAYER_ENCODE_BATCH", "1"))),
    })
    return ctx


def _build_layer_exchange_encode_specs(
    manager, ctx: Dict[str, Any], n: int
) -> Dict[int, Dict[str, Any]]:
    """Build per-stripe encode inputs and remote dependencies for layer exchange."""
    prepared = ctx["prepared"]
    result = prepared["result"]
    state = prepared["state"]
    layer_bufs = prepared["layer_bufs"]
    layer_idx = int(result.layer_idx)
    block_size = int(result.block_size)
    my_node = int(manager.rank_in_group) + 1
    encoder_sids = set(ctx["encoder_sids"])
    recv_offsets = ctx["recv_offsets"]

    specs: Dict[int, Dict[str, Any]] = {}
    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}
    for sid, plan in enumerate(manager.stripe_plans):
        source_block_indices: List[int] = []
        for src_node in plan.source_node_ids:
            source_block_indices.append(src_block_per_node.get(src_node, 0))
        for src_node in plan.source_node_ids:
            src_block_per_node[src_node] = src_block_per_node.get(src_node, 0) + 1
        if sid not in encoder_sids:
            continue
        p1b = layer_bufs.parity1_bufs[sid]
        p2b = layer_bufs.parity2_bufs[sid]
        if p1b is None or p2b is None:
            continue

        data_addrs: List[int] = []
        remote_deps: Set[int] = set()
        for src_node, blk_idx in zip(plan.source_node_ids, source_block_indices):
            nf = manager.get_n_filled_for_node(layer_idx, src_node)
            if blk_idx >= nf:
                data_addrs.append(int(layer_bufs.zero_block.data_ptr()))
            elif src_node == my_node:
                data_addrs.append(state.layer_buf_base + blk_idx * block_size)
            else:
                remote_buf = layer_bufs.remote_layer_bufs.get(src_node)
                off = recv_offsets.get(src_node, {}).get(sid)
                if remote_buf is None or off is None:
                    raise RuntimeError(
                        f"FRCheck layer exchange: missing packed data for node {src_node} stripe {sid}"
                    )
                data_addrs.append(int(remote_buf.data_ptr()) + off)
                remote_deps.add(int(src_node))
        specs[sid] = {
            "data_addrs": data_addrs,
            "p1_addr": int(p1b.data_ptr()),
            "p2_addr": int(p2b.data_ptr()),
            "remote_deps": remote_deps,
            "block_size": block_size,
        }
    return specs


def _encode_layer_exchange_stripes(
    native, ctx: Dict[str, Any], stripe_ids: List[int]
) -> None:
    """Encode a ready subset of local encoder stripes."""
    if not stripe_ids:
        return
    specs = ctx["encode_specs"]
    data_addrs: List[int] = []
    p1_addrs: List[int] = []
    p2_addrs: List[int] = []
    for sid in stripe_ids:
        spec = specs[int(sid)]
        data_addrs.extend(spec["data_addrs"])
        p1_addrs.append(int(spec["p1_addr"]))
        p2_addrs.append(int(spec["p2_addr"]))
    native.encode_layer_stripes(
        [int(sid) for sid in stripe_ids],
        data_addrs,
        p1_addrs,
        p2_addrs,
        int(ctx["prepared"]["result"].block_size),
    )


def _encode_layer_exchange_context(manager, native, ctx: Dict[str, Any], n: int) -> float:
    """Encode local encoder stripes after all layer-exchange network tasks finish."""
    if "encode_specs" not in ctx:
        ctx["encode_specs"] = _build_layer_exchange_encode_specs(manager, ctx, n)
    stripe_ids = sorted(int(sid) for sid in ctx["encode_specs"].keys())
    _encode_layer_exchange_stripes(native, ctx, stripe_ids)
    return time.time() - float(ctx["start_time"])


def _save_frcheck_stripe_files(
    manager,
    output_dir: str,
    rank: int,
    num_stripes: int,
    encode_results: List[_LayerEncodeResult],
    include_encoder_p1: bool = True,
    include_source: bool = True,
    include_p2: bool = True,
) -> None:
    """Write all layer stripe files after encode completes."""
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

    # Job tuple: (layer_name, block_size, sid, role, ncopy, buf, suffix)
    jobs: List[Tuple] = []
    for result in encode_results:
        layer_bufs = manager.get_layer_stripe_bufs(result.layer_idx)
        block_size = result.block_size
        layer_name = result.layer_name
        layer_mirror = layer_bufs.layer_mirror_cpu
        for sid in range(num_stripes):
            plan = stripe_plans[sid]
            if plan.role == StripeRole.SOURCE:
                if not include_source:
                    continue
                blk_idx = _source_blk_idx_for_stripe(stripe_plans, sid)
                if blk_idx < 0 or blk_idx >= result.n_filled_blocks:
                    continue
                src_buf = layer_mirror[
                    blk_idx * block_size : (blk_idx + 1) * block_size
                ]
                jobs.append((
                    layer_name, block_size, sid, 0, block_size,
                    src_buf, "",
                ))
            elif plan.role == StripeRole.ENCODER:
                if not include_encoder_p1:
                    continue
                jobs.append((
                    layer_name, block_size, sid, 1, block_size,
                    layer_bufs.parity1_bufs[sid], "_p1",
                ))
            elif plan.role == StripeRole.PARITY_TARGET:
                if not include_p2:
                    continue
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


def wait_for_frcheck_parity_flush() -> None:
    """Drain FRCheck save-side parity work before in-process recovery."""
    _wait_previous_async_writers(debug=_frcheck_debug_enabled())
    native = FRCheckManager().get_native()
    if native is not None and hasattr(native, "wait_parity_flush"):
        native.wait_parity_flush()


def _async_write_frcheck_encoder_p1_files(
    manager,
    output_dir: str,
    rank: int,
    num_stripes: int,
    encode_results: List[_LayerEncodeResult],
) -> None:
    """Wait for async P1 delivery, then write encoder-owned P1 shards."""
    native = manager.get_native()
    if native is None:
        return
    native.wait_parity_flush()
    _save_frcheck_stripe_files(
        manager,
        output_dir,
        rank,
        num_stripes,
        encode_results,
        include_encoder_p1=True,
        include_source=False,
        include_p2=False,
    )


def _write_frcheck_aggregate_p2_files(
    output_dir: str,
    rank: int,
    segments: List[Dict[str, Any]],
) -> None:
    """Write received aggregate P2 slices using the existing FRBK layout."""
    import concurrent.futures

    def _write(segment: Dict[str, Any]) -> None:
        stripe_dir = Path(output_dir) / segment["layer_name"] / f"stripe_{segment['sid']}"
        stripe_dir.mkdir(parents=True, exist_ok=True)
        path = stripe_dir / f"frcheck_shard_rank{rank}.pt"
        size = int(segment["size"])
        block_size = int(segment["block_size"])
        header = struct.pack("<4sIIQQ", b"FRBK", int(segment["sid"]), 2, size, block_size)
        data = segment["buffer"][int(segment["offset"]):int(segment["offset"]) + size]
        with open(path, "wb") as file:
            file.write(header)
            if size > 0:
                file.write(memoryview(data.numpy()))

    if not segments:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(segments))) as executor:
        futures = [executor.submit(_write, segment) for segment in segments]
        for future in futures:
            future.result()


def _finish_aggregate_p2_context(context: Dict[str, Any], debug: bool = False) -> None:
    """Wait for aggregate P2, write received shards, and release registered buffers."""
    manager = context["manager"]
    native = manager.get_native()
    if native is None:
        raise RuntimeError("FRCheck aggregate P2 native module is unavailable")
    try:
        wait_start = time.time()
        native.wait_aggregate_p2(300)
        if debug:
            logger.info(
                "FRCHECK aggregate P2 wait rank %d: generation=%d elapsed=%.3fs",
                context["rank"], context["generation"], time.time() - wait_start,
            )
        if context["write_to_disk"]:
            _write_frcheck_aggregate_p2_files(
                context["output_dir"], context["rank"], context["recv_segments"],
            )
    finally:
        manager.release_registered_save_buffers(context["buffers"])


def _async_finish_aggregate_p2_context(context: Dict[str, Any], debug: bool = False) -> None:
    global _async_writer_error
    try:
        _finish_aggregate_p2_context(context, debug=debug)
    except BaseException as exc:
        _async_writer_error = exc
        logger.exception("FRCheck async aggregate P2 writer failed")


def _wait_previous_async_writers(debug: bool = False, rank: int = -1) -> None:
    global _async_p1_writer_thread, _async_p2_writer_thread, _async_writer_error
    for attr in ("_async_p1_writer_thread", "_async_p2_writer_thread"):
        thread = globals()[attr]
        if thread is not None:
            start = time.time()
            if debug:
                logger.info("FRCHECK async writer rank %d: joining previous %s", rank, thread.name)
            thread.join()
            globals()[attr] = None
            if debug:
                logger.info(
                    "FRCHECK async writer rank %d: joined previous %s in %.3fs",
                    rank, thread.name, time.time() - start,
                )
    if _async_writer_error is not None:
        error = _async_writer_error
        _async_writer_error = None
        raise RuntimeError("FRCheck asynchronous writer failed") from error


def _start_async_p1_writer(
    manager,
    output_dir: str,
    rank: int,
    num_stripes: int,
    encode_results: List[_LayerEncodeResult],
) -> None:
    global _async_p1_writer_thread
    _wait_previous_async_writers()
    _async_p1_writer_thread = threading.Thread(
        target=_async_write_frcheck_encoder_p1_files,
        args=(manager, output_dir, rank, num_stripes, list(encode_results)),
        name="frcheck-async-p1-writer",
        daemon=False,
    )
    _async_p1_writer_thread.start()


def _start_async_p2_writer(context: Dict[str, Any], debug: bool = False) -> None:
    global _async_p2_writer_thread
    _wait_previous_async_writers(debug=debug, rank=int(context["rank"]))
    _async_p2_writer_thread = threading.Thread(
        target=_async_finish_aggregate_p2_context,
        args=(context, debug),
        name="frcheck-async-p2-writer",
        daemon=False,
    )
    _async_p2_writer_thread.start()


def _build_and_submit_aggregate_p2(
    manager,
    native,
    encode_results: List[_LayerEncodeResult],
    output_dir: str,
    rank: int,
    write_to_disk: bool,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Pack all P2 blocks into one registered buffer per physical peer/lane."""
    global _p2_save_generation
    _p2_save_generation += 1
    generation = _p2_save_generation
    send_groups: Dict[Tuple[int, int], List[Tuple[_LayerEncodeResult, int, torch.Tensor]]] = {}
    recv_groups: Dict[Tuple[int, int], List[Tuple[_LayerEncodeResult, int]]] = {}
    ordered_results = sorted(encode_results, key=lambda result: (result.layer_idx, result.layer_name))
    for result in ordered_results:
        layer_bufs = manager.get_layer_stripe_bufs(result.layer_idx)
        for sid in range(native.num_stripes()):
            route = native.get_p2_route(sid)
            role, peer, lane = int(route[0]), int(route[1]), int(route[2])
            if role == int(StripeRole.ENCODER):
                source = layer_bufs.parity2_bufs[sid]
                if source is None:
                    raise RuntimeError(f"FRCheck aggregate P2 missing encoder buffer for stripe {sid}")
                send_groups.setdefault((peer, lane), []).append((result, sid, source))
            elif role == int(StripeRole.PARITY_TARGET):
                recv_groups.setdefault((peer, lane), []).append((result, sid))

    buffers: List[torch.Tensor] = []
    send_tasks: List[Tuple[int, int, int, int]] = []
    recv_tasks: List[Tuple[int, int, int, int]] = []
    recv_segments: List[Dict[str, Any]] = []
    total_send_bytes = 0
    total_recv_bytes = 0
    for (peer, lane), segments in sorted(send_groups.items()):
        size = sum(int(result.block_size) for result, _sid, _source in segments)
        buffer = manager.allocate_registered_save_buffer(size)
        buffers.append(buffer)
        offset = 0
        for result, _sid, source in segments:
            block_size = int(result.block_size)
            buffer[offset:offset + block_size].copy_(source[:block_size])
            offset += block_size
        send_tasks.append((peer, lane, int(buffer.data_ptr()), size))
        total_send_bytes += size
    for (peer, lane), segments in sorted(recv_groups.items()):
        size = sum(int(result.block_size) for result, _sid in segments)
        buffer = manager.allocate_registered_save_buffer(size)
        buffers.append(buffer)
        offset = 0
        for result, sid in segments:
            block_size = int(result.block_size)
            recv_segments.append({
                "buffer": buffer,
                "offset": offset,
                "size": block_size,
                "block_size": block_size,
                "layer_name": result.layer_name,
                "layer_idx": result.layer_idx,
                "sid": sid,
                "peer": peer,
                "lane": lane,
            })
            offset += block_size
        recv_tasks.append((peer, lane, int(buffer.data_ptr()), size))
        total_recv_bytes += size
    try:
        counts = native.submit_aggregate_p2(send_tasks, recv_tasks, generation)
    except BaseException:
        manager.release_registered_save_buffers(buffers)
        raise
    context = {
        "manager": manager,
        "output_dir": output_dir,
        "rank": rank,
        "generation": generation,
        "buffers": buffers,
        "recv_segments": recv_segments,
        "write_to_disk": write_to_disk,
    }
    summary = {
        "send_tasks": int(counts[0]),
        "recv_tasks": int(counts[1]),
        "send_bytes": total_send_bytes,
        "recv_bytes": total_recv_bytes,
    }
    return context, summary


def save_frcheck_legacy_checkpoint(
    state_dict: Dict[str, Any], checkpoint_name: str, write_to_disk: bool = True
) -> None:
    """Write frcheck_main_rank*.pt + layer-wise source/parity shards."""
    t0 = time.time()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    _dbg = _frcheck_debug_enabled()

    from megatron.training import get_args
    args = get_args()
    _dbg = getattr(args, "frcheck_debug", False)

    optimizer_layer_map = _build_optimizer_layer_map(state_dict)

    # 1. Flatten on a shallow save copy only — mutating live optimizer layout would
    #    duplicate fp32 tensors on iter>=2 and break layer routing below.
    t_decompose = time.time()
    decomposed, save_copy_s, save_flatten_s, save_decompose_s = decompose_state_dict_for_save(state_dict)
    save_pre_group_s = time.time() - t_decompose
    total_tensor_size = decomposed.total_tensor_size_bytes
    if _dbg:
        logger.info(
            "FRCheck save: rank=%d total_tensor_size=%d n_tensors=%d",
            rank, total_tensor_size, len(decomposed.tensor_data),
        )

    if _dbg:
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
    native.set_debug(_dbg)
    if hasattr(native, "reset_ft_timing_stats"):
        native.reset_ft_timing_stats()
    if hasattr(native, "start_mirror_worker"):
        native.start_mirror_worker()

    # Save-scoped aggregate buffers cannot be reused until the previous writer exits.
    _use_async_parity = getattr(args, 'frcheck_async_parity', False)
    _wait_previous_async_writers(debug=_dbg, rank=rank)

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

    # 3. Setup output directory (same layout as ecnaive: files live in mp_rank_* dir)
    checkpoint_path = Path(checkpoint_name)
    checkpoint_dir = checkpoint_path if checkpoint_path.suffix == "" else checkpoint_path.parent
    if write_to_disk:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 4. Group tensors by layer index
    t0 = time.time()
    layer_groups = _group_by_layer(
        decomposed,
        distribute_common=getattr(args, "frcheck_distribute_common", False),
        debug=_dbg,
        optimizer_layer_map=optimizer_layer_map,
        optimizer_model_key_map=_build_optimizer_model_key_map(state_dict),
    )
    layer_groups = _synchronize_layer_groups(layer_groups)
    save_group_s = time.time() - t0
    n_tensors = len(decomposed.tensor_data)
    del decomposed.tensor_data  # GPU refs now held by per-layer groups
    num_layers = len(layer_groups)
    if _dbg:
        logger.info(
            "FRCheck save: grouped %d layers from %d tensors (layers: %s)",
            num_layers, n_tensors,
            [(g.layer_idx, g.total_bytes) for g in layer_groups],
        )

    flat_key_roots = decomposed.flat_key_roots

    if _dbg:
        logger.info(f"FRCHECK save timing: layer group+manager init {time.time()-t0:.3f}s")

    # 4.5 Compute per-layer adaptive block sizes (first save, cached thereafter)
    manager.compute_layer_block_sizes(layer_groups)

    if _dbg:
        n_src_total = (n - 1) * (n - 2)
        n_source_my = sum(1 for p in stripe_plans if p.role == StripeRole.SOURCE)
        n_encoder_my = sum(1 for p in stripe_plans if p.role == StripeRole.ENCODER)
        n_parity_my = sum(1 for p in stripe_plans if p.role == StripeRole.PARITY_TARGET)
        logger.info(
            "[FRCHECK-DEBUG] rank=%d n=%d stripes=%d roles(src=%d enc=%d par=%d) gdr=True",
            rank, n, num_stripes, n_source_my, n_encoder_my, n_parity_my,
        )
        total_cap = 0
        total_data = 0
        for g in layer_groups:
            lidx = g.layer_idx
            blk = manager._layer_block_sizes[lidx]
            cap = blk * n_source_my
            pad = max(0, cap - g.total_bytes)
            pct = 100.0 * pad / cap if cap > 0 else 0
            lname = f"layer_{lidx}" if lidx >= 0 else "layer_common"
            ns = sum(1 for p in stripe_plans if p.role == StripeRole.SOURCE)
            ne = sum(1 for p in stripe_plans if p.role == StripeRole.ENCODER)
            nt = sum(1 for p in stripe_plans if p.role == StripeRole.PARITY_TARGET)
            disk_est = (ns + ne + nt) * blk  # EC expansion: src data + P1 + P2
            logger.info(
                "[FRCHECK-DEBUG] %s: data=%.1fMB cap=%.1fMB blk=%.1fMB pad=%.1fMB(%d%%) "
                "disk_est=%.1fMB (src=%d+P1=%d+P2=%d stripes)",
                lname,
                g.total_bytes / 1e6, cap / 1e6, blk / 1e6,
                pad / 1e6, int(pct),
                disk_est / 1e6, ns, ne, nt,
            )
            total_cap += cap
            total_data += g.total_bytes
        total_pad = total_cap - total_data
        total_pct = 100.0 * total_pad / total_cap if total_cap > 0 else 0
        logger.info(
            "[FRCHECK-DEBUG] rank=%d TOTAL: data=%.1fMB cap=%.1fMB pad=%.1fMB(%.0f%%)",
            rank, total_data / 1e6, total_cap / 1e6,
            total_pad / 1e6, total_pct,
        )

    # 5. Per-layer buffers were allocated and registered by the manager when
    # adaptive block sizes were computed. Avoid a second ibv_reg_mr pass here.
    trace_save = _dbg or os.environ.get("FRCHECK_TRACE_INIT", "0") == "1"
    if trace_save:
        logger.info("FRCHECK save trace rank=%d: layer buffers already registered", rank)

    if world_size > 1:
        if trace_save:
            logger.info("FRCHECK save trace rank=%d: before pre-encode barrier", rank)
        torch.distributed.barrier()
        if trace_save:
            logger.info("FRCHECK save trace rank=%d: after pre-encode barrier", rank)
    e2e_t0 = time.time()

    # Pipeline: pack + encode per layer
    local_layer_order: List[str] = []
    local_layer_metadata: Dict[str, Dict[str, Any]] = {}
    distributed_common_keys = set()
    encode_results: List[_LayerEncodeResult] = []
    prep_stream = torch.cuda.Stream()
    pack_total_s = 0.0
    phase1_total_s = 0.0
    submit_total_s = 0.0
    wait_total_s = 0.0
    noncontig_total = 0
    model_layer_bytes_total = 0
    optimizer_layer_bytes_total = 0
    common_bytes_total = 0
    has_encode_batch = (
        hasattr(native, "submit_source_with_batch")
        and hasattr(native, "submit_enc_recv_with_batch")
        and hasattr(native, "skip_source_batch")
    )
    use_layer_exchange_encode = bool(
        getattr(args, "frcheck_layer_exchange_encode", False)
        and hasattr(native, "send_layer_to_peer")
        and hasattr(native, "recv_layer_from_peer")
        and hasattr(native, "encode_layer_stripes")
        and hasattr(native, "mirror_layer")
    )
    use_cross_layer_encode = (
        has_encode_batch and hasattr(native, "reset_encode_layer")
        and not use_layer_exchange_encode
    )

    prepared_layers: List[Dict[str, Any]] = []

    # Stage A: pack and prepare all layers before launching network/encode.
    for encode_batch_id, group in enumerate(layer_groups, start=1):
        layer_name = f"layer_{group.layer_idx}" if group.layer_idx >= 0 else "layer_common"
        layer_idx = group.layer_idx
        layer_block_size = manager._layer_block_sizes[layer_idx]
        layer_bufs = manager.get_layer_stripe_bufs(layer_idx)

        layer_buf_gpu = layer_bufs.layer_buf_gpu
        pack_t0 = time.time()
        if trace_save:
            logger.info(
                "FRCHECK save trace rank=%d: pack begin batch=%d %s bytes=%.2fMB block=%.2fMB",
                rank, encode_batch_id, layer_name, group.total_bytes / 1e6,
                layer_block_size / 1e6,
            )

        offset = 0
        layer_noncontig = 0
        layer_model_bytes = 0
        layer_optimizer_bytes = 0
        layer_common_bytes = 0
        with torch.cuda.stream(prep_stream):
            for info, tensor in zip(group.tensor_infos, group.tensor_data):
                if not tensor.is_contiguous():
                    layer_noncontig += 1
                ownership = _classify_frcheck_tensor(info.key, optimizer_layer_map)
                info_size = int(getattr(info, "size_bytes", 0))
                if ownership.kind == "model_layer":
                    layer_model_bytes += info_size
                elif ownership.kind == "optimizer_layer":
                    layer_optimizer_bytes += info_size
                else:
                    layer_common_bytes += info_size
                tensor_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
                nbytes = tensor_view.numel()
                layer_buf_gpu[offset : offset + nbytes].copy_(
                    tensor_view, non_blocking=True
                )
                info.offset = offset
                offset += nbytes

            # Only clear padding that can be transmitted in the final used block.
            if offset > 0 and layer_block_size > 0:
                tail_end = ((offset + layer_block_size - 1) // layer_block_size) * layer_block_size
                tail_end = min(tail_end, int(layer_buf_gpu.numel()))
                if tail_end > offset:
                    layer_buf_gpu[offset:tail_end].zero_()
        prep_stream.synchronize()
        layer_pack_s = time.time() - pack_t0
        pack_total_s += layer_pack_s
        noncontig_total += layer_noncontig
        model_layer_bytes_total += layer_model_bytes
        optimizer_layer_bytes_total += layer_optimizer_bytes
        common_bytes_total += layer_common_bytes

        group.tensor_data = []

        phase1_t0 = time.time()
        state = _prep_layer_phase1(
            manager, layer_buf_gpu, layer_bufs.layer_mirror_cpu,
            group.total_bytes, n, num_stripes, layer_block_size, my_node,
            layer_name, layer_idx, _dbg,
        )
        layer_phase1_s = time.time() - phase1_t0
        phase1_total_s += layer_phase1_s

        # Pre-compute encoder active masks for this layer.
        src_blk_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}
        enc_active_masks: Dict[int, List[int]] = {}
        for sid in range(num_stripes):
            plan = stripe_plans[sid]
            if plan.role == StripeRole.ENCODER:
                mask = []
                for src_node in plan.source_node_ids:
                    b = src_blk_per_node.get(src_node, 0)
                    nf = manager.get_n_filled_for_node(layer_idx, src_node)
                    mask.append(1 if b < nf else 0)
                enc_active_masks[sid] = mask
            for src_node in plan.source_node_ids:
                src_blk_per_node[src_node] = src_blk_per_node.get(src_node, 0) + 1

        result = _LayerEncodeResult(
            layer_name=layer_name,
            layer_idx=layer_idx,
            block_size=layer_block_size,
            actual_sizes=state.actual_sizes,
            tensor_infos=group.tensor_infos,
            total_bytes=group.total_bytes,
            n_filled_blocks=state.n_filled_blocks,
        )
        encode_results.append(result)
        prepared_layers.append({
            "batch_id": int(encode_batch_id),
            "group": group,
            "result": result,
            "state": state,
            "layer_bufs": layer_bufs,
            "enc_active_masks": enc_active_masks,
            "pack_s": layer_pack_s,
            "phase1_s": layer_phase1_s,
            "noncontig": layer_noncontig,
            "model_bytes": layer_model_bytes,
            "optimizer_bytes": layer_optimizer_bytes,
            "common_bytes": layer_common_bytes,
        })
        if trace_save:
            logger.info(
                "FRCHECK save trace rank=%d: pack done batch=%d %s pack=%.4fs phase1=%.4fs",
                rank, encode_batch_id, layer_name, layer_pack_s, layer_phase1_s,
            )

    # Stage B: submit prepared layers to native workers.
    if use_cross_layer_encode:
        native.reset_encode_layer()

    layer_submit_times: Dict[int, float] = {
        int(prepared["batch_id"]): 0.0 for prepared in prepared_layers
    }

    def _submit_prepared_sid(prepared: Dict[str, Any], sid: int) -> None:
        encode_batch_id = int(prepared["batch_id"])
        state = prepared["state"]
        layer_bufs = prepared["layer_bufs"]
        enc_active_masks = prepared["enc_active_masks"]
        layer_block_size = int(prepared["result"].block_size)
        plan = stripe_plans[sid]

        if plan.role == StripeRole.SOURCE:
            addr = state.data_addrs[sid]
            if addr == 0:
                if use_cross_layer_encode:
                    native.skip_source_batch(sid, encode_batch_id)
                return
            if use_cross_layer_encode:
                native.submit_source_with_batch(
                    sid, addr, state.mirror_addrs[sid],
                    layer_block_size, encode_batch_id,
                )
            else:
                native.submit_source(sid, addr, state.mirror_addrs[sid], layer_block_size)
        elif plan.role == StripeRole.ENCODER:
            rb = layer_bufs.recv_bufs[sid]
            p1b = layer_bufs.parity1_bufs[sid]
            p2b = layer_bufs.parity2_bufs[sid]
            if rb is None or p1b is None or p2b is None:
                return
            if use_cross_layer_encode:
                native.submit_enc_recv_with_batch(
                    sid, rb.data_ptr(), p1b.data_ptr(), p2b.data_ptr(),
                    layer_block_size, enc_active_masks.get(sid, []),
                    encode_batch_id,
                )
            else:
                native.submit_enc_recv(
                    sid, rb.data_ptr(), p1b.data_ptr(), p2b.data_ptr(),
                    layer_block_size, enc_active_masks.get(sid, []),
                )
        elif plan.role == StripeRole.PARITY_TARGET:
            # Deferred to async phase (after all layers' encoding).
            return

    if use_layer_exchange_encode:
        pack_stream = torch.cuda.Stream()
        layer_exchange_contexts: List[Dict[str, Any]] = []
        for prepared in prepared_layers:
            layer_exchange_contexts.append(
                _prepare_layer_exchange_network(manager, native, prepared, n, pack_stream)
            )
        # No global stage sync: each layer's send waits on its own CUDA event so
        # gather D2H overlaps with network of already-staged layers.
        lx_stage_sync_s = 0.0

        worker_errors: List[BaseException] = []
        worker_errors_lock = threading.Lock()
        send_queues: Dict[Tuple[int, int], queue.Queue] = {}
        recv_queues: Dict[Tuple[int, int], queue.Queue] = {}

        def _mark_task_done(ctx: Dict[str, Any]) -> None:
            with ctx["remaining_lock"]:
                ctx["remaining"] -= 1
                if ctx["remaining"] == 0:
                    ctx["done_event"].set()

        def _record_worker_error(ctx: Dict[str, Any], exc: BaseException) -> None:
            ctx["errors"].append(exc)
            ctx["done_event"].set()
            with worker_errors_lock:
                worker_errors.append(exc)

        def _mark_ready_stripes(ctx: Dict[str, Any], src_node: int, sids: List[int]) -> None:
            if not sids:
                return
            now_s = time.time() - float(ctx["start_time"])
            with ctx["ready_lock"]:
                for sid in sids:
                    sid = int(sid)
                    pending = ctx["ready_pending_remote"].get(sid)
                    if pending is None:
                        continue
                    pending.discard(int(src_node))
                    if pending or sid in ctx["ready_queued_sids"] or sid in ctx["encoded_sids"]:
                        continue
                    ctx["ready_queue"].put(sid)
                    ctx["ready_queued_sids"].add(sid)
                    if float(ctx.get("ready_first_s", 0.0)) == 0.0:
                        ctx["ready_first_s"] = now_s
                    ctx["ready_last_s"] = now_s

        def _send_worker(peer_key: Tuple[int, int], q: queue.Queue) -> None:
            while True:
                item = q.get()
                if item is None:
                    return
                ctx, task = item
                try:
                    # Send from the CPU staging buffer only once this layer's
                    # gather D2H has completed; this overlaps later layers' D2H
                    # with earlier layers' network transfers.
                    stage_event = ctx.get("stage_event")
                    if stage_event is not None:
                        stage_event.synchronize()
                    native.send_layer_to_peer(
                        int(task["peer_rig"]), int(task["addr"]),
                        int(task["size"]), int(task["batch_id"]), int(task["lane_id"]),
                    )
                    _mark_task_done(ctx)
                except BaseException as exc:
                    _record_worker_error(ctx, exc)

        def _recv_worker(peer_key: Tuple[int, int], q: queue.Queue) -> None:
            while True:
                item = q.get()
                if item is None:
                    return
                ctx, task = item
                try:
                    native.recv_layer_from_peer(
                        int(task["peer_rig"]), int(task["addr"]),
                        int(task["size"]), int(task["batch_id"]), int(task["lane_id"]),
                    )
                    _mark_ready_stripes(ctx, int(task["peer_node"]), task.get("ready_sids", []))
                    _mark_task_done(ctx)
                except BaseException as exc:
                    _record_worker_error(ctx, exc)

        lx_queue_setup_t0 = time.time()
        for ctx in layer_exchange_contexts:
            for task in ctx["send_tasks"]:
                key = (int(task["peer_node"]), int(task["lane_id"]))
                send_queues.setdefault(key, queue.Queue())
            for task in ctx["recv_tasks"]:
                key = (int(task["peer_node"]), int(task["lane_id"]))
                recv_queues.setdefault(key, queue.Queue())
        lx_queue_setup_s = time.time() - lx_queue_setup_t0

        exchange_workers: List[threading.Thread] = []
        lx_exchange_thread_start_t0 = time.time()
        for peer_key, q in sorted(send_queues.items()):
            th = threading.Thread(target=_send_worker, args=(peer_key, q), name=f"frcheck-layer-send-peer{peer_key[0]}-lane{peer_key[1]}")
            th.start()
            exchange_workers.append(th)
        for peer_key, q in sorted(recv_queues.items()):
            th = threading.Thread(target=_recv_worker, args=(peer_key, q), name=f"frcheck-layer-recv-peer{peer_key[0]}-lane{peer_key[1]}")
            th.start()
            exchange_workers.append(th)
        lx_exchange_thread_start_s = time.time() - lx_exchange_thread_start_t0

        encode_errors: List[BaseException] = []
        encode_threads: List[threading.Thread] = []

        def _wait_and_encode(ctx: Dict[str, Any]) -> None:
            try:
                wait_t0 = time.time()
                encode_t0 = time.time()
                if os.environ.get("FRCHECK_LAYER_STREAM_ENCODE", "1") == "0":
                    ctx["done_event"].wait()
                    ctx["exchange_wait_s"] = time.time() - wait_t0
                    if ctx["errors"]:
                        raise RuntimeError("FRCheck layer exchange failed") from ctx["errors"][0]
                    ctx["layer_elapsed"] = _encode_layer_exchange_context(manager, native, ctx, n)
                else:
                    total_stripes = len(ctx.get("encode_specs", {}))
                    batch_size = int(ctx.get("encode_batch_size", 1))
                    while len(ctx["encoded_sids"]) < total_stripes:
                        if ctx["errors"]:
                            raise RuntimeError("FRCheck layer exchange failed") from ctx["errors"][0]
                        batch: List[int] = []
                        try:
                            sid = ctx["ready_queue"].get(timeout=0.01)
                        except queue.Empty:
                            if ctx["done_event"].is_set() and ctx["ready_queue"].empty():
                                if ctx["errors"]:
                                    raise RuntimeError("FRCheck layer exchange failed") from ctx["errors"][0]
                                pending = sorted(
                                    int(sid) for sid in ctx["encode_specs"].keys()
                                    if int(sid) not in ctx["encoded_sids"]
                                )
                                if pending:
                                    raise RuntimeError(
                                        f"FRCheck layer exchange encode incomplete: pending stripes {pending[:8]}"
                                    )
                                break
                            continue
                        batch.append(int(sid))
                        while len(batch) < batch_size:
                            try:
                                batch.append(int(ctx["ready_queue"].get_nowait()))
                            except queue.Empty:
                                break
                        with ctx["ready_lock"]:
                            batch = [
                                int(sid) for sid in batch
                                if int(sid) not in ctx["encoded_sids"]
                            ]
                        if not batch:
                            continue
                        batch_encode_t0 = time.time()
                        _encode_layer_exchange_stripes(native, ctx, batch)
                        batch_encode_s = time.time() - batch_encode_t0
                        with ctx["ready_lock"]:
                            ctx["stream_encode_active_s"] += batch_encode_s
                            ctx["encoded_sids"].update(batch)
                            ctx["ready_encode_batches"] += 1
                            if len(ctx["encoded_sids"]) >= total_stripes:
                                ctx["encode_done_event"].set()
                    ctx["done_event"].wait()
                    ctx["exchange_wait_s"] = time.time() - wait_t0
                    if ctx["errors"]:
                        raise RuntimeError("FRCheck layer exchange failed") from ctx["errors"][0]
                    ctx["layer_elapsed"] = time.time() - float(ctx["start_time"])
                ctx["encode_wall_s"] = float(ctx.get("stream_encode_active_s", time.time() - encode_t0))
                ctx["encode_end_s"] = time.time() - float(ctx["start_time"])
            except BaseException as exc:
                encode_errors.append(exc)

        lx_encode_thread_start_t0 = time.time()
        for ctx in layer_exchange_contexts:
            th = threading.Thread(
                target=_wait_and_encode,
                args=(ctx,),
                name=f"frcheck-layer-encode-{ctx['prepared']['result'].layer_name}",
            )
            th.start()
            encode_threads.append(th)
        lx_encode_thread_start_s = time.time() - lx_encode_thread_start_t0

        lx_queue_put_t0 = time.time()
        for ctx in layer_exchange_contexts:
            for task in ctx["recv_tasks"]:
                recv_queues[(int(task["peer_node"]), int(task["lane_id"]))].put((ctx, task))
            for task in ctx["send_tasks"]:
                send_queues[(int(task["peer_node"]), int(task["lane_id"]))].put((ctx, task))
        for q in send_queues.values():
            q.put(None)
        for q in recv_queues.values():
            q.put(None)
        lx_queue_put_s = time.time() - lx_queue_put_t0

        lx_join_t0 = time.time()
        for th in exchange_workers:
            th.join()
        lx_exchange_join_s = time.time() - lx_join_t0
        lx_encode_join_t0 = time.time()
        for th in encode_threads:
            th.join()
        lx_encode_join_s = time.time() - lx_encode_join_t0
        if worker_errors:
            raise RuntimeError("FRCheck layer exchange worker failed") from worker_errors[0]
        if encode_errors:
            raise RuntimeError("FRCheck layer exchange encode failed") from encode_errors[0]

        layer_exchange_send_tasks = sum(len(ctx["send_tasks"]) for ctx in layer_exchange_contexts)
        layer_exchange_recv_tasks = sum(len(ctx["recv_tasks"]) for ctx in layer_exchange_contexts)
        layer_exchange_send_bytes = sum(
            int(task["size"]) for ctx in layer_exchange_contexts for task in ctx["send_tasks"]
        )
        layer_exchange_recv_bytes = sum(
            int(task["size"]) for ctx in layer_exchange_contexts for task in ctx["recv_tasks"]
        )
        lx_stage_pack_s = sum(float(ctx.get("stage_pack_s", 0.0)) for ctx in layer_exchange_contexts)
        lx_stage_pack_bytes = sum(int(ctx.get("stage_pack_bytes", 0)) for ctx in layer_exchange_contexts)
        lx_stage_pack_blocks = sum(int(ctx.get("stage_pack_blocks", 0)) for ctx in layer_exchange_contexts)
        lx_exchange_wait_sum_s = sum(float(ctx.get("exchange_wait_s", 0.0)) for ctx in layer_exchange_contexts)
        lx_exchange_wait_max_s = max(
            (float(ctx.get("exchange_wait_s", 0.0)) for ctx in layer_exchange_contexts),
            default=0.0,
        )
        lx_encode_wall_sum_s = sum(float(ctx.get("encode_wall_s", 0.0)) for ctx in layer_exchange_contexts)
        lx_encode_wall_max_s = max(
            (float(ctx.get("encode_wall_s", 0.0)) for ctx in layer_exchange_contexts),
            default=0.0,
        )
        lx_ready_encode_batches = sum(
            int(ctx.get("ready_encode_batches", 0)) for ctx in layer_exchange_contexts
        )
        lx_ready_first_s = min(
            (float(ctx.get("ready_first_s", 0.0)) for ctx in layer_exchange_contexts
             if float(ctx.get("ready_first_s", 0.0)) > 0.0),
            default=0.0,
        )
        lx_ready_last_s = max(
            (float(ctx.get("ready_last_s", 0.0)) for ctx in layer_exchange_contexts),
            default=0.0,
        )
        lx_layer_elapsed_max = max(
            (float(ctx.get("layer_elapsed", 0.0)) for ctx in layer_exchange_contexts),
            default=0.0,
        )
        slow_layers = sorted(
            (
                (
                    float(ctx.get("layer_elapsed", 0.0)),
                    ctx["prepared"]["result"].layer_name,
                    float(ctx.get("exchange_wait_s", 0.0)),
                    float(ctx.get("encode_wall_s", 0.0)),
                    float(ctx.get("stage_pack_s", 0.0)),
                    len(ctx["send_tasks"]),
                    len(ctx["recv_tasks"]),
                )
                for ctx in layer_exchange_contexts
            ),
            reverse=True,
        )[:3]
        lx_slow_layers = ";".join(
            f"{name}:total={total:.3f},wait={wait:.3f},encode={enc:.3f},stage={stage:.3f},send={send_n},recv={recv_n}"
            for total, name, wait, enc, stage, send_n, recv_n in slow_layers
        )

        for ctx in layer_exchange_contexts:
            prepared = ctx["prepared"]
            batch_id = int(prepared["batch_id"])
            result = prepared["result"]
            layer_elapsed = float(ctx.get("layer_elapsed", 0.0))
            layer_submit_times[batch_id] += layer_elapsed
            submit_total_s = max(submit_total_s, layer_elapsed)
            if _dbg or trace_save:
                group = prepared["group"]
                send_bytes = sum(len(v) for v in ctx["send_blocks"].values()) * int(result.block_size)
                recv_bytes = sum(len(v) for v in ctx["recv_blocks"].values()) * int(result.block_size)
                logger.info(
                    "FRCHECK save layer-exchange profile rank=%d %s: bytes=%.2fMB pack=%.4fs "
                    "phase1=%.4fs exchange_encode=%.4fs send=%.2fMB recv=%.2fMB noncontig=%d "
                    "model_layer=%.2fMB optimizer_layer=%.2fMB common=%.2fMB",
                    rank, result.layer_name, group.total_bytes / 1e6, prepared["pack_s"],
                    prepared["phase1_s"], layer_elapsed, send_bytes / 1e6, recv_bytes / 1e6,
                    prepared["noncontig"], prepared["model_bytes"] / 1e6,
                    prepared["optimizer_bytes"] / 1e6, prepared["common_bytes"] / 1e6,
                )
    elif use_cross_layer_encode:
        stripe_wave_size = max(1, int(n))
        for stripe_wave_id, stripe_start in enumerate(range(0, num_stripes, stripe_wave_size)):
            stripe_end = min(stripe_start + stripe_wave_size, num_stripes)
            if trace_save:
                logger.info(
                    "FRCHECK save trace rank=%d: submit stripe_wave=%d stripes=[%d,%d) layers=%d",
                    rank, stripe_wave_id, stripe_start, stripe_end, len(prepared_layers),
                )
            for prepared in prepared_layers:
                batch_id = int(prepared["batch_id"])
                submit_t0 = time.time()
                for sid in range(stripe_start, stripe_end):
                    _submit_prepared_sid(prepared, sid)
                layer_submit_times[batch_id] += time.time() - submit_t0
    else:
        for prepared in prepared_layers:
            batch_id = int(prepared["batch_id"])
            result = prepared["result"]
            submit_t0 = time.time()
            native.reset_layer()
            for sid in range(num_stripes):
                _submit_prepared_sid(prepared, sid)
            layer_submit_s = time.time() - submit_t0
            layer_submit_times[batch_id] += layer_submit_s
            submit_total_s += layer_submit_s

            layer_wait_s = 0.0
            wait_t0 = time.time()
            if _dbg:
                logger.info("FRCHECK layer %s: wait_encode_only (P2 deferred)", result.layer_name)
            native.wait_encode_only()
            layer_wait_s = time.time() - wait_t0
            wait_total_s += layer_wait_s

            if _dbg or trace_save:
                group = prepared["group"]
                logger.info(
                    "FRCHECK save layer profile rank=%d %s: bytes=%.2fMB pack=%.4fs "
                    "phase1=%.4fs submit=%.4fs wait=%.4fs noncontig=%d "
                    "model_layer=%.2fMB optimizer_layer=%.2fMB common=%.2fMB",
                    rank, result.layer_name, group.total_bytes / 1e6, prepared["pack_s"],
                    prepared["phase1_s"], layer_submit_s, layer_wait_s, prepared["noncontig"],
                    prepared["model_bytes"] / 1e6, prepared["optimizer_bytes"] / 1e6,
                    prepared["common_bytes"] / 1e6,
                )

    if use_cross_layer_encode:
        submit_total_s = sum(layer_submit_times.values())
        if _dbg or trace_save:
            for prepared in prepared_layers:
                batch_id = int(prepared["batch_id"])
                result = prepared["result"]
                group = prepared["group"]
                logger.info(
                    "FRCHECK save layer profile rank=%d %s: bytes=%.2fMB pack=%.4fs "
                    "phase1=%.4fs submit=%.4fs wait=%.4fs noncontig=%d "
                    "model_layer=%.2fMB optimizer_layer=%.2fMB common=%.2fMB",
                    rank, result.layer_name, group.total_bytes / 1e6, prepared["pack_s"],
                    prepared["phase1_s"], layer_submit_times.get(batch_id, 0.0), 0.0,
                    prepared["noncontig"], prepared["model_bytes"] / 1e6,
                    prepared["optimizer_bytes"] / 1e6, prepared["common_bytes"] / 1e6,
                )

    if use_cross_layer_encode:
        wait_t0 = time.time()
        if _dbg or trace_save:
            logger.info(
                "FRCHECK save: wait_encode_only for cross-layer batch layers=%d",
                len(encode_results),
            )
        native.wait_encode_only()
        wait_total_s += time.time() - wait_t0

    network_encode_s = submit_total_s + wait_total_s

    layer_exchange_send_tasks = locals().get("layer_exchange_send_tasks", 0)
    layer_exchange_recv_tasks = locals().get("layer_exchange_recv_tasks", 0)
    layer_exchange_send_bytes = locals().get("layer_exchange_send_bytes", 0)
    layer_exchange_recv_bytes = locals().get("layer_exchange_recv_bytes", 0)
    lx_stage_sync_s = locals().get("lx_stage_sync_s", 0.0)
    lx_stage_pack_s = locals().get("lx_stage_pack_s", 0.0)
    lx_stage_pack_bytes = locals().get("lx_stage_pack_bytes", 0)
    lx_stage_pack_blocks = locals().get("lx_stage_pack_blocks", 0)
    lx_exchange_wait_sum_s = locals().get("lx_exchange_wait_sum_s", 0.0)
    lx_exchange_wait_max_s = locals().get("lx_exchange_wait_max_s", 0.0)
    lx_encode_wall_sum_s = locals().get("lx_encode_wall_sum_s", 0.0)
    lx_encode_wall_max_s = locals().get("lx_encode_wall_max_s", 0.0)
    lx_slow_layers = locals().get("lx_slow_layers", "")
    lx_queue_setup_s = locals().get("lx_queue_setup_s", 0.0)
    lx_exchange_thread_start_s = locals().get("lx_exchange_thread_start_s", 0.0)
    lx_encode_thread_start_s = locals().get("lx_encode_thread_start_s", 0.0)
    lx_queue_put_s = locals().get("lx_queue_put_s", 0.0)
    lx_exchange_join_s = locals().get("lx_exchange_join_s", 0.0)
    lx_encode_join_s = locals().get("lx_encode_join_s", 0.0)
    lx_ready_encode_batches = locals().get("lx_ready_encode_batches", 0)
    lx_ready_first_s = locals().get("lx_ready_first_s", 0.0)
    lx_ready_last_s = locals().get("lx_ready_last_s", 0.0)
    lx_layer_elapsed_max = locals().get("lx_layer_elapsed_max", 0.0)

    # ---- aggregate P2 phase: one transfer per physical peer/lane ----
    _async_p2_submit_t0 = time.time()
    p2_context, p2_summary = _build_and_submit_aggregate_p2(
        manager, native, encode_results, str(checkpoint_dir), rank, write_to_disk,
    )
    _async_p2_submit_elapsed = time.time() - _async_p2_submit_t0
    if rank == 0:
        logger.info(
            "FRCHECK aggregate P2: generation=%d send_tasks=%d recv_tasks=%d "
            "send_bytes=%d recv_bytes=%d async=%s",
            p2_context["generation"], p2_summary["send_tasks"], p2_summary["recv_tasks"],
            p2_summary["send_bytes"], p2_summary["recv_bytes"], _use_async_parity,
        )

    _mirror_t0 = time.time()
    native.wait_mirror_completion()
    _mirror_elapsed = time.time() - _mirror_t0
    if hasattr(native, "start_mirror_worker"):
        native.start_mirror_worker()

    network_encode_s += _async_p2_submit_elapsed
    e2e_s = time.time() - e2e_t0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if world_size > 1:
        torch.distributed.barrier()
    has_native_timing = hasattr(native, "get_ft_timing_stats")
    native_timing = native.get_ft_timing_stats() if has_native_timing else {}
    if has_native_timing:
        mirror_submitted = int(native_timing.get("mirror_tasks_submitted", 0))
        mirror_completed = int(native_timing.get("mirror_tasks_completed", 0))
        mirror_failed = int(native_timing.get("mirror_tasks_failed", 0))
        if mirror_failed or mirror_completed != mirror_submitted:
            raise RuntimeError(
                f"FRCheck mirror incomplete: completed={mirror_completed} "
                f"submitted={mirror_submitted} failed={mirror_failed}"
            )
    summary_fields = {
        "e2e_s": e2e_s,
        "network_encode_s": network_encode_s,
        "pack_total_s": pack_total_s,
        "phase1_total_s": phase1_total_s,
        "submit_total_s": submit_total_s,
        "wait_total_s": wait_total_s,
        "mirror_elapsed_s": _mirror_elapsed,
        "layer_exchange_send_tasks": layer_exchange_send_tasks,
        "layer_exchange_recv_tasks": layer_exchange_recv_tasks,
        "layer_exchange_send_bytes": layer_exchange_send_bytes,
        "layer_exchange_recv_bytes": layer_exchange_recv_bytes,
        "lx_stage_sync_s": lx_stage_sync_s,
        "lx_stage_pack_s": lx_stage_pack_s,
        "lx_stage_pack_bytes": lx_stage_pack_bytes,
        "lx_stage_pack_blocks": lx_stage_pack_blocks,
        "lx_exchange_wait_sum_s": lx_exchange_wait_sum_s,
        "lx_exchange_wait_max_s": lx_exchange_wait_max_s,
        "lx_encode_wall_sum_s": lx_encode_wall_sum_s,
        "lx_encode_wall_max_s": lx_encode_wall_max_s,
        "lx_queue_setup_s": lx_queue_setup_s,
        "lx_exchange_thread_start_s": lx_exchange_thread_start_s,
        "lx_encode_thread_start_s": lx_encode_thread_start_s,
        "lx_queue_put_s": lx_queue_put_s,
        "lx_exchange_join_s": lx_exchange_join_s,
        "lx_encode_join_s": lx_encode_join_s,
        "lx_ready_encode_batches": lx_ready_encode_batches,
        "lx_ready_first_s": lx_ready_first_s,
        "lx_ready_last_s": lx_ready_last_s,
        "lx_layer_elapsed_max": lx_layer_elapsed_max,
    }
    if has_native_timing:
        summary_fields.update({
            "d2h_s": native_timing.get("d2h_s", 0.0),
            "net_s": native_timing.get("net_s", 0.0),
            "encode_s": native_timing.get("encode_s", 0.0),
            "encode_wait_s": native_timing.get("encode_wait_s", 0.0),
            "source_send_sum_s": native_timing.get("source_send_sum_s", 0.0),
            "source_send_max_s": native_timing.get("source_send_max_s", 0.0),
            "enc_recv_sum_s": native_timing.get("enc_recv_sum_s", 0.0),
            "enc_recv_max_s": native_timing.get("enc_recv_max_s", 0.0),
            "source_send_tasks": native_timing.get("source_send_tasks", 0.0),
            "enc_recv_tasks": native_timing.get("enc_recv_tasks", 0.0),
            "source_send_bytes": native_timing.get("source_send_bytes", 0.0),
            "enc_recv_bytes": native_timing.get("enc_recv_bytes", 0.0),
            "mirror_tasks_submitted": native_timing.get("mirror_tasks_submitted", 0.0),
            "mirror_bytes_submitted": native_timing.get("mirror_bytes_submitted", 0.0),
            "mirror_tasks_completed": native_timing.get("mirror_tasks_completed", 0.0),
            "mirror_bytes_completed": native_timing.get("mirror_bytes_completed", 0.0),
            "mirror_tasks_failed": native_timing.get("mirror_tasks_failed", 0.0),
            "mirror_bytes_failed": native_timing.get("mirror_bytes_failed", 0.0),
        })
    summary = _timing_max_dict(summary_fields)
    if rank == 0:
        summary["mode"] = "async" if _use_async_parity else "sync"
        if has_native_timing:
            logger.info(
                "FRCHECK save timing (%(mode)s): e2e_s=%(e2e_s).2fs "
                "d2h_s=%(d2h_s).2fs network_encode_s=%(network_encode_s).2fs "
                "net_s=%(net_s).2fs encode_s=%(encode_s).2fs encode_wait_s=%(encode_wait_s).2fs "
                "source_send_sum_s=%(source_send_sum_s).2fs source_send_max_s=%(source_send_max_s).2fs "
                "enc_recv_sum_s=%(enc_recv_sum_s).2fs enc_recv_max_s=%(enc_recv_max_s).2fs",
                summary,
            )
            logger.info(
                "FRCHECK save detail (%(mode)s): pack_s=%(pack_total_s).2fs "
                "phase1_s=%(phase1_total_s).2fs submit_s=%(submit_total_s).2fs "
                "wait_s=%(wait_total_s).2fs mirror_wait_s=%(mirror_elapsed_s).2fs "
                "layer_exchange_send_tasks=%(layer_exchange_send_tasks).0f "
                "layer_exchange_recv_tasks=%(layer_exchange_recv_tasks).0f "
                "layer_exchange_send_bytes=%(layer_exchange_send_bytes).0f "
                "layer_exchange_recv_bytes=%(layer_exchange_recv_bytes).0f "
                "lx_stage_pack_bytes=%(lx_stage_pack_bytes).0f mirror_tasks=%(mirror_tasks_completed).0f/%(mirror_tasks_submitted).0f "
                "mirror_bytes=%(mirror_bytes_completed).0f/%(mirror_bytes_submitted).0f "
                "mirror_failed_tasks=%(mirror_tasks_failed).0f mirror_failed_bytes=%(mirror_bytes_failed).0f",
                summary,
            )
            if _dbg:
                logger.info(
                    "FRCHECK save native tasks (%(mode)s): source_send_tasks=%(source_send_tasks).0f "
                    "enc_recv_tasks=%(enc_recv_tasks).0f source_send_bytes=%(source_send_bytes).0f "
                    "enc_recv_bytes=%(enc_recv_bytes).0f",
                    summary,
                )
                logger.info(
                    "FRCHECK save submit detail (%(mode)s): queue_setup_s=%(lx_queue_setup_s).4fs "
                    "exchange_thread_start_s=%(lx_exchange_thread_start_s).4fs "
                    "encode_thread_start_s=%(lx_encode_thread_start_s).4fs "
                    "queue_put_s=%(lx_queue_put_s).4fs exchange_join_s=%(lx_exchange_join_s).4fs "
                    "encode_join_s=%(lx_encode_join_s).4fs ready_batches=%(lx_ready_encode_batches).0f "
                    "ready_first_s=%(lx_ready_first_s).4fs ready_last_s=%(lx_ready_last_s).4fs "
                    "layer_elapsed_max_s=%(lx_layer_elapsed_max).4fs",
                    summary,
                )
                logger.info(
                    "FRCHECK save stage detail (%(mode)s): stage_pack_s=%(lx_stage_pack_s).4fs "
                    "stage_sync_s=%(lx_stage_sync_s).4fs "
                    "stage_pack_bytes=%(lx_stage_pack_bytes).0f stage_pack_blocks=%(lx_stage_pack_blocks).0f "
                    "exchange_wait_sum_s=%(lx_exchange_wait_sum_s).4fs "
                    "exchange_wait_max_s=%(lx_exchange_wait_max_s).4fs "
                    "encode_wall_sum_s=%(lx_encode_wall_sum_s).4fs "
                    "encode_wall_max_s=%(lx_encode_wall_max_s).4fs",
                    summary,
                )
                logger.info(
                    "FRCHECK save slow layers (%s): %s",
                    summary["mode"], lx_slow_layers,
                )
        else:
            logger.info(
                "FRCHECK save timing (%(mode)s): e2e_s=%(e2e_s).2fs "
                "network_encode_s=%(network_encode_s).2fs "
                "native_breakdown=unavailable",
                summary,
            )

    if write_to_disk:
        _save_frcheck_stripe_files(
            manager, str(checkpoint_dir), rank, num_stripes, encode_results,
            include_encoder_p1=True, include_source=True, include_p2=False,
        )
    if _use_async_parity:
        _start_async_p2_writer(p2_context, debug=_dbg)
    else:
        _finish_aggregate_p2_context(p2_context, debug=_dbg)

    groups_by_name = {
        (f"layer_{group.layer_idx}" if group.layer_idx >= 0 else "layer_common"): group
        for group in layer_groups
    }
    for result in encode_results:
        group = groups_by_name.get(result.layer_name)
        model_tensor_keys = []
        optimizer_tensor_keys = []
        for info in result.tensor_infos:
            key = getattr(info, "key", "")
            ownership = _classify_frcheck_tensor(key, optimizer_layer_map)
            if result.layer_idx >= 0 and ownership.kind in ("model_layer", "model_common"):
                model_tensor_keys.append(key)
            elif result.layer_idx >= 0 and ownership.kind in ("optimizer_layer", "optimizer_common"):
                optimizer_tensor_keys.append(key)
        local_layer_order.append(result.layer_name)
        layer_distributed_common_keys = sorted(
            (group.distributed_common_keys or []) if group is not None else []
        )
        distributed_common_keys.update(layer_distributed_common_keys)
        local_layer_metadata[result.layer_name] = {
            "block_size": result.block_size,
            "actual_tensor_size": result.total_bytes,
            "tensor_infos": copy.deepcopy(result.tensor_infos),
            "model_tensor_keys": model_tensor_keys,
            "optimizer_tensor_keys": optimizer_tensor_keys,
            "contains_optimizer_state": bool(optimizer_tensor_keys),
            "distributed_common_keys": layer_distributed_common_keys,
        }
        if write_to_disk:
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
                    "gdr": True,
                    "tensor_infos": result.tensor_infos,
                    "actual_tensor_size": result.total_bytes,
                },
                layer_main,
            )

    torch.distributed.barrier()

    # 6. Write metadata-only main file (data_len=0; tensor payload lives in layer FRBK shards).
    main_file = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    from megatron.training.legacy_io_utils import write_main_prepared, MAGIC_FRCHECK
    # restore global offsets (per-layer encode clobbered them with local offsets)
    for info in decomposed.tensor_infos:
        info.offset = _global_offsets[id(info)]

    t_meta = time.time()
    global _cached_all_tensor_infos, _cached_all_layer_order
    global _cached_all_layer_metadata, _cached_all_actual_tensor_sizes
    global _cached_all_optimizer_layer_maps
    if _cached_all_tensor_infos is None:
        (
            all_tensor_infos,
            all_layer_order,
            all_layer_metadata,
            all_optimizer_layer_maps,
        ) = _exchange_frcheck_group_metadata(
            decomposed.tensor_infos, local_layer_order, local_layer_metadata,
            optimizer_layer_map,
        )
        all_actual_tensor_sizes = {
            r: sum(getattr(info, "size_bytes", 0) for info in infos)
            for r, infos in all_tensor_infos.items()
        }
        _cached_all_tensor_infos = all_tensor_infos
        _cached_all_layer_order = all_layer_order
        _cached_all_layer_metadata = all_layer_metadata
        _cached_all_actual_tensor_sizes = all_actual_tensor_sizes
        _cached_all_optimizer_layer_maps = all_optimizer_layer_maps
        if _dbg:
            logger.info(
                "FRCheck save: group metadata exchange %.3fs (ranks=%d)",
                time.time() - t_meta, len(all_tensor_infos),
            )
    else:
        all_tensor_infos = _cached_all_tensor_infos
        all_layer_order = _cached_all_layer_order
        all_layer_metadata = _cached_all_layer_metadata
        all_actual_tensor_sizes = _cached_all_actual_tensor_sizes
        all_optimizer_layer_maps = _cached_all_optimizer_layer_maps
        if _dbg:
            logger.info(
                "FRCheck save: group metadata exchange (cached) %.3fs",
                time.time() - t_meta,
            )

    meta1 = pickle.dumps(decomposed.non_tensor_data)
    meta2 = pickle.dumps(decomposed.tensor_infos)
    extra = pickle.dumps({
        "version": 2, "format": "frcheck_torch_legacy", "rank": rank,
        "group_id": manager.group_id, "rank_in_group": rg,
        "poa_path": manager.get_resolved_table_path() or native.path(),
        "n": n, "num_stripes": num_stripes, "num_layers": num_layers,
        "block_size": block_size, "gdr": True,
        "actual_tensor_size": total_tensor_size,
        "flat_key_roots": list(flat_key_roots) if flat_key_roots else [],
        "state_dict_keys": list(state_dict.keys()),
        "group_member_ranks": manager.group_member_ranks,
        "node_slot_to_global_rank": manager.node_slot_to_global_rank,
        "layer_names": local_layer_order,
        "all_tensor_infos": all_tensor_infos,
        "all_layer_order": all_layer_order,
        "all_layer_metadata": all_layer_metadata,
        "all_actual_tensor_sizes": all_actual_tensor_sizes,
        "all_optimizer_layer_maps": all_optimizer_layer_maps,
        "optimizer_layer_map": optimizer_layer_map,
        "distributed_common_keys": sorted(distributed_common_keys),
    })
    if write_to_disk:
        write_main_prepared(
            str(main_file), MAGIC_FRCHECK, meta1, meta2, extra, memoryview(b""), 0,
        )

    if _dbg:
        logger.info(
            "FRCheck save: done rank=%d node=%d gdr=True layers=%d file=%s (metadata-only, write_to_disk=%s)",
            rank, my_node, num_layers, main_file, write_to_disk,
        )

    del _global_offsets

    if world_size > 1:
        torch.distributed.barrier()

    release_save_buffers = bool(
        getattr(args, "frcheck_layer_exchange_encode", False) and not _use_async_parity
    )
    if release_save_buffers:
        manager.release_save_layer_buffers(empty_cuda_cache=True)

    if _dbg:
        logger.info(
            "FRCheck save: native module kept alive for reuse (no shutdown, rank=%d, released_save_buffers=%s)",
            rank, release_save_buffers,
        )


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


def _read_frbk_block_into(dst: torch.Tensor, filepath: str) -> int:
    """Read FRBK payload directly into dst. Returns bytes written."""
    path = Path(filepath)
    if not path.is_file():
        return 0
    with open(path, "rb") as f:
        hdr = f.read(28)
        if len(hdr) < 28:
            return 0
        magic, _stripe_id, _role, data_size, _block_sz = struct.unpack("<4sIIQQ", hdr)
        if magic != b"FRBK":
            return 0
        ncopy = min(int(data_size), dst.numel())
        if ncopy > 0:
            f.readinto(dst[:ncopy].numpy())
        return ncopy


def _read_frbk_range_into(
    dst: torch.Tensor,
    filepath: str,
    payload_offset: int,
    size: int,
) -> int:
    """Read a byte range from an FRBK payload directly into dst."""
    if size <= 0:
        return 0
    path = Path(filepath)
    if not path.is_file():
        return 0
    with open(path, "rb") as f:
        hdr = f.read(28)
        if len(hdr) < 28:
            return 0
        magic, _stripe_id, _role, data_size, _block_sz = struct.unpack("<4sIIQQ", hdr)
        if magic != b"FRBK":
            return 0
        if payload_offset >= int(data_size):
            return 0
        ncopy = min(int(size), int(data_size) - int(payload_offset), dst.numel())
        if ncopy > 0:
            f.seek(int(payload_offset), os.SEEK_CUR)
            f.readinto(dst[:ncopy].numpy())
        return ncopy


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


def _assert_frcheck_main_metadata_only(main_path: Path) -> None:
    """Reject legacy main files that embed tensor payload (pre-metadata-only save)."""
    from megatron.training.legacy_io_utils import (
        MAGIC_FRCHECK,
        _HEADER_MAGIC_LEN,
        _read_header,
        is_raw_format,
    )
    if is_raw_format(str(main_path), MAGIC_FRCHECK):
        with open(main_path, "rb") as f:
            magic = f.read(_HEADER_MAGIC_LEN)
            if magic != MAGIC_FRCHECK:
                return
            _, _, data_len = _read_header(f)
            if data_len > 0:
                raise RuntimeError(
                    f"FRCheck load: main file {main_path} has data_len={data_len} "
                    "(legacy payload format). Re-save with metadata-only main + FRBK blocks."
                )
        return
    payload = torch.load(main_path, map_location="cpu", weights_only=False)
    tb = payload.get("tensor_buffer")
    if torch.is_tensor(tb) and tb.numel() > 0:
        raise RuntimeError(
            f"FRCheck load: main file {main_path} contains tensor_buffer "
            "(legacy format). Re-save with metadata-only main + FRBK blocks."
        )


def _read_local_frcheck_main_payload(
    checkpoint_dir: Path,
    rank: int,
    load_tensor_buffer: bool = False,
) -> Dict[str, Any]:
    """Read this rank's main.pt from local disk only (no collective)."""
    main_path = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    if not main_path.is_file():
        raise FileNotFoundError(f"FRCheck legacy: missing main file {main_path}")
    if load_tensor_buffer:
        raise RuntimeError(
            "FRCheck load: load_tensor_buffer=True is no longer supported; "
            "assemble from local FRBK SOURCE blocks instead."
        )
    _assert_frcheck_main_metadata_only(main_path)
    from megatron.training.legacy_io_utils import (
        is_raw_format,
        read_raw_checkpoint_metadata,
        MAGIC_FRCHECK,
    )
    if is_raw_format(str(main_path), MAGIC_FRCHECK):
        return read_raw_checkpoint_metadata(str(main_path), MAGIC_FRCHECK)
    payload = torch.load(main_path, map_location="cpu", weights_only=False)
    payload["tensor_buffer"] = None
    return payload


def _load_frcheck_main_payload(
    checkpoint_dir: Path,
    rank: int,
    world_size: int,
    load_tensor_buffer: bool = False,
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
            if _frcheck_debug_enabled():
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



def _optimizer_layer_map_for_rank(main_payload: Dict[str, Any], rank: int) -> Dict[str, int]:
    all_maps = main_payload.get("all_optimizer_layer_maps") or {}
    rank_map = all_maps.get(rank)
    if rank_map is None:
        rank_map = all_maps.get(str(rank))
    if isinstance(rank_map, dict):
        return rank_map
    fallback = main_payload.get("optimizer_layer_map", {})
    return fallback if isinstance(fallback, dict) else {}


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
    dual_failure: bool = False,
    concurrency_override: Optional[int] = None,
) -> Optional[_RecoveryBufPool]:
    """Pre-allocate stripe recovery buffers once before the per-layer network loop."""
    num_helper = max(n - 3, 0)
    num_source_stripes = (n - 1) * (n - 2)
    need_pool = is_failed or is_decoder or is_helper
    if not need_pool:
        return None

    # Global recovery submit can keep windows from multiple layers in flight.
    # Allocate and register all temporary slots once up front; do not allocate
    # per-window in the network path.
    max_concurrency = max(num_source_stripes, 1)
    if concurrency_override is not None:
        concurrency = max(1, int(concurrency_override))
    else:
        env_concurrency = os.environ.get("FRCHECK_RECOVERY_CONCURRENCY")
        if env_concurrency:
            concurrency = max(1, min(int(env_concurrency), max_concurrency))
        else:
            concurrency = max_concurrency
    decoder_recv_buf_slots: List[List[torch.Tensor]] = []
    if is_decoder and num_helper > 0:
        decoder_recv_buf_slots = [
            [
                allocate_hugepage_tensor(max_block_size, fallback_pin_memory=True)
                for _ in range(num_helper)
            ]
            for _ in range(concurrency)
        ]

    failed_recv_bufs = [
        allocate_hugepage_tensor(max_block_size, fallback_pin_memory=True)
        for _ in range(concurrency)
    ] if is_failed else []
    decoder_recovered_bufs = [
        allocate_hugepage_tensor(max_block_size, fallback_pin_memory=True)
        for _ in range(concurrency)
    ] if is_decoder else []
    decoder_recovered_buf2s = [
        allocate_hugepage_tensor(max_block_size, fallback_pin_memory=True)
        for _ in range(concurrency)
    ] if is_decoder and dual_failure else []

    decoder_recv_bufs = decoder_recv_buf_slots[0] if decoder_recv_buf_slots else []
    failed_recv_buf = failed_recv_bufs[0] if failed_recv_bufs else None
    decoder_recovered_buf = decoder_recovered_bufs[0] if decoder_recovered_bufs else None
    decoder_recovered_buf2 = decoder_recovered_buf2s[0] if decoder_recovered_buf2s else None
    failed_layer_buf = (
        allocate_hugepage_tensor(
            num_source_stripes * max_block_size, fallback_pin_memory=True,
        )
        if is_failed else None
    )

    bufs_to_register: List[torch.Tensor] = []
    for recv_slot in decoder_recv_buf_slots:
        bufs_to_register.extend(recv_slot)
    bufs_to_register.extend(failed_recv_bufs)
    bufs_to_register.extend(decoder_recovered_bufs)
    bufs_to_register.extend(decoder_recovered_buf2s)
    if failed_layer_buf is not None:
        bufs_to_register.append(failed_layer_buf)

    for buf in bufs_to_register:
        native.register_recovery_buffer(buf.data_ptr(), buf.numel())

    return _RecoveryBufPool(
        decoder_recv_bufs=decoder_recv_bufs,
        decoder_recv_buf_slots=decoder_recv_buf_slots,
        failed_recv_buf=failed_recv_buf,
        failed_recv_bufs=failed_recv_bufs,
        decoder_recovered_buf=decoder_recovered_buf,
        decoder_recovered_bufs=decoder_recovered_bufs,
        decoder_recovered_buf2=decoder_recovered_buf2,
        decoder_recovered_buf2s=decoder_recovered_buf2s,
        failed_layer_buf=failed_layer_buf,
        max_block_size=max_block_size,
        concurrency=concurrency,
    )


def _preallocate_stable_failed_layer_bufs(
    native,
    buf_pool: Optional[_RecoveryBufPool],
    jobs: List[_FRCheckLayerRecoveryJob],
    n: int,
) -> None:
    """Allocate per-layer failed buffers whose views survive until forward."""
    if buf_pool is None or not jobs:
        return
    num_source_stripes = (n - 1) * (n - 2)
    stable: Dict[int, torch.Tensor] = {}
    for job in jobs:
        if job.layer_idx in stable:
            continue
        size = num_source_stripes * job.layer_block_size
        buf = allocate_hugepage_tensor(size, fallback_pin_memory=True)
        native.register_recovery_buffer(buf.data_ptr(), buf.numel())
        stable[job.layer_idx] = buf
    buf_pool.stable_failed_layer_bufs = stable


def _is_failed_in_recovery_plan(plan: Dict, my_node: int) -> bool:
    if plan.get('dual_failure'):
        return my_node in plan['failed_nodes']
    return my_node == plan.get('failed_node')


def _map_layer_buf_to_full_buf(
    layer_buf: torch.Tensor,
    layer_infos: List,
    global_tensor_infos: List,
    full_buf: torch.Tensor,
    optimizer_layer_map: Optional[Dict[str, int]] = None,
    common_only: bool = False,
) -> int:
    """Copy recovered tensors into full_buf using key-based global offset mapping."""
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
        if common_only:
            ownership = _classify_frcheck_tensor(key, optimizer_layer_map)
            if ownership.layer_idx >= 0 and ownership.kind in ("model_layer", "optimizer_layer"):
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


def _make_layer_recovery_job(
    encode_iter: int,
    rank: int,
    is_failed: bool,
    main_payload: Dict[str, Any],
    all_layer_order: Dict[int, List[str]],
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]],
    all_frcheck_dirs: List[Optional[Path]],
    checkpoint_dir: Path,
    saved_block_size: int,
    recovery_rank: Optional[int] = None,
) -> Optional[_FRCheckLayerRecoveryJob]:
    metadata_rank = recovery_rank if recovery_rank is not None else rank
    my_order = all_layer_order.get(metadata_rank, main_payload.get("layer_names", []))
    if encode_iter >= len(my_order):
        return None

    layer_name = my_order[encode_iter]
    layer_block_size = _resolve_layer_block_size(
        metadata_rank, layer_name, all_layer_metadata, saved_block_size,
    )

    meta = all_layer_metadata.get(metadata_rank, {}).get(layer_name)
    if meta is None:
        failed_dir = (
            all_frcheck_dirs[metadata_rank]
            if metadata_rank < len(all_frcheck_dirs)
            else checkpoint_dir
        )
        meta = _read_local_layer_metadata(failed_dir, layer_name, metadata_rank)
    if meta is None:
        logger.warning(
            "FRCheck recovery: no metadata for rank %d layer %s (iter %d)",
            metadata_rank, layer_name, encode_iter,
        )
        return None
    actual_size = int(meta.get("actual_tensor_size", 0))
    if actual_size == 0:
        return None
    layer_idx = -1
    if layer_name.startswith("layer_"):
        try:
            layer_idx = int(layer_name.split("_", 1)[1])
        except (TypeError, ValueError):
            layer_idx = -1
    return _FRCheckLayerRecoveryJob(
        encode_iter=encode_iter,
        layer_name=layer_name,
        layer_idx=layer_idx,
        layer_block_size=layer_block_size,
        actual_size=actual_size,
        layer_infos=meta.get("tensor_infos", []),
    )


def _materialize_recovered_layer(
    job: _FRCheckLayerRecoveryJob,
    manager,
    is_failed: bool,
    layer_buf: Optional[torch.Tensor],
    full_buf: Optional[torch.Tensor],
    global_tensor_infos: List,
    runtime: Optional[_FRCheckLayerwiseRuntime] = None,
    map_to_full_buf: str = "all",
    clone_runtime_tensors: bool = True,
    recovery_role: str = "unknown",
    tensor_views_by_key: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Optional[_FRCheckLayerReadyRecord], Dict[str, float]]:
    layer_timing: Dict[str, float] = {}
    record = None
    if is_failed and layer_buf is not None:
        t_materialize = time.time()
        copied = 0
        t_map = time.time()
        map_s = 0.0
        if map_to_full_buf != "none":
            if full_buf is None:
                raise RuntimeError(
                    "FRCheck recovery: full_buf is required when map_to_full_buf is enabled"
                )
            copied = _map_layer_buf_to_full_buf(
                layer_buf, job.layer_infos, global_tensor_infos, full_buf,
                optimizer_layer_map=getattr(manager, "_frcheck_optimizer_layer_map", None),
                common_only=(map_to_full_buf == "common_only"),
            )
            map_s = time.time() - t_map
        t_extract = time.time()
        need_layer_tensors = (
            tensor_views_by_key is not None or (runtime is not None and job.layer_idx >= 0)
        )
        layer_tensors = (
            _extract_layer_tensors_from_buf(
                layer_buf, job.layer_infos,
                clone_storage=clone_runtime_tensors,
            )
            if need_layer_tensors else None
        )
        if tensor_views_by_key is not None and layer_tensors:
            tensor_views_by_key.update(layer_tensors)
        extract_s = time.time() - t_extract
        materialize_s = time.time() - t_materialize
        layer_timing["materialize_s"] = materialize_s
        layer_timing["map_to_full_buf_s"] = map_s
        layer_timing["extract_runtime_tensors_s"] = extract_s
        _frcheck_recovery_profile(
            recovery_role, "materialize_done", layer=job.layer_name,
            layer_idx=job.layer_idx, copied_bytes=copied, map_s=map_s,
            extract_runtime_tensors_s=extract_s, materialize_s=materialize_s,
            runtime_tensors=len(layer_tensors or {}),
        )
        record = _layerwise_record_from_layer_buf(
            job.layer_name, job.encode_iter, job.layer_infos, copied,
            materialize_s, tensors=layer_tensors,
            optimizer_layer_map=getattr(manager, "_frcheck_optimizer_layer_map", None),
            model_tensor_keys=(
                runtime._records_by_layer[job.layer_idx].model_tensor_keys
                if runtime is not None and job.layer_idx in runtime._records_by_layer
                else None
            ),
            optimizer_tensor_keys=(
                runtime._records_by_layer[job.layer_idx].optimizer_tensor_keys
                if runtime is not None and job.layer_idx in runtime._records_by_layer
                else None
            ),
        )
        if runtime is not None and job.layer_idx >= 0:
            t_mark = time.time()
            pending_record = runtime._records_by_layer.get(job.layer_idx)
            opt_map = getattr(manager, "_frcheck_optimizer_layer_map", None)
            split_model_keys = (
                pending_record.model_tensor_keys if pending_record is not None else []
            )
            split_optimizer_keys = (
                pending_record.optimizer_tensor_keys if pending_record is not None else []
            )
            model_tensors, optimizer_tensors = _split_recovered_tensors(
                layer_tensors,
                split_model_keys,
                split_optimizer_keys,
                opt_map,
            )
            runtime.mark_model_ready(
                job.layer_idx,
                materialize_s=materialize_s,
                nbytes=copied,
                model_tensors=model_tensors,
            )
            if pending_record is not None and pending_record.contains_optimizer_state:
                runtime.mark_optimizer_ready(
                    job.layer_idx, optimizer_tensors=optimizer_tensors,
                )
            else:
                runtime.mark_optimizer_ready(job.layer_idx)
            _frcheck_recovery_profile(
                recovery_role, "mark_layer_ready_done", layer=job.layer_name,
                layer_idx=job.layer_idx, elapsed_s=time.time() - t_mark,
                model_tensors=len(model_tensors or {}),
                optimizer_tensors=len(optimizer_tensors or {}),
            )
        if _frcheck_debug_enabled():
            destination = "tensor_views" if map_to_full_buf == "none" else "full_buf"
            logger.info(
                "FRCheck recovery: %s materialized %d bytes into %s "
                "(expected %d) in %.4fs",
                job.layer_name, copied, destination, job.actual_size, materialize_s,
            )
    return record, layer_timing


def _run_layer_recovery_job(
    job: _FRCheckLayerRecoveryJob,
    manager,
    native,
    n: int,
    rank: int,
    is_failed: bool,
    preloaded: Dict[int, Dict[int, torch.Tensor]],
    buf_pool: Optional[_RecoveryBufPool],
    full_buf: Optional[torch.Tensor],
    global_tensor_infos: List,
    runtime: Optional[_FRCheckLayerwiseRuntime] = None,
    map_to_full_buf: str = "all",
    clone_runtime_tensors: bool = True,
    recovery_role: str = "unknown",
    tensor_views_by_key: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Optional[_FRCheckLayerReadyRecord], Dict[str, float]]:
    t_job = time.time()
    _frcheck_recovery_profile(
        recovery_role, "job_start", layer=job.layer_name, layer_idx=job.layer_idx,
        encode_iter=job.encode_iter, map_to_full_buf=map_to_full_buf,
        preloaded_blocks=len(preloaded.get(job.encode_iter, {})),
    )
    layer_buf, layer_timing = _recover_one_layer_network(
        manager, native, job.layer_name,
        job.layer_idx, job.layer_block_size, job.actual_size,
        n, rank, preloaded_blocks=preloaded.get(job.encode_iter, {}),
        buf_pool=buf_pool, recovery_role=recovery_role,
    )

    record, materialize_timing = _materialize_recovered_layer(
        job, manager, is_failed, layer_buf, full_buf, global_tensor_infos,
        runtime=runtime, map_to_full_buf=map_to_full_buf,
        clone_runtime_tensors=clone_runtime_tensors,
        recovery_role=recovery_role, tensor_views_by_key=tensor_views_by_key,
    )
    layer_timing.update(materialize_timing)
    layer_timing["job_total_s"] = time.time() - t_job
    _frcheck_recovery_profile(
        recovery_role, "job_done", layer=job.layer_name, layer_idx=job.layer_idx,
        elapsed_s=layer_timing["job_total_s"],
        network_batch_s=layer_timing.get("network_batch_s", 0.0),
        materialize_s=layer_timing.get("materialize_s", 0.0),
    )
    return record, layer_timing
def _recovery_data_failed_pos(plan: Dict[str, Any], n: int, my_node: int) -> Optional[int]:
    if plan.get('dual_failure'):
        positions = [
            int(target.get('failed_pos', n))
            for target in plan.get('failed_targets', [])
            if target.get('failed_node') == my_node
        ]
        data_positions = [pos for pos in positions if pos < n - 2]
        return min(data_positions) if data_positions else None
    pos = int(plan.get('failed_pos', n))
    return pos if pos < n - 2 else None


def _build_failed_source_block_indices(
    plans: List[Dict[str, Any]],
    active_by_stripe: Dict[int, bool],
    n: int,
) -> Dict[Tuple[int, int], int]:
    source_counts: Dict[int, int] = {node: 0 for node in range(1, n + 1)}
    source_indices: Dict[Tuple[int, int], int] = {}
    for plan in plans:
        sid = int(plan['stripe_id'])
        if not active_by_stripe.get(sid, True):
            continue
        if plan.get('dual_failure'):
            for target in plan.get('failed_targets', []):
                failed_node = int(target.get('failed_node', -1))
                original_role = int(target.get('original_role', -1))
                if original_role != int(StripeRole.SOURCE):
                    continue
                source_indices[(sid, failed_node)] = source_counts[failed_node]
                source_counts[failed_node] += 1
        else:
            failed_node = int(plan.get('failed_node', -1))
            original_role = int(plan.get('original_role', -1))
            if original_role != int(StripeRole.SOURCE):
                continue
            source_indices[(sid, failed_node)] = source_counts[failed_node]
            source_counts[failed_node] += 1
    return source_indices


def _group_recovery_data_windows_by_column(
    data_plans: List[Dict[str, Any]],
    n: int,
    my_node: int,
) -> List[List[Dict[str, Any]]]:
    windows: List[List[Dict[str, Any]]] = []
    for failed_pos in range(max(n - 2, 0)):
        column_plans = [
            plan for plan in data_plans
            if _recovery_data_failed_pos(plan, n, my_node) == failed_pos
        ]
        column_plans.sort(key=lambda plan: int(plan['stripe_id']))
        for start in range(0, len(column_plans), max(n - 1, 1)):
            windows.append(column_plans[start:start + max(n - 1, 1)])
    return windows




def _iter_recovery_windows_for_job(
    job: _FRCheckLayerRecoveryJob,
    manager,
    native,
    n: int,
    rank: int,
    preloaded_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool],
    recovery_role: str,
) -> Tuple[List[_FRCheckRecoveryWindow], Dict[int, torch.Tensor], Dict[str, float]]:
    layer_timing: Dict[str, float] = {
        'rdma_xfer_s': 0.0, 'decode_s': 0.0, 'network_batch_s': 0.0,
        'network_submit_s': 0.0, 'network_wait_s': 0.0,
        'submit_s': 0.0, 'wait_s': 0.0, 'waves': 0,
        'pipeline_overlap_s': 0.0,
    }
    if not manager.recovery_stripe_plans:
        return [], {}, layer_timing

    raw_data_plans, parity_plans, active_by_stripe, skipped_padding_stripes, n_filled_blocks = (
        _build_recovery_layer_context(manager, n, job.actual_size, job.layer_block_size)
    )
    data_plans = list(raw_data_plans)
    if not data_plans:
        if _frcheck_debug_enabled():
            logger.info(
                "FRCheck recovery layer %s: rank %d has no data recovery stripes "
                "(skipped parity stripes=%d padding stripes=%d)",
                job.layer_name, rank, len(parity_plans), skipped_padding_stripes,
            )
        return [], {}, layer_timing

    my_blocks, layer_buf, decoder_count, helper_count, failed_count = (
        _prepare_recovery_layer_buffers(
            native, manager, n, job.layer_idx, job.layer_block_size, data_plans,
            preloaded_blocks, buf_pool,
        )
    )
    num_source_stripes = (n - 1) * (n - 2)
    my_node = manager.rank_in_group + 1
    source_block_indices = _build_failed_source_block_indices(
        data_plans, active_by_stripe, n,
    )
    column_windows = _group_recovery_data_windows_by_column(data_plans, n, my_node)
    windows: List[_FRCheckRecoveryWindow] = []
    for wave_plans in column_windows:
        windows.append(_FRCheckRecoveryWindow(
            job=job,
            plans=wave_plans,
            active_by_stripe=active_by_stripe,
            wave_idx=len(windows),
            layer_buf=layer_buf,
            src_block_start=0,
            slot_start=0,
            source_block_indices=source_block_indices,
            decoder_stripes=decoder_count,
            helper_stripes=helper_count,
            failed_stripes=failed_count,
            skipped_padding_stripes=skipped_padding_stripes,
        ))

    max_window_size = max((len(window.plans) for window in windows), default=0)

    if _frcheck_debug_enabled():
        logger.info(
            "FRCheck recovery layer %s: rank %d — data=%d raw_data=%d "
            "parity_skipped=%d padding_skipped=%d filled=%d/%d "
            "decoder=%d helper=%d failed=%d stripes windows=%d max_window_size=%d",
            job.layer_name, rank, len(data_plans), len(raw_data_plans),
            len(parity_plans), skipped_padding_stripes,
            n_filled_blocks, num_source_stripes,
            decoder_count, helper_count, failed_count, len(windows), max_window_size,
        )
    layer_timing['waves'] = len(windows)
    return windows, my_blocks, layer_timing


def _frcheck_recovery_parity_repair_enabled() -> bool:
    """Return whether the active FRCheck hardware recovery requires repair."""
    try:
        from megatron.training import get_args
        args = get_args()
    except Exception:
        return False
    return (
        bool(getattr(args, "use_frcheck", False))
        and bool(getattr(args, "use_frcheck_hardware_failure", False))
    )


def _frcheck_recovery_async_parity_enabled() -> bool:
    """Return whether recovery parity repair runs behind the load thread."""
    if not _frcheck_recovery_parity_repair_enabled():
        return False
    try:
        from megatron.training import get_args
        return bool(getattr(get_args(), "frcheck_recovery_async_parity", False))
    except Exception:
        return False


def _ensure_recovery_parity_layer_buffers(
    manager,
    native,
    jobs: List[_FRCheckLayerRecoveryJob],
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]],
) -> None:
    """Initialize per-layer encode/parity buffers for HW recovery parity repair."""
    if native is None:
        return
    if manager._layer_block_sizes is None:
        manager._layer_block_sizes = {}
    if manager._layer_per_rank_bytes is None:
        manager._layer_per_rank_bytes = {}
    group_members = manager.group_member_ranks or []
    for job in jobs:
        if job.layer_idx < 0:
            continue
        layer_idx = job.layer_idx
        block_size = int(job.layer_block_size)
        manager._layer_block_sizes[layer_idx] = max(
            int(manager._layer_block_sizes.get(layer_idx, 0)), block_size
        )
        per_rank = manager._layer_per_rank_bytes.setdefault(layer_idx, {})
        if group_members:
            for rig, global_rank in enumerate(group_members):
                meta = all_layer_metadata.get(global_rank, {}).get(job.layer_name, {})
                per_rank[rig] = int(meta.get("actual_tensor_size", 0) or 0)
        else:
            per_rank[int(manager.rank_in_group)] = int(job.actual_size)
        manager._allocate_layer_stripe_bufs(
            native, layer_idx, block_size, source_on_cpu=True
        )

        my_total_bytes = int(per_rank.get(int(manager.rank_in_group), job.actual_size) or 0)
        tail_bytes = my_total_bytes % block_size if block_size > 0 else 0
        if tail_bytes:
            layer_bufs = manager.get_layer_stripe_bufs(layer_idx)
            tail_offset = (my_total_bytes // block_size) * block_size + tail_bytes
            tail_end = ((my_total_bytes + block_size - 1) // block_size) * block_size
            layer_bufs.layer_mirror_cpu[tail_offset:tail_end].zero_()


def _submit_recovery_async_parity_repair(
    manager,
    native,
    job: _FRCheckLayerRecoveryJob,
    layer_buf: Optional[torch.Tensor],
    checkpoint_dir: Path,
    rank: int,
    recovery_role: str,
    encode_batch_id: int = 0,
    reset_encode: bool = True,
    wait_encode: bool = True,
    submit_p2: bool = True,
    log_submit: bool = True,
) -> Dict[str, float]:
    """Repair parity using buffers isolated from recovered training data."""
    global _recovery_async_parity_initialized, _recovery_async_parity_submitted
    timing: Dict[str, float] = {
        "source_stage_s": 0.0,
        "encode_submit_s": 0.0,
        "encode_wait_s": 0.0,
        "build_s": 0.0,
        "submit_s": 0.0,
        "parity_tasks": 0.0,
        "send_tasks": 0.0,
    }
    if job.layer_idx < 0 or native is None:
        return timing
    if not _frcheck_recovery_async_parity_enabled():
        return timing
    has_batch_submit = hasattr(native, "submit_async_p2_layer_with_batch")
    has_encode_batch = (
        hasattr(native, "submit_source_with_batch")
        and hasattr(native, "submit_enc_recv_with_batch")
    )
    if not has_batch_submit and not hasattr(native, "submit_async_p2_layer"):
        return timing

    if not _recovery_async_parity_initialized:
        reset_t0 = time.time()
        native.reset_async_parity()
        timing["reset_s"] = time.time() - reset_t0
        _recovery_async_parity_initialized = True

    n = int(native.n())
    num_stripes = int(native.num_stripes())
    my_node = int(manager.rank_in_group) + 1
    layer_name = job.layer_name
    layer_idx = job.layer_idx
    layer_block_size = job.layer_block_size
    layer_bufs = manager.get_layer_stripe_bufs(layer_idx)
    per_rank_bytes = (manager._layer_per_rank_bytes or {}).get(layer_idx, {})
    my_total_bytes = int(per_rank_bytes.get(int(manager.rank_in_group), job.actual_size) or 0)

    src_blk_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}
    enc_active_masks: Dict[int, List[int]] = {}
    for sid in range(num_stripes):
        plan = manager.stripe_plans[sid]
        if plan.role == StripeRole.ENCODER:
            mask = []
            for src_node in plan.source_node_ids:
                b = src_blk_per_node.get(src_node, 0)
                nf = manager.get_n_filled_for_node(layer_idx, src_node)
                mask.append(1 if b < nf else 0)
            enc_active_masks[sid] = mask
        for src_node in plan.source_node_ids:
            src_blk_per_node[src_node] = src_blk_per_node.get(src_node, 0) + 1

    def _stage_source_for_sid(sid: int) -> Tuple[int, int]:
        blk_idx = _source_blk_idx_for_node_stripe(manager.stripe_plans, sid, my_node)
        if blk_idx < 0:
            return 0, 0
        n_filled = manager.get_n_filled_for_node(layer_idx, my_node)
        if blk_idx >= n_filled:
            return 0, 0

        src_offset = blk_idx * layer_block_size
        actual_len = max(0, min(layer_block_size, my_total_bytes - src_offset))
        if actual_len <= 0:
            return 0, 0

        # Failed ranks already reconstructed source blocks in the same contiguous
        # layer layout used by POA source stripes; submit those CPU blocks directly.
        if layer_buf is not None:
            return int(layer_buf[src_offset:src_offset + layer_block_size].data_ptr()), 0

        stage_t0 = time.time()
        blk = _read_stripe_block(
            checkpoint_dir, layer_name, sid, rank, int(StripeRole.SOURCE)
        )
        blk = _normalize_stripe_block(blk, layer_block_size, rank, layer_name, sid)
        blk = blk[:actual_len]
        if not blk.is_contiguous():
            blk = blk.contiguous()

        cpu_view = layer_bufs.layer_mirror_cpu[src_offset:src_offset + layer_block_size]
        copy_len = min(int(blk.numel()), actual_len)
        cpu_view[:copy_len].copy_(blk[:copy_len], non_blocking=False)
        timing["source_stage_s"] += time.time() - stage_t0
        return int(cpu_view.data_ptr()), 0

    submit_t0 = time.time()
    if reset_encode:
        if hasattr(native, "reset_encode_layer"):
            native.reset_encode_layer()
        else:
            native.reset_layer()
    for sid in range(num_stripes):
        plan = manager.stripe_plans[sid]
        if plan.role == StripeRole.SOURCE:
            addr, mirror_addr = _stage_source_for_sid(sid)
            if addr == 0:
                if encode_batch_id and has_encode_batch and hasattr(native, "skip_source_batch"):
                    native.skip_source_batch(sid, int(encode_batch_id))
                continue
            if encode_batch_id and has_encode_batch:
                native.submit_source_with_batch(
                    sid, addr, mirror_addr, layer_block_size, int(encode_batch_id)
                )
            else:
                native.submit_source(sid, addr, mirror_addr, layer_block_size)
        elif plan.role == StripeRole.ENCODER:
            rb = layer_bufs.recv_bufs[sid]
            p1b = layer_bufs.parity1_bufs[sid]
            p2b = layer_bufs.parity2_bufs[sid]
            if rb is None or p1b is None or p2b is None:
                continue
            if encode_batch_id and has_encode_batch:
                native.submit_enc_recv_with_batch(
                    sid, rb.data_ptr(), p1b.data_ptr(), p2b.data_ptr(),
                    layer_block_size, enc_active_masks.get(sid, []), int(encode_batch_id),
                )
            else:
                native.submit_enc_recv(
                    sid, rb.data_ptr(), p1b.data_ptr(), p2b.data_ptr(),
                    layer_block_size, enc_active_masks.get(sid, []),
                )
    timing["encode_submit_s"] = time.time() - submit_t0

    if wait_encode:
        wait_t0 = time.time()
        native.wait_encode_only()
        timing["encode_wait_s"] = time.time() - wait_t0

    if submit_p2:
        _submit_recovery_async_parity_p2(
            manager, native, job, timing, recovery_role, has_batch_submit
        )
    if log_submit:
        _log_recovery_async_parity_submit(recovery_role, job, timing)
    return timing


def _submit_recovery_async_parity_p2(
    manager, native, job: _FRCheckLayerRecoveryJob, timing: Dict[str, float],
    recovery_role: str, has_batch_submit: bool,
) -> None:
    global _recovery_async_parity_submitted
    layer_bufs = manager.get_layer_stripe_bufs(job.layer_idx)
    build_t0 = time.time()
    p2_addrs = [0 if buf is None else int(buf.data_ptr()) for buf in layer_bufs.parity2_bufs]
    timing["build_s"] = timing.get("build_s", 0.0) + (time.time() - build_t0)
    submit_p2_t0 = time.time()
    repair_batch_id = int(job.layer_idx) + 1
    if has_batch_submit:
        counts = native.submit_async_p2_layer_with_batch(
            p2_addrs, job.layer_block_size, repair_batch_id
        )
    else:
        counts = native.submit_async_p2_layer(p2_addrs, job.layer_block_size)
    timing["submit_s"] = timing.get("submit_s", 0.0) + (time.time() - submit_p2_t0)
    if counts is not None and len(counts) >= 2:
        timing["parity_tasks"] = float(int(counts[0]))
        timing["send_tasks"] = float(int(counts[1]))
    if timing.get("parity_tasks", 0.0) or timing.get("send_tasks", 0.0):
        _recovery_async_parity_submitted = True


def _log_recovery_async_parity_submit(
    recovery_role: str, job: _FRCheckLayerRecoveryJob, timing: Dict[str, float]
) -> None:
    _frcheck_recovery_profile(
        recovery_role,
        "recovery_async_parity_submit",
        layer=job.layer_name,
        layer_idx=job.layer_idx,
        parity_tasks=int(timing.get("parity_tasks", 0.0)),
        send_tasks=int(timing.get("send_tasks", 0.0)),
        source_stage_s=timing.get("source_stage_s", 0.0),
        encode_submit_s=timing.get("encode_submit_s", 0.0),
        encode_wait_s=timing.get("encode_wait_s", 0.0),
        submit_s=timing.get("submit_s", 0.0),
        flush_s=timing.get("flush_s", 0.0),
    )

def _flush_recovery_async_parity(reason: str, role: str = "unknown") -> None:
    global _recovery_async_parity_submitted
    if not _recovery_async_parity_submitted:
        return
    # The decode-batch repair core waits every native batch before returning.
    _recovery_async_parity_submitted = False


def _build_recovery_parity_repair_context(
    manager,
    n: int,
    job: _FRCheckLayerRecoveryJob,
) -> Tuple[List[Dict[str, Any]], Dict[int, bool]]:
    parity_plans = [
        p for p in manager.recovery_stripe_plans
        if not _is_data_recovery_plan(p, n)
    ]
    active_by_stripe = {int(plan['stripe_id']): True for plan in parity_plans}
    return parity_plans, active_by_stripe


def _prepare_recovery_parity_repair_buffers(
    native,
    manager,
    n: int,
    job: _FRCheckLayerRecoveryJob,
    parity_plans: List[Dict[str, Any]],
    preloaded_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool],
) -> Tuple[Dict[int, torch.Tensor], int, int, int]:
    my_node = manager.rank_in_group + 1
    decoder_plans = [p for p in parity_plans if my_node == p['decoder_node']]
    helper_plans = [p for p in parity_plans if my_node in p['helper_nodes']]
    failed_plans = [p for p in parity_plans if _is_failed_in_recovery_plan(p, my_node)]

    my_blocks: Dict[int, torch.Tensor] = {}
    for plan in decoder_plans + helper_plans:
        sid = int(plan['stripe_id'])
        blk = preloaded_blocks.get(sid)
        if blk is None:
            role = _stripe_role_for_node(manager, sid, my_node, n)
            raise RuntimeError(
                "FRCheck recovery parity repair: missing preloaded block "
                f"layer={job.layer_name} stripe={sid} role={role} rank_node={my_node}"
            )
        my_blocks[sid] = blk

    return my_blocks, len(decoder_plans), len(helper_plans), len(failed_plans)


def _submit_recovery_parity_repair_window(
    native,
    manager,
    job: _FRCheckLayerRecoveryJob,
    plans: List[Dict[str, Any]],
    active_by_stripe: Dict[int, bool],
    my_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool],
    n: int,
    slot_start: int,
) -> Dict[str, float]:
    my_node = manager.rank_in_group + 1
    t_reset = time.time()
    batch_id = native.begin_recovery_batch()
    reset_s = time.time() - t_reset
    t_submit = time.time()
    active_stripes = 0
    for submit_idx, stri_plan in enumerate(plans):
        sid = int(stri_plan['stripe_id'])
        active = active_by_stripe.get(sid, True)
        if active:
            active_stripes += 1
        buf_slot = slot_start + submit_idx
        helper_block_addr = 0
        decoder_self_block_addr = 0
        decoder_helper_recv_addrs: List[int] = []
        decoder_recovered_addrs: List[int] = []
        failed_recv_buf_addr = 0

        if my_node in stri_plan['helper_nodes']:
            blk = my_blocks.get(sid)
            if blk is not None:
                helper_block_addr = int(blk.data_ptr())

        if my_node == stri_plan['decoder_node']:
            blk = my_blocks.get(sid)
            if blk is not None:
                decoder_self_block_addr = int(blk.data_ptr())
            if buf_pool is not None:
                recv_bufs = (
                    buf_pool.decoder_recv_buf_slots[buf_slot]
                    if buf_slot < len(buf_pool.decoder_recv_buf_slots)
                    else buf_pool.decoder_recv_bufs
                )
                for hi in range(len(stri_plan['helper_nodes'])):
                    if hi < len(recv_bufs):
                        decoder_helper_recv_addrs.append(
                            int(recv_bufs[hi][:job.layer_block_size].data_ptr())
                        )
                if buf_slot < len(buf_pool.decoder_recovered_bufs):
                    decoder_recovered_addrs.append(
                        int(buf_pool.decoder_recovered_bufs[buf_slot][:job.layer_block_size].data_ptr())
                    )
                if (
                    stri_plan.get('dual_failure')
                    and buf_slot < len(buf_pool.decoder_recovered_buf2s)
                ):
                    decoder_recovered_addrs.append(
                        int(buf_pool.decoder_recovered_buf2s[buf_slot][:job.layer_block_size].data_ptr())
                    )

        if _is_failed_in_recovery_plan(stri_plan, my_node):
            if (
                buf_pool is not None
                and buf_slot < len(buf_pool.failed_recv_bufs)
            ):
                failed_recv_buf_addr = int(
                    buf_pool.failed_recv_bufs[buf_slot][:job.layer_block_size].data_ptr()
                )

        native.submit_recovery_stripe_to_batch(
            batch_id,
            sid,
            job.layer_block_size,
            helper_block_addr,
            decoder_self_block_addr,
            decoder_helper_recv_addrs,
            decoder_recovered_addrs,
            failed_recv_buf_addr,
            0,
            0,
            0,
            False,
            active,
        )

    native.end_recovery_batch(batch_id)
    return {
        "batch_id": int(batch_id),
        "reset_s": reset_s,
        "submit_s": time.time() - t_submit,
        "active_stripes": float(active_stripes),
    }


def _submit_recovery_parity_repair(
    manager,
    native,
    job: _FRCheckLayerRecoveryJob,
    preloaded_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool],
    recovery_role: str,
) -> Dict[str, float]:
    timing: Dict[str, float] = {
        "submit_s": 0.0,
        "wait_s": 0.0,
        "network_batch_s": 0.0,
        "decoder_stripes": 0.0,
        "helper_stripes": 0.0,
        "failed_stripes": 0.0,
        "parity_stripes": 0.0,
        "waves": 0.0,
    }
    if native is None:
        return timing
    n = int(native.n())
    parity_plans, active_by_stripe = _build_recovery_parity_repair_context(manager, n, job)
    if not parity_plans:
        return timing
    my_blocks, decoder_count, helper_count, failed_count = _prepare_recovery_parity_repair_buffers(
        native, manager, n, job, parity_plans, preloaded_blocks, buf_pool,
    )
    timing["decoder_stripes"] = float(decoder_count)
    timing["helper_stripes"] = float(helper_count)
    timing["failed_stripes"] = float(failed_count)
    timing["parity_stripes"] = float(len(parity_plans))

    max_inflight = buf_pool.concurrency if buf_pool is not None else max(len(parity_plans), 1)
    t_network = time.time()
    batch_timings: List[Dict[str, float]] = []
    for start in range(0, len(parity_plans), max_inflight):
        wave_plans = parity_plans[start:start + max_inflight]
        batch_timing = _submit_recovery_parity_repair_window(
            native, manager, job, wave_plans, active_by_stripe,
            my_blocks, buf_pool, n, 0,
        )
        batch_timings.append(batch_timing)
        timing["submit_s"] += batch_timing.get("reset_s", 0.0) + batch_timing.get("submit_s", 0.0)
    for batch_timing in batch_timings:
        wait_t0 = time.time()
        batch_id = int(batch_timing.get("batch_id", 0))
        native.wait_recovery_batch_id(batch_id)
        if hasattr(native, "get_recovery_batch_milestones"):
            # Parity repair runs after failed training data is available and is not
            # part of the first-layer recovery critical path. Consume the native
            # record to release it without merging it into first-layer milestones.
            native.get_recovery_batch_milestones(batch_id)
        wait_s = time.time() - wait_t0
        timing["wait_s"] += wait_s
    timing["waves"] = float(len(batch_timings))
    timing["network_batch_s"] = time.time() - t_network
    return timing



def _set_pending_recovery_parity_repair(
    jobs: List[_FRCheckLayerRecoveryJob],
    buf_pool: Optional[_RecoveryBufPool],
    checkpoint_dir: Path,
    rank: int,
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]],
    recovery_role: str,
    parity_preloaded: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
) -> None:
    global _pending_recovery_parity_repair
    if not jobs or not _frcheck_recovery_parity_repair_enabled():
        return
    _pending_recovery_parity_repair = {
        "jobs": list(jobs),
        "buf_pool": buf_pool,
        "checkpoint_dir": checkpoint_dir,
        "rank": rank,
        "all_layer_metadata": all_layer_metadata,
        "recovery_role": recovery_role,
        "parity_preloaded": parity_preloaded or {},
        "execution_mode": (
            "background" if _frcheck_recovery_async_parity_enabled() else "sync"
        ),
    }


def _run_recovery_parity_repair_submissions(pending: Dict[str, Any], reason: str) -> None:
    """Recover failed parity stripes after data recovery completes."""
    global _recovery_async_parity_error, _recovery_async_parity_submitted
    manager = FRCheckManager()
    native = manager.get_native()
    if native is None:
        return
    recovery_role = str(pending.get("recovery_role", "unknown"))
    buf_pool = pending.get("buf_pool")
    jobs = list(pending["jobs"])
    preloaded = pending.get("parity_preloaded", {})
    execution_mode = str(pending.get("execution_mode", "background"))
    t0 = time.time()
    total_stripes = 0
    total_waves = 0
    total_submit_s = 0.0
    total_wait_s = 0.0
    _recovery_async_parity_submitted = True
    _frcheck_recovery_profile(
        recovery_role,
        "recovery_parity_repair_start",
        reason=reason,
        jobs=len(jobs),
        mode=execution_mode,
    )
    try:
        for job in jobs:
            timing = _submit_recovery_parity_repair(
                manager, native, job, preloaded.get(job.encode_iter, {}),
                buf_pool, recovery_role,
            )
            total_stripes += int(timing.get("parity_stripes", 0.0))
            total_waves += int(timing.get("waves", 0.0))
            total_submit_s += float(timing.get("submit_s", 0.0))
            total_wait_s += float(timing.get("wait_s", 0.0))
            _frcheck_recovery_profile(
                recovery_role,
                "recovery_parity_repair_layer_done",
                layer=job.layer_name,
                layer_idx=job.layer_idx,
                mode=execution_mode,
                parity_stripes=int(timing.get("parity_stripes", 0.0)),
                decoder_stripes=int(timing.get("decoder_stripes", 0.0)),
                helper_stripes=int(timing.get("helper_stripes", 0.0)),
                failed_stripes=int(timing.get("failed_stripes", 0.0)),
                waves=int(timing.get("waves", 0.0)),
                submit_s=timing.get("submit_s", 0.0),
                wait_s=timing.get("wait_s", 0.0),
                network_batch_s=timing.get("network_batch_s", 0.0),
            )
        _frcheck_recovery_profile(
            recovery_role,
            "recovery_parity_repair_done",
            jobs=len(jobs),
            tasks=total_stripes,
            waves=total_waves,
            mode=execution_mode,
            submit_s=total_submit_s,
            wait_s=total_wait_s,
            elapsed_s=time.time() - t0,
        )
    except BaseException as exc:
        _recovery_async_parity_error = exc
        _frcheck_recovery_profile(
            recovery_role,
            "recovery_parity_repair_error",
            mode=execution_mode,
            error=f"{type(exc).__name__}: {exc}",
        )

def _start_recovery_parity_repair_submissions(reason: str, role: str = "unknown") -> None:
    """Start parity repair submit work in the background after recovery completes."""
    global _pending_recovery_parity_repair, _recovery_async_parity_thread
    pending = _pending_recovery_parity_repair
    if pending is None or _recovery_async_parity_thread is not None:
        return
    _pending_recovery_parity_repair = None
    recovery_role = str(pending.get("recovery_role", role))
    worker = threading.Thread(
        target=_run_recovery_parity_repair_submissions,
        args=(pending, reason),
        name=f"frcheck-recovery-parity-rank{pending['rank']}",
        daemon=False,
    )
    _recovery_async_parity_thread = worker
    worker.start()


def _finish_recovery_parity_repair_submissions(reason: str, role: str = "unknown") -> None:
    """Join background parity repair and propagate its error before teardown."""
    global _recovery_async_parity_thread, _recovery_async_parity_error
    if _frcheck_recovery_async_parity_enabled():
        _start_recovery_parity_repair_submissions(reason, role)
    worker = _recovery_async_parity_thread
    if worker is not None:
        t_join = time.time()
        _frcheck_recovery_profile(role, "recovery_parity_join_start", reason=reason)
        worker.join()
        _recovery_async_parity_thread = None
        _frcheck_recovery_profile(
            role, "recovery_parity_join_done", reason=reason,
            elapsed_s=time.time() - t_join, mode="background",
        )
    if _recovery_async_parity_error is not None:
        exc = _recovery_async_parity_error
        _recovery_async_parity_error = None
        raise RuntimeError("FRCheck recovery parity repair failed") from exc


def _run_pending_recovery_parity_repair_sync(
    reason: str, role: str = "unknown"
) -> None:
    """Run the same parity repair core synchronously after data recovery."""
    global _pending_recovery_parity_repair
    pending = _pending_recovery_parity_repair
    if pending is None:
        return
    _pending_recovery_parity_repair = None
    pending["execution_mode"] = "sync"
    _run_recovery_parity_repair_submissions(pending, reason)
    _finish_recovery_parity_repair_submissions(reason, role)


def _run_recovery_pipeline(
    jobs: List[_FRCheckLayerRecoveryJob],
    manager,
    native,
    n: int,
    rank: int,
    is_failed: bool,
    preloaded: Dict[int, Dict[int, torch.Tensor]],
    buf_pool: Optional[_RecoveryBufPool],
    full_buf: Optional[torch.Tensor],
    global_tensor_infos: List,
    runtime: Optional[_FRCheckLayerwiseRuntime] = None,
    map_to_full_buf: str = "all",
    clone_runtime_tensors: bool = True,
    recovery_role: str = "unknown",
    tensor_views_by_key: Optional[Dict[str, torch.Tensor]] = None,
) -> List[Tuple[Optional[_FRCheckLayerReadyRecord], Dict[str, float]]]:
    t_pipeline = time.time()
    completed: "queue.Queue[Optional[_FRCheckRecoveredWindow]]" = queue.Queue(maxsize=2)
    results: List[Tuple[Optional[_FRCheckLayerReadyRecord], Dict[str, float]]] = []
    aggregate_timing: Dict[int, Dict[str, float]] = {}
    first_network_done = {"time": 0.0}
    last_materialize_done = {"time": 0.0}
    last_model_done = {"time": 0.0}
    last_optimizer_done = {"time": 0.0}
    last_common_done = {"time": 0.0}
    pipeline_start_delay_s = 0.0
    try:
        from megatron.training.global_vars import get_recovery_to_forward_timer_start
        recovery_to_forward_start_s = get_recovery_to_forward_timer_start()
        if recovery_to_forward_start_s > 0.0:
            pipeline_start_delay_s = max(0.0, t_pipeline - recovery_to_forward_start_s)
    except Exception:
        pipeline_start_delay_s = 0.0
    error_holder: Dict[str, Optional[BaseException]] = {"error": None}

    def _network_worker() -> None:
        def _alloc_slots(
            free_ranges: List[Tuple[int, int]], required: int
        ) -> Optional[int]:
            for idx, (start_slot, length) in enumerate(free_ranges):
                if length < required:
                    continue
                alloc_start = start_slot
                if length == required:
                    free_ranges.pop(idx)
                else:
                    free_ranges[idx] = (start_slot + required, length - required)
                return alloc_start
            return None

        def _free_slots(
            free_ranges: List[Tuple[int, int]], start_slot: int, length: int
        ) -> None:
            free_ranges.append((start_slot, length))
            free_ranges.sort()
            merged: List[Tuple[int, int]] = []
            for cur_start, cur_len in free_ranges:
                if not merged:
                    merged.append((cur_start, cur_len))
                    continue
                prev_start, prev_len = merged[-1]
                prev_end = prev_start + prev_len
                if cur_start <= prev_end:
                    merged[-1] = (prev_start, max(prev_end, cur_start + cur_len) - prev_start)
                else:
                    merged.append((cur_start, cur_len))
            free_ranges[:] = merged

        def _mark_window_done(recovered: _FRCheckRecoveredWindow) -> None:
            job = recovered.window.job
            layer_timing = aggregate_timing.get(id(job), {})
            t_wait = time.time()
            batch_id = int(recovered.timing.get('batch_id', 0))
            native.wait_recovery_batch_id(batch_id)
            wait_s = time.time() - t_wait
            recovered.timing['wait_s'] = wait_s
            if hasattr(native, 'get_recovery_batch_milestones'):
                _record_frcheck_native_batch_milestones(native, batch_id, job.layer_idx)
            batch_timing = {}
            if hasattr(native, 'get_recovery_batch_timing_stats'):
                batch_timing = dict(native.get_recovery_batch_timing_stats())
            for start_key, end_key in (
                ('net_start_us', 'net_end_us'),
                ('decode_start_us', 'decode_end_us'),
            ):
                start_us = float(batch_timing.get(start_key, 0.0) or 0.0)
                end_us = float(batch_timing.get(end_key, 0.0) or 0.0)
                if start_us > 0.0:
                    old_start = float(layer_timing.get(start_key, 0.0) or 0.0)
                    layer_timing[start_key] = start_us if old_start == 0.0 else min(old_start, start_us)
                if end_us > 0.0:
                    layer_timing[end_key] = max(float(layer_timing.get(end_key, 0.0) or 0.0), end_us)
            layer_timing['failed_copy_s'] = layer_timing.get('failed_copy_s', 0.0) + float(batch_timing.get('copy_s', 0.0) or 0.0)
            layer_timing['decoder_decode_sum_s'] = (
                layer_timing.get('decoder_decode_sum_s', 0.0)
                + float(batch_timing.get('decoder_decode_sum_s', 0.0) or 0.0)
            )
            layer_timing['network_wait_s'] += wait_s
            layer_timing['wait_s'] = layer_timing['network_wait_s']
            _frcheck_recovery_profile(
                recovery_role, "network_window_done", layer=job.layer_name,
                layer_idx=job.layer_idx, wave=recovered.window.wave_idx,
                batch_id=recovered.timing.get('batch_id', 0),
                wait_s=wait_s,
                net_s=batch_timing.get('net_s', 0.0),
                decode_s=batch_timing.get('decode_s', 0.0),
            )
            first_network_done["time"] = first_network_done["time"] or time.time()
            if recovered.window.wave_idx + 1 == len(job_windows.get(id(job), [])):
                started = job_network_start.get(id(job), time.time())
                layer_timing['network_batch_s'] = time.time() - started
                layer_timing['rdma_xfer_s'] = layer_timing['network_batch_s']
            completed.put(recovered)

        try:
            pending_windows: List[Tuple[_FRCheckRecoveryWindow, Dict[int, torch.Tensor]]] = []
            job_windows: Dict[int, List[_FRCheckRecoveryWindow]] = {}
            job_network_start: Dict[int, float] = {}
            for job in jobs:
                windows, my_blocks, layer_timing = _iter_recovery_windows_for_job(
                    job, manager, native, n, rank,
                    preloaded.get(job.encode_iter, {}), buf_pool, recovery_role,
                )
                aggregate_timing[id(job)] = layer_timing
                job_windows[id(job)] = windows
                for window in windows:
                    pending_windows.append((window, my_blocks))

            max_inflight_stripes = (
                buf_pool.concurrency if buf_pool is not None else max((n - 1) * (n - 2), 1)
            )
            free_ranges: List[Tuple[int, int]] = [(0, max_inflight_stripes)]
            active_windows: List[Tuple[_FRCheckRecoveredWindow, int, int]] = []
            idx = 0
            while idx < len(pending_windows):
                window, my_blocks = pending_windows[idx]
                required_slots = max(len(window.plans), 1)
                if required_slots > max_inflight_stripes:
                    raise RuntimeError(
                        "FRCheck recovery: one POA recovery window requires "
                        f"{required_slots} slots but buffer pool has {max_inflight_stripes}"
                    )
                slot_start = _alloc_slots(free_ranges, required_slots)
                if slot_start is None:
                    recovered, old_start, old_len = active_windows.pop(0)
                    _mark_window_done(recovered)
                    _free_slots(free_ranges, old_start, old_len)
                    continue

                window.slot_start = slot_start
                job = window.job
                layer_timing = aggregate_timing.get(id(job), {})
                if id(job) not in job_network_start:
                    job_network_start[id(job)] = time.time()
                timing = _submit_recovery_window(
                    native, manager, window, my_blocks, buf_pool, n,
                )
                layer_timing['network_submit_s'] += timing.get('reset_s', 0.0) + timing.get('submit_s', 0.0)
                layer_timing['submit_s'] = layer_timing['network_submit_s']
                recovered = _FRCheckRecoveredWindow(window=window, timing=timing)
                active_windows.append((recovered, slot_start, required_slots))
                idx += 1
                _frcheck_recovery_profile(
                    recovery_role, "network_window_submitted", layer=job.layer_name,
                    layer_idx=job.layer_idx, wave=window.wave_idx,
                    batch_id=timing.get('batch_id', 0), stripes=len(window.plans),
                    active_stripes=int(timing.get('active_stripes', 0.0)),
                    skipped_padding_stripes=(
                        len(window.plans) - int(timing.get('active_stripes', 0.0))
                    ),
                    reset_s=timing.get('reset_s', 0.0),
                    submit_s=timing.get('submit_s', 0.0),
                    slot_start=window.slot_start,
                    inflight_windows=len(active_windows),
                )

            while active_windows:
                recovered, old_start, old_len = active_windows.pop(0)
                _mark_window_done(recovered)
                _free_slots(free_ranges, old_start, old_len)
        except BaseException as exc:
            error_holder["error"] = exc
        finally:
            completed.put(None)

    network_thread = threading.Thread(
        target=_network_worker,
        name=f"frcheck-recovery-network-rank{rank}",
        daemon=False,
    )
    network_thread.start()
    materialized_jobs: Set[int] = set()
    while True:
        item = completed.get()
        if item is None:
            break
        job = item.window.job
        if id(job) in materialized_jobs:
            continue
        layer_timing = aggregate_timing.get(id(job), {})
        if error_holder["error"] is not None:
            break
        # A job can be materialized after its final window arrives.  Since windows
        # are submitted in job order, the last window has the highest wave index.
        if item.window.wave_idx + 1 < int(layer_timing.get('waves', 0)):
            continue
        record, materialize_timing = _materialize_recovered_layer(
            job, manager, is_failed, item.window.layer_buf, full_buf,
            global_tensor_infos, runtime=runtime, map_to_full_buf=map_to_full_buf,
            clone_runtime_tensors=clone_runtime_tensors,
            recovery_role=recovery_role, tensor_views_by_key=tensor_views_by_key,
        )
        layer_timing.update(materialize_timing)
        layer_timing['job_total_s'] = (
            layer_timing.get('network_batch_s', 0.0)
            + layer_timing.get('materialize_s', 0.0)
        )
        results.append((record, layer_timing))
        materialized_jobs.add(id(job))
        done_time = time.time()
        last_materialize_done["time"] = done_time
        if record is not None and record.model_tensor_keys:
            last_model_done["time"] = done_time
        if record is not None and record.optimizer_tensor_keys:
            last_optimizer_done["time"] = done_time
        if record is None or job.layer_idx < 0:
            last_common_done["time"] = done_time
        _frcheck_recovery_profile(
            recovery_role, "pipeline_job_done", layer=job.layer_name,
            layer_idx=job.layer_idx,
            network_batch_s=layer_timing.get('network_batch_s', 0.0),
            network_submit_s=layer_timing.get('network_submit_s', 0.0),
            network_wait_s=layer_timing.get('network_wait_s', 0.0),
            materialize_s=layer_timing.get('materialize_s', 0.0),
            waves=layer_timing.get('waves', 0),
        )
    network_thread.join()
    if error_holder["error"] is not None:
        raise error_holder["error"]
    elapsed = time.time() - t_pipeline
    serial_work_s = sum(
        timing.get('network_batch_s', 0.0) + timing.get('materialize_s', 0.0)
        for _record, timing in results
    )
    overlap_s = max(0.0, serial_work_s - elapsed)
    total_submit_s = sum(timing.get('network_submit_s', 0.0) for _record, timing in results)
    total_wait_s = sum(timing.get('network_wait_s', 0.0) for _record, timing in results)
    total_materialize_s = sum(timing.get('materialize_s', 0.0) for _record, timing in results)
    net_starts = [float(timing.get('net_start_us', 0.0) or 0.0) for _record, timing in results]
    net_ends = [float(timing.get('net_end_us', 0.0) or 0.0) for _record, timing in results]
    net_start_us = min((value for value in net_starts if value > 0.0), default=0.0)
    net_end_us = max(net_ends, default=0.0)
    total_recovery_net_s = ((net_end_us - net_start_us) / 1.0e6) if net_start_us > 0.0 and net_end_us > net_start_us else 0.0
    total_recovery_decode_s = sum(
        timing.get('decoder_decode_sum_s', 0.0) for _record, timing in results
    )
    total_failed_copy_s = sum(timing.get('failed_copy_s', 0.0) for _record, timing in results)
    for _record, timing in results:
        timing['pipeline_overlap_s'] = overlap_s
        timing['pipeline_critical_s'] = elapsed
    first_network_done_s = (
        first_network_done["time"] - t_pipeline if first_network_done["time"] else 0.0
    )
    last_model_done_s = (
        last_model_done["time"] - t_pipeline if last_model_done["time"] else 0.0
    )
    last_optimizer_done_s = (
        last_optimizer_done["time"] - t_pipeline if last_optimizer_done["time"] else 0.0
    )
    last_common_done_s = (
        last_common_done["time"] - t_pipeline if last_common_done["time"] else 0.0
    )
    last_materialize_done_s = (
        last_materialize_done["time"] - t_pipeline if last_materialize_done["time"] else 0.0
    )
    summary = {
        "pipeline_s": elapsed,
        "network_submit_s": total_submit_s,
        "network_wait_s": total_wait_s,
        "materialize_s": total_materialize_s,
        "recovery_net_s": total_recovery_net_s,
        "recovery_decode_s": total_recovery_decode_s,
        "h2d_s": total_failed_copy_s,
        "serial_work_s": serial_work_s,
        "pipeline_overlap_s": overlap_s,
        "first_network_done_s": first_network_done_s,
        "last_model_done_s": last_model_done_s,
        "last_optimizer_done_s": last_optimizer_done_s,
        "last_common_done_s": last_common_done_s,
        "last_materialize_done_s": last_materialize_done_s,
        "pipeline_start_delay_s": pipeline_start_delay_s,
    }
    try:
        from megatron.training.global_vars import stash_recovery_timing_summary
        stash_recovery_timing_summary("frcheck_hw_pipeline", summary)
    except Exception:
        pass
    return results



def _start_layer_recovery_worker(
    jobs: List[_FRCheckLayerRecoveryJob],
    manager,
    native,
    n: int,
    rank: int,
    is_failed: bool,
    preloaded: Dict[int, Dict[int, torch.Tensor]],
    buf_pool: Optional[_RecoveryBufPool],
    full_buf: Optional[torch.Tensor],
    global_tensor_infos: List,
    runtime: Optional[_FRCheckLayerwiseRuntime],
    cleanup_after: bool,
    map_to_full_buf: str = "all",
    clone_runtime_tensors: bool = True,
    recovery_role: str = "unknown",
    tensor_views_by_key: Optional[Dict[str, torch.Tensor]] = None,
    checkpoint_dir: Optional[Path] = None,
    all_layer_metadata: Optional[Dict[int, Dict[str, Dict[str, Any]]]] = None,
) -> threading.Thread:
    error_holder: Dict[str, Optional[BaseException]] = {"error": None}

    def _worker() -> None:
        t_worker = time.time()
        _frcheck_recovery_profile(
            recovery_role, "worker_start", jobs=len(jobs),
            map_to_full_buf=map_to_full_buf, cleanup_after=cleanup_after,
        )
        try:
            try:
                _run_recovery_pipeline(
                    jobs, manager, native, n, rank, is_failed, preloaded,
                    buf_pool, full_buf, global_tensor_infos, runtime=runtime,
                    map_to_full_buf=map_to_full_buf,
                    clone_runtime_tensors=clone_runtime_tensors,
                    recovery_role=recovery_role,
                    tensor_views_by_key=tensor_views_by_key,
                )
            except Exception as exc:
                if runtime is not None:
                    for job in jobs:
                        if job.layer_idx >= 0:
                            runtime.mark_layer_error(
                                job.layer_idx, f"{type(exc).__name__}: {exc}"
                            )
                raise
        except BaseException as exc:
            error_holder["error"] = exc
        finally:
            if (
                error_holder["error"] is None
                and _frcheck_recovery_async_parity_enabled()
            ):
                _start_recovery_parity_repair_submissions(
                    "after_recovery_worker", recovery_role
                )
            if cleanup_after:
                _teardown_frcheck_native_after_load()
            _frcheck_recovery_profile(
                recovery_role, "worker_done", jobs=len(jobs),
                elapsed_s=time.time() - t_worker, cleanup_after=cleanup_after,
            )

    worker = threading.Thread(
        target=_worker,
        name=f"frcheck-layer-recovery-rank{rank}",
        daemon=False,
    )
    setattr(worker, "_frcheck_error_holder", error_holder)
    setattr(worker, "_frcheck_recovery_role", recovery_role)
    setattr(worker, "_frcheck_job_count", len(jobs))
    worker.start()
    return worker


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
    block_cache: Optional[Dict[Tuple[str, int, int], torch.Tensor]] = None,
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

    cache = block_cache if block_cache is not None else {}
    stripe_plans = FRCheckManager().stripe_plans
    source_block_idx_by_sid: Dict[int, int] = {}
    for plan in sorted(participating, key=lambda p: int(p['stripe_id'])):
        sid = int(plan['stripe_id'])
        if _stripe_role_for_node(FRCheckManager(), sid, my_node, FRCheckManager().frcheck_n) == int(StripeRole.SOURCE):
            source_block_idx_by_sid[sid] = len(source_block_idx_by_sid)
    n_source = (
        sum(1 for plan in stripe_plans if my_node in plan.source_node_ids)
        if stripe_plans else len(source_block_idx_by_sid)
    )

    def _load_one(job: Tuple[int, Dict]) -> Tuple[int, int, int, torch.Tensor]:
        encode_iter, plan = job
        sid = int(plan['stripe_id'])
        role = _stripe_role_for_node(
            FRCheckManager(), sid, my_node, FRCheckManager().frcheck_n,
        )
        if encode_iter >= len(my_order):
            return encode_iter, sid, role, torch.zeros(1, dtype=torch.uint8)
        layer_name = my_order[encode_iter]
        meta = all_layer_metadata.get(rank, {}).get(layer_name, {})
        layer_block_size = int(
            meta.get("block_size", saved_block_size) or saved_block_size
        )
        sid = int(plan['stripe_id'])
        if role == int(StripeRole.SOURCE):
            blk_idx = (
                _source_blk_idx_for_node_stripe(stripe_plans, int(sid), my_node)
                if stripe_plans else source_block_idx_by_sid.get(int(sid), -1)
            )
            n_filled = _n_filled_blocks_for_layer(
                int(meta.get("actual_tensor_size", 0)),
                layer_block_size,
                n_source,
            )
            if blk_idx >= n_filled:
                return encode_iter, sid, role, torch.zeros(layer_block_size, dtype=torch.uint8)
        cache_key = (layer_name, sid, role)
        cached = cache.get(cache_key)
        if cached is not None and cached.numel() >= layer_block_size:
            blk = cached
        else:
            blk = _read_stripe_block(
                participant_frcheck_dir,
                layer_name,
                sid,
                rank,
                role,
            )
            blk = _normalize_stripe_block(
                blk, layer_block_size, rank, layer_name, sid,
            )
            if cache_key not in cache:
                cache[cache_key] = blk
        return encode_iter, sid, role, blk

    jobs = [
        (encode_iter, plan)
        for encode_iter in range(n_encode_iters)
        for plan in participating
    ]
    n_workers = min(len(jobs), 8)
    loaded_by_role: Dict[int, int] = {}
    loaded_bytes_by_role: Dict[int, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        for encode_iter, sid, role, blk in executor.map(_load_one, jobs):
            result[encode_iter][sid] = blk
            loaded_by_role[role] = loaded_by_role.get(role, 0) + 1
            loaded_bytes_by_role[role] = loaded_bytes_by_role.get(role, 0) + int(blk.numel())
    if _frcheck_debug_enabled():
        logger.info(
            "FRCheck recovery preload rank=%d: source=%d encoder_p1=%d parity_p2=%d "
            "source_bytes=%d encoder_p1_bytes=%d parity_p2_bytes=%d",
            rank,
            loaded_by_role.get(int(StripeRole.SOURCE), 0),
            loaded_by_role.get(int(StripeRole.ENCODER), 0),
            loaded_by_role.get(int(StripeRole.PARITY_TARGET), 0),
            loaded_bytes_by_role.get(int(StripeRole.SOURCE), 0),
            loaded_bytes_by_role.get(int(StripeRole.ENCODER), 0),
            loaded_bytes_by_role.get(int(StripeRole.PARITY_TARGET), 0),
        )
    return result


def _prep_survivor_hw_disk(
    manager: FRCheckManager,
    checkpoint_dir: Path,
    rank: int,
    main_payload: Dict[str, Any],
    n_encode_iters: int,
    all_layer_order: Dict[int, List[str]],
    all_layer_metadata: Dict[int, Dict[str, Dict[str, Any]]],
    recovery_stripe_plans: List[Dict],
    saved_block_size: int,
    my_node: int,
    all_frcheck_dirs: List[Optional[Path]],
    preload_recovery: bool,
) -> Tuple[torch.Tensor, Dict[int, Dict[int, torch.Tensor]]]:
    """Assemble local full_buf and recovery preload blocks before network recovery."""
    total_tensor_size = int(main_payload.get("actual_tensor_size", 0))
    safety_margin = max(int(total_tensor_size * 0.01), 4096)
    full_buf = manager.allocate_full_buf(total_tensor_size + safety_margin)
    full_buf.zero_()

    cache_source_keys: Set[Tuple[str, int, int]] = set()
    my_order = all_layer_order.get(rank, [])
    if preload_recovery and recovery_stripe_plans:
        participating = [
            p for p in recovery_stripe_plans
            if my_node == p['decoder_node'] or my_node in p['helper_nodes']
        ]
        for encode_iter in range(n_encode_iters):
            if encode_iter >= len(my_order):
                continue
            layer_name = my_order[encode_iter]
            for plan in participating:
                sid = int(plan['stripe_id'])
                role = _stripe_role_for_node(manager, sid, my_node, manager.frcheck_n)
                if role == int(StripeRole.SOURCE):
                    cache_source_keys.add((layer_name, sid, role))

    block_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}
    _build_full_buf_from_local_blocks(
        checkpoint_dir,
        rank,
        main_payload,
        out_buf=full_buf,
        source_block_cache=block_cache,
        cache_source_keys=cache_source_keys,
    )

    preloaded: Dict[int, Dict[int, torch.Tensor]] = {
        i: {} for i in range(n_encode_iters)
    }
    if preload_recovery:
        preloaded = _preload_recovery_stripe_blocks(
            n_encode_iters,
            all_layer_order,
            all_layer_metadata,
            recovery_stripe_plans,
            rank,
            my_node,
            saved_block_size,
            all_frcheck_dirs,
            block_cache=block_cache,
        )
    return full_buf, preloaded


# ---------------------------------------------------------------------------
# Per-layer recovery pipeline (C++ batch network)
# ---------------------------------------------------------------------------

def _is_data_recovery_plan(plan: Dict[str, Any], n: int) -> bool:
    """Return True if this recovery plan restores training data, not parity."""
    if plan.get('dual_failure'):
        return True
    return int(plan.get('failed_pos', n)) < n - 2


def _stripe_role_for_node(manager, sid: int, node_id: int, n: int) -> int:
    row = manager.stripe_plans[int(sid)].row
    pos = row.index(int(node_id))
    if pos < n - 2:
        return int(StripeRole.SOURCE)
    if pos == n - 2:
        return int(StripeRole.ENCODER)
    return int(StripeRole.PARITY_TARGET)


def _recovery_window_rows(n: int) -> int:
    return max(1, n)


def _build_recovery_layer_context(
    manager,
    n: int,
    layer_total_bytes: int,
    layer_block_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, bool], int, int]:
    raw_data_plans = [
        p for p in manager.recovery_stripe_plans
        if _is_data_recovery_plan(p, n)
    ]
    parity_plans = [
        p for p in manager.recovery_stripe_plans
        if not _is_data_recovery_plan(p, n)
    ]
    num_source_stripes = (n - 1) * (n - 2)
    n_filled_blocks = _n_filled_blocks_for_layer(
        layer_total_bytes, layer_block_size, num_source_stripes,
    )
    active_by_stripe: Dict[int, bool] = {}
    skipped_padding_stripes = 0
    enable_padding_skip = os.environ.get("FRCHECK_RECOVERY_SKIP_PADDING", "0") == "1"
    for plan in raw_data_plans:
        active = True
        sid = int(plan['stripe_id'])
        if plan.get('dual_failure'):
            # Match save's source-block indexing exactly instead of relying on
            # recovery-plan order, which can differ from POA stripe order.
            for target in plan.get('failed_targets', []):
                if int(target.get('original_role', -1)) != int(StripeRole.SOURCE):
                    continue
                failed_node = int(target.get('failed_node', -1))
                blk_idx = _source_blk_idx_for_node_stripe(
                    manager.stripe_plans, sid, failed_node,
                )
                if blk_idx >= n_filled_blocks:
                    skipped_padding_stripes += 1
                    active = not enable_padding_skip
                    break
        else:
            original_role = int(plan.get('original_role', -1))
            if original_role == int(StripeRole.SOURCE):
                failed_node = int(plan.get('failed_node', -1))
                blk_idx = _source_blk_idx_for_node_stripe(
                    manager.stripe_plans, sid, failed_node,
                )
                if blk_idx >= n_filled_blocks:
                    skipped_padding_stripes += 1
                    active = not enable_padding_skip
        active_by_stripe[sid] = active
    return raw_data_plans, parity_plans, active_by_stripe, skipped_padding_stripes, n_filled_blocks


def _prepare_recovery_layer_buffers(
    native,
    manager,
    n: int,
    layer_idx: int,
    layer_block_size: int,
    data_plans: List[Dict[str, Any]],
    preloaded_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool],
) -> Tuple[Dict[int, torch.Tensor], Optional[torch.Tensor], int, int, int]:
    my_node = manager.rank_in_group + 1
    num_source_stripes = (n - 1) * (n - 2)
    decoder_stripes = [p for p in data_plans if my_node == p['decoder_node']]
    helper_stripes = [p for p in data_plans if my_node in p['helper_nodes']]
    failed_stripes = [p for p in data_plans if _is_failed_in_recovery_plan(p, my_node)]

    my_blocks: Dict[int, torch.Tensor] = {}
    for plan in decoder_stripes + helper_stripes:
        sid = plan['stripe_id']
        blk = preloaded_blocks.get(sid)
        if blk is None:
            blk = torch.zeros(layer_block_size, dtype=torch.uint8)
            native.register_recovery_buffer(blk.data_ptr(), blk.numel())
        my_blocks[sid] = blk

    layer_buf = None
    if failed_stripes:
        stable_buf = None
        if buf_pool is not None and buf_pool.stable_failed_layer_bufs is not None:
            stable_buf = buf_pool.stable_failed_layer_bufs.get(layer_idx)
        if stable_buf is not None:
            layer_buf = stable_buf[: num_source_stripes * layer_block_size]
        elif buf_pool is not None and buf_pool.failed_layer_buf is not None:
            layer_buf = buf_pool.failed_layer_buf[: num_source_stripes * layer_block_size]
        else:
            layer_buf = allocate_hugepage_tensor(
                num_source_stripes * layer_block_size, fallback_pin_memory=True,
            )
            native.register_recovery_buffer(layer_buf.data_ptr(), layer_buf.numel())

    return my_blocks, layer_buf, len(decoder_stripes), len(helper_stripes), len(failed_stripes)


def _recovery_original_role_for_node(plan: Dict[str, Any], my_node: int) -> Optional[int]:
    if plan.get('dual_failure'):
        for target in plan.get('failed_targets', []):
            if target.get('failed_node') == my_node:
                return target.get('original_role')
        return None
    return plan.get('original_role')


def _submit_recovery_window(
    native,
    manager,
    window: _FRCheckRecoveryWindow,
    my_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool],
    n: int,
) -> Dict[str, float]:
    my_node = manager.rank_in_group + 1
    job = window.job
    t_reset = time.time()
    batch_id = native.begin_recovery_batch()
    reset_s = time.time() - t_reset
    t_submit = time.time()
    wave_active_stripes = 0
    for submit_idx, stri_plan in enumerate(window.plans):
        sid = stri_plan['stripe_id']
        active = window.active_by_stripe.get(int(sid), True)
        if active:
            wave_active_stripes += 1
        buf_slot = window.slot_start + submit_idx
        helper_block_addr = 0
        decoder_self_block_addr = 0
        decoder_helper_recv_addrs: List[int] = []
        decoder_recovered_addrs: List[int] = []
        failed_recv_buf_addr = 0
        failed_layer_buf_addr = 0
        failed_layer_offset = 0
        failed_ncopy = 0
        store_to_layer_buf = False

        if my_node in stri_plan['helper_nodes']:
            blk = my_blocks.get(sid)
            if blk is not None:
                helper_block_addr = blk.data_ptr()

        if my_node == stri_plan['decoder_node']:
            blk = my_blocks.get(sid)
            if blk is not None:
                decoder_self_block_addr = blk.data_ptr()
            if buf_pool is not None:
                recv_bufs = (
                    buf_pool.decoder_recv_buf_slots[buf_slot]
                    if buf_slot < len(buf_pool.decoder_recv_buf_slots)
                    else buf_pool.decoder_recv_bufs
                )
                for hi in range(len(stri_plan['helper_nodes'])):
                    if hi < len(recv_bufs):
                        rb = recv_bufs[hi][:job.layer_block_size]
                        decoder_helper_recv_addrs.append(rb.data_ptr())
                if buf_slot < len(buf_pool.decoder_recovered_bufs):
                    rec_buf = buf_pool.decoder_recovered_bufs[buf_slot]
                    decoder_recovered_addrs.append(
                        rec_buf[:job.layer_block_size].data_ptr()
                    )
                if (
                    stri_plan.get('dual_failure')
                    and buf_slot < len(buf_pool.decoder_recovered_buf2s)
                ):
                    rec_buf2 = buf_pool.decoder_recovered_buf2s[buf_slot]
                    decoder_recovered_addrs.append(
                        rec_buf2[:job.layer_block_size].data_ptr()
                    )

        if _is_failed_in_recovery_plan(stri_plan, my_node):
            if window.layer_buf is not None:
                failed_layer_buf_addr = window.layer_buf.data_ptr()
            original_role = _recovery_original_role_for_node(stri_plan, my_node)
            if active and window.layer_buf is not None and original_role == int(StripeRole.SOURCE):
                store_to_layer_buf = True
                blk_idx = window.source_block_indices.get((int(sid), my_node))
                if blk_idx is None:
                    raise RuntimeError(
                        "FRCheck recovery: missing source block index for "
                        f"stripe={sid} failed_node={my_node}"
                    )
                failed_layer_offset = blk_idx * job.layer_block_size
                failed_ncopy = min(
                    job.layer_block_size,
                    max(0, job.actual_size - failed_layer_offset),
                )
                if failed_ncopy == job.layer_block_size:
                    failed_recv_buf_addr = (
                        window.layer_buf[
                            failed_layer_offset:
                            failed_layer_offset + job.layer_block_size
                        ].data_ptr()
                    )
                    store_to_layer_buf = False
            if (
                failed_recv_buf_addr == 0
                and buf_pool is not None
                and buf_slot < len(buf_pool.failed_recv_bufs)
            ):
                failed_recv_buf_addr = (
                    buf_pool.failed_recv_bufs[buf_slot][:job.layer_block_size].data_ptr()
                )

        native.submit_recovery_stripe_to_batch(
            batch_id,
            sid,
            job.layer_block_size,
            helper_block_addr,
            decoder_self_block_addr,
            decoder_helper_recv_addrs,
            decoder_recovered_addrs,
            failed_recv_buf_addr,
            failed_layer_buf_addr,
            failed_layer_offset,
            failed_ncopy,
            store_to_layer_buf,
            active,
        )

    native.end_recovery_batch(batch_id)
    submit_s = time.time() - t_submit
    return {
        "batch_id": int(batch_id),
        "reset_s": reset_s,
        "submit_s": submit_s,
        "wait_s": 0.0,
        "active_stripes": float(wave_active_stripes),
    }


def _submit_recovery_network(
    native,
    manager,
    layer_name: str,
    layer_idx: int,
    layer_block_size: int,
    layer_total_bytes: int,
    n: int,
    rank: int,
    preloaded_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool] = None,
    recovery_role: str = "unknown",
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Submit one layer of stripe recovery via C++ batch pipeline."""
    my_node = manager.rank_in_group + 1
    layer_timing: Dict[str, float] = {
        'rdma_xfer_s': 0.0, 'decode_s': 0.0, 'network_batch_s': 0.0,
        'submit_s': 0.0, 'wait_s': 0.0, 'waves': 0,
    }

    if not manager.recovery_stripe_plans:
        return None, layer_timing

    raw_data_plans = [
        p for p in manager.recovery_stripe_plans
        if _is_data_recovery_plan(p, n)
    ]
    parity_plans = [
        p for p in manager.recovery_stripe_plans
        if not _is_data_recovery_plan(p, n)
    ]

    _dbg = _frcheck_debug_enabled()
    num_source_stripes = (n - 1) * (n - 2)
    n_filled_blocks = _n_filled_blocks_for_layer(
        layer_total_bytes, layer_block_size, num_source_stripes,
    )
    data_plans: List[Dict[str, Any]] = []
    skipped_padding_stripes = 0
    active_by_stripe: Dict[int, bool] = {}
    enable_padding_skip = os.environ.get("FRCHECK_RECOVERY_SKIP_PADDING", "0") == "1"
    for plan in raw_data_plans:
        active = True
        sid = int(plan['stripe_id'])
        if plan.get('dual_failure'):
            for target in plan.get('failed_targets', []):
                if int(target.get('original_role', -1)) != int(StripeRole.SOURCE):
                    continue
                failed_node = int(target.get('failed_node', -1))
                blk_idx = _source_blk_idx_for_node_stripe(
                    manager.stripe_plans, sid, failed_node,
                )
                if blk_idx >= n_filled_blocks:
                    skipped_padding_stripes += 1
                    active = not enable_padding_skip
                    break
        else:
            original_role = int(plan.get('original_role', -1))
            if original_role == int(StripeRole.SOURCE):
                failed_node = int(plan.get('failed_node', -1))
                blk_idx = _source_blk_idx_for_node_stripe(
                    manager.stripe_plans, sid, failed_node,
                )
                if blk_idx >= n_filled_blocks:
                    skipped_padding_stripes += 1
                    active = not enable_padding_skip
        active_by_stripe[sid] = active
        data_plans.append(plan)

    if not data_plans:
        if _dbg:
            logger.info(
                "FRCheck recovery layer %s: rank %d has no data recovery stripes "
                "(skipped parity stripes=%d padding stripes=%d)",
                layer_name, rank, len(parity_plans), skipped_padding_stripes,
            )
        return None, layer_timing

    decoder_stripes = [p for p in data_plans if my_node == p['decoder_node']]
    helper_stripes = [p for p in data_plans if my_node in p['helper_nodes']]
    failed_stripes = [
        p for p in data_plans if _is_failed_in_recovery_plan(p, my_node)
    ]
    is_failed = len(failed_stripes) > 0

    if _dbg:
        logger.info(
            "FRCheck recovery layer %s: rank %d — data=%d raw_data=%d "
            "parity_skipped=%d padding_skipped=%d filled=%d/%d "
            "decoder=%d helper=%d failed=%d stripes",
            layer_name, rank, len(data_plans), len(raw_data_plans),
            len(parity_plans), skipped_padding_stripes,
            n_filled_blocks, num_source_stripes,
            len(decoder_stripes), len(helper_stripes), len(failed_stripes),
        )

    my_blocks: Dict[int, torch.Tensor] = {}
    for plan in decoder_stripes + helper_stripes:
        sid = plan['stripe_id']
        blk = preloaded_blocks.get(sid)
        if blk is None:
            blk = torch.zeros(layer_block_size, dtype=torch.uint8)
            native.register_recovery_buffer(blk.data_ptr(), blk.numel())
        my_blocks[sid] = blk

    layer_buf = None
    if is_failed:
        stable_buf = None
        if buf_pool is not None and buf_pool.stable_failed_layer_bufs is not None:
            stable_buf = buf_pool.stable_failed_layer_bufs.get(layer_idx)
        if stable_buf is not None:
            layer_buf = stable_buf[: num_source_stripes * layer_block_size]
        elif buf_pool is not None and buf_pool.failed_layer_buf is not None:
            layer_buf = buf_pool.failed_layer_buf[: num_source_stripes * layer_block_size]
        else:
            layer_buf = allocate_hugepage_tensor(
                num_source_stripes * layer_block_size, fallback_pin_memory=True,
            )
            native.register_recovery_buffer(layer_buf.data_ptr(), layer_buf.numel())

    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}

    wave_size = buf_pool.concurrency if buf_pool is not None else max(n, 1)
    t_network = time.time()
    for wave_start in range(0, len(data_plans), wave_size):
        wave_plans = data_plans[wave_start:wave_start + wave_size]
        wave_idx = int(wave_start / wave_size)
        layer_timing['waves'] += 1
        t_reset = time.time()
        native.reset_recovery_batch()
        reset_s = time.time() - t_reset
        t_submit = time.time()
        wave_active_stripes = 0
        for submit_idx, stri_plan in enumerate(wave_plans):
            sid = stri_plan['stripe_id']
            active = active_by_stripe.get(int(sid), True)
            if active:
                wave_active_stripes += 1
            buf_slot = submit_idx
            helper_block_addr = 0
            decoder_self_block_addr = 0
            decoder_helper_recv_addrs: List[int] = []
            decoder_recovered_addrs: List[int] = []
            failed_recv_buf_addr = 0
            failed_layer_buf_addr = 0
            failed_layer_offset = 0
            failed_ncopy = 0
            store_to_layer_buf = False

            if my_node in stri_plan['helper_nodes']:
                blk = my_blocks.get(sid)
                if blk is not None:
                    helper_block_addr = blk.data_ptr()

            if my_node == stri_plan['decoder_node']:
                blk = my_blocks.get(sid)
                if blk is not None:
                    decoder_self_block_addr = blk.data_ptr()
                if buf_pool is not None:
                    recv_bufs = (
                        buf_pool.decoder_recv_buf_slots[buf_slot]
                        if buf_slot < len(buf_pool.decoder_recv_buf_slots)
                        else buf_pool.decoder_recv_bufs
                    )
                    for hi in range(len(stri_plan['helper_nodes'])):
                        if hi < len(recv_bufs):
                            rb = recv_bufs[hi][:layer_block_size]
                            decoder_helper_recv_addrs.append(rb.data_ptr())
                    if buf_slot < len(buf_pool.decoder_recovered_bufs):
                        rec_buf = buf_pool.decoder_recovered_bufs[buf_slot]
                        decoder_recovered_addrs.append(
                            rec_buf[:layer_block_size].data_ptr()
                        )
                    if (
                        stri_plan.get('dual_failure')
                        and buf_slot < len(buf_pool.decoder_recovered_buf2s)
                    ):
                        rec_buf2 = buf_pool.decoder_recovered_buf2s[buf_slot]
                        decoder_recovered_addrs.append(
                            rec_buf2[:layer_block_size].data_ptr()
                        )

            if _is_failed_in_recovery_plan(stri_plan, my_node):
                if layer_buf is not None:
                    failed_layer_buf_addr = layer_buf.data_ptr()

                if stri_plan.get('dual_failure'):
                    original_role = None
                    for target in stri_plan['failed_targets']:
                        if target['failed_node'] == my_node:
                            original_role = target['original_role']
                            break
                else:
                    original_role = stri_plan.get('original_role')

                if (
                    active
                    and layer_buf is not None
                    and original_role == int(StripeRole.SOURCE)
                ):
                    store_to_layer_buf = True
                    blk_idx = src_block_per_node[my_node]
                    src_block_per_node[my_node] += 1
                    failed_layer_offset = blk_idx * layer_block_size
                    failed_ncopy = min(
                        layer_block_size,
                        max(0, layer_total_bytes - failed_layer_offset),
                    )
                    if failed_ncopy == layer_block_size:
                        failed_recv_buf_addr = (
                            layer_buf[
                                failed_layer_offset:
                                failed_layer_offset + layer_block_size
                            ].data_ptr()
                        )
                        store_to_layer_buf = False

                if (
                    failed_recv_buf_addr == 0
                    and buf_pool is not None
                    and buf_slot < len(buf_pool.failed_recv_bufs)
                ):
                    failed_recv_buf_addr = (
                        buf_pool.failed_recv_bufs[buf_slot][:layer_block_size].data_ptr()
                    )

            native.submit_recovery_stripe(
                sid,
                layer_block_size,
                helper_block_addr,
                decoder_self_block_addr,
                decoder_helper_recv_addrs,
                decoder_recovered_addrs,
                failed_recv_buf_addr,
                failed_layer_buf_addr,
                failed_layer_offset,
                failed_ncopy,
                store_to_layer_buf,
                active,
            )

        native.submit_recovery_sentinel()
        submit_s = time.time() - t_submit
        t_wait = time.time()
        native.wait_recovery_batch()
        wait_s = time.time() - t_wait
        layer_timing['submit_s'] += submit_s + reset_s
        layer_timing['wait_s'] += wait_s
        _frcheck_recovery_profile(
            recovery_role, "network_wave_done", layer=layer_name,
            layer_idx=layer_idx, wave=wave_idx, stripes=len(wave_plans),
            wave_size=wave_size, pool_concurrency=(buf_pool.concurrency if buf_pool is not None else 0),
            active_stripes=wave_active_stripes,
            skipped_padding_stripes=len(wave_plans) - wave_active_stripes,
            reset_s=reset_s, submit_s=submit_s, wait_s=wait_s,
        )

    layer_timing['network_batch_s'] = time.time() - t_network
    layer_timing['rdma_xfer_s'] = layer_timing['network_batch_s']
    _frcheck_recovery_profile(
        recovery_role, "network_batch_done", layer=layer_name,
        layer_idx=layer_idx, stripes=len(data_plans),
        raw_stripes=len(raw_data_plans),
        active_stripes=sum(1 for plan in data_plans if active_by_stripe.get(int(plan['stripe_id']), True)),
        skipped_padding_stripes=(
            sum(1 for plan in data_plans if not active_by_stripe.get(int(plan['stripe_id']), True))
        ),
        padding_candidate_stripes=skipped_padding_stripes,
        padding_skip_enabled=enable_padding_skip,
        n_filled_blocks=n_filled_blocks,
        decoder_stripes=len(decoder_stripes), helper_stripes=len(helper_stripes),
        failed_stripes=len(failed_stripes), waves=layer_timing['waves'],
        wave_size=wave_size, pool_concurrency=(buf_pool.concurrency if buf_pool is not None else 0),
        network_batch_s=layer_timing['network_batch_s'],
        submit_s=layer_timing['submit_s'], wait_s=layer_timing['wait_s'],
    )

    if is_failed:
        n_stored = src_block_per_node.get(my_node, 0)
        expected_stored = sum(
            1 for plan in failed_stripes
            if int(plan.get('original_role', -1)) == int(StripeRole.SOURCE)
            and active_by_stripe.get(int(plan['stripe_id']), True)
        )
        if _dbg:
            logger.info(
                "FRCheck recovery: %s — stored %d SOURCE blocks (expected %d)",
                layer_name, n_stored, expected_stored,
            )
        if n_stored != expected_stored:
            raise RuntimeError(
                f"FRCheck recovery: layer {layer_name} stored {n_stored} "
                f"SOURCE blocks, expected {expected_stored}"
            )

    return layer_buf, layer_timing


def _recover_one_layer_network(
    manager,
    native,
    layer_name: str,
    layer_idx: int,
    layer_block_size: int,
    layer_total_bytes: int,
    n: int,
    rank: int,
    preloaded_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool] = None,
    recovery_role: str = "unknown",
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Run stripe-level RS decode recovery for one layer via C++ batch pipeline."""
    return _submit_recovery_network(
        native, manager, layer_name, layer_idx, layer_block_size, layer_total_bytes,
        n, rank, preloaded_blocks, buf_pool, recovery_role=recovery_role,
    )


# ---------------------------------------------------------------------------
# Training exit teardown (ECLATIN-style: explicit stop after eval, not in __del__)
# ---------------------------------------------------------------------------

def _teardown_frcheck_after_training() -> None:
    """Synchronized FRCheck teardown at end of training/eval."""
    from megatron.training import get_args
    args = get_args()
    if not getattr(args, "use_frcheck", False):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    debug = _frcheck_debug_enabled()
    if debug:
        logger.info("FRCHECK teardown trace rank=%d: enter FRCheck teardown", rank)
    service = _get_active_frcheck_recovery_service()
    manager = FRCheckManager()
    if service.safe_point_teardown_done and manager.get_native() is None:
        if debug:
            logger.info("FRCHECK teardown trace rank=%d: teardown already completed", rank)
        return
    if debug:
        logger.info("FRCHECK teardown trace rank=%d: before recovery worker wait", rank)
    frcheck_wait_for_async_recovery()
    if debug:
        logger.info("FRCHECK teardown trace rank=%d: after recovery worker wait", rank)
    if manager.get_native() is None:
        return
    if debug:
        logger.info("FRCheck: tearing down native module after training (rank %d)", rank)
    _wait_previous_async_writers(debug=debug, rank=rank)
    if debug:
        logger.info("FRCHECK teardown trace rank=%d: save writers drained", rank)
    _finish_recovery_parity_repair_submissions("after_training", service.role)
    _flush_recovery_async_parity("after_training", service.role)
    if debug:
        logger.info("FRCHECK teardown trace rank=%d: before manager cleanup", rank)
    manager.cleanup(teardown=True)
    if debug:
        logger.info("FRCHECK teardown trace rank=%d: after manager cleanup", rank)


# ---------------------------------------------------------------------------
# Load teardown (ECLATIN-style: explicit cleanup after recovery only)
# ---------------------------------------------------------------------------

def _teardown_frcheck_native_after_load() -> None:
    """Stop and release native module after load/recovery (not used on save path)."""
    _wait_previous_async_writers(debug=_frcheck_debug_enabled())
    manager = FRCheckManager()
    if manager.get_native() is None:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    role = getattr(manager, "_frcheck_recovery_role", "unknown")
    service = _get_active_frcheck_recovery_service()
    if service.worker is not threading.current_thread():
        service.wait_all(reason="load_teardown")
    _finish_recovery_parity_repair_submissions("load_teardown", role)
    try:
        from megatron.training import get_args
        args = get_args()
    except Exception:
        args = None

    inprocess_reuse = bool(
        args is not None
        and getattr(args, "ft_inprocess_recovery_benchmark", False)
    )
    if inprocess_reuse:
        _flush_recovery_async_parity("load_teardown_inprocess_reuse", role)
        manager.end_recovery()
        _frcheck_recovery_profile(role, "teardown_skipped_inprocess_reuse")
        if _frcheck_debug_enabled():
            logger.info(
                "FRCheck legacy load: keeping native module alive for in-process reuse (rank %d)",
                rank,
            )
        return

    defer_teardown = bool(
        args is not None
        and getattr(args, "frcheck_defer_load_teardown", False)
    )
    if defer_teardown:
        _frcheck_recovery_profile(role, "teardown_deferred")
        if _frcheck_debug_enabled():
            logger.info(
                "FRCheck legacy load: deferred native teardown after load (rank %d)",
                rank,
            )
        return
    skip_barrier = bool(
        args is not None
        and getattr(args, "frcheck_skip_load_teardown_barrier", False)
    )
    _flush_recovery_async_parity("load_teardown", role)
    _frcheck_recovery_profile(role, "teardown_start", sync=not skip_barrier)
    t_cleanup = time.time()
    if _frcheck_debug_enabled():
        logger.info("FRCheck legacy load: cleaning up native module (rank %d)", rank)
    recovery_only = bool(
        args is not None
        and getattr(args, "frcheck_recovery_only_teardown", False)
    )
    if recovery_only and hasattr(manager, "cleanup_recovery"):
        manager.cleanup_recovery(teardown=True, sync=not skip_barrier)
    else:
        manager.cleanup(teardown=True, sync=not skip_barrier)
    teardown_s = time.time() - t_cleanup
    try:
        from megatron.training.global_vars import add_recovery_teardown_time
        add_recovery_teardown_time(teardown_s)
    except Exception:
        pass
    _frcheck_recovery_profile(
        role, "teardown_done", elapsed_s=teardown_s,
        sync=not skip_barrier,
    )


# ---------------------------------------------------------------------------
# Main recovery entry point
# ---------------------------------------------------------------------------

def recover_frcheck_legacy_hardware(
    checkpoint_name: str,
    failed_global_ranks: List[int],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Main entry point for FRCheck hardware recovery.

    Phases (timed separately):
    1. prep — RDMA init, main I/O, disk assemble + recovery preload (before network_encode)
    2. network_encode — per-layer RDMA RS recovery only
    3. rebuild_sd — metadata-only state_dict reconstruct (survivors use prep full_buf)
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    _dbg = _frcheck_debug_enabled()

    timings: Dict[str, float] = {
        'prep': 0.0,
        'main_io': 0.0,
        'disk_io': 0.0,
        'network_encode': 0.0,
        'network_submit': 0.0,
        'network_wait': 0.0,
        'materialize': 0.0,
        'pipeline_overlap': 0.0,
        'pipeline_critical': 0.0,
        'rebuild_sd': 0.0,
    }
    t_prep = time.time()

    manager = FRCheckManager()
    manager.init_frcheck_if_enabled()
    native = manager.get_native()
    if native is None:
        raise RuntimeError("FRCheck hardware recovery: native module not available")

    n = native.n()
    rg = manager.rank_in_group
    gdr = True

    if _dbg:
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
    # lack local main.pt).
    t_main = time.time()
    main_payload = _load_frcheck_main_payload(
        checkpoint_dir, rank, world_size, load_tensor_buffer=False,
    )
    timings['main_io'] = time.time() - t_main

    all_tensor_infos, all_layer_order, all_layer_metadata = _frcheck_metadata_from_payload(
        main_payload, primary_failed,
    )
    optimizer_layer_map = _optimizer_layer_map_for_rank(main_payload, rank)
    main_payload["optimizer_layer_map"] = optimizer_layer_map
    setattr(manager, "_frcheck_optimizer_layer_map", optimizer_layer_map)

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
    if _dbg:
        logger.info(
            "FRCheck recovery: main loaded encode_iters=%d group_layer_orders=%s",
            n_encode_iters,
            {r: all_layer_order.get(r, []) for r in group_members},
        )

    all_frcheck_dirs = _gather_all_frcheck_dirs(checkpoint_dir, world_size)
    involved = is_failed or bool(manager.recovery_stripe_plans)
    my_node = manager.rank_in_group + 1
    data_recovery_plans = [
        p for p in manager.recovery_stripe_plans
        if _is_data_recovery_plan(p, n)
    ]
    parity_recovery_plans = [
        p for p in manager.recovery_stripe_plans
        if not _is_data_recovery_plan(p, n)
    ]
    if manager.recovery_stripe_plans and _dbg:
        logger.info(
            "FRCheck recovery: phase1 data-only mode data_stripes=%d "
            "parity_stripes_skipped=%d",
            len(data_recovery_plans), len(parity_recovery_plans),
        )

    t_disk = time.time()
    preloaded: Dict[int, Dict[int, torch.Tensor]] = {
        i: {} for i in range(n_encode_iters)
    }
    full_buf = None
    if is_survivor:
        full_buf, preloaded = _prep_survivor_hw_disk(
            manager,
            checkpoint_dir,
            rank,
            main_payload,
            n_encode_iters,
            all_layer_order,
            all_layer_metadata,
            data_recovery_plans,
            saved_block_size,
            my_node,
            all_frcheck_dirs,
            preload_recovery=bool(data_recovery_plans),
        )
        for blocks in preloaded.values():
            for blk in blocks.values():
                if blk.numel() > 1:
                    native.register_recovery_buffer(blk.data_ptr(), blk.numel())
    timings['disk_io'] = time.time() - t_disk

    if is_survivor and not involved and _dbg:
        logger.info(
            "FRCheck recovery: rank %d not involved in recovery, local load only",
            rank,
        )

    max_block_size = _max_layer_block_size(
        rank, all_layer_order, all_layer_metadata, saved_block_size,
    )
    role_plans_for_buffers = (
        manager.recovery_stripe_plans
        if _frcheck_recovery_parity_repair_enabled()
        else data_recovery_plans
    )
    is_decoder = any(
        my_node == p['decoder_node'] for p in role_plans_for_buffers
    )
    is_helper = any(
        my_node in p['helper_nodes'] for p in role_plans_for_buffers
    )
    recovery_role = _frcheck_recovery_role(is_failed, is_decoder, is_helper, involved)
    setattr(manager, "_frcheck_recovery_role", recovery_role)
    _get_active_frcheck_recovery_service().reset_for_load(recovery_role)
    _frcheck_recovery_profile(
        recovery_role, "rank_role", is_failed=is_failed, is_decoder=is_decoder,
        is_helper=is_helper, involved=involved, data_stripes=len(data_recovery_plans),
        parity_stripes=len(parity_recovery_plans), encode_iters=n_encode_iters,
        disk_io_s=timings['disk_io'],
    )
    try:
        from megatron.training.global_vars import update_recovery_to_forward_timer_context
        update_recovery_to_forward_timer_context(role=recovery_role)
    except Exception:
        pass

    buf_pool: Optional[_RecoveryBufPool] = None

    layerwise_records: List[_FRCheckLayerReadyRecord] = []
    direct_tensor_views_by_key: Optional[Dict[str, torch.Tensor]] = {} if is_failed else None

    async_forward = False
    async_detach_safe = False
    try:
        from megatron.training import get_args
        args = get_args()
        async_forward = bool(getattr(args, "frcheck_async_recovery_forward", False))
        async_detach_safe = True
    except Exception:
        async_forward = False
        async_detach_safe = False
    runtime_for_recovery: Optional[_FRCheckLayerwiseRuntime] = None
    if async_forward and is_failed:
        pending_records = _make_pending_layerwise_records(
            rank, is_failed, main_payload, all_layer_order, all_layer_metadata,
            all_frcheck_dirs, checkpoint_dir, saved_block_size, n_encode_iters,
            recovery_rank=primary_failed,
        )
        runtime_for_recovery = _FRCheckLayerwiseRuntime(pending_records)
        _set_active_frcheck_layerwise_runtime(runtime_for_recovery)
        if _dbg:
            logger.info(
                "FRCheck recovery: async-forward runtime prepared layers=%d "
                "(worker-backed recovery joins before state_dict rebuild)",
                len(pending_records),
            )

    recovery_jobs: List[_FRCheckLayerRecoveryJob] = []
    for encode_iter in range(n_encode_iters):
        if not involved:
            continue

        job = _make_layer_recovery_job(
            encode_iter, rank, is_failed, main_payload, all_layer_order,
            all_layer_metadata, all_frcheck_dirs, checkpoint_dir, saved_block_size,
            recovery_rank=primary_failed,
        )
        if job is None:
            continue
        recovery_jobs.append(job)

    recovery_jobs = _sort_recovery_jobs_by_forward_priority(recovery_jobs)
    if _dbg:
        logger.info(
            "FRCheck recovery job order rank=%d role=%s: %s",
            rank,
            recovery_role,
            [(job.layer_name, job.layer_idx, job.encode_iter) for job in recovery_jobs],
        )

    buf_pool = _allocate_recovery_buf_pool(
        native, n, max_block_size, is_failed, is_decoder, is_helper,
        dual_failure=getattr(manager, 'recovery_dual_failure', False),
    )

    detached_transformer_recovery = bool(
        async_forward
        and async_detach_safe
        and is_failed
        and runtime_for_recovery is not None
    )
    if direct_tensor_views_by_key is not None:
        _preallocate_stable_failed_layer_bufs(native, buf_pool, recovery_jobs, n)
        _frcheck_recovery_profile(
            recovery_role, "stable_failed_layer_bufs_enabled",
            jobs=len(recovery_jobs), direct_reconstruct=True,
        )
    elif detached_transformer_recovery:
        async_jobs_for_buffers = [job for job in recovery_jobs if job.layer_idx >= 0]
        _preallocate_stable_failed_layer_bufs(native, buf_pool, async_jobs_for_buffers, n)
        _frcheck_recovery_profile(
            recovery_role, "stable_failed_layer_bufs_enabled",
            jobs=len(async_jobs_for_buffers), direct_reconstruct=False,
        )

    if recovery_jobs and _frcheck_recovery_parity_repair_enabled():
        parity_preloaded = _preload_recovery_stripe_blocks(
            n_encode_iters,
            all_layer_order,
            all_layer_metadata,
            parity_recovery_plans,
            rank,
            my_node,
            saved_block_size,
            all_frcheck_dirs,
        )
        for blocks in parity_preloaded.values():
            for blk in blocks.values():
                if blk.numel() > 1:
                    native.register_recovery_buffer(blk.data_ptr(), blk.numel())
        _set_pending_recovery_parity_repair(
            recovery_jobs, buf_pool, checkpoint_dir, rank,
            all_layer_metadata, recovery_role, parity_preloaded=parity_preloaded
        )
    if recovery_jobs and _frcheck_recovery_parity_repair_enabled():
        _frcheck_recovery_profile(
            recovery_role,
            "recovery_parity_repair_prepared",
            mode=("background" if _frcheck_recovery_async_parity_enabled() else "sync"),
            jobs=len(recovery_jobs),
            parity_stripes=len(parity_recovery_plans),
        )

    # Direct recovery keeps failed-rank tensors in stable per-layer views.
    sync_map_to_full_buf = "none" if direct_tensor_views_by_key is not None else "all"
    sync_clone_runtime_tensors = False
    async_map_to_full_buf = "none"
    async_clone_runtime_tensors = False

    timings['prep'] = time.time() - t_prep

    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    first_recovery_layer_idx = min(
        (job.layer_idx for job in recovery_jobs if job.layer_idx >= 0),
        default=None,
    )
    t_net = time.time()
    _start_frcheck_first_layer_recovery_timer(first_recovery_layer_idx)
    recovery_to_forward_started = False

    def _start_recovery_to_forward_from_pipeline_start() -> None:
        nonlocal recovery_to_forward_started
        if recovery_to_forward_started:
            return
        recovery_to_forward_started = True
        try:
            from megatron.training.global_vars import start_recovery_to_forward_timer
            start_recovery_to_forward_timer(
                "FRCheck", "recovery_pipeline_start",
                role=recovery_role, rank0_only_max=True,
            )
            _arm_frcheck_first_layer_cuda_timer()
        except Exception:
            pass

    def _accumulate_layer_timings(
        layer_results: List[Tuple[Optional[_FRCheckLayerReadyRecord], Dict[str, float]]]
    ) -> None:
        if not layer_results:
            return
        timings['network_submit'] += sum(
            item.get('network_submit_s', item.get('submit_s', 0.0))
            for _record, item in layer_results
        )
        timings['network_wait'] += sum(
            item.get('network_wait_s', item.get('wait_s', 0.0))
            for _record, item in layer_results
        )
        timings['materialize'] += sum(
            item.get('materialize_s', 0.0) for _record, item in layer_results
        )
        timings['pipeline_overlap'] = max(
            timings['pipeline_overlap'],
            max((item.get('pipeline_overlap_s', 0.0) for _record, item in layer_results), default=0.0),
        )
        timings['pipeline_critical'] += max(
            (item.get('pipeline_critical_s', 0.0) for _record, item in layer_results),
            default=0.0,
        )

    if _dbg:
        logger.info(
            "FRCheck recovery async decision: rank=%d async_forward=%s "
            "async_detach_safe=%s is_failed=%s involved=%s jobs=%d "
            "runtime=%s detached=%s",
            rank, async_forward, async_detach_safe, is_failed, involved,
            len(recovery_jobs), runtime_for_recovery is not None,
            detached_transformer_recovery,
        )
    if detached_transformer_recovery:
        sync_jobs = [job for job in recovery_jobs if job.layer_idx < 0]
        async_jobs = [job for job in recovery_jobs if job.layer_idx >= 0]
        if sync_jobs:
            _start_recovery_to_forward_from_pipeline_start()
            sync_results = _run_recovery_pipeline(
                sync_jobs, manager, native, n, rank, is_failed, preloaded,
                buf_pool, full_buf, global_tensor_infos,
                runtime=runtime_for_recovery,
                map_to_full_buf=sync_map_to_full_buf,
                clone_runtime_tensors=sync_clone_runtime_tensors,
                recovery_role=recovery_role,
                tensor_views_by_key=direct_tensor_views_by_key,
            )
            _accumulate_layer_timings(sync_results)
            for record, _layer_timing in sync_results:
                if record is not None:
                    layerwise_records.append(record)
        if async_jobs:
            _start_recovery_to_forward_from_pipeline_start()
            worker = _start_layer_recovery_worker(
                async_jobs, manager, native, n, rank, is_failed, preloaded,
                buf_pool, full_buf, global_tensor_infos, runtime_for_recovery,
                cleanup_after=False,
                map_to_full_buf=async_map_to_full_buf,
                clone_runtime_tensors=async_clone_runtime_tensors,
                recovery_role=recovery_role,
                tensor_views_by_key=None,
                checkpoint_dir=checkpoint_dir,
                all_layer_metadata=all_layer_metadata,
            )
            _set_active_frcheck_recovery_worker(worker)
            if _dbg:
                logger.info(
                    "FRCheck recovery: async-forward detached transformer worker "
                    "started rank=%d sync_jobs=%d async_jobs=%d",
                    rank, len(sync_jobs), len(async_jobs),
                )
            safe_point = getattr(args, "frcheck_recovery_safe_point", "load")
            if _dbg:
                logger.info(
                    "FRCheck recovery: deferring detached transformer worker join "
                    "to safe_point=%s rank=%d",
                    safe_point, rank,
                )
    elif async_forward and involved and recovery_jobs:
        _start_recovery_to_forward_from_pipeline_start()
        worker = _start_layer_recovery_worker(
            recovery_jobs, manager, native, n, rank, is_failed, preloaded,
            buf_pool, full_buf, global_tensor_infos, runtime_for_recovery,
            cleanup_after=False,
            map_to_full_buf=sync_map_to_full_buf,
            clone_runtime_tensors=sync_clone_runtime_tensors,
            recovery_role=recovery_role,
            tensor_views_by_key=direct_tensor_views_by_key,
            checkpoint_dir=checkpoint_dir,
            all_layer_metadata=all_layer_metadata,
        )
        _set_active_frcheck_recovery_worker(worker)
        if _dbg:
            logger.info(
                "FRCheck recovery: async-forward worker started rank=%d jobs=%d",
                rank, len(recovery_jobs),
            )
        safe_point = getattr(args, "frcheck_recovery_safe_point", "load")
        if _dbg:
            logger.info(
                "FRCheck recovery: deferring async worker join to safe_point=%s rank=%d",
                safe_point, rank,
            )
    else:
        _start_recovery_to_forward_from_pipeline_start()
        sync_results = _run_recovery_pipeline(
            recovery_jobs, manager, native, n, rank, is_failed, preloaded,
            buf_pool, full_buf, global_tensor_infos,
            runtime=runtime_for_recovery,
            map_to_full_buf=sync_map_to_full_buf,
            clone_runtime_tensors=sync_clone_runtime_tensors,
            recovery_role=recovery_role,
            tensor_views_by_key=direct_tensor_views_by_key,
        )
        _accumulate_layer_timings(sync_results)
        for record, _layer_timing in sync_results:
            if record is not None:
                layerwise_records.append(record)

    if recovery_jobs and _frcheck_recovery_parity_repair_enabled():
        if _frcheck_recovery_async_parity_enabled():
            if _get_active_frcheck_recovery_service().worker is None:
                _start_recovery_parity_repair_submissions(
                    "after_recovery_data", recovery_role
                )
        else:
            # Sync mode makes parity repair part of recovery-to-forward latency.
            _get_active_frcheck_recovery_service().wait_all(
                reason="sync_parity_repair"
            )
            _run_pending_recovery_parity_repair_sync(
                "after_recovery_data", recovery_role
            )

    timings['network_encode'] = time.time() - t_net

    t_rebuild = time.time()
    try:
        if full_buf is not None:
            main_payload['tensor_buffer'] = full_buf[:total_tensor_size]
        if detached_transformer_recovery:
            result = _reconstruct_common_from_tensor_views(
                main_payload, direct_tensor_views_by_key, flat_key_roots,
            )
        elif direct_tensor_views_by_key is not None:
            result = _reconstruct_from_tensor_views(
                main_payload, direct_tensor_views_by_key, flat_key_roots,
            )
        else:
            result = _reconstruct_from_main_payload(main_payload, flat_key_roots)
    except Exception:
        logger.exception(
            "FRCheck recovery: state_dict rebuild failed rank=%d role=%s "
            "detached=%s direct_views=%s full_buf=%s",
            rank, recovery_role, detached_transformer_recovery,
            direct_tensor_views_by_key is not None, full_buf is not None,
        )
        raise
    if is_failed and (layerwise_records or runtime_for_recovery is not None):
        records_to_export = (
            list(runtime_for_recovery._records_by_layer.values())
            if runtime_for_recovery is not None else layerwise_records
        )
        result["__frcheck_layerwise_runtime__"] = {
            "rank": rank,
            "active_runtime": runtime_for_recovery is not None,
            "records": [
                {
                    "layer_name": r.layer_name,
                    "layer_idx": r.layer_idx,
                    "encode_iter": r.encode_iter,
                    "tensor_keys": r.tensor_keys,
                    "model_tensor_keys": r.model_tensor_keys,
                    "optimizer_tensor_keys": r.optimizer_tensor_keys,
                    "contains_optimizer_state": r.contains_optimizer_state,
                    "nbytes": r.nbytes,
                    "materialize_s": r.materialize_s,
                    "ready": r.ready,
                    "model_ready": r.model_ready,
                    "optimizer_ready": r.optimizer_ready,
                    "error": r.error,
                    "tensors": r.tensors,
                    "model_tensors": r.model_tensors,
                    "optimizer_tensors": r.optimizer_tensors,
                }
                for r in records_to_export
            ],
        }
        if _dbg:
            logger.info(
                "FRCheck recovery: layerwise runtime exported layers=%d "
                "materialize_total=%.4fs",
                len(records_to_export),
                sum(r.materialize_s for r in records_to_export),
            )
    timings['rebuild_sd'] = time.time() - t_rebuild

    safe_point = getattr(args, "frcheck_recovery_safe_point", "load")
    if safe_point == "load" and not detached_transformer_recovery:
        _teardown_frcheck_native_after_load()
    elif safe_point != "load":
        _frcheck_recovery_profile(
            recovery_role, "teardown_moved_to_safe_point", safe_point=safe_point
        )
    return result, timings


def _reconstruct_from_main_payload(main_payload: Dict, flat_key_roots: list = None) -> Dict[str, Any]:
    """Reconstruct state_dict from metadata + assembled tensor_buffer."""
    tensor_infos = main_payload.get("tensor_infos", [])
    non_tensor_data = main_payload.get("non_tensor_data", {})
    flat_key_roots = flat_key_roots or main_payload.get("flat_key_roots", [])

    tensor_buffer = main_payload.get("tensor_buffer")
    if tensor_buffer is None:
        raise RuntimeError(
            "FRCheck load: tensor_buffer missing in main payload; "
            "assemble from local FRBK SOURCE blocks before reconstruct."
        )
    buf = tensor_buffer.detach().contiguous().reshape(-1).view(torch.uint8)
    tensor_data = extract_tensors_from_continuous_buffer(buf, tensor_infos)

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


def _reconstruct_from_tensor_views(
    main_payload: Dict,
    tensor_views_by_key: Dict[str, torch.Tensor],
    flat_key_roots: list = None,
) -> Dict[str, Any]:
    """Reconstruct state_dict from recovered per-layer tensor views."""
    tensor_infos = main_payload.get("tensor_infos", [])
    non_tensor_data = main_payload.get("non_tensor_data", {})
    flat_key_roots = flat_key_roots or main_payload.get("flat_key_roots", [])

    tensor_data: List[torch.Tensor] = []
    missing_keys: List[str] = []
    for info in tensor_infos:
        key = getattr(info, "key", "")
        tensor = tensor_views_by_key.get(key)
        if tensor is not None:
            tensor_data.append(tensor)
            continue
        size = int(getattr(info, "size_bytes", 0))
        dtype = getattr(info, "dtype", None)
        shape = tuple(getattr(info, "shape", ()))
        if dtype is not None and shape and size == 0:
            tensor_data.append(torch.empty(shape, dtype=dtype))
            continue
        missing_keys.append(key)
    if missing_keys:
        raise RuntimeError(
            "FRCheck load: missing tensor views for direct reconstruct: "
            f"{missing_keys[:8]} (total {len(missing_keys)})"
        )

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


def _infer_distributed_common_keys_from_metadata(main_payload: Dict) -> set:
    """Infer common tensors that were distributed into real layer groups."""
    optimizer_layer_map = main_payload.get("optimizer_layer_map", {})
    inferred = set()
    all_layer_metadata = main_payload.get("all_layer_metadata") or {}
    for rank_meta in all_layer_metadata.values():
        if not isinstance(rank_meta, dict):
            continue
        for layer_name, layer_meta in rank_meta.items():
            if not str(layer_name).startswith("layer_"):
                continue
            tensor_infos = layer_meta.get("tensor_infos", []) if isinstance(layer_meta, dict) else []
            for info in tensor_infos:
                key = getattr(info, "key", "")
                if not key:
                    continue
                ownership = _classify_frcheck_tensor(key, optimizer_layer_map)
                if ownership.kind in ("model_common", "optimizer_common"):
                    inferred.add(key)
    return inferred


def _reconstruct_common_from_tensor_views(
    main_payload: Dict,
    tensor_views_by_key: Dict[str, torch.Tensor],
    flat_key_roots: list = None,
) -> Dict[str, Any]:
    """Reconstruct common state; layer tensors are injected by runtime."""
    tensor_infos = main_payload.get("tensor_infos", [])
    non_tensor_data = main_payload.get("non_tensor_data", {})
    flat_key_roots = flat_key_roots or main_payload.get("flat_key_roots", [])
    optimizer_layer_map = main_payload.get("optimizer_layer_map", {})
    distributed_common_keys = set(main_payload.get("distributed_common_keys", []))
    if not distributed_common_keys:
        distributed_common_keys = _infer_distributed_common_keys_from_metadata(main_payload)

    common_infos = []
    common_tensor_data: List[torch.Tensor] = []
    skipped = 0
    skipped_bytes = 0
    deferred = 0
    deferred_bytes = 0
    missing_keys: List[str] = []
    for info in tensor_infos:
        key = getattr(info, "key", "")
        ownership = _classify_frcheck_tensor(key, optimizer_layer_map)
        if ownership.layer_idx >= 0 and ownership.kind in ("model_layer", "optimizer_layer"):
            skipped += 1
            skipped_bytes += int(getattr(info, "size_bytes", 0))
            continue

        tensor = tensor_views_by_key.get(key)
        if tensor is not None:
            common_infos.append(info)
            common_tensor_data.append(tensor)
            continue

        size = int(getattr(info, "size_bytes", 0))
        dtype = getattr(info, "dtype", None)
        shape = tuple(getattr(info, "shape", ()))
        if dtype is not None and shape and size == 0:
            common_infos.append(info)
            common_tensor_data.append(torch.empty(shape, dtype=dtype))
            continue
        if key in distributed_common_keys:
            deferred += 1
            deferred_bytes += size
            continue
        missing_keys.append(key)

    if missing_keys:
        raise RuntimeError(
            "FRCheck load: missing common tensor views for async reconstruct: "
            f"{missing_keys[:8]} (total {len(missing_keys)})"
        )

    decomposed = DecomposedStateDict(
        non_tensor_data=non_tensor_data,
        tensor_infos=common_infos,
        tensor_data=common_tensor_data,
        flat_key_roots=set(flat_key_roots) if flat_key_roots else set(),
    )
    result = reconstruct_state_dict(decomposed)
    if _frcheck_debug_enabled():
        logger.info(
            "FRCheck recovery: reconstructed common tensor views "
            "count=%d skipped_layer_tensors=%d bytes=%d "
            "deferred_distributed_common=%d deferred_bytes=%d",
            len(common_infos), skipped, skipped_bytes, deferred, deferred_bytes,
        )
    return result



# ---------------------------------------------------------------------------
# Load from local SOURCE FRBK blocks (metadata-only main)
# ---------------------------------------------------------------------------

def _compile_stripe_plans_for_load(
    poa_path: str, rank_in_group: int,
) -> List[StripePlan]:
    """Lightweight POA compile for load (no RDMA init)."""
    import glob
    import importlib.util
    import os

    from megatron.core.dist_checkpointing.strategies import frcheck_manager as _fm

    strategies_dir = os.path.dirname(os.path.abspath(_fm.__file__))
    so_files = glob.glob(os.path.join(strategies_dir, "frcheck_native*.so"))
    if not so_files:
        raise RuntimeError("FRCheck load: frcheck_native*.so not found")
    spec = importlib.util.spec_from_file_location("frcheck_native", so_files[0])
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    native = mod.FRCheckNative(poa_path)
    native.compile_plans(rank_in_group)
    stripe_plans: List[StripePlan] = []
    for sid in range(native.num_stripes()):
        stripe_plans.append(
            StripePlan(
                stripe_id=sid,
                row=list(native.row(sid)),
                role=StripeRole(native.get_role_for_stripe(sid)),
                source_node_ids=list(native.get_source_node_ids(sid)),
                encoder_node_id=native.get_encoder_node_id(sid),
                parity_target_node_id=native.get_parity_target_node_id(sid),
            )
        )
    return stripe_plans


def _enumerate_source_stripes(stripe_plans: List[StripePlan]) -> List[Tuple[int, int]]:
    """Return (stripe_id, blk_idx) for each SOURCE stripe (matches save order)."""
    result: List[Tuple[int, int]] = []
    blk_idx = 0
    for plan in stripe_plans:
        if plan.role == StripeRole.SOURCE:
            result.append((plan.stripe_id, blk_idx))
            blk_idx += 1
    return result


def _build_full_buf_from_local_blocks(
    checkpoint_dir: Path,
    rank: int,
    main_payload: Dict[str, Any],
    out_buf: Optional[torch.Tensor] = None,
    source_block_cache: Optional[Dict[Tuple[str, int, int], torch.Tensor]] = None,
    cache_source_keys: Optional[Set[Tuple[str, int, int]]] = None,
    timings: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Assemble rank-local tensor bytes into full_buf from SOURCE FRBK shards."""
    poa_path = main_payload.get("poa_path", "")
    if not poa_path:
        raise RuntimeError("FRCheck load: poa_path missing in main metadata")

    rank_in_group = int(main_payload.get("rank_in_group", 0))
    layer_names = main_payload.get("layer_names", [])
    if not layer_names:
        raise RuntimeError("FRCheck load: layer_names empty in main metadata")

    total_tensor_size = int(main_payload.get("actual_tensor_size", 0))
    global_tensor_infos = main_payload.get("tensor_infos", [])

    stripe_plans = _compile_stripe_plans_for_load(poa_path, rank_in_group)
    source_stripes = _enumerate_source_stripes(stripe_plans)
    num_source = len(source_stripes)

    if _frcheck_debug_enabled():
        logger.info(
            "FRCheck load: rank=%d assembling from %d SOURCE stripes, layers=%d",
            rank, num_source, len(layer_names),
        )

    if timings is not None:
        timings.setdefault("alloc", 0.0)
        timings.setdefault("layer_meta", 0.0)
        timings.setdefault("read_blocks", 0.0)
        timings.setdefault("copy_to_full", 0.0)
        timings.setdefault("layers", float(len(layer_names)))
        timings.setdefault("source_blocks", 0.0)
        timings.setdefault("bytes_read", 0.0)
        timings.setdefault("bytes_copied", 0.0)
        timings.setdefault("direct_reads", 0.0)
        timings.setdefault("staged_blocks", 0.0)

    safety_margin = max(int(total_tensor_size * 0.01), 4096)
    t_alloc = time.time()
    if out_buf is not None:
        if out_buf.numel() < total_tensor_size + safety_margin:
            raise RuntimeError(
                f"FRCheck load: out_buf too small ({out_buf.numel()} "
                f"< {total_tensor_size + safety_margin})"
            )
        full_buf = out_buf
    else:
        # Match Gemini SW load: a regular CPU tensor is enough here. Hardware
        # recovery passes an RDMA-registered out_buf from the manager.
        full_buf = torch.empty(
            total_tensor_size + safety_margin,
            dtype=torch.uint8,
            pin_memory=torch.cuda.is_available(),
        )
    if timings is not None:
        timings["alloc"] += time.time() - t_alloc

    cache = source_block_cache if source_block_cache is not None else {}
    cache_keys = cache_source_keys if cache_source_keys is not None else set()
    use_staged_blocks = bool(cache_keys)
    global_info_by_key = {
        getattr(info, "key", ""): info
        for info in global_tensor_infos
        if getattr(info, "key", "")
    }

    for layer_name in layer_names:
        layer_dir = checkpoint_dir / layer_name
        layer_main_path = layer_dir / f"frcheck_layer_main_rank{rank}.pt"
        if not layer_main_path.is_file():
            logger.warning(
                "FRCheck load: missing layer_main for %s, skipping", layer_name,
            )
            continue

        t_meta = time.time()
        layer_meta = torch.load(layer_main_path, map_location="cpu", weights_only=False)
        if timings is not None:
            timings["layer_meta"] += time.time() - t_meta
        layer_block_size = int(layer_meta.get("block_size", 0))
        if layer_block_size <= 0:
            raise RuntimeError(
                f"FRCheck load: invalid block_size for layer {layer_name}"
            )
        layer_infos = layer_meta.get("tensor_infos", [])
        actual_tensor_size = int(layer_meta.get("actual_tensor_size", 0))
        n_filled_blocks = _n_filled_blocks_for_layer(
            actual_tensor_size,
            layer_block_size,
            num_source,
        )

        if use_staged_blocks:
            layer_buf = torch.empty(num_source * layer_block_size, dtype=torch.uint8)
            for stripe_id, blk_idx in source_stripes:
                if blk_idx >= n_filled_blocks:
                    continue
                block_path = (
                    layer_dir / f"stripe_{stripe_id}" / f"frcheck_shard_rank{rank}.pt"
                )
                src_offset = blk_idx * layer_block_size
                dst_slice = layer_buf[src_offset:src_offset + layer_block_size]
                t_read = time.time()
                copy_len = _read_frbk_block_into(dst_slice, str(block_path))
                if timings is not None:
                    timings["read_blocks"] += time.time() - t_read
                if copy_len == 0:
                    logger.warning(
                        "FRCheck load: missing block %s, skipping stripe", block_path,
                    )
                    continue
                if timings is not None:
                    timings["source_blocks"] += 1.0
                    timings["staged_blocks"] += 1.0
                    timings["bytes_read"] += float(copy_len)
                cache_key = (layer_name, stripe_id, int(StripeRole.SOURCE))
                if cache_key in cache_keys:
                    cache[cache_key] = dst_slice[:layer_block_size].clone()

            t_copy = time.time()
            for info in layer_infos:
                key = getattr(info, "key", "")
                global_info = global_info_by_key.get(key)
                if global_info is None:
                    continue
                local_offset = int(getattr(info, "offset", 0))
                global_offset = int(getattr(global_info, "offset", 0))
                size = int(getattr(global_info, "size_bytes", 0))
                if size > 0 and global_offset + size <= full_buf.numel():
                    full_buf[global_offset:global_offset + size].copy_(
                        layer_buf[local_offset:local_offset + size]
                    )
                    if timings is not None:
                        timings["bytes_copied"] += float(size)
            if timings is not None:
                timings["copy_to_full"] += time.time() - t_copy
            continue

        # Normal SW load does not need whole layer staging. Read only tensor ranges
        # directly into their final global offsets in full_buf.
        for info in layer_infos:
            key = getattr(info, "key", "")
            global_info = global_info_by_key.get(key)
            if global_info is None:
                continue
            local_offset = int(getattr(info, "offset", 0))
            global_offset = int(getattr(global_info, "offset", 0))
            remaining = int(getattr(global_info, "size_bytes", 0))
            if remaining <= 0 or global_offset + remaining > full_buf.numel():
                continue
            while remaining > 0:
                blk_idx = local_offset // layer_block_size
                within_block = local_offset - blk_idx * layer_block_size
                if blk_idx >= n_filled_blocks:
                    break
                source = None
                for stripe_id, source_blk_idx in source_stripes:
                    if source_blk_idx == blk_idx:
                        source = (stripe_id, source_blk_idx)
                        break
                if source is None:
                    break
                stripe_id, _source_blk_idx = source
                chunk = min(remaining, layer_block_size - within_block)
                block_path = (
                    layer_dir / f"stripe_{stripe_id}" / f"frcheck_shard_rank{rank}.pt"
                )
                dst = full_buf[global_offset:global_offset + chunk]
                t_read = time.time()
                copy_len = _read_frbk_range_into(
                    dst, str(block_path), within_block, chunk,
                )
                if timings is not None:
                    timings["read_blocks"] += time.time() - t_read
                if copy_len <= 0:
                    logger.warning(
                        "FRCheck load: missing range %s offset=%d size=%d",
                        block_path, within_block, chunk,
                    )
                    break
                if timings is not None:
                    timings["source_blocks"] += 1.0
                    timings["direct_reads"] += 1.0
                    timings["bytes_read"] += float(copy_len)
                local_offset += copy_len
                global_offset += copy_len
                remaining -= copy_len

    return full_buf


def _assemble_state_dict_from_local_blocks(
    checkpoint_name: str,
    main_payload: Optional[Dict[str, Any]] = None,
    teardown_native: bool = True,
) -> Dict[str, Any]:
    """Assemble state_dict from metadata-only main + local SOURCE FRBK shards."""
    checkpoint_dir = _checkpoint_dir_from_path(checkpoint_name)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )

    if main_payload is None:
        main_payload = _load_frcheck_main_payload(
            checkpoint_dir, rank, world_size, load_tensor_buffer=False,
        )

    main_path = checkpoint_dir / f"frcheck_main_rank{rank}.pt"
    if main_path.is_file():
        _assert_frcheck_main_metadata_only(main_path)

    flat_key_roots = main_payload.get("flat_key_roots", [])
    total_tensor_size = int(main_payload.get("actual_tensor_size", 0))
    assemble_timings: Dict[str, float] = {}
    t_assemble = time.time()
    full_buf = _build_full_buf_from_local_blocks(
        checkpoint_dir, rank, main_payload, timings=assemble_timings,
    )
    assemble_s = time.time() - t_assemble

    payload = dict(main_payload)
    payload["tensor_buffer"] = full_buf[:total_tensor_size]
    t_rebuild = time.time()
    result = _reconstruct_from_main_payload(payload, flat_key_roots)
    rebuild_sd_s = time.time() - t_rebuild
    try:
        from megatron.training import get_args
        args = get_args()
        if getattr(args, "use_frcheck", False):
            from megatron.training.global_vars import set_ft_load_timing_context
            timings = {
                "network_encode": 0.0,
                "assemble": assemble_s,
                "assemble_alloc": assemble_timings.get("alloc", 0.0),
                "assemble_layer_meta": assemble_timings.get("layer_meta", 0.0),
                "assemble_read": assemble_timings.get("read_blocks", 0.0),
                "assemble_copy": assemble_timings.get("copy_to_full", 0.0),
                "assemble_source_blocks": assemble_timings.get("source_blocks", 0.0),
                "assemble_bytes_read": assemble_timings.get("bytes_read", 0.0),
                "assemble_bytes_copied": assemble_timings.get("bytes_copied", 0.0),
                "prep_copy": assemble_s,
                "rebuild_sd": rebuild_sd_s,
                "total": rebuild_sd_s,
            }
            set_ft_load_timing_context("FRCHECK", "SW", timings)
    except Exception:
        pass
    if teardown_native:
        try:
            from megatron.training import get_args
            safe_point = getattr(get_args(), "frcheck_recovery_safe_point", "load")
        except Exception:
            safe_point = "load"
        if safe_point == "load":
            _teardown_frcheck_native_after_load()
        else:
            _frcheck_recovery_profile(
                getattr(FRCheckManager(), "_frcheck_recovery_role", "local_load"),
                "teardown_moved_to_safe_point", safe_point=safe_point,
            )
    return result


def load_frcheck_legacy_checkpoint(checkpoint_name: str) -> Dict[str, Any]:
    """Load FRCheck legacy checkpoint.

    Default: all ranks assemble tensor data from local SOURCE FRBK blocks;
    main files provide metadata only.

    Hardware recovery (--use-frcheck-hardware-failure): failed ranks recover
    via RDMA RS decode; survivors assemble from local blocks like normal load.
    """
    from megatron.training import get_args
    args = get_args()

    hw_failure = getattr(args, "use_frcheck_hardware_failure", False)
    failed_ranks_str = getattr(args, "frcheck_failed_ranks", None)
    failed_ranks_parsed = getattr(args, "frcheck_failed_ranks_parsed", None)
    if _frcheck_debug_enabled():
        logger.info(
            "FRCheck load: hw_failure=%s failed_ranks_raw=%r failed_ranks_parsed=%r",
            hw_failure, failed_ranks_str, failed_ranks_parsed,
        )

    failed_ranks = failed_ranks_parsed
    if failed_ranks is None and failed_ranks_str:
        failed_ranks = [int(x.strip()) for x in failed_ranks_str.split(",")]

    if hw_failure and failed_ranks:
        if _frcheck_debug_enabled():
            logger.debug(
                "FRCheck: hardware recovery mode — failed ranks %s", failed_ranks
            )
        result, _t_fr = recover_frcheck_legacy_hardware(checkpoint_name, failed_ranks)
        _t_fr['total'] = _t_fr['network_encode'] + _t_fr['rebuild_sd']
        try:
            from megatron.training.global_vars import set_ft_load_timing_context
            set_ft_load_timing_context("FRCHECK", "HW", _t_fr)
        except Exception:
            pass
        if _frcheck_debug_enabled():
            logger.debug(
                "FRCheck legacy load timing (HW): "
                "total=%(total).2fs network_encode=%(network_encode).2fs "
                "rebuild_sd=%(rebuild_sd).2fs",
                _t_fr,
            )
        try:
            from megatron.training.global_vars import mark_recovery_to_forward_timer
            mark_recovery_to_forward_timer("frcheck_load_return")
        except Exception:
            pass
        return result

    return _assemble_state_dict_from_local_blocks(checkpoint_name)
