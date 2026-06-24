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
import threading
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from dataclasses import dataclass

from megatron.core.dist_checkpointing.strategies.frcheck_manager import (
    FRCheckManager,
    LayerStripeBufs,
    StripePlan,
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

_async_p1_writer_thread: Optional[threading.Thread] = None
_async_p2_writer_thread: Optional[threading.Thread] = None

_LAYER_KEY_RE = re.compile(r"\.layers\.(\d+)\b")

# Cached metadata exchange (identical across iterations for fixed model)
_cached_all_tensor_infos = None
_cached_all_layer_order = None
_cached_all_layer_metadata = None
_cached_all_actual_tensor_sizes = None


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


def _build_optimizer_layer_map(state_dict: Dict[str, Any]) -> Dict[str, int]:
    """Map optimizer tensor keys to transformer layer indexes when possible."""
    model_sd = state_dict.get("model")
    optim_sd = state_dict.get("optimizer")
    if not isinstance(model_sd, dict) or not isinstance(optim_sd, dict):
        return {}

    model_param_entries: List[Tuple[str, torch.Tensor, int]] = []
    for key, tensor in model_sd.items():
        if not torch.is_tensor(tensor):
            continue
        model_param_entries.append((str(key), tensor, _extract_layer_idx(str(key))))
    if not any(layer_idx >= 0 for _, _, layer_idx in model_param_entries):
        return {}

    result: Dict[str, int] = {}
    torch_optim_sd = optim_sd.get("optimizer", optim_sd)
    if not isinstance(torch_optim_sd, dict):
        return {}
    param_groups = torch_optim_sd.get("param_groups", [])

    # Reconstruct the common Megatron optimizer grouping order from model state
    # dict keys. This mirrors _get_param_groups() for the GPT/Adam path:
    # regularized params first, then no-weight-decay params. Common model
    # params stay optimizer_common because they are not consumed by layer hooks.
    grouped_model_entries: Dict[Tuple[float, float, bool, bool], List[Tuple[str, int]]] = {}
    group_order: List[Tuple[float, float, bool, bool]] = []
    for key, tensor, layer_idx in model_param_entries:
        no_wd = key.endswith(".bias") or len(tensor.shape) == 1
        group_key = (0.0 if no_wd else 1.0, 1.0, False, False)
        if group_key not in grouped_model_entries:
            grouped_model_entries[group_key] = []
            group_order.append(group_key)
        grouped_model_entries[group_key].append((key, layer_idx))

    ordered_model_groups = [grouped_model_entries[key] for key in group_order]
    for group_idx, group in enumerate(param_groups):
        params = list(group.get("params", [])) if isinstance(group, dict) else []
        model_group = ordered_model_groups[group_idx] if group_idx < len(ordered_model_groups) else []
        for param_idx, param_ref in enumerate(params):
            layer_idx = -1
            if param_idx < len(model_group):
                _model_key, layer_idx = model_group[param_idx]
            if layer_idx >= 0:
                result[f"optimizer.fp32_params_flat._fp32_group{group_idx}_param{param_idx}"] = layer_idx
                for state_name in ("exp_avg", "exp_avg_sq", "fp32_param"):
                    if "optimizer" in optim_sd:
                        result[f"optimizer.optimizer.state.{param_ref}.{state_name}"] = layer_idx
                    else:
                        result[f"optimizer.state.{param_ref}.{state_name}"] = layer_idx
    return result


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
    error: str = ""
    tensors: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class _FRCheckLayerRecoveryJob:
    encode_iter: int
    layer_name: str
    layer_idx: int
    layer_block_size: int
    actual_size: int
    layer_infos: List


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
        self._events_by_layer: Dict[int, threading.Event] = {}
        self._injected_layers: Set[int] = set()
        for layer_idx, record in self._records_by_layer.items():
            event = threading.Event()
            if record.ready:
                event.set()
            self._events_by_layer[layer_idx] = event
        self.wait_s: float = 0.0
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
        if record is None or not record.tensors:
            return
        t0 = time.time()
        copied = 0
        matched = 0
        missing = 0
        with torch.no_grad():
            for key in record.model_tensor_keys:
                tensor = record.tensors.get(key)
                if tensor is None:
                    continue
                dst = self._lookup_live_tensor(key)
                if dst is None:
                    missing += 1
                    continue
                dst.copy_(tensor.to(device=dst.device, dtype=dst.dtype), non_blocking=True)
                copied += tensor.numel() * tensor.element_size()
                matched += 1
        self._injected_layers.add(layer_idx)
        inject_s = time.time() - t0
        self.inject_s += inject_s
        logger.info(
            "FRCheck layerwise inject: layer=%s idx=%d tensors=%d bytes=%d "
            "matched=%d missing=%d time=%.4fs total_inject=%.4fs",
            record.layer_name, layer_idx, len(record.tensors), copied,
            matched, missing, inject_s, self.inject_s,
        )
        if record.contains_optimizer_state:
            optimizer_key_set = set(record.optimizer_tensor_keys)
            record.tensors = {
                key: tensor
                for key, tensor in record.tensors.items()
                if key in optimizer_key_set
            }
        else:
            record.tensors = None

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
        record.materialize_s = materialize_s
        record.nbytes = nbytes
        if tensors is not None:
            record.tensors = tensors
        record.error = error
        record.ready = not error
        if self.first_layer_ready_s is None:
            self.first_layer_ready_s = time.time()
        self.last_layer_ready_s = time.time()
        event = self._events_by_layer.setdefault(layer_idx, threading.Event())
        event.set()

    def mark_layer_error(self, layer_idx: int, error: str) -> None:
        self.mark_layer_ready(layer_idx, error=error)

    def wait_and_materialize_layer(self, layer_idx: int) -> bool:
        t0 = time.time()
        try:
            record = self._records_by_layer.get(layer_idx)
            if record is None:
                self.missing_layers.add(layer_idx)
                return False
            event = self._events_by_layer.setdefault(layer_idx, threading.Event())
            event.wait()
            if not record.ready:
                raise RuntimeError(
                    f"FRCheck layer {record.layer_name} is not ready: {record.error}"
                )
            self._inject_layer_tensors(layer_idx)
            first_touch = layer_idx not in self.materialized_layers
            self.materialized_layers.add(layer_idx)
            waited = time.time() - t0
            if self.first_wait_s is None:
                self.first_wait_s = waited
            if first_touch:
                logger.info(
                    "FRCheck layerwise forward: layer=%s idx=%d ready "
                    "recovery_materialize=%.4fs wait=%.4fs",
                    record.layer_name, layer_idx, record.materialize_s, waited,
                )
            return True
        finally:
            self.wait_s += time.time() - t0

    def wait_for_optimizer_layers(self) -> None:
        for layer_idx, record in sorted(self._records_by_layer.items()):
            if not record.contains_optimizer_state:
                continue
            event = self._events_by_layer.setdefault(layer_idx, threading.Event())
            event.wait()
            if not record.ready:
                raise RuntimeError(
                    f"FRCheck optimizer layer {record.layer_name} is not ready: {record.error}"
                )

    def summary(self) -> Dict[str, Any]:
        pending_runtime_tensors = sum(
            len(record.tensors or {}) for record in self._records_by_layer.values()
        )
        return {
            "ready_layers": len(self._records_by_layer),
            "materialized_layers": len(self.materialized_layers),
            "missing_layers": sorted(self.missing_layers),
            "wait_s": self.wait_s,
            "first_wait_s": self.first_wait_s,
            "first_layer_ready_s": self.first_layer_ready_s,
            "last_layer_ready_s": self.last_layer_ready_s,
            "inject_s": self.inject_s,
            "injected_layers": len(self._injected_layers),
            "optimizer_materialized": self.optimizer_materialized,
            "pending_runtime_tensors": pending_runtime_tensors,
        }


_active_layerwise_runtime: Optional[_FRCheckLayerwiseRuntime] = None
_active_recovery_worker: Optional[threading.Thread] = None
_pending_optimizer_state: Optional[Dict[str, Any]] = None
_pending_optimizer_container: Optional[Dict[str, Any]] = None


def _set_active_frcheck_layerwise_runtime(
    runtime: Optional[_FRCheckLayerwiseRuntime],
) -> None:
    global _active_layerwise_runtime
    _active_layerwise_runtime = runtime


def _set_active_frcheck_recovery_worker(worker: Optional[threading.Thread]) -> None:
    global _active_recovery_worker
    _active_recovery_worker = worker


def frcheck_wait_for_async_recovery() -> None:
    global _active_recovery_worker
    worker = _active_recovery_worker
    if worker is not None and worker.is_alive():
        logger.info("FRCheck: waiting for async recovery worker to finish")
        worker.join()
    if worker is not None:
        error_holder = getattr(worker, "_frcheck_error_holder", None)
        if isinstance(error_holder, dict) and error_holder.get("error") is not None:
            error = error_holder["error"]
            _active_recovery_worker = None
            raise RuntimeError("FRCheck async recovery worker failed") from error
    _active_recovery_worker = None


def _layerwise_record_from_layer_buf(
    layer_name: str,
    encode_iter: int,
    layer_infos: List,
    nbytes: int,
    materialize_s: float,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
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
    model_tensor_keys = [
        key for key in tensor_keys
        if _classify_frcheck_tensor(key).kind == "model_layer"
    ]
    optimizer_tensor_keys = [
        key for key in tensor_keys
        if _classify_frcheck_tensor(key).kind == "optimizer_layer"
    ]
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
) -> List[_FRCheckLayerReadyRecord]:
    records: List[_FRCheckLayerReadyRecord] = []
    for encode_iter in range(n_encode_iters):
        job = _make_layer_recovery_job(
            encode_iter, rank, is_failed, main_payload, all_layer_order,
            all_layer_metadata, all_frcheck_dirs, checkpoint_dir, saved_block_size,
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
                if _classify_frcheck_tensor(key).kind == "model_layer"
            ]
        if not optimizer_tensor_keys:
            optimizer_tensor_keys = [
                key for key in tensor_keys
                if _classify_frcheck_tensor(key).kind == "optimizer_layer"
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
            )
        )
    return records


def _extract_layer_tensors_from_buf(
    layer_buf: torch.Tensor,
    layer_infos: List,
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
        raw = buf[offset:offset + size].clone()
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
        _active_layerwise_runtime = None
        return

    if _active_layerwise_runtime is not None and metadata.get("active_runtime", False):
        runtime = _active_layerwise_runtime
        model_sd = state_dict.get("model")
        if isinstance(model_sd, dict):
            runtime.attach_model_state_keys(model_sd.keys())
        if model is not None:
            runtime.attach_live_model(model)
        logger.info(
            "FRCheck layerwise runtime attached to active recovery worker: "
            "ready_layers=%d live_tensors=%d",
            len(runtime._records_by_layer), len(runtime._live_tensors),
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
            error=str(item.get("error", "")),
            tensors=item.get("tensors"),
        )
        for item in metadata.get("records", [])
    ]
    runtime = _FRCheckLayerwiseRuntime(records)
    model_sd = state_dict.get("model")
    if isinstance(model_sd, dict):
        runtime.attach_model_state_keys(model_sd.keys())
    if model is not None:
        runtime.attach_live_model(model)
    _active_layerwise_runtime = runtime
    logger.info(
        "FRCheck layerwise runtime installed: ready_layers=%d live_tensors=%d "
        "runtime_tensors=%d",
        len(runtime._records_by_layer),
        len(runtime._live_tensors),
        sum(len(r.tensors or {}) for r in runtime._records_by_layer.values()),
    )


def frcheck_wait_and_materialize_layer(layer_idx: int) -> bool:
    """Wait for a recovered FRCheck layer to be ready before forward."""
    runtime = _active_layerwise_runtime
    if runtime is None:
        return False
    return runtime.wait_and_materialize_layer(layer_idx)


def frcheck_materialize_all_layers() -> None:
    runtime = _active_layerwise_runtime
    if runtime is None:
        return
    for layer_idx in sorted(runtime._records_by_layer):
        runtime.wait_and_materialize_layer(layer_idx)
    frcheck_wait_for_async_recovery()


def frcheck_register_pending_optimizer_state(state_dict: Dict[str, Any]) -> bool:
    """Keep optimizer state_dict for loading at the first optimizer consumption point."""
    global _pending_optimizer_container, _pending_optimizer_state
    runtime = _active_layerwise_runtime
    if runtime is None:
        return False
    if not any(r.contains_optimizer_state for r in runtime._records_by_layer.values()):
        return False
    optim_state = state_dict.pop("optimizer", None)
    if optim_state is None:
        return False
    _pending_optimizer_container = {"optimizer": optim_state}
    _pending_optimizer_state = optim_state
    logger.info(
        "FRCheck optimizer recovery: deferred optimizer load for %d layers",
        sum(1 for r in runtime._records_by_layer.values() if r.contains_optimizer_state),
    )
    return True


def _assign_nested_state_tensor(root: Dict[str, Any], flat_key: str, tensor: torch.Tensor) -> bool:
    first_dot = flat_key.find(".")
    if first_dot >= 0 and flat_key[:first_dot] == "optimizer":
        flat_key = flat_key[first_dot + 1:]
    parts = flat_key.split(".")
    current = root
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            try:
                int_part = int(part)
            except (TypeError, ValueError):
                return False
            if int_part not in current:
                return False
            part = int_part
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        return False
    current[parts[-1]] = tensor
    return True


def _materialize_pending_optimizer_tensors(
    runtime: _FRCheckLayerwiseRuntime,
    optim_state: Dict[str, Any],
) -> int:
    updated = 0
    for record in runtime._records_by_layer.values():
        if not record.contains_optimizer_state or not record.tensors:
            continue
        for key in record.optimizer_tensor_keys:
            tensor = record.tensors.get(key)
            if tensor is None:
                continue
            if _assign_nested_state_tensor(optim_state, key, tensor):
                updated += 1
    return updated


def frcheck_wait_for_optimizer_state(optimizer=None) -> bool:
    """Wait for layerwise optimizer tensors and load deferred optimizer state once."""
    global _pending_optimizer_container, _pending_optimizer_state
    runtime = _active_layerwise_runtime
    if runtime is None or _pending_optimizer_state is None:
        return False
    t0 = time.time()
    runtime.wait_for_optimizer_layers()
    frcheck_wait_for_async_recovery()
    if optimizer is None:
        return False
    updated = _materialize_pending_optimizer_tensors(runtime, _pending_optimizer_state)
    from megatron.core.dist_checkpointing.strategies.state_dict_decomposer import (
        unflatten_optimizer_fp32_params,
    )
    unflatten_optimizer_fp32_params(_pending_optimizer_container)
    optimizer.load_state_dict(_pending_optimizer_state)
    for record in runtime._records_by_layer.values():
        if record.contains_optimizer_state:
            record.tensors = None
    _pending_optimizer_container = None
    _pending_optimizer_state = None
    runtime.optimizer_materialized = True
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info(
        "FRCheck optimizer recovery: loaded deferred optimizer state in %.4fs "
        "(updated_tensors=%d)",
        time.time() - t0, updated,
    )
    return True


def get_frcheck_layerwise_runtime_summary() -> Optional[Dict[str, Any]]:
    runtime = _active_layerwise_runtime
    return None if runtime is None else runtime.summary()


def frcheck_log_layerwise_runtime_summary(context: str) -> None:
    summary = get_frcheck_layerwise_runtime_summary()
    if summary is not None:
        logger.info("FRCheck layerwise runtime summary (%s): %s", context, summary)


def _group_by_layer(
    decomposed,
    distribute_common: bool = False,
    debug: bool = False,
    optimizer_layer_map: Optional[Dict[str, int]] = None,
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
        for info, tensor in zip(common.tensor_infos, common.tensor_data):
            ownership = _classify_frcheck_tensor(info.key, optimizer_layer_map)
            if ownership.kind == "model_common":
                moved_infos.append(info)
                moved_data.append(tensor)
            else:
                kept_infos.append(info)
                kept_data.append(tensor)
        if others and moved_infos:
            for i, (info, tensor) in enumerate(zip(moved_infos, moved_data)):
                target = others[i % len(others)]
                target.tensor_infos.append(info)
                target.tensor_data.append(tensor)
                target.total_bytes += info.size_bytes
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
                "tensors (%d bytes), grouped optimizer_layer=%d, kept common=%d "
                "tensors (%d bytes)",
                len(moved_infos), moved_bytes, mapped_optimizer,
                len(kept_infos), kept_bytes,
            )

    # Sort: non-layer (-1) first, then by layer index
    result = sorted(groups.values(), key=lambda g: (0 if g.layer_idx < 0 else 1, g.layer_idx))
    ownership_counts: Dict[str, int] = {}
    ownership_bytes: Dict[str, int] = {}
    for group in result:
        for info in group.tensor_infos:
            ownership = _classify_frcheck_tensor(info.key, optimizer_layer_map)
            ownership_counts[ownership.kind] = ownership_counts.get(ownership.kind, 0) + 1
            ownership_bytes[ownership.kind] = (
                ownership_bytes.get(ownership.kind, 0)
                + int(getattr(info, "size_bytes", 0))
            )
    layer_sizes = [
        (
            "layer_common" if group.layer_idx < 0 else f"layer_{group.layer_idx}",
            len(group.tensor_infos),
            group.total_bytes,
        )
        for group in result
    ]
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
    if added_empty:
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



def _register_layer_stripe_bufs(manager, native, layer_bufs: LayerStripeBufs) -> None:
    """Register per-stripe recv/parity buffers for RDMA."""
    for buf_list in (
        layer_bufs.recv_bufs,
        layer_bufs.parity1_bufs,
        layer_bufs.parity2_bufs,
    ):
        for buf in buf_list:
            if buf is None:
                continue
            addr = buf.data_ptr()
            if addr in manager._rdma_registered_addrs:
                continue
            native.register_buffer(addr, buf.numel())
            manager._rdma_registered_addrs.add(addr)


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


def _async_write_frcheck_p2_files(
    manager,
    output_dir: str,
    rank: int,
    num_stripes: int,
    encode_results: List[_LayerEncodeResult],
    debug: bool = False,
) -> None:
    """Wait for async P2 delivery, then write P2 shards."""
    native = manager.get_native()
    if native is None:
        return
    try:
        t0 = time.time()
        logger.info(
            "FRCHECK async P2 writer rank %d: wait_parity_flush begin (layers=%d)",
            rank, len(encode_results),
        )
        native.wait_parity_flush()
        flush_elapsed = time.time() - t0
        logger.info(
            "FRCHECK async P2 writer rank %d: wait_parity_flush done %.3fs",
            rank, flush_elapsed,
        )
        write_t0 = time.time()
        _save_frcheck_stripe_files(
            manager,
            output_dir,
            rank,
            num_stripes,
            encode_results,
            include_encoder_p1=False,
            include_source=False,
            include_p2=True,
        )
        logger.info(
            "FRCHECK async P2 writer rank %d: P2 shard write done %.3fs",
            rank, time.time() - write_t0,
        )
    except Exception:
        logger.exception("FRCheck async P2 writer failed")
        raise


def _wait_previous_async_writers(debug: bool = False, rank: int = -1) -> None:
    global _async_p1_writer_thread, _async_p2_writer_thread
    for attr in ("_async_p1_writer_thread", "_async_p2_writer_thread"):
        th = globals()[attr]
        if th is not None:
            t0 = time.time()
            logger.info(
                "FRCHECK async writer rank %d: joining previous %s",
                rank, th.name,
            )
            th.join()
            logger.info(
                "FRCHECK async writer rank %d: joined previous %s in %.3fs",
                rank, th.name, time.time() - t0,
            )
            globals()[attr] = None


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


def _start_async_p2_writer(
    manager,
    output_dir: str,
    rank: int,
    num_stripes: int,
    encode_results: List[_LayerEncodeResult],
    debug: bool = False,
) -> None:
    global _async_p2_writer_thread
    if _async_p2_writer_thread is not None:
        t0 = time.time()
        logger.info(
            "FRCHECK async P2 writer rank %d: joining in-flight writer before restart",
            rank,
        )
        _async_p2_writer_thread.join()
        logger.info(
            "FRCHECK async P2 writer rank %d: in-flight writer joined in %.3fs",
            rank, time.time() - t0,
        )
        _async_p2_writer_thread = None
    _async_p2_writer_thread = threading.Thread(
        target=_async_write_frcheck_p2_files,
        args=(manager, output_dir, rank, num_stripes, list(encode_results), debug),
        name="frcheck-async-p2-writer",
        daemon=False,
    )
    _async_p2_writer_thread.start()
    logger.info(
        "FRCHECK async P2 writer rank %d: started thread for %d layers",
        rank, len(encode_results),
    )


def save_frcheck_legacy_checkpoint(state_dict: Dict[str, Any], checkpoint_name: str) -> None:
    """Write frcheck_main_rank*.pt + layer-wise source/parity shards."""
    t0 = time.time()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    from megatron.training import get_args
    args = get_args()
    _dbg = getattr(args, "frcheck_debug", False)

    optimizer_layer_map = _build_optimizer_layer_map(state_dict)

    # 1. Decompose state_dict
    flatten_optimizer_fp32_params(state_dict)
    decomposed = decompose_state_dict(state_dict)
    total_tensor_size = decomposed.total_tensor_size_bytes
    if _dbg:
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
    native.set_debug(_dbg)

    # Async parity path: drain any pending P2 operations from a previous save.
    _use_async_parity = getattr(args, 'frcheck_async_parity', False)
    if _use_async_parity:
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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 4. Group tensors by layer index
    t0 = time.time()
    layer_groups = _group_by_layer(
        decomposed,
        distribute_common=getattr(args, "frcheck_distribute_common", False),
        debug=_dbg,
        optimizer_layer_map=optimizer_layer_map,
    )
    layer_groups = _synchronize_layer_groups(layer_groups)
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

    # 5. Phase A: prep all layers (copy + register), then Phase B: batch network encode
    t_prep = time.time()
    local_layer_order: List[str] = []
    local_layer_metadata: Dict[str, Dict[str, Any]] = {}
    encode_results: List[_LayerEncodeResult] = []
    prep_stream = torch.cuda.Stream()

    for group in layer_groups:
        layer_name = f"layer_{group.layer_idx}" if group.layer_idx >= 0 else "layer_common"
        layer_idx = group.layer_idx
        layer_block_size = manager._layer_block_sizes[layer_idx]
        layer_bufs = manager.get_layer_stripe_bufs(layer_idx)
        _register_layer_stripe_bufs(manager, native, layer_bufs)

        layer_buf_gpu = layer_bufs.layer_buf_gpu
        layer_buf_gpu.zero_()

        offset = 0
        with torch.cuda.stream(prep_stream):
            for info, tensor in zip(group.tensor_infos, group.tensor_data):
                tensor_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
                nbytes = tensor_view.numel()
                layer_buf_gpu[offset : offset + nbytes].copy_(
                    tensor_view, non_blocking=True
                )
                info.offset = offset
                offset += nbytes
        prep_stream.synchronize()

        group.tensor_data = []

        state = _prep_layer_phase1(
            manager, layer_buf_gpu, layer_bufs.layer_mirror_cpu,
            group.total_bytes, n, num_stripes, layer_block_size, my_node,
            layer_name, layer_idx, _dbg,
        )

        # Pre-compute encoder active masks for this layer
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

        # Submit per-stripe tasks (non-blocking)
        native.reset_layer()
        for sid in range(num_stripes):
            plan = stripe_plans[sid]
            if plan.role == StripeRole.SOURCE:
                addr = state.data_addrs[sid]
                if addr == 0:
                    continue
                native.submit_source(sid, addr, state.mirror_addrs[sid], layer_block_size)
            elif plan.role == StripeRole.ENCODER:
                rb = layer_bufs.recv_bufs[sid]
                p1b = layer_bufs.parity1_bufs[sid]
                p2b = layer_bufs.parity2_bufs[sid]
                if rb is None or p1b is None or p2b is None:
                    continue
                native.submit_enc_recv(sid, rb.data_ptr(), p1b.data_ptr(),
                                      p2b.data_ptr(), layer_block_size,
                                      enc_active_masks.get(sid, []))
            elif plan.role == StripeRole.PARITY_TARGET:
                # Deferred to async phase (after all layers' encoding)
                pass

        # Wait for all stripes in this layer to complete
        if _use_async_parity:
            if _dbg:
                logger.info("FRCHECK layer %s: wait_encode_only (async)", layer_name)
            native.wait_encode_only()
        else:
            native.wait_layer()

        encode_results.append(_LayerEncodeResult(
            layer_name=layer_name,
            layer_idx=layer_idx,
            block_size=layer_block_size,
            actual_sizes=state.actual_sizes,
            tensor_infos=group.tensor_infos,
            total_bytes=group.total_bytes,
            n_filled_blocks=state.n_filled_blocks,
        ))

    t_enc = time.time() - t_prep

    # ---- async parity phase: submit P2 sends + parity receives ----
    # P2 delivery is one independent async batch. Do not call reset_layer()
    # per layer here: sync encode uses per-layer counters, but async P2 is
    # allowed to span layers and continue into the next training iteration.
    _async_p2_submit_elapsed = 0.0
    if _use_async_parity:
        _async_p2_submit_t0 = time.time()
        _async_p2_reset_elapsed = 0.0
        _async_p2_build_elapsed = 0.0
        _async_p2_native_elapsed = 0.0
        _async_p2_parity_tasks = 0
        _async_p2_send_tasks = 0
        _reset_t0 = time.time()
        native.reset_async_parity()
        _async_p2_reset_elapsed = time.time() - _reset_t0
        for result in encode_results:
            layer_bufs = manager.get_layer_stripe_bufs(result.layer_idx)
            _build_t0 = time.time()
            p2_addrs = [
                0 if p2b is None else p2b.data_ptr()
                for p2b in layer_bufs.parity2_bufs
            ]
            _async_p2_build_elapsed += time.time() - _build_t0
            _native_t0 = time.time()
            counts = native.submit_async_p2_layer(p2_addrs, result.block_size)
            _async_p2_native_elapsed += time.time() - _native_t0
            if counts is not None and len(counts) >= 2:
                _async_p2_parity_tasks += int(counts[0])
                _async_p2_send_tasks += int(counts[1])
        _async_p2_submit_elapsed = time.time() - _async_p2_submit_t0
        logger.info(
            "FRCHECK async P2 submit rank %d: layers=%d parity_tasks=%d send_tasks=%d "
            "reset=%.3fs build=%.3fs native=%.3fs total=%.3fs",
            rank, len(encode_results), _async_p2_parity_tasks, _async_p2_send_tasks,
            _async_p2_reset_elapsed, _async_p2_build_elapsed,
            _async_p2_native_elapsed, _async_p2_submit_elapsed,
        )

    _mirror_t0 = time.time()
    native.wait_mirror_completion()
    _mirror_elapsed = time.time() - _mirror_t0

    _post_barrier_elapsed = 0.0
    if world_size > 1:
        _post_barrier_t0 = time.time()
        torch.distributed.barrier()
        _post_barrier_elapsed = time.time() - _post_barrier_t0

    if _use_async_parity:
        logger.info(
            "FRCheck legacy save rank %d: async path submitted "
            "(encode=%.3fs async_p2_submit=%.3fs mirror=%.3fs "
            "post_barrier=%.3fs total=%.3fs)",
            rank, t_enc, _async_p2_submit_elapsed, _mirror_elapsed,
            _post_barrier_elapsed, time.time() - t_prep,
        )
    else:
        logger.info(
            "FRCheck legacy save rank %d: sync path done "
            "(encode=%.3fs mirror=%.3fs post_barrier=%.3fs total=%.3fs)",
            rank, t_enc, _mirror_elapsed,
            _post_barrier_elapsed, time.time() - t_prep,
        )

    if _use_async_parity:
        _save_frcheck_stripe_files(
            manager, str(checkpoint_dir), rank, num_stripes, encode_results,
            include_encoder_p1=True,
            include_source=True,
            include_p2=False,
        )
        _start_async_p2_writer(
            manager, str(checkpoint_dir), rank, num_stripes, encode_results,
            debug=_dbg,
        )
    else:
        _save_frcheck_stripe_files(
            manager, str(checkpoint_dir), rank, num_stripes, encode_results,
        )

    for result in encode_results:
        model_tensor_keys = []
        optimizer_tensor_keys = []
        for info in result.tensor_infos:
            key = getattr(info, "key", "")
            ownership = _classify_frcheck_tensor(key, optimizer_layer_map)
            if ownership.kind == "model_layer":
                model_tensor_keys.append(key)
            elif ownership.kind == "optimizer_layer":
                optimizer_tensor_keys.append(key)
        local_layer_order.append(result.layer_name)
        local_layer_metadata[result.layer_name] = {
            "block_size": result.block_size,
            "actual_tensor_size": result.total_bytes,
            "tensor_infos": copy.deepcopy(result.tensor_infos),
            "model_tensor_keys": model_tensor_keys,
            "optimizer_tensor_keys": optimizer_tensor_keys,
            "contains_optimizer_state": bool(optimizer_tensor_keys),
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
    if _cached_all_tensor_infos is None:
        all_tensor_infos, all_layer_order, all_layer_metadata = _exchange_frcheck_group_metadata(
            decomposed.tensor_infos, local_layer_order, local_layer_metadata,
        )
        all_actual_tensor_sizes = {
            r: sum(getattr(info, "size_bytes", 0) for info in infos)
            for r, infos in all_tensor_infos.items()
        }
        _cached_all_tensor_infos = all_tensor_infos
        _cached_all_layer_order = all_layer_order
        _cached_all_layer_metadata = all_layer_metadata
        _cached_all_actual_tensor_sizes = all_actual_tensor_sizes
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
        "optimizer_layer_map": optimizer_layer_map,
    })
    write_main_prepared(
        str(main_file), MAGIC_FRCHECK, meta1, meta2, extra, memoryview(b""), 0,
    )

    if _dbg:
        logger.info(
            "FRCheck save: done rank=%d node=%d gdr=True layers=%d file=%s (metadata-only)",
            rank, my_node, num_layers, main_file,
        )

    del _global_offsets

    if world_size > 1:
        torch.distributed.barrier()

    if _dbg:
        logger.info(
            "FRCheck save: native module kept alive for reuse (no shutdown, rank=%d)",
            rank,
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
    dual_failure: bool = False,
) -> Optional[_RecoveryBufPool]:
    """Pre-allocate stripe recovery buffers once before the per-layer network loop."""
    num_helper = max(n - 3, 0)
    num_source_stripes = (n - 1) * (n - 2)
    need_pool = is_failed or is_decoder or is_helper
    if not need_pool:
        return None

    # Native recovery starts n workers per role.  Allocate one temporary slot per
    # worker and submit recovery stripes in waves of at most n to avoid reusing a
    # slot while another worker is still using it.
    concurrency = max(n, 1)
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
        native.register_buffer(buf.data_ptr(), buf.numel())

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


def _is_failed_in_recovery_plan(plan: Dict, my_node: int) -> bool:
    if plan.get('dual_failure'):
        return my_node in plan['failed_nodes']
    return my_node == plan.get('failed_node')


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
) -> Optional[_FRCheckLayerRecoveryJob]:
    my_order = all_layer_order.get(rank, main_payload.get("layer_names", []))
    if encode_iter >= len(my_order):
        if is_failed:
            return None
        return _FRCheckLayerRecoveryJob(
            encode_iter=encode_iter,
            layer_name=f"encode_iter_{encode_iter}",
            layer_idx=-1,
            layer_block_size=saved_block_size,
            actual_size=0,
            layer_infos=[],
        )

    layer_name = my_order[encode_iter]
    layer_block_size = _resolve_layer_block_size(
        rank, layer_name, all_layer_metadata, saved_block_size,
    )
    if not is_failed:
        return _FRCheckLayerRecoveryJob(
            encode_iter=encode_iter,
            layer_name=layer_name,
            layer_idx=-1,
            layer_block_size=layer_block_size,
            actual_size=0,
            layer_infos=[],
        )

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
) -> Tuple[Optional[_FRCheckLayerReadyRecord], Dict[str, float]]:
    layer_buf, layer_timing = _recover_one_layer_network(
        manager, native, job.layer_name,
        job.layer_block_size, job.actual_size if is_failed else 0,
        n, rank, preloaded_blocks=preloaded.get(job.encode_iter, {}),
        buf_pool=buf_pool,
    )

    record = None
    if is_failed and layer_buf is not None and full_buf is not None:
        t_materialize = time.time()
        copied = _map_layer_buf_to_full_buf(
            layer_buf, job.layer_infos, global_tensor_infos, full_buf,
        )
        materialize_s = time.time() - t_materialize
        layer_tensors = (
            _extract_layer_tensors_from_buf(layer_buf, job.layer_infos)
            if job.layer_idx >= 0 else None
        )
        record = _layerwise_record_from_layer_buf(
            job.layer_name, job.encode_iter, job.layer_infos, copied,
            materialize_s, tensors=layer_tensors,
        )
        if runtime is not None and job.layer_idx >= 0:
            runtime.mark_layer_ready(
                job.layer_idx, materialize_s=materialize_s,
                nbytes=copied, tensors=layer_tensors,
            )
        logger.info(
            "FRCheck recovery: %s materialized %d bytes into full_buf "
            "(expected %d) in %.4fs",
            job.layer_name, copied, job.actual_size, materialize_s,
        )
    return record, layer_timing


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
) -> threading.Thread:
    error_holder: Dict[str, Optional[BaseException]] = {"error": None}

    def _worker() -> None:
        try:
            for job in jobs:
                try:
                    _run_layer_recovery_job(
                        job, manager, native, n, rank, is_failed, preloaded,
                        buf_pool, full_buf, global_tensor_infos, runtime=runtime,
                    )
                except Exception as exc:
                    if runtime is not None and job.layer_idx >= 0:
                        runtime.mark_layer_error(job.layer_idx, f"{type(exc).__name__}: {exc}")
                    raise
        except BaseException as exc:
            error_holder["error"] = exc
        finally:
            if cleanup_after:
                _teardown_frcheck_native_after_load()

    worker = threading.Thread(
        target=_worker,
        name=f"frcheck-layer-recovery-rank{rank}",
        daemon=False,
    )
    setattr(worker, "_frcheck_error_holder", error_holder)
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
        role = int(plan['original_role'])
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
                if int(plan['original_role']) == int(StripeRole.SOURCE):
                    cache_source_keys.add(
                        (layer_name, plan['stripe_id'], int(StripeRole.SOURCE))
                    )

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


def _submit_recovery_network(
    native,
    manager,
    layer_name: str,
    layer_block_size: int,
    layer_total_bytes: int,
    n: int,
    rank: int,
    preloaded_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Submit one layer of stripe recovery via C++ batch pipeline."""
    my_node = manager.rank_in_group + 1
    layer_timing: Dict[str, float] = {'rdma_xfer_s': 0.0, 'decode_s': 0.0}

    if not manager.recovery_stripe_plans:
        return None, layer_timing

    data_plans = [
        p for p in manager.recovery_stripe_plans
        if _is_data_recovery_plan(p, n)
    ]
    parity_plans = [
        p for p in manager.recovery_stripe_plans
        if not _is_data_recovery_plan(p, n)
    ]

    if not data_plans:
        logger.info(
            "FRCheck recovery layer %s: rank %d has no data recovery stripes "
            "(skipped parity stripes=%d)",
            layer_name, rank, len(parity_plans),
        )
        return None, layer_timing

    decoder_stripes = [p for p in data_plans if my_node == p['decoder_node']]
    helper_stripes = [p for p in data_plans if my_node in p['helper_nodes']]
    failed_stripes = [
        p for p in data_plans if _is_failed_in_recovery_plan(p, my_node)
    ]
    is_failed = len(failed_stripes) > 0

    logger.info(
        "FRCheck recovery layer %s: rank %d — data=%d parity_skipped=%d "
        "decoder=%d helper=%d failed=%d stripes",
        layer_name, rank, len(data_plans), len(parity_plans),
        len(decoder_stripes), len(helper_stripes), len(failed_stripes),
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
            layer_buf = buf_pool.failed_layer_buf[: num_source_stripes * layer_block_size]
        else:
            layer_buf = allocate_hugepage_tensor(
                num_source_stripes * layer_block_size, fallback_pin_memory=True,
            )
            native.register_buffer(layer_buf.data_ptr(), layer_buf.numel())
        layer_buf.zero_()

    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}

    wave_size = buf_pool.concurrency if buf_pool is not None else max(n, 1)
    for wave_start in range(0, len(data_plans), wave_size):
        wave_plans = data_plans[wave_start:wave_start + wave_size]
        native.reset_recovery_batch()
        for submit_idx, stri_plan in enumerate(wave_plans):
            sid = stri_plan['stripe_id']
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

                if layer_buf is not None and original_role == int(StripeRole.SOURCE):
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
            )

        native.submit_recovery_sentinel()
        native.wait_recovery_batch()

    if is_failed:
        n_stored = src_block_per_node.get(my_node, 0)
        expected_stored = sum(
            1 for plan in failed_stripes
            if int(plan.get('original_role', -1)) == int(StripeRole.SOURCE)
        )
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
    layer_block_size: int,
    layer_total_bytes: int,
    n: int,
    rank: int,
    preloaded_blocks: Dict[int, torch.Tensor],
    buf_pool: Optional[_RecoveryBufPool] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Run stripe-level RS decode recovery for one layer via C++ batch pipeline."""
    return _submit_recovery_network(
        native, manager, layer_name, layer_block_size, layer_total_bytes,
        n, rank, preloaded_blocks, buf_pool,
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
    frcheck_wait_for_async_recovery()
    manager = FRCheckManager()
    if manager.get_native() is None:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    logger.info("FRCheck: tearing down native module after training (rank %d)", rank)
    if getattr(args, 'frcheck_async_parity', False):
        _wait_previous_async_writers()
    manager.cleanup(teardown=True)


# ---------------------------------------------------------------------------
# Load teardown (ECLATIN-style: explicit cleanup after recovery only)
# ---------------------------------------------------------------------------

def _teardown_frcheck_native_after_load() -> None:
    """Stop and release native module after load/recovery (not used on save path)."""
    manager = FRCheckManager()
    if manager.get_native() is None:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    logger.info("FRCheck legacy load: cleaning up native module (rank %d)", rank)
    manager.cleanup(teardown=True)


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

    timings: Dict[str, float] = {
        'prep': 0.0,
        'main_io': 0.0,
        'disk_io': 0.0,
        'network_encode': 0.0,
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
    data_recovery_plans = [
        p for p in manager.recovery_stripe_plans
        if _is_data_recovery_plan(p, n)
    ]
    parity_recovery_plans = [
        p for p in manager.recovery_stripe_plans
        if not _is_data_recovery_plan(p, n)
    ]
    if manager.recovery_stripe_plans:
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
                    native.register_buffer(blk.data_ptr(), blk.numel())
        logger.info(
            "FRCheck recovery: survivor disk prep rank=%d preload_blocks=%d",
            rank, sum(len(blocks) for blocks in preloaded.values()),
        )
    n_disk_blocks = sum(len(blocks) for blocks in preloaded.values())
    timings['disk_io'] = time.time() - t_disk
    if is_survivor or involved:
        logger.info(
            "FRCheck recovery: disk prep rank=%d blocks=%d time=%.2fs",
            rank, n_disk_blocks, timings['disk_io'],
        )

    if is_survivor and not involved:
        logger.info(
            "FRCheck recovery: rank %d not involved in recovery, local load only",
            rank,
        )

    max_block_size = _max_layer_block_size(
        rank, all_layer_order, all_layer_metadata, saved_block_size,
    )
    is_decoder = any(
        my_node == p['decoder_node'] for p in data_recovery_plans
    )
    is_helper = any(
        my_node in p['helper_nodes'] for p in data_recovery_plans
    )
    buf_pool = _allocate_recovery_buf_pool(
        native, n, max_block_size, is_failed, is_decoder, is_helper,
        dual_failure=getattr(manager, 'recovery_dual_failure', False),
    )

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
    layerwise_records: List[_FRCheckLayerReadyRecord] = []
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
        )
        runtime_for_recovery = _FRCheckLayerwiseRuntime(pending_records)
        _set_active_frcheck_layerwise_runtime(runtime_for_recovery)
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
        )
        if job is None:
            continue
        recovery_jobs.append(job)

    detached_transformer_recovery = bool(
        async_forward
        and async_detach_safe
        and is_failed
        and runtime_for_recovery is not None
    )
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
        for job in sync_jobs:
            record, _layer_timing = _run_layer_recovery_job(
                job, manager, native, n, rank, is_failed, preloaded,
                buf_pool, full_buf, global_tensor_infos,
                runtime=runtime_for_recovery,
            )
            if record is not None:
                layerwise_records.append(record)
        if async_jobs:
            worker = _start_layer_recovery_worker(
                async_jobs, manager, native, n, rank, is_failed, preloaded,
                buf_pool, full_buf, global_tensor_infos, runtime_for_recovery,
                cleanup_after=True,
            )
            _set_active_frcheck_recovery_worker(worker)
            logger.info(
                "FRCheck recovery: async-forward detached transformer worker "
                "started rank=%d sync_jobs=%d async_jobs=%d",
                rank, len(sync_jobs), len(async_jobs),
            )
    elif async_forward and involved and recovery_jobs:
        worker = _start_layer_recovery_worker(
            recovery_jobs, manager, native, n, rank, is_failed, preloaded,
            buf_pool, full_buf, global_tensor_infos, runtime_for_recovery,
            cleanup_after=False,
        )
        _set_active_frcheck_recovery_worker(worker)
        logger.info(
            "FRCheck recovery: async-forward worker started rank=%d jobs=%d",
            rank, len(recovery_jobs),
        )
        frcheck_wait_for_async_recovery()
    else:
        for job in recovery_jobs:
            record, _layer_timing = _run_layer_recovery_job(
                job, manager, native, n, rank, is_failed, preloaded,
                buf_pool, full_buf, global_tensor_infos,
                runtime=runtime_for_recovery,
            )
            if record is not None:
                layerwise_records.append(record)

    timings['network_encode'] = time.time() - t_net

    t_rebuild = time.time()
    main_payload['tensor_buffer'] = full_buf[:total_tensor_size]
    if detached_transformer_recovery:
        result = _reconstruct_with_layer_placeholders(
            main_payload, full_buf[:total_tensor_size], flat_key_roots,
        )
    else:
        result = _reconstruct_from_main_payload(main_payload, flat_key_roots)
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
                    "error": r.error,
                    "tensors": r.tensors,
                }
                for r in records_to_export
            ],
        }
        logger.info(
            "FRCheck recovery: layerwise runtime exported layers=%d "
            "materialize_total=%.4fs",
            len(records_to_export),
            sum(r.materialize_s for r in records_to_export),
        )
    timings['rebuild_sd'] = time.time() - t_rebuild

    if not detached_transformer_recovery:
        _teardown_frcheck_native_after_load()
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


def _reconstruct_with_layer_placeholders(
    main_payload: Dict,
    full_buf: torch.Tensor,
    flat_key_roots: list = None,
) -> Dict[str, Any]:
    """Reconstruct state_dict with layer-owned model/optimizer tensors as placeholders."""
    tensor_infos = main_payload.get("tensor_infos", [])
    non_tensor_data = main_payload.get("non_tensor_data", {})
    flat_key_roots = flat_key_roots or main_payload.get("flat_key_roots", [])

    buf = full_buf.detach().contiguous().reshape(-1).view(torch.uint8)
    tensor_data: List[torch.Tensor] = []
    placeholder_count = 0
    placeholder_bytes = 0
    for info in tensor_infos:
        key = getattr(info, "key", "")
        ownership = _classify_frcheck_tensor(
            key, main_payload.get("optimizer_layer_map", {})
        )
        if ownership.layer_idx >= 0 and ownership.kind in ("model_layer", "optimizer_layer"):
            tensor_data.append(torch.zeros(tuple(info.shape), dtype=info.dtype))
            placeholder_count += 1
            placeholder_bytes += int(getattr(info, "size_bytes", 0))
            continue
        start = int(getattr(info, "offset", 0))
        size = int(getattr(info, "size_bytes", 0))
        tensor_bytes = buf[start:start + size]
        tensor_data.append(tensor_bytes.view(info.dtype).reshape(info.shape))

    decomposed = DecomposedStateDict(
        non_tensor_data=non_tensor_data,
        tensor_infos=tensor_infos,
        tensor_data=tensor_data,
        flat_key_roots=set(flat_key_roots) if flat_key_roots else set(),
    )
    result = reconstruct_state_dict(decomposed)
    logger.info(
        "FRCheck recovery: reconstructed with layer-owned placeholders "
        "tensors=%d bytes=%d",
        placeholder_count, placeholder_bytes,
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

    logger.info(
        "FRCheck load: rank=%d assembling from %d SOURCE stripes, layers=%d",
        rank, num_source, len(layer_names),
    )

    safety_margin = max(int(total_tensor_size * 0.01), 4096)
    if out_buf is not None:
        if out_buf.numel() < total_tensor_size + safety_margin:
            raise RuntimeError(
                f"FRCheck load: out_buf too small ({out_buf.numel()} "
                f"< {total_tensor_size + safety_margin})"
            )
        full_buf = out_buf
    else:
        full_buf = torch.zeros(total_tensor_size + safety_margin, dtype=torch.uint8)

    cache = source_block_cache if source_block_cache is not None else {}
    cache_keys = cache_source_keys if cache_source_keys is not None else set()
    key_to_local: Dict[str, Tuple[torch.Tensor, int]] = {}

    for layer_name in layer_names:
        layer_dir = checkpoint_dir / layer_name
        layer_main_path = layer_dir / f"frcheck_layer_main_rank{rank}.pt"
        if not layer_main_path.is_file():
            logger.warning(
                "FRCheck load: missing layer_main for %s, skipping", layer_name,
            )
            continue

        layer_meta = torch.load(layer_main_path, map_location="cpu", weights_only=False)
        layer_block_size = int(layer_meta.get("block_size", 0))
        if layer_block_size <= 0:
            raise RuntimeError(
                f"FRCheck load: invalid block_size for layer {layer_name}"
            )
        layer_infos = layer_meta.get("tensor_infos", [])
        layer_buf = torch.zeros(num_source * layer_block_size, dtype=torch.uint8)

        for stripe_id, blk_idx in source_stripes:
            block_path = (
                layer_dir / f"stripe_{stripe_id}" / f"frcheck_shard_rank{rank}.pt"
            )
            src_offset = blk_idx * layer_block_size
            dst_slice = layer_buf[src_offset:src_offset + layer_block_size]
            copy_len = _read_frbk_block_into(dst_slice, str(block_path))
            if copy_len == 0:
                logger.warning(
                    "FRCheck load: missing block %s, skipping stripe", block_path,
                )
                continue
            cache_key = (layer_name, stripe_id, int(StripeRole.SOURCE))
            if cache_key in cache_keys:
                cache[cache_key] = dst_slice[:layer_block_size].clone()

        for info in layer_infos:
            key = getattr(info, "key", "")
            if key:
                key_to_local[key] = (layer_buf, getattr(info, "offset", 0))

    for global_info in global_tensor_infos:
        key = getattr(global_info, "key", "")
        if key not in key_to_local:
            continue
        layer_buf, local_offset = key_to_local[key]
        global_offset = getattr(global_info, "offset", 0)
        size = getattr(global_info, "size_bytes", 0)
        if size > 0 and global_offset + size <= full_buf.numel():
            full_buf[global_offset:global_offset + size].copy_(
                layer_buf[local_offset:local_offset + size]
            )

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
    full_buf = _build_full_buf_from_local_blocks(checkpoint_dir, rank, main_payload)

    payload = dict(main_payload)
    payload["tensor_buffer"] = full_buf[:total_tensor_size]
    result = _reconstruct_from_main_payload(payload, flat_key_roots)
    if teardown_native:
        _teardown_frcheck_native_after_load()
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
    logger.info(
        "FRCheck load: hw_failure=%s failed_ranks_raw=%r failed_ranks_parsed=%r",
        hw_failure, failed_ranks_str, failed_ranks_parsed,
    )

    failed_ranks = failed_ranks_parsed
    if failed_ranks is None and failed_ranks_str:
        failed_ranks = [int(x.strip()) for x in failed_ranks_str.split(",")]

    if hw_failure and failed_ranks:
        logger.info("FRCheck: hardware recovery mode — failed ranks %s", failed_ranks)
        result, _t_fr = recover_frcheck_legacy_hardware(checkpoint_name, failed_ranks)
        _t_fr['total'] = _t_fr['network_encode'] + _t_fr['rebuild_sd']
        logger.info(
            "FRCheck legacy load timing (HW): "
            "total=%(total).2fs prep=%(prep).2fs main_io=%(main_io).2fs "
            "disk_io=%(disk_io).2fs network_encode=%(network_encode).2fs "
            "rebuild_sd=%(rebuild_sd).2fs (prep excluded from total)",
            _t_fr,
        )
        return result

    t0 = time.time()
    result = _assemble_state_dict_from_local_blocks(checkpoint_name)
    logger.info(
        "FRCheck legacy load timing: assemble_from_blocks %.2fs",
        time.time() - t0,
    )
    return result
