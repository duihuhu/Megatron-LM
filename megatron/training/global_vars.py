# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import sys
import threading
import torch
import time

from megatron.core import Timers
from megatron.core import parallel_state
from megatron.core.config import set_experimental_flag
from megatron.core.energy_monitor import EnergyMonitor
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator, unset_num_microbatches_calculator
from megatron.training import dist_signal_handler
from megatron.training.tokenizer import build_tokenizer

_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ONE_LOGGER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_ENERGY_MONITOR = None
_GLOBAL_SIGNAL_HANDLER = None
_GLOBAL_RECOVERY_TO_FORWARD_TIMER = None
_GLOBAL_RECOVERY_TIMING_SUMMARIES = {}
_GLOBAL_RECOVERY_TIMING_SUMMARIES_LOCK = threading.Lock()
_GLOBAL_FT_LOAD_TIMING_CONTEXT = None

def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER

def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS

def get_energy_monitor():
    """Return energy monitor."""
    _ensure_var_is_initialized(_GLOBAL_ENERGY_MONITOR, 'energy monitor')
    return _GLOBAL_ENERGY_MONITOR

def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def start_recovery_to_forward_timer(
    scheme: str, phase: str = "network_decode", **context
) -> None:
    """Start a one-shot timer that ends after the next forward step."""
    global _GLOBAL_RECOVERY_TO_FORWARD_TIMER
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    now = time.time()
    _GLOBAL_RECOVERY_TO_FORWARD_TIMER = {
        "scheme": scheme,
        "phase": phase,
        "rank": rank,
        "start": now,
        "marks": [(phase, now)],
        "context": dict(context),
        "first_layer_start_recorded": False,
    }


def update_recovery_to_forward_timer_context(**context) -> None:
    """Attach metadata such as recovery role to the active recovery timer."""
    timer = _GLOBAL_RECOVERY_TO_FORWARD_TIMER
    if timer is None:
        return
    timer.setdefault("context", {}).update(context)


def mark_recovery_to_forward_timer(label: str) -> None:
    """Record an intermediate recovery-to-forward timing mark."""
    timer = _GLOBAL_RECOVERY_TO_FORWARD_TIMER
    if timer is None:
        return
    timer.setdefault("marks", []).append((label, time.time()))


def get_recovery_to_forward_timer_start() -> float:
    """Return the active recovery-to-forward start timestamp, or 0.0."""
    timer = _GLOBAL_RECOVERY_TO_FORWARD_TIMER
    if timer is None:
        return 0.0
    return float(timer.get("start", 0.0) or 0.0)


def restart_recovery_to_forward_timer(label: str) -> None:
    """Move the active recovery-to-forward timer start to a later phase."""
    timer = _GLOBAL_RECOVERY_TO_FORWARD_TIMER
    if timer is None:
        return
    now = time.time()
    timer["phase"] = label
    timer["start"] = now
    timer["marks"] = [(label, now)]


def _get_ranks_per_node() -> int:
    """Detect local world size using the recovery managers' precedence."""
    for key in (
        "LOCAL_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MPI_LOCALNRANKS",
        "MV2_COMM_WORLD_LOCAL_SIZE",
    ):
        value = os.environ.get(key)
        if not value:
            continue
        try:
            parsed = int(value.strip())
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            continue
    slurm_value = os.environ.get("SLURM_NTASKS_PER_NODE")
    if slurm_value:
        token = slurm_value.split(",", 1)[0].split("(", 1)[0].strip()
        try:
            parsed = int(token)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
    return max(1, torch.cuda.device_count())


def record_recovery_first_layer_start() -> None:
    """Record the first real layer start after an active recovery."""
    timer = _GLOBAL_RECOVERY_TO_FORWARD_TIMER
    if timer is None or timer.get("first_layer_start_recorded", False):
        return
    timer["first_layer_start_recorded"] = True
    elapsed_s = time.time() - float(timer.get("start", 0.0))
    rank = int(timer.get("rank", 0))
    is_node0_rank = rank < _get_ranks_per_node()
    stash_recovery_timing_summary(
        "recovery_first_layer_start",
        {
            "present": bool(is_node0_rank),
            "elapsed_s": elapsed_s if is_node0_rank else -1.0,
            "rank": rank,
            "scheme": str(timer.get("scheme", "FT")),
            "phase": str(timer.get("phase", "")),
        },
    )


def stash_recovery_timing_summary(name: str, values: dict) -> None:
    """Store local recovery timing values for a later safe collective logging point."""
    with _GLOBAL_RECOVERY_TIMING_SUMMARIES_LOCK:
        _GLOBAL_RECOVERY_TIMING_SUMMARIES[name] = dict(values)



def add_recovery_teardown_time(elapsed_s: float) -> None:
    """Accumulate local teardown time to exclude from recovery-to-forward."""
    with _GLOBAL_RECOVERY_TIMING_SUMMARIES_LOCK:
        summary = _GLOBAL_RECOVERY_TIMING_SUMMARIES.setdefault("teardown", {})
        summary["elapsed_s"] = float(summary.get("elapsed_s", 0.0)) + float(elapsed_s)



def finish_recovery_to_forward_timer(label: str = "forward_step_end") -> None:
    """Record elapsed time from recovery start to the next forward step end."""
    global _GLOBAL_RECOVERY_TO_FORWARD_TIMER
    timer = _GLOBAL_RECOVERY_TO_FORWARD_TIMER
    if timer is None:
        return
    mark_recovery_to_forward_timer(label)
    _GLOBAL_RECOVERY_TO_FORWARD_TIMER = None
    marks = timer.get("marks", [])
    start = timer["start"]
    elapsed = marks[-1][1] - start if marks else 0.0
    context = timer.get("context", {}) or {}
    if context.get("rank0_only_max"):
        with _GLOBAL_RECOVERY_TIMING_SUMMARIES_LOCK:
            teardown = dict(_GLOBAL_RECOVERY_TIMING_SUMMARIES.get("teardown", {}))
        teardown_s = float((teardown or {}).get("elapsed_s", 0.0))
        mark_deltas = []
        mark_times = {}
        seen_labels = set()
        for mark_label, mark_time in marks:
            label = str(mark_label)
            mark_times.setdefault(label, float(mark_time))
            if label in seen_labels:
                continue
            seen_labels.add(label)
            mark_deltas.append((label, float(mark_time - start)))

        def delta_between(first_label: str, second_label: str) -> float:
            first = mark_times.get(first_label)
            second = mark_times.get(second_label)
            if first is None or second is None or second < first:
                return 0.0
            return float(second - first)

        pp_rank = -1
        tp_rank = -1
        pp_world_size = -1
        try:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        except Exception:
            pass
        stash_recovery_timing_summary(
            "recovery_to_forward",
            {
                "elapsed_s": elapsed,
                "teardown_s": teardown_s,
                "scheme": timer["scheme"],
                "marks": mark_deltas,
                "h2d_to_loop_iter_s": delta_between("h2d_done", "train_loop_iteration_start"),
                "loop_iter_to_async_save_done_s": delta_between("train_loop_iteration_start", "async_save_finalize_done"),
                "async_save_to_microbatch_done_s": delta_between("async_save_finalize_done", "microbatch_update_done"),
                "microbatch_to_ft_hook_done_s": delta_between("microbatch_update_done", "ft_step_start_hook_done"),
                "ft_hook_to_train_entry_s": delta_between("ft_step_start_hook_done", "train_step_entry"),
                "train_entry_to_zero_grad_done_s": delta_between("train_step_entry", "zero_grad_done"),
                "zero_grad_to_train_step_start_s": delta_between("zero_grad_done", "train_step_start"),
                "train_step_to_forward_backward_start_s": delta_between("train_step_start", "forward_backward_start"),
                "forward_backward_to_timer_start_done_s": delta_between(
                    "forward_backward_start", "forward_backward_timer_start_done"
                ),
                "timer_start_done_to_recv_start_s": delta_between(
                    "forward_backward_timer_start_done", "first_forward_recv_start"
                ),
                "first_forward_recv_s": delta_between(
                    "first_forward_recv_start", "first_forward_recv_done"
                ),
                "recv_done_to_forward_start_s": delta_between(
                    "first_forward_recv_done", "forward_step_start"
                ),
                "forward_backward_to_forward_start_s": delta_between("forward_backward_start", "forward_step_start"),
                "h2d_to_train_step_s": delta_between("h2d_done", "train_step_start"),
                "train_step_to_forward_start_s": delta_between("train_step_start", "forward_step_start"),
                "h2d_to_forward_start_s": delta_between("h2d_done", "forward_step_start"),
                "forward_step_s": delta_between("forward_step_start", label),
                "forward_step_to_forward_backward_done_s": delta_between(label, "forward_backward_done"),
                "rank": float(timer.get("rank", -1)),
                "pp_rank": float(pp_rank),
                "tp_rank": float(tp_rank),
                "pp_world_size": float(pp_world_size),
            },
        )
        return
    role = context.get("role", "")
    role_text = f" role={role}" if role else ""
    print(
        f"{timer['scheme']} recovery-to-forward: "
        f"rank={timer['rank']}{role_text} elapsed={elapsed:.4f}s",
        flush=True,
    )



def flush_recovery_timing_summaries() -> None:
    """All-reduce and log pending recovery timings at a common train-step boundary."""
    global _GLOBAL_RECOVERY_TIMING_SUMMARIES
    with _GLOBAL_RECOVERY_TIMING_SUMMARIES_LOCK:
        has_frcheck_milestones = "frcheck_first_layer_milestones" in (
            _GLOBAL_RECOVERY_TIMING_SUMMARIES
        )
    if has_frcheck_milestones:
        try:
            from megatron.training.frcheck_legacy import resolve_frcheck_first_layer_cuda_events
            resolve_frcheck_first_layer_cuda_events()
        except (ImportError, RuntimeError):
            pass
    pipeline_keys = [
        "pipeline_s",
        "network_submit_s",
        "network_wait_s",
        "materialize_s",
        "recovery_net_s",
        "recovery_decode_s",
        "h2d_s",
        "serial_work_s",
        "pipeline_overlap_s",
        "first_network_done_s",
        "last_model_done_s",
        "last_optimizer_done_s",
        "last_common_done_s",
        "last_materialize_done_s",
        "pipeline_start_delay_s",
    ]
    with _GLOBAL_RECOVERY_TIMING_SUMMARIES_LOCK:
        pending_summaries = dict(_GLOBAL_RECOVERY_TIMING_SUMMARIES)
    pipeline = pending_summaries.get("frcheck_hw_pipeline")
    rtf = pending_summaries.get("recovery_to_forward")
    first_layer = pending_summaries.get("frcheck_first_layer_milestones")
    recovery_first_layer = pending_summaries.get("recovery_first_layer_start")
    frcheck_forward_backward = pending_summaries.get("frcheck_forward_backward")

    rtf_elapsed_s = float((rtf or {}).get("elapsed_s", 0.0))
    rtf_teardown_s = float((rtf or {}).get("teardown_s", 0.0))
    rtf_adjusted_elapsed_s = max(0.0, rtf_elapsed_s - rtf_teardown_s)
    values = [1.0 if pipeline is not None else 0.0]
    values.extend(float((pipeline or {}).get(key, 0.0)) for key in pipeline_keys)
    values.append(1.0 if rtf is not None else 0.0)
    values.append(rtf_adjusted_elapsed_s)
    values.append(-rtf_adjusted_elapsed_s if rtf is not None else -1.0e30)
    values.append(rtf_elapsed_s)
    values.append(rtf_teardown_s)
    breakdown_keys = [
        "h2d_to_loop_iter_s",
        "loop_iter_to_async_save_done_s",
        "async_save_to_microbatch_done_s",
        "microbatch_to_ft_hook_done_s",
        "ft_hook_to_train_entry_s",
        "train_entry_to_zero_grad_done_s",
        "zero_grad_to_train_step_start_s",
        "train_step_to_forward_backward_start_s",
        "forward_backward_to_timer_start_done_s",
        "timer_start_done_to_recv_start_s",
        "first_forward_recv_s",
        "recv_done_to_forward_start_s",
        "forward_backward_to_forward_start_s",
        "h2d_to_train_step_s",
        "train_step_to_forward_start_s",
        "h2d_to_forward_start_s",
        "forward_step_s",
        "forward_step_to_forward_backward_done_s",
    ]
    values.extend(float((rtf or {}).get(key, 0.0)) for key in breakdown_keys)
    recovery_first_layer_offset = len(values)
    recovery_first_layer_present = bool(
        (recovery_first_layer or {}).get("present", False)
    )
    values.extend(
        [
            1.0 if recovery_first_layer_present else 0.0,
            (
                float((recovery_first_layer or {}).get("elapsed_s", -1.0e30))
                if recovery_first_layer_present
                else -1.0e30
            ),
        ]
    )
    frcheck_forward_backward_offset = len(values)
    frcheck_forward_backward_elapsed_s = float(
        (frcheck_forward_backward or {}).get("elapsed_s", 0.0)
    )
    values.extend(
        [
            1.0 if frcheck_forward_backward is not None else 0.0,
            frcheck_forward_backward_elapsed_s,
            (
                -frcheck_forward_backward_elapsed_s
                if frcheck_forward_backward is not None
                else -1.0e30
            ),
        ]
    )

    max_contributor = None
    frcheck_forward_backward_max_contributor = None
    gathered_timing_info = []
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        values = [float(item) for item in tensor.cpu().tolist()]
        rank = torch.distributed.get_rank()

        if (
            values[1 + len(pipeline_keys)] > 0.0
            or values[frcheck_forward_backward_offset] > 0.0
        ):
            local_info = {
                "elapsed_s": rtf_adjusted_elapsed_s,
                "raw_elapsed_s": rtf_elapsed_s,
                "rank": int((rtf or {}).get("rank", rank)),
                "pp_rank": int((rtf or {}).get("pp_rank", -1)),
                "tp_rank": int((rtf or {}).get("tp_rank", -1)),
                "pp_world_size": int((rtf or {}).get("pp_world_size", -1)),
                "h2d_to_loop_iter_s": float((rtf or {}).get("h2d_to_loop_iter_s", 0.0)),
                "loop_iter_to_async_save_done_s": float((rtf or {}).get("loop_iter_to_async_save_done_s", 0.0)),
                "async_save_to_microbatch_done_s": float((rtf or {}).get("async_save_to_microbatch_done_s", 0.0)),
                "microbatch_to_ft_hook_done_s": float((rtf or {}).get("microbatch_to_ft_hook_done_s", 0.0)),
                "ft_hook_to_train_entry_s": float((rtf or {}).get("ft_hook_to_train_entry_s", 0.0)),
                "train_entry_to_zero_grad_done_s": float((rtf or {}).get("train_entry_to_zero_grad_done_s", 0.0)),
                "zero_grad_to_train_step_start_s": float((rtf or {}).get("zero_grad_to_train_step_start_s", 0.0)),
                "train_step_to_forward_backward_start_s": float((rtf or {}).get("train_step_to_forward_backward_start_s", 0.0)),
                "forward_backward_to_timer_start_done_s": float((rtf or {}).get("forward_backward_to_timer_start_done_s", 0.0)),
                "timer_start_done_to_recv_start_s": float((rtf or {}).get("timer_start_done_to_recv_start_s", 0.0)),
                "first_forward_recv_s": float((rtf or {}).get("first_forward_recv_s", 0.0)),
                "recv_done_to_forward_start_s": float((rtf or {}).get("recv_done_to_forward_start_s", 0.0)),
                "forward_backward_to_forward_start_s": float((rtf or {}).get("forward_backward_to_forward_start_s", 0.0)),
                "h2d_to_train_step_s": float((rtf or {}).get("h2d_to_train_step_s", 0.0)),
                "train_step_to_forward_start_s": float((rtf or {}).get("train_step_to_forward_start_s", 0.0)),
                "h2d_to_forward_start_s": float((rtf or {}).get("h2d_to_forward_start_s", 0.0)),
                "forward_step_s": float((rtf or {}).get("forward_step_s", 0.0)),
                "forward_step_to_forward_backward_done_s": float((rtf or {}).get("forward_step_to_forward_backward_done_s", 0.0)),
                "first_layer_milestones": dict(first_layer or {}),
                "marks": list((rtf or {}).get("marks", [])),
                "frcheck_forward_backward_present": frcheck_forward_backward is not None,
                "frcheck_forward_backward_elapsed_s": frcheck_forward_backward_elapsed_s,
                "frcheck_forward_backward_rank": int(
                    (frcheck_forward_backward or {}).get("rank", rank)
                ),
                "frcheck_forward_backward_pp_rank": int(
                    (frcheck_forward_backward or {}).get("pp_rank", -1)
                ),
                "frcheck_forward_backward_tp_rank": int(
                    (frcheck_forward_backward or {}).get("tp_rank", -1)
                ),
            }
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, local_info)
            valid = [item for item in gathered if isinstance(item, dict)]
            gathered_timing_info = valid
            if valid and values[1 + len(pipeline_keys)] > 0.0:
                max_contributor = max(valid, key=lambda item: float(item.get("elapsed_s", -1.0)))
            frcheck_forward_backward_contributors = [
                item
                for item in valid
                if item.get("frcheck_forward_backward_present", False)
            ]
            if frcheck_forward_backward_contributors:
                frcheck_forward_backward_max_contributor = max(
                    frcheck_forward_backward_contributors,
                    key=lambda item: float(
                        item.get("frcheck_forward_backward_elapsed_s", -1.0)
                    ),
                )
    else:
        rank = 0
        if rtf is not None:
            max_contributor = {
                "elapsed_s": rtf_adjusted_elapsed_s,
                "raw_elapsed_s": rtf_elapsed_s,
                "rank": int((rtf or {}).get("rank", 0)),
                "pp_rank": int((rtf or {}).get("pp_rank", -1)),
                "tp_rank": int((rtf or {}).get("tp_rank", -1)),
                "pp_world_size": int((rtf or {}).get("pp_world_size", -1)),
                "h2d_to_loop_iter_s": float((rtf or {}).get("h2d_to_loop_iter_s", 0.0)),
                "loop_iter_to_async_save_done_s": float((rtf or {}).get("loop_iter_to_async_save_done_s", 0.0)),
                "async_save_to_microbatch_done_s": float((rtf or {}).get("async_save_to_microbatch_done_s", 0.0)),
                "microbatch_to_ft_hook_done_s": float((rtf or {}).get("microbatch_to_ft_hook_done_s", 0.0)),
                "ft_hook_to_train_entry_s": float((rtf or {}).get("ft_hook_to_train_entry_s", 0.0)),
                "train_entry_to_zero_grad_done_s": float((rtf or {}).get("train_entry_to_zero_grad_done_s", 0.0)),
                "zero_grad_to_train_step_start_s": float((rtf or {}).get("zero_grad_to_train_step_start_s", 0.0)),
                "train_step_to_forward_backward_start_s": float((rtf or {}).get("train_step_to_forward_backward_start_s", 0.0)),
                "forward_backward_to_timer_start_done_s": float((rtf or {}).get("forward_backward_to_timer_start_done_s", 0.0)),
                "timer_start_done_to_recv_start_s": float((rtf or {}).get("timer_start_done_to_recv_start_s", 0.0)),
                "first_forward_recv_s": float((rtf or {}).get("first_forward_recv_s", 0.0)),
                "recv_done_to_forward_start_s": float((rtf or {}).get("recv_done_to_forward_start_s", 0.0)),
                "forward_backward_to_forward_start_s": float((rtf or {}).get("forward_backward_to_forward_start_s", 0.0)),
                "h2d_to_train_step_s": float((rtf or {}).get("h2d_to_train_step_s", 0.0)),
                "train_step_to_forward_start_s": float((rtf or {}).get("train_step_to_forward_start_s", 0.0)),
                "h2d_to_forward_start_s": float((rtf or {}).get("h2d_to_forward_start_s", 0.0)),
                "forward_step_s": float((rtf or {}).get("forward_step_s", 0.0)),
                "forward_step_to_forward_backward_done_s": float((rtf or {}).get("forward_step_to_forward_backward_done_s", 0.0)),
                "first_layer_milestones": dict(first_layer or {}),
                "marks": list((rtf or {}).get("marks", [])),
            }
            gathered_timing_info = [max_contributor]
        if frcheck_forward_backward is not None:
            frcheck_forward_backward_max_contributor = {
                "frcheck_forward_backward_elapsed_s": frcheck_forward_backward_elapsed_s,
                "frcheck_forward_backward_rank": int(
                    frcheck_forward_backward.get("rank", 0)
                ),
                "frcheck_forward_backward_pp_rank": int(
                    frcheck_forward_backward.get("pp_rank", -1)
                ),
                "frcheck_forward_backward_tp_rank": int(
                    frcheck_forward_backward.get("tp_rank", -1)
                ),
            }

    if rank == 0:
        import logging
        logger = logging.getLogger("megatron.training.frcheck_legacy")
        if values[0] > 0.0:
            summary = dict(zip(pipeline_keys, values[1:1 + len(pipeline_keys)]))
            logger.info(
                "FRCheck HW pipeline breakdown: pipeline_s=%.2fs "
                "network_submit_s=%.2fs network_wait_s=%.2fs materialize_s=%.2fs "
                "net_s=%.2fs decode_s=%.2fs h2d_s=%.2fs "
                "serial_work_s=%.2fs overlap_s=%.2fs first_network_done_s=%.2fs "
                "last_model_done_s=%.2fs last_optimizer_done_s=%.2fs "
                "last_common_done_s=%.2fs last_materialize_done_s=%.2fs "
                "pipeline_start_delay_s=%.4fs",
                summary["pipeline_s"],
                summary["network_submit_s"],
                summary["network_wait_s"],
                summary["materialize_s"],
                summary["recovery_net_s"],
                summary["recovery_decode_s"],
                summary["h2d_s"],
                summary["serial_work_s"],
                summary["pipeline_overlap_s"],
                summary["first_network_done_s"],
                summary["last_model_done_s"],
                summary["last_optimizer_done_s"],
                summary["last_common_done_s"],
                summary["last_materialize_done_s"],
                summary["pipeline_start_delay_s"],
            )
        if values[recovery_first_layer_offset] > 0.0:
            scheme = str(
                (recovery_first_layer or {}).get(
                    "scheme", (rtf or {}).get("scheme", "FT")
                )
            )
            logger.info(
                "%s recovery timing: "
                "node0.all_rank.max(recovery_start_to_train_start_s)=%.6fs",
                scheme,
                values[recovery_first_layer_offset + 1],
            )
        if values[frcheck_forward_backward_offset] > 0.0:
            elapsed_max_s = values[frcheck_forward_backward_offset + 1]
            elapsed_min_s = -values[frcheck_forward_backward_offset + 2]
            contributor = frcheck_forward_backward_max_contributor or {}
            logger.info(
                "FRCheck forward-backward timing: elapsed_min_s=%.6f "
                "elapsed_max_s=%.6f max_rank=%d pp_rank=%d tp_rank=%d",
                elapsed_min_s,
                elapsed_max_s,
                int(contributor.get("frcheck_forward_backward_rank", -1)),
                int(contributor.get("frcheck_forward_backward_pp_rank", -1)),
                int(contributor.get("frcheck_forward_backward_tp_rank", -1)),
            )
        if values[1 + len(pipeline_keys)] > 0.0:
            marks = (rtf or {}).get("marks", [])
            if marks:
                mark_text = " ".join(
                    f"{label}={delta:.4f}s" for label, delta in marks
                )
                scheme_for_marks = str((rtf or {}).get("scheme", "FT"))
                logger.debug("%s recovery-to-forward marks (rank0): %s", scheme_for_marks, mark_text)
            scheme = str((rtf or {}).get("scheme", "FRCheck"))
            elapsed_max_s = values[2 + len(pipeline_keys)]
            elapsed_min_s = -values[3 + len(pipeline_keys)]
            raw_elapsed_max_s = values[4 + len(pipeline_keys)]
            teardown_max_s = values[5 + len(pipeline_keys)]
            breakdown_offset = 6 + len(pipeline_keys)
            breakdown = dict(zip(breakdown_keys, values[breakdown_offset:breakdown_offset + len(breakdown_keys)]))
            logger.info(
                "%s recovery-to-forward post-H2D breakdown: "
                "h2d_to_loop_iter_s=%.4fs loop_iter_to_async_save_done_s=%.4fs "
                "async_save_to_microbatch_done_s=%.4fs microbatch_to_ft_hook_done_s=%.4fs "
                "ft_hook_to_train_entry_s=%.4fs train_entry_to_zero_grad_done_s=%.4fs "
                "zero_grad_to_train_step_start_s=%.4fs train_step_to_forward_backward_start_s=%.4fs "
                "forward_backward_to_timer_start_done_s=%.4fs timer_start_done_to_recv_start_s=%.4fs "
                "first_forward_recv_s=%.4fs recv_done_to_forward_start_s=%.4fs "
                "forward_backward_to_forward_start_s=%.4fs h2d_to_train_step_s=%.4fs "
                "train_step_to_forward_start_s=%.4fs h2d_to_forward_start_s=%.4fs "
                "forward_step_s=%.4fs forward_step_to_forward_backward_done_s=%.4fs",
                scheme,
                breakdown["h2d_to_loop_iter_s"],
                breakdown["loop_iter_to_async_save_done_s"],
                breakdown["async_save_to_microbatch_done_s"],
                breakdown["microbatch_to_ft_hook_done_s"],
                breakdown["ft_hook_to_train_entry_s"],
                breakdown["train_entry_to_zero_grad_done_s"],
                breakdown["zero_grad_to_train_step_start_s"],
                breakdown["train_step_to_forward_backward_start_s"],
                breakdown["forward_backward_to_timer_start_done_s"],
                breakdown["timer_start_done_to_recv_start_s"],
                breakdown["first_forward_recv_s"],
                breakdown["recv_done_to_forward_start_s"],
                breakdown["forward_backward_to_forward_start_s"],
                breakdown["h2d_to_train_step_s"],
                breakdown["train_step_to_forward_start_s"],
                breakdown["h2d_to_forward_start_s"],
                breakdown["forward_step_s"],
                breakdown["forward_step_to_forward_backward_done_s"],
            )
            if max_contributor is not None:
                logger.info(
                    "%s recovery-to-forward max contributor: "
                    "rank=%d pp_rank=%d/%d tp_rank=%d elapsed_s=%.4fs raw_elapsed_s=%.4fs "
                    "h2d_to_loop_iter_s=%.4fs loop_iter_to_async_save_done_s=%.4fs "
                    "async_save_to_microbatch_done_s=%.4fs microbatch_to_ft_hook_done_s=%.4fs "
                    "ft_hook_to_train_entry_s=%.4fs train_entry_to_zero_grad_done_s=%.4fs "
                    "zero_grad_to_train_step_start_s=%.4fs train_step_to_forward_backward_start_s=%.4fs "
                    "forward_backward_to_timer_start_done_s=%.4fs timer_start_done_to_recv_start_s=%.4fs "
                    "first_forward_recv_s=%.4fs recv_done_to_forward_start_s=%.4fs "
                    "forward_backward_to_forward_start_s=%.4fs h2d_to_train_step_s=%.4fs "
                    "train_step_to_forward_start_s=%.4fs h2d_to_forward_start_s=%.4fs "
                    "forward_step_s=%.4fs forward_step_to_forward_backward_done_s=%.4fs",
                    scheme,
                    int(max_contributor.get("rank", -1)),
                    int(max_contributor.get("pp_rank", -1)),
                    int(max_contributor.get("pp_world_size", -1)),
                    int(max_contributor.get("tp_rank", -1)),
                    float(max_contributor.get("elapsed_s", 0.0)),
                    float(max_contributor.get("raw_elapsed_s", 0.0)),
                    float(max_contributor.get("h2d_to_loop_iter_s", 0.0)),
                    float(max_contributor.get("loop_iter_to_async_save_done_s", 0.0)),
                    float(max_contributor.get("async_save_to_microbatch_done_s", 0.0)),
                    float(max_contributor.get("microbatch_to_ft_hook_done_s", 0.0)),
                    float(max_contributor.get("ft_hook_to_train_entry_s", 0.0)),
                    float(max_contributor.get("train_entry_to_zero_grad_done_s", 0.0)),
                    float(max_contributor.get("zero_grad_to_train_step_start_s", 0.0)),
                    float(max_contributor.get("train_step_to_forward_backward_start_s", 0.0)),
                    float(max_contributor.get("forward_backward_to_timer_start_done_s", 0.0)),
                    float(max_contributor.get("timer_start_done_to_recv_start_s", 0.0)),
                    float(max_contributor.get("first_forward_recv_s", 0.0)),
                    float(max_contributor.get("recv_done_to_forward_start_s", 0.0)),
                    float(max_contributor.get("forward_backward_to_forward_start_s", 0.0)),
                    float(max_contributor.get("h2d_to_train_step_s", 0.0)),
                    float(max_contributor.get("train_step_to_forward_start_s", 0.0)),
                    float(max_contributor.get("h2d_to_forward_start_s", 0.0)),
                    float(max_contributor.get("forward_step_s", 0.0)),
                    float(max_contributor.get("forward_step_to_forward_backward_done_s", 0.0)),
                )
                max_marks = max_contributor.get("marks", [])
                if max_marks:
                    logger.debug(
                        "%s recovery-to-forward max contributor marks: %s",
                        scheme,
                        " ".join(
                            f"{label}={float(delta):.4f}s"
                            for label, delta in max_marks
                        ),
                    )
            print(
                f"{scheme} recovery-to-forward: "
                f"elapsed_min={elapsed_min_s:.4f}s elapsed_max={elapsed_max_s:.4f}s "
                f"raw_elapsed_max={raw_elapsed_max_s:.4f}s teardown_max={teardown_max_s:.4f}s",
                flush=True,
            )
            if scheme == "FRCheck":
                milestone_items = [
                    item.get("first_layer_milestones", {})
                    for item in gathered_timing_info
                    if isinstance(item.get("first_layer_milestones", {}), dict)
                ]
                layer_indices = [
                    int(item.get("layer_idx", -1))
                    for item in milestone_items
                    if int(item.get("layer_idx", -1)) >= 0
                ]
                layer_idx = min(layer_indices) if layer_indices else -1

                def select_milestone(name: str, rank0_only: bool = False):
                    candidates = []
                    for item in milestone_items:
                        if not item.get(f"{name}_valid", False):
                            continue
                        contributor_rank = int(item.get("rank", -1))
                        if rank0_only and contributor_rank != 0:
                            continue
                        candidates.append((float(item.get(name, 0.0)), contributor_rank))
                    if not candidates:
                        return None
                    return max(candidates, key=lambda value: value[0])

                repair = select_milestone("repair_s")
                sent_to_failed = select_milestone("sent_to_failed_s")
                delivered = select_milestone("delivered_s")
                h2d = select_milestone("h2d_s")
                forward_done = select_milestone("forward_done_s")

                def format_milestone(value) -> str:
                    if value is None:
                        return "unavailable"
                    return f"{value[0]:.6f}s(rank={value[1]})"

                logger.info(
                    "FRCheck first-layer milestones: layer_idx=%d repair_s=%s "
                    "sent_to_failed_s=%s h2d_s=%s forward_done_s=%s "
                    "delivered_s=%s",
                    layer_idx,
                    format_milestone(repair),
                    format_milestone(sent_to_failed),
                    format_milestone(h2d),
                    format_milestone(forward_done),
                    format_milestone(delivered),
                )

    with _GLOBAL_RECOVERY_TIMING_SUMMARIES_LOCK:
        _GLOBAL_RECOVERY_TIMING_SUMMARIES = {}



def set_ft_load_timing_context(scheme: str, mode: str, timings: dict) -> None:
    """Store local FT load/recovery timings until model/optimizer H2D finishes."""
    global _GLOBAL_FT_LOAD_TIMING_CONTEXT
    _GLOBAL_FT_LOAD_TIMING_CONTEXT = {
        "scheme": scheme,
        "mode": mode,
        "timings": dict(timings),
    }


def get_ft_load_timing_context():
    """Return the pending FT load/recovery timing context, if any."""
    return _GLOBAL_FT_LOAD_TIMING_CONTEXT


def clear_ft_load_timing_context() -> None:
    """Clear the pending FT load/recovery timing context."""
    global _GLOBAL_FT_LOAD_TIMING_CONTEXT
    _GLOBAL_FT_LOAD_TIMING_CONTEXT = None

def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = dist_signal_handler.DistributedSignalHandler().__enter__()



def set_global_variables(args, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    set_args(args)

    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )
    if build_tokenizer:
        _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_wandb_writer(args)
    _set_one_logger(args)
    _set_adlr_autoresume(args)
    _set_timers(args)
    _set_energy_monitor(args)

    if args.enable_experimental:
        set_experimental_flag(True)

    if args.exit_signal_handler:
        _set_signal_handler()


def unset_global_variables():
    """Unset global vars.

    Useful for multiple runs. See `tests/unit_tests/ckpt_converter/test_ckpt_converter.py` for an example.
    """

    global _GLOBAL_ARGS
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    global _GLOBAL_TOKENIZER
    global _GLOBAL_TENSORBOARD_WRITER
    global _GLOBAL_WANDB_WRITER
    global _GLOBAL_ONE_LOGGER
    global _GLOBAL_ADLR_AUTORESUME
    global _GLOBAL_TIMERS
    global _GLOBAL_ENERGY_MONITOR
    global _GLOBAL_SIGNAL_HANDLER
    global _GLOBAL_RECOVERY_TO_FORWARD_TIMER
    global _GLOBAL_RECOVERY_TIMING_SUMMARIES

    _GLOBAL_ARGS = None
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    _GLOBAL_TOKENIZER = None
    _GLOBAL_TENSORBOARD_WRITER = None
    _GLOBAL_WANDB_WRITER = None
    _GLOBAL_ONE_LOGGER = None
    _GLOBAL_ADLR_AUTORESUME = None
    _GLOBAL_TIMERS = None
    _GLOBAL_ENERGY_MONITOR = None
    _GLOBAL_SIGNAL_HANDLER = None
    _GLOBAL_RECOVERY_TO_FORWARD_TIMER = None
    _GLOBAL_RECOVERY_TIMING_SUMMARIES = {}
    _GLOBAL_FT_LOAD_TIMING_CONTEXT = None

    unset_num_microbatches_calculator()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir and args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _set_wandb_writer(args):
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER,
                                   'wandb writer')
    if getattr(args, 'wandb_project', '') and args.rank == (args.world_size - 1):
        if args.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb
        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, 'wandb')
        wandb_config = vars(args)
        if 'kitchen_config_file' in wandb_config and wandb_config['kitchen_config_file'] is not None:
            # Log the contents of the config for discovery of what the quantization
            # settings were.
            with open(wandb_config['kitchen_config_file'], "r") as f:
                wandb_config['kitchen_config_file_contents'] = f.read()
        wandb_kwargs = {
            'dir': save_dir,
            'name': args.wandb_exp_name,
            'project': args.wandb_project,
            'config': wandb_config}
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb


def _set_one_logger(args):
    global _GLOBAL_ONE_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, 'one logger')

    if args.enable_one_logger and args.rank == (args.world_size - 1):
        if args.one_logger_async or getattr(args, 'wandb_project', ''):
            one_logger_async = True
        else:
            one_logger_async = False
        try:
            from one_logger import OneLogger
            config = {
               'project': args.one_logger_project,
               'name': args.one_logger_run_name,
               'async': one_logger_async,
            }
            one_logger = OneLogger(config=config)
            _GLOBAL_ONE_LOGGER = one_logger
        except Exception:
            print('WARNING: one_logger package is required to enable e2e metrics '
                  'tracking. please go to '
                  'https://confluence.nvidia.com/display/MLWFO/Package+Repositories'
                  ' for details to install it')

def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    if args.adlr_autoresume:
        if args.rank == 0:
            print('enabling autoresume ...', flush=True)
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except ImportError:
            print('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)

def _set_energy_monitor(args):
    """Initialize energy monitor."""
    global _GLOBAL_ENERGY_MONITOR
    _ensure_var_is_not_initialized(_GLOBAL_ENERGY_MONITOR, 'energy monitor')
    _GLOBAL_ENERGY_MONITOR = EnergyMonitor()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)

def destroy_global_vars():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None

    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None

    global _GLOBAL_TENSORBOARD_WRITER
    _GLOBAL_TENSORBOARD_WRITER = None

    global _GLOBAL_WANDB_WRITER
    _GLOBAL_WANDB_WRITER = None

    global _GLOBAL_ONE_LOGGER
    _GLOBAL_ONE_LOGGER = None

    global _GLOBAL_ADLR_AUTORESUME
    _GLOBAL_ADLR_AUTORESUME = None

    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = None

    global _GLOBAL_ENERGY_MONITOR
    _GLOBAL_ENERGY_MONITOR = None

    global _GLOBAL_SIGNAL_HANDLER
    _GLOBAL_SIGNAL_HANDLER = None
