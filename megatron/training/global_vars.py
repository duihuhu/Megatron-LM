# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import sys
import torch
import time

from megatron.core import Timers
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


def finish_recovery_to_forward_timer(label: str = "forward_step_end") -> None:
    """Log elapsed time from recovery network/decode start to next forward step end."""
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
    role = context.get("role", "")
    role_text = f" role={role}" if role else ""
    print(
        f"{timer['scheme']} recovery-to-forward: "
        f"rank={timer['rank']}{role_text} elapsed={elapsed:.4f}s",
        flush=True,
    )



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
