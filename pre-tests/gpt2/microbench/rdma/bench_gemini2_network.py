#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Gemini2-style network-only microbenchmark.

Each rank sends one full contiguous buffer to a single replica target and
receives one full contiguous buffer from the previous rank. The timed section
uses GeminiReplicasNative submit/wait APIs and excludes tensor packing,
registration, connection setup, and disk I/O.
"""

import argparse
import glob
import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[4]
STRATEGY_DIR = ROOT / "megatron" / "core" / "dist_checkpointing" / "strategies"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip


def _parse_size(value: str) -> int:
    text = str(value).strip()
    mul = 1
    if text[-1:].lower() == "k":
        mul = 1024
        text = text[:-1]
    elif text[-1:].lower() == "m":
        mul = 1024 ** 2
        text = text[:-1]
    elif text[-1:].lower() == "g":
        mul = 1024 ** 3
        text = text[:-1]
    return int(float(text) * mul)


def _load_native(module_glob: str, module_name: str):
    matches = glob.glob(str(STRATEGY_DIR / module_glob))
    if not matches:
        raise RuntimeError(f"No native module matching {module_glob} in {STRATEGY_DIR}")
    spec = importlib.util.spec_from_file_location(module_name, matches[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load native module {matches[0]}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_ip(rank: int, local_rank: int) -> str:
    return resolve_ip("GEMINI_REPLICAS", rank=rank, local_rank=local_rank)


def _init_dist(args) -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    dist.init_process_group(
        backend=args.backend,
        rank=args.rank,
        world_size=args.world_size,
    )


def _all_gather_strings(value: str) -> List[str]:
    encoded = value.encode("utf-8")
    max_len = 256
    if len(encoded) >= max_len:
        raise RuntimeError(f"String too long for all_gather_strings: {value}")
    local = torch.zeros(max_len, dtype=torch.uint8)
    local[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    values = []
    for item in gathered:
        data = bytes(item.cpu().tolist()).split(b"\0", 1)[0]
        values.append(data.decode("utf-8"))
    return values


def _summary(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    p50 = statistics.median(ordered) if ordered else 0.0
    p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] if ordered else 0.0
    return {
        "min": min(ordered) if ordered else 0.0,
        "p50": p50,
        "p99": p99,
        "max": max(ordered) if ordered else 0.0,
        "avg": sum(ordered) / len(ordered) if ordered else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini2 network-only microbenchmark")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "8")))
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "6000")))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("GEMINI_REPLICAS_BASE_PORT", "36000")))
    parser.add_argument("--bytes", default="1G")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--channels-per-peer", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=int(os.environ.get("GEMINI2_GROUP_SIZE", "8")))
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--cpu-send", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.world_size <= 1:
        raise RuntimeError("Gemini2 bench requires world_size > 1")
    if args.group_size <= 1:
        raise RuntimeError("Gemini2 bench requires group_size > 1")
    if args.world_size % args.group_size != 0:
        raise RuntimeError("Gemini2 bench requires world_size to be divisible by group_size")
    print(f"PROGRESS,rank={args.rank},stage=set_device_start", flush=True)
    cuda_device = 0 if torch.cuda.device_count() == 1 else args.local_rank
    torch.cuda.set_device(cuda_device)
    print(f"PROGRESS,rank={args.rank},stage=init_dist_start,backend={args.backend}", flush=True)
    _init_dist(args)
    print(f"PROGRESS,rank={args.rank},stage=init_dist_done", flush=True)

    print(f"PROGRESS,rank={args.rank},stage=load_native_start", flush=True)
    native_mod = _load_native("gemini_replicas_native*.so", "gemini_replicas_native")
    print(f"PROGRESS,rank={args.rank},stage=load_native_done", flush=True)
    my_ip = _get_ip(args.rank, args.local_rank)
    print(f"PROGRESS,rank={args.rank},stage=all_gather_ip_start,ip={my_ip}", flush=True)
    all_ips = _all_gather_strings(my_ip)
    print(f"PROGRESS,rank={args.rank},stage=all_gather_ip_done", flush=True)

    num_groups = args.world_size // args.group_size
    group_id = args.rank % num_groups
    rank_in_group = args.rank // num_groups
    group_ranks = [group_id + idx * num_groups for idx in range(args.group_size)]
    target = group_ranks[(rank_in_group + 1) % args.group_size]
    source = group_ranks[(rank_in_group - 1 + args.group_size) % args.group_size]
    target_ranks = [target]
    target_ips = [all_ips[target]]
    target_ports = [args.base_port + target * 100]
    my_port = args.base_port + args.rank * 100

    native = native_mod.GeminiReplicasNative(
        args.rank,
        args.world_size,
        target_ranks,
        target_ips,
        target_ports,
        my_ip,
        my_port,
        1,
        True,
        args.channels_per_peer,
    )
    if hasattr(native, "set_debug"):
        native.set_debug(bool(args.debug))
    dist.barrier()
    native.finalize_connections()
    dist.barrier()
    native.start_workers([source])

    size = _parse_size(args.bytes)
    if args.cpu_send:
        send_buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        send_addr = int(send_buf.data_ptr())
    else:
        send_buf = torch.empty(size, dtype=torch.uint8, device="cuda")
        send_addr = int(send_buf.data_ptr())
    recv_buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    native.register_buffer(send_addr, size)
    native.register_buffer(int(recv_buf.data_ptr()), size)

    dist.barrier()
    times: List[float] = []
    total_iters = args.warmup + args.iters
    for it in range(total_iters):
        native.reset_exchange_state()
        dist.barrier()
        t0 = time.time()
        native.submit_recv_buffer(source, int(recv_buf.data_ptr()), size)
        native.submit_send_buffer(send_addr, size)
        native.wait_for_exchange_completion()
        if not args.cpu_send:
            torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.time() - t0
        if it >= args.warmup:
            times.append(elapsed)
            iter_minmax = torch.tensor([elapsed, elapsed], dtype=torch.float64)
            dist.all_reduce(iter_minmax[0:1], op=dist.ReduceOp.MIN)
            dist.all_reduce(iter_minmax[1:2], op=dist.ReduceOp.MAX)
            iter_min_s = float(iter_minmax[0].item())
            iter_max_s = float(iter_minmax[1].item())
            print(
                f"ITER_RESULT,mode=gemini2,rank={args.rank},group_id={group_id},"
                f"rank_in_group={rank_in_group},iter={it - args.warmup},"
                f"seconds={elapsed:.6f},iter_min_s={iter_min_s:.6f},"
                f"iter_max_s={iter_max_s:.6f},iter_spread_s={iter_max_s - iter_min_s:.6f},"
                f"bytes_send={size},bytes_recv={size}",
                flush=True,
            )

    local = _summary(times)
    elapsed_tensor = torch.tensor([local["max"]], dtype=torch.float64)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    max_rank_s = float(elapsed_tensor.item())
    local_send_gib_s = (size / (1024 ** 3)) / local["avg"] if local["avg"] > 0 else 0.0
    e2e_send_gib_s = (size / (1024 ** 3)) / max_rank_s if max_rank_s > 0 else 0.0
    e2e_agg_send_gib_s = (size * args.world_size / (1024 ** 3)) / max_rank_s if max_rank_s > 0 else 0.0
    print(
        f"RESULT,mode=gemini2,rank={args.rank},group_id={group_id},rank_in_group={rank_in_group},"
        f"group_size={args.group_size},iters={args.iters},"
        f"bytes_send={size},bytes_recv={size},avg_s={local['avg']:.6f},"
        f"p50_s={local['p50']:.6f},p99_s={local['p99']:.6f},max_s={local['max']:.6f},"
        f"max_rank_s={max_rank_s:.6f},local_send_gib_s={local_send_gib_s:.3f},"
        f"e2e_send_gib_s={e2e_send_gib_s:.3f},e2e_agg_send_gib_s={e2e_agg_send_gib_s:.3f}",
        flush=True,
    )

    native.stop()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
