#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Launch rdma_ec_bind_bench under torch.distributed (2 ranks, 2 nodes or 1 node).

Uses the same IP resolution as EC/Gemini checkpoint code (network_utils.resolve_ip).
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Repo root on PYTHONPATH when invoked via torchrun from Megatron-LM root.
from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip


def _gather_ips(my_ip: str, world_size: int, use_cuda: bool) -> list[str]:
    ip_bytes = socket.inet_aton(my_ip)
    tensor = torch.tensor([int(b) for b in ip_bytes], dtype=torch.uint8)
    if use_cuda:
        tensor = tensor.cuda()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    ips = []
    for t in gathered:
        raw = bytes(t.cpu().tolist())
        ips.append(socket.inet_ntoa(raw))
    return ips


def main() -> int:
    parser = argparse.ArgumentParser(description="RDMA EC-bind microbench launcher")
    parser.add_argument(
        "--prefix",
        default=os.environ.get("MICROBENCH_PREFIX", "ECNAIVE"),
        help="Env prefix for resolve_ip (ECNAIVE, GEMINI_REPLICAS, ...)",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("MICROBENCH_PORT", "19987")))
    parser.add_argument("--size-mb", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--skip-send", action="store_true", help="Skip send-only phase")
    parser.add_argument("--skip-recv", action="store_true", help="Skip recv-only phase")
    parser.add_argument("--skip-duplex", action="store_true", help="Skip concurrent duplex phase")
    parser.add_argument(
        "--binary",
        default=None,
        help="Path to rdma_ec_bind_bench (default: microbench/rdma/build/)",
    )
    parser.add_argument(
        "--dist-backend",
        choices=("auto", "gloo", "nccl"),
        default=os.environ.get("MICROBENCH_DIST_BACKEND", "auto"),
        help="torch.distributed backend (auto: nccl if CUDA else gloo)",
    )
    args, _unknown = parser.parse_known_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.dist_backend == "gloo":
        backend = "gloo"
    elif args.dist_backend == "nccl":
        backend = "nccl"
    else:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if backend == "nccl":
        if not torch.cuda.is_available():
            print("NCCL backend requested but CUDA is not available", file=sys.stderr)
            return 1
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        if rank == 0:
            print("rdma_ec_bind_bench requires world_size=2", file=sys.stderr)
        dist.destroy_process_group()
        return 1

    my_ip = resolve_ip(args.prefix, rank=rank, local_rank=local_rank)
    ips = _gather_ips(my_ip, world_size, use_cuda=(backend == "nccl"))
    peer_rank = 1 - rank
    peer_ip = ips[peer_rank]

    if rank == 0:
        gpu_info = (
            f"cuda={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES', '(all)')}"
            if torch.cuda.is_available()
            else "cuda=unavailable"
        )
        print(
            f"[launcher] prefix={args.prefix} backend={backend} {gpu_info} ips={ips}"
        )
        if ips[0] == ips[1]:
            print(
                "[launcher] WARNING: both ranks resolved to the same IP. "
                "For single-node RDMA bench, set per-local_rank NICs, e.g.\n"
                f"  export {args.prefix}_LOCAL_RANK_NIC_0=eth0\n"
                f"  export {args.prefix}_LOCAL_RANK_NIC_1=eth1\n"
                "or use two physical nodes with MASTER_ADDR pointing to the peer."
            )
    if backend == "nccl":
        print(
            f"[launcher] rank={rank} local_rank={local_rank} "
            f"gpu={torch.cuda.current_device()} name={torch.cuda.get_device_name(local_rank)}"
        )

    bench_dir = Path(__file__).resolve().parent
    binary = Path(args.binary) if args.binary else bench_dir / "build" / "rdma_ec_bind_bench"
    if not binary.is_file():
        if rank == 0:
            print(
                f"Binary not found: {binary}\n"
                "Build with: cd microbench/rdma && make rdma_ec_bind_bench",
                file=sys.stderr,
            )
        dist.destroy_process_group()
        return 1

    cmd = [
        str(binary),
        "--bind-ip",
        my_ip,
        "--peer-ip",
        peer_ip,
        "--rank",
        str(rank),
        "--peer-rank",
        str(peer_rank),
        "--port",
        str(args.port),
        "--size-mb",
        str(args.size_mb),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    if args.skip_send:
        cmd.append("--skip-send")
    if args.skip_recv:
        cmd.append("--skip-recv")
    if args.skip_duplex:
        cmd.append("--skip-duplex")

    dist.barrier()
    if rank == 0:
        print(f"[launcher] starting bench: port={args.port} size_mb={args.size_mb}")

    result = subprocess.run(cmd, check=False)
    dist.barrier()

    if rank == 0:
        print(f"[launcher] done, exit_code={result.returncode}")

    dist.destroy_process_group()
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
