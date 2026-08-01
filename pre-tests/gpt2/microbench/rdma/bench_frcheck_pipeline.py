#!/usr/bin/env python3
"""Temporary FRCheck CPU-RDMA, CUDA D2H, and ISA-L overlap microbenchmark."""
import argparse
import os
import threading
import time
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist

from bench_frcheck_network import (
    _all_gather_strings, _build_exchange_plan, _build_tasks, _compile_plans,
    _get_ip, _init_dist, _load_native, _parse_size, _run_exchange,
)

GIB = float(1024 ** 3)
MODES = ("net-only", "pcie-only", "encode-only", "serial", "net+pcie",
         "net+encode", "pcie+encode", "all-overlap")
MODE_STAGES = {
    "net-only": ("net",), "pcie-only": ("pcie",), "encode-only": ("encode",),
    "serial": ("net", "pcie", "encode"), "net+pcie": ("net", "pcie"),
    "net+encode": ("net", "encode"), "pcie+encode": ("pcie", "encode"),
    "all-overlap": ("net", "pcie", "encode"),
}


def csv_line(kind: str, fields: Dict[str, Any]) -> None:
    parts = [f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
             for key, value in fields.items()]
    print(kind + "," + ",".join(parts), flush=True)


def global_max(value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def global_sum(value: int) -> int:
    tensor = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def make_encode_buffers(source_target: int, requested_jobs: int):
    n_src = 2
    jobs = max(1, requested_jobs)
    while (source_target + n_src * jobs - 1) // (n_src * jobs) > 0x7FFFFFFF:
        jobs += 1
    block_size = max(4096, ((source_target + n_src * jobs - 1) // (n_src * jobs) + 4095) & ~4095)
    if block_size > 0x7FFFFFFF:
        raise RuntimeError("Encode block size exceeds signed int maximum")
    source_bytes = n_src * jobs * block_size
    parity_bytes = 2 * jobs * block_size
    source = torch.empty(source_bytes, dtype=torch.uint8, pin_memory=True)
    parity = torch.empty(parity_bytes, dtype=torch.uint8, pin_memory=True)
    source.fill_(0x5A)
    stripe_ids = list(range(jobs))
    data_addrs = [int(source.data_ptr()) + (job * n_src + src) * block_size
                  for job in range(jobs) for src in range(n_src)]
    p1_addrs = [int(parity.data_ptr()) + job * 2 * block_size for job in range(jobs)]
    p2_addrs = [addr + block_size for addr in p1_addrs]
    return (source, parity, stripe_ids, data_addrs, p1_addrs, p2_addrs,
            [block_size] * jobs, source_bytes, parity_bytes, block_size, jobs)


def run_mode(stages: List[str], funcs: Dict[str, Callable[[], float]], serial: bool):
    elapsed = {name: 0.0 for name in ("net", "pcie", "encode")}
    if serial:
        wall_t0 = time.perf_counter()
        for stage in stages:
            elapsed[stage] = funcs[stage]()
        return time.perf_counter() - wall_t0, elapsed
    start = threading.Barrier(len(stages) + 1)
    errors: List[BaseException] = []
    lock = threading.Lock()

    def worker(stage: str) -> None:
        try:
            start.wait()
            elapsed[stage] = funcs[stage]()
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(stage,), name=f"bench-{stage}")
               for stage in stages]
    for thread in threads:
        thread.start()
    wall_t0 = time.perf_counter()
    start.wait()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - wall_t0
    if errors:
        raise RuntimeError("Overlapped workload failed") from errors[0]
    return wall, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary FRCheck pipeline overlap microbenchmark")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "16")))
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "10.252.129.35"))
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29610")))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("FRCHECK_BASE_PORT", "28200")))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "2")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", "5")))
    parser.add_argument("--net-bytes", default=os.environ.get("NET_BYTES", "256M"))
    parser.add_argument("--net-source", choices=("cpu", "gpu"),
                        default=os.environ.get("NET_SOURCE", "cpu").lower())
    parser.add_argument("--share-gpu-source", type=int, choices=(0, 1),
                        default=int(os.environ.get("SHARE_GPU_SOURCE", "1")))
    parser.add_argument("--pcie-bytes", default=os.environ.get("PCIE_BYTES", "256M"))
    parser.add_argument("--encode-source-bytes", default=os.environ.get("ENCODE_SOURCE_BYTES", "256M"))
    parser.add_argument("--encode-jobs", type=int, default=int(os.environ.get("ENCODE_JOBS", "8")))
    parser.add_argument("--lanes-per-peer", type=int, default=int(os.environ.get("FRCHECK_RDMA_LANES_PER_PEER", "4")))
    parser.add_argument("--segments", type=int, default=int(os.environ.get("FRCHECK_LAYER_EXCHANGE_SEG", "4")))
    parser.add_argument("--modes", default=os.environ.get("BENCH_MODES", ",".join(MODES)))
    parser.add_argument("--backend", default="gloo")
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODE_STAGES]
    if unknown:
        raise RuntimeError(f"Unknown benchmark modes: {unknown}")
    if len(set(modes)) != len(modes):
        raise RuntimeError("Benchmark modes must be unique")
    if args.world_size != 16 or args.n != 4:
        raise RuntimeError("This benchmark requires world_size=16 and FRCheck n=4")
    if args.warmup < 0 or args.iters <= 0 or args.lanes_per_peer <= 0:
        raise RuntimeError("Invalid iteration or lane count")

    torch.cuda.set_device(args.local_rank)
    _init_dist(args)
    native = None
    try:
        num_groups = args.world_size // args.n
        group_id = args.rank % num_groups
        rank_in_group = args.rank // num_groups
        group_ranks = [group_id + idx * num_groups for idx in range(args.n)]
        group_base_port = args.base_port + group_id * args.n * 100
        my_node = rank_in_group + 1
        lanes = args.lanes_per_peer
        os.environ["FRCHECK_RDMA_LANES_PER_PEER"] = str(lanes * 2)
        if lanes * 2 < args.n * (args.n - 1):
            os.environ.setdefault("FRCHECK_ALLOW_UNSAFE_LANE_SHARING", "1")
        native = _load_native().FRCheckNative(args.n)
        my_ip = _get_ip(args.rank, args.local_rank)
        all_ips = _all_gather_strings(my_ip)
        native.init_rdma(args.n, rank_in_group, group_base_port, my_ip,
                         [all_ips[rank] for rank in group_ranks], True)

        net_target = _parse_size(args.net_bytes)
        plans = _compile_plans(native, args.n, rank_in_group)
        source_blocks = (args.n - 1) * (args.n - 2)
        n_filled = source_blocks
        send_blocks, recv_blocks, _ = _build_exchange_plan(plans, args.n, my_node, n_filled)
        send_block_count = sum(len(blocks) for blocks in send_blocks.values())
        if send_block_count <= 0:
            raise RuntimeError("POA plan produced no network send blocks")
        net_block = max(4096, ((net_target + send_block_count - 1) // send_block_count + 4095) & ~4095)
        send_sizes = {peer: len(blocks) * net_block for peer, blocks in send_blocks.items()}
        net_send_bytes = sum(send_sizes.values())
        pcie_bytes = _parse_size(args.pcie_bytes)
        shared_gpu_source = args.net_source == "gpu" and bool(args.share_gpu_source)
        gpu_backing = None
        if shared_gpu_source:
            gpu_backing = torch.empty(max(net_send_bytes, pcie_bytes), dtype=torch.uint8, device="cuda")
            send_bufs = {}
            offset = 0
            for peer, size in send_sizes.items():
                send_bufs[peer] = gpu_backing.narrow(0, offset, size)
                offset += size
            gpu_source = gpu_backing.narrow(0, 0, pcie_bytes)
        elif args.net_source == "gpu":
            send_bufs = {peer: torch.empty(size, dtype=torch.uint8, device="cuda")
                         for peer, size in send_sizes.items()}
            gpu_source = torch.empty(pcie_bytes, dtype=torch.uint8, device="cuda")
        else:
            send_bufs = {peer: torch.empty(size, dtype=torch.uint8, pin_memory=True)
                         for peer, size in send_sizes.items()}
            gpu_source = torch.empty(pcie_bytes, dtype=torch.uint8, device="cuda")
        recv_bufs = {peer: torch.empty(len(blocks) * net_block, dtype=torch.uint8, pin_memory=True)
                     for peer, blocks in recv_blocks.items()}
        for buf in list(send_bufs.values()) + list(recv_bufs.values()):
            native.register_buffer(int(buf.data_ptr()), int(buf.numel()))
        net_recv_bytes = sum(buf.numel() for buf in recv_bufs.values())

        cpu_destination = torch.empty(pcie_bytes, dtype=torch.uint8, pin_memory=True)
        pcie_stream = torch.cuda.Stream(device=args.local_rank)
        pcie_start_event = torch.cuda.Event(enable_timing=True)
        pcie_end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        encode_values = make_encode_buffers(_parse_size(args.encode_source_bytes), args.encode_jobs)
        (encode_source, encode_parity, stripe_ids, data_addrs, p1_addrs, p2_addrs,
         block_sizes, encode_source_bytes, encode_parity_bytes, encode_block, encode_jobs) = encode_values
        dist.barrier()
        global_send_bytes = global_sum(net_send_bytes)
        summaries: Dict[str, Dict[str, float]] = {}
        for mode_idx, mode in enumerate(modes):
            measured = {name: [] for name in ("wall", "net", "pcie", "encode")}
            for iteration in range(args.warmup + args.iters):
                batch_base = (mode_idx + 1) * 1000000009 + (iteration + 1) * 1000003 + group_id * 10007
                send_tasks, recv_tasks = _build_tasks(
                    send_blocks, recv_blocks, send_bufs, recv_bufs, net_block, args.segments,
                    lanes, lanes, batch_base, my_node)

                def run_net() -> float:
                    if args.net_source == "cpu":
                        return _run_exchange(native, send_tasks, recv_tasks)
                    t0 = time.perf_counter()
                    _run_exchange(native, send_tasks, recv_tasks)
                    torch.cuda.synchronize()
                    return time.perf_counter() - t0

                def run_pcie() -> float:
                    with torch.cuda.stream(pcie_stream):
                        pcie_start_event.record(pcie_stream)
                        cpu_destination.copy_(gpu_source, non_blocking=True)
                        pcie_end_event.record(pcie_stream)
                    pcie_end_event.synchronize()
                    return pcie_start_event.elapsed_time(pcie_end_event) / 1000.0

                def run_encode() -> float:
                    t0 = time.perf_counter()
                    native.encode_layer_stripes_batch(stripe_ids, data_addrs, p1_addrs, p2_addrs, block_sizes)
                    return time.perf_counter() - t0

                funcs = {"net": run_net, "pcie": run_pcie, "encode": run_encode}
                dist.barrier()
                wall, stage_elapsed = run_mode(list(MODE_STAGES[mode]), funcs, mode == "serial")
                dist.barrier()
                global_values = {"wall": global_max(wall)}
                global_values.update({stage: global_max(stage_elapsed[stage])
                                      for stage in ("net", "pcie", "encode")})
                if iteration >= args.warmup:
                    for key, value in global_values.items():
                        measured[key].append(value)
                    csv_line("ITER_RESULT", {
                        "mode": mode, "rank": args.rank, "group_id": group_id,
                        "rank_in_group": rank_in_group, "iter": iteration - args.warmup,
                        "wall_s": wall, "global_max_wall_s": global_values["wall"],
                        "net_s": stage_elapsed["net"], "global_max_net_s": global_values["net"],
                        "pcie_s": stage_elapsed["pcie"], "global_max_pcie_s": global_values["pcie"],
                        "encode_s": stage_elapsed["encode"], "global_max_encode_s": global_values["encode"],
                        "net_send_bytes": net_send_bytes, "net_recv_bytes": net_recv_bytes,
                        "pcie_d2h_bytes": pcie_bytes, "encode_source_bytes": encode_source_bytes,
                        "encode_parity_bytes": encode_parity_bytes,
                    })
            summaries[mode] = {key: sum(values) / len(values) for key, values in measured.items()}
            summaries[mode].update({f"peak_{key}": max(values) for key, values in measured.items()})

        required_only = {"net-only", "pcie-only", "encode-only"}
        if not required_only.issubset(summaries):
            raise RuntimeError("Final slowdown metrics require all three only modes")
        only = {
            "net": summaries["net-only"]["net"],
            "pcie": summaries["pcie-only"]["pcie"],
            "encode": summaries["encode-only"]["encode"],
        }
        serial_sum = sum(only.values())
        for mode in modes:
            result = summaries[mode]
            denominator = serial_sum - max(only.values())
            efficiency = (serial_sum - result["wall"]) / denominator if denominator > 0 else 0.0
            if args.rank == 0:
                csv_line("RESULT", {
                    "mode": mode, "rank": args.rank, "iters": args.iters, "frcheck_n": args.n,
                    "net_source": args.net_source,
                    "share_gpu_source": args.share_gpu_source,
                    "global_max_wall_s": result["peak_wall"],
                    "global_max_net_s": result["peak_net"],
                    "global_max_pcie_s": result["peak_pcie"],
                    "global_max_encode_s": result["peak_encode"],
                    "avg_global_max_wall_s": result["wall"],
                    "avg_global_max_net_s": result["net"],
                    "avg_global_max_pcie_s": result["pcie"],
                    "avg_global_max_encode_s": result["encode"],
                    "slowdown_net": result["net"] / only["net"] if only["net"] and result["net"] else 0.0,
                    "slowdown_pcie": result["pcie"] / only["pcie"] if only["pcie"] and result["pcie"] else 0.0,
                    "slowdown_encode": result["encode"] / only["encode"] if only["encode"] and result["encode"] else 0.0,
                    "serial_sum_s": serial_sum, "overlap_efficiency": efficiency,
                    "net_send_bytes_per_rank": net_send_bytes, "net_recv_bytes_per_rank": net_recv_bytes,
                    "net_aggregate_send_gib_s": (global_send_bytes / GIB) / result["net"] if result["net"] else 0.0,
                    "net_per_rank_send_gib_s": (net_send_bytes / GIB) / result["net"] if result["net"] else 0.0,
                    "pcie_d2h_bytes_per_rank": pcie_bytes,
                    "pcie_per_rank_gib_s": (pcie_bytes / GIB) / result["pcie"] if result["pcie"] else 0.0,
                    "encode_source_bytes_per_rank": encode_source_bytes,
                    "encode_parity_bytes_per_rank": encode_parity_bytes,
                    "encode_block_size": encode_block, "encode_jobs": encode_jobs,
                    "encode_source_gib_s": (encode_source_bytes / GIB) / result["encode"] if result["encode"] else 0.0,
                    "encode_parity_gib_s": (encode_parity_bytes / GIB) / result["encode"] if result["encode"] else 0.0,
                    "encode_memory_gib_s": ((encode_source_bytes + encode_parity_bytes) / GIB) / result["encode"] if result["encode"] else 0.0,
                })
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            finally:
                if native is not None:
                    native.stop()
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
