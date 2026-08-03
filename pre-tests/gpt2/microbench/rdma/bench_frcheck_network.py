#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""FRCheck layer-exchange network-only microbenchmark.

This benchmark reuses FRCheckNative RDMA layer send/recv APIs and constructs a
synthetic POA layer exchange. It does not run ISA-L encode or checkpoint I/O.
"""

import argparse
import glob
import importlib.util
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[4]
STRATEGY_DIR = ROOT / "megatron" / "core" / "dist_checkpointing" / "strategies"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from megatron.core.dist_checkpointing.strategies.network_utils import resolve_ip


class StripeRole:
    SOURCE = 0
    ENCODER = 1
    PARITY_TARGET = 2


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


def _load_native():
    matches = glob.glob(str(STRATEGY_DIR / "frcheck_native*.so"))
    if not matches:
        raise RuntimeError(f"No frcheck_native*.so in {STRATEGY_DIR}")
    spec = importlib.util.spec_from_file_location("frcheck_native", matches[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load native module {matches[0]}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_ip(rank: int, local_rank: int) -> str:
    return resolve_ip("FRCHECK", rank=rank, local_rank=local_rank, fallback_prefixes=["ECLATIN"])


def _init_dist(args) -> None:
    if dist.is_initialized():
        return
    os.environ["MASTER_ADDR"] = str(args.master_addr)
    os.environ["MASTER_PORT"] = str(args.master_port)
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
    return {
        "min": min(ordered) if ordered else 0.0,
        "p50": statistics.median(ordered) if ordered else 0.0,
        "p99": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] if ordered else 0.0,
        "max": max(ordered) if ordered else 0.0,
        "avg": sum(ordered) / len(ordered) if ordered else 0.0,
    }


def _compile_plans(native, n: int, rank_in_group: int) -> List[Dict[str, Any]]:
    native.compile_plans(rank_in_group)
    plans: List[Dict[str, Any]] = []
    for sid in range(native.num_stripes()):
        plans.append({
            "stripe_id": sid,
            "role": int(native.get_role_for_stripe(sid)),
            "source_node_ids": [int(x) for x in native.get_source_node_ids(sid)],
            "encoder_node_id": int(native.get_encoder_node_id(sid)),
            "parity_target_node_id": int(native.get_parity_target_node_id(sid)),
        })
    return plans


def _build_exchange_plan(plans: List[Dict[str, Any]], n: int, my_node: int, n_filled: int):
    send_blocks: Dict[int, List[Tuple[int, int]]] = {}
    recv_blocks: Dict[int, List[Tuple[int, int]]] = {}
    encoder_sids: List[int] = []
    src_block_per_node: Dict[int, int] = {node: 0 for node in range(1, n + 1)}
    for plan in plans:
        sid = int(plan["stripe_id"])
        block_indices = {src: src_block_per_node.get(src, 0) for src in plan["source_node_ids"]}
        if my_node in plan["source_node_ids"] and int(plan["encoder_node_id"]) != my_node:
            blk_idx = block_indices[my_node]
            if blk_idx < n_filled:
                send_blocks.setdefault(int(plan["encoder_node_id"]), []).append((sid, blk_idx))
        if int(plan["role"]) == StripeRole.ENCODER:
            encoder_sids.append(sid)
            for src_node in plan["source_node_ids"]:
                if src_node == my_node:
                    continue
                blk_idx = block_indices[src_node]
                if blk_idx < n_filled:
                    recv_blocks.setdefault(int(src_node), []).append((sid, blk_idx))
        for src_node in plan["source_node_ids"]:
            src_block_per_node[src_node] = src_block_per_node.get(src_node, 0) + 1
    return send_blocks, recv_blocks, encoder_sids


def _iter_segments(block_count: int, seg_count: int):
    if block_count <= 0:
        return
    nseg = min(max(1, int(seg_count)), block_count)
    base = block_count // nseg
    rem = block_count % nseg
    start = 0
    for seg_idx in range(nseg):
        take = base + (1 if seg_idx < rem else 0)
        if take > 0:
            yield seg_idx, start, take
        start += take


def _directed_lane_id(src_node: int, dst_node: int, seg_idx: int, forward_lanes: int, reverse_lanes: int) -> int:
    lanes = forward_lanes if src_node < dst_node else reverse_lanes
    return seg_idx % max(1, lanes)


def _directed_lane_count(src_node: int, dst_node: int, forward_lanes: int, reverse_lanes: int) -> int:
    return max(1, forward_lanes if src_node < dst_node else reverse_lanes)


def _build_tasks(
    send_blocks,
    recv_blocks,
    send_bufs,
    recv_bufs,
    block_size,
    seg_count,
    forward_lanes,
    reverse_lanes,
    batch_base,
    my_node,
):
    send_tasks = []
    recv_tasks = []
    for src_node, blocks in sorted(recv_blocks.items()):
        recv_base = int(recv_bufs[src_node].data_ptr())
        for seg_idx, start, take in _iter_segments(len(blocks), seg_count):
            lane_id = _directed_lane_id(src_node, my_node, seg_idx, forward_lanes, reverse_lanes)
            recv_tasks.append({
                "peer_node": src_node,
                "peer_rig": src_node - 1,
                "addr": recv_base + start * block_size,
                "size": take * block_size,
                "batch_id": batch_base + src_node * 1009 + my_node + seg_idx * 104729,
                "lane_id": lane_id,
            })
    for dst_node, blocks in sorted(send_blocks.items()):
        send_base = int(send_bufs[dst_node].data_ptr())
        for seg_idx, start, take in _iter_segments(len(blocks), seg_count):
            lane_id = _directed_lane_id(my_node, dst_node, seg_idx, forward_lanes, reverse_lanes)
            send_tasks.append({
                "peer_node": dst_node,
                "peer_rig": dst_node - 1,
                "addr": send_base + start * block_size,
                "size": take * block_size,
                "batch_id": batch_base + my_node * 1009 + dst_node + seg_idx * 104729,
                "lane_id": lane_id,
            })
    return send_tasks, recv_tasks


def _build_simple_tasks(
    send_buf,
    recv_buf,
    total_size: int,
    n: int,
    my_node: int,
    batch_base: int,
    forward_lanes: int,
    reverse_lanes: int,
):
    send_tasks = []
    recv_tasks = []
    peer_count = n - 1
    if peer_count <= 0:
        return send_tasks, recv_tasks

    def peer_index(src_node: int, dst_node: int) -> int:
        peers = [node for node in range(1, n + 1) if node != src_node]
        return peers.index(dst_node)

    def shard_bounds(src_node: int, dst_node: int):
        idx = peer_index(src_node, dst_node)
        start = (total_size * idx) // peer_count
        end = (total_size * (idx + 1)) // peer_count
        return start, end, end - start

    def lane_bounds(start: int, size: int, lane_idx: int, lane_count: int):
        seg_start = start + (size * lane_idx) // lane_count
        seg_end = start + (size * (lane_idx + 1)) // lane_count
        return seg_start, seg_end, seg_end - seg_start

    def edge_batch(src_node: int, dst_node: int, lane_idx: int) -> int:
        return batch_base + src_node * 1009 + dst_node + lane_idx * 104729

    for peer_node in (node for node in range(1, n + 1) if node != my_node):
        send_start, _send_end, send_size = shard_bounds(my_node, peer_node)
        recv_start, _recv_end, recv_size = shard_bounds(peer_node, my_node)
        send_lane_count = _directed_lane_count(my_node, peer_node, forward_lanes, reverse_lanes)
        recv_lane_count = _directed_lane_count(peer_node, my_node, forward_lanes, reverse_lanes)
        for lane_idx in range(send_lane_count):
            send_seg_start, _send_seg_end, send_seg_size = lane_bounds(
                send_start, send_size, lane_idx, send_lane_count
            )
            if send_seg_size > 0:
                send_tasks.append({
                    "peer_node": peer_node,
                    "peer_rig": peer_node - 1,
                    "addr": int(send_buf.data_ptr()) + send_seg_start,
                    "size": send_seg_size,
                    "batch_id": edge_batch(my_node, peer_node, lane_idx),
                    "lane_id": _directed_lane_id(my_node, peer_node, lane_idx, forward_lanes, reverse_lanes),
                })
        for lane_idx in range(recv_lane_count):
            recv_seg_start, _recv_seg_end, recv_seg_size = lane_bounds(
                recv_start, recv_size, lane_idx, recv_lane_count
            )
            if recv_seg_size > 0:
                recv_tasks.append({
                    "peer_node": peer_node,
                    "peer_rig": peer_node - 1,
                    "addr": int(recv_buf.data_ptr()) + recv_seg_start,
                    "size": recv_seg_size,
                    "batch_id": edge_batch(peer_node, my_node, lane_idx),
                    "lane_id": _directed_lane_id(peer_node, my_node, lane_idx, forward_lanes, reverse_lanes),
                })
    return send_tasks, recv_tasks


def _run_exchange(native, send_tasks, recv_tasks) -> float:
    errors: List[BaseException] = []
    lock = threading.Lock()

    def record(exc: BaseException) -> None:
        with lock:
            errors.append(exc)

    def send_worker(task):
        try:
            native.send_layer_to_peer(
                int(task["peer_rig"]), int(task["addr"]), int(task["size"]),
                int(task["batch_id"]), int(task["lane_id"]),
            )
        except BaseException as exc:
            record(exc)

    def recv_worker(task):
        try:
            native.recv_layer_from_peer(
                int(task["peer_rig"]), int(task["addr"]), int(task["size"]),
                int(task["batch_id"]), int(task["lane_id"]),
            )
        except BaseException as exc:
            record(exc)

    threads: List[threading.Thread] = []
    t0 = time.time()
    for task in recv_tasks:
        th = threading.Thread(target=recv_worker, args=(task,))
        th.start()
        threads.append(th)
    for task in send_tasks:
        th = threading.Thread(target=send_worker, args=(task,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    elapsed = time.time() - t0
    if errors:
        raise RuntimeError("FRCheck bench exchange failed") from errors[0]
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="FRCheck layer-exchange network-only microbenchmark")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "8")))
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "6000")))
    parser.add_argument("--base-port", type=int, default=int(os.environ.get("FRCHECK_BASE_PORT", "27200")))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--bytes", default="1G")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--lanes-per-peer", type=int, default=int(os.environ.get("FRCHECK_RDMA_LANES_PER_PEER", "8")))
    parser.add_argument("--send-lanes-per-peer", type=int, default=int(os.environ.get("FRCHECK_SEND_LANES_PER_PEER", "0")))
    parser.add_argument("--recv-lanes-per-peer", type=int, default=int(os.environ.get("FRCHECK_RECV_LANES_PER_PEER", "0")))
    parser.add_argument("--segments", type=int, default=int(os.environ.get("FRCHECK_LAYER_EXCHANGE_SEG", "4")))
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--cpu-send", action="store_true")
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.world_size % args.n != 0:
        raise RuntimeError("FRCheck bench requires world_size to be divisible by n")
    if args.n < 2:
        raise RuntimeError("FRCheck bench requires n >= 2")
    if not args.simple and args.n < 3:
        raise RuntimeError("FRCheck normal mode requires n >= 3; use --simple for n=2")
    num_groups = args.world_size // args.n
    group_id = args.rank % num_groups
    rank_in_group = args.rank // num_groups
    group_ranks = [group_id + idx * num_groups for idx in range(args.n)]
    group_base_port = args.base_port + group_id * (args.n * 100)
    if group_base_port > 65000:
        raise RuntimeError(
            f"FRCheck RDMA group_base_port={group_base_port} exceeds uint16 limit; lower --base-port"
        )
    forward_lanes = args.send_lanes_per_peer if args.send_lanes_per_peer > 0 else args.lanes_per_peer
    reverse_lanes = args.recv_lanes_per_peer if args.recv_lanes_per_peer > 0 else args.lanes_per_peer
    if forward_lanes <= 0 or reverse_lanes <= 0:
        raise RuntimeError("FRCheck split lanes require positive send and recv lane counts")
    total_lanes = forward_lanes + reverse_lanes
    print(f"PROGRESS,rank={args.rank},stage=set_device_start", flush=True)
    cuda_device = 0 if torch.cuda.device_count() == 1 else args.local_rank
    torch.cuda.set_device(cuda_device)
    print(f"PROGRESS,rank={args.rank},stage=init_dist_start,backend={args.backend}", flush=True)
    _init_dist(args)
    print(f"PROGRESS,rank={args.rank},stage=init_dist_done", flush=True)
    os.environ["FRCHECK_RDMA_LANES_PER_PEER"] = str(total_lanes)
    if total_lanes < args.n * (args.n - 1):
        os.environ.setdefault("FRCHECK_ALLOW_UNSAFE_LANE_SHARING", "1")

    print(f"PROGRESS,rank={args.rank},stage=load_native_start", flush=True)
    native_mod = _load_native()
    print(f"PROGRESS,rank={args.rank},stage=load_native_done", flush=True)
    native = native_mod.FRCheckNative(args.n)
    if hasattr(native, "set_debug"):
        native.set_debug(bool(args.debug))
    my_ip = _get_ip(args.rank, args.local_rank)
    print(f"PROGRESS,rank={args.rank},stage=all_gather_ip_start,ip={my_ip}", flush=True)
    all_ips = _all_gather_strings(my_ip)
    print(f"PROGRESS,rank={args.rank},stage=all_gather_ip_done", flush=True)
    print(f"PROGRESS,rank={args.rank},stage=init_rdma_start", flush=True)
    native.init_rdma(
        args.n,
        rank_in_group,
        group_base_port,
        my_ip,
        [all_ips[group_rank] for group_rank in group_ranks],
        True,
    )
    print(f"PROGRESS,rank={args.rank},stage=init_rdma_done", flush=True)
    payload_bytes = _parse_size(args.bytes)
    my_node = rank_in_group + 1
    block_size = 0
    n_filled = 0
    simple_send_buf = None
    simple_recv_buf = None
    send_blocks = {}
    recv_blocks = {}
    send_bufs = {}
    recv_bufs = {}
    registered = []

    if args.simple:
        print(f"PROGRESS,rank={args.rank},stage=simple_buffer_alloc_start", flush=True)
        send_bytes = payload_bytes
        recv_bytes = payload_bytes
        if args.cpu_send:
            simple_send_buf = torch.empty(send_bytes, dtype=torch.uint8, pin_memory=True)
        else:
            simple_send_buf = torch.empty(send_bytes, dtype=torch.uint8, device="cuda")
        simple_recv_buf = torch.empty(recv_bytes, dtype=torch.uint8, pin_memory=True)
        native.register_buffer(int(simple_send_buf.data_ptr()), int(simple_send_buf.numel()))
        native.register_buffer(int(simple_recv_buf.data_ptr()), int(simple_recv_buf.numel()))
        registered.extend([simple_send_buf, simple_recv_buf])
        print(f"PROGRESS,rank={args.rank},stage=simple_buffer_alloc_done", flush=True)
    else:
        plans = _compile_plans(native, args.n, rank_in_group)
        n_src = (args.n - 1) * (args.n - 2)
        block_size = max(4096, ((payload_bytes + n_src - 1) // n_src + 4095) & ~4095)
        n_filled = min((payload_bytes + block_size - 1) // block_size, n_src)
        send_blocks, recv_blocks, _encoder_sids = _build_exchange_plan(plans, args.n, my_node, n_filled)
        for dst_node, blocks in send_blocks.items():
            if args.cpu_send:
                buf = torch.empty(len(blocks) * block_size, dtype=torch.uint8, pin_memory=True)
            else:
                buf = torch.empty(len(blocks) * block_size, dtype=torch.uint8, device="cuda")
            native.register_buffer(int(buf.data_ptr()), int(buf.numel()))
            registered.append(buf)
            send_bufs[dst_node] = buf
        for src_node, blocks in recv_blocks.items():
            buf = torch.empty(len(blocks) * block_size, dtype=torch.uint8, pin_memory=True)
            native.register_buffer(int(buf.data_ptr()), int(buf.numel()))
            registered.append(buf)
            recv_bufs[src_node] = buf
        send_bytes = sum(len(v) * block_size for v in send_blocks.values())
        recv_bytes = sum(len(v) * block_size for v in recv_blocks.values())
    print(f"PROGRESS,rank={args.rank},stage=pre_loop_barrier_start", flush=True)
    dist.barrier()
    print(f"PROGRESS,rank={args.rank},stage=pre_loop_barrier_done", flush=True)
    times: List[float] = []
    total_iters = args.warmup + args.iters
    for it in range(total_iters):
        batch_base = (group_id + 1) * 1000000009 + (it + 1) * 1000003
        if args.simple and args.n == 2 and hasattr(native, "simple_exchange_with_peer"):
            peer_node = 2 if my_node == 1 else 1
            send_tasks = []
            recv_tasks = []
        elif args.simple:
            send_tasks, recv_tasks = _build_simple_tasks(
                simple_send_buf, simple_recv_buf, payload_bytes, args.n, my_node, batch_base, forward_lanes, reverse_lanes
            )
        else:
            send_tasks, recv_tasks = _build_tasks(
                send_blocks, recv_blocks, send_bufs, recv_bufs,
                block_size, args.segments, forward_lanes, reverse_lanes, batch_base, my_node,
            )
        if it == 0:
            print(f"PROGRESS,rank={args.rank},stage=iter0_tasks_built,send_tasks={len(send_tasks)},recv_tasks={len(recv_tasks)}", flush=True)
        dist.barrier()
        if it == 0:
            print(f"PROGRESS,rank={args.rank},stage=iter0_exchange_start", flush=True)
        if args.simple and args.n == 2 and hasattr(native, "simple_exchange_with_peer"):
            elapsed = float(native.simple_exchange_with_peer(
                peer_node - 1,
                int(simple_send_buf.data_ptr()),
                int(simple_recv_buf.data_ptr()),
                int(payload_bytes),
                int(forward_lanes),
                int(batch_base + my_node * 1009 + peer_node),
                int(batch_base + peer_node * 1009 + my_node),
            ))
        else:
            elapsed = _run_exchange(native, send_tasks, recv_tasks)
        if not args.cpu_send:
            torch.cuda.synchronize()
        if it == 0:
            print(f"PROGRESS,rank={args.rank},stage=iter0_exchange_done", flush=True)
        dist.barrier()
        if it >= args.warmup:
            times.append(elapsed)
            iter_minmax = torch.tensor([elapsed, elapsed], dtype=torch.float64)
            dist.all_reduce(iter_minmax[0:1], op=dist.ReduceOp.MIN)
            dist.all_reduce(iter_minmax[1:2], op=dist.ReduceOp.MAX)
            iter_min_s = float(iter_minmax[0].item())
            iter_max_s = float(iter_minmax[1].item())
            print(
                f"ITER_RESULT,mode=frcheck,rank={args.rank},group_id={group_id},"
                f"rank_in_group={rank_in_group},iter={it - args.warmup},"
                f"seconds={elapsed:.6f},iter_min_s={iter_min_s:.6f},"
                f"iter_max_s={iter_max_s:.6f},iter_spread_s={iter_max_s - iter_min_s:.6f},"
                f"bytes_send={send_bytes},bytes_recv={recv_bytes},"
                f"send_tasks={len(send_tasks)},recv_tasks={len(recv_tasks)},"
                f"send_lanes_per_peer={forward_lanes},recv_lanes_per_peer={reverse_lanes}",
                flush=True,
            )

    local = _summary(times)
    max_tensor = torch.tensor([local["max"]], dtype=torch.float64)
    send_tensor = torch.tensor([float(send_bytes)], dtype=torch.float64)
    recv_tensor = torch.tensor([float(recv_bytes)], dtype=torch.float64)
    dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(send_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(recv_tensor, op=dist.ReduceOp.SUM)
    max_rank_s = float(max_tensor.item())
    local_agg_send_gib_s = (float(send_tensor.item()) / (1024 ** 3)) / local["avg"] if local["avg"] > 0 else 0.0
    e2e_agg_send_gib_s = (float(send_tensor.item()) / (1024 ** 3)) / max_rank_s if max_rank_s > 0 else 0.0
    print(
        f"RESULT,mode=frcheck,rank={args.rank},group_id={group_id},rank_in_group={rank_in_group},"
        f"frcheck_n={args.n},simple={int(args.simple)},cpu_send={int(args.cpu_send)},iters={args.iters},"
        f"bytes_send={send_bytes},bytes_recv={recv_bytes},send_tasks={len(send_tasks)},"
        f"recv_tasks={len(recv_tasks)},segments={args.segments},"
        f"lanes_per_peer={total_lanes},send_lanes_per_peer={forward_lanes},"
        f"recv_lanes_per_peer={reverse_lanes},block_size={block_size},n_filled={n_filled},"
        f"avg_s={local['avg']:.6f},p50_s={local['p50']:.6f},p99_s={local['p99']:.6f},"
        f"max_s={local['max']:.6f},max_rank_s={max_rank_s:.6f},"
        f"local_agg_send_gib_s={local_agg_send_gib_s:.3f},e2e_agg_send_gib_s={e2e_agg_send_gib_s:.3f}",
        flush=True,
    )
    native.stop()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
