#!/usr/bin/env python3
"""Test Concord stripe-FIFO encode (multi-lane, batch submit, parallel workers)."""

import importlib.util
import os
import sys

sys.path = [p for p in sys.path if "dist_checkpointing/strategies" not in p]

import torch
import torch.distributed as dist

STRATEGIES = "/workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies"
POA_PATH = os.path.join(STRATEGIES, "poa_n4.txt")
SO_PATH = os.path.join(STRATEGIES, "concord_native.cpython-310-x86_64-linux-gnu.so")

spec = importlib.util.spec_from_file_location("concord_native", SO_PATH)
native_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_mod)


def main():
    if not native_mod.ConcordNative.gdr_available():
        print("SKIP: GDR not available (nvidia-peermem required)", flush=True)
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    n = 4
    assert dist.get_world_size() == n

    torch.cuda.set_device(rank)
    native = native_mod.ConcordNative(POA_PATH)
    native.compile_plans(rank)

    base_port = int(os.environ.get("CONCORD_BASE_PORT", "26050"))
    native.init_rdma(n, rank, base_port, "127.0.0.1", ["127.0.0.1"] * n, True)
    dist.barrier()

    bs = 4096
    recv_total = (n - 2) * bs
    num_stripes = native.num_stripes()

    n_source_my = sum(
        1 for sid in range(num_stripes) if native.get_role_for_stripe(sid) == 0
    )
    layer_capacity = max(n_source_my, 1) * bs

    layer_buf_gpu = torch.zeros(layer_capacity, dtype=torch.uint8, device="cuda")
    layer_mirror = torch.zeros(
        layer_capacity, dtype=torch.uint8, device="cpu", pin_memory=True,
    )
    recv_bufs = [
        torch.zeros(recv_total, dtype=torch.uint8, device="cpu", pin_memory=True)
        for _ in range(num_stripes)
    ]
    p1_bufs = [
        torch.zeros(bs, dtype=torch.uint8, device="cpu", pin_memory=True)
        for _ in range(num_stripes)
    ]
    p2_bufs = [
        torch.zeros(bs, dtype=torch.uint8, device="cpu", pin_memory=True)
        for _ in range(num_stripes)
    ]

    native.register_buffer(layer_buf_gpu.data_ptr(), layer_buf_gpu.numel())
    native.register_buffer(layer_mirror.data_ptr(), layer_mirror.numel())
    for sid in range(num_stripes):
        native.register_buffer(recv_bufs[sid].data_ptr(), recv_bufs[sid].numel())
        native.register_buffer(p1_bufs[sid].data_ptr(), p1_bufs[sid].numel())
        native.register_buffer(p2_bufs[sid].data_ptr(), p2_bufs[sid].numel())

    layer_base = layer_buf_gpu.data_ptr()
    mirror_base = layer_mirror.data_ptr()
    src_block_per_node = 0

    native.reset_encoding_batch()
    for sid in range(num_stripes):
        role = native.get_role_for_stripe(sid)
        source_data = 0
        source_mirror = 0
        recv_buf = 0
        p1 = 0
        p2_out = 0
        p2_in = 0

        if role == 0:
            blk_idx = src_block_per_node
            src_block_per_node += 1
            off = blk_idx * bs
            layer_buf_gpu[off : off + bs].fill_((rank * 100 + sid) % 256)
            source_data = layer_base + off
            source_mirror = mirror_base + off
        elif role == 1:
            recv_bufs[sid].zero_()
            p1_bufs[sid].zero_()
            p2_bufs[sid].zero_()
            recv_buf = recv_bufs[sid].data_ptr()
            p1 = p1_bufs[sid].data_ptr()
            p2_out = p2_bufs[sid].data_ptr()
        elif role == 2:
            p2_bufs[sid].zero_()
            p2_in = p2_bufs[sid].data_ptr()

        native.submit_stripe_chunk(
            sid,
            source_data,
            source_mirror,
            recv_buf,
            p1,
            p2_out,
            p2_in,
            bs,
        )

    native.submit_encoding_sentinel()
    native.wait_encoding_batch()

    sample_p2 = p2_bufs[0][0].item() if p2_bufs else 0
    print(
        f"[Rank {rank}] stripe-FIFO worker test PASSED p2[0]={sample_p2}",
        flush=True,
    )

    dist.barrier()
    native.stop()
    print(f"[Rank {rank}] done", flush=True)


if __name__ == "__main__":
    main()
