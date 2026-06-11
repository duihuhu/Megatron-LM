#!/usr/bin/env python3
"""Test FRCheck lockstep worker pipeline (submit_source_send + wait_stripe)."""

import importlib.util
import os
import sys

sys.path = [p for p in sys.path if "dist_checkpointing/strategies" not in p]

import torch
import torch.distributed as dist

STRATEGIES = "/workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies"
POA_PATH = os.path.join(STRATEGIES, "poa_n4.txt")
SO_PATH = os.path.join(STRATEGIES, "frcheck_native.cpython-310-x86_64-linux-gnu.so")

spec = importlib.util.spec_from_file_location("frcheck_native", SO_PATH)
native_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_mod)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    n = 4
    assert dist.get_world_size() == n

    torch.cuda.set_device(rank)
    native = native_mod.FRCheckNative(POA_PATH)
    native.compile_plans(rank)

    base_port = int(os.environ.get("FRCHECK_BASE_PORT", "26050"))
    native.init_rdma(n, rank, base_port, "127.0.0.1", ["127.0.0.1"] * n, True)

    bs = 4096
    recv_total = (n - 2) * bs
    data = torch.zeros(bs, dtype=torch.uint8, device="cpu", pin_memory=True)
    mirror = torch.zeros(bs, dtype=torch.uint8, device="cpu", pin_memory=True)
    recv = torch.zeros(recv_total, dtype=torch.uint8, device="cpu", pin_memory=True)
    p1 = torch.zeros(bs, dtype=torch.uint8, device="cpu", pin_memory=True)
    p2 = torch.zeros(bs, dtype=torch.uint8, device="cpu", pin_memory=True)

    native.register_buffer(data.data_ptr(), data.numel())
    native.register_buffer(recv.data_ptr(), recv.numel())
    native.register_buffer(p1.data_ptr(), p1.numel())
    native.register_buffer(p2.data_ptr(), p2.numel())

    native.reset_encoding_completion()
    for sid in range(native.num_stripes()):
        dist.barrier()
        role = native.get_role_for_stripe(sid)
        if role == 0:
            data.fill_((rank * 100 + sid) % 256)
            native.submit_source_send(sid, data.data_ptr(), 0, bs)
        elif role == 1:
            recv.zero_()
            p1.zero_()
            p2.zero_()
            srcs = native.get_source_node_ids(sid)
            local = [0, 0, 0, 0]
            for i, node in enumerate(srcs):
                if node - 1 == rank:
                    slot = i * bs
                    recv[slot : slot + bs].fill_((rank * 100 + sid) % 256)
                    local[i] = recv.data_ptr() + slot
            native.submit_encoder_encode(
                sid, recv.data_ptr(), p1.data_ptr(), p2.data_ptr(), bs, local
            )
        elif role == 2:
            p2.zero_()
            native.submit_parity_recv(sid, p2.data_ptr(), bs)
        native.wait_stripe(sid)
        dist.barrier()

    print(
        f"[Rank {rank}] lockstep worker test PASSED "
        f"p2[0]={p2[0].item()}"
    )
    native.stop()
    print(f"[Rank {rank}] done", flush=True)


if __name__ == "__main__":
    main()
