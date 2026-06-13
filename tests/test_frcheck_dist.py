#!/usr/bin/env python3
"""
Distributed FRCheck RDMA test (torchrun).

Usage:
  torchrun --nproc_per_node=4 tests/test_frcheck_dist.py

Tests:
  - RDMA full-mesh connection establishment
  - submit_stripe_chunk pipeline (SOURCE->ENCODER->PARITY_TARGET)
  - Data integrity across the encode pipeline
"""

import sys
import os

sys.path = [p for p in sys.path if 'dist_checkpointing/strategies' not in p]

import torch
import torch.distributed as dist
import importlib.util

STRATEGIES = '/workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies'
POA_PATH = os.path.join(STRATEGIES, 'poa_n4.txt')
SO_PATH = os.path.join(STRATEGIES, 'frcheck_native.cpython-310-x86_64-linux-gnu.so')

spec = importlib.util.spec_from_file_location('frcheck_native', SO_PATH)
native_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_mod)


def init_distributed():
    """Initialize torch.distributed."""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def frcheck_dist_test():
    """Test FRCheck RDMA pipeline with 4 processes."""
    if not native_mod.FRCheckNative.gdr_available():
        print("SKIP: GDR not available (nvidia-peermem required)")
        return

    rank, world_size = init_distributed()
    n = 4

    assert world_size == n, f"This test requires exactly {n} processes"

    master_port = int(os.environ.get('MASTER_PORT', '6000'))
    base_port = int(os.environ.get('FRCHECK_BASE_PORT', str(master_port + 20000)))

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Starting FRCheck distributed test (world_size={world_size})")

    native = native_mod.FRCheckNative(POA_PATH)
    native.compile_plans(rank)
    assert native.n() == n, f"n={native.n()} != expected {n}"
    assert native.num_stripes() == n * (n - 1), \
        f"num_stripes={native.num_stripes()} != {n * (n - 1)}"

    my_ip = '127.0.0.1'
    peer_ips = ['127.0.0.1'] * n
    listen_port = base_port + rank
    print(f"[Rank {rank}] RDMA init: base_port={base_port}, listen_port={listen_port}, IP={my_ip}")

    try:
        native.init_rdma(
            group_size=n,
            rank_in_group=rank,
            base_port=base_port,
            my_ip=my_ip,
            peer_ips=peer_ips,
            use_rdma=True,
        )
    except RuntimeError as e:
        print(f"[Rank {rank}] RDMA init failed: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

    print(f"[Rank {rank}] RDMA full-mesh connected: group_size={native.group_size()}")
    dist.barrier()

    block_size = 4096
    recv_total = (n - 2) * block_size
    num_stripes = native.num_stripes()

    n_source_my = sum(
        1 for sid in range(num_stripes) if native.get_role_for_stripe(sid) == 0
    )
    layer_capacity = max(n_source_my, 1) * block_size

    layer_buf_gpu = torch.zeros(layer_capacity, dtype=torch.uint8, device=device)
    layer_mirror = torch.zeros(
        layer_capacity, dtype=torch.uint8, device='cpu', pin_memory=True,
    )
    recv_buf = torch.zeros(recv_total, dtype=torch.uint8, device='cpu', pin_memory=True)
    parity1_buf = torch.zeros(block_size, dtype=torch.uint8, device='cpu', pin_memory=True)
    parity2_buf = torch.zeros(block_size, dtype=torch.uint8, device='cpu', pin_memory=True)

    native.register_buffer(layer_buf_gpu.data_ptr(), layer_buf_gpu.numel())
    native.register_buffer(layer_mirror.data_ptr(), layer_mirror.numel())
    native.register_buffer(recv_buf.data_ptr(), recv_buf.numel())
    native.register_buffer(parity1_buf.data_ptr(), parity1_buf.numel())
    native.register_buffer(parity2_buf.data_ptr(), parity2_buf.numel())

    print(f"[Rank {rank}] Buffers registered: block_size={block_size}")

    layer_base = layer_buf_gpu.data_ptr()
    mirror_base = layer_mirror.data_ptr()
    src_block_per_node = 0

    roles_seen = {0: 0, 1: 0, 2: 0}
    stripe_results = []

    native.reset_encoding_batch()
    for sid in range(num_stripes):
        role = native.get_role_for_stripe(sid)
        roles_seen[role] += 1

        source_data = 0
        source_mirror = 0
        recv_addr = 0
        p1 = 0
        p2_out = 0
        p2_in = 0

        if role == 0:
            val = rank * 100 + sid
            blk_idx = src_block_per_node
            src_block_per_node += 1
            off = blk_idx * block_size
            layer_buf_gpu[off : off + block_size].fill_(val % 256)
            source_data = layer_base + off
            source_mirror = mirror_base + off
        elif role == 1:
            recv_buf.zero_()
            parity1_buf.zero_()
            parity2_buf.zero_()
            recv_addr = recv_buf.data_ptr()
            p1 = parity1_buf.data_ptr()
            p2_out = parity2_buf.data_ptr()
        elif role == 2:
            parity2_buf.zero_()
            p2_in = parity2_buf.data_ptr()

        native.submit_stripe_chunk(
            sid,
            source_data,
            source_mirror,
            recv_addr,
            p1,
            p2_out,
            p2_in,
            block_size,
        )

        result = {
            'stripe_id': sid,
            'role': role,
            'data': layer_mirror[0:8].clone() if role == 0 else parity1_buf[0:8].clone(),
            'parity1': parity1_buf[0:8].clone(),
            'parity2': parity2_buf[0:8].clone(),
        }
        stripe_results.append(result)

    native.submit_encoding_sentinel()
    native.wait_encoding_batch()

    dist.barrier()

    role_name = {0: 'SOURCE', 1: 'ENCODER', 2: 'PARITY_TARGET'}
    print(f"\n[Rank {rank}] Role distribution: "
          f"SOURCE={roles_seen[0]} ENCODER={roles_seen[1]} PARITY_TARGET={roles_seen[2]}")
    print(f"[Rank {rank}] Stripe results (first 8 bytes):")
    for r in stripe_results:
        tag = role_name[r['role']]
        print(f"  stripe {r['stripe_id']:2d} [{tag:14s}] data={r['data'].tolist()}  "
              f"p1={r['parity1'].tolist()}  p2={r['parity2'].tolist()}")

    native.stop()
    dist.barrier()
    print(f"[Rank {rank}] FRCheck distributed test PASSED")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    frcheck_dist_test()
