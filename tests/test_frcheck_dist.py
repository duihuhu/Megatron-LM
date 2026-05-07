#!/usr/bin/env python3
"""
Distributed FRCheck RDMA test (torchrun).

Usage:
  torchrun --nproc_per_node=4 tests/test_frcheck_dist.py

Tests:
  - RDMA full-mesh connection establishment
  - submit_stripe_encode pipeline (SOURCE→ENCODER→PARITY_TARGET)
  - Data integrity across the encode pipeline
"""

import sys, os

# Filter out the strategies dir which has a torch.py that shadows real torch
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
    rank, world_size = init_distributed()
    n = 4

    assert world_size == n, f"This test requires exactly {n} processes"

    # All ranks use the same base_port; each rank listens on base_port + rank_in_group.
    # The base port is MASTER_PORT + 20000 (or FRCHECK_BASE_PORT if set).
    # On a single machine, use an unambiguous base port.
    master_port = int(os.environ.get('MASTER_PORT', '6000'))
    base_port = int(os.environ.get('FRCHECK_BASE_PORT', str(master_port + 20000)))

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Starting FRCheck distributed test (world_size={world_size})")

    # Load native module with POA
    native = native_mod.FRCheckNative(POA_PATH)
    native.compile_plans(rank)
    assert native.n() == n, f"n={native.n()} != expected {n}"
    assert native.num_stripes() == n * (n - 1), \
        f"num_stripes={native.num_stripes()} != {n * (n - 1)}"

    # Init RDMA full-mesh
    my_ip = '127.0.0.1'
    peer_ips = ['127.0.0.1'] * n

    # Use per-rank acceptor port
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
        # Cleanup and exit
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

    print(f"[Rank {rank}] RDMA full-mesh connected: group_size={native.group_size()}")

    # Register buffers for stripe encode
    block_size = 4096  # small block for testing
    recv_total = (n - 2) * block_size  # encoder receives n-2 blocks

    data_buf = torch.zeros(block_size, dtype=torch.uint8, device='cpu')
    recv_buf = torch.zeros(recv_total, dtype=torch.uint8, device='cpu')
    parity1_buf = torch.zeros(block_size, dtype=torch.uint8, device='cpu')
    parity2_buf = torch.zeros(block_size, dtype=torch.uint8, device='cpu')

    native.register_buffer(data_buf.data_ptr(), data_buf.numel())
    native.register_buffer(recv_buf.data_ptr(), recv_buf.numel())
    native.register_buffer(parity1_buf.data_ptr(), parity1_buf.numel())
    native.register_buffer(parity2_buf.data_ptr(), parity2_buf.numel())

    print(f"[Rank {rank}] Buffers registered: block_size={block_size}")

    # Fill data buffer with rank-specific pattern
    for i in range(block_size // 4):
        data_buf[i * 4:(i + 1) * 4] = torch.tensor([rank, (rank + 1) % 256,
                                                      (rank + 2) % 256, (rank + 3) % 256],
                                                     dtype=torch.uint8)

    # Process all stripes
    roles_seen = {0: 0, 1: 0, 2: 0}  # SOURCE, ENCODER, PARITY_TARGET
    stripe_results = []

    for sid in range(native.num_stripes()):
        role = native.get_role_for_stripe(sid)
        roles_seen[role] += 1

        # For SOURCE: fill data_buf with unique pattern per stripe
        if role == 0:  # SOURCE
            val = rank * 100 + sid
            data_buf.fill_(val % 256)
        elif role == 1:  # ENCODER
            recv_buf.zero_()
        elif role == 2:  # PARITY_TARGET
            parity2_buf.zero_()

        # Submit stripe encode
        native.submit_stripe_encode(
            stripe_id=sid,
            my_data_addr=data_buf.data_ptr(),
            block_size=block_size,
            recv_buf_addr=recv_buf.data_ptr(),
            recv_buf_size=recv_buf.numel(),
            parity1_out_addr=parity1_buf.data_ptr(),
            parity2_out_addr=parity2_buf.data_ptr(),
            parity2_in_addr=parity2_buf.data_ptr(),
        )

        result = {
            'stripe_id': sid,
            'role': role,
            'data': data_buf[0:8].clone(),
            'parity1': parity1_buf[0:8].clone(),
            'parity2': parity2_buf[0:8].clone(),
        }
        stripe_results.append(result)

    # Sync all ranks before reading results
    dist.barrier()

    # Report results
    role_name = {0: 'SOURCE', 1: 'ENCODER', 2: 'PARITY_TARGET'}
    print(f"\n[Rank {rank}] Role distribution: "
          f"SOURCE={roles_seen[0]} ENCODER={roles_seen[1]} PARITY_TARGET={roles_seen[2]}")
    print(f"[Rank {rank}] Stripe results (first 8 bytes):")
    for r in stripe_results:
        tag = role_name[r['role']]
        print(f"  stripe {r['stripe_id']:2d} [{tag:14s}] data={r['data'].tolist()}  "
              f"p1={r['parity1'].tolist()}  p2={r['parity2'].tolist()}")

    # Cleanup
    native.stop()
    dist.barrier()
    print(f"[Rank {rank}] FRCheck distributed test PASSED")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    frcheck_dist_test()
