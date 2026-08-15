#!/usr/bin/env python3
"""
Standalone Concord RDMA test for a single process (no distributed).

Tests:
  1. Native module POA loading and stripe plan compilation
  2. RDMA device init and buffer registration
  3. Stripe query consistency

Usage:
  python tests/test_concord_rdma.py
"""

import sys, os

# Filter out the strategies dir which has a torch.py that shadows real torch
sys.path = [p for p in sys.path if 'dist_checkpointing/strategies' not in p]

import torch
import importlib.util

STRATEGIES = '/workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies'
POA_PATH = os.path.join(STRATEGIES, 'poa_n4.txt')
SO_PATH = os.path.join(STRATEGIES, 'concord_native.cpython-310-x86_64-linux-gnu.so')

# Load native module
spec = importlib.util.spec_from_file_location('concord_native', SO_PATH)
native_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_mod)


def test_poa_loading():
    """Test POA loading and stripe plan compilation for all ranks."""
    print("=== Test POA Loading ===")
    for rank in range(4):
        native = native_mod.ConcordNative(POA_PATH)
        native.compile_plans(rank)

        assert native.n() == 4, f"n should be 4, got {native.n()}"
        assert native.num_stripes() == 12, f"num_stripes should be 12, got {native.num_stripes()}"
        assert native.rank_in_group() == rank, f"rank_in_group should be {rank}, got {native.rank_in_group()}"

        roles = [native.get_role_for_stripe(s) for s in range(12)]
        assert roles.count(0) == 6, f"Rank {rank}: expected 6 SOURCE, got {roles.count(0)}"
        assert roles.count(1) == 3, f"Rank {rank}: expected 3 ENCODER, got {roles.count(1)}"
        assert roles.count(2) == 3, f"Rank {rank}: expected 3 PARITY_TARGET, got {roles.count(2)}"

        # Verify each stripe row is a valid permutation
        for sid in range(12):
            row = list(native.row(sid))
            assert sorted(row) == [1, 2, 3, 4], f"Row {sid}: {row} not a permutation"

        # Verify source node IDs are correct
        for sid in range(12):
            src = list(native.get_source_node_ids(sid))
            row = list(native.row(sid))
            assert src == row[:2], f"Source mismatch for stripe {sid}"

        print(f"  Rank {rank}: OK (roles: {roles.count(0)}S/{roles.count(1)}E/{roles.count(2)}P)")

    print("  PASSED\n")


def test_native_rdma_init():
    """Test RDMA device init and buffer registration (single process)."""
    print("=== Test RDMA Init ===")

    # Query RDMA devices
    import subprocess
    result = subprocess.run(['ibv_devinfo', '-d', 'mlx5_0'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'state' in line.lower() or 'hca_id' in line.lower():
            print(f"  RDMA device: {line.strip()}")

    native = native_mod.ConcordNative(POA_PATH)
    native.compile_plans(0)

    # Init RDMA with self as single peer (group_size=1)
    native.init_rdma(
        group_size=4,
        rank_in_group=0,
        base_port=39999,
        my_ip='127.0.0.1',
        peer_ips=['127.0.0.1', '127.0.0.1', '127.0.0.1'],
        use_rdma=True,
    )
    print(f"  RDMA init OK: group_size={native.group_size()}")

    # Register a buffer
    buf = torch.zeros(65536, dtype=torch.uint8)
    native.register_buffer(buf.data_ptr(), buf.numel())
    print(f"  Buffer registration OK: size={buf.numel()}")

    native.stop()
    print("  PASSED\n")


def test_decompose_reconstruct():
    """Test state_dict decomposition and reconstruction."""
    print("=== Test Decompose/Reconstruct ===")
    sys.path.insert(0, STRATEGIES)
    try:
        from megatron.core.dist_checkpointing.strategies.state_dict_decomposer import (
            decompose_state_dict,
            reconstruct_state_dict,
            extract_tensors_from_continuous_buffer,
        )
    finally:
        sys.path.remove(STRATEGIES)

    # Create a simple state dict with various tensor types
    state_dict = {
        'model.layers.0.weight': torch.randn(64, 64),
        'model.layers.0.bias': torch.randn(64),
        'model.layers.1.weight': torch.randn(32, 64),
        'config': {'hidden_size': 64, 'num_layers': 2},
        'optimizer.state': 12345,
    }

    decomposed = decompose_state_dict(state_dict)
    total_size = decomposed.total_tensor_size_bytes
    print(f"  Total tensor size: {total_size} bytes ({total_size / 1024:.1f} KB)")
    print(f"  Num tensors: {len(decomposed.tensor_data)}")
    print(f"  Non-tensor keys: {len(decomposed.non_tensor_data)}")

    # Reconstruct from DecomposedStateDict directly
    reconstructed = reconstruct_state_dict(decomposed)

    # Verify
    assert reconstructed['model']['layers']['0']['weight'] is not None
    assert torch.equal(
        state_dict['model.layers.0.weight'],
        reconstructed['model']['layers']['0']['weight']
    ), "weight mismatch"
    assert torch.equal(
        state_dict['model.layers.0.bias'],
        reconstructed['model']['layers']['0']['bias']
    ), "bias mismatch"
    assert reconstructed['config'] == state_dict['config'], "Non-tensor mismatch"
    assert reconstructed['optimizer']['state'] == state_dict['optimizer.state'], "Optimizer state mismatch"

    print("  Reconstruction verification: PASSED")
    print("  PASSED\n")


def test_concord_manager():
    """Test ConcordManager helper methods without RDMA."""
    print("=== Test ConcordManager ===")
    from megatron.core.dist_checkpointing.strategies.concord_manager import (
        ConcordManager, StripeRole
    )

    mgr = ConcordManager()
    # Test group layout for world_size=4, n=4
    layout = ConcordManager._get_group_layout(4, 4)
    assert layout['num_groups'] == 1, f"Expected 1 group, got {layout['num_groups']}"

    for rank in range(4):
        gid = ConcordManager._get_group_id(rank, 4, 4)
        rig = ConcordManager._get_rank_in_group(rank, 4, 4)
        restored = ConcordManager._get_rank_by_group_position(gid, rig, 4, 4)
        assert gid == 0, f"Rank {rank}: expected group_id=0, got {gid}"
        assert rig == rank, f"Rank {rank}: expected rank_in_group={rank}, got {rig}"
        assert restored == rank, f"Rank {rank}: restored rank mismatch"

    print("  Group layout helpers: OK")

    # Test native through manager interface
    native = native_mod.ConcordNative(POA_PATH)
    for r in range(4):
        n2 = native_mod.ConcordNative(POA_PATH)
        n2.compile_plans(r)
        ns = n2.num_stripes()

        roles = {}
        for sid in range(ns):
            role_val = n2.get_role_for_stripe(sid)
            role = StripeRole(role_val)
            roles[role] = roles.get(role, 0) + 1

        assert roles[StripeRole.SOURCE] == 6, f"Rank {r}: expected 6 SOURCE, got {roles[StripeRole.SOURCE]}"
        assert roles[StripeRole.ENCODER] == 3, f"Rank {r}: expected 3 ENCODER, got {roles[StripeRole.ENCODER]}"
        assert roles[StripeRole.PARITY_TARGET] == 3, f"Rank {r}: expected 3 PARITY_TARGET, got {roles[StripeRole.PARITY_TARGET]}"

    print("  StripePlan distribution: OK")
    print("  PASSED\n")


def test_poa_autogenerate():
    """Test auto-generation of POA tables for various n."""
    print("=== Test POA Auto-generation ===")

    for n in [3, 4, 5, 7, 8, 11, 16]:
        native = native_mod.ConcordNative(n)
        assert native.n() == n, f"n mismatch: {native.n()} != {n}"
        expected_stripes = n * (n - 1)
        assert native.num_stripes() == expected_stripes, \
            f"num_stripes mismatch: {native.num_stripes()} != {expected_stripes}"

        # Verify all rows are permutations
        for sid in range(native.num_stripes()):
            row = list(native.row(sid))
            assert sorted(row) == list(range(1, n + 1)), \
                f"Row {sid} not a permutation: {row}"

        # Verify balance for each rank
        for r in range(min(n, 4)):  # test first few ranks
            n2 = native_mod.ConcordNative(n)
            n2.compile_plans(r)
            roles = [n2.get_role_for_stripe(s) for s in range(n2.num_stripes())]
            # SOURCE count = (n-1)*(n-2), ENCODER = n-1, PARITY_TARGET = n-1
            assert roles.count(0) == (n - 1) * (n - 2), \
                f"n={n} rank={r}: expected {(n-1)*(n-2)} SOURCE, got {roles.count(0)}"
            assert roles.count(1) == n - 1, \
                f"n={n} rank={r}: expected {n-1} ENCODER, got {roles.count(1)}"
            assert roles.count(2) == n - 1, \
                f"n={n} rank={r}: expected {n-1} PARITY_TARGET, got {roles.count(2)}"

        print(f"  n={n}: {expected_stripes} stripes, roles balanced, all permutations OK")

    print("  PASSED\n")


if __name__ == '__main__':
    test_poa_loading()
    test_poa_autogenerate()
    test_decompose_reconstruct()
    test_concord_manager()

    # Only test RDMA if explicitly requested (requires RDMA hardware)
    if os.environ.get('TEST_CONCORD_RDMA', '0') == '1':
        test_native_rdma_init()

    print("All tests passed!")
