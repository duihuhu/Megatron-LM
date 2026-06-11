#!/usr/bin/env python3
"""Verify FRCheck checkpoint RS encoding across all layers and stripes."""
import struct, ctypes
from pathlib import Path
import numpy as np

BASE = Path("/dev/shm/data/checkpoint/models/gpt2-345m-0-frcheck/iter_0000004")
POA = Path("/workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies/poa_n4.txt")
LAYERS = ["layer_common"] + [f"layer_{i}" for i in range(6)]  # adjust if needed
K, ROWS_P, M = 2, 2, 4

def load_poa():
    rows = []
    for line in POA.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append([int(x) for x in line.split()])
    return rows

def rank_dir(r, use_frcheck_sub=False):
    d = BASE / f"mp_rank_00_{r:03d}"
    return d / "frcheck" if use_frcheck_sub else d

def read_frbk(path):
    with open(path, "rb") as f:
        magic, sid, role, sz, blk = struct.unpack("<4sIIQQ", f.read(28))
    if magic != b"FRBK":
        raise ValueError(f"bad magic: {path}")
    with open(path, "rb") as f:
        f.seek(28)
        data = np.frombuffer(f.read(sz), dtype=np.uint8).copy()
    return data, sz, blk

lib = ctypes.CDLL("libisal.so")
def rs_encode(d0, d1):
    n = len(d0)
    enc_mat = (ctypes.c_ubyte * (M * K))()
    lib.gf_gen_rs_matrix(enc_mat, M, K)
    gftbls = (ctypes.c_ubyte * (32 * K * ROWS_P))()
    lib.ec_init_tables(K, ROWS_P, ctypes.byref(enc_mat, K * K), gftbls)
    p1, p2 = np.zeros(n, np.uint8), np.zeros(n, np.uint8)
    dp = (ctypes.c_void_p * K)(d0.ctypes.data, d1.ctypes.data)
    pp = (ctypes.c_void_p * ROWS_P)(p1.ctypes.data, p2.ctypes.data)
    lib.ec_encode_data(n, K, ROWS_P, gftbls, dp, pp)
    return p1, p2

def pct(a, b):
    n = min(len(a), len(b))
    return 100.0 * (a[:n] == b[:n]).sum() / n if n else 0.0

# auto-detect layout
use_sub = not (rank_dir(0) / "frcheck_main_rank0.pt").is_file()
if use_sub:
    print("Using OLD layout: mp_rank_*/frcheck/")
else:
    print("Using NEW layout: mp_rank_*/")

poa = load_poa()
grand = {"ok": 0, "bad": 0, "missing": 0}

for layer in LAYERS:
    layer_ok = layer_bad = layer_miss = 0
    for sid, row in enumerate(poa):
        src_r = [row[0]-1, row[1]-1]
        enc_r, par_r = row[2]-1, row[3]-1
        paths = {
            "d0": rank_dir(src_r[0], use_sub) / layer / f"stripe_{sid}" / f"frcheck_shard_rank{src_r[0]}.pt",
            "d1": rank_dir(src_r[1], use_sub) / layer / f"stripe_{sid}" / f"frcheck_shard_rank{src_r[1]}.pt",
            "p1": rank_dir(enc_r, use_sub) / layer / f"stripe_{sid}" / f"frcheck_shard_rank{enc_r}_p1.pt",
            "p2": rank_dir(enc_r, use_sub) / layer / f"stripe_{sid}" / f"frcheck_shard_rank{enc_r}_p2.pt",
            "tgt": rank_dir(par_r, use_sub) / layer / f"stripe_{sid}" / f"frcheck_shard_rank{par_r}.pt",
        }
        if not all(p.is_file() for p in paths.values()):
            layer_miss += 1
            continue
        d0, sz0, _ = read_frbk(paths["d0"])
        d1, _, _ = read_frbk(paths["d1"])
        p1d, _, _ = read_frbk(paths["p1"])
        p2d, _, _ = read_frbk(paths["p2"])
        tgt, _, _ = read_frbk(paths["tgt"])
        actual = min(sz0, len(d0), len(d1), len(p1d), len(p2d), len(tgt))
        d0, d1 = d0[:actual], d1[:actual]
        p1e, p2e = rs_encode(d0, d1)
        m1, m2, mt = pct(p1e, p1d), pct(p2e, p2d), pct(p2e, tgt)
        ok = m1 > 99.99 and m2 > 99.99 and mt > 99.99
        if ok:
            layer_ok += 1
        else:
            layer_bad += 1
            print(f"FAIL {layer} sid={sid} POA={row}: p1_rs={m1:.2f}% p2_rs={m2:.2f}% p2_vs_tgt={mt:.2f}% actual={actual}")
    print(f"{layer}: OK={layer_ok}/12 bad={layer_bad} missing={layer_miss}")
    grand["ok"] += layer_ok
    grand["bad"] += layer_bad
    grand["missing"] += layer_miss

print(f"TOTAL: OK={grand['ok']} bad={grand['bad']} missing={grand['missing']} (expect OK=84 for 7 layers x 12 stripes)")