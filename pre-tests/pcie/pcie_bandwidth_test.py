#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCIE Bandwidth Test (GPU<->CPU) using PyTorch

Measures H2D (Host-to-Device) and D2H (Device-to-Host) memcpy bandwidth using:
- Pinned host memory
- CUDA streams & events for accurate timing
- Optional size sweep & iterations
- Optional async copies (non_blocking)

Usage examples:
  python pcie_bandwidth_test.py
  python pcie_bandwidth_test.py --direction both --iters 100 --sizes 64,256,1024
  python pcie_bandwidth_test.py --direction h2d --min-mb 16 --max-mb 2048 --steps 6
  python pcie_bandwidth_test.py --device 0

Requirements:
  - Python 3.8+
  - PyTorch with CUDA build (pip install torch --index-url https://download.pytorch.org/whl/cu118)
"""

import argparse
import math
import sys
import time
from typing import List, Tuple, Optional

try:
    import torch
except Exception as e:
    print("ERROR: PyTorch is required. Install a CUDA build of torch.", file=sys.stderr)
    raise

def parse_sizes(args) -> List[int]:
    sizes: List[int] = []
    if args.sizes:
        for tok in args.sizes.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower().endswith("mb"):
                val = float(tok[:-2])
                sizes.append(int(val * (1024**2)))
            elif tok.lower().endswith("gb"):
                val = float(tok[:-2])
                sizes.append(int(val * (1024**3)))
            else:
                # assume MB if unit not provided
                val = float(tok)
                sizes.append(int(val * (1024**2)))
        return sizes
    # logspace sweep between min-mb and max-mb (inclusive) with 'steps' points
    if args.steps <= 1:
        sizes.append(int(args.min_mb * (1024**2)))
        return sizes
    log_min = math.log2(args.min_mb)
    log_max = math.log2(args.max_mb)
    for i in range(args.steps):
        mb = 2 ** (log_min + (log_max - log_min) * i / (args.steps - 1))
        sizes.append(int(round(mb) * (1024**2)))
    # ensure uniqueness and sort
    sizes = sorted(set(sizes))
    return sizes

def human_gbps(bytes_per_sec: float) -> float:
    return bytes_per_sec / 1e9

def alloc_tensors(size_bytes: int, device: torch.device, dtype=torch.uint8):
    elems = size_bytes  # uint8 -> 1 byte each
    # Host pinned and device buffers
    h = torch.empty(elems, dtype=dtype, pin_memory=True)
    d = torch.empty(elems, dtype=dtype, device=device)
    return h, d

@torch.inference_mode()
def measure_copy(
    size_bytes: int,
    device: torch.device,
    direction: str,
    iters: int,
    async_copy: bool,
    warmup: int,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[float, float]:
    """
    Returns (gbps, seconds_per_iter)
    """
    if stream is None:
        stream = torch.cuda.current_stream(device=device)

    h, d = alloc_tensors(size_bytes, device)

    # Touch memory to avoid first-use penalties
    h.fill_(1)
    d.fill_(2)

    # Warmup
    for _ in range(max(warmup, 0)):
        if direction == "h2d":
            d.copy_(h, non_blocking=async_copy)
        elif direction == "d2h":
            h.copy_(d, non_blocking=async_copy)
        else:
            d.copy_(h, non_blocking=async_copy)
            h.copy_(d, non_blocking=async_copy)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    total_bytes = 0
    if direction == "h2d":
        for _ in range(iters):
            d.copy_(h, non_blocking=async_copy)
            total_bytes += size_bytes
    elif direction == "d2h":
        for _ in range(iters):
            h.copy_(d, non_blocking=async_copy)
            total_bytes += size_bytes
    else:  # both
        for _ in range(iters):
            d.copy_(h, non_blocking=async_copy)
            h.copy_(d, non_blocking=async_copy)
            total_bytes += 2 * size_bytes

    end.record(stream)
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1000.0

    gbps = human_gbps(total_bytes / elapsed_s)
    return gbps, elapsed_s / iters

def main():
    parser = argparse.ArgumentParser(description="PCIe bandwidth tester (GPU<->CPU) using PyTorch")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--direction", type=str, default="both", choices=["h2d", "d2h", "both"], help="Copy direction")
    parser.add_argument("--iters", type=int, default=50, help="Iterations per size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--async", dest="async_copy", action="store_true", help="Use non_blocking async copies")
    parser.add_argument("--sizes", type=str, default="", help="Comma-separated sizes (MB/GB), e.g., 64,256,1GB")
    parser.add_argument("--min-mb", type=float, default=32.0, help="Min size (MB) for sweep if --sizes not given")
    parser.add_argument("--max-mb", type=float, default=1024.0, help="Max size (MB) for sweep if --sizes not given")
    parser.add_argument("--steps", type=int, default=6, help="Number of points in size sweep (log-spaced)")
    parser.add_argument("--no-header", action="store_true", help="Do not print header line")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Please run on a machine with a CUDA-capable GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    # Basic device info
    name = torch.cuda.get_device_name(device)
    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)
    if not args.no_header:
        print(f"# Device: {name} (cc {cc_major}.{cc_minor}), PCIe BW test")
        print(f"# Direction: {args.direction}, iters={args.iters}, warmup={args.warmup}, async={args.async_copy}")
        print("# size_MB, gbps, ms_per_iter")

    sizes = parse_sizes(args)
    stream = torch.cuda.Stream(device=device)

    for sz in sizes:
        try:
            with torch.cuda.stream(stream):
                gbps, sec_per_iter = measure_copy(
                    size_bytes=sz,
                    device=device,
                    direction=args.direction,
                    iters=args.iters,
                    async_copy=args.async_copy,
                    warmup=args.warmup,
                    stream=stream,
                )
            print(f"{sz / (1024**2):.0f}, {gbps:.3f}, {sec_per_iter * 1000:.3f}")
        except RuntimeError as e:
            print(f"{sz / (1024**2):.0f}, ERROR, {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
