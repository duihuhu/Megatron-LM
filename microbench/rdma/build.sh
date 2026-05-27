#!/bin/bash

# Build script for RDMA throughput test

set -e

echo "Building RDMA microbench (rdma_ec_bind_bench)..."

# Create build directory
mkdir -p build
cd build

# Run CMake
cmake ..

# Build
make -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "Build completed successfully!"
echo "Executable: build/rdma_ec_bind_bench"
echo ""
echo "Two-rank launch: see run_torch_dist_2node.sh and bench_launcher.py"

