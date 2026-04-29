#!/bin/bash

# Build script for RDMA throughput test

set -e

echo "Building RDMA Throughput Test..."

# Create build directory
mkdir -p build
cd build

# Run CMake
cmake ..

# Build
make -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "Build completed successfully!"
echo "Executable: build/rdma_throughput_test"
echo ""
echo "Run './build/rdma_throughput_test' for usage information"

