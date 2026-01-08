#!/bin/bash

# Build script for network throughput test

set -e

echo "Building Network Throughput Test..."

# Create build directory
mkdir -p build
cd build

# Run CMake
cmake ..

# Build
make -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "Build completed successfully!"
echo "Executable: build/network_throughput_test"
echo ""
echo "Run './build/network_throughput_test --help' for usage information"

