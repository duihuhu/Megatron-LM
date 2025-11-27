#!/bin/bash
#
# Build script for RDMA latency test programs
#
# This script compiles the RDMA server and client programs.
# Requires libibverbs and librdmacm libraries.
#
# Usage:
#   chmod +x build.sh
#   ./build.sh

set -e

echo "========================================="
echo "Building RDMA Latency Test Programs"
echo "========================================="

# Check for required tools
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ compiler not found"
    exit 1
fi

echo "Compiler: $(g++ --version | head -n1)"
echo ""

# Check for RDMA libraries
RDMA_AVAILABLE=false
if pkg-config --exists libibverbs librdmacm 2>/dev/null; then
    RDMA_AVAILABLE=true
    RDMA_CFLAGS=$(pkg-config --cflags libibverbs librdmacm)
    RDMA_LIBS=$(pkg-config --libs libibverbs librdmacm)
    echo "Found RDMA libraries via pkg-config"
elif [ -f "/usr/include/infiniband/verbs.h" ] || [ -f "/usr/local/include/infiniband/verbs.h" ]; then
    RDMA_AVAILABLE=true
    RDMA_CFLAGS=""
    RDMA_LIBS="-libverbs -lrdmacm"
    echo "Found RDMA libraries in standard locations"
fi

if [ "$RDMA_AVAILABLE" = false ]; then
    echo "Error: RDMA libraries not found"
    echo "Please install libibverbs and librdmacm:"
    echo "  Ubuntu/Debian: sudo apt-get install libibverbs-dev librdmacm-dev"
    echo "  RHEL/CentOS:   sudo yum install libibverbs-devel librdmacm-devel"
    exit 1
fi

echo ""

# Compiler flags
CXXFLAGS="-std=c++17 -O3 -Wall -Wextra -pthread"

# Build server
echo "Building rdma_server..."
g++ $CXXFLAGS $RDMA_CFLAGS \
    rdma_server.cpp \
    -o rdma_server \
    $RDMA_LIBS

if [ $? -eq 0 ]; then
    echo "✓ rdma_server compiled successfully"
else
    echo "✗ Failed to compile rdma_server"
    exit 1
fi

echo ""

# Build client
echo "Building rdma_client..."
g++ $CXXFLAGS $RDMA_CFLAGS \
    rdma_client.cpp \
    -o rdma_client \
    $RDMA_LIBS

if [ $? -eq 0 ]; then
    echo "✓ rdma_client compiled successfully"
else
    echo "✗ Failed to compile rdma_client"
    exit 1
fi

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - rdma_server"
echo "  - rdma_client"
echo ""
echo "Note: RDMA requires appropriate hardware and driver support."
echo "Usage:"
echo "  Server: ./rdma_server <port>"
echo "  Client: ./rdma_client <host> <port> <packet_size> <num_packets> [interval_us] [warmup_packets]"
echo ""

