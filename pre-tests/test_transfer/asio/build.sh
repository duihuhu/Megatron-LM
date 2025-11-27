#!/bin/bash
#
# Build script for ASIO latency test programs
#
# This script compiles the ASIO server and client programs.
# Requires Boost.Asio library (header-only, part of Boost.System)
#
# Usage:
#   chmod +x build.sh
#   ./build.sh

set -e

echo "========================================="
echo "Building ASIO Latency Test Programs"
echo "========================================="

# Check for required tools
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ compiler not found"
    exit 1
fi

echo "Compiler: $(g++ --version | head -n1)"
echo ""

# Check for Boost libraries
BOOST_AVAILABLE=false
if pkg-config --exists libboost-system 2>/dev/null; then
    BOOST_AVAILABLE=true
    BOOST_CFLAGS=$(pkg-config --cflags libboost-system)
    BOOST_LIBS=$(pkg-config --libs libboost-system)
    echo "Found Boost via pkg-config"
elif [ -d "/usr/include/boost" ] || [ -d "/usr/local/include/boost" ]; then
    BOOST_AVAILABLE=true
    BOOST_CFLAGS=""
    BOOST_LIBS="-lboost_system"
    echo "Found Boost in standard locations"
fi

if [ "$BOOST_AVAILABLE" = false ]; then
    echo "Warning: Boost libraries not found via pkg-config or in standard locations"
    echo "Attempting to compile anyway (Boost.Asio is header-only)"
    BOOST_CFLAGS=""
    BOOST_LIBS="-lboost_system"
fi

echo ""

# Compiler flags
CXXFLAGS="-std=c++17 -O3 -Wall -Wextra -pthread"

# Build server
echo "Building asio_server..."
g++ $CXXFLAGS $BOOST_CFLAGS \
    asio_server.cpp \
    -o asio_server \
    $BOOST_LIBS

if [ $? -eq 0 ]; then
    echo "✓ asio_server compiled successfully"
else
    echo "✗ Failed to compile asio_server"
    exit 1
fi

echo ""

# Build client
echo "Building asio_client..."
g++ $CXXFLAGS $BOOST_CFLAGS \
    asio_client.cpp \
    -o asio_client \
    $BOOST_LIBS

if [ $? -eq 0 ]; then
    echo "✓ asio_client compiled successfully"
else
    echo "✗ Failed to compile asio_client"
    exit 1
fi

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - asio_server"
echo "  - asio_client"
echo ""
echo "Usage:"
echo "  Server: ./asio_server <port>"
echo "  Client: ./asio_client <host> <port> <packet_size> <num_packets> [interval_us]"
echo ""

