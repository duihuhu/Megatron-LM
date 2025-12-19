#!/bin/bash

# Build script for layer_transfer_cpp module
# This script compiles the C++ extension for layer-by-layer tensor transfer
#
# Usage:
#   bash build_layer_transfer.sh          # Build without CUDA (PyTorch mode)
#   bash build_layer_transfer.sh cuda     # Build with CUDA support

set -e

USE_CUDA=false
if [ "$1" == "cuda" ]; then
    USE_CUDA=true
fi

if [ "$USE_CUDA" == "true" ]; then
    echo "Building layer_transfer_cpp module WITH CUDA support..."
else
    echo "Building layer_transfer_cpp module WITHOUT CUDA (PyTorch mode)..."
fi

# Get Python paths
PYTHON_INCLUDE=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
PYTHON_LIB=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['stdlib'])")

# Get pybind11 include path
PYBIND11_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")

echo "Python include: $PYTHON_INCLUDE"
echo "Pybind11 include: $PYBIND11_INCLUDE"

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SUFFIX=".dylib"
    EXTRA_FLAGS="-undefined dynamic_lookup"
else
    # Linux
    SUFFIX=".so"
    EXTRA_FLAGS=""
fi

if [ "$USE_CUDA" == "true" ]; then
    # Check if nvcc is available
    if ! command -v nvcc &> /dev/null; then
        echo "Error: nvcc not found. Please install CUDA toolkit or build without CUDA."
        exit 1
    fi
    
    CUDA_INCLUDE=$(dirname $(dirname $(which nvcc)))/include
    echo "CUDA include: $CUDA_INCLUDE"
    
    # Compile with nvcc
    nvcc -O3 -std=c++17 --shared --compiler-options '-fPIC' \
        -I"$PYTHON_INCLUDE" \
        -I"$PYBIND11_INCLUDE" \
        -I"$CUDA_INCLUDE" \
        -DUSE_CUDA \
        $EXTRA_FLAGS \
        layer_transfer_cpp.cpp \
        -o layer_transfer_cpp$SUFFIX
    
    echo "Build successful with CUDA support! Output: layer_transfer_cpp$SUFFIX"
else
    # Compile with g++ (no CUDA)
    g++ -O3 -Wall -shared -std=c++17 -fPIC \
        -I"$PYTHON_INCLUDE" \
        -I"$PYBIND11_INCLUDE" \
        $EXTRA_FLAGS \
        layer_transfer_cpp.cpp \
        -o layer_transfer_cpp$SUFFIX
    
    echo "Build successful without CUDA! Output: layer_transfer_cpp$SUFFIX"
fi

echo ""
echo "To test the module:"
echo "  python3 -c 'import layer_transfer_cpp; print(\"CUDA support:\", layer_transfer_cpp.USE_CUDA)'"
echo ""
if [ "$USE_CUDA" == "false" ]; then
    echo "Note: Built in PyTorch mode (no CUDA in C++)"
    echo "      PyTorch will handle GPU->CPU transfers, C++ thread coordinates layers"
    echo "      To build with CUDA support, run: bash build_layer_transfer.sh cuda"
fi

