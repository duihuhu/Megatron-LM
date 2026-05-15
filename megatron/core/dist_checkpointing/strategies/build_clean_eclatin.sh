#!/bin/bash
# Clean build script for ECLATIN native C++ module
# Supports both ASIO (TCP) and RDMA (InfiniBand) transports
# Also supports optional CUDA for GPU operations
#
# Features:
#   - ASIO (TCP): Always available
#   - RDMA: Optional, enabled if libibverbs and librdmacm are detected
#   - CUDA: Optional, enabled with 'cuda' argument
#
# Usage:
#   bash build_clean_eclatin.sh          # Build without CUDA (PyTorch mode)
#   bash build_clean_eclatin.sh cuda     # Build with CUDA support

set -e

USE_CUDA=false
if [ "$1" == "cuda" ]; then
    USE_CUDA=true
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building ECLATIN Native C++ Module"
echo "  - ASIO (TCP) transport: always enabled"
echo "  - RDMA (InfiniBand) transport: optional"
if [ "$USE_CUDA" == "true" ]; then
    echo "  - CUDA support: enabled"
else
    echo "  - CUDA support: disabled (PyTorch mode)"
fi
echo "=========================================="

# Check dependencies
echo "Checking dependencies..."

# Check for pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "❌ Error: pybind11 not found. Please install pybind11 first."
    echo "   pip install pybind11"
    exit 1
fi
echo "✓ pybind11 found"

# Check for Boost (required for ASIO)
if ! python3 -c "import os; exit(0 if os.path.exists('/usr/include/boost/asio.hpp') or os.path.exists('/usr/local/include/boost/asio.hpp') or os.path.exists('/opt/homebrew/include/boost/asio.hpp') else 1)" 2>/dev/null; then
    echo "⚠️  Warning: Boost headers may not be found. If build fails, install Boost:"
    echo "   Ubuntu/Debian: sudo apt-get install libboost-all-dev"
    echo "   CentOS/RHEL: sudo yum install boost-devel"
    echo "   macOS: brew install boost"
else
    echo "✓ Boost headers found"
fi

# Check for RDMA libraries (optional)
RDMA_FOUND=0
if [ -f "/usr/include/infiniband/verbs.h" ] || [ -f "/usr/local/include/infiniband/verbs.h" ]; then
    if ldconfig -p 2>/dev/null | grep -q libibverbs || [ -f "/usr/lib/x86_64-linux-gnu/libibverbs.so" ] || [ -f "/usr/lib64/libibverbs.so" ]; then
        if ldconfig -p 2>/dev/null | grep -q librdmacm || [ -f "/usr/lib/x86_64-linux-gnu/librdmacm.so" ] || [ -f "/usr/lib64/librdmacm.so" ]; then
            echo "✓ RDMA libraries found (libibverbs, librdmacm)"
            RDMA_FOUND=1
        fi
    fi
fi

if [ $RDMA_FOUND -eq 0 ]; then
    echo "⚠️  Note: RDMA libraries not found. Module will build with ASIO/TCP only."
    echo "   RDMA support is optional. ASIO (TCP) transport works without it."
    echo "   To enable RDMA support (optional), install:"
    echo "   Ubuntu/Debian: sudo apt-get install libibverbs-dev librdmacm-dev"
    echo "   CentOS/RHEL: sudo yum install libibverbs-devel librdmacm-devel"
fi

# Check for CUDA (if requested)
if [ "$USE_CUDA" == "true" ]; then
    if ! command -v nvcc &> /dev/null; then
        echo "❌ Error: nvcc not found. Please install CUDA toolkit or build without CUDA."
        exit 1
    fi
    echo "✓ CUDA compiler (nvcc) found: $(which nvcc)"
fi

echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -f eclatin_native*.so
rm -f eclatin_native*.pyd
rm -rf build/
rm -rf *.egg-info/

# Create a completely isolated build environment
TEMP_DIR=$(mktemp -d)
echo "Using isolated build directory: $TEMP_DIR"

# Copy only the necessary files
cp setup_simple_eclatin.py "$TEMP_DIR/"
cp eclatin_native.cpp "$TEMP_DIR/"
cp rdma_device_utils.h "$TEMP_DIR/"

# Build in completely isolated environment
cd "$TEMP_DIR"

# Set completely clean environment
export PYTHONPATH=""
export PYTHONUSERBASE=""
unset PYTHONPATH
unset PYTHONUSERBASE

# Set USE_CUDA environment variable for setup script
if [ "$USE_CUDA" == "true" ]; then
    export USE_CUDA=true
else
    export USE_CUDA=false
fi

# Build the module
echo "Building in isolated environment..."
python3 setup_simple_eclatin.py build_ext --inplace

# Copy built module back
cd "$SCRIPT_DIR"
cp "$TEMP_DIR"/eclatin_native*.so . 2>/dev/null || true
cp "$TEMP_DIR"/eclatin_native*.pyd . 2>/dev/null || true

# Clean up
rm -rf "$TEMP_DIR"

# Check if build was successful
if [ -f eclatin_native*.so ] || [ -f eclatin_native*.pyd ]; then
    echo ""
    echo "=========================================="
    echo "✅ ECLATIN native module built successfully!"
    echo "=========================================="
    echo "Module location: $(pwd)/eclatin_native*.so"
    echo ""
    echo "Available transport modes:"
    echo "  - ASIO (TCP): ✓ Always available"
    if [ $RDMA_FOUND -eq 1 ]; then
        echo "  - RDMA (InfiniBand): ✓ Available (optional)"
    else
        echo "  - RDMA (InfiniBand): ✗ Not available (optional, libraries not found)"
    fi
    echo ""
    echo "CUDA support:"
    if [ "$USE_CUDA" == "true" ]; then
        echo "  - ✓ Enabled (C++ handles GPU->CPU transfers)"
    else
        echo "  - ✗ Disabled (PyTorch handles GPU->CPU transfers)"
        echo "  - To enable: bash build_clean_eclatin.sh cuda"
    fi
    echo ""
    echo "Usage:"
    echo "  Basic (ASIO/TCP):  --use-eclatin"
    echo "  With layerwise:    --use-eclatin --use-eclatin-layerwise"
    if [ $RDMA_FOUND -eq 1 ]; then
        echo "  With RDMA:         --use-eclatin --use-rdma"
        echo "  RDMA + layerwise:  --use-eclatin --use-eclatin-layerwise --use-rdma"
    else
        echo "  With RDMA:         (Install RDMA libraries first)"
    fi
    echo ""
    echo "To test the module, run:"
    echo "  python3 -c 'import eclatin_native; print(\"RDMA available:\", eclatin_native.is_rdma_available())'"
else
    echo ""
    echo "=========================================="
    echo "❌ Build failed!"
    echo "=========================================="
    echo "Please check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  1. Boost not installed (required): sudo apt-get install libboost-all-dev"
    echo "  2. pybind11 not installed (required): pip install pybind11"
    echo "  3. ISA-L not installed (required): sudo apt-get install libisal-dev"
    echo "  4. RDMA libraries missing (optional): sudo apt-get install libibverbs-dev librdmacm-dev"
    echo "  5. CUDA not found (if building with cuda): install CUDA toolkit"
    echo ""
    echo "Note: RDMA support is optional. The module works with ASIO/TCP even without RDMA."
    exit 1
fi
