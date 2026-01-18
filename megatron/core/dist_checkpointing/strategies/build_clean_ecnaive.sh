#!/bin/bash
# Clean build script for EC-NAIVE native C++ module
# Supports both ASIO (TCP) and RDMA (InfiniBand) transports

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building EC-NAIVE Native C++ Module"
echo "  - ASIO (TCP) transport: always enabled"
echo "  - RDMA (InfiniBand) transport: optional"
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

# Check for ISA-L (required for EC encoding)
ISAL_FOUND=0
if [ -f "/usr/include/isa-l.h" ] || [ -f "/usr/local/include/isa-l.h" ] || [ -f "/usr/include/isa-l/erasure_code.h" ] || [ -f "/usr/local/include/isa-l/erasure_code.h" ]; then
    if ldconfig -p 2>/dev/null | grep -q libisal || [ -f "/usr/lib/x86_64-linux-gnu/libisal.so" ] || [ -f "/usr/lib64/libisal.so" ] || [ -f "/usr/local/lib/libisal.so" ]; then
        echo "✓ ISA-L library found"
        ISAL_FOUND=1
    fi
fi

if [ $ISAL_FOUND -eq 0 ]; then
    echo "❌ Error: ISA-L library not found. EC-NAIVE requires ISA-L for erasure coding."
    echo "   Ubuntu/Debian: sudo apt-get install libisal-dev"
    echo "   CentOS/RHEL: sudo yum install libisal-devel"
    echo "   From source: https://github.com/intel/isa-l"
    exit 1
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
    echo "⚠️  Warning: RDMA libraries not found. Module will build but RDMA support will not work."
    echo "   To enable RDMA support, install:"
    echo "   Ubuntu/Debian: sudo apt-get install libibverbs-dev librdmacm-dev"
    echo "   CentOS/RHEL: sudo yum install libibverbs-devel librdmacm-devel"
    echo "   Note: ASIO (TCP) transport will still work without RDMA libraries"
fi

echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -f ecnaive_native*.so
rm -f ecnaive_native*.pyd
rm -rf build/
rm -rf *.egg-info/

# Create a completely isolated build environment
TEMP_DIR=$(mktemp -d)
echo "Using isolated build directory: $TEMP_DIR"

# Copy only the necessary files
cp setup_simple_ecnaive.py "$TEMP_DIR/"
cp ecnaive_native.cpp "$TEMP_DIR/"

# Build in completely isolated environment
cd "$TEMP_DIR"

# Set completely clean environment
export PYTHONPATH=""
export PYTHONUSERBASE=""
unset PYTHONPATH
unset PYTHONUSERBASE

# Build the module
echo "Building in isolated environment..."
python3 setup_simple_ecnaive.py build_ext --inplace

# Copy built module back
cd "$SCRIPT_DIR"
cp "$TEMP_DIR"/ecnaive_native*.so . 2>/dev/null || true
cp "$TEMP_DIR"/ecnaive_native*.pyd . 2>/dev/null || true

# Clean up
rm -rf "$TEMP_DIR"

# Check if build was successful
if [ -f ecnaive_native*.so ] || [ -f ecnaive_native*.pyd ]; then
    echo ""
    echo "=========================================="
    echo "✅ EC-NAIVE native module built successfully!"
    echo "=========================================="
    echo "Module location: $(pwd)/ecnaive_native*.so"
    echo ""
    echo "Available transport modes:"
    echo "  - ASIO (TCP): Always available"
    if [ $RDMA_FOUND -eq 1 ]; then
        echo "  - RDMA (InfiniBand): ✓ Available"
    else
        echo "  - RDMA (InfiniBand): ✗ Not available (libraries not found)"
    fi
    echo ""
    echo "EC encoding library:"
    echo "  - ISA-L: ✓ Available"
    echo ""
    echo "Usage:"
    echo "  Basic (ASIO/TCP):  --use-ecnaive"
    if [ $RDMA_FOUND -eq 1 ]; then
        echo "  With RDMA:         --use-ecnaive --use-rdma"
    else
        echo "  With RDMA:         (Install RDMA libraries first)"
    fi
    echo ""
    echo "To test the module, run:"
    echo "  python3 -c 'import ecnaive_native; print(\"EC-NAIVE module loaded successfully\")'"
else
    echo ""
    echo "=========================================="
    echo "❌ Build failed!"
    echo "=========================================="
    echo "Please check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  1. Boost not installed: sudo apt-get install libboost-all-dev"
    echo "  2. pybind11 not installed: pip install pybind11"
    echo "  3. PyTorch not installed: pip install torch"
    echo "  4. ISA-L not installed (required): sudo apt-get install libisal-dev"
    echo "  5. RDMA libraries missing (optional): sudo apt-get install libibverbs-dev librdmacm-dev"
    exit 1
fi

