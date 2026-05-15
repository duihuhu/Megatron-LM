#!/bin/bash
# Clean build script for EC-CHECK native C++ module
# Supports both ASIO (TCP) and RDMA (InfiniBand) transports

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building EC-CHECK Native C++ Module"
echo "  - ASIO (TCP) transport: always enabled"
echo "  - RDMA (InfiniBand) transport: optional"
echo "  - NCCL transport: optional"
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

# Check for ISA-L (required for erasure coding)
ISA_L_FOUND=0
if [ -f "/usr/include/isa-l/erasure_code.h" ] || [ -f "/usr/local/include/isa-l/erasure_code.h" ]; then
    if ldconfig -p 2>/dev/null | grep -q libisal || [ -f "/usr/lib/x86_64-linux-gnu/libisal.so" ] || [ -f "/usr/lib64/libisal.so" ]; then
        echo "✓ ISA-L library found"
        ISA_L_FOUND=1
    fi
fi

if [ $ISA_L_FOUND -eq 0 ]; then
    echo "⚠️  Warning: ISA-L library not found. EC-CHECK requires ISA-L."
    echo "   Ubuntu/Debian: sudo apt-get install libisal-dev"
    echo "   CentOS/RHEL: sudo yum install isa-l-devel"
    echo "   Or build from source: https://github.com/intel/isa-l"
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
    echo "   Note: ASIO (TCP) and NCCL transports will still work without RDMA libraries"
fi

# Check for NCCL (optional)
NCCL_FOUND=0
if [ -f "/usr/include/nccl.h" ] || [ -f "/usr/local/include/nccl.h" ] || [ -f "/usr/local/cuda/include/nccl.h" ]; then
    if ldconfig -p 2>/dev/null | grep -q libnccl || [ -f "/usr/lib/x86_64-linux-gnu/libnccl.so" ] || [ -f "/usr/lib64/libnccl.so" ]; then
        echo "✓ NCCL library found"
        NCCL_FOUND=1
    fi
fi

if [ $NCCL_FOUND -eq 0 ]; then
    echo "⚠️  Warning: NCCL library not found. NCCL transport will not be available."
    echo "   Note: ASIO (TCP) and RDMA transports will still work without NCCL"
fi

echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -f eccheck_native*.so
rm -f eccheck_native*.pyd
rm -rf build/
rm -rf *.egg-info/

# Create a completely isolated build environment
TEMP_DIR=$(mktemp -d)
echo "Using isolated build directory: $TEMP_DIR"

# Copy only the necessary files
cp setup_simple.py "$TEMP_DIR/"
cp eccheck_native.cpp "$TEMP_DIR/"
cp rdma_device_utils.h "$TEMP_DIR/"

# Build in completely isolated environment
cd "$TEMP_DIR"

# Set completely clean environment
export PYTHONPATH=""
export PYTHONUSERBASE=""
unset PYTHONPATH
unset PYTHONUSERBASE

# Build the module
echo "Building eccheck_native.so..."
python3 setup_simple.py build_ext --inplace

# Copy built module back
cd "$SCRIPT_DIR"
cp "$TEMP_DIR"/eccheck_native*.so . 2>/dev/null || true
cp "$TEMP_DIR"/eccheck_native*.pyd . 2>/dev/null || true

# Clean up
rm -rf "$TEMP_DIR"

# Check if build was successful
if [ -f eccheck_native*.so ] || [ -f eccheck_native*.pyd ]; then
    echo ""
    echo "=========================================="
    echo "✅ EC-CHECK native module built successfully!"
    echo "=========================================="
    echo "Module location: $(pwd)/eccheck_native*.so"
    echo ""
    echo "Available transport modes:"
    echo "  - ASIO (TCP): Always available"
    if [ $RDMA_FOUND -eq 1 ]; then
        echo "  - RDMA (InfiniBand): ✓ Available"
    else
        echo "  - RDMA (InfiniBand): ✗ Not available (libraries not found)"
    fi
    if [ $NCCL_FOUND -eq 1 ]; then
        echo "  - NCCL (GPU): ✓ Available"
    else
        echo "  - NCCL (GPU): ✗ Not available (library not found)"
    fi
    echo ""
    echo "Usage:"
    echo "  Basic (ASIO/TCP):  --use-eccheck"
    if [ $NCCL_FOUND -eq 1 ]; then
        echo "  With NCCL:         --use-eccheck (NCCL is default if available)"
    fi
    if [ $RDMA_FOUND -eq 1 ]; then
        echo "  With RDMA:         --use-eccheck --use-rdma"
        echo "                     (Set ECCHECK_USE_ASIO=true to enable ASIO+RDMA)"
    else
        echo "  With RDMA:         (Install RDMA libraries first)"
    fi
    echo ""
    echo "To test the module, run:"
    echo "  python3 -c 'import eccheck_native; print(\"EC-CHECK module loaded successfully\")'"
    if [ $RDMA_FOUND -eq 1 ]; then
        echo "  python3 -c 'import eccheck_native; print(\"RDMA available:\", eccheck_native.is_rdma_available())'"
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Build failed!"
    echo "=========================================="
    echo "Please check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  1. ISA-L not installed (required): sudo apt-get install libisal-dev"
    echo "  2. Boost not installed (required): sudo apt-get install libboost-all-dev"
    echo "  3. pybind11 not installed: pip install pybind11"
    echo "  4. RDMA libraries missing (optional): sudo apt-get install libibverbs-dev librdmacm-dev"
    echo "  5. NCCL not installed (optional): Install from https://developer.nvidia.com/nccl"
    exit 1
fi

echo ""