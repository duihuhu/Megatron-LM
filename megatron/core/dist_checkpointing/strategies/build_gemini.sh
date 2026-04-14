#!/bin/bash
# Clean build script for Gemini native C++ module
# Supports both ASIO (TCP) and RDMA (InfiniBand) transports

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building Gemini Native C++ Module"
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
rm -f gemini_native*.so
rm -f gemini_native*.pyd
rm -rf build/
rm -rf *.egg-info/

# Build the module
echo "Building gemini_native.so..."
python3 setup_simple_gemini.py build_ext --inplace

# Check if build was successful
if [ -f gemini_native*.so ] || [ -f gemini_native*.pyd ]; then
    echo ""
    echo "=========================================="
    echo "✅ Gemini native module built successfully!"
    echo "=========================================="
    echo "Module location: $(pwd)/gemini_native*.so"
    echo ""
    echo "Available transport modes:"
    echo "  - ASIO (TCP): Always available"
    if [ $RDMA_FOUND -eq 1 ]; then
        echo "  - RDMA (InfiniBand): ✓ Available"
    else
        echo "  - RDMA (InfiniBand): ✗ Not available (libraries not found)"
    fi
    echo ""
    echo "Usage:"
    echo "  Basic (ASIO/TCP):  --use-gemini --use-gemini-optimized"
    if [ $RDMA_FOUND -eq 1 ]; then
        echo "  With RDMA:         --use-gemini --use-gemini-optimized --use-rdma"
    else
        echo "  With RDMA:         (Install RDMA libraries first)"
    fi
    echo ""
    echo "To test the module, run:"
    echo "  python3 -c 'import gemini_native; print(gemini_native.__doc__)'"
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
    echo "  3. RDMA libraries missing (optional): sudo apt-get install libibverbs-dev librdmacm-dev"
    exit 1
fi

