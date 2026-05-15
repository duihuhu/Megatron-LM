#!/bin/bash
# Clean build script for FRCheck native C++ module (RDMA only, no ASIO fallback)
# Compiles frcheck_native.cpp → frcheck_native*.so

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building FRCheck Native C++ Module"
echo "  - RDMA (InfiniBand/RoCE): required"
echo "  - ISA-L (erasure coding): stub (placeholder)"
echo "=========================================="

echo "Checking dependencies..."

# Helper to filter strategies dir from sys.path (avoids local torch.py shadowing)
PY_FILTER_PATH="import sys,os; sys.path=[p for p in sys.path if 'dist_checkpointing/strategies' not in os.path.abspath(p)];"

# pybind11
if ! python3 -c "$PY_FILTER_PATH import pybind11" 2>/dev/null; then
    echo "❌ Error: pybind11 not found."
    echo "   pip install pybind11"
    exit 1
fi
echo "✓ pybind11 found"

# PyTorch
if ! python3 -c "$PY_FILTER_PATH import torch" 2>/dev/null; then
    echo "❌ Error: PyTorch not found."
    exit 1
fi
echo "✓ PyTorch found"

# RDMA (libibverbs)
RDMA_FOUND=0
if [ -f "/usr/include/infiniband/verbs.h" ] || [ -f "/usr/local/include/infiniband/verbs.h" ]; then
    if ldconfig -p 2>/dev/null | grep -q libibverbs || \
       [ -f "/usr/lib/x86_64-linux-gnu/libibverbs.so" ] || \
       [ -f "/usr/lib64/libibverbs.so" ]; then
        echo "✓ libibverbs found"
        RDMA_FOUND=1
    fi
fi

if [ $RDMA_FOUND -eq 0 ]; then
    echo "❌ Error: RDMA libraries (libibverbs) not found."
    echo "   Ubuntu/Debian: sudo apt-get install libibverbs-dev librdmacm-dev"
    echo "   CentOS/RHEL:   sudo yum install libibverbs-devel librdmacm-devel"
    exit 1
fi

# GDR (nvidia-peermem)
GDR_FOUND=0
if grep -q nvidia_peermem /proc/modules 2>/dev/null; then
    echo "✓ nvidia-peermem loaded (GPU Direct RDMA available)"
    GDR_FOUND=1
else
    echo "⚠  nvidia-peermem not loaded — GPU Direct RDMA will not be available"
    echo "   To enable: modprobe nvidia-peermem"
fi

echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -f frcheck_native.cpython-*.so
rm -f frcheck_native*.so
rm -rf build/
rm -rf *.egg-info/

# Build in isolated temp directory to avoid torch.py shadowing
TEMP_DIR=$(mktemp -d)
echo "Using isolated build directory: $TEMP_DIR"

cp setup_simple_frcheck.py "$TEMP_DIR/"
cp frcheck_native.cpp "$TEMP_DIR/"
cp rdma_device_utils.h "$TEMP_DIR/"

cd "$TEMP_DIR"

python3 -c "$PY_FILTER_PATH sys.argv=['build_frcheck','build_ext','--inplace']; exec(open('setup_simple_frcheck.py').read())"

# Copy built module back
cd "$SCRIPT_DIR"
cp "$TEMP_DIR"/frcheck_native.cpython-*.so . 2>/dev/null || true

# Clean up
rm -rf "$TEMP_DIR"

# Verify
if ls frcheck_native.cpython-*.so 1>/dev/null 2>&1; then
    SO_FILE=$(ls frcheck_native.cpython-*.so | head -1)
    echo ""
    echo "=========================================="
    echo "✅ FRCheck native module built successfully!"
    echo "=========================================="
    echo "Module: $(pwd)/$SO_FILE"

    # Quick smoke test (POA auto-generation, no RDMA needed)
    echo ""
    echo "Running smoke test..."
    python3 -c "$PY_FILTER_PATH import importlib.util; spec=importlib.util.spec_from_file_location('frcheck_native', '$SCRIPT_DIR/$SO_FILE'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); n=mod.FRCheckNative(4); n.compile_plans(0); print(f'  n={n.n()} num_stripes={n.num_stripes()}'); print(f'  GDR available: {mod.FRCheckNative.gdr_available()}'); print('  POA auto-generation: OK')" && echo "  Smoke test passed!" || echo "  ⚠ Smoke test failed"

    echo ""
    echo "Usage:"
    echo "  Training:  --use-frcheck --frcheck-n 4 --frcheck-table-dir <dir> --use-rdma"
    echo "  GDR mode:  (automatic if nvidia-peermem loaded)"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Build failed!"
    echo "=========================================="
    echo "Check the error messages above."
    exit 1
fi

echo ""
