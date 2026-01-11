#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

# Build script for Gemini Replicas Native C++ Module with ASIO and RDMA
# 
# This script compiles the gemini_replicas_native.cpp file into a Python extension module
# using pybind11, Boost.ASIO, and InfiniBand verbs (for RDMA support).
#
# Requirements:
#   - Python 3.x with pybind11
#   - Boost libraries (for ASIO)
#   - InfiniBand verbs library (libibverbs-dev) for RDMA support
#   - C++17 compiler (g++ or clang++)
#
# Usage:
#   bash build_gemini_replicas.sh

set -e  # Exit on error

echo "========================================="
echo "Building Gemini Replicas Native Module"
echo "========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Detect Python version
PYTHON_CMD=${PYTHON_CMD:-python3}
echo "Using Python: $PYTHON_CMD"

# Get Python include path
PYTHON_INCLUDE=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_path('include'))")
echo "Python include path: $PYTHON_INCLUDE"

# Get pybind11 include path
PYBIND11_INCLUDE=$($PYTHON_CMD -c "import pybind11; print(pybind11.get_include())")
echo "pybind11 include path: $PYBIND11_INCLUDE"

# Get Python extension suffix
PYTHON_EXT_SUFFIX=$($PYTHON_CMD -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
echo "Python extension suffix: $PYTHON_EXT_SUFFIX"

# Detect compiler
if command -v g++ &> /dev/null; then
    CXX=g++
elif command -v clang++ &> /dev/null; then
    CXX=clang++
else
    echo "Error: No C++ compiler found (g++ or clang++)"
    exit 1
fi
echo "Using compiler: $CXX"

# Compiler flags
CXX_FLAGS="-O3 -Wall -shared -std=c++17 -fPIC"

# Boost library path (auto-detect or use environment variable)
if [ -z "$BOOST_ROOT" ]; then
    # Try common Boost locations
    if [ -d "/usr/include/boost" ]; then
        BOOST_INCLUDE="/usr/include"
    elif [ -d "/usr/local/include/boost" ]; then
        BOOST_INCLUDE="/usr/local/include"
    elif [ -d "$HOME/.local/include/boost" ]; then
        BOOST_INCLUDE="$HOME/.local/include"
    else
        echo "Warning: Boost not found in common locations. Set BOOST_ROOT if needed."
        BOOST_INCLUDE=""
    fi
else
    BOOST_INCLUDE="$BOOST_ROOT/include"
fi

if [ -n "$BOOST_INCLUDE" ]; then
    echo "Boost include path: $BOOST_INCLUDE"
    BOOST_FLAG="-I$BOOST_INCLUDE"
else
    echo "Warning: Using system default Boost path"
    BOOST_FLAG=""
fi

# Boost library linking (ASIO is header-only, but we need system libraries)
# Also link InfiniBand verbs library for RDMA support
BOOST_LIBS="-lboost_system -lpthread -libverbs"

# Output file
OUTPUT_FILE="gemini_replicas_native${PYTHON_EXT_SUFFIX}"

# Compile command
echo ""
echo "Compiling gemini_replicas_native.cpp..."
echo "Command:"
echo "$CXX $CXX_FLAGS \\"
echo "  -I$PYTHON_INCLUDE \\"
echo "  -I$PYBIND11_INCLUDE \\"
if [ -n "$BOOST_FLAG" ]; then
    echo "  $BOOST_FLAG \\"
fi
echo "  gemini_replicas_native.cpp \\"
echo "  -o $OUTPUT_FILE \\"
echo "  $BOOST_LIBS"
echo ""

$CXX $CXX_FLAGS \
    -I"$PYTHON_INCLUDE" \
    -I"$PYBIND11_INCLUDE" \
    $BOOST_FLAG \
    gemini_replicas_native.cpp \
    -o "$OUTPUT_FILE" \
    $BOOST_LIBS

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Build successful!"
    echo "Output: $OUTPUT_FILE"
    echo "========================================="
    echo ""
    echo "To test the module, run:"
    echo "  $PYTHON_CMD -c 'import gemini_replicas_native; print(gemini_replicas_native.__doc__)'"
else
    echo ""
    echo "========================================="
    echo "Build failed!"
    echo "========================================="
    exit 1
fi

