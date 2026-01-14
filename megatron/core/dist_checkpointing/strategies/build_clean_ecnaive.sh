#!/bin/bash
# Clean build script for EC-NAIVE native C++ module
# This script compiles the C++ extension for EC-NAIVE checkpointing
#
# Usage:
#   bash build_clean_ecnaive.sh

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building EC-NAIVE native C++ module..."

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

echo ""
echo "✅ EC-NAIVE native module built successfully!"
echo "Module location: $(pwd)/ecnaive_native*.so"
