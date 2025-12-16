#!/bin/bash
# Clean build script for EC-CHECK native C++ module

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building EC-CHECK native C++ module (clean build)..."

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

# Build in completely isolated environment
cd "$TEMP_DIR"

# Set completely clean environment
export PYTHONPATH=""
export PYTHONUSERBASE=""
unset PYTHONPATH
unset PYTHONUSERBASE

# Build the module
echo "Building in isolated environment..."
python3 setup_simple_eclatin.py build_ext --inplace

# Copy built module back
cd "$SCRIPT_DIR"
cp "$TEMP_DIR"/eclatin_native*.so . 2>/dev/null || true
cp "$TEMP_DIR"/eclatin_native*.pyd . 2>/dev/null || true

# Clean up
rm -rf "$TEMP_DIR"
<< EOF
# Test the built module
echo "Testing built module..."
python3 -c "
import sys
import os
sys.path.insert(0, os.getcwd())
try:
    import eclatin_native
    print('✅ EC-CHECK native module imported successfully')
    print('Available classes:', dir(eclatin_native))
    
    # Test creating an instance
    instance = eclatin_native.ECLATINNative(0, 1, 0)
    print('✅ EC-CHECK native module instance created successfully')
    
    # Test methods
    instance.reset_encoding_completion_flags()
    print('✅ EC-CHECK methods work correctly')
    
except ImportError as e:
    print('❌ Failed to import EC-CHECK native module:', e)
    exit(1)
except Exception as e:
    print('❌ Failed to create EC-CHECK native module instance:', e)
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "✅ EC-CHECK native module built successfully!"
echo "Module location: $(pwd)/eclatin_native*.so"
EOF