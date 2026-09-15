#!/bin/bash
#
# Build the C++ thread extension
#
# Usage:
#   chmod +x build_cpp_extension.sh
#   ./build_cpp_extension.sh
#

set -e

echo "========================================="
echo "Build the C++ thread extension"
echo "========================================="

# Check pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "❌ Error: pybind11 is not installed"
    echo "Install it with: pip install pybind11"
    exit 1
fi

echo "✅ pybind11 is installed"

# Get the Python and pybind11 include paths
PYTHON_INCLUDES=$(python3 -m pybind11 --includes)
EXTENSION_SUFFIX=$(python3-config --extension-suffix)

echo "Python includes: $PYTHON_INCLUDES"
echo "Extension suffix: $EXTENSION_SUFFIX"

# build
echo ""
echo "Starting build..."

g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $PYTHON_INCLUDES \
    cpp_thread_example.cpp \
    -o cpp_thread_example$EXTENSION_SUFFIX \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Build succeeded!"
    echo ""
    echo "Generated file: cpp_thread_example$EXTENSION_SUFFIX"
    echo ""
    echo "Test the compiled extension: "
    python3 -c "import cpp_thread_example; print('  ✅ Import succeeded')" || echo "  ❌ Import failed"
    echo ""
    echo "Run the example: "
    echo "  python3 example_cpp_usage.py"
else
    echo "❌ Build failed"
    exit 1
fi

echo "========================================="

