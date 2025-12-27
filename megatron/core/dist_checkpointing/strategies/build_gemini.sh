#!/bin/bash
# Clean build script for Gemini native C++ module

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building Gemini Native C++ Module"
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

# Check for Boost (optional warning)
if ! python3 -c "import os; exit(0 if os.path.exists('/usr/include/boost/asio.hpp') or os.path.exists('/usr/local/include/boost/asio.hpp') or os.path.exists('/opt/homebrew/include/boost/asio.hpp') else 1)" 2>/dev/null; then
    echo "⚠️  Warning: Boost headers may not be found. If build fails, install Boost:"
    echo "   Ubuntu/Debian: sudo apt-get install libboost-all-dev"
    echo "   CentOS/RHEL: sudo yum install boost-devel"
    echo "   macOS: brew install boost"
else
    echo "✓ Boost headers found"
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
    echo "You can now use Gemini optimized checkpointing with:"
    echo "  --use-gemini --use-gemini-optimized"
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
    exit 1
fi

