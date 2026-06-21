#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v g++ >/dev/null 2>&1; then
    CXX=${CXX:-g++}
elif command -v clang++ >/dev/null 2>&1; then
    CXX=${CXX:-clang++}
else
    echo "Error: no C++ compiler found" >&2
    exit 1
fi

if [ -d "/usr/local/cuda-12.3" ]; then
    CUDA_INCLUDE="-I/usr/local/cuda-12.3/targets/x86_64-linux/include"
    CUDA_LIB="-L/usr/local/cuda-12.3/targets/x86_64-linux/lib -lcudart"
elif [ -d "/usr/local/cuda/include" ]; then
    CUDA_INCLUDE="-I/usr/local/cuda/include"
    CUDA_LIB="-L/usr/local/cuda/lib64 -lcudart"
else
    CUDA_INCLUDE=""
    CUDA_LIB="-lcudart"
fi

echo "Building Gemini GDR send/recv microbench..."
echo "Compiler: $CXX"

"$CXX" -std=c++17 -O3 -Wall -Wextra -pthread \
    $CUDA_INCLUDE \
    gemini_gdr_sendrecv_bench.cpp \
    -o gemini_gdr_sendrecv_bench \
    -libverbs $CUDA_LIB

echo "Done: $SCRIPT_DIR/gemini_gdr_sendrecv_bench"
