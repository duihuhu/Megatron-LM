#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Building rdma_bench..."
g++ -std=c++17 -O2 -o rdma_bench rdma_bench.cpp -libverbs -lpthread
echo "Done: $(pwd)/rdma_bench"
