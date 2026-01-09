#!/bin/bash

# Example script to run RDMA throughput test

set -e

echo "RDMA Throughput Test - Example Script"
echo "========================================="
echo ""

# Configuration
SERVER_HOST="127.0.0.1"
PORT=12345
THREADS=4
DATA_SIZE_MB=10
ITERATIONS=1000
WARMUP=100

# Check if executable exists
if [ ! -f "build/rdma_throughput_test" ]; then
    if [ ! -f "rdma_throughput_test" ]; then
        echo "Error: Executable not found. Please run ./build.sh or make first."
        exit 1
    fi
    EXEC="./rdma_throughput_test"
else
    EXEC="./build/rdma_throughput_test"
fi

# Function to run server
run_server() {
    echo "Starting RDMA server on port $PORT with $THREADS threads..."
    $EXEC server \
        --port $PORT \
        --threads $THREADS \
        --size-mb $DATA_SIZE_MB
}

# Function to run client
run_client() {
    echo "Starting RDMA client connecting to $SERVER_HOST:$PORT..."
    $EXEC client \
        --host $SERVER_HOST \
        --port $PORT \
        --threads $THREADS \
        --size-mb $DATA_SIZE_MB \
        --iterations $ITERATIONS \
        --warmup $WARMUP
}

# Parse command line argument
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [server|client]"
    echo ""
    echo "For two-machine test:"
    echo "  Machine 1 (server): $0 server"
    echo "  Machine 2 (client): Edit SERVER_HOST in this script, then run: $0 client"
    echo ""
    echo "For local test:"
    echo "  Terminal 1: $0 server"
    echo "  Terminal 2: $0 client"
    echo ""
    echo "Note: RDMA requires InfiniBand or RoCE hardware"
    exit 1
fi

MODE=$1

case $MODE in
    server)
        run_server
        ;;
    client)
        run_client
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Use 'server' or 'client'."
        exit 1
        ;;
esac

