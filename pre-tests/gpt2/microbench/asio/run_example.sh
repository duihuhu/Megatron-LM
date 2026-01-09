#!/bin/bash

# Example script to run network throughput test
# This demonstrates running both server and client

set -e

echo "Network Throughput Test - Example Script"
echo "========================================="
echo ""

# Configuration
SERVER_HOST="127.0.0.1"
PORT=12345
THREADS=4
DATA_SIZE=65536  # 64 KB
ITERATIONS=10000
WARMUP=1000

# Check if executable exists
if [ ! -f "build/network_throughput_test" ]; then
    echo "Error: Executable not found. Please run ./build.sh first."
    exit 1
fi

# Function to run server
run_server() {
    echo "Starting server on port $PORT with $THREADS threads..."
    ./build/network_throughput_test \
        --mode server \
        --port $PORT \
        --threads $THREADS \
        --size $DATA_SIZE
}

# Function to run client
run_client() {
    echo "Starting client connecting to $SERVER_HOST:$PORT..."
    ./build/network_throughput_test \
        --mode client \
        --host $SERVER_HOST \
        --port $PORT \
        --threads $THREADS \
        --size $DATA_SIZE \
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

