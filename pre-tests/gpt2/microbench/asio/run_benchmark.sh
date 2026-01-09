#!/bin/bash

# Advanced benchmark script with various test configurations

set -e

EXECUTABLE="build/network_throughput_test"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found. Please run ./build.sh first."
    exit 1
fi

# Default values
MODE=""
HOST="127.0.0.1"
PORT=12345
THREADS=1
DATA_SIZE=65536
ITERATIONS=10000
WARMUP=1000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode|-m)
            MODE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --threads|-t)
            THREADS="$2"
            shift 2
            ;;
        --size|-s)
            DATA_SIZE="$2"
            shift 2
            ;;
        --iterations|-n)
            ITERATIONS="$2"
            shift 2
            ;;
        --warmup|-w)
            WARMUP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --mode [server|client] [options]"
            echo ""
            echo "Options:"
            echo "  --mode, -m        Mode: 'server' or 'client' (required)"
            echo "  --host            Server hostname or IP (default: 127.0.0.1)"
            echo "  --port, -p        Port number (default: 12345)"
            echo "  --threads, -t     Number of threads (default: 1)"
            echo "  --size, -s        Data size in bytes (default: 65536)"
            echo "  --iterations, -n  Number of iterations (default: 10000, client only)"
            echo "  --warmup, -w      Number of warmup iterations (default: 1000, client only)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  Server (single thread):"
            echo "    $0 --mode server --port 12345 --threads 1"
            echo ""
            echo "  Server (4 threads for high concurrency):"
            echo "    $0 --mode server --port 12345 --threads 4"
            echo ""
            echo "  Client (test with 64KB messages, 4 threads):"
            echo "    $0 --mode client --host 192.168.1.100 --port 12345 --threads 4 --size 65536 --iterations 10000"
            echo ""
            echo "  Client (test with 1MB messages, 8 threads):"
            echo "    $0 --mode client --host 192.168.1.100 --port 12345 --threads 8 --size 1048576 --iterations 5000"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Check if mode is specified
if [ -z "$MODE" ]; then
    echo "Error: Mode not specified. Use --mode server or --mode client"
    echo "Run '$0 --help' for usage information"
    exit 1
fi

# Build command
CMD="$EXECUTABLE --mode $MODE --port $PORT --threads $THREADS --size $DATA_SIZE"

if [ "$MODE" == "client" ]; then
    CMD="$CMD --host $HOST --iterations $ITERATIONS --warmup $WARMUP"
fi

# Display configuration
echo "========================================="
echo "Network Throughput Benchmark"
echo "========================================="
echo "Mode:       $MODE"
if [ "$MODE" == "client" ]; then
    echo "Server:     $HOST:$PORT"
fi
echo "Port:       $PORT"
echo "Threads:    $THREADS"
echo "Data Size:  $DATA_SIZE bytes ($(echo "scale=2; $DATA_SIZE/1024" | bc) KB)"
if [ "$MODE" == "client" ]; then
    echo "Iterations: $ITERATIONS"
    echo "Warmup:     $WARMUP"
fi
echo "========================================="
echo ""

# Run the command
exec $CMD

