#!/bin/bash

# Advanced RDMA benchmark script with various test configurations

set -e

# Find executable
if [ -f "build/rdma_throughput_test" ]; then
    EXECUTABLE="build/rdma_throughput_test"
elif [ -f "rdma_throughput_test" ]; then
    EXECUTABLE="rdma_throughput_test"
else
    echo "Error: Executable not found. Please run ./build.sh or make first."
    exit 1
fi

# Default values
MODE=""
HOST="127.0.0.1"
PORT=12345
THREADS=1
DATA_SIZE_MB=10
ITERATIONS=1000
WARMUP=100

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
        --size-mb)
            DATA_SIZE_MB="$2"
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
            echo "  --size-mb         Data size in MB (default: 10)"
            echo "  --iterations, -n  Number of iterations per thread (default: 1000, client only)"
            echo "  --warmup, -w      Number of warmup iterations per thread (default: 100, client only)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  Server (single thread, 10MB):"
            echo "    $0 --mode server --port 12345 --threads 1 --size-mb 10"
            echo ""
            echo "  Server (4 threads for high concurrency):"
            echo "    $0 --mode server --port 12345 --threads 4 --size-mb 10"
            echo ""
            echo "  Client (test with 10MB, 4 threads):"
            echo "    $0 --mode client --host 192.168.1.100 --port 12345 --threads 4 --size-mb 10 --iterations 1000"
            echo ""
            echo "  Client (test with 100MB, 8 threads):"
            echo "    $0 --mode client --host 192.168.1.100 --port 12345 --threads 8 --size-mb 100 --iterations 500"
            echo ""
            echo "Note: RDMA requires InfiniBand or RoCE hardware"
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
CMD="$EXECUTABLE $MODE --port $PORT --threads $THREADS --size-mb $DATA_SIZE_MB"

if [ "$MODE" == "client" ]; then
    CMD="$CMD --host $HOST --iterations $ITERATIONS --warmup $WARMUP"
fi

# Display configuration
echo "========================================="
echo "RDMA Throughput Benchmark"
echo "========================================="
echo "Mode:       $MODE"
if [ "$MODE" == "client" ]; then
    echo "Server:     $HOST:$PORT"
fi
echo "Port:       $PORT"
echo "Threads:    $THREADS"
echo "Data Size:  $DATA_SIZE_MB MB"
if [ "$MODE" == "client" ]; then
    echo "Iterations: $ITERATIONS per thread"
    echo "Warmup:     $WARMUP per thread"
fi
echo "========================================="
echo ""

# Run the command
exec $CMD

