#!/bin/bash
#
# Run ASIO latency test between two nodes
#
# This script helps coordinate running the server on one node
# and the client on another node for latency testing.
#
# Usage:
#   On server node: ./run_test.sh server <port>
#   On client node: ./run_test.sh client <server_host> <port> [packet_size] [total_size_mb|num_packets] [interval_us] [warmup_packets]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_BIN="$SCRIPT_DIR/asio_server"
CLIENT_BIN="$SCRIPT_DIR/asio_client"

# Default test parameters
DEFAULT_PORT=8888
# Packet size: 64KB is a common choice for network latency testing
# - Small packets (1-8KB): Better for latency measurement, more overhead
# - Medium packets (32-64KB): Balance between latency and throughput
# - Large packets (128KB+): Better for throughput testing, may fragment
DEFAULT_PACKET_SIZE=65536  # 64KB
DEFAULT_TOTAL_SIZE_MB=64
DEFAULT_INTERVAL_US=0
DEFAULT_WARMUP_PACKETS=3

print_usage() {
    echo "Usage:"
    echo "  Server: $0 server [port]"
    echo "  Client: $0 client <server_host> [port] [packet_size] [total_size_mb|num_packets] [interval_us] [warmup_packets]"
    echo ""
    echo "Examples:"
    echo "  # On node1 (server):"
    echo "  $0 server 8888"
    echo ""
    echo "  # On node2 (client) - specify total size in MB:"
    echo "  $0 client node1 8888 65536 64 0 3        # Send 64 MB with 64KB packets (default)"
    echo ""
    echo "  # Different packet sizes for different test purposes:"
    echo "  $0 client node1 8888 1024 64 0 3         # Small packets (1KB) - latency focused"
    echo "  $0 client node1 8888 65536 64 0 3        # Medium packets (64KB) - balanced"
    echo "  $0 client node1 8888 131072 64 0 3       # Large packets (128KB) - throughput focused"
    echo ""
    echo "  # Or specify number of packets directly:"
    echo "  $0 client node1 8888 65536 1000 0 3      # Send 1000 packets"
    echo ""
    echo "Default values:"
    echo "  port: $DEFAULT_PORT"
    echo "  packet_size: $DEFAULT_PACKET_SIZE bytes"
    echo "  total_size: $DEFAULT_TOTAL_SIZE_MB MB"
    echo "  interval_us: $DEFAULT_INTERVAL_US"
    echo "  warmup_packets: $DEFAULT_WARMUP_PACKETS (excluded from statistics)"
    echo ""
    echo "Note: The 4th parameter (after packet_size) can be either:"
    echo "  - A number < 1000: treated as total size in MB"
    echo "  - A number >= 1000: treated as number of packets"
    echo ""
    echo "  Examples:"
    echo "    100   -> Send 100 MB total"
    echo "    1000  -> Send 1000 packets"
    echo "    5000  -> Send 5000 packets"
}

# Check if binaries exist
check_binaries() {
    if [ ! -f "$SERVER_BIN" ]; then
        echo "Error: asio_server not found. Please run ./build.sh first."
        exit 1
    fi
    
    if [ ! -f "$CLIENT_BIN" ]; then
        echo "Error: asio_client not found. Please run ./build.sh first."
        exit 1
    fi
}

# Run server
run_server() {
    local port=${1:-$DEFAULT_PORT}
    
    check_binaries
    
    echo "========================================="
    echo "Starting ASIO Server"
    echo "========================================="
    echo "Port: $port"
    echo "Press Ctrl+C to stop"
    echo "========================================="
    echo ""
    
    "$SERVER_BIN" "$port"
}

# Calculate number of packets from total size in MB
calculate_num_packets() {
    local packet_size=$1
    local total_size_mb=$2
    # Calculate: total_bytes / packet_size
    local total_bytes=$((total_size_mb * 1024 * 1024))
    local num_packets=$((total_bytes / packet_size))
    echo $num_packets
}

# Run client
run_client() {
    if [ $# -lt 1 ]; then
        echo "Error: server host is required"
        print_usage
        exit 1
    fi
    
    local server_host=$1
    local port=${2:-$DEFAULT_PORT}
    local packet_size=${3:-$DEFAULT_PACKET_SIZE}
    local size_or_packets=${4:-$DEFAULT_TOTAL_SIZE_MB}
    local interval_us=${5:-$DEFAULT_INTERVAL_US}
    local warmup_packets=${6:-$DEFAULT_WARMUP_PACKETS}
    
    check_binaries
    
    # Determine if size_or_packets is MB or number of packets
    # If it's less than 1000, treat as MB; otherwise as number of packets
    local num_packets
    local total_size_mb
    
    if [ "$size_or_packets" -lt 1000 ] 2>/dev/null; then
        # Treat as MB
        total_size_mb=$size_or_packets
        num_packets=$(calculate_num_packets $packet_size $total_size_mb)
        if [ "$num_packets" -eq 0 ]; then
            echo "Error: Packet size ($packet_size bytes) is larger than total size ($total_size_mb MB)"
            exit 1
        fi
    else
        # Treat as number of packets
        num_packets=$size_or_packets
        # Calculate total size in MB for display (using integer arithmetic)
        local total_bytes=$((num_packets * packet_size))
        total_size_mb=$((total_bytes / 1048576))  # 1024 * 1024
        local remainder_bytes=$((total_bytes % 1048576))
        if [ "$remainder_bytes" -gt 0 ]; then
            # Add decimal part (approximate)
            local decimal_part=$((remainder_bytes * 100 / 1048576))
            total_size_mb="${total_size_mb}.${decimal_part}"
        fi
    fi
    
    # Calculate packet size in KB for display
    local packet_size_kb=$((packet_size / 1024))
    local packet_size_bytes_remainder=$((packet_size % 1024))
    
    echo "========================================="
    echo "Starting ASIO Client"
    echo "========================================="
    echo "Server: $server_host:$port"
    if [ "$packet_size_bytes_remainder" -eq 0 ]; then
        echo "Packet size: $packet_size bytes (${packet_size_kb} KB)"
    else
        echo "Packet size: $packet_size bytes (~${packet_size_kb} KB)"
    fi
    echo "Total size: ${total_size_mb} MB"
    echo "Number of packets: $num_packets"
    echo "Interval: $interval_us us"
    echo "Warmup packets (excluded): $warmup_packets"
    echo "========================================="
    echo ""
    
    "$CLIENT_BIN" "$server_host" "$port" "$packet_size" "$num_packets" "$interval_us" "$warmup_packets"
}

# Main
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

mode=$1
shift

case "$mode" in
    server)
        run_server "$@"
        ;;
    client)
        run_client "$@"
        ;;
    *)
        echo "Error: Unknown mode '$mode'"
        print_usage
        exit 1
        ;;
esac

