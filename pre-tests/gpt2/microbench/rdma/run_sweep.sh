#!/bin/bash
# RDMA QP scaling sweep: measures per-QP and aggregate bandwidth
# as QP count increases. Runs on two nodes/containers.
#
# Usage:
#   On node 0 (server):  ./run_sweep.sh server
#   On node 1 (client):  ./run_sweep.sh client <server_ip>
#
# Or single-node test (loopback):
#   ./run_sweep.sh loopback

set -e
cd "$(dirname "$0")"

MODE="${1:-loopback}"
SERVER_IP="${2:-127.0.0.1}"
BASE_PORT=12000
BLOCK_SIZE_MB=64
N_ITERS=5

# QP counts to sweep
QP_COUNTS=(1 2 4 8 16 32 64 128 256 392)

echo "============================================"
echo "RDMA QP Scaling Benchmark"
echo "block_size = ${BLOCK_SIZE_MB} MB"
echo "n_iters    = ${N_ITERS}"
echo "mode       = ${MODE}"
echo "============================================"
echo ""

printf "%-8s %-12s %-12s %-12s %-12s\n" "n_qps" "agg_bw_gbps" "per_qp_gbps" "per_qp_MBps" "time_s"
printf "%s\n" "----------------------------------------------------------------"

for n_qps in "${QP_COUNTS[@]}"; do
    if [ "$MODE" = "loopback" ]; then
        # Start server in background
        ./rdma_bench \
            --n_qps $n_qps \
            --block_size ${BLOCK_SIZE_MB}M \
            --n_iters $N_ITERS \
            --port $BASE_PORT \
            --server \
            --no_warmup \
            > /tmp/rdma_sweep_server.log 2>&1 &
        SERVER_PID=$!
        sleep 1

        # Run client
        CLIENT_OUT=$(./rdma_bench \
            --n_qps $n_qps \
            --block_size ${BLOCK_SIZE_MB}M \
            --n_iters $N_ITERS \
            --port $BASE_PORT \
            --ip 127.0.0.1 \
            --client \
            --no_warmup 2>&1)
        CLIENT_RC=$?
        wait $SERVER_PID 2>/dev/null || true  # Suppress "Terminated" message
    elif [ "$MODE" = "server" ]; then
        ./rdma_bench \
            --n_qps $n_qps \
            --block_size ${BLOCK_SIZE_MB}M \
            --n_iters $N_ITERS \
            --port $BASE_PORT \
            --server \
            --no_warmup
        # Server exits after client disconnects
        exit 0
    elif [ "$MODE" = "client" ]; then
        CLIENT_OUT=$(./rdma_bench \
            --n_qps $n_qps \
            --block_size ${BLOCK_SIZE_MB}M \
            --n_iters $N_ITERS \
            --port $BASE_PORT \
            --ip "$SERVER_IP" \
            --client \
            --no_warmup 2>&1)
        CLIENT_RC=$?
    fi

    if [ "$MODE" != "server" ]; then
        # Parse output
        AVG_BW=$(echo "$CLIENT_OUT" | grep "avg BW" | awk '{print $NF}')
        PER_QP=$(echo "$CLIENT_OUT" | grep "per-QP BW" | awk '{print $NF}')
        PER_QP_MBS=$(echo "$CLIENT_OUT" | grep "per-QP MB/s" | awk '{print $NF}')
        AVG_TIME=$(echo "$CLIENT_OUT" | grep "avg time" | awk '{print $NF}')

        if [ -n "$AVG_BW" ]; then
            printf "%-8d %-12s %-12s %-12s %-12s\n" \
                "$n_qps" "$AVG_BW" "$PER_QP" "$PER_QP_MBS" "$AVG_TIME"
        else
            printf "%-8d %-12s\n" "$n_qps" "FAILED"
            echo "$CLIENT_OUT" | tail -5
        fi
    fi

    sleep 1
done

echo ""
echo "Done."
