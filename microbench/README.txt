Network Throughput Microbenchmark Suite
========================================

This directory contains network throughput testing tools for both TCP/IP (ASIO) and RDMA.

Directory Structure
-------------------

microbench/
├── asio/     - TCP/IP based throughput test using Boost.Asio
└── rdma/     - RDMA microbenchmarks (libibverbs)
    ├── rdma_ec_bind_bench.cpp   - EC/Gemini-aligned bind + QP + SEND/RECV
    ├── bench_launcher.py        - torchrun 2-rank launcher (resolve_ip + all_gather)
    └── run_torch_dist_2node.sh  - example 2-node wrapper

================================================================================
ASIO (TCP/IP) Version
================================================================================

Location: microbench/asio/

Description:
  - Uses Boost.Asio for standard TCP/IP networking
  - Works on any network (Ethernet, InfiniBand with IP over IB, etc.)
  - Synchronous blocking I/O for simplicity
  - Each client thread establishes its own TCP connection

Prerequisites:
  - Boost libraries (system, program_options)
  - C++14 compiler

Build:
  cd asio
  ./build.sh
  # or
  make

Usage:
  # Server
  ./build/network_throughput_test --mode server --port 12345 --threads 4 --size-mb 10

  # Client
  ./build/network_throughput_test --mode client --host 192.168.1.100 --port 12345 \
      --threads 4 --size-mb 10 --iterations 1000 --warmup 100

================================================================================
RDMA Version
================================================================================

Location: microbench/rdma/

EC/Gemini-aligned bench (recommended for multi-NIC / Aliyun):
  - find_rdma_device_by_ip() from megatron/.../rdma_device_utils.h
  - GID index 1, active_mtu, QP exchange over TCP (same as ecnaive_native)
  - Chunked IBV_WR_SEND / IBV_WR_RECV (64 MiB chunks)
  - Python bench_launcher.py uses network_utils.resolve_ip (ECNAIVE_*, GEMINI_REPLICAS_*, ...)

Prerequisites:
  - RoCE / InfiniBand hardware, libibverbs-dev, netifaces (for resolve_ip NIC binding)
  - PyTorch, torchrun; 2 ranks (2 nodes x 1 GPU or 1 node x 2 processes)

Install:
  sudo apt-get install libibverbs-dev
  pip install netifaces

Build:
  cd microbench/rdma
  make rdma_ec_bind_bench
  # or: ./build.sh

Single-node (2 ranks, specify GPUs):
  cd microbench/rdma
  export ECNAIVE_LOCAL_RANK_NIC_0=eth0
  export ECNAIVE_LOCAL_RANK_NIC_1=eth0   # or eth1 for second NIC
  ./run_single_node.sh 0,1 --size-mb 64 --iterations 20

  # Same without dedicated script:
  SINGLE_NODE=1 CUDA_VISIBLE_DEVICES=0,1 ./run_torch_dist_2node.sh 0

  # Dist on CPU only (RDMA C++ bench still needs RoCE):
  ./run_single_node.sh --dist-backend gloo

Two-node usage:
  export ECNAIVE_LOCAL_RANK_NIC_0=eth0
  export MASTER_ADDR=10.0.0.62 MASTER_PORT=6000

  ./run_torch_dist_2node.sh 0 --size-mb 64 --iterations 50   # node 0
  ./run_torch_dist_2node.sh 1 --size-mb 64 --iterations 50   # node 1

  # Gemini Replicas prefix:
  MICROBENCH_PREFIX=GEMINI_REPLICAS ./run_torch_dist_2node.sh 0

Benchmark phases (default: all three):
  1. send  - peer_rank -> rank, report sender GiB/s (outbound on sender)
  2. recv  - same direction, report receiver GiB/s (inbound on receiver)
  3. duplex - concurrent send on sock_out + recv on sock_in (two TCP ports)

  Skip phases: --skip-send | --skip-recv | --skip-duplex

Direct binary (no torch), after exchanging IPs manually:
  # rank 0 (server):
  ./build/rdma_ec_bind_bench --bind-ip 10.0.0.1 --peer-ip 10.0.0.2 --rank 0 --peer-rank 1
  # rank 1 (client):
  ./build/rdma_ec_bind_bench --bind-ip 10.0.0.2 --peer-ip 10.0.0.1 --rank 1 --peer-rank 0
  Uses TCP ports P and P+1 (default P=19987).

Legacy rdma_throughput_test (device_list[0], not production bind logic):
  see pre-tests/gpt2/microbench/rdma/ if present

================================================================================
Common Features
================================================================================

Both versions support:
  - Multiple concurrent threads
  - Configurable data size (bytes or MB)
  - Warmup phase before actual testing
  - Configurable iterations per thread
  - Detailed throughput statistics

Parameters:
  --mode/-m        : server or client (ASIO), or first argument (RDMA)
  --host           : Server hostname/IP (client mode, default: 127.0.0.1)
  --port/-p        : Port number (default: 12345)
  --threads/-t     : Number of threads (default: 1)
  --size           : Data size in bytes (default: 65536)
  --size-mb        : Data size in MB (overrides --size)
  --iterations/-n  : Iterations per thread (client mode, default: 1000)
  --warmup/-w      : Warmup iterations per thread (client mode, default: 100)

================================================================================
Performance Comparison
================================================================================

Expected Performance (approximate):

ASIO (TCP/IP):
  - Throughput: 1-10 Gbps (depends on network card and CPU)
  - Latency: 10-100 microseconds
  - CPU usage: Medium to High

RDMA:
  - Throughput: 40-200 Gbps (depends on InfiniBand/RoCE hardware)
  - Latency: 1-5 microseconds
  - CPU usage: Very Low (kernel bypass)

================================================================================
Testing Scenarios
================================================================================

Small Message Test (4KB):
  ASIO:  --size 4096 --iterations 50000
  RDMA:  --size 4096 --iterations 50000

Medium Message Test (64KB):
  ASIO:  --size 65536 --iterations 10000
  RDMA:  --size 65536 --iterations 10000

Large Message Test (10MB):
  ASIO:  --size-mb 10 --iterations 1000
  RDMA:  --size-mb 10 --iterations 1000

Very Large Message Test (100MB):
  ASIO:  --size-mb 100 --iterations 500
  RDMA:  --size-mb 100 --iterations 500

Multi-threaded Test:
  --threads 4 --size-mb 10 --iterations 1000

================================================================================
Troubleshooting
================================================================================

ASIO Issues:
  - "Boost not found": Install boost-devel or libboost-all-dev
  - "Connection refused": Ensure server is running first
  - Low throughput: Check network bandwidth, try larger data sizes

RDMA Issues:
  - "No RDMA devices found": Check if InfiniBand/RoCE hardware is installed
  - "Failed to open device": Run 'ibv_devices' to check available devices
  - "Permission denied": May need to run with appropriate permissions
  - Driver issues: Ensure RDMA drivers are installed (OFED or inbox drivers)

Check RDMA Setup:
  ibv_devices          # List available RDMA devices
  ibv_devinfo          # Show detailed device information
  ibstat               # Show InfiniBand port status

================================================================================
Notes
================================================================================

1. Thread Behavior:
   - Each thread sends N iterations independently
   - Total operations = threads × iterations
   - Each client thread creates its own connection

2. Data Transfer:
   - ASIO: Bidirectional (send + receive echo)
   - RDMA: Unidirectional (WRITE only)

3. Warmup Phase:
   - Recommended to avoid cold start effects
   - Default 100 iterations usually sufficient

4. Network Requirements:
   - ASIO: Any TCP/IP network
   - RDMA: InfiniBand or RoCE network adapter required

5. For best results:
   - Use dedicated network interfaces
   - Disable interrupt coalescing if needed
   - Pin threads to cores for consistent results
   - Run tests multiple times and average results

================================================================================

