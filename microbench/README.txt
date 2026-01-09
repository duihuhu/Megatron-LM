Network Throughput Microbenchmark Suite
========================================

This directory contains network throughput testing tools for both TCP/IP (ASIO) and RDMA.

Directory Structure
-------------------

microbench/
├── asio/     - TCP/IP based throughput test using Boost.Asio
└── rdma/     - RDMA based throughput test using libibverbs

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

Description:
  - Uses libibverbs for RDMA operations
  - Requires InfiniBand or RoCE hardware
  - Zero-copy data transfer with kernel bypass
  - Much lower latency and higher throughput than TCP/IP
  - Uses RDMA WRITE operations

Prerequisites:
  - InfiniBand or RoCE network hardware
  - libibverbs-dev package installed
  - RDMA drivers configured
  - C++14 compiler

Install RDMA libraries (Ubuntu/Debian):
  sudo apt-get install libibverbs-dev

Build:
  cd rdma
  ./build.sh
  # or
  make

Usage:
  # Server
  ./build/rdma_throughput_test server --port 12345 --threads 4 --size-mb 10

  # Client
  ./build/rdma_throughput_test client --host 192.168.1.100 --port 12345 \
      --threads 4 --size-mb 10 --iterations 1000 --warmup 100

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

