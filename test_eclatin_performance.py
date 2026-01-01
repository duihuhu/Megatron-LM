#!/usr/bin/env python3
"""
Example script demonstrating how to use EC-LATIN performance statistics.

This script shows how to:
1. Reset performance statistics before a checkpoint
2. Run checkpoint operations
3. Print detailed performance breakdown
"""

import sys
import os

# Add the path to import eclatin_native module
# Adjust this path based on your build directory
# sys.path.insert(0, '/path/to/your/build/directory')

def example_usage():
    """
    Example of how to use the performance statistics in your training code.
    """
    print("=" * 60)
    print("EC-LATIN Performance Statistics Example")
    print("=" * 60)
    print()
    
    print("In your training code, you would typically:")
    print()
    print("1. Before checkpoint operation:")
    print("   eclatin_instance.reset_performance_stats()")
    print()
    print("2. Perform checkpoint operations:")
    print("   - Submit data to pipelines")
    print("   - Wait for completion")
    print()
    print("3. After checkpoint completes:")
    print("   eclatin_instance.print_performance_stats()")
    print()
    print("=" * 60)
    print()
    
    print("Example output you will see:")
    print()
    print("============================================================")
    print("ECLATIN PIPELINE PERFORMANCE BREAKDOWN")
    print("============================================================")
    print()
    print("========== Parity1_Send1 Statistics ==========")
    print("Queue Wait Time:        12.34 ms ( 5.00%)")
    print("Network Send Time:     180.56 ms (73.00%) - 450.23 MB/s")
    print("  Send Count: 100, Total Bytes: 81.92 MB")
    print("Buffer Release Time:     1.23 ms ( 0.50%)")
    print("Total Time:            247.35 ms (100.00%)")
    print("======================================")
    print()
    print("... (similar output for other 5 pipelines)")
    print()
    print("============================================================")
    print("AGGREGATE STATISTICS")
    print("============================================================")
    print("Total Queue Wait:       50.00 ms ( 4.00%)")
    print("Total Network Send:    800.00 ms (64.00%)")
    print("  Total Sent: 327.68 MB")
    print("Total Network Recv:    350.00 ms (28.00%)")
    print("  Total Received: 163.84 MB")
    print("Total XOR Compute:      45.00 ms ( 3.60%)")
    print("Aggregate Total:      1250.00 ms (100.00%)")
    print("============================================================")
    print()
    
    print("=" * 60)
    print("Key Metrics Explained:")
    print("=" * 60)
    print()
    print("1. Queue Wait Time:")
    print("   - Time spent waiting for tasks in the queue")
    print("   - High value indicates producer is slower than consumer")
    print()
    print("2. Network Send/Recv Time:")
    print("   - Actual time spent in network I/O operations")
    print("   - Includes throughput (MB/s) for performance analysis")
    print()
    print("3. XOR Compute Time:")
    print("   - Time spent calculating parity using XOR operations")
    print("   - Only present in recv_xor pipelines")
    print()
    print("4. Buffer Release Time:")
    print("   - Time spent managing buffer release queues")
    print("   - Usually very small (<1%)")
    print()
    print("=" * 60)
    print("Performance Optimization Tips:")
    print("=" * 60)
    print()
    print("- If Network Send/Recv > 70%: Network is the bottleneck")
    print("  → Consider faster network or compression")
    print()
    print("- If Queue Wait > 10%: Pipeline starvation")
    print("  → Increase batch size or parallelize data preparation")
    print()
    print("- If XOR Compute > 10%: CPU computation bottleneck")
    print("  → Consider using hardware acceleration (if available)")
    print()
    print("=" * 60)

if __name__ == "__main__":
    example_usage()

