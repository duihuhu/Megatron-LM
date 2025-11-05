#!/usr/bin/env python3
#
# GPU XOR vs EC Performance Analysis Script
# 分析benchmark结果并生成对比报告
#

import csv
import sys
import glob
import os
from collections import defaultdict

def analyze_results(csv_file):
    """分析benchmark结果"""
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return
    
    results = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['Block Size (KB)'], row['Threads Per Block'], 
                   row['Task Num'])
            results[key].append({
                'type': row['Test Type'],
                'time': float(row['Avg Time (ms)']) if row['Avg Time (ms)'] != 'N/A' else None,
                'throughput': float(row['Throughput (MB/s)']) if row['Throughput (MB/s)'] != 'N/A' else None,
                'blocks': row['Blocks Per Grid']
            })
    
    print("=" * 80)
    print("GPU XOR vs EC Performance Analysis")
    print("=" * 80)
    print()
    
    # 按配置分组统计
    print("Performance Comparison by Configuration:")
    print("-" * 80)
    print(f"{'Block Size':<12} {'Threads':<8} {'Tasks':<8} {'XOR Time':<12} {'EC Time':<12} {'Speedup':<10} {'XOR TP':<12} {'EC TP':<12}")
    print("-" * 80)
    
    for key in sorted(results.keys()):
        block_size, threads, tasks = key
        config_results = results[key]
        
        xor_time = None
        ec_time = None
        xor_tp = None
        ec_tp = None
        
        for r in config_results:
            if r['type'] == 'XOR':
                xor_time = r['time']
                xor_tp = r['throughput']
            elif r['type'] == 'EC':
                ec_time = r['time']
                ec_tp = r['throughput']
        
        if xor_time and ec_time:
            speedup = ec_time / xor_time if xor_time > 0 else 0
            print(f"{block_size:<12} {threads:<8} {tasks:<8} "
                  f"{xor_time:<12.3f} {ec_time:<12.3f} {speedup:<10.2f}x "
                  f"{xor_tp or 'N/A':<12} {ec_tp or 'N/A':<12}")
    
    print()
    
    # 统计总体性能
    xor_times = []
    ec_times = []
    xor_throughputs = []
    ec_throughputs = []
    
    for config_results in results.values():
        for r in config_results:
            if r['type'] == 'XOR' and r['time']:
                xor_times.append(r['time'])
                if r['throughput']:
                    xor_throughputs.append(r['throughput'])
            elif r['type'] == 'EC' and r['time']:
                ec_times.append(r['time'])
                if r['throughput']:
                    ec_throughputs.append(r['throughput'])
    
    if xor_times and ec_times:
        print("Overall Statistics:")
        print("-" * 80)
        print(f"XOR Average Time: {sum(xor_times)/len(xor_times):.3f} ms")
        print(f"EC Average Time: {sum(ec_times)/len(ec_times):.3f} ms")
        print(f"EC is {sum(ec_times)/len(ec_times) / (sum(xor_times)/len(xor_times)):.2f}x slower than XOR")
        
        if xor_throughputs and ec_throughputs:
            print(f"XOR Average Throughput: {sum(xor_throughputs)/len(xor_throughputs):.2f} MB/s")
            print(f"EC Average Throughput: {sum(ec_throughputs)/len(ec_throughputs):.2f} MB/s")
    
    print()
    
    # 最佳配置推荐
    print("Best Configurations:")
    print("-" * 80)
    
    # 最快XOR配置
    best_xor = None
    best_xor_time = float('inf')
    for config_results in results.values():
        for r in config_results:
            if r['type'] == 'XOR' and r['time'] and r['time'] < best_xor_time:
                best_xor_time = r['time']
                best_xor = r
    
    # 最快EC配置
    best_ec = None
    best_ec_time = float('inf')
    for config_results in results.values():
        for r in config_results:
            if r['type'] == 'EC' and r['time'] and r['time'] < best_ec_time:
                best_ec_time = r['time']
                best_ec = r
    
    if best_xor:
        print(f"Best XOR Configuration: {best_xor_time:.3f} ms")
    if best_ec:
        print(f"Best EC Configuration: {best_ec_time:.3f} ms")
    
    print()

if __name__ == '__main__':
    # 查找最新的CSV文件
    csv_files = glob.glob('benchmark_results_*.csv')
    
    if not csv_files:
        print("Error: No benchmark results found. Please run benchmark_xor_vs_ec.sh first.")
        sys.exit(1)
    
    # 使用最新的文件
    latest_file = sorted(csv_files)[-1]
    print(f"Analyzing: {latest_file}")
    print()
    
    analyze_results(latest_file)

