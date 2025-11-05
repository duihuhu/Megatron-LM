#!/bin/bash
#
# GPU XOR vs EC Performance Benchmark Script
# 测试纯GPU计算中XOR和EC编码的性能对比
# 编码参数: k=2, m=1
#

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查测试程序是否存在
if [ ! -f "./test_gpu_tensor" ]; then
    echo -e "${RED}Error: test_gpu_tensor not found. Please compile first: make test${NC}"
    exit 1
fi

# 输出文件
OUTPUT_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).csv"
LOG_FILE="benchmark_log_$(date +%Y%m%d_%H%M%S).log"

# 创建CSV头
echo "Test Type,Block Size (KB),Threads Per Block,Blocks Per Grid,Task Num,Avg Time (ms),Throughput (MB/s)" > $OUTPUT_FILE

# 打印头部信息
echo "=========================================="
echo "GPU XOR vs EC Performance Benchmark"
echo "=========================================="
echo "Encoding Parameters: k=2, m=1"
echo "Output CSV: $OUTPUT_FILE"
echo "Log File: $LOG_FILE"
echo "=========================================="
echo ""

# 测试配置
BLOCK_SIZES=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)  # KB
THREADS_PER_BLOCK=(16 32 64 128 256 512 1024)      # 线程数
TASK_NUMS=(1 2 4)                       # 流水线深度
BLOCKS_PER_GRID=(0 16 32 64 128 256 512 1024 2048)  # Blocks per grid (0=auto)

# 运行单个测试
run_test() {
    local test_type=$1
    local block_size_kb=$2
    local threads=$3
    local blocks=$4
    local task_num=$5
    
    echo -e "${BLUE}Running: $test_type | Block=$block_size_kb KB | Threads=$threads | Blocks=$blocks | Tasks=$task_num${NC}" | tee -a $LOG_FILE
    
    # 根据测试类型选择参数
    if [ "$test_type" == "XOR" ]; then
        # XOR测试
        result=$(./test_gpu_tensor --test 4 -k 2 -b $block_size_kb --threads $threads --blocks $blocks 2>&1 | tee -a $LOG_FILE)
    else
        # EC编码测试
        result=$(./test_gpu_tensor --test 2 -k 2 -m 1 -w 8 -b $block_size_kb -t $task_num --threads $threads --blocks $blocks 2>&1 | tee -a $LOG_FILE)
    fi
    
    # 提取时间和吞吐量
    avg_time=$(echo "$result" | grep -E "(Average time|Encoding time)" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    throughput=$(echo "$result" | grep "Throughput" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    if [ -z "$avg_time" ]; then
        avg_time="N/A"
    fi
    if [ -z "$throughput" ]; then
        throughput="N/A"
    fi
    
    # 写入CSV
    echo "$test_type,$block_size_kb,$threads,$blocks,$task_num,$avg_time,$throughput" >> $OUTPUT_FILE
    
    echo -e "${GREEN}  Result: Time=$avg_time ms, Throughput=$throughput MB/s${NC}"
    echo ""
}

# 测试1: 固定线程数，变化块大小（XOR vs EC）
echo "=========================================="
echo "Test 1: Variable Block Size"
echo "Threads per block: 128 (fixed)"
echo "Task num: 1 (fixed)"
echo "=========================================="
for block_size in "${BLOCK_SIZES[@]}"; do
    # XOR测试
    run_test "XOR" $block_size 128 0 1
    
    # EC编码测试
    run_test "EC" $block_size 128 0 1
    
    sleep 1  # 短暂休息，避免GPU过热
done

# 测试2: 固定块大小，变化线程数（XOR vs EC）
echo "=========================================="
echo "Test 2: Variable Threads Per Block"
echo "Block size: 256 MB (fixed)"
echo "Task num: 1 (fixed)"
echo "=========================================="
for threads in "${THREADS_PER_BLOCK[@]}"; do
    # XOR测试
    #run_test "XOR" 524288 $threads 0 1
    
    # EC编码测试
    #run_test "EC" 524288 $threads 0 1
    
    sleep 1
done

# 测试3: 固定配置，变化流水线深度（仅EC）
#echo "=========================================="
#echo "Test 3: Variable Pipeline Depth (EC only)"
#echo "Block size: 256 KB (fixed)"
#echo "Threads per block: 128 (fixed)"
#echo "=========================================="
#for task_num in "${TASK_NUMS[@]}"; do
#    run_test "EC" 256 128 0 $task_num
#    sleep 1
#done

# 测试4: 大块大小性能测试（XOR vs EC）
#echo "=========================================="
#echo "Test 4: Large Block Size Performance"
#echo "Threads per block: 256 (fixed)"
#cho "Task num: 1 (fixed)"
#echo "=========================================="
#large_blocks=(1024 2048 4096 8192 16384 32768 65536)
#for block_size in "${large_blocks[@]}"; do
#    run_test "XOR" $block_size 256 0 1
#    run_test "EC" $block_size 256 0 1
#    sleep 1
#done

# 测试5: 最优配置对比（XOR vs EC）
echo "=========================================="
echo "Test 5: Optimal Configuration Comparison"
echo "=========================================="
optimal_configs=(
    "1024 256 0 1"   # 标准配置
    "2048 256 0 1"  # 大块配置
    "4096 256 0 1"  # 大块配置
    "8192 256 0 1"  # 大块配置
    "16384 256 0 1"  # 超大块配置
    "32768 256 0 1"  # 超大块配置
    "65536 256 0 1"  # 超大块配置
)

for config in "${optimal_configs[@]}"; do
    read block_size threads blocks task_num <<< "$config"
    #run_test "XOR" $block_size $threads $blocks $task_num
    #run_test "EC" $block_size $threads $blocks $task_num
    #sleep 1
done

# 测试6: 固定块大小和线程数，变化Blocks Per Grid（XOR vs EC）
echo "=========================================="
echo "Test 6: Variable Blocks Per Grid"
echo "Block size: 256 MB (262144 KB, fixed)"
echo "Threads per block: 128 (fixed)"
echo "Task num: 1 (fixed)"
echo "=========================================="
for blocks in "${BLOCKS_PER_GRID[@]}"; do
    # XOR测试
    #run_test "XOR" 262144 128 $blocks 1
    
    # EC编码测试
    #run_test "EC" 262144 128 $blocks 1
    
    sleep 1  # 短暂休息，避免GPU过热
done

# 生成汇总报告
echo ""
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="
echo "Results saved to: $OUTPUT_FILE"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Quick Summary:"
echo "----------------------------------------"

# 计算平均性能提升
python3 << EOF
import csv
import sys
import glob

try:
    # 找到最新的CSV文件
    files = glob.glob('benchmark_results_*.csv')
    if not files:
        print("No results file found")
        sys.exit(0)
    latest_file = sorted(files)[-1]
    
    with open(latest_file, 'r') as f:
        reader = csv.DictReader(f)
        xor_times = []
        ec_times = []
        
        for row in reader:
            if row['Test Type'] == 'XOR' and row['Avg Time (ms)'] != 'N/A':
                xor_times.append(float(row['Avg Time (ms)']))
            elif row['Test Type'] == 'EC' and row['Avg Time (ms)'] != 'N/A':
                ec_times.append(float(row['Avg Time (ms)']))
        
        if xor_times and ec_times:
            avg_xor = sum(xor_times) / len(xor_times)
            avg_ec = sum(ec_times) / len(ec_times)
            speedup = avg_ec / avg_xor if avg_xor > 0 else 0
            print(f"Average XOR time: {avg_xor:.3f} ms")
            print(f"Average EC time: {avg_ec:.3f} ms")
            print(f"EC is {speedup:.2f}x slower than XOR")
except Exception as e:
    print(f"Summary generation failed: {e}")
EOF

echo ""
echo "To analyze results, run:"
echo "  cat $OUTPUT_FILE | column -t -s,"
echo ""

