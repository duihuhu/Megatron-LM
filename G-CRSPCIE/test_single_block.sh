#!/bin/bash
#
# Single-Block GPU Performance Test Script
# 测试GPU在单线程块（minimal resources）下的EC和XOR性能
# 用于评估GPU在pretrain时匀出来的性能
#

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查测试程序是否存在
if [ ! -f "./test_gpu_tensor" ]; then
    echo -e "${RED}Error: test_gpu_tensor not found. Please compile first: make test${NC}"
    exit 1
fi

# 输出文件
OUTPUT_FILE="single_block_results_$(date +%Y%m%d_%H%M%S).csv"
LOG_FILE="single_block_log_$(date +%Y%m%d_%H%M%S).log"

# 创建CSV头
echo "Test Type,Block Size (KB),Avg Time (ms),Throughput (MB/s),Notes" > $OUTPUT_FILE

# 打印头部信息
echo "=========================================="
echo "Single-Block GPU Performance Test"
echo "=========================================="
echo "Purpose: Test GPU performance with minimal resources"
echo "         (simulates GPU during pretraining)"
echo "Encoding Parameters: k=2, m=1"
echo "Config: 1 block, 32 threads (warp size)"
echo "Output CSV: $OUTPUT_FILE"
echo "Log File: $LOG_FILE"
echo "=========================================="
echo ""

# 测试配置
BLOCK_SIZES=(64 128 256 512 1024 2048 4096 8192)  # KB

# 运行单个测试
run_test() {
    local test_type=$1
    local block_size_kb=$2
    local test_num=$3
    
    echo -e "${BLUE}Running: $test_type | Block=$block_size_kb KB${NC}" | tee -a $LOG_FILE
    
    # 根据测试类型选择参数
    if [ "$test_type" == "Single-Block XOR" ]; then
        result=$(./test_gpu_tensor --test 6 -k 2 -b $block_size_kb 2>&1 | tee -a $LOG_FILE)
    else
        result=$(./test_gpu_tensor --test 5 -k 2 -m 1 -w 4 -b $block_size_kb 2>&1 | tee -a $LOG_FILE)
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
    echo "$test_type,$block_size_kb,$avg_time,$throughput,Minimal resources (1 block, 32 threads)" >> $OUTPUT_FILE
    
    echo -e "${GREEN}  Result: Time=$avg_time ms, Throughput=$throughput MB/s${NC}"
    echo ""
}

# 测试：不同块大小下的单线程块性能
echo "=========================================="
echo "Testing Single-Block Performance"
echo "=========================================="
for block_size in "${BLOCK_SIZES[@]}"; do
    echo "Testing block size: ${block_size} KB"
    
    # Single-block XOR测试
    run_test "Single-Block XOR" $block_size 6
    
    # Single-block EC测试
    run_test "Single-Block EC" $block_size 5
    
    sleep 1  # 短暂休息
done

# 生成汇总报告
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Results saved to: $OUTPUT_FILE"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Quick Summary:"
echo "----------------------------------------"

# 计算平均性能
python3 << EOF
import csv
import sys
import glob

try:
    files = glob.glob('single_block_results_*.csv')
    if not files:
        print("No results file found")
        sys.exit(0)
    latest_file = sorted(files)[-1]
    
    with open(latest_file, 'r') as f:
        reader = csv.DictReader(f)
        xor_times = []
        ec_times = []
        xor_tp = []
        ec_tp = []
        
        for row in reader:
            if 'XOR' in row['Test Type'] and row['Avg Time (ms)'] != 'N/A':
                xor_times.append(float(row['Avg Time (ms)']))
                if row['Throughput (MB/s)'] != 'N/A':
                    xor_tp.append(float(row['Throughput (MB/s)']))
            elif 'EC' in row['Test Type'] and row['Avg Time (ms)'] != 'N/A':
                ec_times.append(float(row['Avg Time (ms)']))
                if row['Throughput (MB/s)'] != 'N/A':
                    ec_tp.append(float(row['Throughput (MB/s)']))
        
        if xor_times and ec_times:
            avg_xor = sum(xor_times) / len(xor_times)
            avg_ec = sum(ec_times) / len(ec_times)
            print(f"Single-Block XOR Average Time: {avg_xor:.3f} ms")
            print(f"Single-Block EC Average Time: {avg_ec:.3f} ms")
            if avg_xor > 0:
                print(f"EC is {avg_ec/avg_xor:.2f}x slower than XOR")
            
            if xor_tp and ec_tp:
                avg_xor_tp = sum(xor_tp) / len(xor_tp)
                avg_ec_tp = sum(ec_tp) / len(ec_tp)
                print(f"Single-Block XOR Average Throughput: {avg_xor_tp:.2f} MB/s")
                print(f"Single-Block EC Average Throughput: {avg_ec_tp:.2f} MB/s")
except Exception as e:
    print(f"Summary generation failed: {e}")
EOF

echo ""
echo "To analyze results, run:"
echo "  cat $OUTPUT_FILE | column -t -s,"
echo ""
echo "Note: These results represent GPU performance when using minimal resources"
echo "      (1 thread block with 32 threads), which simulates GPU during pretraining."



