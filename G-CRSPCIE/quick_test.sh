# 快速测试脚本 - 小规模测试
#!/bin/bash
#
# Quick Test: 快速验证测试脚本功能
# 仅测试几个配置，用于验证脚本是否正常工作
#

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ ! -f "./test_gpu_tensor" ]; then
    echo -e "${RED}Error: test_gpu_tensor not found. Please compile first: make test${NC}"
    exit 1
fi

echo "=========================================="
echo "Quick XOR vs EC Test (k=2, m=1)"
echo "=========================================="
echo ""

# 测试配置
BLOCK_SIZES=(64 256 512)
THREADS=(128 256)

for block_size in "${BLOCK_SIZES[@]}"; do
    for threads in "${THREADS[@]}"; do
        echo -e "${BLUE}Testing: Block=$block_size KB, Threads=$threads${NC}"
        
        # XOR测试
        echo "  XOR:"
        ./test_gpu_tensor --test 4 -k 2 -b $block_size --threads $threads 2>&1 | grep -E "(Average time|Throughput)" | sed 's/^/    /'
        
        # EC测试
        echo "  EC:"
        ./test_gpu_tensor --test 2 -k 2 -m 1 -w 4 -b $block_size --threads $threads 2>&1 | grep -E "(Encoding time|Throughput)" | sed 's/^/    /'
        
        echo ""
        sleep 1
    done
done

echo "Quick test completed!"

