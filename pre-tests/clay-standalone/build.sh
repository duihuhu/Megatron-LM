#!/bin/bash
# Clay 测试程序编译脚本

set -e

echo "=========================================="
echo "   Clay 测试程序编译脚本"
echo "=========================================="

# 检查依赖
echo "检查依赖..."

# 检查 CMake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake，请先安装 cmake"
    exit 1
fi

# 检查 Ceph RADOS
if [ ! -f /usr/include/rados/buffer.h ] && [ ! -f /usr/local/include/rados/buffer.h ]; then
    echo "警告: 未找到 rados/buffer.h"
    echo "请安装 librados-dev (Ubuntu/Debian) 或 librados-devel (CentOS/RHEL)"
fi

# 创建构建目录
mkdir -p build
cd build

# 运行 CMake
echo ""
echo "运行 CMake..."
cmake ..

# 编译
echo ""
echo "编译测试程序..."
make

echo ""
echo "=========================================="
echo "编译完成！"
echo "运行测试: cd build && ./test_clay"
echo "=========================================="

