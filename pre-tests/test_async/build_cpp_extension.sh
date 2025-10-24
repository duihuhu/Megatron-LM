#!/bin/bash
#
# 编译 C++ 线程扩展
#
# 使用方法：
#   chmod +x build_cpp_extension.sh
#   ./build_cpp_extension.sh
#

set -e

echo "========================================="
echo "编译 C++ 线程扩展"
echo "========================================="

# 检查 pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "❌ 错误：pybind11 未安装"
    echo "请安装：pip install pybind11"
    exit 1
fi

echo "✅ pybind11 已安装"

# 获取 Python 和 pybind11 的 include 路径
PYTHON_INCLUDES=$(python3 -m pybind11 --includes)
EXTENSION_SUFFIX=$(python3-config --extension-suffix)

echo "Python includes: $PYTHON_INCLUDES"
echo "Extension suffix: $EXTENSION_SUFFIX"

# 编译
echo ""
echo "开始编译..."

g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $PYTHON_INCLUDES \
    cpp_thread_example.cpp \
    -o cpp_thread_example$EXTENSION_SUFFIX \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "生成文件: cpp_thread_example$EXTENSION_SUFFIX"
    echo ""
    echo "测试编译结果："
    python3 -c "import cpp_thread_example; print('  ✅ 导入成功')" || echo "  ❌ 导入失败"
    echo ""
    echo "运行示例："
    echo "  python3 example_cpp_usage.py"
else
    echo "❌ 编译失败"
    exit 1
fi

echo "========================================="

