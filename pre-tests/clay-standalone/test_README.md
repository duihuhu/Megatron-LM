# Clay 测试程序使用说明

## 概述

`test_clay.cc` 是一个完整的测试程序，用于验证 Clay 纠删码的编码和解码功能。

## 测试内容

### 测试 1: 基本编码和解码
- 创建 Clay 编码器
- 编码测试数据
- 完整解码（所有块都可用）
- 验证数据正确性

### 测试 2: 数据丢失恢复
- 编码数据
- 模拟数据块丢失（随机丢失 1-2 个块）
- 使用剩余块恢复丢失的数据
- 验证恢复的数据是否正确

### 测试 3: 不同数据大小
- 测试不同大小的数据（1KB, 4KB, 16KB, 64KB）
- 验证 Clay 在不同数据大小下的表现

## 编译要求

### 依赖库
1. **Ceph RADOS 开发库**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install librados-dev
   
   # CentOS/RHEL
   sudo yum install librados-devel
   ```

2. **Boost 库**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libboost-dev
   
   # CentOS/RHEL
   sudo yum install boost-devel
   ```

3. **Jerasure 库**
   - 需要 Jerasure C 库
   - 如果项目中有 `lib/ec/jerasure/` 目录，CMakeLists.txt 会自动使用
   - 否则需要系统安装的 Jerasure 库

4. **CMake** (版本 >= 3.12)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cmake
   
   # CentOS/RHEL
   sudo yum install cmake
   ```

5. **C++ 编译器** (支持 C++20)
   - GCC >= 10 或 Clang >= 10

## 编译步骤

### 方法 1: 使用 CMake

```bash
cd clay-standalone
mkdir build
cd build
cmake ..
make
```

### 方法 2: 手动编译（如果 CMake 配置有问题）

```bash
cd clay-standalone

# 设置包含路径和库路径（根据实际情况调整）
export CXXFLAGS="-I./include -I./clay -I./jerasure -I./utils -I/usr/include/rados -std=c++20"
export LDFLAGS="-lrados -lboost_system"

# 编译
g++ ${CXXFLAGS} test_clay.cc \
    clay/erasure_code_clay.cc \
    clay/erasure_code_clay_factory.cc \
    jerasure/erasure_code_jerasure.cc \
    jerasure/erasure_code_jerasure_factory.cc \
    utils/erasure_code.cc \
    utils/str_util.cc \
    ${LDFLAGS} \
    -o test_clay
```

## 运行测试

```bash
cd build  # 如果使用 CMake
./test_clay
```

## 预期输出

测试程序会输出详细的测试过程：

```
========================================
    Clay 纠删码功能测试程序
========================================

=== 测试 1: 基本编码和解码 (k=4, m=2, data_size=4096) ===
✓ 编码器创建成功
  - 数据块数 (k): 4
  - 校验块数 (m): 2
  - 总块数: 6
✓ 测试数据准备完成 (大小: 4096 字节)
✓ 编码成功，生成 6 个块
  - 每个块大小: XXX 字节
✓ 解码成功
✓ 数据验证通过！原始数据与恢复数据完全匹配

=== 测试 2: 数据丢失恢复 (k=4, m=2, 丢失 1 个块) ===
...
```

## 测试参数

可以在 `main()` 函数中修改测试参数：

```cpp
int k = 4;          // 数据块数
int m = 2;          // 校验块数
size_t data_size = 4096;  // 测试数据大小
```

## 故障排除

### 1. 找不到 rados/buffer.h
```
错误: fatal error: rados/buffer.h: No such file or directory
解决: 安装 librados-dev 或 librados-devel
```

### 2. 链接错误
```
错误: undefined reference to `jerasure_*`
解决: 确保 Jerasure 库已正确链接
      检查 CMakeLists.txt 中的 JERASURE_LIB 设置
```

### 3. 运行时错误
```
错误: 创建编码器失败
解决: 检查错误输出，可能是参数配置问题
      确保 k, m, d 参数合理（d 应该在 [k, k+m-1] 范围内）
```

## 测试结果验证

所有测试通过的标准：
1. ✓ 编码器创建成功
2. ✓ 编码成功
3. ✓ 解码成功
4. ✓ 数据验证通过

如果所有测试都通过，程序会返回 0，否则返回 1。

## 扩展测试

可以添加更多测试用例：
- 测试不同的 (k, m) 组合
- 测试更大的数据大小
- 测试丢失更多块的情况
- 测试边界情况（最小/最大数据大小）

