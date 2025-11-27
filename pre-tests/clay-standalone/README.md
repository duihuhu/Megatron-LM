# Clay 独立提取代码

本目录包含从 NCBlob 项目中提取的 Clay 纠删码实现，所有代码保持原样，未做任何修改。

## 目录结构

```
clay-standalone/
├── clay/                          # Clay 核心实现
│   ├── erasure_code_clay.hh       # Clay 头文件 (143 行)
│   ├── erasure_code_clay.cc       # Clay 实现文件 (905 行)
│   └── erasure_code_clay_factory.cc  # Clay 工厂类 (14 行)
│
├── include/                       # 接口和基类定义
│   ├── erasure_code_intf.hpp      # 纠删码接口定义
│   ├── erasure_code.hh            # 基类定义
│   ├── erasure_code_factory.hpp   # 工厂接口
│   ├── exception.hpp              # 异常处理
│   ├── utils.hpp                  # 工具函数头文件
│   └── ec_intf.hh                 # EC 接口
│
├── jerasure/                      # Jerasure 实现 (Clay 的底层依赖)
│   ├── erasure_code_jerasure.hh   # Jerasure 头文件
│   ├── erasure_code_jerasure.cc   # Jerasure 实现
│   └── erasure_code_jerasure_factory.cc  # Jerasure 工厂
│
├── utils/                         # 工具函数
│   ├── erasure_code.cc            # 基类实现
│   ├── str_util.hh               # 字符串工具头文件
│   └── str_util.cc               # 字符串工具实现
│
└── clay-config/                   # Clay 配置文件
    └── *.bin                      # 各种 (k, m) 组合的配置文件
```

## 代码统计

- **Clay 核心代码**: 约 1062 行
  - `erasure_code_clay.hh`: 143 行
  - `erasure_code_clay.cc`: 905 行
  - `erasure_code_clay_factory.cc`: 14 行

- **依赖代码**: 约 2000+ 行
  - 接口和基类
  - Jerasure 实现
  - 工具函数

## 依赖关系

### 1. 接口层
- `ErasureCodeInterface` → `ErasureCode` → `ErasureCodeClay`
- 所有接口定义在 `include/` 目录

### 2. 运行时依赖
- **Jerasure**: Clay 内部使用 Jerasure 作为标量 MDS 码（MDS 和 PFT）
  - 在 `init()` 方法中创建两个 Jerasure 实例
  - `mds.erasure_code` - 用于 MDS 编码
  - `pft.erasure_code` - 用于 PFT 编码

### 3. 外部依赖
- **Ceph RADOS 库**: 
  - `rados/buffer.h` / `rados/buffer_fwd.h`
  - `ceph::bufferlist` - 缓冲区列表类型
  - `ceph::bufferptr` - 缓冲区指针类型

### 4. 标准库
- STL 容器: `<algorithm>`, `<vector>`, `<map>`, `<set>`
- 输入输出: `<iostream>`, `<ostream>`
- 其他: `<cassert>`, `<cstring>`

## 主要功能

### Clay 核心功能
1. **编码 (Encode)**
   - `encode_chunks()` - 编码块
   - 支持分层编码（Layered Encoding）

2. **解码 (Decode)**
   - `decode()` - 标准解码
   - `decode_chunks()` - 解码块
   - `decode_layered()` - 分层解码
   - `decode_erasures()` - 擦除解码
   - `decode_uncoupled()` - 解耦解码

3. **修复 (Repair)**
   - `repair()` - 修复丢失的块
   - `repair_one_lost_chunk()` - 修复单个丢失块
   - `minimum_to_repair()` - 计算修复所需的最小块
   - `is_repair()` - 判断是否为修复操作

## 使用说明

### 编译要求
1. **Ceph RADOS 开发库**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install librados-dev
   
   # CentOS/RHEL
   sudo yum install librados-devel
   ```

2. **Jerasure 库**
   - 需要 Jerasure C 库（在 `lib/ec/jerasure/` 目录）

3. **C++ 编译器**
   - 支持 C++20 标准

### 基本使用

```cpp
#include "clay/erasure_code_clay.hh"
#include "include/erasure_code_factory.hpp"

// 创建 Clay 编码器
ec::ErasureCodeProfile profile;
profile["k"] = "4";
profile["m"] = "2";
profile["d"] = "5";

ec::ErasureCodeClayFactory factory;
std::ostringstream errors;
auto encoder = factory.make(profile, errors);

if (encoder) {
    // 使用编码器进行编码/解码
    // ...
}
```

## 注意事项

1. **代码未修改**: 所有代码都是从原项目直接复制，未做任何修改
2. **依赖完整**: 包含所有必要的依赖文件
3. **Ceph 依赖**: 仍需要 Ceph RADOS 库，如需独立使用需要替换 `ceph::bufferlist`
4. **Jerasure 依赖**: Clay 内部依赖 Jerasure，必须保留或提供替代实现

## 文件清单

### Clay 核心文件
- `clay/erasure_code_clay.hh`
- `clay/erasure_code_clay.cc`
- `clay/erasure_code_clay_factory.cc`

### 接口文件
- `include/erasure_code_intf.hpp`
- `include/erasure_code.hh`
- `include/erasure_code_factory.hpp`
- `include/exception.hpp`
- `include/utils.hpp`
- `include/ec_intf.hh`

### Jerasure 文件
- `jerasure/erasure_code_jerasure.hh`
- `jerasure/erasure_code_jerasure.cc`
- `jerasure/erasure_code_jerasure_factory.cc`

### 工具文件
- `utils/erasure_code.cc`
- `utils/str_util.hh`
- `utils/str_util.cc`

## 提取日期

提取日期: 2024年（从 NCBlob 项目）

## 许可证

所有代码保持原项目的许可证（GNU Lesser General Public License v2.1+）

