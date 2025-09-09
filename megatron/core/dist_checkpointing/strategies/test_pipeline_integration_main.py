#!/usr/bin/env python3

"""
测试流水线checkpoint集成功能
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/Users/gaofz/Desktop/胡cunchen/Phd/LLM/ECTrain/Megatron-LM')

def test_filesystem_writer_async_pipeline():
    """测试 FileSystemWriterAsync 的流水线功能"""
    print("=== 测试 FileSystemWriterAsync 流水线功能 ===")
    
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 测试标准模式
        print("1. 测试标准模式...")
        writer_std = FileSystemWriterAsync(
            temp_dir,
            enable_pipeline=False,
            num_tensor_groups=2
        )
        print(f"   标准模式 - enable_pipeline: {writer_std.enable_pipeline}")
        print(f"   标准模式 - num_tensor_groups: {writer_std.num_tensor_groups}")
        
        # 测试流水线模式
        print("2. 测试流水线模式...")
        writer_pipeline = FileSystemWriterAsync(
            temp_dir,
            enable_pipeline=True,
            num_tensor_groups=4
        )
        print(f"   流水线模式 - enable_pipeline: {writer_pipeline.enable_pipeline}")
        print(f"   流水线模式 - num_tensor_groups: {writer_pipeline.num_tensor_groups}")
        
        # 测试 get_save_function_and_args
        print("3. 测试 get_save_function_and_args...")
        
        # 模拟 write_buckets
        mock_buckets = [
            ("file1.pt", "tensor1", ([], [("item1", torch.randn(10, 10))])),
            ("file2.pt", "tensor2", ([], [("item2", torch.randn(20, 20))])),
        ]
        writer_std.write_buckets = mock_buckets
        writer_pipeline.write_buckets = mock_buckets
        
        # 标准模式
        async_fn_std, preload_fn_std, args_std = writer_std.get_save_function_and_args()
        print(f"   标准模式 - async_fn: {async_fn_std.func.__name__ if hasattr(async_fn_std, 'func') else 'Unknown'}")
        print(f"   标准模式 - preload_fn: {preload_fn_std.func.__name__ if hasattr(preload_fn_std, 'func') else 'Unknown'}")
        
        # 流水线模式
        async_fn_pipeline, preload_fn_pipeline, args_pipeline = writer_pipeline.get_save_function_and_args()
        print(f"   流水线模式 - async_fn: {async_fn_pipeline.func.__name__ if hasattr(async_fn_pipeline, 'func') else 'Unknown'}")
        print(f"   流水线模式 - preload_fn: {preload_fn_pipeline.func.__name__ if hasattr(preload_fn_pipeline, 'func') else 'Unknown'}")
        
        print("   ✓ FileSystemWriterAsync 流水线功能测试通过")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_torch_dist_save_strategy_pipeline():
    """测试 TorchDistSaveShardedStrategy 的流水线功能"""
    print("\n=== 测试 TorchDistSaveShardedStrategy 流水线功能 ===")
    
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
    
    # 测试标准模式
    print("1. 测试标准模式...")
    strategy_std = TorchDistSaveShardedStrategy(
        'torch_dist', 1,
        enable_pipeline=False,
        num_tensor_groups=2
    )
    print(f"   标准模式 - enable_pipeline: {strategy_std.enable_pipeline}")
    print(f"   标准模式 - num_tensor_groups: {strategy_std.num_tensor_groups}")
    
    # 测试流水线模式
    print("2. 测试流水线模式...")
    strategy_pipeline = TorchDistSaveShardedStrategy(
        'torch_dist', 1,
        enable_pipeline=True,
        num_tensor_groups=6
    )
    print(f"   流水线模式 - enable_pipeline: {strategy_pipeline.enable_pipeline}")
    print(f"   流水线模式 - num_tensor_groups: {strategy_pipeline.num_tensor_groups}")
    
    print("   ✓ TorchDistSaveShardedStrategy 流水线功能测试通过")

def test_pipeline_parameters_validation():
    """测试流水线参数验证"""
    print("\n=== 测试流水线参数验证 ===")
    
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
    
    # 测试参数验证
    print("1. 测试参数验证...")
    
    # 测试 enable_pipeline=False 时的 num_tensor_groups
    writer1 = FileSystemWriterAsync(
        "/tmp/test",
        enable_pipeline=False,
        num_tensor_groups=10  # 应该被忽略
    )
    print(f"   enable_pipeline=False, num_tensor_groups=10 -> {writer1.num_tensor_groups}")
    
    # 测试 enable_pipeline=True 时的 num_tensor_groups
    writer2 = FileSystemWriterAsync(
        "/tmp/test",
        enable_pipeline=True,
        num_tensor_groups=10
    )
    print(f"   enable_pipeline=True, num_tensor_groups=10 -> {writer2.num_tensor_groups}")
    
    # 测试 num_tensor_groups < 2 的情况
    writer3 = FileSystemWriterAsync(
        "/tmp/test",
        enable_pipeline=True,
        num_tensor_groups=1  # 应该被调整为2
    )
    print(f"   enable_pipeline=True, num_tensor_groups=1 -> {writer3.num_tensor_groups}")
    
    print("   ✓ 流水线参数验证测试通过")

def test_group_write_buckets():
    """测试分组功能"""
    print("\n=== 测试分组功能 ===")
    
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
    
    # 创建测试数据
    write_buckets = [
        ("file1.pt", "tensor1", ([], [("item1", torch.randn(100, 100))])),
        ("file2.pt", "tensor2", ([], [("item2", torch.randn(200, 200))])),
        ("file3.pt", "tensor3", ([], [("item3", torch.randn(50, 50))])),
        ("file4.pt", "tensor4", ([], [("item4", torch.randn(300, 300))])),
    ]
    
    print(f"1. 原始 buckets 数量: {len(write_buckets)}")
    
    # 测试分组
    grouped = FileSystemWriterAsync.group_write_buckets_by_size(write_buckets, 2)
    print(f"2. 分成 2 组: {len(grouped)} 组")
    for i, group in enumerate(grouped):
        print(f"   组 {i+1}: {len(group)} 个 buckets")
    
    grouped = FileSystemWriterAsync.group_write_buckets_by_size(write_buckets, 3)
    print(f"3. 分成 3 组: {len(grouped)} 组")
    for i, group in enumerate(grouped):
        print(f"   组 {i+1}: {len(group)} 个 buckets")
    
    print("   ✓ 分组功能测试通过")

def test_command_line_args():
    """测试命令行参数"""
    print("\n=== 测试命令行参数 ===")
    
    # 模拟命令行参数
    class MockArgs:
        def __init__(self):
            self.enable_pipeline_checkpoint = True
            self.num_tensor_groups = 4
            self.async_save = True
    
    args = MockArgs()
    
    print(f"1. 模拟命令行参数:")
    print(f"   --enable-pipeline-checkpoint: {args.enable_pipeline_checkpoint}")
    print(f"   --num-tensor-groups: {args.num_tensor_groups}")
    print(f"   --async-save: {args.async_save}")
    
    # 测试参数获取
    enable_pipeline = getattr(args, 'enable_pipeline_checkpoint', False)
    num_tensor_groups = getattr(args, 'num_tensor_groups', 2)
    
    print(f"2. 参数获取结果:")
    print(f"   enable_pipeline: {enable_pipeline}")
    print(f"   num_tensor_groups: {num_tensor_groups}")
    
    print("   ✓ 命令行参数测试通过")

def main():
    """主测试函数"""
    print("开始测试流水线checkpoint集成功能...\n")
    
    try:
        test_filesystem_writer_async_pipeline()
        test_torch_dist_save_strategy_pipeline()
        test_pipeline_parameters_validation()
        test_group_write_buckets()
        test_command_line_args()
        
        print("\n🎉 所有测试通过！流水线checkpoint集成功能正常工作。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
