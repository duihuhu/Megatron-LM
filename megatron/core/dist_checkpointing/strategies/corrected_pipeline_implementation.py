#!/usr/bin/env python3

"""
修正后的流水线实现 - 正确实现stage间的数据传输与数据写入并行
"""

import threading
import time
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class PipelineStage:
    """流水线阶段管理"""
    stage_id: int
    group_data: List[Any]
    gpu_to_cpu_done: threading.Event
    cpu_to_disk_done: threading.Event
    preloaded_data: Any = None
    error: Exception = None

class CorrectedPipelineImplementation:
    """修正后的流水线实现"""
    
    def __init__(self, num_groups: int = 3):
        self.num_groups = num_groups
        self.stages = []
        self.stage_locks = []  # 每个stage的锁，确保顺序执行
        
    def create_stages(self):
        """创建流水线阶段"""
        for i in range(self.num_groups):
            stage = PipelineStage(
                stage_id=i,
                group_data=f"group_{i}_data",
                gpu_to_cpu_done=threading.Event(),
                cpu_to_disk_done=threading.Event()
            )
            self.stages.append(stage)
            self.stage_locks.append(threading.Lock())
    
    def gpu_to_cpu_worker(self, stage: PipelineStage):
        """GPU→CPU传输工作线程 - 需要等待前一个stage完成"""
        try:
            print(f"[Stage {stage.stage_id}] 等待前一个stage的GPU→CPU传输完成...")
            
            # 关键：等待前一个stage的GPU→CPU传输完成
            if stage.stage_id > 0:
                prev_stage = self.stages[stage.stage_id - 1]
                prev_stage.gpu_to_cpu_done.wait()
                print(f"[Stage {stage.stage_id}] 前一个stage的GPU→CPU传输已完成，开始当前stage传输")
            
            print(f"[Stage {stage.stage_id}] 开始GPU→CPU传输...")
            time.sleep(1)  # 模拟传输时间
            
            stage.preloaded_data = f"cpu_data_{stage.stage_id}"
            stage.gpu_to_cpu_done.set()
            print(f"[Stage {stage.stage_id}] GPU→CPU传输完成 ✓")
            
        except Exception as e:
            stage.error = e
            stage.gpu_to_cpu_done.set()
            print(f"[Stage {stage.stage_id}] GPU→CPU传输失败: {e}")
    
    def cpu_to_disk_worker(self, stage: PipelineStage):
        """CPU→磁盘写入工作线程 - 需要等待当前stage的GPU→CPU和前一个stage的CPU→磁盘写入"""
        try:
            print(f"[Stage {stage.stage_id}] 等待当前stage的GPU→CPU传输完成...")
            
            # 等待当前stage的GPU→CPU传输完成
            stage.gpu_to_cpu_done.wait()
            print(f"[Stage {stage.stage_id}] 当前stage的GPU→CPU传输已完成")
            
            # 关键：等待前一个stage的CPU→磁盘写入完成
            if stage.stage_id > 0:
                prev_stage = self.stages[stage.stage_id - 1]
                print(f"[Stage {stage.stage_id}] 等待前一个stage的CPU→磁盘写入完成...")
                prev_stage.cpu_to_disk_done.wait()
                print(f"[Stage {stage.stage_id}] 前一个stage的CPU→磁盘写入已完成")
            
            print(f"[Stage {stage.stage_id}] 开始CPU→磁盘写入...")
            time.sleep(1.5)  # 模拟写入时间
            
            stage.cpu_to_disk_done.set()
            print(f"[Stage {stage.stage_id}] CPU→磁盘写入完成 ✓")
            
        except Exception as e:
            stage.error = e
            stage.cpu_to_disk_done.set()
            print(f"[Stage {stage.stage_id}] CPU→磁盘写入失败: {e}")
    
    def run_pipeline(self):
        """运行修正后的流水线"""
        print("=== 修正后的流水线实现 ===\n")
        
        # 创建阶段
        self.create_stages()
        print(f"创建了 {len(self.stages)} 个流水线阶段\n")
        
        # 启动所有GPU→CPU传输线程
        gpu_to_cpu_threads = []
        print("1. 启动GPU→CPU传输线程:")
        for stage in self.stages:
            thread = threading.Thread(target=self.gpu_to_cpu_worker, args=(stage,))
            thread.start()
            gpu_to_cpu_threads.append(thread)
            print(f"   - Stage {stage.stage_id}: GPU→CPU线程启动")
        
        # 启动所有CPU→磁盘写入线程
        cpu_to_disk_threads = []
        print("\n2. 启动CPU→磁盘写入线程:")
        for stage in self.stages:
            thread = threading.Thread(target=self.cpu_to_disk_worker, args=(stage,))
            thread.start()
            cpu_to_disk_threads.append(thread)
            print(f"   - Stage {stage.stage_id}: CPU→磁盘线程启动")
        
        # 等待所有线程完成
        print("\n3. 等待所有线程完成:")
        all_threads = gpu_to_cpu_threads + cpu_to_disk_threads
        for thread in all_threads:
            thread.join()
        
        print("\n=== 流水线执行完成 ===")
        
        # 验证结果
        print("\n4. 验证流水线结果:")
        for stage in self.stages:
            print(f"   - Stage {stage.stage_id}: GPU→CPU={stage.gpu_to_cpu_done.is_set()}, "
                  f"CPU→磁盘={stage.cpu_to_disk_done.is_set()}")

def demonstrate_corrected_pipeline():
    """演示修正后的流水线"""
    pipeline = CorrectedPipelineImplementation(num_groups=3)
    pipeline.run_pipeline()

def explain_corrected_logic():
    """解释修正后的逻辑"""
    print("\n=== 修正后的流水线逻辑解释 ===\n")
    
    print("正确的流水线时序:")
    print("""
    Stage 0: [GPU→CPU传输] → [CPU→磁盘写入........................]
    Stage 1:                [GPU→CPU传输] → [CPU→磁盘写入........]
    Stage 2:                                [GPU→CPU传输] → [CPU→磁盘写入]
    
    关键约束:
    1. Stage N的GPU→CPU传输必须等待Stage N-1的GPU→CPU传输完成
    2. Stage N的CPU→磁盘写入必须等待:
       - Stage N的GPU→CPU传输完成
       - Stage N-1的CPU→磁盘写入完成
    """)
    
    print("修正后的关键代码:")
    print("""
    def gpu_to_cpu_worker(self, stage):
        # 等待前一个stage的GPU→CPU传输完成
        if stage.stage_id > 0:
            prev_stage = self.stages[stage.stage_id - 1]
            prev_stage.gpu_to_cpu_done.wait()  # 关键同步点1
        
        # 执行当前stage的GPU→CPU传输
        # ...
        stage.gpu_to_cpu_done.set()
    
    def cpu_to_disk_worker(self, stage):
        # 等待当前stage的GPU→CPU传输完成
        stage.gpu_to_cpu_done.wait()  # 关键同步点2
        
        # 等待前一个stage的CPU→磁盘写入完成
        if stage.stage_id > 0:
            prev_stage = self.stages[stage.stage_id - 1]
            prev_stage.cpu_to_disk_done.wait()  # 关键同步点3
        
        # 执行当前stage的CPU→磁盘写入
        # ...
        stage.cpu_to_disk_done.set()
    """)

if __name__ == "__main__":
    # 演示修正后的流水线
    demonstrate_corrected_pipeline()
    
    # 解释修正后的逻辑
    explain_corrected_logic()

