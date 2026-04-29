import torch
import torch.distributed as dist
import os
import sys

def init_process(rank, world_size, backend="nccl"):
    # 设置必要的环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    # 设置每个进程使用相同的 GPU
    # 节点0: rank 0,1 使用 GPU 0
    # 节点1: rank 2,3 使用 GPU 1
    device_id = rank // 2
    print(f"rank {rank} set device {device_id}")
    torch.cuda.set_device(device_id)
    
    # 初始化分布式环境
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cross_node_communication(rank, world_size):
    """跨节点通信：A(0)-C(2) 和 B(1)-D(3) 相互通信"""
    # 使用与init_process中相同的设备分配方式
    device_id = rank // 2
    device = torch.device(f'cuda:{device_id}')
    
    # 定义通信对
    # A(0) <-> C(2), B(1) <-> D(3)
    communication_pairs = {0: 2, 1: 3, 2: 0, 3: 1}
    
    if rank in communication_pairs:
        partner_rank = communication_pairs[rank]
        print(f"Process {rank} (Node {rank//2}, GPU {device_id}) will communicate with Process {partner_rank} (Node {partner_rank//2})")
        
        # 创建要发送的数据
        send_data = torch.tensor([rank, rank * 10, rank * 100, rank * 1000], device=device)
        print(f"Process {rank} prepared data to send: {send_data}")
        
        # 同步所有进程
        # dist.barrier()
        
        if rank in [0, 1]:  # A和B先发送
            # 发送数据到对应的伙伴进程
            dist.send(send_data, dst=partner_rank)
            print(f"Process {rank} sent {send_data} to process {partner_rank}")
            
            # 接收来自伙伴进程的数据
            recv_data = torch.zeros_like(send_data, device=device)
            dist.recv(recv_data, src=partner_rank)
            print(f"Process {rank} received {recv_data} from process {partner_rank}")
            
        else:  # C和D先接收
            # 接收来自伙伴进程的数据
            recv_data = torch.zeros_like(send_data, device=device)
            dist.recv(recv_data, src=partner_rank)
            print(f"Process {rank} received {recv_data} from process {partner_rank}")
            
            # 发送数据到对应的伙伴进程
            dist.send(send_data, dst=partner_rank)
            print(f"Process {rank} sent {send_data} to process {partner_rank}")

def all_gather_example(rank, world_size):
    """使用all-gather收集所有进程的数据"""
    # 使用与init_process中相同的设备分配方式
    device_id = rank // 2
    device = torch.device(f'cuda:{device_id}')
    
    print(f"Process {rank} starting all-gather communication...")
    
    # 每个进程创建自己的数据
    local_data = torch.tensor([rank, rank * 2, rank * 3, rank * 4], device=device)
    print(f"Process {rank} local data: {local_data}")
    
    # 同步所有进程
    dist.barrier()
    
    # 准备接收所有进程数据的列表
    gathered_data = [torch.zeros_like(local_data) for _ in range(world_size)]
    
    # 执行all-gather操作
    dist.all_gather(gathered_data, local_data)
    
    print(f"Process {rank} gathered all data:")
    for i, data in enumerate(gathered_data):
        print(f"  From process {i}: {data}")

def run(rank, world_size):
    init_process(rank, world_size)
    
    # 确保分布式环境已经初始化
    if dist.is_initialized():
        print(f"Process {rank} (Node {rank//2}) initialized successfully")
        
        # 等待所有进程初始化完成
        # dist.barrier()
        
        # 1. 跨节点点对点通信：A-C 和 B-D
        cross_node_communication(rank, world_size)
        
        # 等待所有进程完成点对点通信
        # dist.barrier()
        
        # 2. 使用all-gather收集所有进程数据
        # all_gather_example(rank, world_size)
        
        # 最终同步
        # dist.barrier()
        print(f"Process {rank} completed all operations")

if __name__ == "__main__":
    # 4个进程：A(0), B(1), C(2), D(3)
    # 节点0: A(0), B(1) - 使用 GPU 0
    # 节点1: C(2), D(3) - 使用 GPU 1
    world_size = 4  # 总共 4 个进程
    rank = int(sys.argv[1])
    run(rank, world_size)
