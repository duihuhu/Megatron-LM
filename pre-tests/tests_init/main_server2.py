import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os

# 初始化分布式进程组
def setup(rank, world_size):
    print(f"before Process {rank} initialized, {world_size} world_size")

    dist.init_process_group(
        backend='nccl',               # 使用 NCCL 后端（适用于多GPU）
        init_method='env://',          # 使用环境变量初始化
        world_size=world_size,        # 总进程数
        rank=rank                      # 当前进程的 rank
    )
    torch.cuda.set_device(rank)  # 设置每个进程使用的 GPU
    print(f"Process {rank} initialized.")

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 创建模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# 分布式训练函数
def train(rank, world_size):
    setup(rank, world_size)

    # model = SimpleModel().cuda(rank)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # data = torch.randn(100, 10).cuda(rank)
    # target = torch.randn(100, 10).cuda(rank)
    # dataset = TensorDataset(data, target)
    # train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # optimizer = optim.SGD(model.parameters(), lr=0.01)

    # for epoch in range(5):
    #     model.train()
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = nn.MSELoss()(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

# 启动分布式训练
def main():
    world_size = 4  # 总进程数为 4，因为每台机器有 2 个进程，每个进程使用一个 GPU
    rank = 0  # 主节点的 rank（Server 1）

    # 设置主节点的地址和端口
    os.environ['MASTER_ADDR'] = '10.156.154.36'  # 主节点的 IP 地址
    os.environ['MASTER_PORT'] = '6000'       # 端口号

    # 使用 spawn 启动每个进程，这里每个机器启动 2 个进程
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=2, join=True)

if __name__ == '__main__':
    main()
