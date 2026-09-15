import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os

# Initialize the distributed process group
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',               # Use the NCCL backend (for multiple GPUs)
        init_method='env://',          # Initialize using environment variables
        world_size=world_size,        # Total number of processes
        rank=rank                      # Rank of the current process
    )
    torch.cuda.set_device(rank)  # Set the GPU used by each process
    print(f"Process {rank} initialized.")

# Clean up the distributed environment
def cleanup():
    dist.destroy_process_group()

# Create the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Distributed training function
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

# Start distributed training
def main():
    world_size = 2  # Two processes launched on separate machines, one GPU per machine
    rank = 0  # Primary node rank (Server 1)

    # Set the primary node address and port
    os.environ['MASTER_ADDR'] = '10.156.154.36'  # Primary node IP address
    os.environ['MASTER_PORT'] = '6000'       # Port number

    train(rank, world_size)

if __name__ == '__main__':
    main()
