import torch
import torch.distributed as dist
import os
import sys

def init_process(rank, world_size, backend="nccl"):
    # Set required environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    # Assign the same GPU to each process
    # Node 0: rank 0,1 use GPU 0
    # Node 1: rank 2,3 use GPU 1
    device_id = rank // 2
    print(f"rank {rank} set device {device_id}")
    torch.cuda.set_device(device_id)

    # Initialize distributed environment
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cross_node_communication(rank, world_size):
    """Cross-node communication: A(0)-C(2) and B(1)-D(3)"""
    # Use the same device assignment as init_process
    device_id = rank // 2
    device = torch.device(f'cuda:{device_id}')

    # Define communication pairs
    # A(0) <-> C(2), B(1) <-> D(3)
    communication_pairs = {0: 2, 1: 3, 2: 0, 3: 1}

    if rank in communication_pairs:
        partner_rank = communication_pairs[rank]
        print(f"Process {rank} (Node {rank//2}, GPU {device_id}) will communicate with Process {partner_rank} (Node {partner_rank//2})")

        # Create data to send
        send_data = torch.tensor([rank, rank * 10, rank * 100, rank * 1000], device=device)
        print(f"Process {rank} prepared data to send: {send_data}")

        # Synchronize all processes
        # dist.barrier()

        if rank in [0, 1]:  # A and B send first
            # Send data to the corresponding peer process
            dist.send(send_data, dst=partner_rank)
            print(f"Process {rank} sent {send_data} to process {partner_rank}")

            # Receive data from the peer process
            recv_data = torch.zeros_like(send_data, device=device)
            dist.recv(recv_data, src=partner_rank)
            print(f"Process {rank} received {recv_data} from process {partner_rank}")

        else:  # C and D receive first
            # Receive data from the peer process
            recv_data = torch.zeros_like(send_data, device=device)
            dist.recv(recv_data, src=partner_rank)
            print(f"Process {rank} received {recv_data} from process {partner_rank}")

            # Send data to the corresponding peer process
            dist.send(send_data, dst=partner_rank)
            print(f"Process {rank} sent {send_data} to process {partner_rank}")

def all_gather_example(rank, world_size):
    """Use all-gather to collect data from all processes"""
    # Use the same device assignment as init_process
    device_id = rank // 2
    device = torch.device(f'cuda:{device_id}')

    print(f"Process {rank} starting all-gather communication...")

    # Each process creates its own data
    local_data = torch.tensor([rank, rank * 2, rank * 3, rank * 4], device=device)
    print(f"Process {rank} local data: {local_data}")

    # Synchronize all processes
    dist.barrier()

    # Prepare a list to receive data from all processes
    gathered_data = [torch.zeros_like(local_data) for _ in range(world_size)]

    # Execute the all-gather operation
    dist.all_gather(gathered_data, local_data)

    print(f"Process {rank} gathered all data:")
    for i, data in enumerate(gathered_data):
        print(f"  From process {i}: {data}")

def run(rank, world_size):
    init_process(rank, world_size)

    # Ensure the distributed environment is initialized
    if dist.is_initialized():
        print(f"Process {rank} (Node {rank//2}) initialized successfully")

        # Wait for all processes to initialize
        # dist.barrier()

        # 1. Cross-node point-to-point communication: A-C and B-D
        cross_node_communication(rank, world_size)

        # Wait for all processes to complete point-to-point communication
        # dist.barrier()

        # 2. Use all-gather to collect data from all processes
        # all_gather_example(rank, world_size)

        # Final synchronization
        # dist.barrier()
        print(f"Process {rank} completed all operations")

if __name__ == "__main__":
    # 4 processes: A(0), B(1), C(2), D(3)
    # Node 0: A(0), B(1) - use GPU 0
    # Node 1: C(2), D(3) - use GPU 1
    world_size = 4  # Four processes total
    rank = int(sys.argv[1])
    run(rank, world_size)
