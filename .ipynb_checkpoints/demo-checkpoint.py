import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


########################################
# 1. Dataset giả lập
########################################

class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.x = torch.randn(size, 10)
        self.y = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


########################################
# 2. Model demo
########################################

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


########################################
# 3. Setup distributed
########################################

def setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )


def cleanup():
    dist.destroy_process_group()


########################################
# 4. Train function
########################################

def train(rank, world_size):

    print(f"Running training on GPU {rank}")

    setup(rank, world_size)

    torch.cuda.set_device(rank)

    dataset = DummyDataset()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )

    model = MyModel().to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()

    epochs = 3

    for epoch in range(epochs):

        sampler.set_epoch(epoch)

        for data, target in dataloader:

            data = data.to(rank)
            target = target.to(rank)

            optimizer.zero_grad()

            output = ddp_model(data)

            loss = loss_fn(output, target)

            loss.backward()

            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch} finished | Loss: {loss.item():.4f}")

    cleanup()


########################################
# 5. Main
########################################

def main():

    world_size = torch.cuda.device_count()

    print("Using GPUs:", world_size)

    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()