import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import torch.optim as optim

def setup(rank, world_size):
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train(rank, world_size, epochs=5, batch_size=64):
    setup(rank, world_size)

    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = SimpleNet()
    ddp_model = DDP(model)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01 * world_size)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for data, target in loader:
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    if rank == 0:
        print("Training Time:", time.time() - start)

    cleanup()

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    train(rank, world_size)
