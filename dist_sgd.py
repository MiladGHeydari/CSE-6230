import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from random import Random
import math
import time

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
    
class DriveDataset(Dataset):
    def __init__(self, csv_file):
        columns = [f'col{i+1}' for i in range(48)] + ['Class']
        self.data = pd.read_csv('Sensorless_drive_diagnosis.txt', delimiter=' ', header=None, names= columns)

        # Encode target column
        self.label_encoder = LabelEncoder()
        self.data['Class'] = self.label_encoder.fit_transform(self.data['Class'])

        # Standardize features
        self.scaler = StandardScaler()
        self.data.iloc[:, :-1] = self.scaler.fit_transform(self.data.iloc[:, :-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, :-1], dtype=torch.float32)
        target = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)
        return features, target
    
def partition_dataset():
    dataset = dataset = DriveDataset('Sensorless_drive_diagnosis.txt')

    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz

input_dim = 48
hidden_size1 = 32
hidden_size2 = 64
num_classes = 11

class ThreeLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2, num_classes):
        super(ThreeLayerNet, self).__init__()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(input_dim, hidden_size1)
        self.lin2 = nn.Linear(hidden_size1, hidden_size2)
        self.lin3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.LeakyReLU()  # Use one instance for all activations

    def forward(self, x):
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        # No activation after the last layer as CrossEntropyLoss includes SoftMax
        return x

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def run(rank, size):
    """ Distributed function to be implemented later. """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = ThreeLayerNet(input_dim, hidden_size1, hidden_size2, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0025)
    num_batches = math.ceil(len(train_set.dataset) / float(bsz))
    num_epochs = 20
    LossThreshold = 0.9
    Converged = False
    
    for epoch in range(num_epochs):
        if Converged:
            break
        model.train()
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Check if loss is below threshold
            if loss.item() < LossThreshold:
                Converged = True

        print(f'Rank {dist.get_rank()}, epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / num_batches:.4f}')

    # Ensure all processes have stopped training before exiting
    dist.barrier()



def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 8
    processes = []
    mp.set_start_method("spawn")
    start = time.perf_counter()
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    end = time.perf_counter()
    ms = (end - start)
    print(f"Elapsed {ms:.03f} secs for {size} processes.")
