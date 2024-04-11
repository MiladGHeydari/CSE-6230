import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp
import time
import warnings
import sys

import ctypes
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

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

class ThreeLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super(ThreeLayerNet, self).__init__()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.LeakyReLU()  # Use one instance for all activations

    def forward(self, x):
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        # No activation after the last layer as CrossEntropyLoss includes SoftMax
        return x
    
def train(model, data_loader, v_rank, num_processes, starttime):
    optimizer = optim.Adam(model.parameters(), lr = 0.00001)
    criterion = nn.CrossEntropyLoss()


    SGD_precent = 0.075
    total_batches = len(data_loader)
    batches_to_use = int(total_batches * SGD_precent)

    for epoch in range(3000):
        selected_indices = np.random.choice(range(total_batches), batches_to_use, replace=False)
        epoch_loss = 0.0
        for i, (data, labels) in enumerate(data_loader):
            if i in selected_indices:
                optimizer.zero_grad()
                loss = criterion(model(data), labels)
                epoch_loss += loss.item()
                loss.backward()        
                optimizer.step()
        epoch_loss = epoch_loss / batches_to_use
        if v_rank == 0 and epoch % 5 == 0:
            print(f'Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, time:', time.time() - starttime)
        if epoch_loss < 0.3:
            return


#==========

input_dim = 48
hidden_size = 12
num_classes = 11
num_processes = None

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: script.py num_processes")
        sys.exit(1)

    num_processes = int(sys.argv[1])

    model = ThreeLayerNet(input_dim, hidden_size, num_classes)
    model.share_memory()

    dataset = DriveDataset('Sensorless_drive_diagnosis.txt')
    dataset, test_data = train_test_split(dataset, test_size=0.1, random_state = 40)

    processes = []
    for rank in range(num_processes):
        data_loader = DataLoader(
            dataset=dataset,
            batch_size = 16
        )
        p = mp.Process(target=train, args=(model, data_loader, rank, num_processes, time.time()))
        processes.append(p)

    start = time.perf_counter()
    for p in processes:
        p.start()
        print("starting")

    for p in processes:
        p.join()
        print("joining")

    print("done")
    end = time.perf_counter()
    ms = (end - start)
    print(f"Elapsed {ms:.03f} secs for {num_processes} processes.")

    #----

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in DataLoader(test_data, batch_size=64):
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

