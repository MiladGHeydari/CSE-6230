import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tqdm
import torch.multiprocessing as mp
import time

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
    
def train(model, data_loader):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        epoch_loss = 0.0
        for data, labels in data_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), labels)
            epoch_loss += loss.item()
            loss.backward()        
            optimizer.step()
        print(f'Epoch [{epoch+1}/20], Loss: {epoch_loss:.4f}')

input_dim = 48
hidden_size1 = 32
hidden_size2 = 64
num_classes = 11
num_processes = 8

if __name__ == "__main__":
    model = ThreeLayerNet(input_dim, hidden_size1, hidden_size2, num_classes)
    model.share_memory()

    dataset = DriveDataset('Sensorless_drive_diagnosis.txt')
    processes = []
    start = time.perf_counter()
    for rank in range(num_processes):
        data_loader = DataLoader(
            dataset=dataset,
            sampler=DistributedSampler(
                dataset=dataset,
                num_replicas=num_processes,
                rank=rank
            ),
            batch_size=32
        )
        p = mp.Process(target=train, args=(model, data_loader))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    end = time.perf_counter()
    ms = (end - start)
    print(f"Elapsed {ms:.03f} secs for {num_processes} processes.")
