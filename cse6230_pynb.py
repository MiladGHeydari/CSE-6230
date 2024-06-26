import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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

dataset = DriveDataset('Sensorless_drive_diagnosis.txt')
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

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

# Updated model instantiation with the new layer sizes
model = ThreeLayerNet(input_dim, hidden_size1, hidden_size2, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0025)

import time
num_epochs = 100
Converged = False

start_time = time.time()

LossThreshold = 0.1

for epoch in range(num_epochs):
    if Converged:
      break
    model.train()
    for inputs, targets in train_loader:

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        if loss.item() < LossThreshold:
          Converged = True
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

end_time = time.time()
duration = end_time - start_time
print(f" {duration} seconds to converge.")

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

