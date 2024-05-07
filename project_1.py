from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import torch

class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        label = self.data[idx,-1]
        return features, label
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.sigmioid(self.fc3(x))
        return x

dataset_path = "/Users/behnaz/Desktop/pytorch_projects/water_potability.csv"
dataset_train = WaterDataset(dataset_path)
dataloader_train = DataLoader(dataset_train, batch_size = 2, shuffle = True)

features, labels = next(iter(dataloader_train))
print(f"Features: {features}, \nLabels: {labels}")



# Optimizer, trainer and evaluation
net = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01)
# optimizer = optim.Adagrad(net.parameter(), lr = 0.01)
# optimizer = optim.RMSprop(net.parameter(), lr = 0.01)
# optimizer = optim.Adam(net.parameter(), lr = 0.01)

for epoch in range(1000):
    for features, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(features)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()




acc = Accuracy(task = "binary")
net.eval()
with torch.no_grad():
    for features, labels in dataloader_test:
        outputs =net(features)
        pred = (outputs>=0.5).float()

accuracy = acc.compute()
print(f"Accuracy : {accuracy}")