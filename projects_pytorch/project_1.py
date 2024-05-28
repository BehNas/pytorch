from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import torch
import multiprocessing
import torch.nn.init as init
import torch.optim.lr_scheduler as lr_scheduler



multiprocessing.freeze_support() 
class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        df = pd.read_csv(csv_path)
        # drop instances with nan values
        df = df.dropna()
        # Normalize the columns
        df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean(axis=0)) / df.iloc[:, :-1].std(axis=0)       
        self.data = df.to_numpy().astype(np.float32)

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
        # Add two batch normalization layers
        # self.bn1 = nn.BatchNorm1d(16)
        # self.bn2 = nn.BatchNorm1d(8)
        
        # Apply He initialization
        # init.kaiming_uniform_(self.fc1.weight)
        # init.kaiming_uniform_(self.fc2.weight)
        # init.kaiming_uniform_(self.fc3.weight, nonlinearity = "sigmoid")
    
    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = nn.functional.elu(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = nn.functional.elu(x)
        x = nn.functional.sigmoid(self.fc3(x))
        return x

dataset_path = "/Users/behnaz/Desktop/pytorch_projects/water_potability.csv"
dataset = WaterDataset(dataset_path)



# Define the sizes of training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

dataset_train, dataset_test = random_split(dataset, [train_size, test_size])


dataloader_train = DataLoader(dataset_train, batch_size = 2, shuffle = True, num_workers=0)
dataloader_test = DataLoader(dataset_test, batch_size = 2, shuffle = True,  num_workers=0)
# import pdb
# pdb.set_trace()
features, labels = next(iter(dataloader_train))

print(f"Features: {features}, \nLabels: {labels}")

# Optimizer, trainer and evaluation
net = Net()
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01)
# optimizer = optim.Adagrad(net.parameter(), lr = 0.01)
# optimizer = optim.RMSprop(net.parameter(), lr = 0.01)
# optimizer = optim.Adam(net.parameter(), lr = 0.01)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

net.to(device='cpu')
net.train()
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
        acc(pred, labels.view(-1, 1))

accuracy = acc.compute()
print(f"Accuracy : {accuracy}")