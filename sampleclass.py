import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Network(nn.Module):

    def __init__(self, in_features, out_features, num_hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.l1 = nn.Linear(in_features, 16)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(16, 16)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(16, out_features)
        
    
    def forward(self, input):
        out = input
        out = self.l1(out)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        return out
    
class MyData(Dataset):
    def __init__(self, data):
        self.x = data[:, :3]
        self.y = data[:, 3]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]