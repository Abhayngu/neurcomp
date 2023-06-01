import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Network(nn.Module):

    def __init__(self, in_features, out_features, num_hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.layers.append(nn.Linear(in_features, 16))
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(16, 16))
        self.layers.append(nn.Linear(16, out_features))
    
    def forward(self, input):
        out = input
        for ndx,net_layer in enumerate(self.layers):
            out = net_layer(out)
            out = self.relu(out)
        return out
    
class MyData(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data[:, :3])
        self.y = torch.tensor(data[:, 3])

    def __len__(self):
        return self.y.shape[0].item()

    def __getitem__(self, index):
        # x, y, z = torch.randint(low=0, high=150, size=(3, 1))
        # # x = x.item()
        # # y = y.item()
        # # z = z.item()
        # return torch.tensor([x.item(), y.item(), z.item()], dtype=torch.float), self.data[x, y, z]
        return self.x[index], self.y[index]