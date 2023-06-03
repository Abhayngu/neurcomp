import time
import torch
import torch.nn as nn
import numpy as np
import math
from sampleclass import Network, MyData
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import tiled_net_out
from vtk import *



vol = np.load('volumes/test_vol.npy')
device = 'cuda' if torch.cuda.is_available else 'cpu'

tvol = torch.from_numpy(vol)
resol = torch.prod(torch.tensor([v for v in tvol.shape])).item()
raw_min = torch.tensor([torch.min(tvol)], dtype=tvol.dtype )
raw_max = torch.tensor([torch.max(tvol)], dtype=tvol.dtype )
normalizedVolume = 2.0*((tvol-raw_min)/(raw_max-raw_min)-0.5)

vol2 = np.zeros(shape=(150*150*150, 4))
ind = 0
for i in range(150):
    for j in range(150):
        for k in range(150):
            vol2[ind] = [i, j, k, normalizedVolume[i, j, k]]
            ind = ind + 1
tensorData = torch.from_numpy(vol2)
tensorData = tensorData.to(torch.float)

net = Network(3, 1, 3)
print(net)
bs = 2048
loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=5e-5, betas=(0.9, 0.999))
dataset = MyData(tensorData)
dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True)

# print(normalizedVolume.shape)

n_epochs = 10

for i in range(n_epochs):
    net.train()
    train_loss = 0
    
    start = time.time()
    
    for input, output in dataloader:
        #print(type(input))
        preds = net(input)
        preds = preds.view(-1, 1)
        l = loss(output, preds)
        optimizer.zero_grad()
        train_loss += l
        l.backward()
        optimizer.step()
    
    end = time.time()
    
    print(f"epoch: {i} train loss: {train_loss.item() / bs} time: {round(end - start, 2)}")
        
net.eval()

resizedVol = np.zeros(shape=(150*150*150, 3))
ind = 0
for i in range(150):
    for j in range(150):
        for k in range(150):
            resizedVol[ind] = [i, j, k]
            ind = ind + 1
resizeTensor = torch.from_numpy(resizedVol)
resizeTensor = resizeTensor.to(torch.float)



predicted_volume = net(resizeTensor)
print(predicted_volume)

torch.save(net, 'model.pth')
