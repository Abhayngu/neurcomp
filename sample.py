import time
import torch
import torch.nn as nn
import numpy as np
import math
from sampleclass import Network, MyData
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

vol = np.load('volumes/test_vol.npy')
device = 'cuda' if torch.cuda.is_available else 'cpu'

tvol = torch.from_numpy(vol)
resol = torch.prod(torch.tensor([v for v in tvol.shape])).item()
raw_min = torch.tensor([torch.min(tvol)], dtype=tvol.dtype )
raw_max = torch.tensor([torch.max(tvol)], dtype=tvol.dtype )
normalizedVolume = 2.0*((tvol-raw_min)/(raw_max-raw_min)-0.5)

net = Network(3, 1, 6)
# print(net)
bs = 2048
loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=5e-5, betas=(0.9, 0.999))
dataset = MyData(normalizedVolume)
dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True)
#print(dataset[0])
# print(resol/len(dataloader))
# for i, (input, label) in enumerate(dataloader):
#     print('i : ', i)
#     # (x, y, z), val = data
#     # print('x : ', x, ' y : ', y, ' z : ', z , ' val : ', val)
#     # print(data.shape)
#     print('input : ', input)
#     print('label : ', label)

n_epochs = 50

for i in range(n_epochs):
    net.train()
    train_loss = 0
    
    start = time.time()
    
    for input, output in dataloader:
        #print(type(input))
        preds = net(input)
        l = loss(output, preds)
        optimizer.zero_grad()
        train_loss += l
        l.backward()
        optimizer.step()
    
    end = time.time()
    
    print(f"epoch: {i} train loss: {train_loss.item() / bs} time: {round(end - start, 2)}")
        
net.eval()
# predict the entire volume and scale back to original
predicted_volume = net(normalizedVolume)


# n_epoch = 10
# totalSamples = resol
# itr = math.ceil(totalSamples/bs)
# print('Total iterations : ', itr)

# for epoch in range(n_epoch):
#     for i, data in enumerate(dataloader):
#         (x, y, z), val = data
        