import torch as torch
import numpy as np
from sampleclass import Network
import torch.optim as optim

vol = np.load('volumes/test_vol.npy')
device = 'cuda' if torch.cuda.is_available else 'cpu'

tvol = torch.from_numpy(vol)
raw_min = torch.tensor([torch.min(tvol)], dtype=tvol.dtype )
raw_max = torch.tensor([torch.max(tvol)], dtype=tvol.dtype )
normalizedVolume = 2.0*((tvol-raw_min)/(raw_max-raw_min)-0.5)

net = Network(3, 1, 6)
print(net)
optimizer = optim.Adam(net.parameters(), lr=5e-5, betas=(0.9, 0.999))