import numpy as np
import json
import time
import random

import sys
import torch as th
from torch.utils.data.dataset import Dataset

class VolumeDataset(Dataset):
    def __init__(self,volume,oversample=16):
        # Resolution of the data [150, 150, 150]
        self.vol_res = volume.shape

        # Number of voxels 150*150*150
        self.n_voxels = th.prod(th.tensor(self.vol_res,dtype=th.int)).item()

        # Resolution in float [150., 150., 150.]
        self.vol_res_float = th.tensor([self.vol_res[0],self.vol_res[1],self.vol_res[2]],dtype=th.float)

        # Minimum value in the volume
        self.min_volume = th.tensor([th.min(volume)],dtype=volume.dtype)

        # Maximum value in the volume
        self.max_volume = th.tensor([th.max(volume)],dtype=volume.dtype)

        # minimum index which can be accessed across all the dimension
        self.min_bb = th.tensor([0.0,0.0,0.0],dtype=th.float)

        # maximum index which can be accessed across all the dimension
        self.max_bb = th.tensor([float(volume.size()[0]-1),float(volume.size()[1]-1),float(volume.size()[2]-1)],dtype=th.float)

        # Difference between max and min indices which can be accessed across all the dimension
        self.diag = self.max_bb-self.min_bb

        # very small value
        self.pos_eps = 1e-8

        # Multiplying by value of almost 1 in diag_eps
        # Remains same for not very long precision
        self.diag_eps = self.diag*(1.0-2.0*self.pos_eps)

        # Dimension having maximum length
        self.max_dim = th.max(self.diag)

        # scaling every dimension such that everything comes in [0, 1]
        self.scales = self.diag/self.max_dim
        #self.scales = th.ones(3)

        # Tile sampling
        self.lattice = self.tile_sampling(self.min_bb,self.max_bb,res=self.vol_res,normalize=False)

        # Linear Tile sampling
        self.full_tiling = self.tile_sampling(self.min_bb,self.max_bb,res=self.vol_res,normalize=False).view(-1,3)

        # Total number of voxels from the linear tiling
        self.actual_voxels = self.full_tiling.shape[0]

        # Setting oversample
        self.oversample = oversample
    #

    def tile_sampling(self, sub_min_bb, sub_max_bb, res=None, normalize=True):
        if res is None:
            res = th.tensor([self.tile_res,self.tile_res,self.tile_res],dtype=th.int)
        positional_data = th.zeros(res[0],res[1],res[2],3)

        start = sub_min_bb / (self.max_bb-self.min_bb) if normalize else sub_min_bb
        end = sub_max_bb / (self.max_bb-self.min_bb) if normalize else sub_max_bb
        positional_data[:,:,:,0] = th.linspace(start[0],end[0],res[0],dtype=th.float).view(res[0],1,1)
        positional_data[:,:,:,1] = th.linspace(start[1],end[1],res[1],dtype=th.float).view(1,res[1],1)
        positional_data[:,:,:,2] = th.linspace(start[2],end[2],res[2],dtype=th.float).view(1,1,res[2])

        return 2.0*positional_data - 1.0 if normalize else positional_data
    #

    def uniform_sampling(self,n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        return self.pos_eps + self.min_bb.unsqueeze(0) + th.rand(n_samples,3)*self.diag_eps.unsqueeze(0)
    #

    def __len__(self):
        return self.n_voxels
    #

    def __getitem__(self, index):
        random_positions = self.full_tiling[th.randint(self.actual_voxels,(self.oversample,))]
        normalized_positions = 2.0 * ( (random_positions - self.min_bb.unsqueeze(0)) / (self.max_bb-self.min_bb).unsqueeze(0) ) - 1.0
        normalized_positions = self.scales.unsqueeze(0)*normalized_positions
        return random_positions, normalized_positions
    #
#
