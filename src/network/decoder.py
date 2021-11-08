import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ipdb import set_trace as st
from src.network.utils import normalize_3d_coordinate, ResnetBlockFC, \
    normalize_coordinate


class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128, out_dim=3,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, map2local=None):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
        self.map2local = map2local
        self.out_dim = out_dim
    

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', 
                                    align_corners=True, 
                                    mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone())
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', 
                                    align_corners=True, 
                                    mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        batch_size = p.shape[0]
        plane_type = list(c_plane.keys())
        c = 0
        if 'grid' in plane_type:
            c += self.sample_grid_feature(p, c_plane['grid'])
        if 'xz' in plane_type:
            c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
        if 'xy' in plane_type:
            c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
        if 'yz' in plane_type:
            c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
        c = c.transpose(1, 2)

        p = p.float()

        if self.map2local:
            p = self.map2local(p)
        
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        

        if self.out_dim > 3:
            out = out.reshape(batch_size, -1, 3)
            
        return out