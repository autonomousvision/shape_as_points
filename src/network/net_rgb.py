# code from IDR (https://github.com/lioryariv/idr/blob/main/code/model/implicit_differentiable_renderer.py)
import torch
import torch.nn as nn
import numpy as np
from src.network.utils import get_embedder
from pdb import set_trace as st

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            fea_size=0,
            mode='naive',
            d_out=3,
            dims=[512, 512, 512, 512],
            weight_norm=True,
            pe_freq_view=0 # for positional encoding
    ):
        super().__init__()
        
        self.mode = mode
        if mode == 'naive':
            d_in = 3
        elif mode == 'no_feature':
            d_in = 3 + 3 + 3
            fea_size = 0
        elif mode == 'full':
            d_in = 3 + 3 + 3
        else:
            d_in = 3 + 3
        dims = [d_in + fea_size] + dims + [d_out]

        self.embedview_fn = None
        if pe_freq_view > 0:
            embedview_fn, input_ch = get_embedder(pe_freq_view, d_in=3)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals=None, view_dirs=None, feature_vectors=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
            # points = self.embedview_fn(points)

        if (self.mode == 'full') & (feature_vectors is not None):
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif (self.mode == 'no_feature') | ((self.mode == 'full') & (feature_vectors is None)):
            rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs], dim=-1)
        else:
            rendering_input = points

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x


class NeRFRenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size=0,
            mode='naive',
            d_in=3,
            d_out=3,
            dims=[512, 512, 512, 256],
            weight_norm=True,
            multires=0, # positional encoding of points
            multires_view=0 # positional encoding of view
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims


        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, d_in=d_in)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)
        
        self.num_layers = len(dims)

        self.pts_net = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(self.num_layers - 1)])

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, view_ch = get_embedder(multires_view, d_in=3)
            self.embedview_fn = embedview_fn
            # dims[0] += (input_ch - 3)

        if mode == 'full':
            self.view_net = nn.ModuleList([nn.Linear(dims[-1]+view_ch, 128)])
            self.rgb_net = nn.Linear(128, 3)
        else: 
            self.rgb_net = nn.Linear(dims[-1], 3)

            
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals=None, view_dirs=None, feature_vectors=None):
        if self.embed_fn is not None:
            points = self.embed_fn(points)
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        x = points
        for net in self.pts_net:
            x = net(x)
            x = self.relu(x)

        if self.mode=='full':
            x = torch.cat([x, view_dirs], -1)
            for net in self.view_net:
                x = net(x)
                x = self.relu(x)

        x = self.rgb_net(x)
        x = self.tanh(x)
        return x

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            feature_vector_size=0,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)