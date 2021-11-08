import torch
import numpy as np
import time
from src.utils import point_rasterize, grid_interp, mc_from_psr, \
calc_inters_points
from src.dpsr import DPSR
import torch.nn as nn
from src.network import encoder_dict, decoder_dict
from src.network.utils import map2local

class PSR2Mesh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psr_grid):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        verts, faces, normals = mc_from_psr(psr_grid, pytorchify=True)
        verts = verts.unsqueeze(0)
        faces = faces.unsqueeze(0)
        normals = normals.unsqueeze(0)

        res = torch.tensor(psr_grid.detach().shape[2])
        ctx.save_for_backward(verts, normals, res)

        return verts, faces, normals

    @staticmethod
    def backward(ctx, dL_dVertex, dL_dFace, dL_dNormals):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        vert_pts, normals, res = ctx.saved_tensors
        res = (res.item(), res.item(), res.item())
        # matrix multiplication between dL/dV and dV/dPSR
        # dV/dPSR = - normals
        grad_vert = torch.matmul(dL_dVertex.permute(1, 0, 2), -normals.permute(1, 2, 0))
        grad_grid = point_rasterize(vert_pts, grad_vert.permute(1, 0, 2), res) # b x 1 x res x res x res
        
        return grad_grid

class PSR2SurfacePoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psr_grid, poses, img_size, uv, psr_grad, mask_sample):
        verts, faces, normals = mc_from_psr(psr_grid, pytorchify=True)
        verts = verts * 2. - 1. # within the range of [-1, 1]

        
        p_all, n_all, mask_all = [], [], []

        for i in range(len(poses)):
            pose = poses[i]
            if mask_sample is not None:
                p_inters, mask, _, _ = calc_inters_points(verts, faces, pose, img_size, mask_gt=mask_sample[i])
            else:
                p_inters, mask, _, _ = calc_inters_points(verts, faces, pose, img_size)

            n_inters = grid_interp(psr_grad[None], (p_inters[None].detach() + 1) / 2).squeeze()
            p_all.append(p_inters)
            n_all.append(n_inters)
            mask_all.append(mask)
        p_inters_all = torch.cat(p_all, dim=0)
        n_inters_all = torch.cat(n_all, dim=0)
        mask_visible = torch.stack(mask_all, dim=0)


        res = torch.tensor(psr_grid.detach().shape[2])
        ctx.save_for_backward(p_inters_all, n_inters_all, res)

        return p_inters_all, mask_visible

    @staticmethod
    def backward(ctx, dL_dp, dL_dmask):
        pts, pts_n, res = ctx.saved_tensors
        res = (res.item(), res.item(), res.item())

        # grad from the p_inters via MLP renderer
        grad_pts = torch.matmul(dL_dp[:, None], -pts_n[..., None])
        grad_grid_pts = point_rasterize((pts[None]+1)/2, grad_pts.permute(1, 0, 2), res) # b x 1 x res x res x res
        
        return grad_grid_pts, None, None, None, None, None

class Encode2Points(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        encoder = cfg['model']['encoder']
        decoder = cfg['model']['decoder']
        dim = cfg['data']['dim'] # input dim
        c_dim = cfg['model']['c_dim']
        encoder_kwargs = cfg['model']['encoder_kwargs']
        if encoder_kwargs == None:
            encoder_kwargs = {}
        decoder_kwargs = cfg['model']['decoder_kwargs']
        padding = cfg['data']['padding']
        self.predict_normal = cfg['model']['predict_normal']
        self.predict_offset = cfg['model']['predict_offset']

        out_dim = 3
        out_dim_offset = 3
        num_offset = cfg['data']['num_offset']
        # each point predict more than one offset to add output points
        if num_offset > 1:
            out_dim_offset = out_dim * num_offset
        self.num_offset = num_offset

        # local mapping
        self.map2local = None
        if cfg['model']['local_coord']:
            if 'unet' in encoder_kwargs.keys():
                unit_size = 1 / encoder_kwargs['plane_resolution']
            else:
                unit_size = 1 / encoder_kwargs['grid_resolution']
            
            local_mapping = map2local(unit_size)

        self.encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, map2local=local_mapping,
            **encoder_kwargs
        )

        if self.predict_normal:
            # decoder for normal prediction
            self.decoder_normal = decoder_dict[decoder](
                dim=dim, c_dim=c_dim, out_dim=out_dim,
                **decoder_kwargs)
        if self.predict_offset:
            # decoder for offset prediction
            self.decoder_offset = decoder_dict[decoder](
                dim=dim, c_dim=c_dim, out_dim=out_dim_offset,
                map2local=local_mapping,
                **decoder_kwargs)

            self.s_off = cfg['model']['s_offset']
        
        
    def forward(self, p):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): input unoriented points
        '''

        time_dict = {}
        mask = None
        
        batch_size = p.size(0)
        points = p.clone()

        # encode the input point cloud to a feature volume
        t0 = time.perf_counter()
        c = self.encoder(p)
        t1 = time.perf_counter()
        if self.predict_offset:
            offset = self.decoder_offset(p, c)
            # more than one offset is predicted per-point
            if self.num_offset > 1:
                points = points.repeat(1, 1, self.num_offset).reshape(batch_size, -1, 3)
            points = points + self.s_off * offset
        else:
            points = p

        if self.predict_normal:
            normals = self.decoder_normal(points, c)
        t2 = time.perf_counter()
        
        time_dict['encode'] = t1 - t0
        time_dict['predict'] = t2 - t1
        
        points = torch.clamp(points, 0.0, 0.99)
        if self.cfg['model']['normal_normalize']:
            normals = normals / (normals.norm(dim=-1, keepdim=True)+1e-8)

        
        return points, normals
    