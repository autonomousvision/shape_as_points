import os
import numpy as np
import torch
from torch.nn import functional as F
from collections import defaultdict
import trimesh
from tqdm import tqdm

from src.dpsr import DPSR
from src.utils import grid_interp, export_pointcloud, export_mesh, \
                      mc_from_psr, scale2onet, GaussianSmoothing
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import chamfer_distance
from pdb import set_trace as st

class Trainer(object):
    '''
    Args:
        model (nn.Module): our defined model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
    '''
    def __init__(self, cfg, optimizer, device=None):
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        if self.cfg['train']['w_raw'] != 0:
            from src.model import PSR2Mesh
            self.psr2mesh = PSR2Mesh.apply

        # initialize DPSR
        self.dpsr = DPSR(res=(cfg['model']['grid_res'], 
                            cfg['model']['grid_res'], 
                            cfg['model']['grid_res']), 
                        sig=cfg['model']['psr_sigma'])
        if torch.cuda.device_count() > 1:    
            self.dpsr = torch.nn.DataParallel(self.dpsr) # parallell DPSR
        self.dpsr = self.dpsr.to(device)

        if cfg['train']['gauss_weight']>0.:
            self.gauss_smooth = GaussianSmoothing(1, 7, 2).to(device)
        
    def train_step(self, inputs, data, model):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.optimizer.zero_grad()
        p = data.get('inputs').to(self.device)
        
        out = model(p)
        
        points, normals = out

        loss = 0
        loss_each = {}
        if self.cfg['train']['w_psr'] != 0:
            psr_gt = data.get('gt_psr').to(self.device)
            if self.cfg['model']['psr_tanh']:
                psr_gt = torch.tanh(psr_gt)
            
            psr_grid = self.dpsr(points, normals)
            if self.cfg['model']['psr_tanh']:
                psr_grid = torch.tanh(psr_grid)

            # apply a rescaling weight based on GT SDF values
            if self.cfg['train']['gauss_weight']>0:
                gauss_sigma = self.cfg['train']['gauss_weight']
                # set up the weighting for loss, higher weights 
                # for points near to the surface
                psr_gt_pad = torch.nn.ReplicationPad3d(1)(psr_gt.unsqueeze(1)).squeeze(1)
                delta_x = delta_y = delta_z = 1
                grad_x = (psr_gt_pad[:, 2:, :, :] - psr_gt_pad[:, :-2, :, :]) / 2 / delta_x
                grad_y = (psr_gt_pad[:, :, 2:, :] - psr_gt_pad[:, :, :-2, :]) / 2 / delta_y
                grad_z = (psr_gt_pad[:, :, :, 2:] - psr_gt_pad[:, :, :, :-2]) / 2 / delta_z
                grad_x = grad_x[:, :, 1:-1, 1:-1]
                grad_y = grad_y[:, 1:-1, :, 1:-1]
                grad_z = grad_z[:, 1:-1, 1:-1, :]
                psr_grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)
                psr_grad_norm = psr_grad.norm(dim=-1)[:, None]
                w = torch.nn.ReplicationPad3d(3)(psr_grad_norm)
                w = 2*self.gauss_smooth(w).squeeze(1)
                loss_each['psr'] = self.cfg['train']['w_psr'] * F.mse_loss(w*psr_grid, w*psr_gt)
            else:
                loss_each['psr'] = self.cfg['train']['w_psr'] * F.mse_loss(psr_grid, psr_gt)

            loss += loss_each['psr']

        # regularization on the input point positions via chamfer distance
        if self.cfg['train']['w_reg_point'] != 0.:
            points_gt = data.get('gt_points').to(self.device)
            loss_reg, loss_norm = chamfer_distance(points, points_gt)
                
            loss_each['reg'] = self.cfg['train']['w_reg_point'] * loss_reg
            loss += loss_each['reg']
            
        if self.cfg['train']['w_normals'] != 0.:
            points_gt = data.get('gt_points').to(self.device)
            normals_gt = data.get('gt_points.normals').to(self.device)
            x_nn = knn_points(points, points_gt, K=1)
            x_normals_near = knn_gather(normals_gt, x_nn.idx)[..., 0, :]
            
            cham_norm_x = F.l1_loss(normals, x_normals_near)
            loss_norm = cham_norm_x

            loss_each['normals'] = self.cfg['train']['w_normals'] * loss_norm
            loss += loss_each['normals']    
            
        if self.cfg['train']['w_raw'] != 0:
            res = self.cfg['model']['grid_res']
            # DPSR to get grid
            psr_grid = self.dpsr(points, normals)
            if self.cfg['model']['psr_tanh']:
                psr_grid = torch.tanh(psr_grid)
            
            v, f, n = self.psr2mesh(psr_grid)

            pts_gt = data.get('gt_points').to(self.device)
            
            loss, _ = chamfer_distance(v, pts_gt)

        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_each
    
    def save(self, model, data, epoch, id):
        
        p = data.get('inputs').to(self.device)

        exp_pcl = self.cfg['train']['exp_pcl']
        exp_mesh = self.cfg['train']['exp_mesh']
        exp_gt = self.cfg['generation']['exp_gt']
        exp_input = self.cfg['generation']['exp_input']
        
        model.eval()
        with torch.no_grad():
            points, normals = model(p)

        if exp_gt:
            points_gt = data.get('gt_points').to(self.device)
            normals_gt = data.get('gt_points.normals').to(self.device)

        if exp_pcl:
            dir_pcl = self.cfg['train']['dir_pcl']
            export_pointcloud(os.path.join(dir_pcl, '{:04d}_{:01d}.ply'.format(epoch, id)), scale2onet(points), normals)
            if exp_gt:
                export_pointcloud(os.path.join(dir_pcl, '{:04d}_{:01d}_oracle.ply'.format(epoch, id)), scale2onet(points_gt), normals_gt)
            if exp_input:
                export_pointcloud(os.path.join(dir_pcl, '{:04d}_{:01d}_input.ply'.format(epoch, id)), scale2onet(p))

        if exp_mesh:
            dir_mesh = self.cfg['train']['dir_mesh']
            psr_grid = self.dpsr(points, normals)
            # psr_grid = torch.tanh(psr_grid)
            with torch.no_grad():
                v, f, _ = mc_from_psr(psr_grid, 
                            zero_level=self.cfg['data']['zero_level'])
            outdir_mesh = os.path.join(dir_mesh, '{:04d}_{:01d}.ply'.format(epoch, id))
            export_mesh(outdir_mesh, scale2onet(v), f)
            if exp_gt:
                psr_gt = self.dpsr(points_gt, normals_gt)
                with torch.no_grad():
                    v, f, _ = mc_from_psr(psr_gt,
                            zero_level=self.cfg['data']['zero_level'])
                export_mesh(os.path.join(dir_mesh, '{:04d}_{:01d}_oracle.ply'.format(epoch, id)), scale2onet(v), f)
        
    def evaluate(self, val_loader, model):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data, model)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict
    
    def eval_step(self, data, model):
        ''' Performs an evaluation step.
        Args:
            data (dict): data dictionary
        '''
        model.eval()
        eval_dict = {}

        p = data.get('inputs').to(self.device)
        psr_gt = data.get('gt_psr').to(self.device)
        
        with torch.no_grad():
            # forward pass
            points, normals = model(p)
            # DPSR to get predicted psr grid
            psr_grid = self.dpsr(points, normals)

        eval_dict['psr_l1'] = F.l1_loss(psr_grid, psr_gt).item()
        eval_dict['psr_l2'] = F.mse_loss(psr_grid, psr_gt).item()

        return eval_dict