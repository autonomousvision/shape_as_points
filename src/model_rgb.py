import torch
from src.network.net_rgb import RenderingNetwork
from src.utils import approx_psr_grad
from pytorch3d.renderer import (
    RasterizationSettings, 
    PerspectiveCameras,
    MeshRenderer, 
    MeshRasterizer,  
    SoftSilhouetteShader)
from pytorch3d.structures import Meshes


def approx_psr_grad(psr_grid, res, normalize=True):
    delta_x = delta_y = delta_z = 1/res
    psr_pad = torch.nn.ReplicationPad3d(1)(psr_grid).squeeze()

    grad_x = (psr_pad[2:, :, :] - psr_pad[:-2, :, :]) / 2 / delta_x
    grad_y = (psr_pad[:, 2:, :] - psr_pad[:, :-2, :]) / 2 / delta_y
    grad_z = (psr_pad[:, :, 2:] - psr_pad[:, :, :-2]) / 2 / delta_z
    grad_x = grad_x[:, 1:-1, 1:-1]
    grad_y = grad_y[1:-1, :, 1:-1]
    grad_z = grad_z[1:-1, 1:-1, :]

    psr_grad = torch.stack([grad_x, grad_y, grad_z], dim=3)  # [res_x, res_y, res_z, 3]
    if normalize:
        psr_grad = psr_grad / (psr_grad.norm(dim=3, keepdim=True) + 1e-12)

    return psr_grad


class SAP2Image(nn.Module):
    def __init__(self, cfg, img_size):
        super().__init__()

        self.psr2sur = PSR2SurfacePoints.apply
        self.psr2mesh = PSR2Mesh.apply
        # initialize DPSR
        self.dpsr = DPSR(res=(cfg['model']['grid_res'], 
                              cfg['model']['grid_res'], 
                              cfg['model']['grid_res']), 
                         sig=cfg['model']['psr_sigma'])
        self.cfg = cfg
        if cfg['train']['l_weight']['rgb'] != 0.:
            self.rendering_network = RenderingNetwork(**cfg['model']['renderer'])

        if cfg['train']['l_weight']['mask'] != 0.:
            # initialize rasterizer
            sigma = 1e-4
            raster_settings_soft = RasterizationSettings(
                image_size=img_size, 
                blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
                faces_per_pixel=150,
                perspective_correct=False
            )

            # initialize silhouette renderer 
            self.mesh_rasterizer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    raster_settings=raster_settings_soft
                ),
                shader=SoftSilhouetteShader()
            )

        self.cfg = cfg
        self.img_size = img_size
    
    def forward(self, inputs, data):
        points, normals = inputs[...,:3], inputs[...,3:]
        points = torch.sigmoid(points)
        normals = normals / normals.norm(dim=-1, keepdim=True)

        # DPSR to get grid
        psr_grid = self.dpsr(points, normals).unsqueeze(1)
        psr_grid = torch.tanh(psr_grid)
        
        return self.render_img(psr_grid, data)

    def render_img(self, psr_grid, data):
        
        n_views = len(data['masks'])
        n_views_per_iter = self.cfg['data']['n_views_per_iter']
    
        rgb_render_mode = self.cfg['model']['renderer']['mode']
        uv = data['uv']

        idx = np.random.randint(0, n_views, n_views_per_iter)
        pose = [data['poses'][i] for i in idx]
        rgb = data['rgbs'][idx]
        mask_gt = data['masks'][idx]
        ray = None
        pred_rgb = None
        pred_mask = None
        
        if self.cfg['train']['l_weight']['rgb'] != 0.:
            psr_grad = approx_psr_grad(psr_grid, self.cfg['model']['grid_res'])
            p_inters, visible_mask = self.psr2sur(psr_grid, pose, self.img_size, uv, psr_grad, None)
            n_inters = grid_interp(psr_grad[None], (p_inters.detach()[None] + 1) / 2)
            fea_interp = None
            if 'rays' in data.keys():
                ray = data['rays'].squeeze()[idx][visible_mask]
            pred_rgb = self.rendering_network(p_inters, normals=n_inters.squeeze(), view_dirs=ray, feature_vectors=fea_interp)
            
        # silhouette loss
        if self.cfg['train']['l_weight']['mask'] != 0.:
            # build mesh
            v, f, _ = self.psr2mesh(psr_grid)
            v = v * 2. - 1 # within the range of [-1, 1]
            # ! Fast but more GPU usage
            mesh = Meshes(verts=[v.squeeze()], faces=[f.squeeze()])
            if True:
                #! PyTorch3D silhouette loss
                # build pose
                R = torch.cat([p.R for p in pose], dim=0)
                T = torch.cat([p.T for p in pose], dim=0)
                focal = torch.cat([p.focal_length for p in pose], dim=0)
                pp = torch.cat([p.principal_point for p in pose], dim=0)
                pose_cur = PerspectiveCameras(
                                    focal_length=focal, 
                                    principal_point=pp, 
                                    R=R, T=T,
                                    device=R.device)
                pred_mask = self.mesh_rasterizer(mesh.extend(n_views_per_iter), cameras=pose_cur)[..., 3]
            else:
                pred_mask = []
                # ! Slow but less GPU usage
                for i in range(n_views_per_iter):
                    #! PyTorch3D silhouette loss
                    pred_mask.append(self.mesh_rasterizer(mesh, cameras=pose[i])[..., 3])
                pred_mask = torch.cat(pred_mask, dim=0)

        output = {
            'rgb': pred_rgb,
            'rgb_gt': rgb,
            'mask': pred_mask,
            'mask_gt': mask_gt,
            'vis_mask': visible_mask,
            }
        
        return output