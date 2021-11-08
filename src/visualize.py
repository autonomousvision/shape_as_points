import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from skimage import measure
from src.utils import calc_inters_points, grid_interp
from scipy import ndimage
from tqdm import trange
from torchvision.utils import save_image
from pdb import set_trace as st


def visualize_points_mesh(vis, points, normals, verts, faces, cfg, it, epoch, color_v=None):
    ''' Visualization.

    Args:
        data (dict): data dictionary
        depth (int): PSR depth
        out_path (str): output path for the mesh 
    '''
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color(np.array([0.7,0.7,0.7]))
    if color_v is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(color_v)
    
    if vis is not None:
        dir_o3d = cfg['train']['dir_o3d']
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

        p = points.squeeze(0).detach().cpu().numpy()
        n = normals.squeeze(0).detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        pcd.normals = o3d.utility.Vector3dVector(n)
        pcd.paint_uniform_color(np.array([0.7,0.7,1.0]))
        # pcd = pcd.uniform_down_sample(5)

        vis.clear_geometries()
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)
        
        #! Thingi wheel - an example for how to change cameras in Open3D viewers
        vis.get_view_control().set_front([ 0.0461, -0.7467, 0.6635 ])
        vis.get_view_control().set_lookat([ 0.0092, 0.0078, 0.0638 ])
        vis.get_view_control().set_up([ 0.0520, 0.6651, 0.7449 ])
        vis.get_view_control().set_zoom(0.7)
        vis.poll_events()
        
        out_path = os.path.join(dir_o3d, '{}.jpg'.format(it))
        vis.capture_screen_image(out_path)

        vis.clear_geometries()
        vis.add_geometry(pcd, reset_bounding_box=False)
        vis.update_geometry(pcd)
        vis.get_render_option().point_show_normal=True # visualize point normals

        vis.get_view_control().set_front([ 0.0461, -0.7467, 0.6635 ])
        vis.get_view_control().set_lookat([ 0.0092, 0.0078, 0.0638 ])
        vis.get_view_control().set_up([ 0.0520, 0.6651, 0.7449 ])
        vis.get_view_control().set_zoom(0.7)
        vis.poll_events()

        out_path = os.path.join(dir_o3d, '{}_pcd.jpg'.format(it))
        vis.capture_screen_image(out_path)

def visualize_psr_grid(psr_grid, pose=None, out_dir=None, out_video_name='video.mp4'):
    if pose is not None:
        device = psr_grid.device
        # get world coordinate of grid points [-1, 1]
        res = psr_grid.shape[-1]
        x = torch.linspace(-1, 1, steps=res)
        co_x, co_y, co_z = torch.meshgrid(x, x, x)
        co_grid = torch.stack(
                [co_x.reshape(-1), co_y.reshape(-1), co_z.reshape(-1)], 
                dim=1).to(device).unsqueeze(0)

        # visualize the projected occ_soft value
        res = 128
        psr_grid = psr_grid.reshape(-1)
        out_mask = psr_grid>0
        in_mask = psr_grid<0
        pix = pose.transform_points_screen(co_grid, ((res, res),))[..., :2].round().long().squeeze()
        vis_mask = (pix[..., 0]>=0) & (pix[..., 0]<=res-1) & \
                    (pix[..., 1]>=0) & (pix[..., 1]<=res-1)
        pix_out = pix[vis_mask & out_mask]
        pix_in = pix[vis_mask & in_mask]
        
        img = torch.ones([res,res]).to(device)
        psr_grid = torch.sigmoid(- psr_grid * 5)
        img[pix_out[:, 1], pix_out[:, 0]] = psr_grid[vis_mask & out_mask]
        img[pix_in[:, 1], pix_in[:, 0]] = psr_grid[vis_mask & in_mask]
        # save_image(img, 'tmp.png', normalize=True)
        return img
    elif out_dir is not None:
        dir_psr_vis = out_dir
        os.makedirs(dir_psr_vis, exist_ok=True)
        psr_grid = psr_grid.squeeze().detach().cpu().numpy()
        axis = ['x', 'y', 'z']
        s = psr_grid.shape[0]
        for idx in trange(s):
            my_dpi = 100
            plt.figure(figsize=(1000/my_dpi, 300/my_dpi), dpi=my_dpi)
            plt.subplot(1, 3, 1)
            plt.imshow(ndimage.rotate(psr_grid[idx], 180, mode='nearest'), cmap='nipy_spectral')
            plt.clim(-1, 1)
            plt.colorbar()
            plt.title('x')
            plt.grid("off")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(ndimage.rotate(psr_grid[:, idx], 180, mode='nearest'), cmap='nipy_spectral')
            plt.clim(-1, 1)
            plt.colorbar()
            plt.title('y')
            plt.grid("off")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(ndimage.rotate(psr_grid[:,:,idx], 90, mode='nearest'), cmap='nipy_spectral')
            plt.clim(-1, 1)
            plt.colorbar()
            plt.title('z')
            plt.grid("off")
            plt.axis("off")


            plt.savefig(os.path.join(dir_psr_vis, '{}'.format(idx)), pad_inches = 0, dpi=100)
            plt.close()
        os.system("rm {}/{}".format(dir_psr_vis, out_video_name))
        os.system("ffmpeg -framerate 25 -start_number 0 -i {}/%d.png -pix_fmt yuv420p -crf 17 {}/{}".format(dir_psr_vis, dir_psr_vis, out_video_name))

def visualize_mesh_phong(v, f, n, pose, img_size, name, device='cpu'):
    #! Mesh rendering using Phong shading model
    _, mask, f_p, w = calc_inters_points(v, f, pose, img_size)
    n_a, n_b, n_c = n[:, f_p[..., 0]], n[:, f_p[..., 1]], n[:, f_p[..., 2]]
    n_inters = w[..., 0, None] * n_a.squeeze() + \
                w[..., 1, None] * n_b.squeeze() + \
                w[..., 2, None] * n_c.squeeze()
    n_inters = n_inters.detach().to(device)
    light_source = -pose.R@pose.T.squeeze()
    light = (light_source / light_source.norm(2)).permute(1, 0).to(device).float()
    diffuse_per = torch.Tensor([0.7,0.7,0.7]).float()
    ambiant = torch.Tensor([0.3,0.3,0.3]).float()

    diffuse = torch.mm(n_inters, light).clamp_min(0).repeat(1, 3) * diffuse_per.unsqueeze(0).to(device)

    phong = torch.ones([img_size[0]*img_size[1], 3]).to(device)
    phong[mask] = (ambiant.unsqueeze(0).to(device) + diffuse).clamp_max(1.0)
    pp = phong.reshape(img_size[0], img_size[1], -1)
    save_image(pp.permute(2, 0, 1), name)

def render_rgb(v, f, n, pose, renderer, img_size, mask_gt=None, ray=None, fea_grid=None):
        p_inters, mask, f_p, w = calc_inters_points(v.detach(), f, pose, img_size, mask_gt=mask_gt)
        # normals for p_inters
        n_inters = None
        if n is not None:
            n_a, n_b, n_c = n[:, f_p[..., 0]], n[:, f_p[..., 1]], n[:, f_p[..., 2]]
            n_inters = w[..., 0, None] * n_a.squeeze() + \
                       w[..., 1, None] * n_b.squeeze() + \
                       w[..., 2, None] * n_c.squeeze()
        if ray is not None:
            ray = ray.squeeze()[mask]
        
        fea = None
        if fea_grid is not None:
            fea = grid_interp(fea_grid, (p_inters.detach()[None] + 1) / 2).squeeze()

        # use MLP to regress color
        color_pred = renderer(p_inters, normals=n_inters, view_dirs=ray, feature_vectors=fea).squeeze()

        return color_pred, mask