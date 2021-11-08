import os
import cv2
import torch
import numpy as np
from glob import glob
from torch.utils import data
from src.utils import load_rgb, load_mask, get_camera_params
from pytorch3d.renderer import PerspectiveCameras
from skimage import img_as_float32

##################################################
# Below are for the differentiable renderer
# Taken from https://github.com/lioryariv/idr/blob/main/code/utils/rend_util.py

def load_rgb(path):
    img = imageio.imread(path)
    img = img_as_float32(img)

    # pixel values between [-1,1]
    # img -= 0.5
    # img *= 2.

    # img = img.transpose(2, 0, 1)
    return img

def load_mask(path):
    alpha = imageio.imread(path, as_gray=True)
    alpha = img_as_float32(alpha)
    object_mask = alpha > 127.5

    return object_mask


def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples))
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc

def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def lift(x, y, z, intrinsics):
    # parse intrinsics
    # intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)


class PixelNeRFDTUDataset(data.Dataset):
    """
    Processed DTU from pixelNeRF
    """
    def __init__(self,
                 data_dir='data/DTU',
                 scan_id=65,
                 img_size=None,
                 device=None,
                 fixed_scale=0,
                 ):
        data_dir = os.path.join(data_dir, "scan{}".format(scan_id))
        rgb_paths = [
            x for x in glob(os.path.join(data_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob(os.path.join(data_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)
        sel_indices = np.arange(len(rgb_paths))

        cam_path = os.path.join(data_dir, "cameras.npz")
        all_cam = np.load(cam_path)
        all_imgs = []
        all_poses = []
        all_masks = []
        all_rays = []
        all_light_pose = []
        all_K = []
        all_R = []
        all_T = []

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):

            i = sel_indices[idx]
            rgb = load_rgb(rgb_path)
            mask = load_mask(mask_path)
            rgb[~mask] = 0.
            rgb = torch.from_numpy(rgb).float().to(device)
            mask = torch.from_numpy(mask).float().to(device)
            x_scale = y_scale = 1.0
            xy_delta = 0.0

            P = all_cam["world_mat_" + str(i)]
            P = P[:3]

            # scale the original shape to really [-0.9, 0.9]
            if fixed_scale!=0.:
                scale_mat_new = np.eye(4, 4)
                scale_mat_new[:3, :3] *= fixed_scale # scale to [-0.9, 0.9]
                P = all_cam["world_mat_" + str(i)] @ all_cam["scale_mat_" + str(i)] @ scale_mat_new
            else:
                P = all_cam["world_mat_" + str(i)] @ all_cam["scale_mat_" + str(i)]
            
            P = P[:3, :4]
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            K = K / K[2, 2]

            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            ########!!!!!
            RR = torch.from_numpy(R).permute(1, 0).unsqueeze(0)
            tt = torch.from_numpy(-R@(t[:3] / t[3])).permute(1, 0)
            focal = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)
            pc = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
            im_size = (rgb.shape[1], rgb.shape[0])
            
            # check https://pytorch3d.org/docs/cameras for how to transform from screen to NDC
            s = min(im_size)
            focal[:, 0] = focal[:, 0] * 2 / (s-1)
            focal[:, 1] = focal[:, 1] * 2 /(s-1)
            pc[:, 0] = -(pc[:, 0] - (im_size[0]-1)/2) * 2 / (s-1)
            pc[:, 1] = -(pc[:, 1] - (im_size[1]-1)/2) * 2 / (s-1)

            camera = PerspectiveCameras(focal_length=-focal, principal_point=pc, 
                                        device=device, R=RR, T=tt)

            # calculate camera rays
            uv = uv_creation(im_size)[None].float()
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R.transpose()
            pose[:3,3] = (t[:3] / t[3])[:,0]
            pose = torch.from_numpy(pose)[None].float()
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = K
            intrinsics[0, 1] = 0. #! remove skew for now
            intrinsics = torch.from_numpy(intrinsics)[None].float()
            

            rays, _ = get_camera_params(uv, pose, intrinsics)
            rays = -rays.to(device)

            
            
            all_poses.append(camera)
            all_imgs.append(rgb)
            all_masks.append(mask)
            all_rays.append(rays)
            all_light_pose.append(pose)
            # only for neural renderer
            all_K.append(torch.tensor(K).to(device))
            all_R.append(torch.tensor(R).to(device))
            all_T.append(torch.tensor(t[:3]/t[3]).to(device))

        all_imgs = torch.stack(all_imgs)
        all_masks = torch.stack(all_masks)
        all_rays = torch.stack(all_rays)
        all_light_pose = torch.stack(all_light_pose).squeeze()
        # only for neural renderer
        all_K = torch.stack(all_K).float()
        all_R = torch.stack(all_R).float()
        all_T = torch.stack(all_T).permute(0, 2, 1).float()

        uv = uv_creation((all_imgs.size(2), all_imgs.size(1)))
        self.data = {'rgbs': all_imgs,
                     'masks': all_masks,
                     'poses': all_poses,
                     'rays': all_rays,
                     'uv': uv,
                     'light_pose': all_light_pose, # for rendering lights
                     'K': all_K,
                     'R': all_R,
                     'T': all_T,
                     }
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data