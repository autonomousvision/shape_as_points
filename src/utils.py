import torch
import io, os, logging, urllib
import yaml
import trimesh
import imageio
import numbers
import math
import numpy as np
from collections import OrderedDict
from plyfile import PlyData
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from skimage import measure, img_as_float32
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, rasterize_meshes
from igl import adjacency_matrix, connected_components
import open3d as o3d

##################################################
# Below are functions for DPSR

def fftfreqs(res, dtype=torch.float32, exact=True):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes
    :return:
    """

    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)

    return omega

def img(x, deg=1): # imaginary of tensor (assume last dim: real/imag)
    """
    multiply tensor x by i ** deg
    """
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res

def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res, dtype=torch.float64) # [dim0, dim1, dim2, d]
    dis = torch.sqrt(torch.sum(omega ** 2, dim=-1))
    filter_ = torch.exp(-0.5*((sig*2*dis/res[0])**2)).unsqueeze(-1).unsqueeze(-1)
    filter_.requires_grad = False

    return filter_

def grid_interp(grid, pts, batched=True):
    """
    :param grid: tensor of shape (batch, *size, in_features)
    :param pts: tensor of shape (batch, num_points, dim) within range (0, 1)
    :return values at query points
    """
    if not batched:
        grid = grid.unsqueeze(0)
        pts = pts.unsqueeze(0)
    dim = pts.shape[-1]
    bs = grid.shape[0]
    size = torch.tensor(grid.shape[1:-1]).to(grid.device).type(pts.dtype)
    cubesize = 1.0 / size
    
    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long() # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0) # (2, batch, num_points, dim)
    tmp = torch.tensor([0,1],dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1) # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]   # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1) # (batch, num_points, 2**dim, dim)
    ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    # latent code on neighbor nodes
    if dim == 2:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1]] # (batch, num_points, 2**dim, in_features)
    else:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1], ind_n[..., 2]] # (batch, num_points, 2**dim, in_features)

    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize        # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0) # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1-com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False) # (batch, num_points, 2**dim)
    query_values = torch.sum(lat * weights.unsqueeze(-1), dim=-2)  # (batch, num_points, in_features)
    if not batched:
        query_values = query_values.squeeze(0)
        
    return query_values

def scatter_to_grid(inds, vals, size):
    """
    Scatter update values into empty tensor of size size.
    :param inds: (#values, dims)
    :param vals: (#values)
    :param size: tuple for size. len(size)=dims
    """
    dims = inds.shape[1]
    assert(inds.shape[0] == vals.shape[0])
    assert(len(size) == dims)
    dev = vals.device
    # result = torch.zeros(*size).view(-1).to(dev).type(vals.dtype)  # flatten
    # # flatten inds
    result = torch.zeros(*size, device=dev).view(-1).type(vals.dtype)  # flatten
    # flatten inds
    fac = [np.prod(size[i+1:]) for i in range(len(size)-1)] + [1]
    fac = torch.tensor(fac, device=dev).type(inds.dtype)
    inds_fold = torch.sum(inds*fac, dim=-1)  # [#values,]
    result.scatter_add_(0, inds_fold, vals)
    result = result.view(*size)
    return result

def point_rasterize(pts, vals, size):
    """
    :param pts: point coords, tensor of shape (batch, num_points, dim) within range (0, 1)
    :param vals: point values, tensor of shape (batch, num_points, features)
    :param size: len(size)=dim tuple for grid size
    :return rasterized values (batch, features, res0, res1, res2)
    """
    dim = pts.shape[-1]
    assert(pts.shape[:2] == vals.shape[:2])
    assert(pts.shape[2] == dim)
    size_list = list(size)
    size = torch.tensor(size).to(pts.device).float()
    cubesize = 1.0 / size
    bs = pts.shape[0]
    nf = vals.shape[-1]
    npts = pts.shape[1]
    dev = pts.device
    
    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long() # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0) # (2, batch, num_points, dim)
    tmp = torch.tensor([0,1],dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1) # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]   # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1) # (batch, num_points, 2**dim, dim)
    # ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    ind_b = torch.arange(bs, device=dev).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    
    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize        # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0) # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1-com_, ..., dim_].permute(2,3,0,1)   # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False) # (batch, num_points, 2**dim)
    
    ind_b = ind_b.unsqueeze(-1).unsqueeze(-1)      # (batch, num_points, 2**dim, 1, 1)
    ind_n = ind_n.unsqueeze(-2)                    # (batch, num_points, 2**dim, 1, dim)
    ind_f = torch.arange(nf, device=dev).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)
    # ind_f = torch.arange(nf).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)
    
    ind_b = ind_b.expand(bs, npts, 2**dim, nf, 1)
    ind_n = ind_n.expand(bs, npts, 2**dim, nf, dim).to(dev)
    ind_f = ind_f.expand(bs, npts, 2**dim, nf, 1)
    inds = torch.cat([ind_b, ind_f, ind_n], dim=-1)  # (batch, num_points, 2**dim, nf, 1+1+dim)
     
    # weighted values
    vals = weights.unsqueeze(-1) * vals.unsqueeze(-2)   # (batch, num_points, 2**dim, nf)
    
    inds = inds.view(-1, dim+2).permute(1, 0).long()  # (1+dim+1, bs*npts*2**dim*nf)
    vals = vals.reshape(-1) # (bs*npts*2**dim*nf)
    tensor_size = [bs, nf] + size_list
    raster = scatter_to_grid(inds.permute(1, 0), vals, [bs, nf] + size_list)
    
    return raster  # [batch, nf, res, res, res]



##################################################
# Below are the utilization functions in general

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.n = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.n = n
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def valcavg(self):
        return self.val.sum().item() / (self.n != 0).sum().item()

    @property
    def avgcavg(self):
        return self.avg.sum().item() / (self.count != 0).sum().item()

def load_model_manual(state_dict, model):
    new_state_dict = OrderedDict()
    is_model_parallel = isinstance(model, torch.nn.DataParallel)
    for k, v in state_dict.items():
        if k.startswith('module.') != is_model_parallel:
            if k.startswith('module.'):
                # remove module
                k = k[7:]
            else:
                # add module
                k = 'module.' + k

        new_state_dict[k]=v

    model.load_state_dict(new_state_dict)

def mc_from_psr(psr_grid, pytorchify=False, real_scale=False, zero_level=0):
    '''
    Run marching cubes from PSR grid
    '''
    batch_size = psr_grid.shape[0]
    s = psr_grid.shape[-1] # size of psr_grid
    psr_grid_numpy = psr_grid.squeeze().detach().cpu().numpy()
    
    if batch_size>1:
        verts, faces, normals = [], [], []
        for i in range(batch_size):
            verts_cur, faces_cur, normals_cur, values = measure.marching_cubes(psr_grid_numpy[i], level=0)
            verts.append(verts_cur)
            faces.append(faces_cur)
            normals.append(normals_cur)
        verts = np.stack(verts, axis = 0)
        faces = np.stack(faces, axis = 0)
        normals = np.stack(normals, axis = 0)
    else:
        try:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy, level=zero_level)
        except:
            verts, faces, normals, values = measure.marching_cubes(psr_grid_numpy)
    if real_scale:
        verts = verts / (s-1) # scale to range [0, 1]
    else:
        verts = verts / s # scale to range [0, 1)

    if pytorchify:
        device = psr_grid.device
        verts = torch.Tensor(np.ascontiguousarray(verts)).to(device)
        faces = torch.Tensor(np.ascontiguousarray(faces)).to(device)
        normals = torch.Tensor(np.ascontiguousarray(-normals)).to(device)

    return verts, faces, normals
        
def calc_inters_points(verts, faces, pose, img_size, mask_gt=None):
    verts = verts.squeeze()
    faces = faces.squeeze()
    pix_to_face, w, mask = mesh_rasterization(verts, faces, pose, img_size)
    if mask_gt is not None:
        #! only evaluate within the intersection
        mask = mask & mask_gt
    # find 3D points intesected on the mesh
    if True:
        w_masked = w[mask]
        f_p = faces[pix_to_face[mask]].long() # cooresponding faces for each pixel
        # corresponding vertices for p_closest
        v_a, v_b, v_c = verts[f_p[..., 0]], verts[f_p[..., 1]], verts[f_p[..., 2]]
        
        # calculate the intersection point of each pixel and the mesh
        p_inters = w_masked[..., 0, None] * v_a + \
                w_masked[..., 1, None] * v_b + \
                w_masked[..., 2, None] * v_c
    else:
        # backproject ndc to world coordinates using z-buffer
        W, H = img_size[1], img_size[0]
        xy = uv.to(mask.device)[mask]
        x_ndc = 1 - (2*xy[:, 0]) / (W - 1)
        y_ndc = 1 - (2*xy[:, 1]) / (H - 1)
        z = zbuf.squeeze().reshape(H * W)[mask]
        xy_depth = torch.stack((x_ndc, y_ndc, z), dim=1)
        
        p_inters = pose.unproject_points(xy_depth, world_coordinates=True)
    
        # if there are outlier points, we should remove it
        if (p_inters.max()>1) | (p_inters.min()<-1):
            mask_bound = (p_inters>=-1) & (p_inters<=1)
            mask_bound = (mask_bound.sum(dim=-1)==3)
            mask[mask==True] = mask_bound
            p_inters = p_inters[mask_bound]
            print('!!!!!find outlier!')

    return p_inters, mask, f_p, w_masked

def mesh_rasterization(verts, faces, pose, img_size):
    '''
    Use PyTorch3D to rasterize the mesh given a camera 
    '''
    transformed_v = pose.transform_points(verts.detach()) # world -> ndc coordinate system
    if isinstance(pose, PerspectiveCameras):
        transformed_v[..., 2] = 1/transformed_v[..., 2]
    # find p_closest on mesh of each pixel via rasterization
    transformed_mesh = Meshes(verts=[transformed_v], faces=[faces])
    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        transformed_mesh,
        image_size=img_size,
        blur_radius=0,
        faces_per_pixel=1,
        perspective_correct=False
    )
    pix_to_face = pix_to_face.reshape(1, -1) # B x reso x reso -> B x (reso x reso)
    mask = pix_to_face.clone() != -1
    mask = mask.squeeze()
    pix_to_face = pix_to_face.squeeze()
    w = bary_coords.reshape(-1, 3)

    return pix_to_face, w, mask

def verts_on_largest_mesh(verts, faces):
    '''
    verts: Numpy array or Torch.Tensor (N, 3)
    faces: Numpy array (N, 3)
    '''
    if torch.is_tensor(faces):
        verts = verts.squeeze().detach().cpu().numpy()
        faces = faces.squeeze().int().detach().cpu().numpy()

    A = adjacency_matrix(faces)
    num, conn_idx, conn_size = connected_components(A)
    if num == 0:
        v_large, f_large = verts, faces
    else:
        max_idx = conn_size.argmax() # find the index of the largest component
        v_large = verts[conn_idx==max_idx] # keep points on the largest component

        if True:
            mesh_largest = trimesh.Trimesh(verts, faces)
            connected_comp = mesh_largest.split(only_watertight=False)
            mesh_largest = connected_comp[max_idx]
            v_large, f_large = mesh_largest.vertices, mesh_largest.faces
            v_large = v_large.astype(np.float32)
    return v_large, f_large

def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices

# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

def update_config(config, unknown):
    # update config given args
    for idx,arg in enumerate(unknown):
        if arg.startswith("--"):
            keys = arg.replace("--","").split(':')
            assert(len(keys)==2)
            k1, k2 = keys
            argtype = type(config[k1][k2])
            if argtype == bool:
                v = unknown[idx+1].lower() == 'true'
            else:
                if config[k1][k2] is not None:
                    v = type(config[k1][k2])(unknown[idx+1])
                else:
                    v = unknown[idx+1]
            print(f'Changing {k1}:{k2} ---- {config[k1][k2]} to {v}')
            config[k1][k2] = v
    return config

def initialize_logger(cfg):
    out_dir = cfg['train']['out_dir']
    if not out_dir:
        os.makedirs(out_dir)
    
    cfg['train']['dir_model'] = os.path.join(out_dir, 'model')
    os.makedirs(cfg['train']['dir_model'], exist_ok=True)

    if cfg['train']['exp_mesh']:
        cfg['train']['dir_mesh'] = os.path.join(out_dir, 'vis/mesh')
        os.makedirs(cfg['train']['dir_mesh'], exist_ok=True)
    if cfg['train']['exp_pcl']:
        cfg['train']['dir_pcl'] = os.path.join(out_dir, 'vis/pointcloud')
        os.makedirs(cfg['train']['dir_pcl'], exist_ok=True)
    if cfg['train']['vis_rendering']:
        cfg['train']['dir_rendering'] = os.path.join(out_dir, 'vis/rendering')
        os.makedirs(cfg['train']['dir_rendering'], exist_ok=True)
    if cfg['train']['o3d_show']:
        cfg['train']['dir_o3d'] = os.path.join(out_dir, 'vis/o3d')
        os.makedirs(cfg['train']['dir_o3d'], exist_ok=True)


    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    # ch = logging.StreamHandler()
    # logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(cfg['train']['out_dir'], "log.txt"))
    logger.addHandler(fh)
    logger.info('Outout dir: %s', out_dir)
    return logger

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def export_pointcloud(name, points, normals=None):
    if len(points.shape) > 2:
        points = points[0]
        if normals is not None:
            normals = normals[0]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if normals is not None:
            normals = normals.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(name, pcd)

def export_mesh(name, v, f):
    if len(v.shape) > 2:
        v, f = v[0], f[0]
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    o3d.io.write_triangle_mesh(name, mesh)

def scale2onet(p, scale=1.2):
    '''
    Scale the point cloud from SAP to ONet range
    '''
    return (p - 0.5) * scale
    
def update_optimizer(inputs, cfg, epoch, model=None, schedule=None):
    if model is not None:
        if schedule is not None:
            optimizer = torch.optim.Adam([
                {"params": model.parameters(),
                "lr": schedule[0].get_learning_rate(epoch)},
                {"params": inputs,
                "lr": schedule[1].get_learning_rate(epoch)}])
        elif 'lr' in cfg['train']:
            optimizer = torch.optim.Adam([
                {"params": model.parameters(),
                    "lr": float(cfg['train']['lr'])},
                {"params": inputs,
                "lr": float(cfg['train']['lr_pcl'])}])
        else:
            raise Exception('no known learning rate')
    else:
        if schedule is not None:
            optimizer = torch.optim.Adam([inputs], lr=schedule[0].get_learning_rate(epoch))
        else:
            optimizer = torch.optim.Adam([inputs], lr=float(cfg['train']['lr_pcl']))

    return optimizer


def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

def load_url(url):
    '''Load a module dictionary from url.
    
    Args:
        url (str): url to saved model
    '''
    print(url)
    print('=> Loading checkpoint from url...')
    state_dict = model_zoo.load_url(url, progress=True)
    
    return state_dict


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
    
# Originally from https://github.com/amosgropp/IGR/blob/0db06b1273/code/utils/general.py
def get_learning_rate_schedules(schedule_specs):

    schedules = []

    for key in schedule_specs.keys():
        schedules.append(StepLearningRateSchedule(
                schedule_specs[key]['initial'],
                schedule_specs[key]["interval"],
                schedule_specs[key]["factor"], 
                schedule_specs[key]["final"]))
    return schedules

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass
class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor, final=1e-6):
        self.initial = float(initial)
        self.interval = interval
        self.factor = factor
        self.final = float(final)

    def get_learning_rate(self, epoch):
        lr = np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)
        if lr > self.final:
            return lr
        else:
            return self.final

def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
