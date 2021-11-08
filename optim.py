import torch
import trimesh
import shutil, argparse, time, os, glob

import numpy as np; np.set_printoptions(precision=4)
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision.io import write_video

from src.optimization import Trainer
from src.utils import load_config, update_config, initialize_logger, \
    get_learning_rate_schedules, adjust_learning_rate, AverageMeter,\
         update_optimizer, export_pointcloud
from skimage import measure
from plyfile import PlyData
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes


def main():
    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')    
    parser.add_argument('--seed', type=int, default=1457, metavar='S', 
                        help='random seed')
    
    args, unknown = parser.parse_known_args() 
    cfg = load_config(args.config, 'configs/default.yaml')
    cfg = update_config(cfg, unknown)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    data_type = cfg['data']['data_type']
    data_class = cfg['data']['class']

    print(cfg['train']['out_dir'])

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    # boiler-plate
    if cfg['train']['timestamp']:
        cfg['train']['out_dir'] += '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = initialize_logger(cfg)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    shutil.copyfile(args.config, 
                    os.path.join(cfg['train']['out_dir'], 'config.yaml'))

    # tensorboardX writer
    tblogdir = os.path.join(cfg['train']['out_dir'], "tensorboard_log")
    if not os.path.exists(tblogdir):
        os.makedirs(tblogdir)
    writer = SummaryWriter(log_dir=tblogdir)

    # initialize o3d visualizer
    vis = None
    if cfg['train']['o3d_show']:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=cfg['train']['o3d_window_size'], 
                          height=cfg['train']['o3d_window_size'])

    # initialize dataset
    if data_type == 'point':
        if cfg['data']['object_id'] != -1:
            data_paths = sorted(glob.glob(cfg['data']['data_path']))
            data_path = data_paths[cfg['data']['object_id']]
            print('Loaded %d/%d object' % (cfg['data']['object_id']+1, len(data_paths)))
        else:
            data_path = cfg['data']['data_path']
            print('Data loaded')
        ext = data_path.split('.')[-1]
        if ext == 'obj': # have GT mesh
            mesh = load_objs_as_meshes([data_path], device=device)
            # scale the mesh into unit cube
            verts = mesh.verts_packed()
            N = verts.shape[0]
            center = verts.mean(0)
            mesh.offset_verts_(-center.expand(N, 3))
            scale = max((verts - center).abs().max(0)[0])
            mesh.scale_verts_((1.0 / float(scale)))
            # important for our DPSR to have the range in [0, 1), not reaching 1
            mesh.scale_verts_(0.9)

            target_pts, target_normals = sample_points_from_meshes(mesh, 
                                            num_samples=200000, return_normals=True)
        elif ext == 'ply': # only have the point cloud
            plydata = PlyData.read(data_path)
            vertices = np.stack([plydata['vertex']['x'],
                                    plydata['vertex']['y'],
                                    plydata['vertex']['z']], axis=1)
            normals = np.stack([plydata['vertex']['nx'],
                                plydata['vertex']['ny'],
                                plydata['vertex']['nz']], axis=1)
            N = vertices.shape[0]
            center = vertices.mean(0)
            scale = np.max(np.max(np.abs(vertices - center), axis=0))
            vertices -= center
            vertices /= scale
            vertices *= 0.9

            target_pts = torch.tensor(vertices, device=device)[None].float()
            target_normals = torch.tensor(normals, device=device)[None].float()
            mesh = None # no GT mesh

        if not torch.is_tensor(center):
            center = torch.from_numpy(center)
        if not torch.is_tensor(scale):
            scale = torch.from_numpy(np.array([scale]))

        data = {'target_points': target_pts,
                'target_normals': target_normals, # normals are never used
                'gt_mesh': mesh}

    else:
        raise NotImplementedError

    # save the input point cloud
    if 'target_points' in data.keys():
        outdir_pcl = os.path.join(cfg['train']['out_dir'], 'target_pcl.ply')
        if 'target_normals' in data.keys():
            export_pointcloud(outdir_pcl, data['target_points'], data['target_normals'])
        else:
            export_pointcloud(outdir_pcl, data['target_points'])

    # save oracle PSR mesh (mesh from our PSR using GT point+normals)
    if data.get('gt_mesh') is not None:
        gt_verts, gt_faces = data['gt_mesh'].get_mesh_verts_faces(0)
        pts_gt, norms_gt = sample_points_from_meshes(data['gt_mesh'], 
                                    num_samples=500000, return_normals=True)
        pts_gt = (pts_gt + 1) / 2
        from src.dpsr import DPSR
        dpsr_tmp = DPSR(res=(cfg['model']['grid_res'], 
                            cfg['model']['grid_res'], 
                            cfg['model']['grid_res']), 
                        sig=cfg['model']['psr_sigma']).to(device)
        target = dpsr_tmp(pts_gt, norms_gt).unsqueeze(1).to(device)
        target = torch.tanh(target)
        s = target.shape[-1] # size of psr_grid
        psr_grid_numpy = target.squeeze().detach().cpu().numpy()
        verts, faces, _, _ = measure.marching_cubes(psr_grid_numpy)
        verts = verts / s * 2. - 1 # [-1, 1]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        outdir_mesh = os.path.join(cfg['train']['out_dir'], 'oracle_mesh.ply')
        o3d.io.write_triangle_mesh(outdir_mesh, mesh)

    # initialize the source point cloud given an input mesh
    if 'input_mesh' in cfg['train'].keys() and \
                    os.path.isfile(cfg['train']['input_mesh']):
        if cfg['train']['input_mesh'].split('/')[-2] == 'mesh':
            mesh_tmp = trimesh.load_mesh(cfg['train']['input_mesh'])
            verts = torch.from_numpy(mesh_tmp.vertices[None]).float().to(device)
            faces = torch.from_numpy(mesh_tmp.faces[None]).to(device)
            mesh = Meshes(verts=verts, faces=faces)
            points, normals = sample_points_from_meshes(mesh, 
                        num_samples=cfg['data']['num_points'], return_normals=True)
            # mesh is saved in the original scale of the gt
            points -= center.float().to(device)
            points /= scale.float().to(device)
            points *= 0.9
            # make sure the points are within the range of [0, 1)
            points = points / 2. + 0.5
        else:
            # directly initialize from a point cloud
            pcd = o3d.io.read_point_cloud(cfg['train']['input_mesh'])
            points = torch.from_numpy(np.array(pcd.points)[None]).float().to(device)
            normals = torch.from_numpy(np.array(pcd.normals)[None]).float().to(device)
            points -= center.float().to(device)
            points /= scale.float().to(device)
            points *= 0.9
            points = points / 2. + 0.5
    else: #! initialize our source point cloud from a sphere
        sphere_radius = cfg['model']['sphere_radius']
        sphere_mesh = trimesh.creation.uv_sphere(radius=sphere_radius, 
                                                 count=[256,256])
        points, idx = sphere_mesh.sample(cfg['data']['num_points'], 
                                         return_index=True)
        points += 0.5 # make sure the points are within the range of [0, 1)
        normals = sphere_mesh.face_normals[idx]
        points = torch.from_numpy(points).unsqueeze(0).to(device)
        normals = torch.from_numpy(normals).unsqueeze(0).to(device)

    
    points = torch.log(points/(1-points)) # inverse sigmoid
    inputs = torch.cat([points, normals], axis=-1).float()
    inputs.requires_grad = True

    model = None # no network
    
    # initialize optimizer
    cfg['train']['schedule']['pcl']['initial'] = cfg['train']['lr_pcl']
    print('Initial learning rate:', cfg['train']['schedule']['pcl']['initial'])
    if 'schedule' in cfg['train']:
        lr_schedules = get_learning_rate_schedules(cfg['train']['schedule'])
    else:
        lr_schedules = None

    optimizer = update_optimizer(inputs, cfg, 
                            epoch=0, model=model, schedule=lr_schedules)

    try:
        # load model
        state_dict = torch.load(os.path.join(cfg['train']['out_dir'], 'model.pt'))
        if ('pcl' in state_dict.keys()) & (state_dict['pcl'] is not None):
            inputs = state_dict['pcl'].to(device)
            inputs.requires_grad = True
        
        optimizer = update_optimizer(inputs, cfg, 
                                epoch=state_dict.get('epoch'), schedule=lr_schedules)
            
        out = "Load model from epoch %d" % state_dict.get('epoch', 0)
        print(out)
        logger.info(out)
    except:
        state_dict = dict()

    start_epoch = state_dict.get('epoch', -1)

    trainer = Trainer(cfg, optimizer, device=device)
    runtime = {}
    runtime['all'] = AverageMeter()
    
    # training loop
    for epoch in range(start_epoch+1, cfg['train']['total_epochs']+1):
        
        # schedule the learning rate
        if (epoch>0) & (lr_schedules is not None):
            if (epoch % lr_schedules[0].interval == 0):
                adjust_learning_rate(lr_schedules, optimizer, epoch)
                if len(lr_schedules) >1:
                    print('[epoch {}] net_lr: {}, pcl_lr: {}'.format(epoch, 
                                        lr_schedules[0].get_learning_rate(epoch), 
                                        lr_schedules[1].get_learning_rate(epoch)))
                else:
                    print('[epoch {}] adjust pcl_lr to: {}'.format(epoch, 
                                        lr_schedules[0].get_learning_rate(epoch)))

        start = time.time()
        loss, loss_each = trainer.train_step(data, inputs, model, epoch)
        runtime['all'].update(time.time() - start)

        if epoch % cfg['train']['print_every'] == 0:
            log_text = ('[Epoch %02d] loss=%.5f') %(epoch, loss)
            if loss_each is not None:
                for k, l in loss_each.items():
                    if l.item() != 0.:
                        log_text += (' loss_%s=%.5f') % (k, l.item())
            
            log_text += (' time=%.3f / %.3f') % (runtime['all'].val, 
                                                 runtime['all'].sum)
            logger.info(log_text)
            print(log_text)

        # visualize point clouds and meshes
        if (epoch % cfg['train']['visualize_every'] == 0) & (vis is not None):
            trainer.visualize(data, inputs, model, epoch, o3d_vis=vis)
        
        # save outputs
        if epoch % cfg['train']['save_every'] == 0:
            trainer.save_mesh_pointclouds(inputs, epoch, 
                                        center.cpu().numpy(), 
                                        scale.cpu().numpy()*(1/0.9))

        # save checkpoints
        if (epoch > 0) & (epoch % cfg['train']['checkpoint_every'] == 0):
            state = {'epoch': epoch}
            pcl = None
            if isinstance(inputs, torch.Tensor):
                state['pcl'] = inputs.detach().cpu()
            
            torch.save(state, os.path.join(cfg['train']['dir_model'], 
                                                '%04d' % epoch + '.pt'))
            print("Save new model at epoch %d" % epoch)
            logger.info("Save new model at epoch %d" % epoch)
            torch.save(state, os.path.join(cfg['train']['out_dir'], 'model.pt'))
        
        # resample and gradually add new points to the source pcl
        if (epoch > 0) & \
           (cfg['train']['resample_every']!=0) & \
           (epoch % cfg['train']['resample_every'] == 0) & \
           (epoch < cfg['train']['total_epochs']):
                inputs = trainer.point_resampling(inputs)
                optimizer = update_optimizer(inputs, cfg, 
                            epoch=epoch, model=model, schedule=lr_schedules)
                trainer = Trainer(cfg, optimizer, device=device)
    
    # visualize the Open3D outputs
    if cfg['train']['o3d_show']:
        out_video_dir = os.path.join(cfg['train']['out_dir'], 
                                        'vis/o3d/video.mp4')
        if os.path.isfile(out_video_dir):
            os.system('rm {}'.format(out_video_dir))
        os.system('ffmpeg -framerate 30 \
                    -start_number 0 \
                    -i {}/vis/o3d/%04d.jpg \
                    -pix_fmt yuv420p \
                    -crf 17 {}'.format(cfg['train']['out_dir'], out_video_dir))
        out_video_dir = os.path.join(cfg['train']['out_dir'], 
                                        'vis/o3d/video_pcd.mp4')
        if os.path.isfile(out_video_dir):
            os.system('rm {}'.format(out_video_dir))
        os.system('ffmpeg -framerate 30 \
                    -start_number 0 \
                    -i {}/vis/o3d/%04d_pcd.jpg \
                    -pix_fmt yuv420p \
                    -crf 17 {}'.format(cfg['train']['out_dir'], out_video_dir))
        print('Video saved.')

if __name__ == '__main__':
    main()
