import torch
import trimesh
from torch.utils.data import Dataset, DataLoader
import numpy as np; np.set_printoptions(precision=4)
import shutil, argparse, time, os
import pandas as pd
from src.data import collate_remove_none, collate_stack_together, worker_init_fn
from src.training import Trainer
from src.model import Encode2Points
from src.data import PointCloudField, IndexField, Shapes3dDataset
from src.utils import load_config, load_pointcloud
from src.eval import MeshEvaluator
from tqdm import tqdm
from pdb import set_trace as st


def main():
    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')    
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--iter', type=int, metavar='S', help='the training iteration to be evaluated.')
    
    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    data_type = cfg['data']['data_type']
    # Shorthands
    out_dir = cfg['train']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])

    if cfg['generation'].get('iter', 0)!=0:
        generation_dir += '_%04d'%cfg['generation']['iter']
    elif args.iter is not None:
        generation_dir += '_%04d'%args.iter
            
    print('Evaluate meshes under %s'%generation_dir)
    
    out_file = os.path.join(generation_dir, 'eval_meshes_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_meshes.csv')
    
    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    pointcloud_field = PointCloudField(cfg['data']['pointcloud_file'])
    fields = {
        'pointcloud': pointcloud_field,
        'idx': IndexField(),
    }

    print('Test split: ', cfg['data']['test_split'])

    dataset_folder = cfg['data']['path']
    dataset = Shapes3dDataset(
        dataset_folder, fields,
        cfg['data']['test_split'],
        categories=cfg['data']['class'], cfg=cfg)
    
    # Loader
    test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)
    
    # Evaluator
    evaluator = MeshEvaluator(n_points=100000)

    eval_dicts = []   
    print('Evaluating meshes...')
    for it, data in enumerate(tqdm(test_loader)):

        if data is None:
            print('Invalid data.')
            continue

        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')

        
        # Get index etc.
        idx = data['idx'].item()
        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}

        modelname = model_dict['model']
        category_id = model_dict['category']

        try:
            category_name = dataset.metadata[category_id].get('name', 'n/a')
        except AttributeError:
            category_name = 'n/a'

        if category_id != 'n/a':
            mesh_dir = os.path.join(mesh_dir, category_id)
            pointcloud_dir = os.path.join(pointcloud_dir, category_id)

        # Evaluate
        pointcloud_tgt = data['pointcloud'].squeeze(0).numpy()
        normals_tgt = data['pointcloud.normals'].squeeze(0).numpy()

        
        eval_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname':modelname,
        }
        eval_dicts.append(eval_dict)
        
        # Evaluate mesh
        if cfg['test']['eval_mesh']:
            mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)

            if os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file, process=False)
                eval_dict_mesh = evaluator.eval_mesh(
                    mesh, pointcloud_tgt, normals_tgt)
                for k, v in eval_dict_mesh.items():
                    eval_dict[k + ' (mesh)'] = v
            else:
                print('Warning: mesh does not exist: %s' % mesh_file)

        # Evaluate point cloud
        if cfg['test']['eval_pointcloud']:
            pointcloud_file = os.path.join(
                pointcloud_dir, '%s.ply' % modelname)

            if os.path.exists(pointcloud_file):
                pointcloud = load_pointcloud(pointcloud_file).astype(np.float32)
                eval_dict_pcl = evaluator.eval_pointcloud(
                    pointcloud, pointcloud_tgt)
                for k, v in eval_dict_pcl.items():
                    eval_dict[k + ' (pcl)'] = v
            else:
                print('Warning: pointcloud does not exist: %s'
                        % pointcloud_file)
            
        
    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)
    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class name']).mean()
    eval_df_class.loc['mean'] = eval_df_class.mean()
    eval_df_class.to_csv(out_file_class)

    # Print results
    print(eval_df_class)

if __name__ == '__main__':
    main()