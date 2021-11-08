import os
import torch
import time
import multiprocessing
import numpy as np
from tqdm import tqdm
from src.dpsr import DPSR

data_path = 'data/ShapeNet' # path for ShapeNet from ONet
base = 'data' # output base directory
dataset_name = 'shapenet_psr'
multiprocess = True
njobs = 8
save_pointcloud = True
save_psr_field = True
resolution = 128
zero_level = 0.0
num_points = 100000
padding = 1.2

dpsr = DPSR(res=(resolution, resolution, resolution), sig=0)

def process_one(obj):

    obj_name = obj.split('/')[-1]
    c = obj.split('/')[-2]

    # create new for the current object
    out_path_cur = os.path.join(base, dataset_name, c)
    out_path_cur_obj = os.path.join(out_path_cur, obj_name)
    os.makedirs(out_path_cur_obj, exist_ok=True)

    gt_path = os.path.join(data_path, c, obj_name, 'pointcloud.npz')
    data = np.load(gt_path)
    points = data['points']
    normals = data['normals']

    # normalize the point to [0, 1)
    points = points / padding + 0.5
    # to scale back during inference, we should:
    #! p = (p - 0.5) * padding
    
    if save_pointcloud:
        outdir = os.path.join(out_path_cur_obj, 'pointcloud.npz')
        # np.savez(outdir, points=points, normals=normals)
        np.savez(outdir, points=data['points'], normals=data['normals'])
        # return
    
    if save_psr_field:
        psr_gt = dpsr(torch.from_numpy(points.astype(np.float32))[None], 
                      torch.from_numpy(normals.astype(np.float32))[None]).squeeze().cpu().numpy().astype(np.float16)

        outdir = os.path.join(out_path_cur_obj, 'psr.npz')
        np.savez(outdir, psr=psr_gt)
    

def main(c):

    print('---------------------------------------')
    print('Processing {} {}'.format(c, split))
    print('---------------------------------------')

    for split in ['train', 'val', 'test']:
        fname = os.path.join(data_path, c, split+'.lst')
        with open(fname, 'r') as f:
            obj_list = f.read().splitlines() 
        
        obj_list = [c+'/'+s for s in obj_list]

        if multiprocess:
            # multiprocessing.set_start_method('spawn', force=True)
            pool = multiprocessing.Pool(njobs)
            try:
                for _ in tqdm(pool.imap_unordered(process_one, obj_list), total=len(obj_list)):
                    pass
                # pool.map_async(process_one, obj_list).get()
            except KeyboardInterrupt:
                # Allow ^C to interrupt from any thread.
                exit()
            pool.close()
        else:
            for obj in tqdm(obj_list):
                process_one(obj)
        
        print('Done Processing {} {}!'.format(c, split))
        
                
if __name__ == "__main__":

    classes = ['02691156', '02828884', '02933112', 
               '02958343', '03211117', '03001627',
               '03636649', '03691459', '04090263',
               '04256520', '04379243', '04401088', '04530566']

    
    t_start = time.time()
    for c in classes:
        main(c)
    
    t_end = time.time()
    print('Total processing time: ', t_end - t_start)
