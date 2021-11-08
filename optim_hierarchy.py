import sys, os
import argparse
from src.utils import load_config
import subprocess
os.environ['MKL_THREADING_LAYER'] = 'GNU'

def main():

    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--start_res', type=int, default=-1, help='Resolution to start with.')
    parser.add_argument('--object_id', type=int, default=-1, help='Object index.')

    args, unknown = parser.parse_known_args() 
    cfg = load_config(args.config, 'configs/default.yaml')

    resolutions=[32, 64, 128, 256]
    iterations=[1000, 1000, 1000, 200]
    lrs=[2e-3, 2e-3*0.7, 2e-3*(0.7**2), 2e-3*(0.7**3)] # reduce lr
    for idx,(res, iteration, lr) in enumerate(zip(resolutions, iterations, lrs)):

        if res<args.start_res:
            continue

        if res>cfg['model']['grid_res']:
            continue

        psr_sigma= 2 if res<=128 else 3
        
        if res > 128:
            psr_sigma = 5 if 'thingi_noisy' in args.config else 3

        if args.object_id != -1:
            out_dir = os.path.join(cfg['train']['out_dir'], 'object_%02d'%args.object_id, 'res_%d'%res)
        else:
            out_dir = os.path.join(cfg['train']['out_dir'], 'res_%d'%res)
        
        # sample from mesh when resampling is enabled, otherwise reuse the pointcloud
        init_shape='mesh' if cfg['train']['resample_every']>0 else 'pointcloud'
                
        
        if args.object_id != -1:
            input_mesh='None' if idx==0 else os.path.join(cfg['train']['out_dir'], 
                            'object_%02d'%args.object_id, 'res_%d' % (resolutions[idx-1]), 
                            'vis', init_shape, '%04d.ply' % (iterations[idx-1]))
        else:
            input_mesh='None' if idx==0 else os.path.join(cfg['train']['out_dir'],
                            'res_%d' % (resolutions[idx-1]), 
                            'vis', init_shape, '%04d.ply' % (iterations[idx-1]))
        
        
        cmd = 'export MKL_SERVICE_FORCE_INTEL=1 && '
        cmd += "python optim.py %s --model:grid_res %d --model:psr_sigma %d \
                                   --train:input_mesh %s --train:total_epochs %d \
                                   --train:out_dir %s --train:lr_pcl %f \
                                   --data:object_id %d" % (
                                           args.config,
                                           res,
                                           psr_sigma,
                                           input_mesh,
                                           iteration,
                                           out_dir,
                                           lr,
                                           args.object_id)
        print(cmd)
        os.system(cmd)

if __name__=="__main__":
    main()
