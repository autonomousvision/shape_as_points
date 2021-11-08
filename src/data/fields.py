import os
import glob
import time
import random
from PIL import Image
import numpy as np
import trimesh
from src.data.core import Field
from pdb import set_trace as st


class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

class FullPSRField(Field):
    def __init__(self, transform=None, multi_files=None):
        self.transform = transform
        # self.unpackbits = unpackbits
        self.multi_files = multi_files
    
    def load(self, model_path, idx, category):

        # try:
        # t0 = time.time()
        if self.multi_files is not None:
            psr_path = os.path.join(model_path, 'psr', 'psr_{:02d}.npz'.format(idx))
        else:
            psr_path = os.path.join(model_path, 'psr.npz')
        psr_dict = np.load(psr_path)
        # t1 = time.time()
        psr = psr_dict['psr']
        psr = psr.astype(np.float32)
        # t2 = time.time()
        # print('load PSR: {:.4f}, change type: {:.4f}, total: {:.4f}'.format(t1 - t0, t2 - t1, t2-t0))
        data = {None: psr}
        
        if self.transform is not None:
            data = self.transform(data)

        return data

class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, data_type=None, transform=None, multi_files=None, padding=0.1, scale=1.2):
        self.file_name = file_name
        self.data_type = data_type # to make sure the range of input is correct
        self.transform = transform
        self.multi_files = multi_files
        self.padding = padding
        self.scale = scale

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            # num = np.random.randint(self.multi_files)
            # file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
            file_path = os.path.join(model_path, self.file_name, 'pointcloud_%02d.npz' % (idx))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)
        
        data = {
            None: points,
            'normals': normals,
        }
        if self.transform is not None:
            data = self.transform(data)
        
        if self.data_type == 'psr_full':
            # scale the point cloud to the range of (0, 1)
            data[None] = data[None] / self.scale + 0.5

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
