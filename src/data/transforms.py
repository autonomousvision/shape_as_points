import numpy as np


# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out

class PointcloudOutliers(object):
    ''' Point cloud outlier transformation class.

    It adds outliers to point cloud data.

    Args:
        ratio (int): outlier percentage to the entire point cloud
    '''

    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        n_points = points.shape[0]
        n_outlier_points = int(n_points*self.ratio)
        ind = np.random.randint(0, n_points, n_outlier_points)
        
        outliers = np.random.uniform(-0.55, 0.55, (n_outlier_points, 3))
        outliers = outliers.astype(np.float32)
        points[ind] = outliers
        data_out[None] = points
        return data_out

class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]
        if 'normals' in data.keys():
            normals = data['normals']
            data_out['normals'] = normals[indices, :]

        return data_out