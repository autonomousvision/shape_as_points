import yaml
from torchvision import transforms
from src import data, generation
from src.dpsr import DPSR
from ipdb import set_trace as st


# Generator for final mesh extraction
def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    
    if cfg['generation']['psr_resolution'] == 0:
        psr_res = cfg['model']['grid_res']
        psr_sigma = cfg['model']['psr_sigma']
    else:
        psr_res = cfg['generation']['psr_resolution']
        psr_sigma = cfg['generation']['psr_sigma']
    
    dpsr = DPSR(res=(psr_res, psr_res, psr_res), 
                sig= psr_sigma).to(device)

    
    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['data']['zero_level'],
        sample=cfg['generation']['use_sampling'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        dpsr=dpsr,
        psr_tanh=cfg['model']['psr_tanh']
    )
    return generator

# Datasets
def get_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['class']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'vis': cfg['data']['val_split'],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        fields = get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
            cfg = cfg
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
 
    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        noise_level = cfg['data']['pointcloud_noise']
        if cfg['data']['pointcloud_outlier_ratio']>0:
            transform = transforms.Compose([
                data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
                data.PointcloudNoise(noise_level),
                data.PointcloudOutliers(cfg['data']['pointcloud_outlier_ratio'])
            ])
        else:
            transform = transforms.Compose([
                data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
                data.PointcloudNoise(noise_level)
            ])

        data_type = cfg['data']['data_type']
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], data_type, transform,
            multi_files= cfg['data']['multi_files']
        )    
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field

def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    data_type = cfg['data']['data_type']
    fields = {}
    
    if (mode in ('val', 'test')):
        transform = data.SubsamplePointcloud(100000)
    else:
        transform = data.SubsamplePointcloud(cfg['data']['num_gt_points'])
    
    data_name = cfg['data']['pointcloud_file']
    fields['gt_points'] = data.PointCloudField(data_name, 
                transform=transform, data_type=data_type, multi_files=cfg['data']['multi_files'])
    if data_type == 'psr_full':
        if mode != 'test':
            fields['gt_psr'] = data.FullPSRField(multi_files=cfg['data']['multi_files'])
    else:
        raise ValueError('Invalid data type (%s)' % data_type)

    return fields