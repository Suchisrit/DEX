import os

import numpy as np
from PIL import Image

import torch
from torchvision import transforms


def downsample_depth_to_size_min(depth_map, out_height, out_width):
    '''Downsample a depth map to (out_height, out_width) with min pooling'''

    in_height, in_width = depth_map.shape
    scale_y = in_height / out_height
    scale_x = in_width / out_width

    output = np.zeros((out_height, out_width), dtype=depth_map.dtype)

    for y in range(out_height):
        for x in range(out_width):
            y0 = int(y * scale_y)
            y1 = int((y + 1) * scale_y)
            x0 = int(x * scale_x)
            x1 = int((x + 1) * scale_x)

            patch = depth_map[y0:y1, x0:x1]
            if patch.size != 0:
                output[y, x] = np.min(patch)

    return output


class ScanNetPP(torch.utils.data.Dataset):
    '''
    ScanNet++ DSLR fisheye evaluation set

    Arg(s):
        model_name : str
            backbone model name, selects the image normalization
        eval_shape : tuple[int, int]
            H, W to resize the image and ground truth to
        split_path : str
            text file listing "image_path depth_path" pairs
        data_root : str
            folder containing scannetplusplus/ and scannetplusplus_depth/
    '''

    def __init__(self, model_name, eval_shape, split_path='splits/scannetpp_test.txt', data_root='data'):
        super().__init__()

        self.depth_scale = 1000.0
        self.height, self.width = eval_shape

        if model_name == 'midas':
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize([self.height, self.width])

        self.rgb_paths = []
        self.gt_paths = []
        self.intrinsics_paths = []

        with open(split_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            image, depth = line.split(' ')[:2]
            scene_name = depth.split('/')[1]

            self.rgb_paths.append(os.path.join(data_root, 'scannetplusplus', image))
            self.gt_paths.append(os.path.join(data_root, 'scannetplusplus_depth', *depth.split('/')[1:]))
            self.intrinsics_paths.append(
                os.path.join(data_root, 'scannetplusplus/data/', scene_name, 'dslr/colmap/cameras.txt'))

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # 4th line of the COLMAP cameras file: fx fy cx cy k1 k2 k3 k4
        with open(self.intrinsics_paths[idx], 'r') as f:
            params = f.readlines()[3].split(' ')

        K = np.eye(3)
        K[0, 0] = params[4]
        K[1, 1] = params[5]
        K[0, 2] = params[6]
        K[1, 2] = params[7]
        D = np.array([float(params[8]), float(params[9]), float(params[10]), float(params[11].split('\n')[0])])

        image = Image.open(self.rgb_paths[idx])
        depth_gt = Image.open(self.gt_paths[idx])

        sx = self.width / image.size[0]
        sy = self.height / image.size[1]
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        image = np.array(self.resize(image), copy=True)
        image = self.normalize(self.to_tensor(image))

        depth_gt = np.array(depth_gt, copy=True, dtype=np.float32) / self.depth_scale
        depth_gt = downsample_depth_to_size_min(depth_gt, self.height, self.width)
        depth_gt = self.to_tensor(np.copy(depth_gt))

        return image, depth_gt, K, D
