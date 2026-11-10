import os

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms


class KITTI360(torch.utils.data.Dataset):
    '''
    KITTI-360 fisheye evaluation set (distance ground truth, 700 x 700)

    Arg(s):
        model_name : str
            backbone model name, selects the image normalization
        split_path : str
            text file listing "image_path distance_path" pairs
        mask_path : str
            image of the valid (inside the fisheye circle) region
        data_root : str
            folder containing KITTI-360/
    '''

    def __init__(self, model_name, split_path='splits/kitti360_test.txt', mask_path='splits/kitti_360_mask.png',
                 data_root='data'):
        super().__init__()

        height, width = 700, 700

        if model_name == 'midas':
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize([height, width], interpolation=transforms.InterpolationMode.BICUBIC,
                                        antialias=True)
        self.resize_depth = transforms.Resize([height, width], transforms.InterpolationMode.NEAREST_EXACT)

        mask = cv2.resize(cv2.imread(mask_path), (width, height), interpolation=cv2.INTER_LINEAR)
        self.mask = (mask.mean(2) > 150).astype(np.float32)

        self.rgb_path = os.path.join(data_root, 'KITTI-360', 'RGB')
        self.gt_path = os.path.join(data_root, 'KITTI-360', 'Depth')

        with open(split_path, 'r') as f:
            lines = f.readlines()

        # only the 90 degree field of view crops are evaluated
        self.filenames = [line.split()[:2] for line in lines if '_90' in line.split()[0]]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name, depth_name = self.filenames[idx]

        image = Image.open(os.path.join(self.rgb_path, image_name.lstrip('/\\')))
        depth_gt = np.load(os.path.join(self.gt_path, depth_name.lstrip('/\\')))

        image = np.array(self.resize(image), copy=True)
        image = self.normalize(self.to_tensor(image))

        depth_gt = np.asarray(self.resize_depth(Image.fromarray(depth_gt)))
        depth_gt = np.array(depth_gt, copy=True, dtype=np.float32) * self.mask
        depth_gt = self.to_tensor(np.copy(depth_gt))

        return image, depth_gt
