import argparse
import os
import numpy as np
import random
import time
from collections import defaultdict
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from utils import Logger
from dataset import KITTI, NYUD2, SYNWoodScape, KITTI_360, SCANNETPP

from external_model import ExternalMonocularDepthEstimationModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate monocular depth estimation (DepthAnything or MiDaS)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--result_dir', type=str, default='',
                        help='Where to store logs/results.')
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'),
                        help='Base path of your dataset images.')
    parser.add_argument('-j', '--workers', default=4, type=int, help='Data loading workers.')
    parser.add_argument('--batch_size', default=32, type=int, help='Mini-batch size.')
    parser.add_argument('--dataset', type=str, default="NYU",
                        help='Which dataset')
    parser.add_argument('--patch_size', type=int, default=14,
                    help='Patch size for the model (e.g., 14, 16).')


    parser.add_argument('--model', type=str, default='depthanything',
                        help='Which external model')


    args = parser.parse_args()
    return args


import cv2

def normalize_255(depth):
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-8: 
        return np.zeros_like(depth, dtype=np.uint8)
    norm = (depth - d_min) / (d_max - d_min)
    return (norm * 255).astype(np.uint8)


def save_depth_comparison(depth_pred, depth_gt, save_path):
    pred_np = depth_pred.squeeze().detach().cpu().numpy()
    gt_np   = depth_gt.squeeze().detach().cpu().numpy()

    p_255 = normalize_255(pred_np)
    g_255 = normalize_255(gt_np)

    pred_color = cv2.applyColorMap(p_255, cv2.COLORMAP_INFERNO)
    gt_color   = cv2.applyColorMap(g_255, cv2.COLORMAP_INFERNO)

    comparison = np.concatenate([pred_color, gt_color], axis=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, comparison)


def evaluate(args, model, test_loader, logger):
    """
    Evaluate the model on test_loader, including flip TTA,
    and save side-by-side color maps of prediction vs. GT.
    """
    
    model.eval()
    total_error = defaultdict(float)

    vis_dir = os.path.join(args.result_dir, 'comparison_vis')
    os.makedirs(vis_dir, exist_ok=True)

    sample_index = 0
    for batch_i, (rgb_data, gt_data) in enumerate(tqdm(test_loader)):
        rgb_data = rgb_data.cuda()

        with torch.no_grad():
            # Forward pass
            depth_pred = model.forward(rgb_data)
            rgb_flipped = torch.flip(rgb_data, dims=[-1])
            depth_pred_flipped = model.forward(rgb_flipped)
            depth_pred_flipped = torch.flip(depth_pred_flipped, dims=[-1])
            # Final prediction
            depth_pred = (depth_pred + depth_pred_flipped) / 2

            # depth_pred *= 10/256
        
        errors = logger.compute_depth_losses(depth_pred, gt_data)
        for key in logger.metric_names:
            total_error[key] += errors[key]

        batch_size = rgb_data.size(0)
        for i in range(batch_size):
            pred_i = depth_pred[i:i+1]  
            gt_i   = gt_data[i:i+1]   
            save_path = os.path.join(vis_dir, f"compare_{sample_index:05d}.png")
            save_depth_comparison(pred_i, gt_i, save_path)
            sample_index += 1

    n_batches = len(test_loader)
    for key in logger.metric_names:
        total_error[key] /= n_batches
    logger.print_perf(24, total_error)



def main(rank, args):

    if args.dataset == 'KITTI':
        args.min_depth = 0.01
        args.max_depth = 80.0
        args.kitti_split_eval = os.path.join(
            os.path.dirname(__file__),
            'dataset', 'splits', 'kitti_eigen_test.txt'
        )
        eval_set = KITTI(args, mode='test')

    elif args.dataset == 'NYU':
        args.min_depth = 0.01
        args.max_depth = 10.0
        args.nyu_split_eval = os.path.join(
            os.path.dirname(__file__),
            'dataset', 'splits', 'nyu_test.txt'
        )
        eval_set = NYUD2(args, mode='test')

    elif args.dataset == 'SynWoodScape':
        args.min_depth = 0.01
        args.max_depth = 40.0
        args.sws_split_eval = os.path.join(
            os.path.dirname(__file__),
            'dataset', 'splits', 'synwoodscape_test.txt'
        )
        eval_set = SYNWoodScape(args, mode='test')

    elif args.dataset == 'KITTI_360':
        args.min_depth = 0.01
        args.max_depth = 40.0
        args.k360_split_eval = os.path.join(
            os.path.dirname(__file__),
            'dataset', 'splits', 'kitti360_test.txt'
        )
        eval_set = KITTI_360(args, mode='test')

    elif args.dataset == 'SCANNETPP':
        args.min_depth = 0.01
        args.max_depth = 20.0
        eval_set = SCANNETPP(args, mode='test')

    if args.result_dir == '':
        args.result_dir = './workspace/' + args.dataset + '_eval_results'

    print("=> Dataset:", args.dataset)

    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    print('=> Number of evaluation samples:', len(eval_set))

    logger = Logger(
        args.batch_size,
        dataset=args.dataset,
        log_path=args.result_dir,
        eval_visualize=True
    )

    print(f"=> Creating external monocular model: {args.model}")

    model = ExternalMonocularDepthEstimationModel(
        model_name=args.model,
        min_predict_depth=args.min_depth,
        max_predict_depth=args.max_depth,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )


    print(f"DepthAnythingBaseModel initialized: {model}")


    model.to(rank)
    print("Model Initialized")

    evaluate(args, model, eval_loader, logger)

    print("Done")
    logger.close_tb()


if __name__ == "__main__":
    args = parse_args()
    main(0, args)
