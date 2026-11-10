import argparse
import random

import numpy as np
import torch

from model import Model
from utils.dataset import KITTI360, ScanNetPP
from utils.depth_evaluate import METRIC_NAMES, evaluate

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Distortion Extenders on ScanNet++ and KITTI-360 fisheye images')

    parser.add_argument('--num_extenders', type=int, required=True,
                        help='Number of Distortion Extenders (must match the checkpoint)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the checkpoint .pth file')
    parser.add_argument('--model_name', type=str, default='unidepth',
                        choices=['unidepth', 'depthanything', 'midas'],
                        help='Backbone depth model name')
    parser.add_argument('--datasets', nargs='+', type=str, default=['scannetpp', 'kitti360'],
                        choices=['scannetpp', 'kitti360'],
                        help='Datasets to evaluate on')
    parser.add_argument('--eval_shape', nargs=2, type=int, default=[462, 616], metavar=('H', 'W'),
                        help='H W to resize ScanNet++ to (KITTI-360 is always 700 x 700)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Evaluation batch size')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Folder containing scannetplusplus/, scannetplusplus_depth/ and KITTI-360/')
    parser.add_argument('--n_thread', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (e.g. cuda, cuda:0)')

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(args.device)

    model = Model(
        num_extenders=args.num_extenders,
        model_name=args.model_name,
        device=device,
    )
    model.to(device)

    print(f'Loading checkpoint from {args.checkpoint_path}')
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = {k[len('dex.'):]: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('dex.')}
    model.dex.load_state_dict(state_dict)
    model.eval()

    for dataset_name in args.datasets:
        if dataset_name == 'scannetpp':
            dataset = ScanNetPP(args.model_name, args.eval_shape, data_root=args.data_root)
        else:
            dataset = KITTI360(args.model_name, data_root=args.data_root)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_thread, pin_memory=True)

        results = evaluate(model, loader, dataset_name, device)

        print(f'Evaluation: {dataset_name}')
        print(' | '.join(f'{name}: {results[name]:.3f}' for name in METRIC_NAMES))


if __name__ == '__main__':
    main()
