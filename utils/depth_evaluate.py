import torch
from tqdm import tqdm

from utils.fisheye_ray_conversion import fisheye_distance_to_fisheye_depth

METRIC_NAMES = ['rms', 'a1']

# (alignment range, metric range) in meters per dataset
# KITTI-360 is aligned on 0.5-40 m but scored on 0.01-80 m, as in the original evaluation
SCANNETPP_RANGES = ((0.01, 20.0), (0.01, 20.0))
KITTI360_RANGES = ((0.5, 40.0), (0.01, 80.0))


def align_scale(depth_pred, depth_gt, min_depth, max_depth):
    '''Aligns the prediction to the ground truth with a per-image least squares scale and shift'''

    for b in range(depth_gt.shape[0]):
        mask = ((depth_gt[b] > min_depth) * (depth_gt[b] <= max_depth)).bool()

        depth_pred_masked = depth_pred[b][mask]
        depth_gt_masked = depth_gt[b][mask]

        homo_gt_masked = torch.stack([depth_gt_masked, torch.ones_like(depth_gt_masked)], 1)
        s, t = torch.linalg.lstsq(homo_gt_masked, depth_pred_masked.unsqueeze(1)).solution
        depth_pred[b, ...] = (depth_pred[b] - t) / s

    return depth_pred


def depth_errors(pred, target):
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    a1 = (torch.max(target / pred, pred / target) < 1.25).float().mean()

    return [rmse, a1]


def evaluate(model, loader, dataset_name, device):
    '''
    Evaluates the model with Distortion Extenders on fisheye images

    Arg(s):
        model : Model
            model in eval mode with Extenders loaded
        loader : torch.utils.data.DataLoader
            loader of a ScanNetPP or KITTI360 dataset
        dataset_name : str
            'scannetpp' (z-depth ground truth) or 'kitti360' (distance ground truth)
        device : torch.device
            device to run on
    Returns:
        dict[str, float] : metric averaged over the images with valid ground truth
    '''

    align_range, metric_range = SCANNETPP_RANGES if dataset_name == 'scannetpp' else KITTI360_RANGES
    min_depth, max_depth = metric_range

    total_error = {name: 0.0 for name in METRIC_NAMES}
    n_samples = 0

    for batch in tqdm(loader, desc=dataset_name):
        image, depth_gt = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            depth_pred = model.infer(image)

            if torch.isinf(depth_pred.mean()):
                depth_pred = 0.1 * torch.ones_like(depth_pred)

            if dataset_name == 'scannetpp':
                # the model predicts distance, the ground truth is z-depth
                K, D = batch[2].to(device), batch[3].to(device)
                depth_pred = fisheye_distance_to_fisheye_depth(depth_pred, K, D)

            depth_pred = align_scale(depth_pred, depth_gt, *align_range)
            depth_pred = torch.clamp(depth_pred, min_depth, max_depth)

            for b in range(depth_gt.shape[0]):
                mask = (depth_gt[b] > min_depth) * (depth_gt[b] < max_depth)
                mask = (mask * (depth_pred[b] > min_depth) * (depth_pred[b] < max_depth)).bool()

                if mask.sum() == 0:
                    continue

                errors = depth_errors(depth_pred[b][mask], depth_gt[b][mask])
                for name, error in zip(METRIC_NAMES, errors):
                    total_error[name] += error.item()
                n_samples += 1

    return {name: error / n_samples for name, error in total_error.items()}
