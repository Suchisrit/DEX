import argparse, os
import torch
from main import train

def parse_args():
    parser = argparse.ArgumentParser(description='Train Distortion Extenders (DEX)')

    parser.add_argument('--train_batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--datasets', nargs='+', type=str, help="Dataset names/tags (e.g. irs waymo), used for logging")
    parser.add_argument('--image_txts', nargs='+', type=str, help="Paths to text files with one image path per line (one per dataset)")
    parser.add_argument('--intrinsics_paths', nargs='+', type=str, help="Per-dataset intrinsics: shared .npy or .txt list of per-image .npy paths (one per dataset)")
    parser.add_argument('--crop_shapes', nargs='+', type=int, help="H W pairs per dataset (2*N values total)")
    parser.add_argument('--k1_ranges', nargs='+', type=float, help="k1 min/max pairs per dataset (2*N values)")
    parser.add_argument('--k2_ranges', nargs='+', type=float, help="k2 min/max pairs per dataset (2*N values)")
    parser.add_argument('--k3_ranges', nargs='+', type=float, help="k3 min/max pairs per dataset (2*N values)")
    parser.add_argument('--k4_ranges', nargs='+', type=float, help="k4 min/max pairs per dataset (2*N values)")
    parser.add_argument('--n_batches_per_update', type=int, default=2, help="Number of dataset batches to accumulate per optimizer step")
    parser.add_argument('--dataset_weights', nargs='+', type=float, default=None, help="Loss weight per dataset (N values, defaults to 1.0 for all)")

    parser.add_argument('--learning_rates', nargs='+', type=float, help="Learning rates for the optimizer")
    parser.add_argument('--learning_schedule', nargs='+', type=int, help="Learning rate schedule (in iterations)")

    parser.add_argument('--num_extenders', type=int, default=12, help="Number of Distortion Extenders to train per encoder block")

    parser.add_argument('--output_reference_frame', type=str, default='distance', choices=['distance', 'depth'],
                        help="Reference frame the standard prediction is aligned to for the loss: 'distance' (default) converts z-depth to Euclidean distance from the camera center (R); 'depth' keeps the raw z-depth")

    parser.add_argument('--aug_prob_brightness', type=float, default=0.0, help="Probability of brightness augmentation")
    parser.add_argument('--aug_prob_contrast', type=float, default=0.0, help="Probability of contrast augmentation")
    parser.add_argument('--aug_prob_saturation', type=float, default=0.0, help="Probability of saturation augmentation")
    parser.add_argument('--aug_prob_hue', type=float, default=0.0, help="Probability of hue augmentation")
    parser.add_argument('--aug_prob_blur', type=float, default=0.0, help="Probability of Gaussian blur augmentation")
    parser.add_argument('--aug_brightness_range', nargs=2, type=float, default=[0.6, 1.4], metavar=('MIN', 'MAX'), help="Brightness enhancement factor range")
    parser.add_argument('--aug_contrast_range', nargs=2, type=float, default=[0.6, 1.4], metavar=('MIN', 'MAX'), help="Contrast enhancement factor range")
    parser.add_argument('--aug_saturation_range', nargs=2, type=float, default=[0.6, 1.4], metavar=('MIN', 'MAX'), help="Saturation enhancement factor range")
    parser.add_argument('--aug_hue_range', nargs=2, type=float, default=[-0.1, 0.1], metavar=('MIN', 'MAX'), help="Hue shift range (fraction of hue circle)")
    parser.add_argument('--aug_blur_range', nargs=2, type=float, default=[0.1, 0.5],  metavar=('MIN', 'MAX'), help="Gaussian blur sigma range")

    parser.add_argument('--checkpoint_path', type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument('--n_step_per_checkpoint', type=int, default=1, help="Number of steps per checkpoint")
    parser.add_argument('--n_step_per_summary', type=int, default=1, help="Number of steps per summary")
    parser.add_argument('--restore_path', type=str, default=None, help="Path to restore model from checkpoint")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for training")
    parser.add_argument('--n_thread', type=int, default=4, help="Number of threads to use")
    parser.add_argument('--model_name', type=str, default='unidepth', help="Name of backbone model to train Distortion Extenders for, [depthanything, unidepth, midas]")
    parser.add_argument('--enable_ddp', action='store_true', help="Whether to ddp to train the model using multi-gpu")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    if args.enable_ddp:
        world_size = torch.cuda.device_count()
        rank = int(os.environ["RANK"]) 
        print(f"World size={world_size}, Starting DDP training on Rank {rank}")
        train(rank, world_size, args)
    else:
        train(0, 1, args)