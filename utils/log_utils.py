import os
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt

KITTI_RANGE = [0.01, 80.0]
KITTI_MEAN = [123.675, 116.28, 103.53]
KITTI_STD  = [58.395, 57.12, 57.375]
NYU_RANGE = [0.01, 10.0]
NYU_MEAN = [123.675, 116.28, 103.53]
NYU_STD  = [58.395, 57.12, 57.375]
SWS_RANGE = [0.5, 40.0]
K360_RANGE = [0.5, 40.0]
SPP_RANGE = [0.01, 20.0]


def log_summary(summary_writer,
                tag,
                step,
                dataset='NYU',
                image0=None,
                output_depth0=None,
                ground_truth0=None,
                scalars={},
                n_image_per_summary=4):
    '''
    Logs summary to Tensorboard

    Arg(s):
        summary_writer : SummaryWriter
            Tensorboard summary writer
        tag : str
            tag that prefixes names to log
        step : int
            current step in training
        image0 : torch.Tensor[float32]
            image at time step t
        image1to0 : torch.Tensor[float32]
            image at time step t-1 warped to time step t
        image2to0 : torch.Tensor[float32]
            image at time step t+1 warped to time step t
        output_depth0 : torch.Tensor[float32]
            output depth at time t
        sparse_depth0 : torch.Tensor[float32]
            sparse_depth at time t
        validity_map0 : torch.Tensor[float32]
            validity map of sparse depth at time t
        ground_truth0 : torch.Tensor[float32]
            ground truth depth at time t
        pose0to1 : torch.Tensor[float32]
            4 x 4 relative pose from image at time t to t-1
        pose0to2 : torch.Tensor[float32]
            4 x 4 relative pose from image at time t to t+1
        scalars : dict[str, float]
            dictionary of scalars to log
        n_image_per_summary : int
            number of images to display within a summary
    '''

    if 'KITTI' in dataset:
        min_depth, max_depth = KITTI_RANGE[0], KITTI_RANGE[1]
        mean = KITTI_MEAN
        std = KITTI_STD
    elif 'NYU' in dataset:
        min_depth, max_depth = NYU_RANGE[0], NYU_RANGE[1]
        mean = NYU_MEAN
        std = NYU_STD
    elif 'SynWoodScape' in dataset:
        min_depth, max_depth = SWS_RANGE[0], SWS_RANGE[1]
        mean = KITTI_MEAN
        std = KITTI_STD
    elif 'KITTI_360' in dataset:
        min_depth, max_depth = K360_RANGE[0], K360_RANGE[1]
        mean = KITTI_MEAN
        std = KITTI_STD
    elif 'SCANNETPP' in dataset:
        min_depth, max_depth = SPP_RANGE[0], SPP_RANGE[1]
        mean = NYU_MEAN
        std = NYU_STD
    elif 'Matterport' in dataset:
        min_depth, max_depth = 0.01, 80.0
        mean = NYU_MEAN
        std = NYU_STD

    with torch.no_grad():

        display_summary_image = []
        display_summary_depth = []

        display_summary_image_text = tag
        display_summary_depth_text = tag

        if image0 is not None:
            image0_summary = image0[0:n_image_per_summary, ...]

            display_summary_image_text += '_image0'
            display_summary_depth_text += '_image0'

            # Normalize for display if necessary
            image0_summary = (image0_summary - torch.min(image0_summary)) / (torch.max(image0_summary) - torch.min(image0_summary))
            # if torch.max(image0_summary) > 1:
            #     image0_summary = image0_summary / 255.0

            # Add to list of images to log
            display_summary_image.append(
                torch.cat([
                    image0_summary.cpu(),
                    torch.zeros_like(image0_summary, device=torch.device('cpu'))],
                    dim=-1))

            display_summary_depth.append(display_summary_image[-1])

        if output_depth0 is not None:
            output_depth0_summary = output_depth0[0:n_image_per_summary, ...]

            display_summary_depth_text += '_output0'

            # Add to list of images to log
            n_batch, _, n_height, n_width = output_depth0_summary.shape

            display_summary_depth.append(
                torch.cat([
                    colorize(
                        (output_depth0_summary / max_depth).cpu(),
                        colormap='magma'),
                    torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                    dim=3))

            # Log distribution of output depth
            summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)

        if output_depth0 is not None and ground_truth0 is not None:

            ground_truth0_summary = ground_truth0[0:n_image_per_summary, ...]
            validity_map0_summary = torch.where(
                ground_truth0_summary > 0,
                torch.ones_like(ground_truth0_summary),
                ground_truth0_summary)

            display_summary_depth_text += '_groundtruth0-error'

            ground_truth0_summary = torch.clamp(ground_truth0_summary, 0.0, max_depth)

            # Compute output error w.r.t. ground truth
            ground_truth0_error_summary = \
                torch.abs(output_depth0_summary - ground_truth0_summary)

            ground_truth0_error_summary = torch.where(
                validity_map0_summary == 1.0,
                (ground_truth0_error_summary + 1e-8) / (ground_truth0_summary + 1e-8),
                validity_map0_summary)

            # Add to list of images to log
            ground_truth0_summary = colorize(
                (ground_truth0_summary / max_depth).cpu(),
                colormap='magma')
            ground_truth0_error_summary = colorize(
                (ground_truth0_error_summary / 0.05).cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    ground_truth0_summary,
                    ground_truth0_error_summary],
                    dim=3))

            # Log distribution of ground truth
            summary_writer.add_histogram(tag + '_ground_truth0_distro', ground_truth0, global_step=step)

    # Log scalars to tensorboard
    for (name, value) in scalars.items():
        summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

    # Log image summaries to tensorboard
    if len(display_summary_image) > 1:
        display_summary_image = torch.cat(display_summary_image, dim=2)

        summary_writer.add_image(
            display_summary_image_text,
            torchvision.utils.make_grid(display_summary_image, nrow=n_image_per_summary),
            global_step=step)

    if len(display_summary_depth) > 1:
        display_summary_depth = torch.cat(display_summary_depth, dim=2)

        summary_writer.add_image(
            display_summary_depth_text,
            torchvision.utils.make_grid(display_summary_depth, nrow=n_image_per_summary),
            global_step=step)


def log(s, filepath=None, to_console=False):
    '''
    Logs a string or list of strings to either a file or console.

    Arg(s):
        s : str or list
            string or list of strings to log
        filepath : str or None
            output filepath for logging, if None logs to console only
        to_console : bool
            log to console if True
    '''

    if to_console:
        print(s if isinstance(s, str) else '\n'.join(s))

    if filepath is not None:
        # Ensure the directory exists
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        # Open the file in append mode
        with open(filepath, 'a+') as o:
            if isinstance(s, list):
                for item in s:
                    o.write(str(item) + '\n')
            else:
                o.write(str(s) + '\n')

def colorize(T, colormap='magma'):
    '''
    Colorizes a 1-channel tensor with matplotlib colormaps

    Arg(s):
        T : torch.Tensor[float32]
            1-channel tensor
        colormap : str
            matplotlib colormap
    '''

    cm = plt.cm.get_cmap(colormap)
    shape = T.shape

    # Convert to numpy array and transpose
    if shape[0] > 1:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)))
    else:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)), axis=-1)

    # Colorize using colormap and transpose back
    color = np.concatenate([
        np.expand_dims(cm(T[n, ...])[..., 0:3], 0) for n in range(T.shape[0])],
        axis=0)
    color = np.transpose(color, (0, 3, 1, 2))

    # Convert back to tensor
    return torch.from_numpy(color.astype(np.float32))