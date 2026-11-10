import torch, torchvision
import torchvision.transforms.functional as functional
import numpy as np
from PIL import Image
import cv2
import math

import scipy.ndimage.interpolation as itpl
import scipy.misc as misc

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import transforms


def crop_inputs(inputs, shape, intrinsics=None, crop_type=['bottom']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[torch.Tensor]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : tensor[float32]
            3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[torch.Tensor] : list of cropped inputs
        tensor[float32] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, :, y_start:y_end, x_start:x_end] for T in inputs
    ]

    if intrinsics is not None:
        # Adjust intrinsics
        intrinsics = intrinsics + torch.tensor([[0.0, 0.0, -x_start],
                                   [0.0, 0.0, -y_start],
                                   [0.0, 0.0, 0.0     ]]).cuda()

        return outputs, intrinsics
    else:
        return outputs


def resize_to_scale(tensors, resize_shape, intrinsics=None):
    '''
    Resize images, ground truth, spare depth, and adjust intrinsics accordingly

    Arg(s):
        tensors : list[torch.Tensor]
            list of N x C x H x W tensors
        intrinsics : tensor.Tensor
            1 x 3 x 3 intrinsics matrix
        resize_shape : list[int, int]
            height and width to resize
    Returns:
        list[torch.Tensor] : list of transformed N x C x H x W image tensors
        torch.Tensor : adjusted 1 x 3 x 3 intrinsics matrix
        list[tuple] : list of original height and width for each tensor
    '''
    resize_height, resize_width = resize_shape
    resized_tensors = []
    original_shapes = []

    for tensor in tensors:
        # Get original dimensions
        n, c, h, w = tensor.shape
        original_shapes.append((h, w))

        # Calculate the scale for resizing
        scale_x = resize_width / w
        scale_y = resize_height / h

        # Resize the tensor
        resized_tensor = functional.resize(tensor, size=[resize_height, resize_width])
        resized_tensors.append(resized_tensor)

    if intrinsics is not None:
        # Ensure intrinsics is of shape [1, 3, 3]
        if intrinsics.shape != (1, 3, 3):
            raise ValueError("Intrinsics matrix must be of shape [1, 3, 3]")

        # Extract the single 3x3 intrinsics matrix
        intrinsics_matrix = intrinsics[0]

        # Adjust intrinsics matrix
        intrinsics_matrix[0, 0] *= scale_x  # fx
        intrinsics_matrix[1, 1] *= scale_y  # fy
        intrinsics_matrix[0, 2] *= scale_x  # cx
        intrinsics_matrix[1, 2] *= scale_y  # cy

        # Convert back to [1, 3, 3] shape
        intrinsics = intrinsics_matrix.unsqueeze(0)

    return resized_tensors, intrinsics, original_shapes


def undo_resize(tensors, intrinsics, original_shapes):
    '''
    Undo resize images

    Arg(s):
        tensors : list[torch.Tensor]
            list of N x C x H x W tensors
        intrinsics : torch.Tensor
            1 x 3 x 3 intrinsics matrix
        original_shapes : list[tuple]
            list of original height and width for each tensor
    Returns:
        list[torch.Tensor] : list of restored N x C x H x W image tensors
        torch.Tensor : adjusted 1 x 3 x 3 intrinsics matrix
    '''
    restored_tensors = []

    for tensor, (original_height, original_width) in zip(tensors, original_shapes):
        # Get current dimensions
        n, c, h, w = tensor.shape

        # Calculate the scale for restoring
        scale_x = original_width / w
        scale_y = original_height / h

        # Resize the tensor back to the original dimensions
        restored_tensor = functional.resize(tensor, [original_height, original_width])
        restored_tensors.append(restored_tensor)

    if intrinsics is not None:
        # Ensure intrinsics is of shape [1, 3, 3]
        if intrinsics.shape != (1, 3, 3):
            raise ValueError("Intrinsics matrix must be of shape [1, 3, 3]")

        # Extract the single 3x3 intrinsics matrix
        intrinsics_matrix = intrinsics[0]

        # Adjust intrinsics matrix back
        intrinsics_matrix[0, 0] /= scale_x  # fx
        intrinsics_matrix[1, 1] /= scale_y  # fy
        intrinsics_matrix[0, 2] /= scale_x  # cx
        intrinsics_matrix[1, 2] /= scale_y  # cy

        # Convert back to [1, 3, 3] shape
        intrinsics = intrinsics_matrix.unsqueeze(0)

    return restored_tensors, intrinsics


def resize_and_crop(images_arr,
                    do_resize_and_crop,
                    resize_shape,
                    start_yx,
                    end_yx,
                    interpolation_modes=[Image.NEAREST]):
    '''
    Resize and crop to shape

    Arg(s):
        images_arr : list[torch.Tensor]
            list of N x C x H x W tensors
        do_resize_and_crop : bool
            N booleans to determine if image will be resized and crop based on input (y, x)
        resize_shape : list[int, int]
            height and width to resize
        start_yx : list[int, int]
            top left corner y, x coordinate
        end_yx : list
            bottom right corner y, x coordinate
        interpolation_modes : list[int]
            list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
    Returns:
        list[torch.Tensor] : list of transformed N x C x H x W image tensors
    '''

    n_images_arr = len(images_arr)

    if len(interpolation_modes) < n_images_arr:
        interpolation_modes = \
            interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))

    for i, (images, interpolation_mode) in enumerate(zip(images_arr, interpolation_modes)):

        images_cropped = []

        for b, image in enumerate(images):
            if do_resize_and_crop[b]:

                r_height = resize_shape[0][b]
                r_width = resize_shape[1][b]

                # Resize image
                image = functional.resize(
                    image,
                    size=(r_height, r_width),
                    interpolation=interpolation_mode)

                start_y = start_yx[0][b]
                start_x = start_yx[1][b]
                end_y = end_yx[0][b]
                end_x = end_yx[1][b]

                # Crop image
                image = image[..., start_y:end_y, start_x:end_x]

                images_cropped.append(image)
            else:
                images_cropped.append(image)

        images_arr[i] = torch.stack(images_cropped, dim=0)

    return images_arr


def resize_image(tensor, width, height, keep_aspect_ratio=True, resize_method='lower_bound', ensure_multiple_of=1, interpolation_method=cv2.INTER_AREA):
    """
    Resize image tensor.

    Args:
        tensor : torch.Tensor
            N x C x H x W image tensor
        width : int
            desired output width
        height : int
            desired output height
        keep_aspect_ratio : bool
            whether to keep the aspect ratio
        resize_method : str
            resize method: 'lower_bound', 'upper_bound', 'minimal'
        ensure_multiple_of : int
            output width and height are constrained to be multiples of this value
        interpolation_method : str
            interpolation method: 'nearest', 'bilinear', etc.
    Returns:
        torch.Tensor : resized image tensor
        tuple : original height and width of the image tensor
    """
    # Get original dimensions
    N, C, H, W = tensor.shape
    resized_list = []

    for i in range(N):
        single_tensor = tensor[i].unsqueeze(0)  # shape (1, C, H, W)

        h = single_tensor.shape[2]
        w = single_tensor.shape[3]

        scale_height = height / h
        scale_width = width / w

        if keep_aspect_ratio:
            if resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif resize_method == "minimal":
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {resize_method} not implemented")

        new_height = (np.round(scale_height * h / ensure_multiple_of) * ensure_multiple_of).astype(int)
        new_width = (np.round(scale_width * w / ensure_multiple_of) * ensure_multiple_of).astype(int)

        if new_height < 0:
            new_height = (np.ceil(scale_height * h / ensure_multiple_of) * ensure_multiple_of).astype(int)
        if new_width < 0:
            new_width = (np.ceil(scale_width * w / ensure_multiple_of) * ensure_multiple_of).astype(int)

        np_tensor = single_tensor.squeeze(0).permute(1,2,0).cpu().numpy()

        resized_np_tensor = cv2.resize(np_tensor, (new_width, new_height), interpolation=interpolation_method)

        resized_single_tensor = torch.from_numpy(resized_np_tensor).permute(2,0,1).unsqueeze(0)

        resized_list.append(resized_single_tensor)

    resized_tensor = torch.cat(resized_list, dim=0)

    return resized_tensor


def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_VIRIDIS)

    if bits == 1:
        cv2.imwrite(path, out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path, out.astype("uint16"))

    return


def zju_crop(images_arr, radar_points, intrinsic=None):
    '''
    Crops a list of images and processes radar points according to specific regions of interest for zju.

    Arg(s):
        images_arr : list[numpy.ndarray]
            List of C x H x W numpy arrays 
        radar_points : numpy.ndarray
            N x 3 numpy array representing the radar points
        intrinsic : numpy.ndarray, optional
            3 x 3 intrinsics matrix 

    Returns:
        list[numpy.ndarray] : A list of cropped images as numpy arrays.
        numpy.ndarray : Processed radar points adjusted to the cropped region.
        numpy.ndarray : The original intrinsic matrix (unchanged).
    '''
    
    # Processing radar points
    if radar_points.size != 0:
        if len(radar_points.shape) > 2:
            radar_points = np.squeeze(radar_points)

        radar_points = radar_points[radar_points[:, 1] < 720 // 4 * 3]
        radar_points[:, 1] = radar_points[:, 1] - 720 // 3
        radar_points = radar_points[radar_points[:, 1] >= 0]

    outputs = []

    for idx, image in enumerate(images_arr):
        if image.size > 0:
            if image.shape[0] == 720:
                image = image[720 // 3: 720 // 4 * 3, :]
            if image.shape[1] == 720:
                image = image[:, 720 // 3: 720 // 4 * 3, :]
            if image.shape[2] == 720:
                image = image[:, :, 720 // 3: 720 // 4 * 3]
        else:
            image = np.array([])
        outputs.append(image)

    return outputs, radar_points, intrinsic


def vod_crop(images_arr, radar_points, intrinsic=None):
    '''
    Crops a list of images and processes radar points according to specific regions of interest for vod.

    Arg(s):
        images_arr : list[numpy.ndarray]
            List of C x H x W numpy arrays 
        radar_points : numpy.ndarray
            N x 3 numpy array representing the radar points
        intrinsic : numpy.ndarray, optional
            3 x 3 intrinsics matrix 

    Returns:
        list[numpy.ndarray] : A list of cropped images as numpy arrays.
        numpy.ndarray : Processed radar points adjusted to the cropped region.
        numpy.ndarray : The original intrinsic matrix (unchanged).
    '''
    
    # Processing radar points
    if radar_points.size != 0:
        if len(radar_points.shape) > 2:
            radar_points = np.squeeze(radar_points)

        radar_points = radar_points[radar_points[:, 1] < 1216 // 2]
        radar_points = radar_points[radar_points[:, 1] >= 0]

        radar_points = radar_points[:, :3]

    outputs = []

    for idx, image in enumerate(images_arr):
        if image.size > 0:
            if image.shape[0] == 1216:
                image = image[1216 // 2 : 1216, :]
            if image.shape[1] == 1216:
                image = image[:, 1216 // 2 : 1216, :]
            if image.shape[2] == 1216:
                image = image[:, :, 1216 // 2 : 1216]
        else:
            image = np.array([])
        outputs.append(image)

    return outputs, radar_points, intrinsic

# Use more loose depth threshold for points with larger depth value
def sid_depth_thresh(input_depth):
    alpha = 5
    beta = 16
    K = 100

    depth_thresh = np.exp(((input_depth * np.log(beta / alpha)) / K) + np.log(alpha))

    return depth_thresh

# Use more strict distance threshold for points with larger depth value
def sid_dist_thresh(input_depth):
    alpha = 14
    beta = 4
    K = 100

    dist_thresh = np.exp(((input_depth * np.log(beta / alpha)) / K) + np.log(alpha))

    return dist_thresh

# Select the depth value of the topk candidates from ndarray
def select_topk_depth(depth, topk_idx):
    # Select topk index and topk value
    point_count = topk_idx.shape[0]

    topk_depth_lst = []
    for i in range(point_count):
        # ipdb.set_trace()
        topk_depth = depth[topk_idx[i, :]]
        topk_depth_lst.append(topk_depth)

    return np.asarray(topk_depth_lst)

# Check topk candidates using depth threshold
def check_valid_depth(depth_dist, dist_valid_count, depth_value):
    point_count = depth_dist.shape[0]

    # ipdb.set_trace()
    # 0 => invalid, 1 => valid, 2 => unknown
    valid_labels = np.zeros([point_count, 1])
    # Iterate through all the radar points
    for i in range(point_count):
        # ipdb.set_trace()
        depth_thresh_new = sid_depth_thresh(depth_value[i, :dist_valid_count[i, 0]])
        depth_valid_count = np.sum((depth_dist[i, :dist_valid_count[i, 0]] < depth_thresh_new).astype(np.int16))
        # ipdb.set_trace()
        if dist_valid_count[i, 0] == 0:
            valid_labels[i, 0] = 2
        elif depth_valid_count >= np.ceil(dist_valid_count[i, 0] / 2):
            valid_labels[i, 0] = 1

    return valid_labels

def lin_filter_radar_points(radar_points, radar_depth_points, lidar_points, lidar_depth_points):
    # Find k nearest neighbors whithin distance threshold
    k = 3
    dist_thresh = 10

    # Fetch only the x, y coord
    radar_points = radar_points[:2, :].transpose(1, 0)
    lidar_points = lidar_points[:2, :].transpose(1, 0)

    # Mask out points > 80m
    # radar_mask = radar_depth_points < 80.
    # radar_depth_points = radar_depth_points[radar_mask]
    # radar_points = radar_points[radar_mask, :]

    radar_points_exp = np.expand_dims(radar_points, axis=1)
    lidar_points_exp = np.expand_dims(lidar_points, axis=0)

    dist = np.sqrt(np.sum((radar_points_exp - lidar_points_exp) ** 2, axis=-1))

    # Fetch the topk index
    dist_topk_index = np.argsort(dist)[:, :k][..., None]
    # Get dist topk value
    dist_topk_val = np.sort(dist, axis=-1)[:, :k]
    # Get depth topk depth value
    depth_topk_val = np.squeeze(select_topk_depth(lidar_depth_points, dist_topk_index))

    # Get depth-aware dist thresh
    dist_thresh_sid = sid_dist_thresh(depth_topk_val)
    dist_valid_count = np.sum((dist_topk_val <= dist_thresh_sid).astype(np.int16), axis=-1)[..., None]

    # print(sid_dist_thresh(depth_topk_val))
    depth_dist = radar_depth_points[..., None] - depth_topk_val
    valid_labels = check_valid_depth(depth_dist, dist_valid_count, depth_topk_val)

    # ipdb.set_trace()
    # Perform the filtering
    valid_mask_final = np.squeeze(valid_labels > 0)
    radar_points_filtered = radar_points[valid_mask_final, :].transpose(1, 0)
    radar_depth_points_filtered = radar_depth_points[valid_mask_final]

    return valid_labels, valid_mask_final, radar_points_filtered, radar_depth_points_filtered

def lin_train_transform(input_data, patch_size=[450, 800], max_depth=80):
    rgb = np.array(input_data["image"]).astype(np.float32)
    lidar_depth = np.array(input_data["lidar_depth"]).astype(np.float32)
    radar_depth = np.array(input_data["radar_depth"]).astype(np.float32)
    
    if 'index_map' in input_data.keys():
        index_map = np.array(input_data["index_map"]).astype(np.int)

    # Define augmentation factor
    scale_factor = np.random.uniform(1, 1.5)  # random scaling
    angle_factor = np.random.uniform(-5., 5.)  # random rotation degrees
    flip_factor = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    # Compose customized transform for RGB and Depth separately
    color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2)
    resize_image = transforms.LinResize(scale_factor, interpolation="bilinear")
    resize_depth = transforms.LinResize(scale_factor, interpolation="nearest")

    h_new = patch_size[0]
    w_new = patch_size[1]
    resize_image_method = transforms.LinResize([h_new, w_new], interpolation="bilinear")
    resize_depth_method = transforms.LinResize([h_new, w_new], interpolation="nearest")

    # Get the border of random crop
    h_scaled, w_scaled = math.floor(h_new * scale_factor), math.floor((w_new * scale_factor))
    h_bound, w_bound = h_scaled - patch_size[0], w_scaled - patch_size[1]
    h_startpoint = round(np.random.uniform(0, h_bound))
    w_startpoint = round(np.random.uniform(0, w_bound))

    # Compose the transforms for RGB
    transform_rgb = transforms.Compose([
        transforms.Rotate(angle_factor),
        resize_image,
        transforms.Crop(h_startpoint, w_startpoint, patch_size[0], patch_size[1]),
        transforms.HorizontalFlip(flip_factor)
    ])

    # Compose the transforms for Depth
    transform_depth = transforms.Compose([
        transforms.Rotate(angle_factor),
        resize_depth,
        transforms.Crop(h_startpoint, w_startpoint, patch_size[0], patch_size[1]),
        transforms.HorizontalFlip(flip_factor)
    ])

    # Perform transform on rgb data
    # ToDo: whether we need to - imagenet mean here
    rgb = transform_rgb(rgb)
    rgb = color_jitter(rgb)
    rgb = rgb / 255.

    # Perform transform on lidar depth data
    lidar_depth /= float(scale_factor)
    lidar_depth = transform_depth(lidar_depth)

    rgb = np.array(rgb).astype(np.float32)
    lidar_depth = np.array(lidar_depth).astype(np.float32)

    rgb = transforms.ToTensor(rgb)
    lidar_depth = transforms.ToTensor(lidar_depth)

    # Perform transform on radar depth data
    radar_depth /= float(scale_factor)
    radar_depth = transform_depth(radar_depth)

    radar_depth = np.array(radar_depth).astype(np.float32)
    radar_depth = transforms.ToTensor(radar_depth)

    # Perform transform on index map
    if 'index_map' in input_data.keys():
        index_map = transform_depth(index_map)
        index_map = np.array(index_map).astype(np.int)
        index_map = transforms.ToTensor(index_map)
        index_map = index_map.unsqueeze(0)

    lidar_depth = lidar_depth.unsqueeze(0)
    radar_depth = radar_depth.unsqueeze(0)

    # Filter out the the points exceeding max_depth
    mask = (radar_depth > max_depth)
    radar_depth[mask] = 0
    
    return rgb, lidar_depth, radar_depth

def lin_validation_transform(input_data, patch_size=[450,800], max_depth=80):
    rgb = np.array(input_data["image"]).astype(np.float32)
    lidar_depth = np.array(input_data["lidar_depth"]).astype(np.float32)
    radar_depth = np.array(input_data["radar_depth"]).astype(np.float32)
    
    if 'index_map' in input_data.keys():
        index_map = np.array(input_data["index_map"]).astype(np.int)

    # Define augmentation factor
    scale_factor = np.random.uniform(1, 1.5)  # random scaling
    angle_factor = np.random.uniform(-5., 5.)  # random rotation degrees
    flip_factor = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    # Compose customized transform for RGB and Depth separately
    color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2)
    resize_image = transforms.LinResize(scale_factor, interpolation="bilinear")
    resize_depth = transforms.LinResize(scale_factor, interpolation="nearest")

    h_new = patch_size[0]
    w_new = patch_size[1]
    resize_image_method = transforms.LinResize([h_new, w_new], interpolation="bilinear")
    resize_depth_method = transforms.LinResize([h_new, w_new], interpolation="nearest")

    transform_rgb = transforms.Compose([
            # resize_image_method,
            transforms.CenterCrop(patch_size)
        ])
    transform_depth = transforms.Compose([
            # resize_depth_method,
            transforms.CenterCrop(patch_size)
        ])

    # Perform transform on rgb data
    rgb = transform_rgb(rgb)

    # Perform transform on lidar depth data
    lidar_depth = transform_depth(lidar_depth)

    rgb = np.array(rgb).astype(np.float32)
    lidar_depth = np.array(lidar_depth).astype(np.float32)

    rgb = transforms.ToTensor()(rgb)
    lidar_depth = transforms.ToTensor()(lidar_depth)

    # Perform transform on radar depth data
    radar_depth = transform_depth(radar_depth)

    radar_depth = np.array(radar_depth).astype(np.float32)
    radar_depth = transforms.ToTensor()(radar_depth)

    # Perform transform on index map
    if 'index_map' in input_data.keys():
        index_map = transform_depth(index_map)
        index_map = np.array(index_map).astype(np.int)
        index_map = transforms.ToTensor()(index_map)
        index_map = index_map.unsqueeze(0)

    lidar_depth = lidar_depth.unsqueeze(0)
    radar_depth = radar_depth.unsqueeze(0)

    # Filter out the the points exceeding max_depth
    mask = (radar_depth > max_depth)
    radar_depth[mask] = 0

    return rgb, lidar_depth, radar_depth

def lin_nuscenes_evaluation_resize(validation_depth, ground_truth):
    '''
    Resize the output depth map and ground truth for Lin evaluation on Nuscenes

    Arg(s):
        validation_depth : torch.Tensor[float32]
            H x W output depth fusion
        ground_truth : torch.Tensor[float32]
            H x W output depth fusion

    Returns:
        resized_validation_depth : torch.Tensor[float32]
            H x W output depth fusion
        resized_ground_truth : torch.Tensor[float32]
            H x W output depth fusion
    '''
    target_h, target_w = 900, 1600

    if validation_depth.shape[0] != target_h and validation_depth.shape[1] != target_w:
        # Convert numpy arrays to PIL images
        validation_depth_img = Image.fromarray(validation_depth)
        
        # Resize to target dimensions using bilinear interpolation
        resized_validation_depth_img = validation_depth_img.resize((target_w, target_h), Image.BILINEAR)
        
        # Convert resized PIL images back to numpy arrays
        validation_depth = np.array(resized_validation_depth_img)

    if ground_truth.shape[0] != target_h and ground_truth.shape[1] != target_w:
        # Convert numpy arrays to PIL images
        ground_truth_img = Image.fromarray(ground_truth)

        # Resize to target dimensions using bilinear interpolation
        resized_ground_truth_img = ground_truth_img.resize((target_w, target_h), Image.BILINEAR)

        # Convert resized PIL images back to numpy arrays
        ground_truth = np.array(resized_ground_truth_img)

    return validation_depth, ground_truth