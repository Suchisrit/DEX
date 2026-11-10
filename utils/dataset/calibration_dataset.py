import logging, cv2
import numpy as np
import random
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset


class CalibrationDataset(Dataset):
    # TODO: Make repo consistent in use of double and single quotes
    '''
    Unified dataset for fisheye Distortion Extender training.

    Each dataset is described by:
      - image_txt:        path to a text file with one absolute/relative image path per line
      - intrinsics_path:  either a single shared .npy (3x3 K matrix) or a .txt file with
                          one .npy path per line (must match the image list length)

    All other dataset-specific behaviour (crop, distortion ranges) is passed explicitly.
    '''

    def __init__(self, args, patch_size, crop_shape, distortion_ranges, image_txt, intrinsics_path):
        super().__init__()
        self.patch_size = patch_size
        self.crop_shape = crop_shape
        self.distortion_ranges = distortion_ranges
        self.logger = logging.getLogger(__name__)

        # Normalization settings
        if args.model_name == 'midas':
            mean = [0.5, 0.5, 0.5]
            std  = [0.5, 0.5, 0.5]
        else:
            mean = [0.485, 0.456, 0.406]  # ViT mean
            std  = [0.229, 0.224, 0.225]  # ViT std

        self.normalize  = transforms.Normalize(mean=mean, std=std)
        self.to_tensor  = transforms.ToTensor()

        # Augmentation probabilities and ranges: [brightness, contrast, saturation, hue, blur]
        aug_probs = args.augmentation_probabilities
        aug_ranges = args.augmentation_ranges
        assert aug_probs is not None and len(aug_probs) == 5, (
            f"augmentation_probabilities must have exactly 5 values "
            f"[brightness, contrast, saturation, hue, blur], got: {aug_probs}")
        assert aug_ranges is not None and len(aug_ranges) == 5, (
            f"augmentation_ranges must have exactly 5 entries "
            f"[brightness, contrast, saturation, hue, blur], got: {aug_ranges}")
        self.augmentation_probabilities = aug_probs
        self.augmentation_ranges = aug_ranges

        # Load image paths
        with open(image_txt, 'r') as f:
            self.image_paths = [line.strip() for line in f if line.strip()]

        # Load intrinsics — shared (.npy) or per-image (.txt)
        if intrinsics_path.endswith('.npy'):
            self._shared_intrinsics = np.load(intrinsics_path)
            self._intrinsic_paths   = None
        elif intrinsics_path.endswith('.txt'):
            self._shared_intrinsics = None
            with open(intrinsics_path, 'r') as f:
                self._intrinsic_paths = [line.strip() for line in f if line.strip()]
            if len(self._intrinsic_paths) != len(self.image_paths):
                raise ValueError(
                    f"Intrinsics list ({len(self._intrinsic_paths)}) does not match "
                    f"image list ({len(self.image_paths)}) for {image_txt}")
        else:
            raise ValueError(f"intrinsics_path must end in .npy or .txt, got: {intrinsics_path}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        standard_image = np.array(Image.open(image_path).convert('RGB'), copy=True)

        if self._shared_intrinsics is not None:
            intrinsics = self._shared_intrinsics.copy()
        else:
            intrinsics = np.load(self._intrinsic_paths[idx])

        orig_h, orig_w = standard_image.shape[:2]
        crop_h, crop_w = self.crop_shape

        if crop_h > orig_h or crop_w > orig_w:
            raise ValueError(
                f"Crop size {self.crop_shape} is larger than image size {(orig_h, orig_w)} "
                f"for {image_path}")

        dr = self.distortion_ranges
        distortion_parameters = np.array([
            random.uniform(dr[0], dr[1]),
            random.uniform(dr[2], dr[3]),
            random.uniform(dr[4], dr[5]),
            random.uniform(dr[6], dr[7]),
        ], dtype=np.float32)

        standard_image, intrinsics = self._crop_and_resize_image(
            standard_image, intrinsics, output_size=[crop_w, crop_h])

        # Photometric augmentation applied before fisheye generation so both images share the same appearance
        standard_image = self._photometric_augment(standard_image)

        fisheye_image, distortion_map, undistortion_map = self._standard_to_fisheye(
            standard_image, intrinsics, distortion_parameters)

        standard_image = Image.fromarray(standard_image.astype(np.uint8))
        standard_image = self.normalize(self.to_tensor(standard_image))

        fisheye_image = Image.fromarray(fisheye_image.astype(np.uint8))
        fisheye_image = self.normalize(self.to_tensor(fisheye_image))

        grid_tensor     = self._create_grid_tensor(undistortion_map[..., 0], undistortion_map[..., 1])
        for_grid_tensor = self._create_grid_tensor(distortion_map[..., 0],   distortion_map[..., 1])

        return standard_image, fisheye_image, grid_tensor, for_grid_tensor, intrinsics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _photometric_augment(self, image):
        # TODO: Make repo consistent in use of double and single quotes
        '''
        Apply independent photometric augmentations to a uint8 numpy RGB image (H, W, 3).
        Each augmentation fires independently with its own probability and uses its own range.
        Returns an augmented uint8 numpy array.
        '''
        p = self.augmentation_probabilities
        r = self.augmentation_ranges
        pil_img = Image.fromarray(image)

        if random.random() < p[0]:  # brightness
            pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(r[0][0], r[0][1]))

        if random.random() < p[1]:  # contrast
            pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(r[1][0], r[1][1]))

        if random.random() < p[2]:  # saturation
            pil_img = ImageEnhance.Color(pil_img).enhance(random.uniform(r[2][0], r[2][1]))

        if random.random() < p[3]:  # hue
            pil_img = transforms.functional.adjust_hue(pil_img, random.uniform(r[3][0], r[3][1]))

        if random.random() < p[4]:  # Gaussian blur
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(r[4][0], r[4][1])))

        return np.array(pil_img, dtype=np.uint8)

    def _standard_to_fisheye(self, img, intrinsics, distortion_parameters):
        '''Apply fisheye distortion to a standard image. Returns (fisheye_img, distortion_map, undistortion_map).'''
        height, width = img.shape[:2]
        K = intrinsics
        D = distortion_parameters

        xv, yv = np.meshgrid(np.arange(width), np.arange(height))
        points_2d = np.stack((xv, yv), axis=-1).astype(np.float32).reshape(-1, 2)

        points3d = np.concatenate([points_2d, np.ones_like(points_2d[:, 0:1])], 1)
        points3d = (np.linalg.inv(K) @ points3d.T).T

        # Build distortion map: undistorted pixel -> distorted coords
        undistorted_points = cv2.fisheye.undistortPoints(
            points_2d.reshape(-1, 1, 2), K, D, P=K)
        undistorted_points = undistorted_points.reshape(height, width, 2)

        distort_map_x = undistorted_points[..., 0]
        distort_map_y = undistorted_points[..., 1]

        fisheye_img = cv2.remap(
            img, distort_map_x, distort_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)

        undist_map_x, undist_map_y = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (width, height), cv2.CV_32FC1)

        distortion_map   = undistorted_points                          # H x W x 2
        undistortion_map = np.stack([undist_map_x, undist_map_y], -1)  # H x W x 2

        return fisheye_img, distortion_map, undistortion_map

    def _create_grid_tensor(self, map_x, map_y):
        '''Convert pixel-coordinate maps to a normalised [-1,1] grid tensor of shape [H,W,2].'''
        H, W = map_x.shape
        X_norm = (map_x / (W - 1)) * 2 - 1
        Y_norm = (map_y / (H - 1)) * 2 - 1
        grid = np.stack((X_norm, Y_norm), axis=-1)
        return torch.from_numpy(grid).float()

    def _crop_and_resize_image(self, image, intrinsic, output_size=(518, 518)):
        '''Resize image to output_size and adjust intrinsics accordingly.'''
        W_out, H_out = output_size
        H = image.shape[0]
        K_out = np.array(intrinsic, dtype=np.float32).copy()
        s = H_out / float(H)
        img_out = cv2.resize(image, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
        K_out[0, 0] *= s
        K_out[1, 1] *= s
        K_out[0, 2] *= s
        K_out[1, 2] *= s
        return img_out, K_out
