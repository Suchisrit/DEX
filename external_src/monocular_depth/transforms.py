'''
This module contains transforms needed for training augmentations
'''
import torch
import torchvision.transforms.functional as functional
from PIL import Image, ImageEnhance
import numpy as np
import numbers
import types
import collections
try:
    import accimage
except ImportError:
    accimage = None
import scipy.ndimage.interpolation as itpl
import scipy.misc as misc

def normalize_images(images_arr, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            images_arr[torch.Tensor[float32]] : list of normalized N x C x H x W tensors
        '''

        do_normalization_standard = any([
            isinstance(value, tuple) or isinstance(value, list)
            for value in normalized_image_range])

        if normalized_image_range == [0, 1]:
            images_arr = [
                images / 255.0 for images in images_arr
            ]

        elif do_normalization_standard:
            # Perform standard normalization
            mean, std = normalized_image_range[0], normalized_image_range[1]
            images_arr = [
                functional.normalize(
                    images / 255.0,
                    mean,
                    std)
                for images in images_arr
            ]

        elif normalized_image_range == [-1, 1]:
            images_arr = [
                2.0 * (images / 255.0) - 1.0 for images in images_arr
            ]
        elif normalized_image_range == [0, 255]:
            pass
        else:
            raise ValueError('Unsupported normalization range: {}'.format(
                normalized_image_range))

        return images_arr


# MiDAS - specific transforms # TODO: Genearalize late
class Resize(object):
    """
    Resize sample to given size (width, height).
    """
    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method='bicubic',
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__interpolation_mode_map = {
            'nearest' : Image.NEAREST,
            'bilinear' : Image.BILINEAR,
            'bicubic' : Image.BICUBIC,
            'lanczos' : Image.LANCZOS,#in previous version: Image.ANTIALIAS,
        }

        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = self.__interpolation_mode_map[image_interpolation_method]

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y
    
    def get_size(self, width, height):

        # Determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"resize method {self.__resize_method } is not implemented")
            
        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize method {self.__resize_method } is not implemented")
        
        return new_width, new_height#(new_width.item(), new_height.item())
    
    def __call__(self, images_batch):

        images_batch = [images.float() for images in images_batch]
        #print(f'Image batch len and element shape {len(images_batch)} {images_batch[0].shape}')
        n_dim = images_batch[0].dim()
        if n_dim == 4:
            n_batch, n_channel, n_height, n_width = images_batch[0].shape
        else:
            #print(f'n_dim is {n_dim}')
            raise ValueError(f'Resize transform input {n_dim} is wrong')
        
        width, height = self.get_size(height=n_height, width=n_width)
        #print(f'new height and width {height} {width}')
        n_images_batch = len(images_batch)
        interpolation_modes = n_images_batch * [self.__image_interpolation_method]

        for i, (images, interpolation_mode) in enumerate(zip(images_batch, interpolation_modes)):
            images_resized = []

            for b, image in enumerate(images):

                # Resize image
                image = functional.resize(
                    image,
                    size=(height,width),
                    interpolation=interpolation_mode
                )
                images_resized.append(image)
            
            images_batch[i] = torch.stack(images_resized, dim=0)

        return images_batch
        

class Transforms(object):

    def __init__(self,
                 normalized_image_range=[0, 255],
                 random_brightness=[-1],
                 random_contrast=[-1],
                 random_saturation=[-1],
                 random_noise_type='none',
                 random_noise_spread=-1,
                 random_flip_type=['none']):
        '''
        Transforms and augmentation class
        Note: brightness, contrast, gamma, hue, saturation augmentations expect
        either type int in [0, 255] or float in [0, 1]

        Arg(s):
            normalized_image_range : list[float]
                intensity range after normalizing images
            random_brightness : list[float]
                brightness adjustment [0, B], from 0 (black image) to B factor increase
            random_contrast : list[float]
                contrast adjustment [0, C], from 0 (gray image) to C factor increase
            random_saturation : list[float]
                saturation adjustment [0, S], from 0 (black image) to S factor increase
            random_noise_type : str
                type of noise to add: gaussian, uniform
            random_noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
            random_flip_type : list[str]
                none, horizontal, vertical
        '''
        # Image normalization
        self.normalized_image_range = normalized_image_range

        # Photometric augmentations
        self.do_random_brightness = True if -1 not in random_brightness else False
        self.random_brightness = random_brightness
        self.do_random_contrast = True if -1 not in random_contrast else False
        self.random_contrast = random_contrast
        self.do_random_saturation = True if -1 not in random_saturation else False
        self.random_saturation = random_saturation

        self.do_random_noise = \
            True if (random_noise_type != 'none' and random_noise_spread > -1) else False

        self.random_noise_type = random_noise_type
        self.random_noise_spread = random_noise_spread

        # Geometric augmentations
        self.do_random_horizontal_flip = True if 'horizontal' in random_flip_type else False
        self.do_random_vertical_flip = True if 'vertical' in random_flip_type else False

    def transform(self,
                  images_arr,
                  range_maps_arr=[],
                  points_arr=[],
                  bounding_boxes_arr=[],
                  random_transform_probability=0.50):
        '''
        Applies transform to images and ground truth

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            range_maps_arr : list[torch.Tensor]
                list of N x c x H x W tensors
            points_arr : list[torch.Tensor]
                list of N x 3 tensors
            bounding_boxes_arr : list[torch.Tensor]
                list of N x 4 tensors
            random_transform_probability : float
                probability to perform transform
        Returns:
            list[torch.Tensor] : list of transformed tensors
        '''

        device = images_arr[0].device

        n_dim = images_arr[0].ndim

        if n_dim == 4:
            n_batch, _, n_height, n_width = images_arr[0].shape
        else:
            raise ValueError('Unsupported number of dimensions: {}'.format(n_dim))

        do_random_transform = \
            torch.rand(n_batch, device=device) <= random_transform_probability

        '''
        Photometric Transformations (applied only to images)
        '''
        for idx, images in enumerate(images_arr):
            # In case user pass in [0, 255] range image as float type
            if torch.max(images) > 1.0:
                images_arr[idx] = images.int()

        if self.do_random_brightness:

            do_brightness = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            brightness_min, brightness_max = self.random_brightness
            factors = (brightness_max - brightness_min) * values + brightness_min

            images_arr = self.adjust_brightness(images_arr, do_brightness, factors)

        if self.do_random_contrast:

            do_contrast = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            contrast_min, contrast_max = self.random_contrast
            factors = (contrast_max - contrast_min) * values + contrast_min

            images_arr = self.adjust_contrast(images_arr, do_contrast, factors)

        if self.do_random_saturation:

            do_saturation = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            values = torch.rand(n_batch, device=device)

            saturation_min, saturation_max = self.random_saturation
            factors = (saturation_max - saturation_min) * values + saturation_min

            images_arr = self.adjust_saturation(images_arr, do_saturation, factors)

        '''
        Convert all images to float and normalize
        '''
        images_arr = [
            images.float() for images in images_arr
        ]

        # Normalize images to a given range
        images_arr = self.normalize_images(
            images_arr,
            normalized_image_range=self.normalized_image_range)

        '''
        Points augmentation
        '''
        if self.do_random_noise:
            do_add_noise = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)
            points_arr = self.add_noise(
                points_arr,
                do_add_noise=do_add_noise,
                noise_type=self.random_noise_type,
                noise_spread=self.random_noise_spread)

        '''
        Geometric Transformations
        '''
        if self.do_random_horizontal_flip:

            do_horizontal_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.horizontal_flip(
                images_arr,
                do_horizontal_flip)

            range_maps_arr = self.horizontal_flip(
                range_maps_arr,
                do_horizontal_flip)

            for bounding_boxes in bounding_boxes_arr:
                for bounding_box_idx in range(0,bounding_boxes.shape[0]):
                    do_hflip = do_horizontal_flip[bounding_box_idx]
                    for box_idx in range(0,bounding_boxes.shape[1]):
                        if do_hflip:
                            temp = bounding_boxes[bounding_box_idx, box_idx, 0].clone()
                            bounding_boxes[bounding_box_idx, box_idx, 0] = n_width - bounding_boxes[bounding_box_idx, box_idx, 2]
                            bounding_boxes[bounding_box_idx, box_idx, 2] = n_width - temp

        if self.do_random_vertical_flip:

            do_vertical_flip = torch.logical_and(
                do_random_transform,
                torch.rand(n_batch, device=device) <= 0.50)

            images_arr = self.vertical_flip(
                images_arr,
                do_vertical_flip)

            range_maps_arr = self.vertical_flip(
                range_maps_arr,
                do_vertical_flip)

            for bounding_boxes in bounding_boxes_arr:
                for bounding_box_idx in range(0,bounding_boxes.shape[0]):
                    do_vflip = do_vertical_flip[bounding_box_idx]
                    if do_vflip:
                        temp = bounding_boxes[bounding_box_idx, 1].clone()
                        bounding_boxes[bounding_box_idx, 1] = n_height - bounding_boxes[bounding_box_idx, 3]
                        bounding_boxes[bounding_box_idx, 3] = n_height - temp

        # Return the transformed inputs
        outputs = []

        if len(images_arr) > 0:
            outputs.append(images_arr)

        if len(range_maps_arr) > 0:
            outputs.append(range_maps_arr)

        if len(points_arr) > 0:
            outputs.append(points_arr)

        if len(bounding_boxes_arr) > 0:
            outputs.append(bounding_boxes_arr)
        

        return outputs[0] if len(outputs) == 1 else outputs

    '''
    Photometric transforms
    '''
    def normalize_images(self, images_arr, normalized_image_range=[0, 1]):
        '''
        Normalize image to a given range

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            normalized_image_range : list[float]
                intensity range after normalizing images
        Returns:
            images_arr[torch.Tensor[float32]] : list of normalized N x C x H x W tensors
        '''

        if normalized_image_range == [0, 1]:
            images_arr = [
                images / 255.0 for images in images_arr
            ]
        elif normalized_image_range == [-1, 1]:
            images_arr = [
                2.0 * (images / 255.0) - 1.0 for images in images_arr
            ]
        elif normalized_image_range == [0, 255]:
            pass
        else:
            raise ValueError('Unsupported normalization range: {}'.format(
                normalized_image_range))

        return images_arr

    def adjust_brightness(self, images_arr, do_brightness, factors):
        '''
        Adjust brightness on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_brightness : bool
                N booleans to determine if brightness is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_brightness[b]:
                    images[b, ...] = functional.adjust_brightness(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_contrast(self, images_arr, do_contrast, factors):
        '''
        Adjust contrast on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_contrast : bool
                N booleans to determine if contrast is adjusted on each sample
            factors : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_contrast[b]:
                    images[b, ...] = functional.adjust_contrast(image, factors[b])

            images_arr[i] = images

        return images_arr

    def adjust_saturation(self, images_arr, do_saturation, factors):
        '''
        Adjust saturation on each sample

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_saturation : bool
                N booleans to determine if saturation is adjusted on each sample
            gammas : float
                N floats to determine how much to adjust
        Returns:
            list[torch.Tensor] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_saturation[b]:
                    images[b, ...] = functional.adjust_saturation(image, factors[b])

            images_arr[i] = images

        return images_arr

    '''
    Geometric transforms
    '''
    def horizontal_flip(self, images_arr, do_horizontal_flip):
        '''
        Perform horizontal flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_horizontal_flip : bool
                N booleans to determine if horizontal flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_horizontal_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-1])

            images_arr[i] = images

        return images_arr

    def vertical_flip(self, images_arr, do_vertical_flip):
        '''
        Perform vertical flip on each sample

        Arg(s):
            images_arr : list[torch.Tensor[float32]]
                list of N x C x H x W tensors
            do_vertical_flip : bool
                N booleans to determine if vertical flip is performed on each sample
        Returns:
            list[torch.Tensor[float32]] : list of transformed N x C x H x W image tensors
        '''

        for i, images in enumerate(images_arr):

            for b, image in enumerate(images):
                if do_vertical_flip[b]:
                    images[b, ...] = torch.flip(image, dims=[-2])

            images_arr[i] = images

        return images_arr

    def add_noise(self, images_arr, do_add_noise, noise_type, noise_spread):
        '''
        Add noise to images

        Arg(s):
            images_arr : list[torch.Tensor]
                list of N x C x H x W tensors
            do_add_noise : bool
                N booleans to determine if noise will be added
            noise_type : str
                gaussian, uniform
            noise_spread : float
                if gaussian, then standard deviation; if uniform, then min-max range
        '''

        for i, images in enumerate(images_arr):
            device = images.device

            for b, image in enumerate(images):
                if do_add_noise[b]:

                    shape = image.shape

                    if noise_type == 'gaussian':
                        image = image + noise_spread * torch.randn(*shape, device=device)
                    elif noise_type == 'uniform':
                        image = image + noise_spread * (torch.rand(*shape, device=device) - 0.5)
                    else:
                        raise ValueError('Unsupported noise type: {}'.format(noise_type))

                    images[b, ...] = image

            images_arr[i] = images

        return images_arr

# Transformation for the lin model

# Check whether the input is a numpy array
def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# Check whether the input is a PIL image
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


# Check wheter the input is a tensor
def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        PIL Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
    See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    np_img = np.array(img, dtype=np.float32)
    np_img = 255 * gain * ((np_img / 255) ** gamma)
    np_img = np.uint8(np.clip(np_img, 0, 255))

    img = Image.fromarray(np_img, 'RGB').convert(input_mode)
    return img

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Rotate(object):
    """Rotates the given ``numpy.ndarray``.
    Args:
        angle (float): The rotation angle in degrees.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be rotated.
        Returns:
            img (numpy.ndarray (C x H x W)): Rotated image.
        """

        # order=0 means nearest-neighbor type interpolation
        return itpl.rotate(img, self.angle, reshape=False, prefilter=False, order=0)


class LinResize(object):
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation='nearest'):
        assert isinstance(size, int) or isinstance(size, float) or \
               (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        if img.ndim == 3:
            return misc.imresize(img, self.size, self.interpolation)
        elif img.ndim == 2:
            return misc.imresize(img, self.size, self.interpolation, 'F')
        else:
            RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

class HorizontalFlip(object):
    """Horizontally flip the given ``numpy.ndarray``.
    Args:
        do_flip (boolean): whether or not do horizontal flip.
    """

    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be flipped.
        Returns:
            img (numpy.ndarray (C x H x W)): flipped image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if self.do_flip:
            return np.fliplr(img)
        else:
            return img

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Input image.
        Returns:
            img (numpy.ndarray (C x H x W)): Color jittered image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        pil = Image.fromarray(img)
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return np.array(transform(pil))

class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    """

    def __call__(self, img):
        """Convert a ``numpy.ndarray`` to tensor.
        Args:
            img (numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if isinstance(img, np.ndarray):
            # handle numpy array
            if img.ndim == 3:
                img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
            elif img.ndim == 2:
                img = torch.from_numpy(img.copy())
            else:
                raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

            # backward compatibility
            # return img.float().div(255)
            return img.float()

class CenterCrop(object):
    """Crops the given ``numpy.ndarray`` at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for center crop.
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        # # randomized cropping
        # i = np.random.randint(i-3, i+4)
        # j = np.random.randint(j-3, j+4)

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)

        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i+h, j:j+w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

