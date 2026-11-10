import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from matplotlib import cm
_JET_LUT = torch.from_numpy((cm.get_cmap('jet')(np.linspace(0,1,256))[:, :3])).float()

def depth_to_jet_rgb(d: torch.Tensor) -> torch.Tensor:
    '''
    d: (1,H,W) or (H,W) float tensor on CPU, any range
    returns: (3,H,W) float in [0,1]
    '''
    if d.ndim == 3 and d.size(0) == 1:
        d = d.squeeze(0)
    d = d.float()
    d = (d - d.min()) / (d.max() - d.min() + 1e-12)      # [0,1]
    idx = (d * 255).round().to(torch.long).clamp_(0,255) # (H,W)
    rgb = _JET_LUT[idx]                                   # (H,W,3)
    return rgb.permute(2, 0, 1)                             # (3,H,W)


class Model(nn.Module):
    def __init__(self,
                 device=torch.device('cuda'),
                 num_extenders=12,
                 model_name='unidepth',
                 output_reference_frame='distance'):
        '''
        Arg(s):
            device : torch.device
                device to run the model on
            num_extenders : int
                number of learnable Distortion Extenders per encoder block
            model_name : str
                backbone depth model: 'unidepth', 'depthanything', or 'midas'
            output_reference_frame : str
                reference frame the standard prediction is aligned to before computing the loss:
                'distance' (default) converts z-depth to Euclidean distance from the camera center (R),
                'depth' keeps the raw z-depth prediction
        '''

        super().__init__()
        self.device = device
        self.model_name = model_name
        self.output_reference_frame = output_reference_frame

        patch_size = 16

        if 'depthanything' == model_name:
            from src.monocular_depth.depth_anything_model import DepthAnythingModel
            self.model = DepthAnythingModel(
                device=device,
                use_pretrained=True,
                model_type='vitl'
            )
            patch_size = 14
            embed_dim = 1024
            num_blocks = 24

        elif 'unidepth' == model_name:
            from src.monocular_depth.unidepth_model import UniDepthModel
            self.model = UniDepthModel(
                device=device
            )
            patch_size = 14
            embed_dim = 384
            num_blocks = 12

        elif 'midas' == model_name:
            from src.monocular_depth.midas_model import MiDaSModel
            self.model = MiDaSModel(
                device=device
            )
            embed_dim = 1024
            num_blocks = 24

        else:
            raise ValueError('Unsupported monocular depth estimation model: {}'.format(model_name))

        # per-block Extenders E (M x D) and low-rank selection factors W1 (D x d), W2 (d x D)
        self.num_extenders = num_extenders
        low_rank_dim = 20
        fw = {'device': device, 'dtype': torch.float32}
        self.dex = nn.ModuleList([
            nn.ParameterDict({
                'E': nn.Parameter(torch.empty(num_extenders, embed_dim, **fw)),
                'W1': nn.Parameter(torch.empty(embed_dim, low_rank_dim, **fw)),
                'W2': nn.Parameter(torch.empty(low_rank_dim, embed_dim, **fw)),
            }) for _ in range(num_blocks)
        ])

        for block_dex in self.dex:
            for param in block_dex.values():
                nn.init.xavier_uniform_(param)

        self.patch_size = patch_size

        self.train_summary_writer = None
        self.log_step = None
        self.current_dataset = None

    def infer(self, image):
        '''
        Predicts from a fisheye image with the Extenders, averaging with the prediction of the horizontally flipped image

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W normalized fisheye image
        Returns:
            torch.Tensor[float32] : N x 1 x H x W prediction
        '''

        pred = self.model.forward(image, self.dex)
        pred_flipped = self.model.forward(torch.flip(image, [-1]), self.dex)

        return (pred + torch.flip(pred_flipped, [-1])) / 2

    def depth_to_distance(self, d, K):
        '''
        Convert z-depth (along camera Z) to Euclidean distance from the camera center.

        d: (B,1,H,W) depth in meters
        K: (B,3,3) intrinsics

        returns: (B,1,H,W) Euclidean distance
        '''
        B, _, H, W = d.shape
        device, dtype = d.device, d.dtype

        # Find inverse intrinsics
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]

        Kinv = torch.zeros((B, 3, 3), device=device, dtype=dtype)
        Kinv[:, 0, 0] = 1.0/fx
        Kinv[:, 1, 1] = 1.0/fy
        Kinv[:, 0, 2] = -cx/fx
        Kinv[:, 1, 2] = -cy/fy
        Kinv[:, 2, 2] = 1.0

        u = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(1, 1, H, W)
        v = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(1, 1, H, W)
        ones = torch.ones((1, 1, H, W), device=device, dtype=dtype)
        p = torch.cat([u, v, ones], dim=1) # (1,3,H,W)
        p = p.expand(B, -1, -1, -1).contiguous() # (B,3,H,W)

        p_camera = torch.einsum('bij,bjhw->bihw', Kinv, p)   # (B,3,H,W)
        p_backproj = p_camera * d # (B,3,H,W)
        R = torch.norm(p_backproj, dim=1, keepdim=True) # (B,1,H,W)

        return R

    def forward(self, standard_img, fisheye_img, intrinsics, grid_tensor, for_grid_tensor, rank):
        '''
        Arg(s):
            standard_img : torch.Tensor[float32]
                N x 3 x H x W perspective image
            fisheye_img : torch.Tensor[float32]
                N x 3 x H x W fisheye image
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsic matrix for the perspective image
            grid_tensor : torch.Tensor[float32]
                N x H x W x 2 sampling grid to undistort the fisheye image
            for_grid_tensor : torch.Tensor[float32]
                N x H x W x 2 sampling grid to distort a perspective image to fisheye
            rank : int
                process rank (used to gate logging to rank 0)
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                standard_distance_pred : N x 1 x H x W Euclidean distance from the perspective image
                fisheye_depth_pred     : N x 1 x H x W prediction from the fisheye image (training to output distance as well)
        '''

        standard_img = standard_img.to(self.device)
        fisheye_img = fisheye_img.to(self.device)

        with torch.no_grad():
            standard_depth_pred = self.model.forward(standard_img)

        fisheye_depth_pred = self.model.forward(fisheye_img, self.dex)

        # Logging visualizations
        if self.log_step is not None and rank == 0:

            undist_fisheye_img = F.grid_sample(fisheye_img, grid_tensor, mode='nearest',
                    padding_mode='zeros', align_corners=False)
            vis_standard_img = standard_img.clone().cpu()
            vis_fisheye_img = fisheye_img.clone().cpu()

            for i in range(standard_img.shape[0]):
                standard_image = vis_standard_img[i]
                fisheye_image = vis_fisheye_img[i]
                undist_fisheye_image = undist_fisheye_img[i]

                norm_standard_image = (standard_image - standard_image.min()) / (standard_image.max() - standard_image.min() + 1e-12)
                norm_fisheye_image = (fisheye_image - fisheye_image.min()) / (fisheye_image.max() - fisheye_image.min() + 1e-12)
                norm_undist_fisheye_image = (undist_fisheye_image - undist_fisheye_image.min()) / (undist_fisheye_image.max() - undist_fisheye_image.min() + 1e-12)

                vis_standard_img[i] = norm_standard_image
                vis_fisheye_img[i] = norm_fisheye_image
                undist_fisheye_img[i] = norm_undist_fisheye_image

            self.train_summary_writer.add_image(f'train_image/standard_image_{self.current_dataset}', make_grid(vis_standard_img, nrow=(standard_img.shape[0] // 2)), self.log_step)
            self.train_summary_writer.add_image(f'train_image/fisheye_image_{self.current_dataset}', make_grid(vis_fisheye_img, nrow=(fisheye_img.shape[0] // 2)), self.log_step)
            self.train_summary_writer.add_image(f'train_image/undistorted_fisheye_image_{self.current_dataset}', make_grid(undist_fisheye_img, nrow=(undist_fisheye_img.shape[0] // 2)), self.log_step)

        # Optionally convert z-depth to Euclidean distance (R) from the camera center.
        # When output_reference_frame == 'depth', the raw z-depth prediction is kept instead.
        if self.output_reference_frame == 'distance':
            standard_pred = self.depth_to_distance(standard_depth_pred, intrinsics)
        else:
            standard_pred = standard_depth_pred

        return standard_pred, fisheye_depth_pred

    def direct_loss(self,
                    standard_depth,
                    fisheye_depth,
                    grid_tensor,
                    for_grid_tensor,
                    rank=0):
        '''
        Computes the undistortion prediction loss: aligns fisheye depth with standard depth
        by undistorting the fisheye prediction using the grid maps.

        returns:
            The computed regression loss (torch.Tensor)
        '''

        # Log depth prediction visualizations
        if self.log_step is not None and rank == 0:
            with torch.no_grad():
                vis_standard_depth = standard_depth.clone().cpu()
                vis_fisheye_depth = fisheye_depth.clone().cpu()
                vis_undist_fisheye_depth = F.grid_sample(
                    fisheye_depth.clone().cpu(),
                    grid_tensor.clone().cpu(),
                    mode='nearest',
                    padding_mode='zeros',
                    align_corners=False)

                vis_standard_depth_result = torch.zeros((vis_standard_depth.shape[0], 3, vis_standard_depth.shape[2], vis_standard_depth.shape[3]))
                vis_fisheye_depth_result = torch.zeros((vis_fisheye_depth.shape[0], 3, vis_fisheye_depth.shape[2], vis_fisheye_depth.shape[3]))
                vis_undist_fisheye_depth_result = torch.zeros((vis_undist_fisheye_depth.shape[0], 3, vis_undist_fisheye_depth.shape[2], vis_undist_fisheye_depth.shape[3]))

                for i in range(vis_standard_depth.shape[0]):
                    standard_depth1 = vis_standard_depth[i]
                    fisheye_depth1 = vis_fisheye_depth[i]
                    undist_fisheye_depth1 = vis_undist_fisheye_depth[i]

                    norm_standard_depth = (standard_depth1 - standard_depth1.min()) / (standard_depth1.max() - standard_depth1.min() + 1e-12)
                    norm_fisheye_depth = (fisheye_depth1 - fisheye_depth1.min()) / (fisheye_depth1.max() - fisheye_depth1.min() + 1e-12)
                    norm_undist_fisheye_depth = (undist_fisheye_depth1 - undist_fisheye_depth1.min()) / (undist_fisheye_depth1.max() - undist_fisheye_depth1.min() + 1e-12)

                    vis_standard_depth_result[i] = depth_to_jet_rgb(norm_standard_depth)
                    vis_fisheye_depth_result[i] = depth_to_jet_rgb(norm_fisheye_depth)
                    vis_undist_fisheye_depth_result[i] = depth_to_jet_rgb(norm_undist_fisheye_depth)

                self.train_summary_writer.add_image(f'train_depth/standard_depth_{self.current_dataset}', make_grid(vis_standard_depth_result, nrow=(vis_standard_depth.shape[0] // 2)), self.log_step)
                self.train_summary_writer.add_image(f'train_depth/fisheye_depth_{self.current_dataset}', make_grid(vis_fisheye_depth_result, nrow=(vis_fisheye_depth.shape[0] // 2)), self.log_step)
                self.train_summary_writer.add_image(f'train_depth/undistorted_fisheye_depth_{self.current_dataset}', make_grid(vis_undist_fisheye_depth_result, nrow=(vis_undist_fisheye_depth.shape[0] // 2)), self.log_step)

                # Compute and visualize error maps
                import matplotlib.pyplot as plt
                error_map = (vis_standard_depth - vis_undist_fisheye_depth).abs()
                hot_cmap = plt.get_cmap('hot')
                vis_error_result = torch.zeros((error_map.shape[0], 3, error_map.shape[2], error_map.shape[3]))
                max_error = 5.0  # maximum error for visualization normalization
                for i in range(error_map.shape[0]):
                    err = error_map[i, 0]
                    err_norm = (err / max_error).clamp(0, 1)
                    err_rgb = hot_cmap(err_norm.numpy())[:, :, :3]
                    vis_error_result[i] = torch.from_numpy(err_rgb).permute(2, 0, 1).float()
                self.train_summary_writer.add_image(f'train_depth/error_map_{self.current_dataset}', make_grid(vis_error_result, nrow=(error_map.shape[0] // 2)), self.log_step)

                del vis_standard_depth_result, vis_fisheye_depth_result, vis_undist_fisheye_depth_result, vis_error_result
                torch.cuda.empty_cache()

        # Create mask for transformation artifacts
        fisheye_mask = torch.ones_like(fisheye_depth)
        dist_mask = F.grid_sample(fisheye_mask, for_grid_tensor, mode='nearest',
                                        padding_mode='zeros', align_corners=False).bool()

        undist_mask = F.grid_sample(dist_mask.float(), grid_tensor, mode='nearest',
                                        padding_mode='zeros', align_corners=False).bool()
        undist_depth_pred = F.grid_sample(fisheye_depth, grid_tensor, mode='nearest',
                                            padding_mode='zeros', align_corners=False)
        undist_mask = undist_mask.bool()

        loss = torch.log(1 + ((standard_depth - undist_depth_pred)[undist_mask]).abs()).mean()

        return loss

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        super().to(device)

    def restore_model(self,
                      restore_path,
                      optimizer=None,
                      enable_ddp=False,
                      rank= 0):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_paths : list[str]
                path to model weights, 1st for depth model and 2nd for pose model (if exists)
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            int : training step
            torch.optimizer : optimizer for depth or None if no optimizer is passed in
            torch.optimizer : optimizer for pose or None if no optimizer is passed in
        '''
        if enable_ddp :
            checkpoint = torch.load(restore_path, map_location={"cuda:0": f"cuda:{rank}"})
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('dex.')}
            self.load_state_dict(pretrained_dict, strict=False)
        else:
            checkpoint = torch.load(restore_path, map_location=self.device)
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()}
            self.load_state_dict(pretrained_dict, strict=False)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        step = checkpoint.get('step', 0)

        return step, optimizer

    def save_model(self,
                   checkpoint_dirpath,
                   step,
                   optimizer=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_dirpath : str
                path to save directory to save checkpoints
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''
        os.makedirs(checkpoint_dirpath, exist_ok=True)

        checkpoint_filepath = os.path.join(checkpoint_dirpath, f'checkpoint_{step}.pth')
        state_dict = self.state_dict()

        checkpoint = {
            'model_state_dict': {k: v for k, v in state_dict.items() if k.startswith('dex.')},
            'step': step
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_filepath)

        print(f'Model checkpoint saved at {checkpoint_filepath}')
