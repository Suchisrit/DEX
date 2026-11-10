import os, sys
import torch
sys.path.insert(0, os.path.join('external_src', 'monocular_depth', 'MiDaS', 'midas'))
from dpt_depth import DPTDepthModel


class MiDaSModel(object):
    '''
    Class for interfacing with the MiDaS DPT depth model.

    Arg(s):
        device : torch.device
            device to run model on
        model_type : str
            model variant; currently only 'dpt_large_384' is supported
    '''
    def __init__(self,
                 device=torch.device('cuda'),
                 model_type='dpt_large_384'):

        self.device = device

        ckpt_path = os.path.join('model_ckpt', 'dpt_large_384.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        if model_type == 'dpt_large_384':
            self.model = DPTDepthModel(
                path=ckpt_path,
                backbone='vitl16_384',
                non_negative=True
            )
        else:
            raise NotImplementedError(f"Unsupported MiDaS model type: {model_type}")

        self.to(self.device)

    def forward(self, image, dex=None):
        '''
        Forwards inputs through the network.

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image, already normalized with mean 0.5 and std 0.5
            dex : nn.ModuleList or None
                per-block Distortion Extender parameters (E, W1, W2)
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''
        prediction = self.model.forward(image, dex)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[2:],
            mode='bicubic',
            align_corners=False)
        prediction = 1.0 / (prediction + 1e-7)
        return prediction

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.to(device)
