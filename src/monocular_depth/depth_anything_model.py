import os, sys
import torch
sys.path.insert(0, os.path.join('external_src', 'monocular_depth', 'Depth_Anything'))
from depth_anything.dpt import DepthAnything as DepthAnythingBaseModel


class DepthAnythingModel(object):
    '''
    Class for interfacing with Depth-Anything model.

    Arg(s):
        device : torch.device
            device to run model on
        use_pretrained : bool
            if True, load pretrained weights from model_ckpt/depthanything/
        model_type : str
            encoder backbone: 'vitl', 'vitb', or 'vits'
    '''
    def __init__(self, device=torch.device('cuda'), use_pretrained=True, model_type='vitl'):
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]}
        }

        self.model = DepthAnythingBaseModel(model_configs[model_type])
        self.device = device

        if use_pretrained:
            ckpt_path = os.path.join('model_ckpt', 'depthanything', f'depth_anything_{model_type}14.pth')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            print(f"Loading DepthAnything checkpoint from: {ckpt_path}")
            self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

        self.to(self.device)

    def forward(self, image, dex=None):
        '''
        Forwards inputs through the network.

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            dex : nn.ModuleList or None
                per-block Distortion Extender parameters (E, W1, W2)
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''
        image = image.to(self.device)
        output = self.model(image, dex=dex)
        output = torch.nn.functional.interpolate(output.unsqueeze(1), image.shape[2:], mode='nearest')
        output = 1.0 / (output + 1e-7)
        return output

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.to(device)
