import torch
from external_src.monocular_depth.MiDaS.midas.dpt_depth import DPTDepthModel
from external_src.monocular_depth.transforms import Resize

class DPTModel():
    '''
    Class for interfacing with DPT model vitl16_384

    Arg(s):
        device : torch.device
            device to run model on
    '''
    def __init__(self, device=torch.device('cuda')):

        self.device = device

        self.model = DPTDepthModel(
            path=None,
            backbone='vitl16_384',
            non_negative=True)

        self.net_w, self.net_h = 384, 384

        self.resize_transform = Resize(
            self.net_w,
            self.net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method='minimal',
            image_interpolation_method='bicubic')

        # move model to device
        self.to(self.device)

    def forward(self, image):
        '''
        Forwards inputs through the network

        Arg(s):
            image : tensor[float32]
                N x 3 x H x W image
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''
        prediction = self.model.forward(image)

        prediction = 1.0/(prediction+1e-7)
        return prediction

    def restore_model(self,
                      restore_path,
                      optimizer=None):
        '''
        Loads weights from checkpoint and loads and returns optimizer

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            torch.optimizer if optimizer is passed in
        '''

        if isinstance(restore_path, list):
            restore_path = restore_path[0]

        checkpoint_dict = torch.load(restore_path, map_location=self.device)

        if 'state_dict' in checkpoint_dict:

            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint_dict['state_dict'])
            elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(checkpoint_dict['state_dict'])
            else:
                self.model.load_state_dict(checkpoint_dict['state_dict'])

            if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
                optimizer.load_state_dict(checkpoint_dict['optimizer'])

            if 'train_step' in checkpoint_dict.keys():
                train_step = checkpoint_dict['train_step']
                return train_step, optimizer
            else:
                return optimizer
        else:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint_dict)
            elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(checkpoint_dict)
            else:
                self.model.load_state_dict(checkpoint_dict)

            return None

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model = torch.nn.DataParallel(self.model)

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        if isinstance(self.model, torch.nn.DataParallel):
            checkpoint = {
                'state_dict': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            checkpoint = {
                'net': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        else:
            checkpoint = {
                'net': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }

        torch.save(checkpoint, checkpoint_path)

    def compute_loss(self,
                     output_depth,
                     ground_truth_depth,
                     l1_weight=1.0,
                     l2_weight=1.0):
        '''

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth_depth : torch.Tensor[float32]
                N x 2 x H x W ground_truth depth and ground truth validity map
            l1_weight : float
                weight of l1 loss
            l2_weight : float
                weight of l2 loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        l1_loss = torch.abs(output_depth - ground_truth_depth[:, 0:1])
        l2_loss = (output_depth - ground_truth_depth[:, 0:1]) ** 2

        loss = l1_weight * l1_loss.mean() + l2_weight * l2_loss.mean()

        loss_info = {
            "l1_loss": l1_loss.mean().item(),
            "l2_loss": l2_loss.mean().item(),
            "total_loss": loss.item()
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                parameters.append(param)

        parameters = torch.nn.ParameterList(parameters)

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''
        self.model.eval()

    def to(self, device):
        '''
        Moves model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.model.to(device)
