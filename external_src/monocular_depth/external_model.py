import torch, torchvision
import log_utils

class ExternalMonocularDepthEstimationModel(object):
    '''
    Wrapper class for all external monocular depth estimation models

    Arg(s):
        model_name : str
            monocular depth estimation model name
        min_predict_depth : float
            minimum depth to predict
        max_predict_depth : float
            maximum depth to predict
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 model_name,
                 min_predict_depth,
                 max_predict_depth,
                 device=torch.device('cuda')):

        self.model_name = model_name
        self.device = device
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        if 'depthanything' == model_name:
            from depth_anything_model import DepthAnythingModel
            self.model = DepthAnythingModel(
                device=device,
                max_predict_depth=max_predict_depth,
                use_pretrained=True,
                model_type='vitl'
            )
        elif 'dpt' == model_name:
            from dpt_model import DPTModel
            self.model = DPTModel(
                device=device,
                max_predict_depth=max_predict_depth,
                use_pretrained=True,
                model_type="dpt_hybrid_kitti"
            )
        elif 'unidepth' == model_name:
            from unidepth_model import UniDepthModel
            self.model = UniDepthModel(
                device=device,
                max_predict_depth=max_predict_depth,
                use_pretrained=True
            )
        elif 'midas' == model_name:
            from midas_model import MiDaSModel
            self.model = MiDaSModel(
                device=device
            )
        else:
            raise ValueError('Unsupported monocular depth estimation model: {}'.format(model_name))

    def forward(self, image):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''

        return self.model.forward(image)

    def compute_loss(self, output_depth, ground_truth, w_losses={}):
        '''
        Call the model's compute loss function

        Arg(s):
            output_depth : list[torch.Tensor[float32]]
                N x 1 x H x W dense output depth already masked with validity map or list of all outputs
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground_truth depth with only valid values
            w_losses : dict[str, float]
                dictionary of weights for each loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''
        return self.model.compute_loss(output_depth=output_depth, ground_truth_depth=ground_truth)

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters()

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
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.model.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model.data_parallel()

    def distributed_data_parallel(self, rank):

        self.model.distributed_data_parallel(rank)

    def restore_model(self, restore_paths, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_paths : list[str]
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        return self.model.restore_model(restore_paths, optimizer)

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

        self.model.save_model(checkpoint_path, step, optimizer)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image=None,
                    sparse_depth=None,
                    output_depth=None,
                    validity_map=None,
                    ground_truth=None,
                    scalars={},
                    n_image_per_summary=4,
                    scalars_secondary_loss={},
                    scaffnet_output=None,
                    scaffnet_uncertainty=None,
                    uncertainty_threshold=None,
                    image_uncertainty=None):
        '''
        Logs summary to Tensorboard
        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image : torch.Tensor[float32]
                N x 3 x H x W image from camera
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse_depth from LiDAR
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth for image
            validity_map : torch.Tensor[float32]
                N x 1 x H x W validity map from sparse depth
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth depth image
            scalars : dict[str, float]
                dictionary of scalars to log
            n_image_per_summary : int
                number of images to display
            scalars_secondary_loss : dict[str, float]
                dictionary of uncertainty loss info when training Fusion Net
            scaffnet_output : torch.Tensor[float32]
                N x 1 x H x W depth densified by scaffnet when training Fusion Net
        '''

        with torch.no_grad():
            display_summary_image = []
            display_summary_depth = []
            display_summary_scaffnet = []

            display_summary_image_text = tag
            display_summary_depth_text = tag
            display_summary_scaffnet_text = tag

            # Log image
            if image is not None:
                # Normalize for display if necessary
                if torch.max(image) > 1.0:
                    image = image / 255.0

                image_summary = image[0:n_image_per_summary, ...]
                display_summary_image_text += '_image'
                display_summary_depth_text += '_image'
                display_summary_scaffnet_text += '_image'
                display_summary_image.append(
                    torch.cat([
                        image_summary.cpu(),
                        torch.zeros_like(image_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            if image_uncertainty is not None:
                display_summary_scaffnet_text += '_image-unc'
                image_uncertainty_summary = image_uncertainty[0:n_image_per_summary]
                image_uncertainty_summary = (image_uncertainty_summary - torch.min(image_uncertainty_summary)) / \
                                            (torch.max(image_uncertainty_summary) - torch.min(
                                                image_uncertainty_summary))
                display_summary_scaffnet.append(
                    log_utils.colorize(image_uncertainty_summary.cpu(),
                                       colormap='plasma'))

            if scaffnet_output is not None:
                display_summary_scaffnet.append(image_summary.cpu())
                scaffnet_summary = scaffnet_output[0:n_image_per_summary]
                display_summary_scaffnet_text += '_scaffnet-output'

                display_summary_scaffnet.append(
                    log_utils.colorize((scaffnet_summary / self.max_predict_depth).cpu(),
                                       colormap='viridis'))

            if scaffnet_uncertainty is not None:
                display_summary_scaffnet_text += '_scaffnete-unc'
                scaffnet_uncertainty_summary = scaffnet_uncertainty[0:n_image_per_summary]
                scaffnet_uncertainty_summary = (scaffnet_uncertainty_summary - torch.min(
                    scaffnet_uncertainty_summary)) / \
                                               (torch.max(scaffnet_uncertainty_summary) - torch.min(
                                                   scaffnet_uncertainty_summary))
                display_summary_scaffnet.append(
                    log_utils.colorize(scaffnet_uncertainty_summary.cpu(),
                                       colormap='plasma'))

            if output_depth is not None:
                if output_depth.shape[1] > 1:
                    output_depth = output_depth[:, 0:1, :, :]

                output_depth_summary = output_depth[0:n_image_per_summary]
                display_summary_depth_text += '_output'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth_distro', output_depth, global_step=step)

            # Log output depth vs sparse depth
            if output_depth is not None and sparse_depth is not None and validity_map is not None:
                sparse_depth_summary = sparse_depth[0:n_image_per_summary]
                validity_map_summary = validity_map[0:n_image_per_summary]

                display_summary_depth_text += '_sparse-error'

                # Compute output error w.r.t. input sparse depth
                sparse_depth_error_summary = \
                    torch.abs(output_depth_summary - sparse_depth_summary)

                sparse_depth_error_summary = torch.where(
                    validity_map_summary == 1.0,
                    sparse_depth_error_summary / (sparse_depth_summary + 1e-8),
                    validity_map_summary)

                # Add to list of images to log
                sparse_depth_summary = log_utils.colorize(
                    (sparse_depth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                sparse_depth_error_summary = log_utils.colorize(
                    (sparse_depth_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        sparse_depth_summary,
                        sparse_depth_error_summary],
                        dim=3))

                # Log distribution of sparse depth
                summary_writer.add_histogram(tag + '_sparse_depth_distro', sparse_depth, global_step=step)

            # Log output depth vs ground truth depth
            if output_depth is not None and ground_truth is not None:
                validity_map_ground_truth = torch.where(
                    ground_truth > 0,
                    torch.ones_like(ground_truth),
                    torch.zeros_like(ground_truth))

                validity_map_ground_truth_summary = validity_map_ground_truth[0:n_image_per_summary]
                ground_truth_summary = ground_truth[0:n_image_per_summary]

                display_summary_depth_text += '_groundtruth-error'

                # Compute output error w.r.t. ground truth
                ground_truth_error_summary = \
                    torch.abs(output_depth_summary - ground_truth_summary)

                ground_truth_error_summary = torch.where(
                    validity_map_ground_truth_summary == 1.0,
                    (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                    validity_map_ground_truth_summary)

                # Add to list of images to log
                ground_truth_summary = log_utils.colorize(
                    (ground_truth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth_error_summary = log_utils.colorize(
                    (ground_truth_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth_summary,
                        ground_truth_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth_distro', ground_truth, global_step=step)

            if uncertainty_threshold is not None:
                display_summary_scaffnet_text += '_unc-thresh-input'

                sparse_input_summary = sparse_depth[0:n_image_per_summary]
                scaffnet_input_depth_summary = scaffnet_output[0:n_image_per_summary]
                scaffnet_input_uncertainty_summary = scaffnet_uncertainty[0:n_image_per_summary]
                dense_input_summary = torch.where(sparse_input_summary > 0,
                                                  sparse_input_summary,
                                                  scaffnet_input_depth_summary)

                dense_input_summary = torch.where(scaffnet_input_uncertainty_summary < uncertainty_threshold,
                                                  dense_input_summary,
                                                  torch.zeros_like(dense_input_summary))

                display_summary_scaffnet.append(
                    log_utils.colorize((dense_input_summary / self.max_predict_depth).cpu(),
                                       colormap='viridis'))

            # Log scalars to tensorboard
            for (name, value) in scalars.items():
                summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

            # Log scalars to tensorboard
            if len(scalars_secondary_loss) > 0:
                for (name, value) in scalars_secondary_loss.items():
                    summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

            # Log image summaries to tensorboard
            if len(display_summary_image) >= 1:
                display_summary_image = torch.cat(display_summary_image, dim=2)

                summary_writer.add_image(
                    display_summary_image_text,
                    torchvision.utils.make_grid(display_summary_image, nrow=n_image_per_summary),
                    global_step=step)
            if len(display_summary_scaffnet) >= 1:
                display_summary_scaffnet = torch.cat(display_summary_scaffnet, dim=2)

                summary_writer.add_image(
                    display_summary_scaffnet_text,
                    torchvision.utils.make_grid(display_summary_scaffnet, nrow=n_image_per_summary),
                    global_step=step)

            if len(display_summary_depth) >= 1:
                display_summary_depth = torch.cat(display_summary_depth, dim=2)

                summary_writer.add_image(
                    display_summary_depth_text,
                    torchvision.utils.make_grid(display_summary_depth, nrow=n_image_per_summary),
                    global_step=step)


class ExternalRadarCameraFusionModel(object):
    '''
    Wrapper class for all external monocular depth estimation models

    Arg(s):
        model_name : str
            model name of fusion models i.e. radarcamdepth, singh, dorn
        dataset_name : str
            model for a given dataset
        network_modules : list[str]
            list of additional network modules to build for model i.e. rcnet, sml, freeze_all
        min_predict_depth : float
            minimum depth to predict
        max_predict_depth : float
            maximum depth to predict
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 model_name,
                 dataset_name,
                 network_modules=[],
                 patch_size=[0, 0],
                 mde_model_name='dpt',
                 min_predict_depth=0.1,
                 max_predict_depth=100.0,
                 min_evaluate_depth=0.0,
                 max_evaluate_depth=80.0,
                 device=torch.device('cuda'),):
        self.model_name = model_name
        self.network_modules = network_modules
        self.device = device
        
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.min_evaluate_depth = min_evaluate_depth
        self.max_evaluate_depth = max_evaluate_depth

        if model_name == 'radarcamdepth':
            from radarcam_model import RadarCamDepthModel
            self.model = RadarCamDepthModel(network_modules=network_modules,
                                            patch_size=patch_size,
                                            min_predict_depth=min_predict_depth,
                                            max_predict_depth=max_predict_depth,   
                                            min_evaluate_depth=min_evaluate_depth,
                                            max_evaluate_depth=max_evaluate_depth,
                                            mde_model_name=mde_model_name, 
                                            device=device)
        elif model_name == 'singh':
            from radarnet_fusionnet_model import SinghModel
            self.model = SinghModel(network_modules=network_modules, 
                                    patch_size=patch_size, 
                                    min_predict_depth=min_predict_depth,
                                    max_predict_depth=max_predict_depth,   
                                    min_evaluate_depth=min_evaluate_depth,
                                    max_evaluate_depth=max_evaluate_depth,
                                    device=device)
        elif model_name == 'cafnet':
            from cafnet_model import CaFNetModel
            self.model = CaFNetModel(network_modules=network_modules, 
                                     patch_size=patch_size, 
                                     min_predict_depth=min_predict_depth,
                                     max_predict_depth=max_predict_depth,  
                                     min_evaluate_depth=min_evaluate_depth,
                                     max_evaluate_depth=max_evaluate_depth,
                                     device=device)
        elif model_name == 'lin':
            from lin_model import LinModel
            self.model = LinModel(network_modules=network_modules, 
                                  patch_size=patch_size, 
                                  min_predict_depth=min_predict_depth,
                                  max_predict_depth=max_predict_depth,  
                                  min_evaluate_depth=min_evaluate_depth,
                                  max_evaluate_depth=max_evaluate_depth,
                                  device=device)
        elif model_name == 'sparse_beats_dense':
            from sparse_beats_dense_model import SparseBeatsDenseModel
            
            self.model = SparseBeatsDenseModel(
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,  
                max_predict_depth=max_predict_depth,
                min_evaluate_depth=min_evaluate_depth,
                max_evaluate_depth=max_evaluate_depth,
                device=device)
        else:
            raise ValueError('Unsupported radar camera fusion depth estimation model: {}'.format(model_name))
        
        self.common_log_args = [
            "summary_writer", "tag", "step", 
            "image", "output_depth_fusion", "ground_truth", "validity_map"
            "scalars", "n_display"
        ]
        # default args usually unchanged independnt of model
        self.default_args = {
            "n_display" : 4
        }
        # Model-specific configuration avoid passing unnecessary arguments for logging
        self.model_specific_log_config = {
            "sparse_beats_dense": [
                "segmentation_pred", "segmentation_gt"
            ],
            # define model-specific arguments below if needed
            "default" : [
                 "input_response_map", "input_global_aligned_mono",
                 "output_logits", "output_quasi_dense_depth",
                 "ground_truth_label", "bounding_boxes_list"
            ]

        }

        print("\nExternal Radar Camera Fusion Model Initialized...\n")

    def forward(self,
                image,
                radar_points,
                sparse_depth,
                quasi_dense_depth=None,
                response_map=None,
                global_aligned_mono=None,
                bounding_boxes_list=[],
                valid_radar_pts_cnts=0,
                return_logits=True):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            radar_points : torch.Tensor[float32]
                N x 3 input point
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse depth
            quasi_dense_depth : torch.Tensor[float32]
                N x 1 x H x W quasi dense depth
            response_map : torch.Tensor[float32]
                N x 1 x H x W response map
            global_aligned_mono : torch.Tensor[float32]
                N x 1 x H x W global aligned monocular depth
            bounding_boxes_list : list[]
                list of bounding boxes
            return_logits : bool
                if set, then return logits otherwise sigmoid
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
            torch.Tensor[float32] : N x 1 x H x W quasi dense depth
            torch.Tensor[float32] : N x 1 x H x W response map
            torch.Tensor[float32] : N x 1 x H x W global aligned monocular depth
        '''
        return self.model.forward(image,
                                  radar_points,
                                  sparse_depth,
                                  quasi_dense_depth,
                                  response_map,
                                  global_aligned_mono,
                                  bounding_boxes_list,
                                  valid_radar_pts_cnts,
                                  return_logits)

    def compute_loss(self,
                     image,
                     output_depth_fusion,
                     ground_truth,
                     lidar_map,
                     loss_func,
                     w_smoothness,
                     loss_smoothness_kernel_size,
                     w_lidar_loss,
                     logits,
                     validity_map,
                     w_positive_class):
        '''
        Call the model's compute loss function

        Arg(s):
            image :  torch.Tensor[float32]
                N x 3 x H x W image
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth
            lidar_map : torch.Tensor[float32]
                N x 1 x H x W single lidar scan
            loss_func : str
                loss function to minimize
            w_smoothness : float
                weight of local smoothness loss
            loss_smoothness_kernel_size : tuple[int]
                kernel size of loss smoothness
            w_lidar_loss : float
                weight of lidar loss
            logits : torch.Tensor[float32]
                N x 1 x H x W logits
            validity_map : torch.Tensor[float32]
                N x 1 x H x W validity map
            w_positive_class : float
                weight of positive class
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''
        return self.model.compute_loss(image,
                                       output_depth_fusion,
                                       ground_truth,
                                       lidar_map,
                                       loss_func,
                                       w_smoothness,
                                       loss_smoothness_kernel_size,
                                       w_lidar_loss,
                                       logits,
                                       validity_map,
                                       w_positive_class)

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        return self.model.parameters()

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
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''
        self.device = device
        self.model.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''
        self.model.data_parallel()

    def restore_model(self, checkpoint_paths, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            checkpoint_paths : list[str]
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''
        return self.model.restore_model(checkpoint_paths, optimizer)

    def save_model(self, checkpoint_paths, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : list[str]
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''
        return self.model.save_model(checkpoint_paths, step, optimizer)

    def log_summary(self, **kwargs):
        '''
        Logs summary to Tensorboard using model-specific arguments.
        (See self.model_specific_log_config)

        Args:
            kwargs: Arguments passed to the log_summary_method
        '''
        
        model_specific_args = self.model_specific_log_config.get(
            self.model_name, 
            self.model_specific_log_config["default"] # this arg for unspecified models
        )

        required_args = self.common_log_args + model_specific_args

        # add default args if weren't provided
        for key, value in self.default_args.items():
            if key not in kwargs:
                kwargs[key] = value
        # only choose necessary arguments (to avoid dealing with them inside model)
        filtered_args = {key : kwargs[key] for key in required_args if key in kwargs}

        self.model.log_summary(**filtered_args)
