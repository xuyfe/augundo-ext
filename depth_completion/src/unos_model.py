import os, sys
import torch
import torch.nn.functional as F
# Path to UnOS package: augundo-ext/external-src/stereo_depth_completion/UnOS (script is in augundo-ext/depth_completion/src)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
_unos_root = os.path.join(_repo_root, 'external-src', 'stereo_depth_completion')
if _unos_root not in sys.path:
    sys.path.insert(0, _unos_root)
from UnOS.unos_stereo import UnOSStereo, UnOSDepth, UnOSDepthFlow
from UnOS.pose_net import PoseNet


class UnOSModel(object):
    '''
    Wrapper for UnOS stereo depth completion model.

    STEREO-SPECIFIC DIFFERENCES FROM MONOCULAR TEMPLATE:
    - forward_depth takes additional right_image parameter
    - Produces depth from stereo disparity (1/disparity)
    - Loss computation uses stereo photometric + LR consistency + smoothness
    - No separate pose network needed for stereo-only mode
    - For depth/depthflow modes, pose net operates on temporal pairs

    Arg(s):
        dataset_name : str
            model for a given dataset
        network_modules : list[str]
            network modules to build for model.
            Can contain: 'stereo', 'depth', 'depthflow'
            'stereo' = pure stereo (UnOSStereo)
            'depth' = stereo + temporal (UnOSDepth)
            'depthflow' = full model (UnOSDepthFlow)
        min_predict_depth : float
            minimum value of predicted depth
        max_predict_depth : float
            maximum value of predicted depth
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 dataset_name='kitti',
                 network_modules=['stereo'],
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        self.network_modules = network_modules
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        # Default config matching UnOS original
        if dataset_name == 'kitti' or dataset_name == 'vkitti':
            img_height = 256
            img_width = 832
        else:
            img_height = 256
            img_width = 512

        params = {
            'height': img_height,
            'width': img_width,
            'batch_size': 4,
            'alpha_image_loss': 0.85,
            'disp_gradient_loss_weight': 0.1,
            'lr_loss_weight': 1.0,
            'photo_loss_weight': 1.0,
            'smooth_loss_weight': 0.1,
            'flow_smooth_weight': 0.1,
            'flow_consist_weight': 0.01,
        }

        # Select model variant based on network_modules
        if 'depthflow' in network_modules:
            self.model_depth = UnOSDepthFlow(params)
            self.mode = 'depthflow'
        elif 'depth' in network_modules:
            self.model_depth = UnOSDepth(params)
            self.mode = 'depth'
        else:
            self.model_depth = UnOSStereo(params)
            self.mode = 'stereo'

        # Pose network is built into UnOSDepth and UnOSDepthFlow
        # For stereo-only mode, there is no pose network
        if self.mode in ('depth', 'depthflow'):
            self.model_pose = self.model_depth.pose_net
        else:
            self.model_pose = None

        # Move to device
        self.device = device
        self.to(self.device)
        self.eval()

    def transform_inputs(self, image, sparse_depth):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
        Returns:
            torch.Tensor[float32] : N x 3 x H x W image
            torch.Tensor[float32] : N x 1 x H x W sparse depth map
        '''

        # UnOS does not require special preprocessing
        return image, sparse_depth

    def forward_depth(self, image, sparse_depth, validity_map, intrinsics,
                      right_image=None, return_all_outputs=False):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W left image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            validity_map : torch.Tensor[float32]
                N x 1 x H x W valid locations of projected sparse point cloud
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
            right_image : torch.Tensor[float32]
                N x 3 x H x W right image (required for stereo)
            return_all_outputs : bool
                if set, return all outputs
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''

        assert right_image is not None, \
            'UnOS is a stereo model and requires right_image'

        # Run stereo disparity estimation
        if self.mode == 'stereo':
            disps = self.model_depth(image, right_image)
        elif self.mode == 'depth':
            result = self.model_depth(image, right_image)
            disps = result['disps']
        elif self.mode == 'depthflow':
            result = self.model_depth(image, right_image)
            disps = result['disps']

        # Convert finest scale disparity to depth
        # disps[0] is N x 2 x H/4 x W/4 (left and right disparity)
        disp_left = disps[0][:, 0:1, :, :]  # N x 1 x H/4 x W/4

        # Convert disparity to depth: depth = focal_x / (disp * width_at_scale)
        _, _, dh, dw = disp_left.shape
        fx = intrinsics[:, 0, 0].reshape(-1, 1, 1, 1)
        depth = fx / (disp_left.clamp(min=1e-6) * dw)

        # Upsample depth to original resolution
        _, _, orig_h, orig_w = image.shape
        output_depth = F.interpolate(
            depth, size=(orig_h, orig_w),
            mode='bilinear', align_corners=True)

        # Clamp depth to valid range
        output_depth = torch.clamp(
            output_depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)

        if return_all_outputs:
            return [output_depth, disps]
        else:
            return output_depth

    def forward_pose(self, image0, image1):
        '''
        Forwards a pair of images through the network to output pose from time 0 to 1

        Only available for 'depth' and 'depthflow' modes.

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W tensor
            image1 : torch.Tensor[float32]
                N x 3 x H x W tensor
        Returns:
            torch.Tensor[float32] : N x 6 pose vector
        '''

        assert self.model_pose is not None, \
            'Pose network only available in depth or depthflow mode'

        return self.model_pose(image0, image1)

    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     sparse_depth0,
                     validity_map0,
                     intrinsics,
                     pose0to1,
                     pose0to2,
                     w_losses,
                     right_image0=None,
                     right_image1=None,
                     right_image2=None):
        '''
        Computes loss function

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W left image at time step t
            image1 : torch.Tensor[float32]
                N x 3 x H x W left image at time step t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W left image at time step t+1
            output_depth0 : list[torch.Tensor[float32]]
                list of N x 1 x H x W output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth at time t
            validity_map0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth at time t
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t-1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t+1
            w_losses : dict[str, float]
                dictionary of weights for each loss
            right_image0 : torch.Tensor[float32]
                N x 3 x H x W right image at time step t
            right_image1 : torch.Tensor[float32]
                N x 3 x H x W right image at time step t-1
            right_image2 : torch.Tensor[float32]
                N x 3 x H x W right image at time step t+1
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        assert right_image0 is not None, \
            'UnOS is a stereo model and requires right_image0 for loss computation'

        if self.mode == 'stereo':
            # Pure stereo loss: photometric + smoothness + LR consistency
            disps, total_loss = self.model_depth.compute_loss(image0, right_image0)

            loss_info = {
                'loss': total_loss.item(),
                'stereo_loss': total_loss.item(),
            }

            return total_loss, loss_info

        elif self.mode == 'depth':
            # Stereo + temporal loss
            # image2 serves as the temporal next frame
            left_next = image2 if image2 is not None else image1
            result = self.model_depth.compute_loss(
                image0, right_image0, left_next, intrinsics)

            loss_info = {
                'loss': result['total_loss'].item(),
                'stereo_loss': result['stereo_loss'].item(),
                'temporal_loss': result['temporal_loss'].item(),
            }

            return result['total_loss'], loss_info

        elif self.mode == 'depthflow':
            # Full model: stereo + flow + pose + consistency
            left_next = image2 if image2 is not None else image1
            result = self.model_depth.compute_loss(
                image0, right_image0, left_next, intrinsics)

            loss_info = {
                'loss': result['total_loss'].item(),
                'stereo_loss': result['stereo_loss'].item(),
                'flow_photo_loss': result['flow_photo_loss'].item(),
                'flow_smooth_loss': result['flow_smooth_loss'].item(),
                'consist_loss': result['consist_loss'].item(),
            }

            return result['total_loss'], loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return list(self.model_depth.parameters())

    def parameters_depth(self):
        '''
        Returns the list of parameters for depth network modules

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        if self.mode == 'stereo':
            return list(self.model_depth.parameters())
        else:
            # Return parameters excluding pose net
            depth_params = []
            for name, param in self.model_depth.named_parameters():
                if 'pose_net' not in name:
                    depth_params.append(param)
            return depth_params

    def parameters_pose(self):
        '''
        Fetches model parameters for pose network modules

        Returns:
            list[torch.Tensor[float32]] : list of model parameters for pose network modules
        '''

        if self.model_pose is not None:
            return list(self.model_pose.parameters())
        else:
            raise ValueError(
                'Pose network not available in stereo-only mode. '
                'Use depth or depthflow network_modules.')

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model_depth.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model_depth.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.model_depth.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model_depth = torch.nn.DataParallel(self.model_depth)

    def restore_model(self,
                      model_depth_restore_path,
                      model_pose_restore_path=None,
                      optimizer_depth=None,
                      optimizer_pose=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            model_depth_restore_path : str
                path to model weights for depth network
            model_pose_restore_path : str
                path to model weights for pose network (unused, pose is part of model)
            optimizer_depth : torch.optim
                optimizer for depth network
            optimizer_pose : torch.optim
                optimizer for pose network
        Returns:
            int : training step
            torch.optim : optimizer for depth
            torch.optim : optimizer for pose
        '''

        checkpoint = torch.load(
            model_depth_restore_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model_depth.load_state_dict(checkpoint['model_state_dict'])
            train_step = checkpoint.get('train_step', 0)
            if optimizer_depth is not None and 'optimizer_state_dict' in checkpoint:
                optimizer_depth.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # Assume checkpoint is just state_dict
            self.model_depth.load_state_dict(checkpoint)
            train_step = 0

        return train_step, optimizer_depth, optimizer_pose

    def save_model(self,
                   model_depth_checkpoint_path,
                   step,
                   optimizer_depth,
                   model_pose_checkpoint_path=None,
                   optimizer_pose=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            model_depth_checkpoint_path : str
                path to save checkpoint for depth network
            step : int
                current training step
            optimizer_depth : torch.optim
                optimizer for depth network
            model_pose_checkpoint_path : str
                path to save checkpoint for pose network (unused, pose is part of model)
            optimizer_pose : torch.optim
                optimizer for pose network
        '''

        checkpoint = {
            'model_state_dict': self.model_depth.state_dict(),
            'train_step': step,
        }

        if optimizer_depth is not None:
            checkpoint['optimizer_state_dict'] = optimizer_depth.state_dict()

        torch.save(checkpoint, model_depth_checkpoint_path)
