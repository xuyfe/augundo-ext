import os, sys
import torch
import torch.nn.functional as F
# Path to BridgeDepthFlow package: augundo-ext/external-src/stereo_depth_completion/BridgeDepthFlow (script is in augundo-ext/depth_completion/src)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))
_bdf_root = os.path.join(_repo_root, 'external-src', 'stereo_depth_completion')
if _bdf_root not in sys.path:
    sys.path.insert(0, _bdf_root)
sys.path.insert(0, os.path.join('external-src', 'stereo_depth_completion'))
from BridgeDepthFlow.bridge_depth_flow import BridgeDepthFlowModel


class BridgeDepthFlowModelWrapper(object):
    '''
    Wrapper for BridgeDepthFlow stereo depth completion model.

    STEREO-SPECIFIC DIFFERENCES FROM MONOCULAR TEMPLATE:
    - forward_depth takes additional right_image parameter
    - Produces depth from stereo disparity (1/disparity)
    - Loss computation uses stereo photometric + LR consistency + smoothness
    - No pose network (pure stereo model)

    Arg(s):
        dataset_name : str
            model for a given dataset
        network_modules : list[str]
            network modules to build for model
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

        if dataset_name == 'kitti' or dataset_name == 'vkitti':
            img_height = 256
            img_width = 832
        else:
            img_height = 256
            img_width = 512

        self.model_depth = BridgeDepthFlowModel(
            img_height=img_height,
            img_width=img_width)

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
            'BridgeDepthFlow is a stereo model and requires right_image'

        depth, disp_est_scale, disp_est = self.model_depth(image, right_image)

        # Clamp depth to valid range
        output_depth = torch.clamp(
            depth,
            min=self.min_predict_depth,
            max=self.max_predict_depth)

        if return_all_outputs:
            return [output_depth, disp_est_scale, disp_est]
        else:
            return output_depth

    def forward_pose(self, image0, image1):
        '''
        Forwards a pair of images through the network to output pose from time 0 to 1

        BridgeDepthFlow is a pure stereo model and does not have a pose network.

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W tensor
            image1 : torch.Tensor[float32]
                N x 3 x H x W tensor
        Returns:
            torch.Tensor[float32] : N x 6 pose vector
        '''

        raise ValueError(
            'BridgeDepthFlow is a pure stereo model and does not have a pose network.')

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
            'BridgeDepthFlow is a stereo model and requires right_image0 for loss computation'

        loss, loss_info = self.model_depth.compute_training_loss(
            image0, right_image0)

        return loss, loss_info

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

        return list(self.model_depth.parameters())

    def parameters_pose(self):
        '''
        Fetches model parameters for pose network modules

        Returns:
            list[torch.Tensor[float32]] : list of model parameters for pose network modules
        '''

        raise ValueError(
            'BridgeDepthFlow is a pure stereo model and does not have a pose network.')

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
                unused (no pose network)
            optimizer_depth : torch.optim
                optimizer for depth network
            optimizer_pose : torch.optim
                unused (no pose network)
        Returns:
            int : training step
            torch.optim : optimizer for depth
            torch.optim : optimizer for pose (always None)
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
                unused (no pose network)
            optimizer_pose : torch.optim
                unused (no pose network)
        '''

        checkpoint = {
            'model_state_dict': self.model_depth.state_dict(),
            'train_step': step,
        }

        if optimizer_depth is not None:
            checkpoint['optimizer_state_dict'] = optimizer_depth.state_dict()

        torch.save(checkpoint, model_depth_checkpoint_path)
