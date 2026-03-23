import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root (augundo-ext) is on path so we can import UnOS by full package path.
# Do not use "from models import ..." here: stereo_depth_completion_model imports BDF first,
# which caches "models" as BDF's package; we need UnOS's Model_stereo and Model_depthflow.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from external_src.stereo_depth_completion.UnOS.models import Model_stereo, Model_depthflow
from external_src.stereo_depth_completion.UnOS.monodepth_model import disp_godard


class UnOSModel(object):
    '''
    Wrapper for UnOS under the AugUndo stereo interface.

    Arg(s):
        mode : str
            'stereo' or 'depthflow'
        img_height : int
            input image height
        img_width : int
            input image width
        depth_smooth_weight : float
            weight for depth smoothness loss
        ssim_weight : float
            weight for SSIM in pixel loss
        flow_smooth_weight : float
            weight for flow smoothness loss
        flow_consist_weight : float
            weight for flow consistency loss
        flow_diff_threshold : float
            threshold for comparing optical flow and rigid flow
        num_scales : int
            number of multi-scale levels
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 mode='depthflow',
                 img_height=256,
                 img_width=832,
                 depth_smooth_weight=10.0,
                 ssim_weight=0.85,
                 flow_smooth_weight=10.0,
                 flow_consist_weight=0.01,
                 flow_diff_threshold=4.0,
                 num_scales=4,
                 device=torch.device('cuda')):

        self.mode = mode
        self.device = device

        # Build an opt-like object that UnOS models expect
        class _Opt:
            pass

        self.opt = _Opt()
        self.opt.img_height = img_height
        self.opt.img_width = img_width
        self.opt.depth_smooth_weight = depth_smooth_weight
        self.opt.ssim_weight = ssim_weight
        self.opt.flow_smooth_weight = flow_smooth_weight
        self.opt.flow_consist_weight = flow_consist_weight
        self.opt.flow_diff_threshold = flow_diff_threshold
        self.opt.num_scales = num_scales

        if mode == 'depthflow':
            self.model = Model_depthflow(self.opt)
        elif mode == 'stereo':
            self.model = Model_stereo(self.opt)
        else:
            raise ValueError('Unsupported UnOS mode: {}. Use "stereo" or "depthflow".'.format(mode))

        self.model = self.model.to(device)

    def forward(self, image_left_t, image_right_t, image_left_t1, image_right_t1,
                cam2pix=None, pix2cam=None):
        '''
        Forward pass through UnOS network.

        Arg(s):
            image_left_t : torch.Tensor[float32]
                N x 3 x H x W left image at time t
            image_right_t : torch.Tensor[float32]
                N x 3 x H x W right image at time t
            image_left_t1 : torch.Tensor[float32]
                N x 3 x H x W left image at time t+1
            image_right_t1 : torch.Tensor[float32]
                N x 3 x H x W right image at time t+1
            cam2pix : torch.Tensor[float32]
                N x num_scales x 3 x 3 camera intrinsics
            pix2cam : torch.Tensor[float32]
                N x num_scales x 3 x 3 inverse camera intrinsics

        Returns:
            torch.Tensor[float32] : total loss
            dict : loss info dictionary
        '''

        loss, info = self.model(
            image_left_t, image_right_t,
            image_left_t1, image_right_t1,
            cam2pix, pix2cam)

        return loss, info

    def forward_disparity(self, left, right):
        '''
        Predict stereo disparity without computing loss.

        Calls the shared feature_pyramid_disp and pwc_disp networks in
        inference mode (is_training=False) to obtain disparity only.
        Gradients still flow through the shared weights for backpropagation.

        Arg(s):
            left : torch.Tensor[float32]
                N x 3 x H x W left image
            right : torch.Tensor[float32]
                N x 3 x H x W right image

        Returns:
            disp_left : list[torch.Tensor[float32]]
                4 tensors each (N, 1, H_s, W_s) positive normalised left disparity
            disp_right : list[torch.Tensor[float32]]
                4 tensors each (N, 1, H_s, W_s) positive normalised right disparity
        '''

        feat_left = self.model.feature_pyramid_disp(left)
        feat_right = self.model.feature_pyramid_disp(right)

        # is_training=False skips internal loss computation but keeps gradients
        disp_est = disp_godard(
            left, right, feat_left, feat_right, self.opt,
            is_training=False, pwc_disp_net=self.model.pwc_disp)

        # disp_est: list of 4 tensors (B, 2, H_s, W_s)
        # ch0 = left disparity (positive), ch1 = right disparity (positive)
        disp_left = [d[:, 0:1] for d in disp_est]
        disp_right = [d[:, 1:2] for d in disp_est]

        return disp_left, disp_right

    def compute_loss(self, output_loss, output_info):
        '''
        For UnOS, loss is computed inside the model forward pass.
        This method simply returns the pre-computed loss and info.

        Arg(s):
            output_loss : torch.Tensor[float32]
                loss from forward()
            output_info : dict
                loss info from forward()

        Returns:
            torch.Tensor[float32] : total loss
            dict[str, float] : loss info
        '''

        return output_loss, output_info

    def parameters(self):
        return list(self.model.parameters())

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)

    def data_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                path to model checkpoint
            optimizer : torch.optim
                optimizer to restore

        Returns:
            int : training step/iteration from checkpoint
            torch.optim : restored optimizer
        '''

        checkpoint = torch.load(restore_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            step = checkpoint.get('iteration', 0)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
            step = 0

        return step, optimizer

    def save_model(self, checkpoint_path, step, optimizer=None):
        '''
        Save model checkpoint.

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training iteration
            optimizer : torch.optim
                optimizer to save
        '''

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        state = {
            'iteration': step,
            'model_state_dict': raw_model.state_dict(),
        }
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(state, checkpoint_path)
