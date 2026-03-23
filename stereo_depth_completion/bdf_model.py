import importlib.util
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add BDF source to path for models.* imports (MonodepthModel, PWC_net, etc.)
_bdf_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'external_src', 'stereo_depth_completion', 'BDF')
if _bdf_root not in sys.path:
    sys.path.insert(0, _bdf_root)

from models.MonodepthModel import MonodepthNet
from models.PWC_net import pwc_dc_net
from models.networks.resample2d_package import Resample2d

# Import BDF's utils.utils via importlib to avoid name collision with the
# project-level utils/ directory (augundo-ext/utils/) which may also be on
# sys.path.
_bdf_utils_spec = importlib.util.spec_from_file_location(
    '_bdf_utils', os.path.join(_bdf_root, 'utils', 'utils.py'))
_bdf_utils = importlib.util.module_from_spec(_bdf_utils_spec)
_bdf_utils_spec.loader.exec_module(_bdf_utils)
SSIM = _bdf_utils.SSIM
cal_grad2_error = _bdf_utils.cal_grad2_error
make_pyramid = _bdf_utils.make_pyramid
get_mask = _bdf_utils.get_mask
create_border_mask = _bdf_utils.create_border_mask


class BDFModel(object):
    '''
    Wrapper for BridgeDepthFlow (BDF) under the AugUndo stereo interface.

    Arg(s):
        model_name : str
            'monodepth' or 'pwc'
        input_height : int
            input image height
        input_width : int
            input image width
        lr_loss_weight : float
            left-right consistency loss weight
        alpha_image_loss : float
            weight between SSIM and L1 in image loss
        disp_gradient_loss_weight : float
            disparity smoothness weight
        type_of_2warp : int
            two-warp loss type (0, 1, 2, or 3)
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 model_name='monodepth',
                 input_height=256,
                 input_width=512,
                 lr_loss_weight=0.5,
                 alpha_image_loss=0.85,
                 disp_gradient_loss_weight=0.1,
                 type_of_2warp=0,
                 device=torch.device('cuda')):

        self.model_name = model_name
        self.input_height = input_height
        self.input_width = input_width
        self.lr_loss_weight = lr_loss_weight
        self.alpha_image_loss = alpha_image_loss
        self.disp_gradient_loss_weight = disp_gradient_loss_weight
        self.type_of_2warp = type_of_2warp
        self.device = device

        if model_name == 'monodepth':
            self.net = MonodepthNet()
        elif model_name == 'pwc':
            self.net = pwc_dc_net()
            self.input_width = 832
        else:
            raise ValueError('Unsupported BDF model name: {}'.format(model_name))

        self.resample = Resample2d()
        self.net = self.net.to(device)

    def forward(self, image_left_t, image_right_t, image_left_t1, image_right_t1):
        '''
        Forward pass through BDF network.

        The BDF training concatenates 4 directional pairs into a batch of 8:
            former = [left_t1, left_t, right_t, left_t]
            latter = [right_t1, left_t1, right_t1, right_t]
        Then runs the model on cat(former, latter) along channel dim, and its reverse.

        For AugUndo, we run the full BDF forward and return multi-scale disparity
        predictions along with auxiliary outputs needed for loss computation.

        Arg(s):
            image_left_t : torch.Tensor[float32]
                N x 3 x H x W left image at time t
            image_right_t : torch.Tensor[float32]
                N x 3 x H x W right image at time t
            image_left_t1 : torch.Tensor[float32]
                N x 3 x H x W left image at time t+1
            image_right_t1 : torch.Tensor[float32]
                N x 3 x H x W right image at time t+1

        Returns:
            dict : dictionary containing:
                'disp_est' : list of disparity estimates (normalized) at 4 scales
                'disp_est_scale' : list of disparity estimates (pixel-scaled) at 4 scales
                'disp_est_2' : reverse direction disparity estimates
                'disp_est_scale_2' : reverse direction pixel-scaled disparity estimates
                'former' : the former image tensor (batch of 4*N)
                'latter' : the latter image tensor (batch of 4*N)
        '''

        # Build the 4 directional pairs as in BDF train.py
        former = torch.cat((image_left_t1, image_left_t, image_right_t, image_left_t), dim=0)
        latter = torch.cat((image_right_t1, image_left_t1, image_right_t1, image_right_t), dim=0)

        model_input = torch.cat((former, latter), dim=1)
        model_input_2 = torch.cat((latter, former), dim=1)

        if self.model_name == 'monodepth':
            disp_est_scale, disp_est = self.net(model_input)
            disp_est_scale_2, disp_est_2 = self.net(model_input_2)
        elif self.model_name == 'pwc':
            disp_est_scale = self.net(model_input)
            disp_est = [
                torch.cat((
                    disp_est_scale[i][:, 0, :, :].unsqueeze(1) / disp_est_scale[i].shape[3],
                    disp_est_scale[i][:, 1, :, :].unsqueeze(1) / disp_est_scale[i].shape[2]
                ), 1) for i in range(4)
            ]
            disp_est_scale_2 = self.net(model_input_2)
            disp_est_2 = [
                torch.cat((
                    disp_est_scale_2[i][:, 0, :, :].unsqueeze(1) / disp_est_scale_2[i].shape[3],
                    disp_est_scale_2[i][:, 1, :, :].unsqueeze(1) / disp_est_scale_2[i].shape[2]
                ), 1) for i in range(4)
            ]

        return {
            'disp_est': disp_est,
            'disp_est_scale': disp_est_scale,
            'disp_est_2': disp_est_2,
            'disp_est_scale_2': disp_est_scale_2,
            'former': former,
            'latter': latter,
        }

    def forward_stereo_disparity(self, left, right):
        '''
        Predict stereo disparity for a single left-right pair.

        Runs the network on (left, right) and its reverse (right, left)
        to obtain bidirectional disparity.  Unlike forward(), this does not
        build the 4-directional-pair batch.

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

        model_input = torch.cat((left, right), dim=1)    # (N, 6, H, W)
        model_input_2 = torch.cat((right, left), dim=1)   # reverse

        if self.model_name == 'monodepth':
            _disp_scale, disp_norm = self.net(model_input)
            _disp_scale_2, disp_norm_2 = self.net(model_input_2)
        elif self.model_name == 'pwc':
            disp_scale = self.net(model_input)
            disp_norm = [
                torch.cat((
                    disp_scale[i][:, 0, :, :].unsqueeze(1) / disp_scale[i].shape[3],
                    disp_scale[i][:, 1, :, :].unsqueeze(1) / disp_scale[i].shape[2]
                ), 1) for i in range(4)
            ]
            disp_scale_2 = self.net(model_input_2)
            disp_norm_2 = [
                torch.cat((
                    disp_scale_2[i][:, 0, :, :].unsqueeze(1) / disp_scale_2[i].shape[3],
                    disp_scale_2[i][:, 1, :, :].unsqueeze(1) / disp_scale_2[i].shape[2]
                ), 1) for i in range(4)
            ]

        # Forward direction: cat(left, right) -> Resample2d samples right at
        # grid_x + flow_u to reconstruct left.  For standard stereo the
        # horizontal component (ch0) is NEGATIVE (sample leftward in right).
        # Negate to get positive normalised left disparity.
        disp_left = [-disp_norm[s][:, 0:1] for s in range(4)]

        # Reverse direction: cat(right, left) -> flow to reconstruct right from
        # left.  Horizontal component is POSITIVE.
        disp_right = [disp_norm_2[s][:, 0:1] for s in range(4)]

        return disp_left, disp_right

    def forward_temporal_flow(self, image_t, image_t1):
        '''
        Predict temporal 2D flow from image_t to image_t1.

        Runs the network on cat(image_t, image_t1) to produce a flow field
        that, when used for backward warping, reconstructs image_t from
        image_t1.

        Arg(s):
            image_t : torch.Tensor[float32]
                N x 3 x H x W image at time t
            image_t1 : torch.Tensor[float32]
                N x 3 x H x W image at time t+1

        Returns:
            list[torch.Tensor[float32]] :
                4 tensors each (N, 2, H_s, W_s) normalized flow
                (ch0 = fraction of width, ch1 = fraction of height)
        '''

        model_input = torch.cat((image_t, image_t1), dim=1)

        if self.model_name == 'monodepth':
            _, flow_norm = self.net(model_input)
        elif self.model_name == 'pwc':
            flow_scale = self.net(model_input)
            flow_norm = [
                torch.cat((
                    flow_scale[i][:, 0:1] / flow_scale[i].shape[3],
                    flow_scale[i][:, 1:2] / flow_scale[i].shape[2]
                ), 1) for i in range(4)
            ]

        return flow_norm

    def compute_loss(self, output, batch):
        '''
        Computes the full BDF loss in the original (un-augmented) coordinate frame.

        The loss includes:
            L_rec : photometric reconstruction loss (SSIM + L1)
            L_sm  : disparity smoothness loss (image-edge-aware 2nd order)
            L_lr  : left-right consistency loss
            L_2warp : optional two-warp consistency loss

        Arg(s):
            output : dict
                Output from forward(), containing disparity estimates and input images
            batch : dict
                Original (un-augmented) batch with keys:
                    'image_left_t', 'image_right_t', 'image_left_t1', 'image_right_t1'

        Returns:
            torch.Tensor[float32] : total scalar loss
            dict[str, float] : dictionary of individual loss components
        '''

        disp_est = output['disp_est']
        disp_est_scale = output['disp_est_scale']
        disp_est_2 = output['disp_est_2']
        disp_est_scale_2 = output['disp_est_scale_2']
        former = output['former']
        latter = output['latter']

        left_pyramid = make_pyramid(former, 4)
        right_pyramid = make_pyramid(latter, 4)

        # Border mask for occlusion handling
        border_mask = [create_border_mask(left_pyramid[i], 0.1) for i in range(4)]

        # Forward-backward consistency masks
        fw_mask = []
        bw_mask = []
        for i in range(4):
            fw, bw, _, _ = get_mask(disp_est_scale[i], disp_est_scale_2[i], border_mask[i])
            fw = fw + 1e-3
            bw = bw + 1e-3
            n = batch['image_left_t'].shape[0]
            # Stereo pair indices get full mask (no occlusion masking)
            fw[[0, 1, 6, 7]] = fw[[0, 1, 6, 7]] * 0 + 1
            bw[[0, 1, 6, 7]] = bw[[0, 1, 6, 7]] * 0 + 1
            fw_mask.append(fw.clone().detach())
            bw_mask.append(bw.clone().detach())

        # Reconstruction loss: right-to-left
        left_est = [self.resample(right_pyramid[i], disp_est_scale[i]) for i in range(4)]
        l1_left = [torch.abs(left_est[i] - left_pyramid[i]) * fw_mask[i] for i in range(4)]
        l1_loss_left = [torch.mean(l1_left[i]) / torch.mean(fw_mask[i]) for i in range(4)]
        ssim_left = [SSIM(left_est[i] * fw_mask[i], left_pyramid[i] * fw_mask[i]) for i in range(4)]
        ssim_loss_left = [torch.mean(ssim_left[i]) / torch.mean(fw_mask[i]) for i in range(4)]
        image_loss_left = [
            self.alpha_image_loss * ssim_loss_left[i] +
            (1 - self.alpha_image_loss) * l1_loss_left[i] for i in range(4)
        ]
        image_loss = sum(image_loss_left)

        # Disparity smoothness loss
        disp_loss = [cal_grad2_error(disp_est_scale[i] / 20, left_pyramid[i], 1.0) for i in range(4)]
        disp_gradient_loss = sum(disp_loss)

        # Reconstruction loss: left-to-right
        right_est = [self.resample(left_pyramid[i], disp_est_scale_2[i]) for i in range(4)]
        l1_right = [torch.abs(right_est[i] - right_pyramid[i]) * bw_mask[i] for i in range(4)]
        l1_loss_right = [torch.mean(l1_right[i]) / torch.mean(bw_mask[i]) for i in range(4)]
        ssim_right = [SSIM(right_est[i] * bw_mask[i], right_pyramid[i] * bw_mask[i]) for i in range(4)]
        ssim_loss_right = [torch.mean(ssim_right[i]) / torch.mean(bw_mask[i]) for i in range(4)]
        image_loss_right = [
            self.alpha_image_loss * ssim_loss_right[i] +
            (1 - self.alpha_image_loss) * l1_loss_right[i] for i in range(4)
        ]
        image_loss_2 = sum(image_loss_right)

        # Smoothness loss (reverse direction)
        disp_loss_2 = [cal_grad2_error(disp_est_scale_2[i] / 20, right_pyramid[i], 1.0) for i in range(4)]
        disp_gradient_loss_2 = sum(disp_loss_2)

        # Left-right consistency loss
        right_to_left_disp = [-self.resample(disp_est_2[i], disp_est_scale[i]) for i in range(4)]
        left_to_right_disp = [-self.resample(disp_est[i], disp_est_scale_2[i]) for i in range(4)]

        lr_left_loss = [
            torch.mean(torch.abs(right_to_left_disp[i][[0, 1, 6, 7]] - disp_est[i][[0, 1, 6, 7]]))
            for i in range(4)
        ]
        lr_right_loss = [
            torch.mean(torch.abs(left_to_right_disp[i][[0, 1, 6, 7]] - disp_est_2[i][[0, 1, 6, 7]]))
            for i in range(4)
        ]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Total loss
        loss = (image_loss + image_loss_2 +
                10 * (disp_gradient_loss + disp_gradient_loss_2) +
                self.lr_loss_weight * lr_loss)

        loss_info = {
            'loss': loss.item(),
            'loss_image': (image_loss + image_loss_2).item(),
            'loss_sm': (disp_gradient_loss + disp_gradient_loss_2).item(),
            'loss_lr': lr_loss.item(),
        }

        # Optional 2-warp losses
        warp_2 = _bdf_utils.warp_2

        class _Args:
            alpha_image_loss = self.alpha_image_loss

        _args = _Args()

        if self.type_of_2warp == 1:
            mask_4 = [fw_mask[i][[2, 3]] for i in range(4)]
            warp2_est_4 = [self.resample(left_est[i][[0, 1]], disp_est_scale[i][[2, 3]]) for i in range(4)]
            loss_2warp = 0.1 * sum([warp_2(warp2_est_4[i], left_pyramid[i][[6, 7]], mask_4[i], _args) for i in range(4)])
            mask_5 = [bw_mask[i][[2, 3]] for i in range(4)]
            warp2_est_5 = [self.resample(left_est[i][[6, 7]], disp_est_scale_2[i][[2, 3]]) for i in range(4)]
            loss_2warp += 0.1 * sum([warp_2(warp2_est_5[i], left_pyramid[i][[0, 1]], mask_5[i], _args) for i in range(4)])
            loss = loss + loss_2warp
            loss_info['loss_2warp'] = loss_2warp.item()

        elif self.type_of_2warp == 2:
            mask = [self.resample(fw_mask[i][[2, 3]], disp_est_scale_2[i][[0, 1]]) for i in range(4)]
            warp2_est = [self.resample(left_est[i][[2, 3]], disp_est_scale_2[i][[6, 7]]) for i in range(4)]
            loss_2warp = 0.1 * sum([warp_2(warp2_est[i], right_pyramid[i][[6, 7]], mask[i], _args) for i in range(4)])
            mask_3 = [self.resample(fw_mask[i][[4, 5]], disp_est_scale[i][[0, 1]]) for i in range(4)]
            warp2_est_3 = [self.resample(left_est[i][[4, 5]], disp_est_scale[i][[6, 7]]) for i in range(4)]
            loss_2warp += 0.1 * sum([warp_2(warp2_est_3[i], left_pyramid[i][[6, 7]], mask_3[i], _args) for i in range(4)])
            loss = loss + loss_2warp
            loss_info['loss_2warp'] = loss_2warp.item()

        elif self.type_of_2warp == 3:
            mask = [self.resample(fw_mask[i][[2, 3]], disp_est_scale_2[i][[0, 1]]) for i in range(4)]
            warp2_est = [self.resample(left_est[i][[2, 3]], disp_est_scale_2[i][[6, 7]]) for i in range(4)]
            loss_2warp = 0.1 * sum([warp_2(warp2_est[i], right_pyramid[i][[6, 7]], mask[i], _args) for i in range(4)])
            mask_2 = [fw_mask[i][[4, 5]] for i in range(4)]
            warp2_est_2 = [self.resample(right_est[i][[0, 1]], disp_est_scale[i][[4, 5]]) for i in range(4)]
            loss_2warp += 0.1 * sum([warp_2(warp2_est_2[i], right_pyramid[i][[6, 7]], mask_2[i], _args) for i in range(4)])
            loss = loss + loss_2warp
            loss_info['loss_2warp'] = loss_2warp.item()

        loss_info['loss'] = loss.item()

        return loss, loss_info

    def parameters(self):
        return list(self.net.parameters())

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def to(self, device):
        self.device = device
        self.net = self.net.to(device)

    def data_parallel(self):
        self.net = torch.nn.DataParallel(self.net)

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                path to model checkpoint
            optimizer : torch.optim
                optimizer to restore

        Returns:
            int : training step from checkpoint
            torch.optim : restored optimizer
        '''

        checkpoint = torch.load(restore_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['state_dict'])
            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint.get('epoch', 0)
        else:
            self.net.load_state_dict(checkpoint)
            step = 0

        return step, optimizer

    def save_model(self, checkpoint_path, step, optimizer=None):
        '''
        Save model checkpoint.

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer to save
        '''

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        state = {
            'epoch': step,
            'state_dict': self.net.state_dict(),
        }
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        torch.save(state, checkpoint_path)
