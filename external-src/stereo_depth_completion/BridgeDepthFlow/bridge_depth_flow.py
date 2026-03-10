"""
BridgeDepthFlow main model wrapper.

Provides a clean nn.Module interface around MonodepthNet with:
- forward(): runs stereo pair through the network, returns depth map
- compute_training_loss(): full self-supervised training loss from train.py

The training loss includes:
- L1 + SSIM photometric reconstruction loss (forward and backward)
- 2nd-order edge-aware disparity smoothness
- Left-right consistency loss

All CUDA custom ops (Resample2d) have been replaced with pure-PyTorch
F.grid_sample via resample2d(). Variable usage has been removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .monodepth_model import MonodepthNet
from .loss_utils import (
    resample2d,
    SSIM,
    cal_grad2_error,
    get_mask,
    create_border_mask,
    make_pyramid,
    warp_2,
)


class BridgeDepthFlowModel(nn.Module):
    """
    Wrapper around MonodepthNet for stereo depth estimation.

    Args:
        img_height: expected input image height (default 256)
        img_width: expected input image width (default 512)
        alpha_image_loss: weight between SSIM and L1 in photometric loss (default 0.85)
        disp_gradient_loss_weight: weight for 2nd-order smoothness loss (default 10.0)
        lr_loss_weight: weight for left-right consistency loss (default 0.5)
    """

    def __init__(
        self,
        img_height=256,
        img_width=512,
        alpha_image_loss=0.85,
        disp_gradient_loss_weight=10.0,
        lr_loss_weight=0.5,
    ):
        super(BridgeDepthFlowModel, self).__init__()
        self.net = MonodepthNet()
        self.img_height = img_height
        self.img_width = img_width
        self.alpha_image_loss = alpha_image_loss
        self.disp_gradient_loss_weight = disp_gradient_loss_weight
        self.lr_loss_weight = lr_loss_weight

    def forward(self, left_image, right_image):
        """
        Run stereo pair through MonodepthNet and return depth.

        Args:
            left_image: (N, 3, H, W) left image tensor, values in [0, 1]
            right_image: (N, 3, H, W) right image tensor, values in [0, 1]

        Returns:
            depth: (N, 1, H, W) estimated depth map (1 / horizontal disparity).
                   Uses the finest scale (scale 0) horizontal disparity.
            disp_est_scale: list of 4 pixel-scale disparity tensors (for inspection)
            disp_est: list of 4 normalized disparity tensors (for inspection)
        """
        model_input = torch.cat((left_image, right_image), dim=1)
        disp_est_scale, disp_est = self.net(model_input)

        # Extract horizontal disparity at finest scale
        # disp_est_scale[0] shape: (N, 2, H, W), channel 0 is horizontal
        horiz_disp = disp_est_scale[0][:, 0:1, :, :]  # (N, 1, H, W)

        # Convert disparity to depth (depth = 1 / |disparity|)
        # Add small epsilon to avoid division by zero
        depth = 1.0 / (torch.abs(horiz_disp) + 1e-6)

        return depth, disp_est_scale, disp_est

    def compute_training_loss(self, left_image, right_image):
        """
        Compute the full self-supervised training loss.

        Implements the training procedure from BridgeDepthFlow/train.py for a
        single stereo pair (without the temporal cycle consistency of the
        original 4-pair batching).

        Loss = image_loss + image_loss_2
               + disp_gradient_loss_weight * (smooth_loss + smooth_loss_2)
               + lr_loss_weight * lr_loss

        Args:
            left_image: (N, 3, H, W) left image, values in [0, 1]
            right_image: (N, 3, H, W) right image, values in [0, 1]

        Returns:
            loss: scalar total loss
            loss_info: dict with individual loss components for logging
        """
        # Forward pass: left->right disparity
        model_input = torch.cat((left_image, right_image), dim=1)
        disp_est_scale, disp_est = self.net(model_input)

        # Backward pass: right->left disparity
        model_input_2 = torch.cat((right_image, left_image), dim=1)
        disp_est_scale_2, disp_est_2 = self.net(model_input_2)

        # Build image pyramids
        left_pyramid = make_pyramid(left_image, 4)
        right_pyramid = make_pyramid(right_image, 4)

        # Border masks
        border_mask = [create_border_mask(left_pyramid[i], 0.1) for i in range(4)]

        # Occlusion masks via forward-backward consistency
        fw_mask = []
        bw_mask = []
        for i in range(4):
            fw, bw, diff_fw, diff_bw = get_mask(
                disp_est_scale[i], disp_est_scale_2[i], border_mask[i]
            )
            fw = fw + 1e-3
            bw = bw + 1e-3
            fw_mask.append(fw.clone().detach())
            bw_mask.append(bw.clone().detach())

        # ---- Reconstruction from right to left (forward direction) ----
        left_est = [resample2d(right_pyramid[i], disp_est_scale[i]) for i in range(4)]

        l1_left = [
            torch.abs(left_est[i] - left_pyramid[i]) * fw_mask[i] for i in range(4)
        ]
        l1_reconstruction_loss_left = [
            torch.mean(l1_left[i]) / torch.mean(fw_mask[i]) for i in range(4)
        ]

        ssim_left = [
            SSIM(left_est[i] * fw_mask[i], left_pyramid[i] * fw_mask[i])
            for i in range(4)
        ]
        ssim_loss_left = [
            torch.mean(ssim_left[i]) / torch.mean(fw_mask[i]) for i in range(4)
        ]

        image_loss_left = [
            self.alpha_image_loss * ssim_loss_left[i]
            + (1 - self.alpha_image_loss) * l1_reconstruction_loss_left[i]
            for i in range(4)
        ]
        image_loss = sum(image_loss_left)

        # Forward smoothness
        disp_loss = [
            cal_grad2_error(disp_est_scale[i] / 20, left_pyramid[i], 1.0)
            for i in range(4)
        ]
        disp_gradient_loss = sum(disp_loss)

        # ---- Reconstruction from left to right (backward direction) ----
        right_est = [resample2d(left_pyramid[i], disp_est_scale_2[i]) for i in range(4)]

        l1_right = [
            torch.abs(right_est[i] - right_pyramid[i]) * bw_mask[i] for i in range(4)
        ]
        l1_reconstruction_loss_right = [
            torch.mean(l1_right[i]) / torch.mean(bw_mask[i]) for i in range(4)
        ]

        ssim_right = [
            SSIM(right_est[i] * bw_mask[i], right_pyramid[i] * bw_mask[i])
            for i in range(4)
        ]
        ssim_loss_right = [
            torch.mean(ssim_right[i]) / torch.mean(bw_mask[i]) for i in range(4)
        ]

        image_loss_right = [
            self.alpha_image_loss * ssim_loss_right[i]
            + (1 - self.alpha_image_loss) * l1_reconstruction_loss_right[i]
            for i in range(4)
        ]
        image_loss_2 = sum(image_loss_right)

        # Backward smoothness
        disp_loss_2 = [
            cal_grad2_error(disp_est_scale_2[i] / 20, right_pyramid[i], 1.0)
            for i in range(4)
        ]
        disp_gradient_loss_2 = sum(disp_loss_2)

        # ---- Left-right consistency loss ----
        right_to_left_disp = [
            -resample2d(disp_est_2[i], disp_est_scale[i]) for i in range(4)
        ]
        left_to_right_disp = [
            -resample2d(disp_est[i], disp_est_scale_2[i]) for i in range(4)
        ]

        lr_left_loss = [
            torch.mean(torch.abs(right_to_left_disp[i] - disp_est[i]))
            for i in range(4)
        ]
        lr_right_loss = [
            torch.mean(torch.abs(left_to_right_disp[i] - disp_est_2[i]))
            for i in range(4)
        ]
        lr_loss = sum(lr_left_loss) + sum(lr_right_loss)

        # ---- Total loss ----
        smooth_loss = disp_gradient_loss + disp_gradient_loss_2
        loss = (
            image_loss
            + image_loss_2
            + self.disp_gradient_loss_weight * smooth_loss
            + self.lr_loss_weight * lr_loss
        )

        loss_info = {
            "total_loss": loss.item(),
            "image_loss": image_loss.item(),
            "image_loss_2": image_loss_2.item(),
            "smooth_loss": smooth_loss.item(),
            "lr_loss": lr_loss.item(),
        }

        return loss, loss_info
