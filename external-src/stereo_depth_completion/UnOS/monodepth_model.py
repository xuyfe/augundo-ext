"""
Monodepth model with stereo reconstruction losses for UnOS.

Ported from the original TensorFlow implementation.
Combines PWC-Net disparity estimation with photometric reconstruction loss
(SSIM + L1), edge-aware smoothness loss, and left-right consistency loss.
Uses forward warping for occlusion mask computation.

All tensors use PyTorch NCHW format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import SSIM, cal_grad2_error, charbonnier_loss
from .warping import backward_warp, forward_warp
from .pwc_disp import PWCDisp


def disp_godard(left, right, disp_left, disp_right, alpha_image_loss=0.85,
                disp_gradient_loss_weight=0.1, lr_loss_weight=1.0):
    """Compute stereo disparity losses for a single scale.

    Computes:
    1. Image reconstruction loss (alpha * SSIM + (1-alpha) * L1)
    2. 2nd-order edge-aware disparity smoothness loss
    3. Left-right consistency loss
    4. Occlusion-aware masking via forward warping

    Args:
        left: Left image at current scale, N x 3 x H x W.
        right: Right image at current scale, N x 3 x H x W.
        disp_left: Left disparity (normalized), N x 1 x H x W.
        disp_right: Right disparity (normalized), N x 1 x H x W.
        alpha_image_loss: Weight for SSIM vs L1 in image loss.
        disp_gradient_loss_weight: Weight for smoothness loss.
        lr_loss_weight: Weight for left-right consistency loss.

    Returns:
        total_loss: Combined scalar loss.
        image_loss: Image reconstruction loss.
        disp_gradient_loss: Smoothness loss.
        lr_loss: Left-right consistency loss.
    """
    _, _, H, W = left.shape

    # Convert normalized disparity to pixel displacement
    # disp_left is positive, disp_right is negative (already signed)
    disp_left_pixel = disp_left * W
    disp_right_pixel = disp_right * W

    # Create flow fields for warping (disparity is horizontal only)
    # Left-to-right: negative displacement (right image warped to left view)
    flow_left = torch.cat([-disp_left_pixel, torch.zeros_like(disp_left_pixel)], dim=1)
    # Right-to-left: positive displacement
    flow_right = torch.cat([disp_right_pixel, torch.zeros_like(disp_right_pixel)], dim=1)

    # Reconstruct images
    left_reconstructed = backward_warp(right, flow_left)
    right_reconstructed = backward_warp(left, flow_right)

    # Occlusion masks via forward warping
    ones = torch.ones_like(disp_left)
    left_occu = forward_warp(ones, flow_left)
    right_occu = forward_warp(ones, flow_right)

    # Threshold occlusion maps: occluded if splatted count < 1
    left_occu_mask = (left_occu >= 1.0).float()
    right_occu_mask = (right_occu >= 1.0).float()

    # --- Image reconstruction loss ---
    # SSIM (output is smaller due to VALID padding in avg_pool)
    ssim_left = SSIM(left_reconstructed, left)
    ssim_right = SSIM(right_reconstructed, right)

    # L1 loss (trim to match SSIM size)
    l1_left = torch.abs(left_reconstructed - left)
    l1_right = torch.abs(right_reconstructed - right)

    # Trim L1 and occlusion mask to match SSIM spatial size (1 pixel border removed)
    l1_left_trimmed = l1_left[:, :, 1:-1, 1:-1]
    l1_right_trimmed = l1_right[:, :, 1:-1, 1:-1]
    left_occu_trimmed = left_occu_mask[:, :, 1:-1, 1:-1]
    right_occu_trimmed = right_occu_mask[:, :, 1:-1, 1:-1]

    # Combine SSIM and L1 with occlusion masking
    image_loss_left = (alpha_image_loss * ssim_left + (1 - alpha_image_loss) * l1_left_trimmed)
    image_loss_right = (alpha_image_loss * ssim_right + (1 - alpha_image_loss) * l1_right_trimmed)

    # Apply occlusion masks
    image_loss_left = image_loss_left * left_occu_trimmed
    image_loss_right = image_loss_right * right_occu_trimmed

    # Normalize by mask area
    image_loss = (
        torch.sum(image_loss_left) / (torch.sum(left_occu_trimmed) + 1e-8) +
        torch.sum(image_loss_right) / (torch.sum(right_occu_trimmed) + 1e-8)
    ) / 2.0

    # --- Disparity smoothness loss (2nd order, edge-aware) ---
    disp_gradient_loss = (
        cal_grad2_error(disp_left_pixel, left, beta=10.0) +
        cal_grad2_error(disp_right_pixel, right, beta=10.0)
    ) / 2.0

    # --- Left-right consistency loss ---
    # Warp right disparity to left view and compare
    disp_right_warped = backward_warp(disp_right_pixel, flow_left)
    disp_left_warped = backward_warp(disp_left_pixel, flow_right)

    lr_loss_left = torch.abs(disp_left_pixel - disp_right_warped) * left_occu_mask
    lr_loss_right = torch.abs(disp_right_pixel + disp_left_warped) * right_occu_mask

    lr_loss = (
        torch.sum(lr_loss_left) / (torch.sum(left_occu_mask) + 1e-8) +
        torch.sum(lr_loss_right) / (torch.sum(right_occu_mask) + 1e-8)
    ) / 2.0

    # --- Total loss ---
    total_loss = (
        image_loss +
        disp_gradient_loss_weight * disp_gradient_loss +
        lr_loss_weight * lr_loss
    )

    return total_loss, image_loss, disp_gradient_loss, lr_loss


class MonodepthModel(nn.Module):
    """Stereo disparity model with photometric reconstruction losses.

    Combines PWC-Net based disparity estimation with self-supervised
    training losses from stereo image pairs.

    The model produces left and right disparities at 4 pyramid scales
    and computes:
    - Photometric reconstruction loss (SSIM + L1)
    - 2nd-order edge-aware disparity smoothness
    - Left-right consistency loss
    - Forward-warping based occlusion handling
    """

    def __init__(self, params):
        """
        Args:
            params: Configuration object/dict with fields:
                - alpha_image_loss: SSIM vs L1 balance (default 0.85)
                - disp_gradient_loss_weight: Smoothness loss weight
                - lr_loss_weight: Left-right consistency loss weight
                - height: Input image height
                - width: Input image width
                - batch_size: Training batch size
        """
        super().__init__()

        if isinstance(params, dict):
            self.alpha_image_loss = params.get('alpha_image_loss', 0.85)
            self.disp_gradient_loss_weight = params.get('disp_gradient_loss_weight', 0.1)
            self.lr_loss_weight = params.get('lr_loss_weight', 1.0)
            self.height = params.get('height', 256)
            self.width = params.get('width', 512)
            self.batch_size = params.get('batch_size', 4)
        else:
            self.alpha_image_loss = getattr(params, 'alpha_image_loss', 0.85)
            self.disp_gradient_loss_weight = getattr(params, 'disp_gradient_loss_weight', 0.1)
            self.lr_loss_weight = getattr(params, 'lr_loss_weight', 1.0)
            self.height = getattr(params, 'height', 256)
            self.width = getattr(params, 'width', 512)
            self.batch_size = getattr(params, 'batch_size', 4)

        self.pwc_disp = PWCDisp()

    def forward(self, left, right, left_feature, right_feature):
        """Estimate disparities from stereo features.

        Args:
            left: Left image, N x 3 x H x W.
            right: Right image, N x 3 x H x W.
            left_feature: Left feature pyramid (tuple of 6 feature maps).
            right_feature: Right feature pyramid (tuple of 6 feature maps).

        Returns:
            List of 4 disparity tensors at scales [1/4, 1/8, 1/16, 1/32].
            Each tensor is N x 2 x H_s x W_s (channel 0: left disp,
            channel 1: right disp, both with eps added).
        """
        disps = self.pwc_disp(left_feature, right_feature, left, right)
        return disps

    def compute_loss(self, left, right, left_feature, right_feature):
        """Estimate disparities and compute training loss.

        Args:
            left: Left image, N x 3 x H x W.
            right: Right image, N x 3 x H x W.
            left_feature: Left feature pyramid (tuple of 6 feature maps).
            right_feature: Right feature pyramid (tuple of 6 feature maps).

        Returns:
            disps: List of 4 disparity tensors.
            total_loss: Scalar training loss.
        """
        disps = self.forward(left, right, left_feature, right_feature)

        total_loss = 0.0
        num_scales = len(disps)

        for i, disp in enumerate(disps):
            # Split left and right disparities
            disp_left = disp[:, 0:1, :, :]   # N x 1 x H_s x W_s
            disp_right = disp[:, 1:2, :, :]  # N x 1 x H_s x W_s

            # Resize images to match disparity scale
            scale_h, scale_w = disp.shape[2], disp.shape[3]
            left_scaled = F.interpolate(left, size=(scale_h, scale_w),
                                        mode='bilinear', align_corners=True)
            right_scaled = F.interpolate(right, size=(scale_h, scale_w),
                                         mode='bilinear', align_corners=True)

            # Compute loss at this scale
            scale_loss, _, _, _ = disp_godard(
                left_scaled, right_scaled, disp_left, disp_right,
                alpha_image_loss=self.alpha_image_loss,
                disp_gradient_loss_weight=self.disp_gradient_loss_weight,
                lr_loss_weight=self.lr_loss_weight
            )

            # Weight by scale (finer scales get more weight)
            scale_weight = 1.0 / (2 ** i)
            total_loss = total_loss + scale_weight * scale_loss

        return disps, total_loss
