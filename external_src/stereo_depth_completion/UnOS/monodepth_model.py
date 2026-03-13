"""
Stereo disparity estimation model -- PyTorch reimplementation of UnOS / UnDepthflow.
Mirrors the TensorFlow original in monodepth_model.py.

Adopted from https://github.com/mrharicot/monodepth
"""
from __future__ import absolute_import, division, print_function
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets.pwc_disp import PWCDisp
from .optical_flow_warp_old import transformer_old
from .optical_flow_warp_fwd import transformerFwd


# ---------------------------------------------------------------------------
# Parameter container (mirrors the TF namedtuple)
# ---------------------------------------------------------------------------

monodepth_parameters = namedtuple(
    'parameters',
    'encoder, do_stereo, wrap_mode, use_deconv, alpha_image_loss, '
    'disp_gradient_loss_weight, lr_loss_weight, full_summary, '
    'height, width, batch_size')


# ---------------------------------------------------------------------------
# MonodepthModel
# ---------------------------------------------------------------------------

class MonodepthModel(nn.Module):
    """Stereo disparity model wrapping PWCDisp with multi-scale loss computation.

    Input images are expected in NCHW format with values in [0, 1].
    Disparity outputs have 2 channels: channel 0 = left, channel 1 = right.
    """

    def __init__(self, params, mode, pwc_disp_net=None):
        super().__init__()
        self.params = params
        self.mode = mode

        # The actual network -- shared externally or created here
        if pwc_disp_net is not None:
            self.pwc_disp = pwc_disp_net
        else:
            self.pwc_disp = PWCDisp()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def gradient_x(img):
        """img: (N, C, H, W)"""
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    @staticmethod
    def gradient_y(img):
        """img: (N, C, H, W)"""
        return img[:, :-1, :, :] - img[:, 1:, :, :]

    @staticmethod
    def scale_pyramid(img, num_scales):
        """Create a multi-scale image pyramid using area interpolation.

        Args:
            img: (N, C, H, W) tensor.
            num_scales: number of scales (the first is the original).
        Returns:
            List of tensors at decreasing spatial resolution.
        """
        scaled_imgs = [img]
        _, _, H, W = img.shape
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = H // ratio
            nw = W // ratio
            scaled_imgs.append(
                F.interpolate(img, size=(nh, nw), mode='area'))
        return scaled_imgs

    def generate_flow_left(self, disp, scale):
        """Convert left disparity to a 2-channel flow (NHWC) for warping.

        disp: (B, 1, H_s, W_s)  -- normalised disparity
        Returns: (B, H_s, W_s, 2)
        """
        B, _, H, W = disp.shape
        ltr_flow_x = -disp * W  # (B, 1, H, W)
        ltr_flow_y = torch.zeros_like(ltr_flow_x)
        # Stack to (B, 2, H, W) then permute to NHWC
        ltr_flow = torch.cat([ltr_flow_x, ltr_flow_y], dim=1)
        return ltr_flow.permute(0, 2, 3, 1)  # (B, H, W, 2)

    def generate_flow_right(self, disp, scale):
        return self.generate_flow_left(-disp, scale)

    def generate_transformed(self, img, flow, scale):
        """Warp *img* (NCHW) by *flow* (NHWC) using transformer_old.

        transformer_old expects NHWC inputs and returns NHWC.
        """
        H = self.params.height // (2 ** scale)
        W = self.params.width // (2 ** scale)
        img_nhwc = img.permute(0, 2, 3, 1)
        out_nhwc = transformer_old(img_nhwc, flow, [H, W])
        return out_nhwc.permute(0, 3, 1, 2)  # back to NCHW

    @staticmethod
    def SSIM(x, y):
        """Structural similarity loss (NCHW).

        Uses 'VALID' style pooling (no padding) to match the TF original.
        Returns a loss map clipped to [0, 1].
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 'VALID' pooling: kernel=3, stride=1, no padding
        mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
        mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=0)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM_val = SSIM_n / SSIM_d
        return torch.clamp((1 - SSIM_val) / 2, 0, 1)

    def get_disparity_smoothness_2nd(self, disp, pyramid):
        """Edge-aware second-order disparity smoothness.

        Args:
            disp: list of 4 disparity tensors (NCHW).
            pyramid: list of 4 image tensors (NCHW).
        Returns:
            List of 8 tensors (4 x-smoothness + 4 y-smoothness).
        """
        disp_gx = [self.gradient_x(d) for d in disp]
        disp_gy = [self.gradient_y(d) for d in disp]

        disp_gxx = [self.gradient_x(dg) for dg in disp_gx]
        disp_gyy = [self.gradient_y(dg) for dg in disp_gy]

        img_gx = [self.gradient_x(img) for img in pyramid]
        img_gy = [self.gradient_y(img) for img in pyramid]

        weights_x = [
            torch.exp(-torch.mean(10.0 * torch.abs(g), dim=1, keepdim=True))
            for g in img_gx
        ]
        weights_y = [
            torch.exp(-torch.mean(10.0 * torch.abs(g), dim=1, keepdim=True))
            for g in img_gy
        ]

        smoothness_x = [
            disp_gxx[i] * weights_x[i][:, :, :, :-1] for i in range(4)
        ]
        smoothness_y = [
            disp_gyy[i] * weights_y[i][:, :-1, :, :] for i in range(4)
        ]
        return smoothness_x + smoothness_y

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def build_outputs(self, left, right, left_feature=None, right_feature=None):
        """Run the PWCDisp network and compute warped images / LR consistency.

        Args:
            left, right: (B, 3, H, W) stereo images.
            left_feature, right_feature: optional precomputed feature tuples.
        Returns:
            disp_est: list of 4 disparity tensors [(B,2,H_s,W_s), ...].
        """
        H, W = self.params.height, self.params.width

        # Run network
        if left_feature is not None and right_feature is not None:
            disp_est = self.pwc_disp.forward_with_features(
                left, right, left_feature, right_feature)
        else:
            disp_est = self.pwc_disp(left, right)

        # Split into left / right
        disp_left_est = [d[:, 0:1, :, :] for d in disp_est]
        disp_right_est = [d[:, 1:2, :, :] for d in disp_est]

        if self.mode == 'test':
            return disp_est, disp_left_est, disp_right_est, {}, {}

        # Image pyramids
        left_pyramid = self.scale_pyramid(left, 4)
        right_pyramid = self.scale_pyramid(right, 4)

        # Flows
        ltr_flow = [self.generate_flow_left(disp_left_est[i], i) for i in range(4)]
        rtl_flow = [self.generate_flow_right(disp_right_est[i], i) for i in range(4)]

        # Occlusion masks via forward warping
        B = left.shape[0]
        right_occ_mask = []
        left_occ_mask = []
        for i in range(4):
            Hs = H // (2 ** i)
            Ws = W // (2 ** i)
            ones_nhwc = torch.ones(B, Hs, Ws, 1, device=left.device)
            right_occ = torch.clamp(
                transformerFwd(ones_nhwc, ltr_flow[i], [Hs, Ws]),
                min=0.0, max=1.0)
            left_occ = torch.clamp(
                transformerFwd(ones_nhwc, rtl_flow[i], [Hs, Ws]),
                min=0.0, max=1.0)
            # Convert to NCHW
            right_occ_mask.append(right_occ.permute(0, 3, 1, 2))
            left_occ_mask.append(left_occ.permute(0, 3, 1, 2))

        right_occ_mask_avg = [torch.mean(m) + 1e-12 for m in right_occ_mask]
        left_occ_mask_avg = [torch.mean(m) + 1e-12 for m in left_occ_mask]

        # Warped images (NCHW throughout)
        left_est = [
            self.generate_transformed(right_pyramid[i], ltr_flow[i], i)
            for i in range(4)
        ]
        right_est = [
            self.generate_transformed(left_pyramid[i], rtl_flow[i], i)
            for i in range(4)
        ]

        # LR consistency
        right_to_left_disp = [
            self.generate_transformed(disp_right_est[i], ltr_flow[i], i)
            for i in range(4)
        ]
        left_to_right_disp = [
            self.generate_transformed(disp_left_est[i], rtl_flow[i], i)
            for i in range(4)
        ]

        # Smoothness
        disp_left_smoothness = self.get_disparity_smoothness_2nd(
            disp_left_est, left_pyramid)
        disp_right_smoothness = self.get_disparity_smoothness_2nd(
            disp_right_est, right_pyramid)

        outputs = {
            'left_pyramid': left_pyramid,
            'right_pyramid': right_pyramid,
            'disp_left_est': disp_left_est,
            'disp_right_est': disp_right_est,
            'left_est': left_est,
            'right_est': right_est,
            'left_occ_mask': left_occ_mask,
            'right_occ_mask': right_occ_mask,
            'left_occ_mask_avg': left_occ_mask_avg,
            'right_occ_mask_avg': right_occ_mask_avg,
            'right_to_left_disp': right_to_left_disp,
            'left_to_right_disp': left_to_right_disp,
            'disp_left_smoothness': disp_left_smoothness,
            'disp_right_smoothness': disp_right_smoothness,
        }
        return disp_est, disp_left_est, disp_right_est, outputs, {}

    def build_losses(self, outputs):
        """Compute image reconstruction, smoothness and LR consistency losses.

        Returns:
            total_loss: scalar tensor.
        """
        params = self.params

        # IMAGE RECONSTRUCTION -- L1
        l1_left = [
            torch.abs(outputs['left_est'][i] - outputs['left_pyramid'][i]) *
            outputs['left_occ_mask'][i]
            for i in range(4)
        ]
        l1_reconstruction_loss_left = [
            torch.mean(l) / outputs['left_occ_mask_avg'][i]
            for i, l in enumerate(l1_left)
        ]

        l1_right = [
            torch.abs(outputs['right_est'][i] - outputs['right_pyramid'][i]) *
            outputs['right_occ_mask'][i]
            for i in range(4)
        ]
        l1_reconstruction_loss_right = [
            torch.mean(l) / outputs['right_occ_mask_avg'][i]
            for i, l in enumerate(l1_right)
        ]

        # SSIM
        ssim_loss_left = [
            torch.mean(self.SSIM(
                outputs['left_est'][i] * outputs['left_occ_mask'][i],
                outputs['left_pyramid'][i] * outputs['left_occ_mask'][i]
            )) / outputs['left_occ_mask_avg'][i]
            for i in range(4)
        ]
        ssim_loss_right = [
            torch.mean(self.SSIM(
                outputs['right_est'][i] * outputs['right_occ_mask'][i],
                outputs['right_pyramid'][i] * outputs['right_occ_mask'][i]
            )) / outputs['right_occ_mask_avg'][i]
            for i in range(4)
        ]

        # Weighted sum
        image_loss_left = [
            params.alpha_image_loss * ssim_loss_left[i] +
            (1 - params.alpha_image_loss) * l1_reconstruction_loss_left[i]
            for i in range(4)
        ]
        image_loss_right = [
            params.alpha_image_loss * ssim_loss_right[i] +
            (1 - params.alpha_image_loss) * l1_reconstruction_loss_right[i]
            for i in range(4)
        ]
        image_loss = sum(image_loss_left) + sum(image_loss_right)

        # DISPARITY SMOOTHNESS
        disp_left_loss = [
            torch.mean(torch.abs(outputs['disp_left_smoothness'][i])) / (2 ** i)
            for i in range(8)
        ]
        disp_right_loss = [
            torch.mean(torch.abs(outputs['disp_right_smoothness'][i])) / (2 ** i)
            for i in range(8)
        ]
        disp_gradient_loss = (sum(disp_left_loss) + sum(disp_right_loss)) * 0.5

        # LR CONSISTENCY
        lr_left_loss = [
            torch.mean(
                torch.abs(outputs['right_to_left_disp'][i] -
                          outputs['disp_left_est'][i]) *
                outputs['left_occ_mask'][i]
            ) / outputs['left_occ_mask_avg'][i]
            for i in range(4)
        ]
        lr_right_loss = [
            torch.mean(
                torch.abs(outputs['left_to_right_disp'][i] -
                          outputs['disp_right_est'][i]) *
                outputs['right_occ_mask'][i]
            ) / outputs['right_occ_mask_avg'][i]
            for i in range(4)
        ]
        lr_loss = sum(lr_left_loss) + sum(lr_right_loss)

        # TOTAL
        total_loss = (image_loss +
                      params.disp_gradient_loss_weight * disp_gradient_loss +
                      params.lr_loss_weight * lr_loss)
        return total_loss

    def forward(self, left, right, left_feature=None, right_feature=None):
        """Full forward pass: network + loss.

        Args:
            left, right: (B, 3, H, W) stereo pair.
            left_feature, right_feature: optional precomputed features.
        Returns:
            If training: (disp_est, total_loss)
            If testing: disp_est
        """
        disp_est, disp_left, disp_right, outputs, _ = self.build_outputs(
            left, right, left_feature, right_feature)
        if self.mode == 'test':
            return disp_est
        total_loss = self.build_losses(outputs)
        return disp_est, total_loss


# ---------------------------------------------------------------------------
# Standalone helper matching TF interface
# ---------------------------------------------------------------------------

def disp_godard(left_img, right_img, left_feature, right_feature, opt,
                is_training=True, pwc_disp_net=None):
    """Compute stereo disparity and loss using MonodepthModel.

    Args:
        left_img, right_img: (B, 3, H, W) stereo pair (NCHW, float [0,1]).
        left_feature, right_feature: precomputed feature tuples from
            FeaturePyramidDisp.
        opt: options namespace with attributes ssim_weight, depth_smooth_weight,
            img_height, img_width.
        is_training: if True, also computes and returns the total loss.
        pwc_disp_net: optional PWCDisp module (avoids creating a new one).
    Returns:
        If is_training:
            ([disp1, disp2, disp3, disp4], total_loss)
        Else:
            [disp1, disp2, disp3, disp4]

        Each disp is (B, 2, H_s, W_s) with channel 0 = left, channel 1 = right.
    """
    B = left_img.shape[0]
    params = monodepth_parameters(
        encoder='pwc',
        do_stereo=True,
        wrap_mode='border',
        use_deconv=False,
        alpha_image_loss=opt.ssim_weight,
        disp_gradient_loss_weight=opt.depth_smooth_weight,
        lr_loss_weight=1.0,
        full_summary=False,
        height=opt.img_height,
        width=opt.img_width,
        batch_size=B)

    mode = 'train' if is_training else 'test'
    model = MonodepthModel(params, mode, pwc_disp_net=pwc_disp_net)
    # Move to same device as input
    model = model.to(left_img.device)

    if is_training:
        disp_est, total_loss = model(left_img, right_img,
                                     left_feature, right_feature)
        return disp_est, total_loss
    else:
        disp_est = model(left_img, right_img,
                         left_feature, right_feature)
        return disp_est
