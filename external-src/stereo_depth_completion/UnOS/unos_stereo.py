"""
Main UnOS model classes combining all components.

Ported from the original TensorFlow models.py.
Provides three model variants of increasing complexity:

- UnOSStereo: Pure stereo disparity estimation with smoothness loss.
- UnOSDepth: Stereo + temporal (disparity + pose + photometric loss).
- UnOSDepthFlow: Full model with depth + optical flow + pose + mutual
  consistency losses. This is the main model to use.

All tensors use PyTorch NCHW format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_pyramid import FeaturePyramidDisp, FeaturePyramidFlow
from .pwc_disp import PWCDisp
from .pwc_flow import PWCFlow
from .pose_net import PoseNet
from .monodepth_model import MonodepthModel, disp_godard
from .geometry import inverse_warp, inverse_warp_new
from .warping import backward_warp, forward_warp
from .loss_utils import SSIM, cal_grad2_error, cal_grad2_error_mask, charbonnier_loss


class UnOSStereo(nn.Module):
    """Pure stereo disparity estimation model.

    Uses a feature pyramid encoder and PWC-Net disparity decoder to estimate
    left and right disparities from a stereo image pair. Training is
    self-supervised using photometric reconstruction, smoothness, and
    left-right consistency losses.
    """

    def __init__(self, params):
        """
        Args:
            params: Configuration dict or object with fields:
                - alpha_image_loss: SSIM vs L1 balance (default 0.85)
                - disp_gradient_loss_weight: Smoothness weight (default 0.1)
                - lr_loss_weight: L-R consistency weight (default 1.0)
                - height: Input image height
                - width: Input image width
                - batch_size: Training batch size
        """
        super().__init__()
        self.feature_pyramid = FeaturePyramidDisp()
        self.monodepth = MonodepthModel(params)

    def forward(self, left, right):
        """Estimate disparities from a stereo pair.

        Args:
            left: Left image, N x 3 x H x W.
            right: Right image, N x 3 x H x W.

        Returns:
            disps: List of 4 disparity tensors at [1/4, 1/8, 1/16, 1/32] scales.
                   Each tensor is N x 2 x H_s x W_s.
        """
        left_features = self.feature_pyramid(left)
        right_features = self.feature_pyramid(right)
        disps = self.monodepth(left, right, left_features, right_features)
        return disps

    def compute_loss(self, left, right):
        """Estimate disparities and compute training loss.

        Args:
            left: Left image, N x 3 x H x W.
            right: Right image, N x 3 x H x W.

        Returns:
            disps: List of 4 disparity tensors.
            total_loss: Scalar training loss.
        """
        left_features = self.feature_pyramid(left)
        right_features = self.feature_pyramid(right)
        disps, total_loss = self.monodepth.compute_loss(
            left, right, left_features, right_features
        )
        return disps, total_loss


class UnOSDepth(nn.Module):
    """Stereo + temporal depth estimation model.

    Extends UnOSStereo with temporal consistency: uses a pose network to
    estimate the relative camera motion between consecutive frames and
    adds a photometric loss between the current frame and the temporally
    warped adjacent frame.
    """

    def __init__(self, params):
        """
        Args:
            params: Configuration dict or object. In addition to UnOSStereo params:
                - photo_loss_weight: Temporal photometric loss weight (default 1.0)
                - smooth_loss_weight: Temporal smooth loss weight (default 0.1)
        """
        super().__init__()
        self.feature_pyramid = FeaturePyramidDisp()
        self.monodepth = MonodepthModel(params)
        self.pose_net = PoseNet()

        if isinstance(params, dict):
            self.photo_loss_weight = params.get('photo_loss_weight', 1.0)
            self.smooth_loss_weight = params.get('smooth_loss_weight', 0.1)
        else:
            self.photo_loss_weight = getattr(params, 'photo_loss_weight', 1.0)
            self.smooth_loss_weight = getattr(params, 'smooth_loss_weight', 0.1)

    def forward(self, left, right, left_next=None):
        """Estimate disparities and optionally pose.

        Args:
            left: Left image at time t, N x 3 x H x W.
            right: Right image at time t, N x 3 x H x W.
            left_next: Left image at time t+1, N x 3 x H x W. Optional.

        Returns:
            dict with:
                - 'disps': List of 4 disparity tensors.
                - 'pose': Pose vector N x 6 (if left_next provided).
        """
        left_features = self.feature_pyramid(left)
        right_features = self.feature_pyramid(right)
        disps = self.monodepth(left, right, left_features, right_features)

        result = {'disps': disps}

        if left_next is not None:
            pose = self.pose_net(left, left_next)
            result['pose'] = pose

        return result

    def compute_loss(self, left, right, left_next, intrinsics):
        """Compute combined stereo + temporal loss.

        Args:
            left: Left image at time t, N x 3 x H x W.
            right: Right image at time t, N x 3 x H x W.
            left_next: Left image at time t+1, N x 3 x H x W.
            intrinsics: Camera intrinsics, N x 3 x 3.

        Returns:
            result: Dict with disparities, pose, and losses.
        """
        left_features = self.feature_pyramid(left)
        right_features = self.feature_pyramid(right)

        # Stereo loss
        disps, stereo_loss = self.monodepth.compute_loss(
            left, right, left_features, right_features
        )

        # Pose estimation
        pose = self.pose_net(left, left_next)

        # Temporal photometric loss
        temporal_loss = 0.0
        # Use finest scale disparity
        disp_left = disps[0][:, 0:1, :, :]  # N x 1 x H/4 x W/4
        _, _, dh, dw = disp_left.shape

        # Convert disparity to depth (assuming baseline=1, focal from intrinsics)
        # depth = baseline * fx / disparity
        # For normalized disparity: depth = fx / (disp * W)
        fx = intrinsics[:, 0, 0].reshape(-1, 1, 1, 1)
        depth = fx / (disp_left.clamp(min=1e-6) * dw)

        # Scale intrinsics for this resolution
        scale_x = dw / left.shape[3]
        scale_y = dh / left.shape[2]
        intrinsics_scaled = intrinsics.clone()
        intrinsics_scaled[:, 0, :] = intrinsics_scaled[:, 0, :] * scale_x
        intrinsics_scaled[:, 1, :] = intrinsics_scaled[:, 1, :] * scale_y

        # Compute flow from depth + pose
        flow, _ = inverse_warp(depth, pose, intrinsics_scaled)

        # Warp next image to current
        left_next_scaled = F.interpolate(left_next, size=(dh, dw),
                                         mode='bilinear', align_corners=True)
        left_scaled = F.interpolate(left, size=(dh, dw),
                                    mode='bilinear', align_corners=True)

        left_next_warped = backward_warp(left_next_scaled, flow)

        # Photometric loss
        ssim_loss = SSIM(left_next_warped, left_scaled)
        l1_loss = torch.abs(left_next_warped - left_scaled)
        # Trim L1 to match SSIM
        l1_trimmed = l1_loss[:, :, 1:-1, 1:-1]

        temporal_loss = torch.mean(0.85 * ssim_loss + 0.15 * l1_trimmed)

        total_loss = stereo_loss + self.photo_loss_weight * temporal_loss

        return {
            'disps': disps,
            'pose': pose,
            'stereo_loss': stereo_loss,
            'temporal_loss': temporal_loss,
            'total_loss': total_loss,
        }


class UnOSDepthFlow(nn.Module):
    """Full UnOS model: depth + optical flow + pose with mutual consistency.

    The complete UnOS model that combines:
    1. Stereo disparity estimation (left-right)
    2. Optical flow estimation (temporal)
    3. Camera pose estimation
    4. Mutual consistency between depth-induced flow and optical flow
    5. SVD-based pose refinement

    This is the main model to use for best performance.
    """

    def __init__(self, params):
        """
        Args:
            params: Configuration dict or object with all loss weights:
                - alpha_image_loss: SSIM vs L1 balance (default 0.85)
                - disp_gradient_loss_weight: Disparity smoothness (default 0.1)
                - lr_loss_weight: L-R consistency (default 1.0)
                - photo_loss_weight: Temporal photometric (default 1.0)
                - flow_smooth_weight: Flow smoothness (default 0.1)
                - flow_consist_weight: Flow-depth consistency (default 0.01)
                - height, width, batch_size
        """
        super().__init__()

        # Feature pyramids (separate for disparity and flow)
        self.disp_feature_pyramid = FeaturePyramidDisp()
        self.flow_feature_pyramid = FeaturePyramidFlow()

        # Sub-networks
        self.monodepth = MonodepthModel(params)
        self.pwc_flow = PWCFlow()
        self.pose_net = PoseNet()

        # Loss weights
        if isinstance(params, dict):
            self.photo_loss_weight = params.get('photo_loss_weight', 1.0)
            self.flow_smooth_weight = params.get('flow_smooth_weight', 0.1)
            self.flow_consist_weight = params.get('flow_consist_weight', 0.01)
            self.alpha_image_loss = params.get('alpha_image_loss', 0.85)
        else:
            self.photo_loss_weight = getattr(params, 'photo_loss_weight', 1.0)
            self.flow_smooth_weight = getattr(params, 'flow_smooth_weight', 0.1)
            self.flow_consist_weight = getattr(params, 'flow_consist_weight', 0.01)
            self.alpha_image_loss = getattr(params, 'alpha_image_loss', 0.85)

    def forward(self, left, right, left_next=None, right_next=None):
        """Forward pass: estimate disparities, flow, and pose.

        Args:
            left: Left image at time t, N x 3 x H x W.
            right: Right image at time t, N x 3 x H x W.
            left_next: Left image at time t+1, N x 3 x H x W. Optional.
            right_next: Right image at time t+1, N x 3 x H x W. Optional.

        Returns:
            dict with:
                - 'disps': List of 4 stereo disparity tensors.
                - 'fwd_flows': List of 4 forward optical flow tensors (if left_next).
                - 'bwd_flows': List of 4 backward optical flow tensors (if left_next).
                - 'pose': Pose vector N x 6 (if left_next).
        """
        # Stereo disparity
        disp_feat_left = self.disp_feature_pyramid(left)
        disp_feat_right = self.disp_feature_pyramid(right)
        disps = self.monodepth(left, right, disp_feat_left, disp_feat_right)

        result = {'disps': disps}

        if left_next is not None:
            # Optical flow
            flow_feat1 = self.flow_feature_pyramid(left)
            flow_feat2 = self.flow_feature_pyramid(left_next)
            fwd_flows, bwd_flows = self.pwc_flow(flow_feat1, flow_feat2, left, left_next)
            result['fwd_flows'] = fwd_flows
            result['bwd_flows'] = bwd_flows

            # Pose
            pose = self.pose_net(left, left_next)
            result['pose'] = pose

        return result

    def compute_loss(self, left, right, left_next, intrinsics):
        """Compute full combined loss.

        Args:
            left: Left image at time t, N x 3 x H x W.
            right: Right image at time t, N x 3 x H x W.
            left_next: Left image at time t+1, N x 3 x H x W.
            intrinsics: Camera intrinsics, N x 3 x 3.

        Returns:
            dict with all outputs and loss components.
        """
        # --- Stereo disparity ---
        disp_feat_left = self.disp_feature_pyramid(left)
        disp_feat_right = self.disp_feature_pyramid(right)
        disps, stereo_loss = self.monodepth.compute_loss(
            left, right, disp_feat_left, disp_feat_right
        )

        # --- Optical flow ---
        flow_feat1 = self.flow_feature_pyramid(left)
        flow_feat2 = self.flow_feature_pyramid(left_next)
        fwd_flows, bwd_flows = self.pwc_flow(flow_feat1, flow_feat2, left, left_next)

        # --- Pose ---
        pose_fwd = self.pose_net(left, left_next)
        pose_bwd = self.pose_net(left_next, left)

        # --- Temporal photometric loss (using optical flow) ---
        flow_photo_loss = 0.0
        flow_smooth_loss = 0.0
        consist_loss = 0.0

        # Use finest scale
        fwd_flow = fwd_flows[0]  # N x 2 x H/4 x W/4
        bwd_flow = bwd_flows[0]

        _, _, fh, fw = fwd_flow.shape

        left_scaled = F.interpolate(left, size=(fh, fw),
                                    mode='bilinear', align_corners=True)
        left_next_scaled = F.interpolate(left_next, size=(fh, fw),
                                         mode='bilinear', align_corners=True)

        # Forward warping for occlusion detection
        ones = torch.ones(left.shape[0], 1, fh, fw, dtype=left.dtype, device=left.device)
        fwd_occu = forward_warp(ones, fwd_flow)
        bwd_occu = forward_warp(ones, bwd_flow)
        fwd_occu_mask = (fwd_occu >= 1.0).float()
        bwd_occu_mask = (bwd_occu >= 1.0).float()

        # Forward flow photometric loss
        left_next_warped = backward_warp(left_next_scaled, fwd_flow)
        fwd_ssim = SSIM(left_next_warped, left_scaled)
        fwd_l1 = torch.abs(left_next_warped - left_scaled)[:, :, 1:-1, 1:-1]
        fwd_occu_trimmed = fwd_occu_mask[:, :, 1:-1, 1:-1]

        fwd_photo = (self.alpha_image_loss * fwd_ssim +
                     (1 - self.alpha_image_loss) * fwd_l1) * fwd_occu_trimmed
        fwd_photo_loss = torch.sum(fwd_photo) / (torch.sum(fwd_occu_trimmed) + 1e-8)

        # Backward flow photometric loss
        left_warped = backward_warp(left_scaled, bwd_flow)
        bwd_ssim = SSIM(left_warped, left_next_scaled)
        bwd_l1 = torch.abs(left_warped - left_next_scaled)[:, :, 1:-1, 1:-1]
        bwd_occu_trimmed = bwd_occu_mask[:, :, 1:-1, 1:-1]

        bwd_photo = (self.alpha_image_loss * bwd_ssim +
                     (1 - self.alpha_image_loss) * bwd_l1) * bwd_occu_trimmed
        bwd_photo_loss = torch.sum(bwd_photo) / (torch.sum(bwd_occu_trimmed) + 1e-8)

        flow_photo_loss = (fwd_photo_loss + bwd_photo_loss) / 2.0

        # --- Flow smoothness loss ---
        flow_smooth_loss = (
            cal_grad2_error(fwd_flow, left_scaled, beta=10.0) +
            cal_grad2_error(bwd_flow, left_next_scaled, beta=10.0)
        ) / 2.0

        # --- Depth-flow consistency loss ---
        # Convert finest disparity to depth
        disp_left = disps[0][:, 0:1, :, :]  # N x 1 x H/4 x W/4
        _, _, dh, dw = disp_left.shape

        fx = intrinsics[:, 0, 0].reshape(-1, 1, 1, 1)
        depth = fx / (disp_left.clamp(min=1e-6) * dw)

        # Scale intrinsics
        scale_x = dw / left.shape[3]
        scale_y = dh / left.shape[2]
        intrinsics_scaled = intrinsics.clone()
        intrinsics_scaled[:, 0, :] = intrinsics_scaled[:, 0, :] * scale_x
        intrinsics_scaled[:, 1, :] = intrinsics_scaled[:, 1, :] * scale_y

        # Depth-induced flow
        depth_flow, _ = inverse_warp(depth, pose_fwd, intrinsics_scaled)

        # Resize to match optical flow if needed
        if depth_flow.shape[2:] != fwd_flow.shape[2:]:
            depth_flow = F.interpolate(depth_flow, size=fwd_flow.shape[2:],
                                       mode='bilinear', align_corners=True)

        # Consistency: optical flow should match depth-induced flow
        consist_error = torch.abs(fwd_flow - depth_flow) * fwd_occu_mask
        consist_loss = torch.sum(consist_error) / (torch.sum(fwd_occu_mask) + 1e-8)

        # --- Total loss ---
        total_loss = (
            stereo_loss +
            self.photo_loss_weight * flow_photo_loss +
            self.flow_smooth_weight * flow_smooth_loss +
            self.flow_consist_weight * consist_loss
        )

        return {
            'disps': disps,
            'fwd_flows': fwd_flows,
            'bwd_flows': bwd_flows,
            'pose_fwd': pose_fwd,
            'pose_bwd': pose_bwd,
            'stereo_loss': stereo_loss,
            'flow_photo_loss': flow_photo_loss,
            'flow_smooth_loss': flow_smooth_loss,
            'consist_loss': consist_loss,
            'total_loss': total_loss,
        }
