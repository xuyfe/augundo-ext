"""
High-level model classes for UnOS -- PyTorch reimplementation of models.py.

Each class combines the sub-networks (PWC-Flow, PWC-Disp, PoseNet) with the
appropriate loss functions for the four training modes:
  stereo, flow, depth, depthflow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets.pose_net import PoseExpNet
from .monodepth_model import disp_godard, MonodepthModel
from .nets.pwc_flow import PWCFlow, FeaturePyramidFlow
from .nets.pwc_disp import FeaturePyramidDisp, PWCDisp
from .optical_flow_warp_fwd import transformerFwd
from .optical_flow_warp_old import transformer_old
from .monodepth_dataloader import get_multi_scale_intrinsics
from .utils import inverse_warp, inverse_warp_new
from .loss_utils import (SSIM, deprocess_image, preprocess_image,
                         cal_grad2_error_mask, charbonnier_loss,
                         cal_grad2_error)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_area(img, size):
    """Resize NCHW tensor to (h, w) using area interpolation."""
    return F.interpolate(img, size=size, mode='area')


def _to_nhwc(t):
    """NCHW -> NHWC"""
    return t.permute(0, 2, 3, 1)


def _to_nchw(t):
    """NHWC -> NCHW"""
    return t.permute(0, 3, 1, 2)


def _compute_occu_masks(optical_flows_rev, B, H, W, num_scales, device):
    """Compute occlusion masks by forward-warping ones with reverse flows."""
    occu_masks = []
    for s in range(num_scales):
        Hs = H // (2 ** s)
        Ws = W // (2 ** s)
        ones_nhwc = torch.ones(B, Hs, Ws, 1, device=device)
        flowr = optical_flows_rev[s]  # NHWC
        mask = torch.clamp(
            transformerFwd(ones_nhwc, flowr, [Hs, Ws]),
            min=0.0, max=1.0)
        occu_masks.append(mask)
    return occu_masks


# ===================================================================
# TRAINING MODELS
# ===================================================================

class Model_stereo(nn.Module):
    """Mode='stereo': only train PWC-Disp using stereo pairs."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.feature_pyramid_disp = FeaturePyramidDisp()
        self.pwc_disp = PWCDisp(feature_pyramid=self.feature_pyramid_disp)

    def forward(self, image1, image1r, image2=None, image2r=None,
                cam2pix=None, pix2cam=None):
        """
        Args:
            image1, image1r: (B, 3, H, W) stereo pair.
        Returns:
            loss: scalar tensor.
            info: dict with sub-losses for logging.
        """
        opt = self.opt
        feature1_disp = self.feature_pyramid_disp(image1)
        feature1r_disp = self.feature_pyramid_disp(image1r)

        pred_disp, stereo_smooth_loss = disp_godard(
            image1, image1r, feature1_disp, feature1r_disp, opt,
            is_training=True, pwc_disp_net=self.pwc_disp)

        self.pred_disp = pred_disp
        self.pred_depth = [1.0 / d for d in pred_disp]
        self.loss = stereo_smooth_loss

        return self.loss, {
            'total_loss': self.loss.item(),
            'stereo_smooth_loss': stereo_smooth_loss.item(),
        }


class Model_flow(nn.Module):
    """Mode='flow': only train PWC-Flow for optical flow."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.feature_pyramid_flow = FeaturePyramidFlow()
        self.pwc_flow = PWCFlow(feature_pyramid=self.feature_pyramid_flow)

    def forward(self, image1, image1r=None, image2=None, image2r=None,
                cam2pix=None, pix2cam=None):
        opt = self.opt
        B, _, H, W = image1.shape
        device = image1.device

        feature1 = self.feature_pyramid_flow(image1)
        feature2 = self.feature_pyramid_flow(image2)

        optical_flows = self.pwc_flow.construct_model_pwc_full(
            image1, image2, feature1, feature2)  # list of NHWC tensors
        optical_flows_rev = self.pwc_flow.construct_model_pwc_full(
            image2, image1, feature2, feature1)

        occu_masks = _compute_occu_masks(
            optical_flows_rev, B, H, W, opt.num_scales, device)

        pixel_loss_optical = torch.tensor(0.0, device=device)
        flow_smooth_loss = torch.tensor(0.0, device=device)

        for s in range(opt.num_scales):
            Hs = H // (2 ** s)
            Ws = W // (2 ** s)
            curr_tgt = _to_nhwc(_resize_area(image1, (Hs, Ws)))
            curr_src = _to_nhwc(_resize_area(image2, (Hs, Ws)))

            occu_mask = occu_masks[s]
            occu_mask_avg = torch.mean(occu_mask) + 1e-12

            curr_proj_optical = transformer_old(
                curr_src, optical_flows[s], [Hs, Ws])
            curr_proj_error = torch.abs(curr_proj_optical - curr_tgt)

            pixel_loss_optical += (
                (1.0 - opt.ssim_weight) *
                torch.mean(curr_proj_error * occu_mask) / occu_mask_avg)

            if opt.ssim_weight > 0:
                pixel_loss_optical += (
                    opt.ssim_weight *
                    torch.mean(SSIM(
                        _to_nchw(curr_proj_optical * occu_mask),
                        _to_nchw(curr_tgt * occu_mask))) / occu_mask_avg)

            flow_smooth_loss += (
                opt.flow_smooth_weight *
                cal_grad2_error(
                    _to_nchw(optical_flows[s] / 20.0),
                    _to_nchw(curr_tgt), 1.0))

        self.loss = pixel_loss_optical + flow_smooth_loss
        return self.loss, {
            'total_loss': self.loss.item(),
            'pixel_loss_optical': pixel_loss_optical.item(),
            'flow_smooth_loss': flow_smooth_loss.item(),
        }


class Model_depth(nn.Module):
    """Mode='depth': train PWC-Disp + PoseNet using stereo + temporal."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.feature_pyramid_flow = FeaturePyramidFlow()
        self.feature_pyramid_disp = FeaturePyramidDisp()
        self.pwc_disp = PWCDisp(feature_pyramid=self.feature_pyramid_disp)
        self.pwc_flow = PWCFlow(feature_pyramid=self.feature_pyramid_flow)
        self.pose_net = PoseExpNet()

    def forward(self, image1, image1r, image2, image2r,
                cam2pix, pix2cam):
        opt = self.opt
        B, _, H, W = image1.shape
        device = image1.device

        feature1_flow = self.feature_pyramid_flow(image1)
        feature2_flow = self.feature_pyramid_flow(image2)

        feature1_disp = self.feature_pyramid_disp(image1)
        feature1r_disp = self.feature_pyramid_disp(image1r)

        pred_disp, stereo_smooth_loss = disp_godard(
            image1, image1r, feature1_disp, feature1r_disp, opt,
            is_training=True, pwc_disp_net=self.pwc_disp)

        pred_depth = [1.0 / d for d in pred_disp]
        pred_poses = self.pose_net(image1, image2)

        optical_flows_rev = self.pwc_flow.construct_model_pwc_full(
            image2, image1, feature2_flow, feature1_flow)

        occu_masks = _compute_occu_masks(
            optical_flows_rev, B, H, W, opt.num_scales, device)

        pixel_loss_depth = torch.tensor(0.0, device=device)

        for s in range(opt.num_scales):
            Hs = H // (2 ** s)
            Ws = W // (2 ** s)
            curr_tgt = _to_nhwc(_resize_area(image1, (Hs, Ws)))
            curr_src = _to_nhwc(_resize_area(image2, (Hs, Ws)))

            # depth_flow from depth + pose
            depth_s = _to_nhwc(pred_depth[s][:, 0:1, :, :])  # left depth only
            depth_flow, pose_mat = inverse_warp(
                depth_s, pred_poses,
                cam2pix[:, s, :, :], pix2cam[:, s, :, :])

            occu_mask = occu_masks[s]
            occu_mask_avg = torch.mean(occu_mask) + 1e-12

            curr_proj_depth = transformer_old(
                curr_src, depth_flow, [Hs, Ws])
            curr_proj_error = torch.abs(curr_proj_depth - curr_tgt)

            pixel_loss_depth += (
                (1.0 - opt.ssim_weight) *
                torch.mean(curr_proj_error * occu_mask) / occu_mask_avg)

            if opt.ssim_weight > 0:
                pixel_loss_depth += (
                    opt.ssim_weight *
                    torch.mean(SSIM(
                        _to_nchw(curr_proj_depth * occu_mask),
                        _to_nchw(curr_tgt * occu_mask))) / occu_mask_avg)

        self.loss = 10.0 * pixel_loss_depth + stereo_smooth_loss
        self.pred_disp = pred_disp
        return self.loss, {
            'total_loss': self.loss.item(),
            'pixel_loss_depth': pixel_loss_depth.item(),
            'stereo_smooth_loss': stereo_smooth_loss.item(),
        }


class Model_depthflow(nn.Module):
    """Mode='depthflow': joint depth, flow, pose, motion segmentation."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.feature_pyramid_flow = FeaturePyramidFlow()
        self.feature_pyramid_disp = FeaturePyramidDisp()
        self.pwc_disp = PWCDisp(feature_pyramid=self.feature_pyramid_disp)
        self.pwc_flow = PWCFlow(feature_pyramid=self.feature_pyramid_flow)
        self.pose_net = PoseExpNet()

    def forward(self, image1, image1r, image2, image2r,
                cam2pix, pix2cam):
        opt = self.opt
        B, _, H, W = image1.shape
        device = image1.device

        feature1_flow = self.feature_pyramid_flow(image1)
        feature2_flow = self.feature_pyramid_flow(image2)

        feature1_disp = self.feature_pyramid_disp(image1)
        feature1r_disp = self.feature_pyramid_disp(image1r)

        pred_disp, stereo_smooth_loss = disp_godard(
            image1, image1r, feature1_disp, feature1r_disp, opt,
            is_training=True, pwc_disp_net=self.pwc_disp)

        pred_depth = [1.0 / d for d in pred_disp]
        pred_poses = self.pose_net(image1, image2)

        optical_flows_rev = self.pwc_flow.construct_model_pwc_full(
            image2, image1, feature2_flow, feature1_flow)

        # Also compute forward disparity for frame 2 (for inverse_warp_new)
        feature2_disp = self.feature_pyramid_disp(image2)
        feature2r_disp = self.feature_pyramid_disp(image2r)
        pred_disp_rev = disp_godard(
            image2, image2r, feature2_disp, feature2r_disp, opt,
            is_training=False, pwc_disp_net=self.pwc_disp)

        optical_flows = self.pwc_flow.construct_model_pwc_full(
            image1, image2, feature1_flow, feature2_flow)

        occu_masks = _compute_occu_masks(
            optical_flows_rev, B, H, W, opt.num_scales, device)

        # Refined pose via inverse_warp_new at scale 0
        depth1_s0 = _to_nhwc(pred_depth[0][:, 0:1, :, :])
        depth2_s0 = _to_nhwc(1.0 / pred_disp_rev[0][:, 0:1, :, :])
        _, pose_mat, _, _ = inverse_warp_new(
            depth1_s0, depth2_s0, pred_poses,
            cam2pix[:, 0, :, :], pix2cam[:, 0, :, :],
            optical_flows[0], occu_masks[0])

        pixel_loss_depth = torch.tensor(0.0, device=device)
        pixel_loss_optical = torch.tensor(0.0, device=device)
        flow_smooth_loss = torch.tensor(0.0, device=device)
        flow_consist_loss = torch.tensor(0.0, device=device)

        for s in range(opt.num_scales):
            Hs = H // (2 ** s)
            Ws = W // (2 ** s)
            curr_tgt = _to_nhwc(_resize_area(image1, (Hs, Ws)))
            curr_src = _to_nhwc(_resize_area(image2, (Hs, Ws)))
            occu_mask = occu_masks[s]

            # Rigid flow from depth + refined pose
            depth_s = _to_nhwc(pred_depth[s][:, 0:1, :, :])
            depth_flow, pose_mat = inverse_warp(
                depth_s, pose_mat.detach(),
                cam2pix[:, s, :, :], pix2cam[:, s, :, :])

            # Rigid flow from depth + original pose
            depth_flow_orig, _ = inverse_warp(
                depth_s.detach(), pred_poses,
                cam2pix[:, s, :, :], pix2cam[:, s, :, :])

            # Flow consistency mask
            flow_diff = torch.sqrt(
                torch.sum((depth_flow - optical_flows[s]) ** 2,
                          dim=3, keepdim=True) + 1e-12)
            flow_diff_mask = (flow_diff < (opt.flow_diff_threshold / (2 ** s))).float()
            occu_region = (occu_mask < 0.5).float()
            ref_exp_mask = torch.clamp(
                flow_diff_mask + occu_region, min=0.0, max=1.0)

            occu_mask_avg = torch.mean(occu_mask) + 1e-12

            # Depth reconstruction loss (refined pose)
            curr_proj_depth = transformer_old(
                curr_src, depth_flow, [Hs, Ws])
            curr_proj_error_depth = torch.abs(
                curr_proj_depth - curr_tgt) * ref_exp_mask
            pixel_loss_depth += (
                (1.0 - opt.ssim_weight) *
                torch.mean(curr_proj_error_depth * occu_mask) / occu_mask_avg)

            # Depth reconstruction loss (original pose)
            curr_proj_depth_orig = transformer_old(
                curr_src, depth_flow_orig, [Hs, Ws])
            curr_proj_error_orig = torch.abs(
                curr_proj_depth_orig - curr_tgt) * ref_exp_mask
            pixel_loss_depth += (
                (1.0 - opt.ssim_weight) *
                torch.mean(curr_proj_error_orig * occu_mask) / occu_mask_avg)

            # Optical flow reconstruction loss
            curr_proj_optical = transformer_old(
                curr_src, optical_flows[s], [Hs, Ws])
            curr_proj_error_optical = torch.abs(
                curr_proj_optical - curr_tgt)
            pixel_loss_optical += (
                (1.0 - opt.ssim_weight) *
                torch.mean(curr_proj_error_optical * occu_mask) / occu_mask_avg)

            if opt.ssim_weight > 0:
                pixel_loss_depth += opt.ssim_weight * torch.mean(
                    SSIM(_to_nchw(curr_proj_depth * occu_mask * ref_exp_mask),
                         _to_nchw(curr_tgt * occu_mask * ref_exp_mask))
                ) / occu_mask_avg
                pixel_loss_depth += opt.ssim_weight * torch.mean(
                    SSIM(_to_nchw(curr_proj_depth_orig * occu_mask * ref_exp_mask),
                         _to_nchw(curr_tgt * occu_mask * ref_exp_mask))
                ) / occu_mask_avg
                pixel_loss_optical += opt.ssim_weight * torch.mean(
                    SSIM(_to_nchw(curr_proj_optical * occu_mask),
                         _to_nchw(curr_tgt * occu_mask))
                ) / occu_mask_avg

            flow_smooth_loss += opt.flow_smooth_weight * cal_grad2_error_mask(
                _to_nchw(optical_flows[s] / 20.0),
                _to_nchw(curr_tgt), 1.0,
                _to_nchw(1.0 - ref_exp_mask))

            depth_flow_stop = depth_flow.detach()
            flow_consist_loss += opt.flow_consist_weight * charbonnier_loss(
                _to_nchw(depth_flow_stop - optical_flows[s]),
                _to_nchw(ref_exp_mask))

        self.loss = (
            10.0 * pixel_loss_depth + stereo_smooth_loss +
            pixel_loss_optical + flow_smooth_loss + flow_consist_loss)
        self.pred_disp = pred_disp
        return self.loss, {
            'total_loss': self.loss.item(),
            'pixel_loss_depth': pixel_loss_depth.item(),
            'pixel_loss_optical': pixel_loss_optical.item(),
            'stereo_smooth_loss': stereo_smooth_loss.item(),
            'flow_smooth_loss': flow_smooth_loss.item(),
            'flow_consist_loss': flow_consist_loss.item(),
        }


# ===================================================================
# EVALUATION MODELS
# ===================================================================

class Model_eval_stereo(nn.Module):
    """Inference-only stereo disparity model."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.feature_pyramid_disp = FeaturePyramidDisp()
        self.pwc_disp = PWCDisp(feature_pyramid=self.feature_pyramid_disp)

    @torch.no_grad()
    def forward(self, image1, image1r, image2=None, image2r=None,
                intrinsic=None):
        opt = self.opt
        input_1 = preprocess_image(image1)
        input_1r = preprocess_image(image1r)

        feat1 = self.feature_pyramid_disp(input_1)
        feat1r = self.feature_pyramid_disp(input_1r)

        pred_disp = disp_godard(
            input_1, input_1r, feat1, feat1r, opt,
            is_training=False, pwc_disp_net=self.pwc_disp)

        self.pred_disp = pred_disp[0][:, 0:1, :, :]  # left disp, finest scale
        self.pred_flow_optical = torch.tensor(0.0)
        self.pred_flow_rigid = torch.tensor(0.0)
        self.pred_disp2 = torch.tensor(0.0)
        self.pred_mask = torch.tensor(0.0)
        return self


class Model_eval_flow(nn.Module):
    """Inference-only optical flow model."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.feature_pyramid_flow = FeaturePyramidFlow()
        self.pwc_flow = PWCFlow(feature_pyramid=self.feature_pyramid_flow)

    @torch.no_grad()
    def forward(self, image1, image1r, image2, image2r=None,
                intrinsic=None):
        input_1 = preprocess_image(image1)
        input_2 = preprocess_image(image2)

        feat1 = self.feature_pyramid_flow(input_1)
        feat2 = self.feature_pyramid_flow(input_2)

        optical_flows = self.pwc_flow.construct_model_pwc_full(
            input_1, input_2, feat1, feat2)

        self.pred_flow_optical = optical_flows[0]  # finest scale NHWC
        self.pred_flow_rigid = torch.tensor(0.0)
        self.pred_disp = torch.tensor(0.0)
        self.pred_disp2 = torch.tensor(0.0)
        self.pred_mask = torch.tensor(0.0)
        return self


class Model_eval_depth(nn.Module):
    """Inference-only depth + pose model."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.feature_pyramid_disp = FeaturePyramidDisp()
        self.feature_pyramid_flow = FeaturePyramidFlow()
        self.pwc_disp = PWCDisp(feature_pyramid=self.feature_pyramid_disp)
        self.pwc_flow = PWCFlow(feature_pyramid=self.feature_pyramid_flow)
        self.pose_net = PoseExpNet()

    @torch.no_grad()
    def forward(self, image1, image1r, image2, image2r=None,
                intrinsic=None):
        opt = self.opt
        input_1 = preprocess_image(image1)
        input_2 = preprocess_image(image2)
        input_1r = preprocess_image(image1r)

        feat1_disp = self.feature_pyramid_disp(input_1)
        feat1r_disp = self.feature_pyramid_disp(input_1r)
        feat1_flow = self.feature_pyramid_flow(input_1)
        feat2_flow = self.feature_pyramid_flow(input_2)

        pred_disp = disp_godard(
            input_1, input_1r, feat1_disp, feat1r_disp, opt,
            is_training=False, pwc_disp_net=self.pwc_disp)
        pred_poses = self.pose_net(input_1, input_2)
        optical_flows = self.pwc_flow.construct_model_pwc_full(
            input_1, input_2, feat1_flow, feat2_flow)

        cam2pix, pix2cam = get_multi_scale_intrinsics(intrinsic, opt.num_scales)
        cam2pix = cam2pix.unsqueeze(0).to(input_1.device)
        pix2cam = pix2cam.unsqueeze(0).to(input_1.device)

        s = 0
        depth_s = _to_nhwc(1.0 / pred_disp[s][:, 0:1, :, :])
        depth_flow, pose_mat = inverse_warp(
            depth_s, pred_poses,
            cam2pix[:, s, :, :], pix2cam[:, s, :, :])

        self.pred_flow_rigid = depth_flow      # NHWC
        self.pred_flow_optical = optical_flows[0]  # NHWC
        self.pred_disp = pred_disp[0][:, 0:1, :, :]  # NCHW left disp
        self.pred_pose_mat = pose_mat[0, :, :]
        self.pred_disp2 = torch.tensor(0.0)
        self.pred_mask = torch.tensor(0.0)
        return self


class Model_eval_depthflow(nn.Module):
    """Inference-only joint depth + flow model."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.feature_pyramid_disp = FeaturePyramidDisp()
        self.feature_pyramid_flow = FeaturePyramidFlow()
        self.pwc_disp = PWCDisp(feature_pyramid=self.feature_pyramid_disp)
        self.pwc_flow = PWCFlow(feature_pyramid=self.feature_pyramid_flow)
        self.pose_net = PoseExpNet()

    @torch.no_grad()
    def forward(self, image1, image1r, image2, image2r,
                intrinsic=None):
        opt = self.opt
        input_1 = preprocess_image(image1)
        input_2 = preprocess_image(image2)
        input_1r = preprocess_image(image1r)
        input_2r = preprocess_image(image2r)

        feat1_disp = self.feature_pyramid_disp(input_1)
        feat1r_disp = self.feature_pyramid_disp(input_1r)
        feat2_disp = self.feature_pyramid_disp(input_2)
        feat2r_disp = self.feature_pyramid_disp(input_2r)
        feat1_flow = self.feature_pyramid_flow(input_1)
        feat2_flow = self.feature_pyramid_flow(input_2)

        pred_disp = disp_godard(
            input_1, input_1r, feat1_disp, feat1r_disp, opt,
            is_training=False, pwc_disp_net=self.pwc_disp)
        pred_disp_rev = disp_godard(
            input_2, input_2r, feat2_disp, feat2r_disp, opt,
            is_training=False, pwc_disp_net=self.pwc_disp)

        pred_poses = self.pose_net(input_1, input_2)

        optical_flows = self.pwc_flow.construct_model_pwc_full(
            input_1, input_2, feat1_flow, feat2_flow)
        optical_flows_rev = self.pwc_flow.construct_model_pwc_full(
            input_2, input_1, feat2_flow, feat1_flow)

        cam2pix, pix2cam = get_multi_scale_intrinsics(intrinsic, opt.num_scales)
        cam2pix = cam2pix.unsqueeze(0).to(input_1.device)
        pix2cam = pix2cam.unsqueeze(0).to(input_1.device)

        s = 0
        Hs = opt.img_height // (2 ** s)
        Ws = opt.img_width // (2 ** s)
        B = input_1.shape[0]
        ones_nhwc = torch.ones(B, Hs, Ws, 1, device=input_1.device)
        occu_mask = torch.clamp(
            transformerFwd(ones_nhwc, optical_flows_rev[s], [Hs, Ws]),
            min=0.0, max=1.0)

        depth1 = _to_nhwc(1.0 / pred_disp[0][:, 0:1, :, :])
        depth2 = _to_nhwc(1.0 / pred_disp_rev[0][:, 0:1, :, :])
        depth_flow, pose_mat, disp1_trans, small_mask = inverse_warp_new(
            depth1, depth2, pred_poses,
            cam2pix[:, 0, :, :], pix2cam[:, 0, :, :],
            optical_flows[0], occu_mask)

        flow_diff = torch.sqrt(
            torch.sum((depth_flow - optical_flows[0]) ** 2,
                      dim=3, keepdim=True) + 1e-12)
        flow_diff_mask = (flow_diff < opt.flow_diff_threshold).float()
        occu_region = (occu_mask < 0.5).float()
        ref_exp_mask = torch.clamp(
            flow_diff_mask + occu_region, min=0.0, max=1.0)

        self.pred_flow_rigid = depth_flow
        self.pred_flow_optical = optical_flows[0]
        self.pred_disp = pred_disp[0][:, 0:1, :, :]
        self.pred_pose_mat = pose_mat[0, :, :]

        # pred_disp2: warped reverse disparity via optical flow
        disp_rev_nhwc = _to_nhwc(pred_disp_rev[0][:, 0:1, :, :])
        warped_disp_rev = transformer_old(
            disp_rev_nhwc, optical_flows[0],
            [opt.img_height, opt.img_width])
        self.pred_disp2 = _to_nchw(warped_disp_rev)
        self.pred_mask = 1.0 - ref_exp_mask  # NHWC
        return self
