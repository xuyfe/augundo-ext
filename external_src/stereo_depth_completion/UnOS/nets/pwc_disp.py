"""
PWC stereo disparity network -- PyTorch reimplementation of UnOS / UnDepthflow.
Mirrors the TensorFlow original in nets/pwc_disp.py.

Key difference from pwc_flow: cost volume searches only along the horizontal
(x) axis, so its output has (2*d+1) channels instead of (2*d+1)^2.
The decoder outputs 1-channel disparity (+ a zero y-channel for warping).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.1)


def resize_like(inputs, ref):
    """Resize *inputs* to the spatial size of *ref* (both NCHW)."""
    if inputs.shape[2:] == ref.shape[2:]:
        return inputs
    return F.interpolate(
        inputs, size=ref.shape[2:], mode='bilinear', align_corners=False)


def warp(x, flow):
    """Warp tensor *x* (NCHW) by *flow* (N,2,H,W) using grid_sample.

    flow[:,0] is horizontal (x-direction), flow[:,1] is vertical.
    """
    B, C, H, W = x.size()
    xx = torch.arange(0, W, device=x.device, dtype=x.dtype).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device, dtype=x.dtype).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
    mask = torch.ones_like(x)
    mask = F.grid_sample(mask, vgrid, align_corners=True)
    mask = (mask >= 0.9999).float()
    return output * mask


# ---------------------------------------------------------------------------
# Feature pyramid (identical structure to flow, but separate weights)
# ---------------------------------------------------------------------------

class FeaturePyramidDisp(nn.Module):
    """3 -> 16 -> 16 -> 32 -> 32 -> 64 -> 64 -> 96 -> 96 -> 128 -> 128 -> 192 -> 192"""

    def __init__(self):
        super().__init__()
        self.cnv1  = nn.Conv2d(3,   16,  3, stride=2, padding=1)
        self.cnv2  = nn.Conv2d(16,  16,  3, stride=1, padding=1)
        self.cnv3  = nn.Conv2d(16,  32,  3, stride=2, padding=1)
        self.cnv4  = nn.Conv2d(32,  32,  3, stride=1, padding=1)
        self.cnv5  = nn.Conv2d(32,  64,  3, stride=2, padding=1)
        self.cnv6  = nn.Conv2d(64,  64,  3, stride=1, padding=1)
        self.cnv7  = nn.Conv2d(64,  96,  3, stride=2, padding=1)
        self.cnv8  = nn.Conv2d(96,  96,  3, stride=1, padding=1)
        self.cnv9  = nn.Conv2d(96,  128, 3, stride=2, padding=1)
        self.cnv10 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.cnv11 = nn.Conv2d(128, 192, 3, stride=2, padding=1)
        self.cnv12 = nn.Conv2d(192, 192, 3, stride=1, padding=1)

    def forward(self, image):
        c1  = leaky_relu(self.cnv1(image))
        c2  = leaky_relu(self.cnv2(c1))
        c3  = leaky_relu(self.cnv3(c2))
        c4  = leaky_relu(self.cnv4(c3))
        c5  = leaky_relu(self.cnv5(c4))
        c6  = leaky_relu(self.cnv6(c5))
        c7  = leaky_relu(self.cnv7(c6))
        c8  = leaky_relu(self.cnv8(c7))
        c9  = leaky_relu(self.cnv9(c8))
        c10 = leaky_relu(self.cnv10(c9))
        c11 = leaky_relu(self.cnv11(c10))
        c12 = leaky_relu(self.cnv12(c11))
        return c2, c4, c6, c8, c10, c12


# ---------------------------------------------------------------------------
# 1-D cost volume (horizontal search only, d=4 -> 2*d+1 = 9 channels)
# ---------------------------------------------------------------------------

class CostVolumeDisp(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.d = d

    def forward(self, feature1, feature2):
        """
        1-D horizontal cost volume.
        feature1, feature2: (B, C, H, W)
        returns: (B, 2*d+1, H, W)
        """
        d = self.d
        B, C, H, W = feature1.size()
        # Pad only in the width (horizontal) dimension
        feature2_pad = F.pad(feature2, [d, d, 0, 0], mode='constant', value=0)
        cv = []
        for j in range(2 * d + 1):
            cost = torch.mean(
                feature1 * feature2_pad[:, :, :, j:j + W],
                dim=1, keepdim=True)
            cv.append(cost)
        return torch.cat(cv, dim=1)


def cost_volume_disp(feature1, feature2, d=4):
    """Functional wrapper matching the original ``cost_volumn()`` signature."""
    B, C, H, W = feature1.size()
    feature2_pad = F.pad(feature2, [d, d, 0, 0], mode='constant', value=0)
    cv = []
    for j in range(2 * d + 1):
        cost = torch.mean(
            feature1 * feature2_pad[:, :, :, j:j + W],
            dim=1, keepdim=True)
        cv.append(cost)
    return torch.cat(cv, dim=1)


# ---------------------------------------------------------------------------
# Disparity decoder (outputs 1-channel disparity + zero y-channel)
# ---------------------------------------------------------------------------

class OpticalFlowDecoderDisp(nn.Module):
    """
    Replaces ``optical_flow_decoder_dc(inputs, level)`` for disparity.

    Outputs 2-channel flow where channel 0 is disparity_x and channel 1 is
    always zero. Also returns the 32-channel feature (cnv5).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.cnv1 = nn.Conv2d(in_channels,  128, 3, 1, 1)
        self.cnv2 = nn.Conv2d(128,           128, 3, 1, 1)
        self.cnv3 = nn.Conv2d(128 + 128,      96, 3, 1, 1)
        self.cnv4 = nn.Conv2d(128 + 96,       64, 3, 1, 1)
        self.cnv5 = nn.Conv2d(96 + 64,        32, 3, 1, 1)
        self.cnv6 = nn.Conv2d(64 + 32,         1, 3, 1, 1)   # 1-channel disparity_x

    def forward(self, x):
        c1 = leaky_relu(self.cnv1(x))
        c2 = leaky_relu(self.cnv2(c1))
        c3 = leaky_relu(self.cnv3(torch.cat([c1, c2], dim=1)))
        c4 = leaky_relu(self.cnv4(torch.cat([c2, c3], dim=1)))
        c5 = leaky_relu(self.cnv5(torch.cat([c3, c4], dim=1)))
        flow_x = self.cnv6(torch.cat([c4, c5], dim=1))  # no activation
        flow_y = torch.zeros_like(flow_x)
        flow = torch.cat([flow_x, flow_y], dim=1)  # (B, 2, H, W)
        return flow, c5


# ---------------------------------------------------------------------------
# Context network (dilated convolutions, 1-channel disparity output)
# ---------------------------------------------------------------------------

class ContextNetDisp(nn.Module):
    """Replaces ``context_net(inputs)`` for disparity."""

    def __init__(self, in_channels):
        super().__init__()
        self.cnv1 = nn.Conv2d(in_channels, 128, 3, 1, padding=1,  dilation=1)
        self.cnv2 = nn.Conv2d(128,         128, 3, 1, padding=2,  dilation=2)
        self.cnv3 = nn.Conv2d(128,         128, 3, 1, padding=4,  dilation=4)
        self.cnv4 = nn.Conv2d(128,          96, 3, 1, padding=8,  dilation=8)
        self.cnv5 = nn.Conv2d(96,           64, 3, 1, padding=16, dilation=16)
        self.cnv6 = nn.Conv2d(64,           32, 3, 1, padding=1,  dilation=1)
        self.cnv7 = nn.Conv2d(32,            1, 3, 1, padding=1,  dilation=1)  # 1-channel

    def forward(self, x):
        c1 = leaky_relu(self.cnv1(x))
        c2 = leaky_relu(self.cnv2(c1))
        c3 = leaky_relu(self.cnv3(c2))
        c4 = leaky_relu(self.cnv4(c3))
        c5 = leaky_relu(self.cnv5(c4))
        c6 = leaky_relu(self.cnv6(c5))
        flow_x = self.cnv7(c6)  # no activation
        flow_y = torch.zeros_like(flow_x)
        flow = torch.cat([flow_x, flow_y], dim=1)
        return flow


# ---------------------------------------------------------------------------
# construct_model_pwc_full_disp
# ---------------------------------------------------------------------------

class PWCDispDecoder(nn.Module):
    """
    Decoder half of the stereo disparity network.  Implements
    ``construct_model_pwc_full_disp(feature1, feature2, image1, neg)``.

    Separate instances are used for left->right and right->left directions
    so that they have independent weights (matching TF variable scopes
    ``left_disp`` and ``right_disp``).
    """

    def __init__(self):
        super().__init__()
        cv_ch = 9  # 2*4+1

        self.cost_vol = CostVolumeDisp(d=4)
        self.decoder6 = OpticalFlowDecoderDisp(cv_ch)
        # Level 5: cv(9) + feature(128) + flow(2)
        self.decoder5 = OpticalFlowDecoderDisp(cv_ch + 128 + 2)
        # Level 4: cv(9) + feature(96) + flow_x(1)
        self.decoder4 = OpticalFlowDecoderDisp(cv_ch + 96 + 1)
        # Level 3: cv(9) + feature(64) + flow_x(1)
        self.decoder3 = OpticalFlowDecoderDisp(cv_ch + 64 + 1)
        # Level 2: cv(9) + feature(32) + flow_x(1)
        self.decoder2 = OpticalFlowDecoderDisp(cv_ch + 32 + 1)
        # Context: flow_x(1) + feature(32)
        self.context = ContextNetDisp(1 + 32)

    def forward(self, feature1, feature2, image1, neg=False):
        """
        Args:
            feature1, feature2: tuples of 6 feature maps from FeaturePyramidDisp
            image1: (B, 3, H, W) -- used only for spatial dimensions
            neg: if True, disparity is negative (left-to-right)
        Returns:
            (disp0, disp1, disp2, disp3) -- normalised disparities at 4 scales
        """
        B, _, H, W = image1.shape
        f1_1, f1_2, f1_3, f1_4, f1_5, f1_6 = feature1
        f2_1, f2_2, f2_3, f2_4, f2_5, f2_6 = feature2

        def _clamp(flow):
            if neg:
                return -F.relu(-flow)
            else:
                return F.relu(flow)

        # --- Level 6 ---
        cv6 = self.cost_vol(f1_6, f2_6)
        flow6, _ = self.decoder6(cv6)
        flow6 = _clamp(flow6)

        # --- Level 5 ---
        H5, W5 = H // 32, W // 32
        flow6to5 = F.interpolate(
            flow6, size=(H5, W5), mode='bilinear', align_corners=False) * 2.0
        f2_5w = warp(f2_5, flow6to5)
        cv5 = self.cost_vol(f1_5, f2_5w)
        flow5, _ = self.decoder5(torch.cat([cv5, f1_5, flow6to5], dim=1))
        flow5 = flow5 + flow6to5
        flow5 = _clamp(flow5)

        # --- Level 4 ---
        H4, W4 = H // 16, W // 16
        flow5to4 = F.interpolate(
            flow5, size=(H4, W4), mode='bilinear', align_corners=False) * 2.0
        f2_4w = warp(f2_4, flow5to4)
        cv4 = self.cost_vol(f1_4, f2_4w)
        flow4, _ = self.decoder4(
            torch.cat([cv4, f1_4, flow5to4[:, 0:1, :, :]], dim=1))
        flow4 = flow4 + flow5to4
        flow4 = _clamp(flow4)

        # --- Level 3 ---
        H3, W3 = H // 8, W // 8
        flow4to3 = F.interpolate(
            flow4, size=(H3, W3), mode='bilinear', align_corners=False) * 2.0
        f2_3w = warp(f2_3, flow4to3)
        cv3 = self.cost_vol(f1_3, f2_3w)
        flow3, _ = self.decoder3(
            torch.cat([cv3, f1_3, flow4to3[:, 0:1, :, :]], dim=1))
        flow3 = flow3 + flow4to3
        flow3 = _clamp(flow3)

        # --- Level 2 ---
        H2, W2 = H // 4, W // 4
        flow3to2 = F.interpolate(
            flow3, size=(H2, W2), mode='bilinear', align_corners=False) * 2.0
        f2_2w = warp(f2_2, flow3to2)
        cv2 = self.cost_vol(f1_2, f2_2w)
        flow2_raw, f2_feat = self.decoder2(
            torch.cat([cv2, f1_2, flow3to2[:, 0:1, :, :]], dim=1))
        flow2_raw = flow2_raw + flow3to2
        flow2_raw = _clamp(flow2_raw)

        # Context refinement
        flow2 = self.context(
            torch.cat([flow2_raw[:, 0:1, :, :], f2_feat], dim=1)) + flow2_raw
        flow2 = _clamp(flow2)

        # --- Normalise to [0, 1] disparity and resize ---
        disp0 = F.interpolate(
            flow2[:, 0:1, :, :] / (W / 4.0),
            size=(H, W), mode='bilinear', align_corners=False)
        disp1 = F.interpolate(
            flow3[:, 0:1, :, :] / (W / 8.0),
            size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        disp2 = F.interpolate(
            flow4[:, 0:1, :, :] / (W / 16.0),
            size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        disp3 = F.interpolate(
            flow5[:, 0:1, :, :] / (W / 32.0),
            size=(H // 8, W // 8), mode='bilinear', align_corners=False)

        if neg:
            return -disp0, -disp1, -disp2, -disp3
        else:
            return disp0, disp1, disp2, disp3


# ---------------------------------------------------------------------------
# Top-level PWCDisp: left-right + right-left disparity
# ---------------------------------------------------------------------------

class PWCDisp(nn.Module):
    """
    Full stereo disparity model.  Computes left-to-right and right-to-left
    disparities, concatenates them, and adds ``min_eps`` for numerical safety.

    Replaces ``pwc_disp(image1, image2, feature1, feature2)`` in the original.
    """

    def __init__(self):
        super().__init__()
        self.feature_pyramid = FeaturePyramidDisp()
        self.left_disp = PWCDispDecoder()
        self.right_disp = PWCDispDecoder()

    def forward(self, image1, image2):
        """
        Args:
            image1: left image  (B, 3, H, W)
            image2: right image (B, 3, H, W)
        Returns:
            list of 4 tensors (one per scale), each (B, 2, H_s, W_s) where
            channel 0 = left disparity and channel 1 = right disparity.
            All values are positive (min_eps added).
        """
        min_eps = 1e-6

        feature1 = self.feature_pyramid(image1)
        feature2 = self.feature_pyramid(image2)

        ltr_disp = self.left_disp(feature1, feature2, image1, neg=True)
        rtl_disp = self.right_disp(feature2, feature1, image2, neg=False)

        return [
            torch.cat([ltr + min_eps, rtl + min_eps], dim=1)
            for ltr, rtl in zip(ltr_disp, rtl_disp)
        ]

    def forward_with_features(self, image1, image2, feature1, feature2):
        """Alternative entry point when features are precomputed externally."""
        min_eps = 1e-6

        ltr_disp = self.left_disp(feature1, feature2, image1, neg=True)
        rtl_disp = self.right_disp(feature2, feature1, image2, neg=False)

        return [
            torch.cat([ltr + min_eps, rtl + min_eps], dim=1)
            for ltr, rtl in zip(ltr_disp, rtl_disp)
        ]
