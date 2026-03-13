"""
PWC optical flow network -- PyTorch reimplementation of UnOS / UnDepthflow.
Mirrors the TensorFlow original in nets/pwc_flow.py.
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
    # Base grid
    xx = torch.arange(0, W, device=x.device, dtype=x.dtype).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device, dtype=x.dtype).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)  # (B, 2, H, W)
    vgrid = grid + flow
    # Normalise to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # (B, H, W, 2)
    output = F.grid_sample(x, vgrid, align_corners=True)
    # Occlusion mask -- positions that land outside valid range
    mask = torch.ones_like(x)
    mask = F.grid_sample(mask, vgrid, align_corners=True)
    mask = (mask >= 0.9999).float()
    return output * mask


# ---------------------------------------------------------------------------
# Feature pyramid
# ---------------------------------------------------------------------------

class FeaturePyramidFlow(nn.Module):
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
# Cost volume (2-D search, d=4 -> (2d+1)^2 = 81 channels)
# ---------------------------------------------------------------------------

class CostVolume(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.d = d

    def forward(self, feature1, feature2):
        """
        feature1, feature2: (B, C, H, W)
        returns: (B, (2d+1)^2, H, W)
        """
        d = self.d
        B, C, H, W = feature1.size()
        feature2_pad = F.pad(feature2, [d, d, d, d], mode='constant', value=0)
        cv = []
        for i in range(2 * d + 1):
            for j in range(2 * d + 1):
                cost = torch.mean(
                    feature1 * feature2_pad[:, :, i:i + H, j:j + W],
                    dim=1, keepdim=True)
                cv.append(cost)
        return torch.cat(cv, dim=1)


def cost_volume(feature1, feature2, d=4):
    """Functional wrapper matching the original ``cost_volumn()`` signature."""
    B, C, H, W = feature1.size()
    feature2_pad = F.pad(feature2, [d, d, d, d], mode='constant', value=0)
    cv = []
    for i in range(2 * d + 1):
        for j in range(2 * d + 1):
            cost = torch.mean(
                feature1 * feature2_pad[:, :, i:i + H, j:j + W],
                dim=1, keepdim=True)
            cv.append(cost)
    return torch.cat(cv, dim=1)


# ---------------------------------------------------------------------------
# Optical flow decoder (dense connections)
# ---------------------------------------------------------------------------

class OpticalFlowDecoder(nn.Module):
    """
    Replaces ``optical_flow_decoder_dc(inputs, level)``.

    Dense-connection decoder that produces a 2-channel flow prediction and
    a 32-channel feature map (cnv5).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.cnv1 = nn.Conv2d(in_channels,  128, 3, 1, 1)
        self.cnv2 = nn.Conv2d(128,           128, 3, 1, 1)
        self.cnv3 = nn.Conv2d(128 + 128,      96, 3, 1, 1)
        self.cnv4 = nn.Conv2d(128 + 96,       64, 3, 1, 1)
        self.cnv5 = nn.Conv2d(96 + 64,        32, 3, 1, 1)
        self.cnv6 = nn.Conv2d(64 + 32,         2, 3, 1, 1)   # 2-channel flow

    def forward(self, x):
        c1 = leaky_relu(self.cnv1(x))
        c2 = leaky_relu(self.cnv2(c1))
        c3 = leaky_relu(self.cnv3(torch.cat([c1, c2], dim=1)))
        c4 = leaky_relu(self.cnv4(torch.cat([c2, c3], dim=1)))
        c5 = leaky_relu(self.cnv5(torch.cat([c3, c4], dim=1)))
        flow = self.cnv6(torch.cat([c4, c5], dim=1))  # no activation
        return flow, c5


# ---------------------------------------------------------------------------
# Context network (dilated convolutions)
# ---------------------------------------------------------------------------

class ContextNet(nn.Module):
    """Replaces ``context_net(inputs)``."""

    def __init__(self, in_channels):
        super().__init__()
        self.cnv1 = nn.Conv2d(in_channels, 128, 3, 1, padding=1,  dilation=1)
        self.cnv2 = nn.Conv2d(128,         128, 3, 1, padding=2,  dilation=2)
        self.cnv3 = nn.Conv2d(128,         128, 3, 1, padding=4,  dilation=4)
        self.cnv4 = nn.Conv2d(128,          96, 3, 1, padding=8,  dilation=8)
        self.cnv5 = nn.Conv2d(96,           64, 3, 1, padding=16, dilation=16)
        self.cnv6 = nn.Conv2d(64,           32, 3, 1, padding=1,  dilation=1)
        self.cnv7 = nn.Conv2d(32,            2, 3, 1, padding=1,  dilation=1)  # 2-channel

    def forward(self, x):
        c1 = leaky_relu(self.cnv1(x))
        c2 = leaky_relu(self.cnv2(c1))
        c3 = leaky_relu(self.cnv3(c2))
        c4 = leaky_relu(self.cnv4(c3))
        c5 = leaky_relu(self.cnv5(c4))
        c6 = leaky_relu(self.cnv6(c5))
        flow = self.cnv7(c6)  # no activation
        return flow


# ---------------------------------------------------------------------------
# Full PWC-Flow model
# ---------------------------------------------------------------------------

class PWCFlow(nn.Module):
    """
    Full PWC optical flow network.

    ``construct_model_pwc_full(image1, image2, feature1, feature2)``
    is implemented as the ``forward`` method.
    """

    def __init__(self):
        super().__init__()
        self.feature_pyramid = FeaturePyramidFlow()
        self.cost_vol = CostVolume(d=4)

        cv_ch = 81  # (2*4+1)^2

        # Level 6: cost volume only
        self.decoder6 = OpticalFlowDecoder(cv_ch)
        # Level 5: cv + feature1_5 (128) + upsampled flow (2)
        self.decoder5 = OpticalFlowDecoder(cv_ch + 128 + 2)
        # Level 4: cv + feature1_4 (96) + upsampled flow (2)
        self.decoder4 = OpticalFlowDecoder(cv_ch + 96 + 2)
        # Level 3: cv + feature1_3 (64) + upsampled flow (2)
        self.decoder3 = OpticalFlowDecoder(cv_ch + 64 + 2)
        # Level 2: cv + feature1_2 (32) + upsampled flow (2)
        self.decoder2 = OpticalFlowDecoder(cv_ch + 32 + 2)

        # Context refinement: flow (2) + decoder feature (32)
        self.context = ContextNet(2 + 32)

    def construct_model_pwc_full(self, image1, image2, feature1, feature2):
        """
        Args:
            image1, image2: (B, 3, H, W)
            feature1, feature2: each a tuple of 6 feature tensors from
                FeaturePyramidFlow
        Returns:
            (flow0, flow1, flow2, flow3) -- flows at 4 scales, each in
            pixel units at the respective target resolution.
        """
        B, _, H, W = image1.shape
        f1_1, f1_2, f1_3, f1_4, f1_5, f1_6 = feature1
        f2_1, f2_2, f2_3, f2_4, f2_5, f2_6 = feature2

        # --- Level 6 ---
        cv6 = self.cost_vol(f1_6, f2_6)
        flow6, _ = self.decoder6(cv6)

        # --- Level 5 ---
        H5, W5 = H // 32, W // 32
        flow6to5 = F.interpolate(
            flow6, size=(H5, W5), mode='bilinear', align_corners=False) * 2.0
        f2_5w = warp(f2_5, flow6to5)
        cv5 = self.cost_vol(f1_5, f2_5w)
        flow5, _ = self.decoder5(torch.cat([cv5, f1_5, flow6to5], dim=1))
        flow5 = flow5 + flow6to5

        # --- Level 4 ---
        H4, W4 = H // 16, W // 16
        flow5to4 = F.interpolate(
            flow5, size=(H4, W4), mode='bilinear', align_corners=False) * 2.0
        f2_4w = warp(f2_4, flow5to4)
        cv4 = self.cost_vol(f1_4, f2_4w)
        flow4, _ = self.decoder4(torch.cat([cv4, f1_4, flow5to4], dim=1))
        flow4 = flow4 + flow5to4

        # --- Level 3 ---
        H3, W3 = H // 8, W // 8
        flow4to3 = F.interpolate(
            flow4, size=(H3, W3), mode='bilinear', align_corners=False) * 2.0
        f2_3w = warp(f2_3, flow4to3)
        cv3 = self.cost_vol(f1_3, f2_3w)
        flow3, _ = self.decoder3(torch.cat([cv3, f1_3, flow4to3], dim=1))
        flow3 = flow3 + flow4to3

        # --- Level 2 ---
        H2, W2 = H // 4, W // 4
        flow3to2 = F.interpolate(
            flow3, size=(H2, W2), mode='bilinear', align_corners=False) * 2.0
        f2_2w = warp(f2_2, flow3to2)
        cv2 = self.cost_vol(f1_2, f2_2w)
        flow2_raw, f2_feat = self.decoder2(
            torch.cat([cv2, f1_2, flow3to2], dim=1))
        flow2_raw = flow2_raw + flow3to2

        # Context refinement
        flow2 = self.context(torch.cat([flow2_raw, f2_feat], dim=1)) + flow2_raw

        # --- Up-sample to target scales (multiply by 4 to convert to full-res pixels) ---
        flow0_enlarge = F.interpolate(
            flow2 * 4.0, size=(H, W), mode='bilinear', align_corners=False)
        flow1_enlarge = F.interpolate(
            flow3 * 4.0, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        flow2_enlarge = F.interpolate(
            flow4 * 4.0, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        flow3_enlarge = F.interpolate(
            flow5 * 4.0, size=(H // 8, W // 8), mode='bilinear', align_corners=False)

        return flow0_enlarge, flow1_enlarge, flow2_enlarge, flow3_enlarge

    def forward(self, image1, image2):
        """Convenience: extract features then compute flow."""
        feature1 = self.feature_pyramid(image1)
        feature2 = self.feature_pyramid(image2)
        return self.construct_model_pwc_full(image1, image2, feature1, feature2)
