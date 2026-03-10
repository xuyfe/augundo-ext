"""
PWC-Net based disparity estimation network for UnOS.

Ported from the original TensorFlow pwc_disp.py.
Implements a coarse-to-fine disparity estimation pipeline using:
- 1D cost volumes (horizontal search only for stereo disparity)
- Dense decoders with skip connections
- Context network for refinement
- ReLU gating to ensure correct disparity sign

All tensors use PyTorch NCHW format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warping import backward_warp


class CostVolume1D(nn.Module):
    """1D cost volume for stereo disparity estimation.

    Computes correlation between feature1 and horizontally shifted versions
    of feature2 over a search range d. Produces 2*d+1 correlation channels.

    The cost volume searches only along the width dimension (horizontal),
    which is appropriate for rectified stereo pairs.
    """

    def __init__(self, d=4):
        """
        Args:
            d: Search range. Will search [-d, d] positions, producing
               2*d+1 = 9 output channels.
        """
        super().__init__()
        self.d = d

    def forward(self, feature1, feature2):
        """Compute 1D cost volume.

        Args:
            feature1: Left features, shape N x C x H x W.
            feature2: Right features (possibly warped), shape N x C x H x W.

        Returns:
            Cost volume, shape N x (2*d+1) x H x W.
        """
        N, C, H, W = feature1.shape
        d = self.d

        # Pad feature2 along width dimension: d on each side
        feature2_padded = F.pad(feature2, [d, d, 0, 0])  # pad W: (left, right, top, bottom)

        cost_list = []
        for i in range(2 * d + 1):
            # Extract shifted version of feature2
            shifted = feature2_padded[:, :, :, i:i + W]  # N x C x H x W
            # Inner product (correlation) normalized by channels
            cost = torch.mean(feature1 * shifted, dim=1, keepdim=True)  # N x 1 x H x W
            cost_list.append(cost)

        cost_volume = torch.cat(cost_list, dim=1)  # N x (2*d+1) x H x W
        return cost_volume


class DispDecoder(nn.Module):
    """Dense decoder for disparity estimation at a single pyramid level.

    Uses dense connections (concatenation of intermediate features) to
    produce a disparity prediction and intermediate features for the
    context network.

    Architecture:
        input -> cnv1(128) -> cnv2(128) -> cat(cnv1,cnv2) ->
        cnv3(96) -> cat(cnv2,cnv3) -> cnv4(64) -> cat(cnv3,cnv4) ->
        cnv5(32) -> cat(cnv4,cnv5) -> flow_x(1, no activation)
        flow_y = zeros

    Output is 2-channel flow (x, y) where y is always zero (horizontal only).
    """

    def __init__(self, in_channels):
        """
        Args:
            in_channels: Number of input channels (varies by pyramid level).
        """
        super().__init__()
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.cnv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.cnv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.cnv3 = nn.Conv2d(128 + 128, 96, kernel_size=3, stride=1, padding=1)
        self.cnv4 = nn.Conv2d(128 + 96, 64, kernel_size=3, stride=1, padding=1)
        self.cnv5 = nn.Conv2d(96 + 64, 32, kernel_size=3, stride=1, padding=1)
        self.flow_x = nn.Conv2d(64 + 32, 1, kernel_size=3, stride=1, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cnv1, self.cnv2, self.cnv3, self.cnv4, self.cnv5]:
            nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Flow prediction layer: zero init for stable training start
        nn.init.zeros_(self.flow_x.weight)
        nn.init.zeros_(self.flow_x.bias)

    def forward(self, x):
        """
        Args:
            x: Input features, shape N x in_channels x H x W.

        Returns:
            flow: Disparity as 2-channel flow (x, 0), shape N x 2 x H x W.
            cnv5_out: Intermediate features, shape N x 32 x H x W.
        """
        cnv1 = self.activation(self.cnv1(x))
        cnv2 = self.activation(self.cnv2(cnv1))
        cnv3 = self.activation(self.cnv3(torch.cat([cnv1, cnv2], dim=1)))
        cnv4 = self.activation(self.cnv4(torch.cat([cnv2, cnv3], dim=1)))
        cnv5 = self.activation(self.cnv5(torch.cat([cnv3, cnv4], dim=1)))
        flow_x = self.flow_x(torch.cat([cnv4, cnv5], dim=1))  # N x 1 x H x W

        # Disparity is horizontal only: flow_y = 0
        flow_y = torch.zeros_like(flow_x)
        flow = torch.cat([flow_x, flow_y], dim=1)  # N x 2 x H x W

        return flow, cnv5


class DispContextNet(nn.Module):
    """Context network for disparity refinement.

    Uses dilated convolutions to aggregate context over a large receptive
    field for refining the initial disparity estimate.

    Architecture:
        cnv1: 128, dilation=1
        cnv2: 128, dilation=2
        cnv3: 128, dilation=4
        cnv4: 96,  dilation=8
        cnv5: 64,  dilation=16
        cnv6: 32,  dilation=1
        flow_x: 1, dilation=1 (no activation)
    """

    def __init__(self, in_channels):
        """
        Args:
            in_channels: Number of input channels.
        """
        super().__init__()
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.cnv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.cnv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.cnv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.cnv4 = nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.cnv5 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.cnv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.flow_x = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, dilation=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cnv1, self.cnv2, self.cnv3, self.cnv4, self.cnv5, self.cnv6]:
            nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.flow_x.weight)
        nn.init.zeros_(self.flow_x.bias)

    def forward(self, x):
        """
        Args:
            x: Input features, shape N x in_channels x H x W.

        Returns:
            Residual disparity as 2-channel flow (x, 0), shape N x 2 x H x W.
        """
        out = self.activation(self.cnv1(x))
        out = self.activation(self.cnv2(out))
        out = self.activation(self.cnv3(out))
        out = self.activation(self.cnv4(out))
        out = self.activation(self.cnv5(out))
        out = self.activation(self.cnv6(out))
        flow_x = self.flow_x(out)  # N x 1 x H x W

        flow_y = torch.zeros_like(flow_x)
        flow = torch.cat([flow_x, flow_y], dim=1)  # N x 2 x H x W

        return flow


def construct_model_pwc_full_disp(feature_list1, feature_list2, image1, decoders,
                                  context_net, neg=False):
    """Coarse-to-fine disparity estimation.

    Iterates from the coarsest feature level (level 6) to the finest
    (level 2), progressively refining the disparity estimate.

    At each level:
    1. Upsample previous flow by 2x (and scale values by 2)
    2. Warp feature2 using current flow estimate
    3. Compute 1D cost volume between feature1 and warped feature2
    4. Decode residual flow from cost volume + features + current flow
    5. Add residual to current flow
    6. Apply ReLU gating (positive for left disp, negative for right disp)

    At the finest level (level 2), apply context network refinement.

    Args:
        feature_list1: Tuple of 6 feature maps from image 1.
        feature_list2: Tuple of 6 feature maps from image 2.
        image1: Original image for loss computation, N x 3 x H x W.
        decoders: List of 5 DispDecoder modules (levels 6 to 2).
        context_net: DispContextNet module.
        neg: If False, disparity is positive (left image). If True, negative (right).

    Returns:
        List of 4 disparity maps at scales [1/4, 1/8, 1/16, 1/32] of input,
        each normalized by image width.
    """
    _, _, H, W = image1.shape
    cost_volume = CostVolume1D(d=4).to(image1.device)

    flow_list = []

    # Level 6 (coarsest): no previous flow
    feat1_6 = feature_list1[5]
    feat2_6 = feature_list2[5]
    cv6 = cost_volume(feat1_6, feat2_6)
    decoder_input = torch.cat([cv6, feat1_6], dim=1)
    flow6, cnv5_6 = decoders[0](decoder_input)

    # Apply ReLU gating
    if not neg:
        flow6 = F.relu(flow6)
    else:
        flow6 = -F.relu(-flow6)

    # Levels 5 to 2
    for level in range(4, 0, -1):  # level index 4,3,2,1 -> feature levels 5,4,3,2
        feat1 = feature_list1[level]
        feat2 = feature_list2[level]

        # Upsample previous flow
        prev_flow = flow6 if level == 4 else flow_prev
        flow_up = F.interpolate(prev_flow, size=feat1.shape[2:], mode='bilinear',
                                align_corners=True) * 2.0

        # Warp feature2
        feat2_warped = backward_warp(feat2, flow_up)

        # Cost volume
        cv = cost_volume(feat1, feat2_warped)

        # Decode
        decoder_idx = 5 - level  # maps level 4->1, 3->2, 2->3, 1->4
        decoder_input = torch.cat([cv, feat1, flow_up], dim=1)
        flow_res, cnv5 = decoders[decoder_idx](decoder_input)

        flow_prev = flow_up + flow_res

        # ReLU gating
        if not neg:
            flow_prev = F.relu(flow_prev)
        else:
            flow_prev = -F.relu(-flow_prev)

        # Store disparity maps at levels 4,3,2,1 (scales 1/32 to 1/4)
        # We want maps at 4 scales from coarse to fine
        if level <= 4:
            flow_list.append(flow_prev)

    # Context network refinement at finest level
    context_input = torch.cat([cnv5, flow_prev], dim=1)
    flow_res_ctx = context_net(context_input)
    flow_final = flow_prev + flow_res_ctx

    if not neg:
        flow_final = F.relu(flow_final)
    else:
        flow_final = -F.relu(-flow_final)

    # Replace the finest level with context-refined version
    flow_list[-1] = flow_final

    # Normalize disparity maps by width
    # flow_list has maps from coarse to fine: [1/32, 1/16, 1/8, 1/4]
    # Reverse to get [1/4, 1/8, 1/16, 1/32] ordering
    flow_list = flow_list[::-1]

    disp_list = []
    for flow in flow_list:
        # Only x-component matters, normalize by width
        disp = flow[:, 0:1, :, :] / W  # N x 1 x H_level x W_level
        disp_list.append(disp)

    return disp_list


class PWCDisp(nn.Module):
    """PWC-Net based stereo disparity estimation.

    Estimates left and right disparities using two sub-networks with shared
    feature pyramids. Output is concatenated left+right disparities at
    4 pyramid scales.

    The left disparity is positive (left-to-right displacement) and the
    right disparity is negative (right-to-left displacement).
    """

    def __init__(self):
        super().__init__()
        # Cost volume output channels
        cv_channels = 9  # 2*4+1

        # Decoders for left disparity (5 levels: 6 to 2)
        # Level 6: input = cv + feat channels
        # Level 6: feat=192, cv=9 -> 201
        self.left_decoder_6 = DispDecoder(cv_channels + 192)
        # Levels 5-2: input = cv + feat + flow(2)
        self.left_decoder_5 = DispDecoder(cv_channels + 128 + 2)
        self.left_decoder_4 = DispDecoder(cv_channels + 96 + 2)
        self.left_decoder_3 = DispDecoder(cv_channels + 64 + 2)
        self.left_decoder_2 = DispDecoder(cv_channels + 32 + 2)
        # Context net: input = cnv5(32) + flow(2)
        self.left_context = DispContextNet(32 + 2)

        # Decoders for right disparity
        self.right_decoder_6 = DispDecoder(cv_channels + 192)
        self.right_decoder_5 = DispDecoder(cv_channels + 128 + 2)
        self.right_decoder_4 = DispDecoder(cv_channels + 96 + 2)
        self.right_decoder_3 = DispDecoder(cv_channels + 64 + 2)
        self.right_decoder_2 = DispDecoder(cv_channels + 32 + 2)
        self.right_context = DispContextNet(32 + 2)

    def forward(self, left_features, right_features, left_image, right_image, eps=1e-6):
        """Estimate left and right disparities.

        Args:
            left_features: Tuple of 6 feature maps from left image.
            right_features: Tuple of 6 feature maps from right image.
            left_image: Left image, N x 3 x H x W.
            right_image: Right image, N x 3 x H x W.
            eps: Small constant added to disparities for numerical stability.

        Returns:
            List of 4 tensors at scales [1/4, 1/8, 1/16, 1/32].
            Each tensor has shape N x 2 x H_s x W_s, where channel 0 is
            left disparity + eps and channel 1 is right disparity + eps.
        """
        left_decoders = [self.left_decoder_6, self.left_decoder_5,
                         self.left_decoder_4, self.left_decoder_3, self.left_decoder_2]
        right_decoders = [self.right_decoder_6, self.right_decoder_5,
                          self.right_decoder_4, self.right_decoder_3, self.right_decoder_2]

        # Left disparity: left->right, positive displacement
        left_disps = construct_model_pwc_full_disp(
            left_features, right_features, left_image,
            left_decoders, self.left_context, neg=False
        )

        # Right disparity: right->left, negative displacement
        right_disps = construct_model_pwc_full_disp(
            right_features, left_features, right_image,
            right_decoders, self.right_context, neg=True
        )

        # Concatenate: [left_disp + eps, right_disp + eps]
        disp_list = []
        for ld, rd in zip(left_disps, right_disps):
            combined = torch.cat([ld + eps, rd + eps], dim=1)  # N x 2 x H x W
            disp_list.append(combined)

        return disp_list
