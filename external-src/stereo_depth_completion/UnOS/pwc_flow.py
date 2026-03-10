"""
PWC-Net based optical flow estimation network for UnOS.

Ported from the original TensorFlow pwc_flow.py.
Similar architecture to pwc_disp.py but uses 2D cost volumes (search in
both horizontal and vertical directions) and outputs 2-channel flow
without ReLU gating (flow can be in any direction).

All tensors use PyTorch NCHW format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warping import backward_warp


class CostVolume2D(nn.Module):
    """2D cost volume for optical flow estimation.

    Computes correlation between feature1 and shifted versions of feature2
    over a 2D search range d in both horizontal and vertical directions.
    Produces (2*d+1)^2 correlation channels.
    """

    def __init__(self, d=4):
        """
        Args:
            d: Search range in each direction. Produces (2*d+1)^2 = 81
               output channels for d=4.
        """
        super().__init__()
        self.d = d

    def forward(self, feature1, feature2):
        """Compute 2D cost volume.

        Args:
            feature1: Source features, shape N x C x H x W.
            feature2: Target features, shape N x C x H x W.

        Returns:
            Cost volume, shape N x (2*d+1)^2 x H x W.
        """
        N, C, H, W = feature1.shape
        d = self.d

        # Pad feature2 along both H and W dimensions
        feature2_padded = F.pad(feature2, [d, d, d, d])  # pad (left, right, top, bottom)

        cost_list = []
        for dy in range(2 * d + 1):
            for dx in range(2 * d + 1):
                shifted = feature2_padded[:, :, dy:dy + H, dx:dx + W]
                cost = torch.mean(feature1 * shifted, dim=1, keepdim=True)
                cost_list.append(cost)

        cost_volume = torch.cat(cost_list, dim=1)  # N x (2*d+1)^2 x H x W
        return cost_volume


class FlowDecoder(nn.Module):
    """Dense decoder for optical flow estimation at a single pyramid level.

    Same architecture as DispDecoder but outputs 2 channels (x and y flow).

    Architecture:
        input -> cnv1(128) -> cnv2(128) -> cat(cnv1,cnv2) ->
        cnv3(96) -> cat(cnv2,cnv3) -> cnv4(64) -> cat(cnv3,cnv4) ->
        cnv5(32) -> cat(cnv4,cnv5) -> flow(2)
    """

    def __init__(self, in_channels):
        """
        Args:
            in_channels: Number of input channels.
        """
        super().__init__()
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.cnv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.cnv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.cnv3 = nn.Conv2d(128 + 128, 96, kernel_size=3, stride=1, padding=1)
        self.cnv4 = nn.Conv2d(128 + 96, 64, kernel_size=3, stride=1, padding=1)
        self.cnv5 = nn.Conv2d(96 + 64, 32, kernel_size=3, stride=1, padding=1)
        self.flow = nn.Conv2d(64 + 32, 2, kernel_size=3, stride=1, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cnv1, self.cnv2, self.cnv3, self.cnv4, self.cnv5]:
            nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, x):
        """
        Args:
            x: Input features, shape N x in_channels x H x W.

        Returns:
            flow: Optical flow, shape N x 2 x H x W.
            cnv5_out: Intermediate features, shape N x 32 x H x W.
        """
        cnv1 = self.activation(self.cnv1(x))
        cnv2 = self.activation(self.cnv2(cnv1))
        cnv3 = self.activation(self.cnv3(torch.cat([cnv1, cnv2], dim=1)))
        cnv4 = self.activation(self.cnv4(torch.cat([cnv2, cnv3], dim=1)))
        cnv5 = self.activation(self.cnv5(torch.cat([cnv3, cnv4], dim=1)))
        flow = self.flow(torch.cat([cnv4, cnv5], dim=1))

        return flow, cnv5


class FlowContextNet(nn.Module):
    """Context network for optical flow refinement.

    Uses dilated convolutions to aggregate context for refining flow.
    Same architecture as DispContextNet but outputs 2 channels.

    Architecture:
        cnv1: 128, dilation=1
        cnv2: 128, dilation=2
        cnv3: 128, dilation=4
        cnv4: 96,  dilation=8
        cnv5: 64,  dilation=16
        cnv6: 32,  dilation=1
        flow: 2,   dilation=1 (no activation)
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
        self.flow = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, dilation=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cnv1, self.cnv2, self.cnv3, self.cnv4, self.cnv5, self.cnv6]:
            nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, x):
        """
        Args:
            x: Input features, shape N x in_channels x H x W.

        Returns:
            Residual flow, shape N x 2 x H x W.
        """
        out = self.activation(self.cnv1(x))
        out = self.activation(self.cnv2(out))
        out = self.activation(self.cnv3(out))
        out = self.activation(self.cnv4(out))
        out = self.activation(self.cnv5(out))
        out = self.activation(self.cnv6(out))
        flow = self.flow(out)

        return flow


def construct_model_pwc_full(feature_list1, feature_list2, image1, image2,
                             decoders, context_net):
    """Coarse-to-fine optical flow estimation.

    Same structure as disparity estimation but:
    - Uses 2D cost volumes (search in both x and y)
    - No ReLU gating (flow can be in any direction)
    - Outputs 2-channel flow (x, y)

    Args:
        feature_list1: Tuple of 6 feature maps from image 1.
        feature_list2: Tuple of 6 feature maps from image 2.
        image1: Source image, N x 3 x H x W.
        image2: Target image, N x 3 x H x W.
        decoders: List of 5 FlowDecoder modules (levels 6 to 2).
        context_net: FlowContextNet module.

    Returns:
        List of 4 flow maps at scales [1/4, 1/8, 1/16, 1/32] of input.
        Each has shape N x 2 x H_s x W_s.
    """
    cost_volume = CostVolume2D(d=4).to(image1.device)
    cv_channels = 81  # (2*4+1)^2

    flow_list = []

    # Level 6 (coarsest)
    feat1_6 = feature_list1[5]
    feat2_6 = feature_list2[5]
    cv6 = cost_volume(feat1_6, feat2_6)
    decoder_input = torch.cat([cv6, feat1_6], dim=1)
    flow6, cnv5_6 = decoders[0](decoder_input)

    # Levels 5 to 2
    for level in range(4, 0, -1):
        feat1 = feature_list1[level]
        feat2 = feature_list2[level]

        prev_flow = flow6 if level == 4 else flow_prev
        flow_up = F.interpolate(prev_flow, size=feat1.shape[2:], mode='bilinear',
                                align_corners=True) * 2.0

        feat2_warped = backward_warp(feat2, flow_up)

        cv = cost_volume(feat1, feat2_warped)

        decoder_idx = 5 - level
        decoder_input = torch.cat([cv, feat1, flow_up], dim=1)
        flow_res, cnv5 = decoders[decoder_idx](decoder_input)

        flow_prev = flow_up + flow_res

        if level <= 4:
            flow_list.append(flow_prev)

    # Context network refinement at finest level
    context_input = torch.cat([cnv5, flow_prev], dim=1)
    flow_res_ctx = context_net(context_input)
    flow_final = flow_prev + flow_res_ctx

    flow_list[-1] = flow_final

    # Reverse to get [finest, ..., coarsest] ordering
    flow_list = flow_list[::-1]

    return flow_list


class PWCFlow(nn.Module):
    """PWC-Net based optical flow estimation.

    Estimates bidirectional optical flow (forward and backward) between
    two frames using a coarse-to-fine approach with 2D cost volumes.
    """

    def __init__(self):
        super().__init__()
        cv_channels = 81  # (2*4+1)^2

        # Forward flow decoders (image1 -> image2)
        self.fwd_decoder_6 = FlowDecoder(cv_channels + 192)
        self.fwd_decoder_5 = FlowDecoder(cv_channels + 128 + 2)
        self.fwd_decoder_4 = FlowDecoder(cv_channels + 96 + 2)
        self.fwd_decoder_3 = FlowDecoder(cv_channels + 64 + 2)
        self.fwd_decoder_2 = FlowDecoder(cv_channels + 32 + 2)
        self.fwd_context = FlowContextNet(32 + 2)

        # Backward flow decoders (image2 -> image1)
        self.bwd_decoder_6 = FlowDecoder(cv_channels + 192)
        self.bwd_decoder_5 = FlowDecoder(cv_channels + 128 + 2)
        self.bwd_decoder_4 = FlowDecoder(cv_channels + 96 + 2)
        self.bwd_decoder_3 = FlowDecoder(cv_channels + 64 + 2)
        self.bwd_decoder_2 = FlowDecoder(cv_channels + 32 + 2)
        self.bwd_context = FlowContextNet(32 + 2)

    def forward(self, features1, features2, image1, image2):
        """Estimate bidirectional optical flow.

        Args:
            features1: Tuple of 6 feature maps from image 1.
            features2: Tuple of 6 feature maps from image 2.
            image1: First image, N x 3 x H x W.
            image2: Second image, N x 3 x H x W.

        Returns:
            fwd_flows: List of 4 forward flow maps [1/4, 1/8, 1/16, 1/32].
            bwd_flows: List of 4 backward flow maps [1/4, 1/8, 1/16, 1/32].
        """
        fwd_decoders = [self.fwd_decoder_6, self.fwd_decoder_5,
                        self.fwd_decoder_4, self.fwd_decoder_3, self.fwd_decoder_2]
        bwd_decoders = [self.bwd_decoder_6, self.bwd_decoder_5,
                        self.bwd_decoder_4, self.bwd_decoder_3, self.bwd_decoder_2]

        fwd_flows = construct_model_pwc_full(
            features1, features2, image1, image2,
            fwd_decoders, self.fwd_context
        )

        bwd_flows = construct_model_pwc_full(
            features2, features1, image2, image1,
            bwd_decoders, self.bwd_context
        )

        return fwd_flows, bwd_flows
