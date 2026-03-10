"""
Feature pyramid encoder networks for UnOS stereo depth estimation.

Ported from the original TensorFlow implementation (pwc_disp.py and pwc_flow.py).
Two separate feature pyramid networks with identical architecture but independent
weights: one for disparity estimation, one for optical flow estimation.

Architecture: 12 convolutional layers producing 6 feature levels at
progressively lower resolutions (stride-2 downsampling every other layer).

All tensors use PyTorch NCHW format.
"""

import torch
import torch.nn as nn


class FeaturePyramidDisp(nn.Module):
    """Feature pyramid encoder for disparity estimation.

    Produces 6 feature levels from an input image, each at half the resolution
    of the previous level. Uses LeakyReLU(0.1) activation.

    Architecture:
        cnv1:  3 -> 16,  stride=2
        cnv2:  16 -> 16, stride=1
        cnv3:  16 -> 32, stride=2
        cnv4:  32 -> 32, stride=1
        cnv5:  32 -> 64, stride=2
        cnv6:  64 -> 64, stride=1
        cnv7:  64 -> 96, stride=2
        cnv8:  96 -> 96, stride=1
        cnv9:  96 -> 128, stride=2
        cnv10: 128 -> 128, stride=1
        cnv11: 128 -> 192, stride=2
        cnv12: 192 -> 192, stride=1

    Returns feature maps from layers 2, 4, 6, 8, 10, 12 (6 levels).
    L2 regularization is handled externally via optimizer weight_decay.
    """

    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        # Level 1: 1/2 resolution
        self.cnv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.cnv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        # Level 2: 1/4 resolution
        self.cnv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.cnv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # Level 3: 1/8 resolution
        self.cnv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.cnv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Level 4: 1/16 resolution
        self.cnv7 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        self.cnv8 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)

        # Level 5: 1/32 resolution
        self.cnv9 = nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1)
        self.cnv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Level 6: 1/64 resolution
        self.cnv11 = nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1)
        self.cnv12 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Extract multi-scale features.

        Args:
            x: Input image, shape N x 3 x H x W.

        Returns:
            Tuple of 6 feature maps (cnv2, cnv4, cnv6, cnv8, cnv10, cnv12)
            at resolutions 1/2, 1/4, 1/8, 1/16, 1/32, 1/64 of input.
        """
        cnv1 = self.activation(self.cnv1(x))
        cnv2 = self.activation(self.cnv2(cnv1))

        cnv3 = self.activation(self.cnv3(cnv2))
        cnv4 = self.activation(self.cnv4(cnv3))

        cnv5 = self.activation(self.cnv5(cnv4))
        cnv6 = self.activation(self.cnv6(cnv5))

        cnv7 = self.activation(self.cnv7(cnv6))
        cnv8 = self.activation(self.cnv8(cnv7))

        cnv9 = self.activation(self.cnv9(cnv8))
        cnv10 = self.activation(self.cnv10(cnv9))

        cnv11 = self.activation(self.cnv11(cnv10))
        cnv12 = self.activation(self.cnv12(cnv11))

        return (cnv2, cnv4, cnv6, cnv8, cnv10, cnv12)


class FeaturePyramidFlow(nn.Module):
    """Feature pyramid encoder for optical flow estimation.

    Identical architecture to FeaturePyramidDisp but with separate
    (independent) weights. This separation allows the disparity and flow
    encoders to specialize for their respective tasks.

    See FeaturePyramidDisp for architecture details.
    """

    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        # Level 1: 1/2 resolution
        self.cnv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.cnv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        # Level 2: 1/4 resolution
        self.cnv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.cnv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # Level 3: 1/8 resolution
        self.cnv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.cnv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Level 4: 1/16 resolution
        self.cnv7 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        self.cnv8 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)

        # Level 5: 1/32 resolution
        self.cnv9 = nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1)
        self.cnv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Level 6: 1/64 resolution
        self.cnv11 = nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1)
        self.cnv12 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Extract multi-scale features.

        Args:
            x: Input image, shape N x 3 x H x W.

        Returns:
            Tuple of 6 feature maps (cnv2, cnv4, cnv6, cnv8, cnv10, cnv12)
            at resolutions 1/2, 1/4, 1/8, 1/16, 1/32, 1/64 of input.
        """
        cnv1 = self.activation(self.cnv1(x))
        cnv2 = self.activation(self.cnv2(cnv1))

        cnv3 = self.activation(self.cnv3(cnv2))
        cnv4 = self.activation(self.cnv4(cnv3))

        cnv5 = self.activation(self.cnv5(cnv4))
        cnv6 = self.activation(self.cnv6(cnv5))

        cnv7 = self.activation(self.cnv7(cnv6))
        cnv8 = self.activation(self.cnv8(cnv7))

        cnv9 = self.activation(self.cnv9(cnv8))
        cnv10 = self.activation(self.cnv10(cnv9))

        cnv11 = self.activation(self.cnv11(cnv10))
        cnv12 = self.activation(self.cnv12(cnv11))

        return (cnv2, cnv4, cnv6, cnv8, cnv10, cnv12)
