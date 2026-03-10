"""
6-DoF camera pose estimation network for UnOS.

Ported from the original TensorFlow implementation.
Takes a pair of images concatenated along the channel dimension and
predicts the relative camera pose (3 translation + 3 rotation parameters).

All tensors use PyTorch NCHW format.
"""

import torch
import torch.nn as nn


class PoseNet(nn.Module):
    """Camera pose estimation network.

    Takes two concatenated images (6 channels) and predicts the relative
    6-DoF camera pose between them. The network uses a series of strided
    convolutions to progressively reduce spatial resolution, followed by
    global average pooling and a final 1x1 convolution.

    The rotation components are scaled by 0.01 for stable training.

    Architecture:
        cnv1:  6  -> 16,  k=7, s=2   + cnv1b: 16  -> 16,  k=7, s=1
        cnv2:  16 -> 32,  k=5, s=2   + cnv2b: 32  -> 32,  k=5, s=1
        cnv3:  32 -> 64,  k=3, s=2   + cnv3b: 64  -> 64,  k=3, s=1
        cnv4:  64 -> 128, k=3, s=2   + cnv4b: 128 -> 128, k=3, s=1
        cnv5:  128-> 256, k=3, s=2   + cnv5b: 256 -> 256, k=3, s=1
        cnv6:  256-> 256, k=3, s=2   + cnv6b: 256 -> 256, k=3, s=1
        cnv7:  256-> 256, k=3, s=2   + cnv7b: 256 -> 256, k=3, s=1
        pred:  256-> 6,   k=1, s=1   (no activation)
        Global average pool -> [B, 6]
        Output: [translation(3), 0.01 * rotation(3)]

    All convolutions use ReLU activation (no batch normalization).
    """

    def __init__(self):
        super().__init__()

        # Build conv layers: (in_ch, out_ch, kernel_size, stride)
        layers_config = [
            # (in, out, kernel, stride)
            (6, 16, 7, 2),    # cnv1
            (16, 16, 7, 1),   # cnv1b
            (16, 32, 5, 2),   # cnv2
            (32, 32, 5, 1),   # cnv2b
            (32, 64, 3, 2),   # cnv3
            (64, 64, 3, 1),   # cnv3b
            (64, 128, 3, 2),  # cnv4
            (128, 128, 3, 1), # cnv4b
            (128, 256, 3, 2), # cnv5
            (256, 256, 3, 1), # cnv5b
            (256, 256, 3, 2), # cnv6
            (256, 256, 3, 1), # cnv6b
            (256, 256, 3, 2), # cnv7
            (256, 256, 3, 1), # cnv7b
        ]

        self.convs = nn.ModuleList()
        for in_ch, out_ch, k, s in layers_config:
            # Compute padding for 'SAME' behavior
            p = k // 2
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))

        self.relu = nn.ReLU(inplace=True)

        # Prediction layer: 1x1 conv, no activation
        self.pred = nn.Conv2d(256, 6, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        for m in self.convs:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Small init for prediction layer
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)

    def forward(self, image1, image2):
        """Estimate relative camera pose between two images.

        Args:
            image1: First image, shape N x 3 x H x W.
            image2: Second image, shape N x 3 x H x W.

        Returns:
            pose: 6-DoF pose vector [tx, ty, tz, 0.01*rx, 0.01*ry, 0.01*rz],
                  shape N x 6.
        """
        # Concatenate images along channel dimension
        x = torch.cat([image1, image2], dim=1)  # N x 6 x H x W

        # Apply conv layers with ReLU
        for conv in self.convs:
            x = self.relu(conv(x))

        # Prediction (no activation)
        x = self.pred(x)  # N x 6 x H' x W'

        # Global average pooling
        pose = x.mean(dim=[2, 3])  # N x 6

        # Scale rotation components by 0.01
        pose_final = torch.cat([
            pose[:, :3],          # translation (tx, ty, tz)
            0.01 * pose[:, 3:6],  # rotation (rx, ry, rz) scaled
        ], dim=1)

        return pose_final
