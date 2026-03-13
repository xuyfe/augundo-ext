"""
Camera pose estimation network -- PyTorch reimplementation of UnOS / UnDepthflow.
Mirrors the TensorFlow original in nets/pose_net.py.
Adopted from https://github.com/tinghuiz/SfMLearner
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseExpNet(nn.Module):
    """
    Pose estimation CNN.

    Input : concatenation of target + source images -> (B, 6, H, W)
    Output: (B, 6) pose vector [tx, ty, tz, rx, ry, rz]
            with translations (first 3) scaled by 0.01.
    """

    def __init__(self):
        super().__init__()
        # Shared encoder
        self.cnv1  = nn.Conv2d(6,   16,  7, stride=2, padding=3)
        self.cnv1b = nn.Conv2d(16,  16,  7, stride=1, padding=3)
        self.cnv2  = nn.Conv2d(16,  32,  5, stride=2, padding=2)
        self.cnv2b = nn.Conv2d(32,  32,  5, stride=1, padding=2)
        self.cnv3  = nn.Conv2d(32,  64,  3, stride=2, padding=1)
        self.cnv3b = nn.Conv2d(64,  64,  3, stride=1, padding=1)
        self.cnv4  = nn.Conv2d(64,  128, 3, stride=2, padding=1)
        self.cnv4b = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.cnv5  = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.cnv5b = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # Pose-specific layers
        self.cnv6  = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.cnv6b = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.cnv7  = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.cnv7b = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # 1x1 prediction head
        self.pred  = nn.Conv2d(256, 6, 1, stride=1, padding=0)

    def forward(self, tgt_image, src_image):
        """
        Args:
            tgt_image: (B, 3, H, W)
            src_image: (B, 3, H, W)
        Returns:
            pose: (B, 6) -- [tx, ty, tz, rx, ry, rz]
        """
        x = torch.cat([tgt_image, src_image], dim=1)  # (B, 6, H, W)

        x = F.relu(self.cnv1(x))
        x = F.relu(self.cnv1b(x))
        x = F.relu(self.cnv2(x))
        x = F.relu(self.cnv2b(x))
        x = F.relu(self.cnv3(x))
        x = F.relu(self.cnv3b(x))
        x = F.relu(self.cnv4(x))
        x = F.relu(self.cnv4b(x))
        x = F.relu(self.cnv5(x))
        x = F.relu(self.cnv5b(x))
        x = F.relu(self.cnv6(x))
        x = F.relu(self.cnv6b(x))
        x = F.relu(self.cnv7(x))
        x = F.relu(self.cnv7b(x))

        pose_pred = self.pred(x)                           # (B, 6, h, w)
        pose_avg = pose_pred.mean(dim=[2, 3])              # (B, 6)
        pose_final = pose_avg.view(-1, 6)
        # Scale translations by 0.01 (empirical facilitator for training)
        pose_final = torch.cat(
            [pose_final[:, 0:3], 0.01 * pose_final[:, 3:6]], dim=1)
        return pose_final
