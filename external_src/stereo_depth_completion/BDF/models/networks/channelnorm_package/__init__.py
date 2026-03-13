import torch
import torch.nn as nn


class ChannelNorm(nn.Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        # L2 (or general Lp) normalize across channel dimension
        return input1 / (torch.norm(input1, p=self.norm_deg, dim=1, keepdim=True) + 1e-5)
