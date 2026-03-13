import torch
import torch.nn as nn
import torch.nn.functional as F


class Correlation(nn.Module):
    def __init__(self, pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        """
        Pure PyTorch correlation layer.
        input1: [B, C, H, W]
        input2: [B, C, H, W]
        """
        B, C, H, W = input1.shape

        # Pad input2
        input2_padded = F.pad(input2, [self.pad_size] * 4)

        # Number of displacement steps
        max_disp = self.max_displacement
        stride2 = self.stride2

        # Compute output size based on displacements
        neighborhood_grid_radius = max_disp // stride2
        neighborhood_grid_width = 2 * neighborhood_grid_radius + 1
        out_channels = neighborhood_grid_width * neighborhood_grid_width

        output = torch.zeros(B, out_channels, H, W, device=input1.device, dtype=input1.dtype)

        # Loop over displacement offsets
        idx = 0
        for dy in range(-neighborhood_grid_radius, neighborhood_grid_radius + 1):
            for dx in range(-neighborhood_grid_radius, neighborhood_grid_radius + 1):
                # Offset in padded coordinates
                y_offset = self.pad_size + dy * stride2
                x_offset = self.pad_size + dx * stride2

                # Extract shifted patch from padded input2
                input2_shifted = input2_padded[:, :, y_offset:y_offset + H, x_offset:x_offset + W]

                # Correlation: dot product across channels
                output[:, idx, :, :] = torch.sum(input1 * input2_shifted, dim=1) * self.corr_multiply / C

                idx += 1

        return output
