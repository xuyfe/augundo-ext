import torch
import torch.nn as nn
import torch.nn.functional as F


class Resample2d(nn.Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        """
        Warp input1 using the flow field input2 via F.grid_sample.
        input1: [B, C, H1, W1] - image/features to warp
        input2: [B, 2, H2, W2] - flow field (u, v)
        Output: [B, C, H2, W2]
        """
        B, _, H, W = input2.shape

        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=input2.device, dtype=input2.dtype),
            torch.arange(W, device=input2.device, dtype=input2.dtype),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

        # Add flow to base grid
        flow_u = input2[:, 0, :, :]  # horizontal displacement
        flow_v = input2[:, 1, :, :]  # vertical displacement

        x = grid_x + flow_u
        y = grid_y + flow_v

        # Normalize to [-1, 1] for grid_sample
        _, _, H1, W1 = input1.shape
        x = 2.0 * x / max(W1 - 1, 1) - 1.0
        y = 2.0 * y / max(H1 - 1, 1) - 1.0

        grid = torch.stack([x, y], dim=-1)  # [B, H, W, 2]

        output = F.grid_sample(input1, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return output
