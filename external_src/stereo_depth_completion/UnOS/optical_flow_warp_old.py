import torch
import torch.nn.functional as F
import numpy as np


def transformer_old(U, flo, out_size, name='SpatialTransformer', **kwargs):
    """Backward warping layer.

    Implements a backward warping layer described in
    "Unsupervised Deep Learning for Optical Flow Estimation, Zhe Ren et al"

    Parameters
    ----------
    U : torch.Tensor
        The source image/feature map of shape [num_batch, height, width, num_channels].
        (NHWC format to match original TF interface)
    flo : torch.Tensor
        The optical flow used to do the backward warping.
        Shape is [num_batch, height, width, 2] (NHWC, flow_x and flow_y).
    out_size : tuple of two ints
        The size of the output of the network (height, width).

    Returns
    -------
    output : torch.Tensor
        Backward-warped result of shape [num_batch, out_height, out_width, num_channels].
    """
    num_batch = U.shape[0]
    height = U.shape[1]
    width = U.shape[2]
    channels = U.shape[3]
    device = U.device

    out_height = out_size[0]
    out_width = out_size[1]

    # Convert U from NHWC to NCHW for grid_sample
    U_nchw = U.permute(0, 3, 1, 2)  # [B, C, H, W]

    # Create meshgrid in [-1, 1]
    y_t = torch.linspace(-1.0, 1.0, out_height, device=device)
    x_t = torch.linspace(-1.0, 1.0, out_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_t, x_t, indexing='ij')

    x_s = x_grid.unsqueeze(0).expand(num_batch, -1, -1)
    y_s = y_grid.unsqueeze(0).expand(num_batch, -1, -1)

    # Add flow (convert flow from pixel coords to normalized [-1, 1] space)
    # flo[:,:,:,0] is x-displacement in pixels, flo[:,:,:,1] is y-displacement
    x_out = x_s + flo[:, :, :, 0] / ((float(out_width) - 1.0) / 2.0)
    y_out = y_s + flo[:, :, :, 1] / ((float(out_height) - 1.0) / 2.0)

    # Stack into grid for grid_sample: [B, H, W, 2] with (x, y) in [-1, 1]
    grid = torch.stack([x_out, y_out], dim=-1)

    # grid_sample expects NCHW input, returns NCHW
    output_nchw = F.grid_sample(
        U_nchw, grid, mode='bilinear', padding_mode='zeros',
        align_corners=True)

    # Convert back to NHWC
    output = output_nchw.permute(0, 2, 3, 1)  # [B, H, W, C]

    return output
