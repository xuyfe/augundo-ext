"""
Warping operations for UnOS stereo depth estimation.

Ported from the original TensorFlow implementation.
Provides backward warping (bilinear sampling) and forward warping (splatting)
for image reconstruction and occlusion detection.

All tensors use PyTorch NCHW format (batch, channels, height, width).
"""

import torch
import torch.nn.functional as F


def backward_warp(image, flow):
    """Backward warp an image using bilinear sampling.

    Given a flow field, sample pixels from the source image to produce
    a warped image. Uses F.grid_sample for differentiable bilinear interpolation.

    Args:
        image: Source image tensor, shape N x C x H x W.
        flow: Optical flow / displacement field, shape N x 2 x H x W.
              Channel 0 is horizontal (x) displacement in pixels,
              Channel 1 is vertical (y) displacement in pixels.

    Returns:
        Warped image tensor, shape N x C x H x W.
    """
    N, C, H, W = image.shape

    # Create base meshgrid of pixel coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, dtype=flow.dtype, device=flow.device),
        torch.arange(0, W, dtype=flow.dtype, device=flow.device),
        indexing='ij'
    )

    # grid_x, grid_y: H x W
    grid_x = grid_x.unsqueeze(0).expand(N, -1, -1)  # N x H x W
    grid_y = grid_y.unsqueeze(0).expand(N, -1, -1)  # N x H x W

    # Add flow displacement to base coordinates
    # flow[:, 0] is x displacement, flow[:, 1] is y displacement
    x = grid_x + flow[:, 0, :, :]  # N x H x W
    y = grid_y + flow[:, 1, :, :]  # N x H x W

    # Normalize to [-1, 1] for grid_sample
    x = 2.0 * x / (W - 1) - 1.0
    y = 2.0 * y / (H - 1) - 1.0

    # Stack into grid: N x H x W x 2 (last dim is x, y)
    grid = torch.stack([x, y], dim=-1)

    # Bilinear sampling
    warped = F.grid_sample(
        image, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    return warped


def forward_warp(tensor, flow):
    """Forward warp (splatting) for occlusion detection.

    Splats values from the source tensor to target locations defined by the
    flow field. Primarily used to splat a tensor of ones to detect which
    pixels in the target are covered (non-occluded) vs uncovered (occluded).

    Uses scatter_add for the splatting operation with bilinear weight
    distribution to the 4 nearest integer pixel locations.

    Args:
        tensor: Source tensor to splat, shape N x C x H x W.
        flow: Flow field defining target locations, shape N x 2 x H x W.
              Channel 0 is x displacement, channel 1 is y displacement.

    Returns:
        Splatted tensor, shape N x C x H x W.
    """
    N, C, H, W = tensor.shape

    # Create base meshgrid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, dtype=flow.dtype, device=flow.device),
        torch.arange(0, W, dtype=flow.dtype, device=flow.device),
        indexing='ij'
    )

    grid_x = grid_x.unsqueeze(0).expand(N, -1, -1)  # N x H x W
    grid_y = grid_y.unsqueeze(0).expand(N, -1, -1)  # N x H x W

    # Target coordinates
    x = grid_x + flow[:, 0, :, :]  # N x H x W
    y = grid_y + flow[:, 1, :, :]  # N x H x W

    # Get the four nearest integer coordinates
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # Bilinear weights
    wa = (x1.float() - x) * (y1.float() - y)  # top-left weight
    wb = (x1.float() - x) * (y - y0.float())  # bottom-left weight
    wc = (x - x0.float()) * (y1.float() - y)  # top-right weight
    wd = (x - x0.float()) * (y - y0.float())  # bottom-right weight

    # Initialize output
    output = torch.zeros_like(tensor)

    # Flatten spatial dimensions for scatter
    # For each of the 4 corners, compute linear index and scatter
    for dx, dy, w in [(x0, y0, wa), (x0, y1, wb), (x1, y0, wc), (x1, y1, wd)]:
        # Clamp to valid range
        valid = (dx >= 0) & (dx < W) & (dy >= 0) & (dy < H)
        dx_c = dx.clamp(0, W - 1)
        dy_c = dy.clamp(0, H - 1)

        # Linear index into H*W
        linear_idx = dy_c * W + dx_c  # N x H x W

        # Apply validity mask to weights
        w_valid = w * valid.float()  # N x H x W

        # Scatter for each batch and channel
        for n in range(N):
            for c in range(C):
                weighted_vals = tensor[n, c, :, :].reshape(-1) * w_valid[n].reshape(-1)
                idx = linear_idx[n].reshape(-1)
                output[n, c].reshape(-1).scatter_add_(0, idx, weighted_vals)

    return output
