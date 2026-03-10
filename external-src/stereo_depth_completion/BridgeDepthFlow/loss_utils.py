"""
Loss utility functions for BridgeDepthFlow.

Extracted and adapted from BridgeDepthFlow/utils/utils.py.

Key modifications from original:
- Replaced Resample2d (CUDA custom op) with pure-PyTorch resample2d using F.grid_sample
- Replaced Variable with direct tensors (Variable is deprecated in modern PyTorch)
- Replaced make_pyramid's PIL-based resizing with pure-PyTorch F.interpolate
- Removed hardcoded .cuda() calls; tensors follow the device of their inputs
- create_border_mask uses F.pad instead of nn.ZeroPad2d with Variable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def resample2d(image, flow):
    """
    Warp an image using a flow field, pure-PyTorch replacement for Resample2d().

    This replaces the CUDA-only Resample2d custom op with F.grid_sample.

    Args:
        image: (N, C, H, W) tensor to warp
        flow: (N, 2, H, W) tensor of pixel displacements (dx, dy)

    Returns:
        Warped image tensor (N, C, H, W)
    """
    N, C, H, W = image.shape

    # Create base grid of pixel coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=flow.dtype),
        torch.arange(W, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )

    # Add flow displacements to base grid
    # flow[:, 0] is horizontal (x) displacement, flow[:, 1] is vertical (y) displacement
    grid_x = grid_x.unsqueeze(0).expand(N, -1, -1) + flow[:, 0, :, :]
    grid_y = grid_y.unsqueeze(0).expand(N, -1, -1) + flow[:, 1, :, :]

    # Normalize to [-1, 1] for F.grid_sample
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0

    # Stack into grid (N, H, W, 2) with (x, y) order for grid_sample
    grid = torch.stack([grid_x, grid_y], dim=-1)

    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


def SSIM(x, y):
    """
    Compute structural similarity index map between two images.

    Args:
        x, y: (N, C, H, W) tensors

    Returns:
        (N, C, H', W') tensor of per-pixel dissimilarity in [0, 1],
        where H' = H-2, W' = W-2 due to avg_pool2d with kernel=3, no padding.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM_val = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM_val) / 2, 0, 1)


def cal_grad2_error(flo, image, beta):
    """
    Calculate image-edge-aware second-order smoothness loss for a flow/disparity field.

    Args:
        flo: (N, C, H, W) flow or disparity field
        image: (N, C, H, W) reference image for edge-awareness
        beta: scalar weight multiplier

    Returns:
        Scalar loss value
    """
    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (
        torch.mean(beta * weights_x[:, :, :, 1:] * torch.abs(dx2))
        + torch.mean(beta * weights_y[:, :, 1:, :] * torch.abs(dy2))
    ) / 2.0


def length_sq(x):
    """Compute squared length along channel dimension."""
    return torch.sum(x ** 2, 1, keepdim=True)


def create_outgoing_mask(flow):
    """
    Create a mask indicating pixels that remain inside the image after warping.

    Args:
        flow: (N, 2, H, W) flow field

    Returns:
        (N, 1, H, W) binary mask (1 = stays inside, 0 = goes outside)
    """
    num_batch, channel, height, width = flow.shape

    grid_x = torch.arange(width, device=flow.device, dtype=flow.dtype).view(1, 1, width)
    grid_x = grid_x.expand(num_batch, height, -1)
    grid_y = torch.arange(height, device=flow.device, dtype=flow.dtype).view(1, height, 1)
    grid_y = grid_y.expand(num_batch, -1, width)

    flow_u, flow_v = torch.unbind(flow, 1)
    pos_x = grid_x + flow_u
    pos_y = grid_y + flow_v
    inside_x = (pos_x <= (width - 1)) & (pos_x >= 0.0)
    inside_y = (pos_y <= (height - 1)) & (pos_y >= 0.0)
    inside = inside_x & inside_y
    return inside.float().unsqueeze(1)


def _create_mask(tensor, paddings):
    """
    Create a binary mask with zeros at the borders.

    Args:
        tensor: reference tensor for shape and device
        paddings: [[top, bottom], [left, right]]

    Returns:
        (N, 1, H, W) border mask
    """
    shape = tensor.shape
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones((inner_height, inner_width), device=tensor.device)

    mask2d = F.pad(inner, (paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]))
    mask4d = mask2d.unsqueeze(0).unsqueeze(0).expand(shape[0], 1, -1, -1)
    return mask4d.detach()


def create_border_mask(tensor, border_ratio=0.1):
    """
    Create a border mask that zeros out a border_ratio fraction of the image edges.

    Args:
        tensor: (N, C, H, W) reference tensor
        border_ratio: fraction of height to use as border size

    Returns:
        (N, 1, H, W) binary mask with 0 at borders, 1 in the interior
    """
    num_batch, _, height, width = tensor.shape
    sz = int(np.ceil(height * border_ratio))
    border_mask = _create_mask(tensor, [[sz, sz], [sz, sz]])
    return border_mask.detach()


def get_mask(forward, backward, border_mask):
    """
    Compute forward-backward occlusion masks using consistency check.

    Uses resample2d (F.grid_sample based) instead of CUDA Resample2d.

    Args:
        forward: (N, 2, H, W) forward flow/disparity
        backward: (N, 2, H, W) backward flow/disparity
        border_mask: (N, 1, H, W) border mask or None

    Returns:
        fw: (N, 1, H, W) forward visibility mask
        bw: (N, 1, H, W) backward visibility mask
        flow_diff_fw: (N, 2, H, W) forward flow consistency error
        flow_diff_bw: (N, 2, H, W) backward flow consistency error
    """
    flow_fw = forward
    flow_bw = backward
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)

    flow_bw_warped = resample2d(flow_bw, flow_fw)
    flow_fw_warped = resample2d(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh = 0.01 * mag_sq + 0.5
    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh).float()

    if border_mask is None:
        mask_fw = create_outgoing_mask(flow_fw)
        mask_bw = create_outgoing_mask(flow_bw)
    else:
        mask_fw = border_mask
        mask_bw = border_mask
    fw = mask_fw * (1 - fb_occ_fw)
    bw = mask_bw * (1 - fb_occ_bw)

    return fw, bw, flow_diff_fw, flow_diff_bw


def make_pyramid(image, num_scales):
    """
    Create a multi-scale image pyramid using pure-PyTorch interpolation.

    Replaces the original PIL-based implementation with F.interpolate.

    Args:
        image: (N, C, H, W) tensor
        num_scales: number of scales (e.g. 4)

    Returns:
        List of num_scales tensors, from original resolution to smallest
    """
    scale_image = [image]
    height, width = image.shape[2:]

    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = height // ratio
        nw = width // ratio
        scaled = F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=True)
        scale_image.append(scaled)

    return scale_image


def warp_2(est, img, occ_mask, alpha_image_loss=0.85):
    """
    Compute secondary warp photometric loss (L1 + SSIM) with occlusion masking.

    Args:
        est: (N, C, H, W) warped estimate
        img: (N, C, H, W) target image
        occ_mask: (N, 1, H, W) occlusion mask
        alpha_image_loss: weight between SSIM and L1

    Returns:
        Scalar loss value
    """
    l1_warp2 = torch.abs(est - img) * occ_mask
    l1_reconstruction_loss_warp2 = torch.mean(l1_warp2) / torch.mean(occ_mask)
    ssim_warp2 = SSIM(est * occ_mask, img * occ_mask)
    ssim_loss_warp2 = torch.mean(ssim_warp2) / torch.mean(occ_mask)
    image_loss_warp2 = alpha_image_loss * ssim_loss_warp2 + (1 - alpha_image_loss) * l1_reconstruction_loss_warp2
    return image_loss_warp2
