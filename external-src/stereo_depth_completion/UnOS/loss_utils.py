"""
Loss utility functions for UnOS stereo depth estimation.

Ported from the original TensorFlow implementation.
Includes SSIM, edge-aware smoothness, and Charbonnier loss functions.

All tensors use PyTorch NCHW format (batch, channels, height, width).
Note: The original TF code uses NHWC. Gradient computations along spatial
dimensions have been adjusted accordingly (H is dim 2, W is dim 3 in NCHW).
"""

import torch
import torch.nn.functional as F


def SSIM(x, y):
    """Compute the Structural Similarity Index (SSIM) loss.

    Computes a per-pixel SSIM map and returns clamped (1 - SSIM) / 2
    as a loss value (lower SSIM = higher loss).

    Uses 3x3 average pooling with 'VALID' padding (no padding) to compute
    local statistics, matching the original TF implementation.

    Args:
        x: Predicted image, shape N x C x H x W.
        y: Target image, shape N x C x H x W.

    Returns:
        SSIM loss map, shape N x C x (H-2) x (W-2), values in [0, 1].
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 'VALID' padding = no padding (padding=0)
    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=0)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_x_mu_y = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x ** 2, kernel_size=3, stride=1, padding=0) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y ** 2, kernel_size=3, stride=1, padding=0) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=0) - mu_x_mu_y

    numerator = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / denominator

    return torch.clamp((1 - ssim_map) / 2, 0, 1)


def cal_grad2_error(flo, image, beta):
    """Compute 2nd-order edge-aware smoothness loss.

    Penalizes second-order gradients of the flow/disparity, weighted by
    image gradients (edge-aware: less smoothness at image edges).

    Args:
        flo: Flow or disparity map, shape N x C x H x W.
        image: Reference image, shape N x C_img x H x W.
        beta: Weighting factor for edge-aware term.

    Returns:
        Scalar smoothness loss.
    """
    def gradient_h(x):
        """Compute gradient along H dimension (dim 2 in NCHW)."""
        return x[:, :, :-1, :] - x[:, :, 1:, :]

    def gradient_w(x):
        """Compute gradient along W dimension (dim 3 in NCHW)."""
        return x[:, :, :, :-1] - x[:, :, :, 1:]

    # Image gradients for edge-aware weighting
    img_grad_h = gradient_h(image)
    img_grad_w = gradient_w(image)

    # Edge-aware weights: exponential decay at edges
    # Mean across channel dimension for multi-channel images
    weights_h = torch.exp(-beta * torch.mean(torch.abs(img_grad_h), dim=1, keepdim=True))
    weights_w = torch.exp(-beta * torch.mean(torch.abs(img_grad_w), dim=1, keepdim=True))

    # First-order flow gradients
    flo_grad_h = gradient_h(flo)
    flo_grad_w = gradient_w(flo)

    # Second-order flow gradients
    flo_grad2_h = gradient_h(flo_grad_h)
    flo_grad2_w = gradient_w(flo_grad_w)

    # Trim weights to match second-order gradient sizes
    # flo_grad2_h has H-2, weights_h has H-1, so trim one row
    weights_h_trimmed = weights_h[:, :, :-1, :]
    weights_w_trimmed = weights_w[:, :, :, :-1]

    error = (
        torch.mean(weights_h_trimmed * torch.abs(flo_grad2_h)) +
        torch.mean(weights_w_trimmed * torch.abs(flo_grad2_w))
    )

    return error / 2.0


def cal_grad2_error_mask(flo, image, beta, mask):
    """Compute 2nd-order edge-aware smoothness loss with a validity mask.

    Same as cal_grad2_error but applies a mask to only compute the loss
    over valid regions.

    Args:
        flo: Flow or disparity map, shape N x C x H x W.
        image: Reference image, shape N x C_img x H x W.
        beta: Weighting factor for edge-aware term.
        mask: Validity mask, shape N x 1 x H x W. Values in [0, 1].

    Returns:
        Scalar smoothness loss over masked region.
    """
    def gradient_h(x):
        return x[:, :, :-1, :] - x[:, :, 1:, :]

    def gradient_w(x):
        return x[:, :, :, :-1] - x[:, :, :, 1:]

    img_grad_h = gradient_h(image)
    img_grad_w = gradient_w(image)

    weights_h = torch.exp(-beta * torch.mean(torch.abs(img_grad_h), dim=1, keepdim=True))
    weights_w = torch.exp(-beta * torch.mean(torch.abs(img_grad_w), dim=1, keepdim=True))

    flo_grad_h = gradient_h(flo)
    flo_grad_w = gradient_w(flo)

    flo_grad2_h = gradient_h(flo_grad_h)
    flo_grad2_w = gradient_w(flo_grad_w)

    weights_h_trimmed = weights_h[:, :, :-1, :]
    weights_w_trimmed = weights_w[:, :, :, :-1]

    # Trim mask to match second-order gradient sizes
    mask_h = mask[:, :, 1:-1, :]  # trim 1 from each end along H
    mask_w = mask[:, :, :, 1:-1]  # trim 1 from each end along W

    error_h = weights_h_trimmed * torch.abs(flo_grad2_h) * mask_h
    error_w = weights_w_trimmed * torch.abs(flo_grad2_w) * mask_w

    # Normalize by mask sum to avoid division effects from mask size
    norm_h = torch.sum(mask_h) + 1e-8
    norm_w = torch.sum(mask_w) + 1e-8

    error = torch.sum(error_h) / norm_h + torch.sum(error_w) / norm_w

    return error / 2.0


def charbonnier_loss(x, mask=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the Charbonnier loss (generalized L1).

    Charbonnier loss is a smooth approximation of L1 loss:
    loss = (x^2 * beta^2 + epsilon^2)^alpha

    Args:
        x: Input error tensor.
        mask: Optional validity mask. If provided, only masked regions
              contribute to the loss.
        alpha: Exponent parameter (default 0.45).
        beta: Scaling factor for input (default 1.0).
        epsilon: Small constant for numerical stability (default 0.001).

    Returns:
        Scalar Charbonnier loss.
    """
    error = torch.pow(x * x * beta * beta + epsilon * epsilon, alpha)

    if mask is not None:
        error = error * mask
        return torch.sum(error) / (torch.sum(mask) + 1e-8)
    else:
        return torch.mean(error)
