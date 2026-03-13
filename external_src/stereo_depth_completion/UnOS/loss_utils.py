import torch
import torch.nn.functional as F
import numpy as np


def average_gradients(tower_grads):
    """No-op in PyTorch. Gradient averaging is handled by DDP."""
    pass


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
        true: the ground truth image.
        pred: the predicted image.
    Returns:
        mean squared error between ground truth and predicted image.
    """
    return torch.sum((true - pred) ** 2) / float(pred.numel())


def weighted_mean_squared_error(true, pred, weight):
    """Weighted L2 distance between tensors true and pred.

    Args:
        true: the ground truth image.
        pred: the predicted image.
        weight: per-pixel weight map.
    Returns:
        weighted mean squared error between ground truth and predicted image.
    """
    # NCHW format: sum over H,W (dims 2,3), keep dims
    tmp = torch.sum(
        weight * (true - pred) ** 2, dim=[2, 3], keepdim=True
    ) / torch.sum(weight, dim=[2, 3], keepdim=True)
    return torch.mean(tmp)


def mean_L1_error(true, pred):
    """L1 distance between tensors true and pred.

    Args:
        true: the ground truth image.
        pred: the predicted image.
    Returns:
        mean L1 error between ground truth and predicted image.
    """
    return torch.sum(torch.abs(true - pred)) / float(pred.numel())


def weighted_mean_L1_error(true, pred, weight):
    """Weighted L1 distance between tensors true and pred.

    Args:
        true: the ground truth image.
        pred: the predicted image.
        weight: per-pixel weight map.
    Returns:
        weighted mean L1 error between ground truth and predicted image.
    """
    return torch.sum(torch.abs(true - pred) * weight) / float(pred.numel())


def cal_grad2_error(flo, image, beta):
    """Calculate the image-edge-aware second-order smoothness loss for flo.

    Args:
        flo: predicted flow or depth, NCHW.
        image: reference image, NCHW.
        beta: weighting factor.
    Returns:
        Weighted second-order smoothness loss.
    """
    def gradient(pred):
        # NCHW: D_dy along H (dim 2), D_dx along W (dim 3)
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-10.0 * torch.mean(
        torch.abs(img_grad_x), dim=1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(
        torch.abs(img_grad_y), dim=1, keepdim=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (torch.mean(beta * weights_x[:, :, :, 1:] * torch.abs(dx2)) +
            torch.mean(beta * weights_y[:, :, 1:, :] * torch.abs(dy2))) / 2.0


def cal_grad2_error_mask(flo, image, beta, mask):
    """Calculate the image-edge-aware second-order smoothness loss for flo
    within the given mask.

    Args:
        flo: predicted flow or depth, NCHW.
        image: reference image, NCHW.
        beta: weighting factor.
        mask: binary mask, NCHW.
    Returns:
        Weighted second-order smoothness loss within mask.
    """
    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-10.0 * torch.mean(
        torch.abs(img_grad_x), dim=1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(
        torch.abs(img_grad_y), dim=1, keepdim=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (torch.mean(beta * weights_x[:, :, :, 1:] * torch.abs(dx2) * mask[:, :, :, 1:-1]) +
            torch.mean(beta * weights_y[:, :, 1:, :] * torch.abs(dy2) * mask[:, :, 1:-1, :])) / 2.0


def SSIM(x, y):
    """Structural similarity loss.

    Args:
        x: image tensor, NCHW.
        y: image tensor, NCHW.
    Returns:
        SSIM loss map, clipped to [0, 1].
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(x ** 2, kernel_size=3, stride=1, padding=1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, kernel_size=3, stride=1, padding=1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM_val = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM_val) / 2, 0, 1)


def deprocess_image(image):
    """Convert float image [0,1] to uint8 [0,255]."""
    return (torch.clamp(image, 0, 1) * 255).to(torch.uint8)


def preprocess_image(image):
    """Convert uint8 image [0,255] to float [0,1]."""
    return image.float() / 255.0


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0,
                     epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [N, C, H, W].
        mask: a mask of shape [N, C_mask, H, W], where C_mask must be either 1
            or the same number as channels of x. Entries should be 0 or 1.
        truncate: optional truncation value.
        alpha: exponent (default 0.45).
        beta: scaling factor (default 1.0).
        epsilon: small constant for numerical stability (default 0.001).
    Returns:
        loss as float.
    """
    batch, channels, height, width = x.shape
    normalization = float(batch * channels * height * width)

    error = ((x * beta) ** 2 + epsilon ** 2) ** alpha

    if mask is not None:
        error = mask * error

    if truncate is not None:
        error = torch.minimum(error, torch.tensor(truncate, dtype=error.dtype,
                                                   device=error.device))

    return torch.sum(error) / normalization
