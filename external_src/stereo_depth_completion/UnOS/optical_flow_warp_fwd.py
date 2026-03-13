import torch
import torch.nn.functional as F
import numpy as np


def transformerFwd(U, flo, out_size, name='SpatialTransformerFwd',
                   backprop=False, **kwargs):
    """Forward Warping Layer described in
    'Occlusion Aware Unsupervised Learning of Optical Flow by Yang Wang et al'

    Parameters
    ----------
    U : torch.Tensor
        The source image/feature map of shape [num_batch, height, width, num_channels].
        (NHWC format to match original TF interface)
    flo : torch.Tensor
        The optical flow used for forward warping of shape
        [num_batch, height, width, 2].
    out_size : tuple of two ints
        The size of the output (height, width).
    backprop : boolean
        Indicates whether to back-propagate through the forward warping layer.
        If False, uses scatter_add_ (no gradient). If True, uses a
        differentiable scatter approach.

    Returns
    -------
    output : torch.Tensor
        Forward-warped result of shape [num_batch, out_height, out_width, num_channels].
    """
    num_batch = U.shape[0]
    height = U.shape[1]
    width = U.shape[2]
    channels = U.shape[3]

    out_height = out_size[0]
    out_width = out_size[1]
    device = U.device

    # Create meshgrid in [-1, 1]
    y_t = torch.linspace(-1.0, 1.0, out_height, device=device)
    x_t = torch.linspace(-1.0, 1.0, out_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_t, x_t, indexing='ij')

    x_s = x_grid.unsqueeze(0).expand(num_batch, -1, -1)
    y_s = y_grid.unsqueeze(0).expand(num_batch, -1, -1)

    # Add flow (convert flow from pixel space to [-1,1] space)
    x_t = x_s + flo[:, :, :, 0] / ((out_width - 1.0) / 2.0)
    y_t = y_s + flo[:, :, :, 1] / ((out_height - 1.0) / 2.0)

    # Scale from [-1,1] to [0, width-1] / [0, height-1]
    x = (x_t + 1.0) * (width - 1) / 2.0
    y = (y_t + 1.0) * (height - 1) / 2.0

    # Flatten
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)

    # Floor and ceil
    x0 = torch.floor(x_flat).long()
    x1 = x0 + 1
    y0 = torch.floor(y_flat).long()
    y1 = y0 + 1

    # Clipped versions for bounds checking
    max_x = width - 1
    max_y = height - 1
    x0_c = torch.clamp(x0, 0, max_x)
    x1_c = torch.clamp(x1, 0, max_x)
    y0_c = torch.clamp(y0, 0, max_y)
    y1_c = torch.clamp(y1, 0, max_y)

    # Compute flat indices
    dim2 = width
    dim1 = width * height
    base = torch.arange(num_batch, device=device).unsqueeze(1).expand(
        num_batch, out_height * out_width).reshape(-1) * dim1

    base_y0 = base + y0_c * dim2
    base_y1 = base + y1_c * dim2
    idx_a = base_y0 + x0_c
    idx_b = base_y1 + x0_c
    idx_c = base_y0 + x1_c
    idx_d = base_y1 + x1_c

    # Bilinear weights
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    wa = ((x1_f - x_flat) * (y1_f - y_flat)).unsqueeze(1)  # [N, 1]
    wb = ((x1_f - x_flat) * (y_flat - y0_f)).unsqueeze(1)
    wc = ((x_flat - x0_f) * (y1_f - y_flat)).unsqueeze(1)
    wd = ((x_flat - x0_f) * (y_flat - y0_f)).unsqueeze(1)

    # Zero out weights for out-of-bounds coordinates
    zerof = torch.zeros_like(wa)
    wa = torch.where(
        ((x0_c == x0) & (y0_c == y0)).unsqueeze(1), wa, zerof)
    wb = torch.where(
        ((x0_c == x0) & (y1_c == y1)).unsqueeze(1), wb, zerof)
    wc = torch.where(
        ((x1_c == x1) & (y0_c == y0)).unsqueeze(1), wc, zerof)
    wd = torch.where(
        ((x1_c == x1) & (y1_c == y1)).unsqueeze(1), wd, zerof)

    # Flatten input
    im_flat = U.reshape(-1, channels).float()  # [B*H*W, C]

    total_size = num_batch * height * width

    if not backprop:
        # Use scatter_add_ (no gradient)
        output = torch.zeros(total_size, channels, device=device,
                             dtype=torch.float32)
        output.scatter_add_(0, idx_a.unsqueeze(1).expand(-1, channels),
                            im_flat * wa)
        output.scatter_add_(0, idx_b.unsqueeze(1).expand(-1, channels),
                            im_flat * wb)
        output.scatter_add_(0, idx_c.unsqueeze(1).expand(-1, channels),
                            im_flat * wc)
        output.scatter_add_(0, idx_d.unsqueeze(1).expand(-1, channels),
                            im_flat * wd)
    else:
        # Differentiable version using zeros + scatter via index
        output = torch.zeros(total_size, channels, device=device,
                             dtype=torch.float32)
        # Use put_ or manual accumulation for differentiable scatter
        # We accumulate by creating sparse additions
        for idx, w in [(idx_a, wa), (idx_b, wb), (idx_c, wc), (idx_d, wd)]:
            idx_exp = idx.unsqueeze(1).expand(-1, channels)
            src = im_flat * w
            output = output.scatter_add(0, idx_exp, src)

    output = output.reshape(num_batch, out_height, out_width, channels)
    return output
