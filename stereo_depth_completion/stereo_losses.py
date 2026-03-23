'''
Stereo reconstruction loss for AugUndo training.

Computes loss in the original (un-augmented) coordinate frame using
undone disparity predictions and original images. Independent of
model-specific code in external_src/.

Loss terms:
    L_rec   : photometric reconstruction (alpha * SSIM + (1-alpha) * L1)
    L_sm    : 2nd-order image-edge-aware disparity smoothness
    L_lr    : left-right disparity consistency
'''

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Warping helpers
# ---------------------------------------------------------------------------

def warp_right_to_left(right_img, disp_left):
    '''
    Reconstruct left image by backward-warping right image with left disparity.

    Arg(s):
        right_img : torch.Tensor[float32]
            (B, C, H, W) right / source image
        disp_left : torch.Tensor[float32]
            (B, 1, H, W) positive normalized left disparity (fraction of width)

    Returns:
        torch.Tensor[float32] : (B, C, H, W) reconstructed left image
    '''

    B, C, H, W = right_img.shape

    gy = torch.linspace(-1.0, 1.0, H, device=right_img.device, dtype=right_img.dtype)
    gx = torch.linspace(-1.0, 1.0, W, device=right_img.device, dtype=right_img.dtype)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Left pixel x corresponds to right pixel x - d*W.
    # In normalised [-1,1] coordinates: x_norm - d * 2W/(W-1).
    offset = disp_left[:, 0] * (2.0 * W / max(W - 1, 1))
    grid_x_warped = grid_x - offset

    grid = torch.stack([grid_x_warped, grid_y], dim=-1)
    return F.grid_sample(
        right_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def warp_left_to_right(left_img, disp_right):
    '''
    Reconstruct right image by backward-warping left image with right disparity.

    Arg(s):
        left_img : torch.Tensor[float32]
            (B, C, H, W) left / source image
        disp_right : torch.Tensor[float32]
            (B, 1, H, W) positive normalized right disparity (fraction of width)

    Returns:
        torch.Tensor[float32] : (B, C, H, W) reconstructed right image
    '''

    B, C, H, W = left_img.shape

    gy = torch.linspace(-1.0, 1.0, H, device=left_img.device, dtype=left_img.dtype)
    gx = torch.linspace(-1.0, 1.0, W, device=left_img.device, dtype=left_img.dtype)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Right pixel x corresponds to left pixel x + d*W.
    offset = disp_right[:, 0] * (2.0 * W / max(W - 1, 1))
    grid_x_warped = grid_x + offset

    grid = torch.stack([grid_x_warped, grid_y], dim=-1)
    return F.grid_sample(
        left_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def ssim(x, y):
    '''SSIM loss with VALID pooling (no padding).  Returns (1-SSIM)/2 in [0,1].'''

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y,  3, 1, 0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def _gradient_x(img):
    '''Horizontal gradient (right difference): (N, C, H, W) -> (N, C, H, W-1).'''
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img):
    '''Vertical gradient (down difference): (N, C, H, W) -> (N, C, H-1, W).'''
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def smoothness_loss_2nd_order(disp, image):
    '''
    2nd-order image-edge-aware disparity smoothness loss.

    Matches the formulation used by both UnOS (MonodepthModel) and BDF
    (cal_grad2_error): second-order disparity gradients weighted by
    exp(-10 * |image_gradient|).

    Arg(s):
        disp  : torch.Tensor[float32]  (B, 1, H, W) disparity map
        image : torch.Tensor[float32]  (B, 3, H, W) reference image for edges

    Returns:
        torch.Tensor[float32] : scalar smoothness loss
    '''

    disp_gx  = _gradient_x(disp)
    disp_gy  = _gradient_y(disp)
    disp_gxx = _gradient_x(disp_gx)
    disp_gyy = _gradient_y(disp_gy)

    img_gx = _gradient_x(image)
    img_gy = _gradient_y(image)

    weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_gx), dim=1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_gy), dim=1, keepdim=True))

    loss_x = torch.mean(torch.abs(disp_gxx) * weights_x[:, :, :, :-1])
    loss_y = torch.mean(torch.abs(disp_gyy) * weights_y[:, :, :-1, :])

    return (loss_x + loss_y) / 2.0


def scale_pyramid(img, num_scales):
    '''Multi-scale image pyramid using area interpolation.'''

    scaled = [img]
    _, _, H, W = img.shape
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        scaled.append(
            F.interpolate(img, size=(H // ratio, W // ratio), mode='area'))
    return scaled


# ---------------------------------------------------------------------------
# Full stereo loss
# ---------------------------------------------------------------------------

def compute_stereo_loss(left_img, right_img,
                        disp_left, disp_right,
                        alpha_image_loss=0.85,
                        disp_smooth_weight=10.0,
                        lr_loss_weight=1.0,
                        num_scales=4):
    '''
    Compute stereo reconstruction loss in the original (un-augmented) frame.

    Arg(s):
        left_img : torch.Tensor[float32]
            (B, 3, H, W) original left image
        right_img : torch.Tensor[float32]
            (B, 3, H, W) original right image
        disp_left : list[torch.Tensor[float32]]
            num_scales tensors each (B, 1, H_s, W_s), positive normalised left
            disparity (fraction of image width)
        disp_right : list[torch.Tensor[float32]]
            num_scales tensors each (B, 1, H_s, W_s), positive normalised right
            disparity
        alpha_image_loss : float
            weight between SSIM and L1 (0.85 matches BDF/UnOS default)
        disp_smooth_weight : float
            smoothness loss weight (10.0 matches both models)
        lr_loss_weight : float
            left-right consistency weight (1.0 matches UnOS)
        num_scales : int
            number of pyramid scales (4 for both models)

    Returns:
        torch.Tensor[float32] : total scalar loss
        dict[str, float] : dictionary of individual loss components
    '''

    left_pyramid  = scale_pyramid(left_img, num_scales)
    right_pyramid = scale_pyramid(right_img, num_scales)

    image_loss  = 0.0
    smooth_loss = 0.0
    lr_loss     = 0.0

    for s in range(num_scales):
        # -- Photometric reconstruction (SSIM + L1) --
        left_est  = warp_right_to_left(right_pyramid[s], disp_left[s])
        right_est = warp_left_to_right(left_pyramid[s],  disp_right[s])

        l1_left   = torch.mean(torch.abs(left_est  - left_pyramid[s]))
        l1_right  = torch.mean(torch.abs(right_est - right_pyramid[s]))
        ssim_left  = torch.mean(ssim(left_est,  left_pyramid[s]))
        ssim_right = torch.mean(ssim(right_est, right_pyramid[s]))

        image_loss += (
            alpha_image_loss * (ssim_left + ssim_right) +
            (1 - alpha_image_loss) * (l1_left + l1_right))

        # -- 2nd-order smoothness --
        smooth_loss += (
            smoothness_loss_2nd_order(disp_left[s],  left_pyramid[s]) +
            smoothness_loss_2nd_order(disp_right[s], right_pyramid[s])
        ) / (2 ** s)

        # -- Left-right consistency --
        right_to_left_disp = warp_right_to_left(disp_right[s], disp_left[s])
        left_to_right_disp = warp_left_to_right(disp_left[s],  disp_right[s])

        lr_loss += (
            torch.mean(torch.abs(right_to_left_disp - disp_left[s])) +
            torch.mean(torch.abs(left_to_right_disp - disp_right[s])))

    smooth_loss *= 0.5

    total_loss = (image_loss +
                  disp_smooth_weight * smooth_loss +
                  lr_loss_weight * lr_loss)

    loss_info = {
        'loss_image':  image_loss.item()  if torch.is_tensor(image_loss)  else image_loss,
        'loss_smooth': smooth_loss.item() if torch.is_tensor(smooth_loss) else smooth_loss,
        'loss_lr':     lr_loss.item()     if torch.is_tensor(lr_loss)     else lr_loss,
        'loss':        total_loss.item()  if torch.is_tensor(total_loss)  else total_loss,
    }

    return total_loss, loss_info
