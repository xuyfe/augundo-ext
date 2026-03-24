'''
Stereo reconstruction loss for AugUndo training.

Computes loss in the original (un-augmented) coordinate frame using
undone disparity predictions and original images. Independent of
model-specific code in external_src/.

Loss terms:
    L_rec   : photometric reconstruction (alpha * SSIM + (1-alpha) * L1)
    L_sm    : 2nd-order image-edge-aware disparity smoothness
    L_lr    : left-right disparity consistency

Occlusion masking matches the forward-warp approach from UnOS
(monodepth_model.py build_outputs): forward-warp ones from each view
to the other, clamp to [0, 1], and use as pixel-wise loss weights.
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


def warp_with_flow_2d(source, flow_norm):
    '''
    Backward-warp source image using 2D normalized flow.

    Arg(s):
        source : torch.Tensor[float32]
            (B, C, H, W) source image to sample from
        flow_norm : torch.Tensor[float32]
            (B, 2, H, W) normalized flow (ch0 = fraction of width,
            ch1 = fraction of height)

    Returns:
        torch.Tensor[float32] : (B, C, H, W) warped image
    '''

    B, C, H, W = source.shape

    gy = torch.linspace(-1.0, 1.0, H, device=source.device, dtype=source.dtype)
    gx = torch.linspace(-1.0, 1.0, W, device=source.device, dtype=source.dtype)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    grid_x_warped = grid_x + flow_norm[:, 0] * (2.0 * W / max(W - 1, 1))
    grid_y_warped = grid_y + flow_norm[:, 1] * (2.0 * H / max(H - 1, 1))

    grid = torch.stack([grid_x_warped, grid_y_warped], dim=-1)
    return F.grid_sample(
        source, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


# ---------------------------------------------------------------------------
# Occlusion masking via forward warping
# ---------------------------------------------------------------------------

def _forward_warp_ones(pixel_disp, direction, H, W):
    '''
    Forward-warp a ones tensor horizontally using stereo disparity.

    Implements bilinear splatting: each source pixel at position x is
    scattered to target position (x +/- d) with bilinear weights. The
    output indicates how many source pixels cover each target pixel.
    Clamped to [0, 1] to produce a visibility mask.

    This matches the occlusion masking approach in UnOS
    (transformerFwd applied to ones tensors in monodepth_model.py).

    Arg(s):
        pixel_disp : torch.Tensor[float32]
            (B, H, W) disparity in pixel units (positive)
        direction : str
            'left_to_right' or 'right_to_left'
        H : int
            image height
        W : int
            image width

    Returns:
        torch.Tensor[float32] : (B, 1, H, W) visibility mask in [0, 1]
    '''

    B = pixel_disp.shape[0]
    device = pixel_disp.device
    dtype = pixel_disp.dtype

    # Source pixel x-coordinates: (B, H, W)
    x_s = torch.arange(W, device=device, dtype=dtype)
    x_s = x_s.view(1, 1, W).expand(B, H, -1)

    # Target x-coordinate after disparity shift
    if direction == 'left_to_right':
        x_t = x_s - pixel_disp     # left pixel x -> right pixel x - d
    else:
        x_t = x_s + pixel_disp     # right pixel x -> left pixel x + d

    # Bilinear weights (horizontal only; y is unchanged for stereo)
    x0 = torch.floor(x_t).long()
    x1 = x0 + 1
    w1 = (x_t - x0.float())        # weight for x1 bin
    w0 = 1.0 - w1                   # weight for x0 bin

    # Flat index: batch_offset + row_offset + col
    batch_offset = torch.arange(B, device=device).view(B, 1, 1) * (H * W)
    row_offset = torch.arange(H, device=device).view(1, H, 1) * W
    base = (batch_offset + row_offset).expand_as(x0).reshape(-1)

    x0_flat = x0.reshape(-1)
    x1_flat = x1.reshape(-1)
    w0_flat = w0.reshape(-1)
    w1_flat = w1.reshape(-1)

    # Mask out-of-bounds columns
    valid0 = (x0_flat >= 0) & (x0_flat < W)
    valid1 = (x1_flat >= 0) & (x1_flat < W)

    idx0 = base + torch.clamp(x0_flat, 0, W - 1)
    idx1 = base + torch.clamp(x1_flat, 0, W - 1)

    output = torch.zeros(B * H * W, device=device, dtype=dtype)
    output.scatter_add_(0, idx0[valid0], w0_flat[valid0])
    output.scatter_add_(0, idx1[valid1], w1_flat[valid1])

    return torch.clamp(output.reshape(B, 1, H, W), 0.0, 1.0)


def compute_occlusion_masks(disp_left, disp_right):
    '''
    Compute occlusion masks via forward warping of ones tensors.

    Matches the approach in UnOS (monodepth_model.py build_outputs):
    forward-warp ones from each view to the other using the predicted
    disparity, then clamp to [0, 1].

    Arg(s):
        disp_left : torch.Tensor[float32]
            (B, 1, H, W) positive normalized left disparity (fraction of width)
        disp_right : torch.Tensor[float32]
            (B, 1, H, W) positive normalized right disparity (fraction of width)

    Returns:
        left_occ_mask : torch.Tensor[float32]
            (B, 1, H, W) left visibility mask (1 = visible from right view)
        right_occ_mask : torch.Tensor[float32]
            (B, 1, H, W) right visibility mask (1 = visible from left view)
    '''

    _, _, H, W = disp_left.shape

    # Right mask: forward-warp ones from left to right using left disparity
    right_occ_mask = _forward_warp_ones(
        disp_left[:, 0] * W, 'left_to_right', H, W)

    # Left mask: forward-warp ones from right to left using right disparity
    left_occ_mask = _forward_warp_ones(
        disp_right[:, 0] * W, 'right_to_left', H, W)

    return left_occ_mask.detach(), right_occ_mask.detach()


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
                        num_scales=4,
                        smooth_flow_left=None,
                        smooth_flow_right=None,
                        smooth_pixel_divisor=0.0,
                        smooth_per_scale=True):
    '''
    Compute stereo reconstruction loss in the original (un-augmented) frame.

    Uses forward-warped occlusion masks (matching UnOS monodepth_model.py)
    to weight photometric, SSIM, and LR consistency losses so that occluded
    pixels do not contribute invalid gradients.

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
        smooth_flow_left : list[torch.Tensor[float32]] or None
            if provided, use these (B, C, H_s, W_s) tensors for smoothness
            instead of disp_left. Allows passing multi-channel flow (BDF)
        smooth_flow_right : list[torch.Tensor[float32]] or None
            same for right view
        smooth_pixel_divisor : float
            if > 0, convert normalised disparity/flow to pixel scale and
            divide by this value before smoothness (BDF uses 20.0, matching
            native ``cal_grad2_error(disp_scale / 20, ...)``)
        smooth_per_scale : bool
            if True, weight smoothness by 1/(2^s) per scale (UnOS default).
            if False, all scales contribute equally (BDF default)

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
        # -- Occlusion masks at this scale --
        left_occ_mask, right_occ_mask = compute_occlusion_masks(
            disp_left[s], disp_right[s])
        left_occ_avg = torch.mean(left_occ_mask) + 1e-12
        right_occ_avg = torch.mean(right_occ_mask) + 1e-12

        # -- Photometric reconstruction (SSIM + L1) with occlusion masking --
        left_est  = warp_right_to_left(right_pyramid[s], disp_left[s])
        right_est = warp_left_to_right(left_pyramid[s],  disp_right[s])

        # L1 masked by visibility
        l1_left = torch.mean(
            torch.abs(left_est - left_pyramid[s]) * left_occ_mask
        ) / left_occ_avg
        l1_right = torch.mean(
            torch.abs(right_est - right_pyramid[s]) * right_occ_mask
        ) / right_occ_avg

        # SSIM masked by visibility (mask applied to both inputs)
        ssim_left = torch.mean(
            ssim(left_est * left_occ_mask, left_pyramid[s] * left_occ_mask)
        ) / left_occ_avg
        ssim_right = torch.mean(
            ssim(right_est * right_occ_mask, right_pyramid[s] * right_occ_mask)
        ) / right_occ_avg

        image_loss += (
            alpha_image_loss * (ssim_left + ssim_right) +
            (1 - alpha_image_loss) * (l1_left + l1_right))

        # -- 2nd-order smoothness --
        # Select what to compute smoothness on: explicit flow or disparity
        sm_left = smooth_flow_left[s] if smooth_flow_left is not None else disp_left[s]
        sm_right = smooth_flow_right[s] if smooth_flow_right is not None else disp_right[s]

        # Optionally convert from normalised to pixel-scale / divisor
        # For 2-channel flow: ch0 is fraction of width, ch1 is fraction of height,
        # so scale each channel by its respective spatial dimension.
        if smooth_pixel_divisor > 0:
            W_s = sm_left.shape[3]
            H_s = sm_left.shape[2]
            if sm_left.shape[1] >= 2:
                scale_x = W_s / smooth_pixel_divisor
                scale_y = H_s / smooth_pixel_divisor
                sm_left = torch.cat([
                    sm_left[:, 0:1] * scale_x,
                    sm_left[:, 1:] * scale_y], dim=1)
                sm_right = torch.cat([
                    sm_right[:, 0:1] * scale_x,
                    sm_right[:, 1:] * scale_y], dim=1)
            else:
                sm_left = sm_left * (W_s / smooth_pixel_divisor)
                sm_right = sm_right * (W_s / smooth_pixel_divisor)

        scale_weight = 1.0 / (2 ** s) if smooth_per_scale else 1.0
        smooth_loss += (
            smoothness_loss_2nd_order(sm_left,  left_pyramid[s]) +
            smoothness_loss_2nd_order(sm_right, right_pyramid[s])
        ) * scale_weight

        # -- Left-right consistency with occlusion masking --
        right_to_left_disp = warp_right_to_left(disp_right[s], disp_left[s])
        left_to_right_disp = warp_left_to_right(disp_left[s],  disp_right[s])

        lr_loss += (
            torch.mean(
                torch.abs(right_to_left_disp - disp_left[s]) * left_occ_mask
            ) / left_occ_avg +
            torch.mean(
                torch.abs(left_to_right_disp - disp_right[s]) * right_occ_mask
            ) / right_occ_avg)

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


# ---------------------------------------------------------------------------
# Temporal photometric loss (for BDF temporal pairs)
# ---------------------------------------------------------------------------

def compute_temporal_photometric_loss(image_t, image_t1, flow_norm,
                                      alpha_image_loss=0.85):
    '''
    Compute temporal photometric reconstruction loss at finest scale only.

    Warps image_t1 to image_t using the predicted 2D flow, then computes
    alpha * SSIM + (1-alpha) * L1.

    Arg(s):
        image_t : torch.Tensor[float32]
            (B, 3, H, W) target image at time t
        image_t1 : torch.Tensor[float32]
            (B, 3, H, W) source image at time t+1
        flow_norm : torch.Tensor[float32]
            (B, 2, H, W) normalized flow from t to t+1
            (ch0 = fraction of width, ch1 = fraction of height)
        alpha_image_loss : float
            weight between SSIM and L1

    Returns:
        torch.Tensor[float32] : scalar loss
    '''

    image_t_est = warp_with_flow_2d(image_t1, flow_norm)

    l1_loss = torch.mean(torch.abs(image_t_est - image_t))
    ssim_loss = torch.mean(ssim(image_t_est, image_t))

    return alpha_image_loss * ssim_loss + (1 - alpha_image_loss) * l1_loss


def compute_flow_smoothness(flow_list, image, num_scales=4,
                            pixel_divisor=0.0, per_scale=True):
    '''
    Compute 2nd-order smoothness loss on multi-scale 2D flow.

    Used to add temporal flow smoothness for BDF (matching native BDF which
    computes smoothness on all 4 directional pairs including temporal flows).

    Arg(s):
        flow_list : list[torch.Tensor[float32]]
            num_scales tensors each (B, 2, H_s, W_s) normalised flow
        image : torch.Tensor[float32]
            (B, 3, H, W) reference image for edge-aware weighting
        num_scales : int
            number of scales
        pixel_divisor : float
            if > 0, convert normalised flow to pixel scale / divisor
        per_scale : bool
            if True, weight by 1/(2^s) per scale

    Returns:
        torch.Tensor[float32] : scalar smoothness loss
    '''

    image_pyramid = scale_pyramid(image, num_scales)
    loss = 0.0

    for s in range(num_scales):
        flow_s = flow_list[s]
        if pixel_divisor > 0:
            W_s = flow_s.shape[3]
            H_s = flow_s.shape[2]
            if flow_s.shape[1] >= 2:
                scale_x = W_s / pixel_divisor
                scale_y = H_s / pixel_divisor
                flow_s = torch.cat([
                    flow_s[:, 0:1] * scale_x,
                    flow_s[:, 1:] * scale_y], dim=1)
            else:
                flow_s = flow_s * (W_s / pixel_divisor)

        scale_weight = 1.0 / (2 ** s) if per_scale else 1.0
        loss += smoothness_loss_2nd_order(flow_s, image_pyramid[s]) * scale_weight

    return loss * 0.5
