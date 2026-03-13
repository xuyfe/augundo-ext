from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
from .optical_flow_warp_old import transformer_old


def gray2rgb(im, cmap='gray'):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None,
                                cmap='gray'):
    """Convert depth to disparity and normalize for display."""
    depth = 1.0 / (depth + 1e-6)
    if normalizer is not None:
        depth = depth / normalizer
    else:
        depth = depth / (np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1 - crop_percent))
    depth = depth[:keep_H]
    return depth


def _pixel2cam(depth, pixel_coords, intrinsics_inv):
    """Transform coordinates in the pixel frame to the camera frame.

    Args:
        depth: [B, 1, H*W]
        pixel_coords: [B, 3, H*W]
        intrinsics_inv: [B, 3, 3]
    Returns:
        cam_coords: [B, 3, H*W]
    """
    cam_coords = torch.matmul(intrinsics_inv, pixel_coords) * depth
    return cam_coords


def _cam2pixel(cam_coords, proj_c2p):
    """Transform coordinates in the camera frame to the pixel frame.

    Args:
        cam_coords: [B, 4, H*W]
        proj_c2p: [B, 4, 4]
    Returns:
        pixel_coords: [B, 2, H*W]
    """
    pcoords = torch.matmul(proj_c2p, cam_coords)
    X = pcoords[:, 0:1, :]
    Y = pcoords[:, 1:2, :]
    Z = pcoords[:, 2:3, :]
    X_norm = X / (Z + 1e-10)
    Y_norm = Y / (Z + 1e-10)
    pixel_coords = torch.cat([X_norm, Y_norm], dim=1)
    return pixel_coords


def _meshgrid_abs(height, width, device='cpu'):
    """Meshgrid in absolute coordinates.

    Returns:
        grid: [3, H*W] tensor with (x, y, 1) coordinates
    """
    x_t = torch.linspace(0.0, float(width), width, device=device)
    y_t = torch.linspace(0.0, float(height), height, device=device)
    y_grid, x_grid = torch.meshgrid(y_t, x_t, indexing='ij')
    x_t_flat = x_grid.reshape(1, -1)
    y_t_flat = y_grid.reshape(1, -1)
    ones = torch.ones_like(x_t_flat)
    grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
    return grid


def _meshgrid_abs_xy(batch, height, width, device='cpu'):
    """Meshgrid in absolute coordinates, returning x and y separately.

    Returns:
        x: [B, H, W]
        y: [B, H, W]
    """
    x_t = torch.linspace(0.0, float(width), width, device=device)
    y_t = torch.linspace(0.0, float(height), height, device=device)
    y_grid, x_grid = torch.meshgrid(y_t, x_t, indexing='ij')
    x_out = x_grid.unsqueeze(0).expand(batch, -1, -1)
    y_out = y_grid.unsqueeze(0).expand(batch, -1, -1)
    return x_out, y_out


def _euler2mat(z, y, x):
    """Converts euler angles to rotation matrix.

    Args:
        z: rotation angle along z axis (in radians) -- [B, 1]
        y: rotation angle along y axis (in radians) -- [B, 1]
        x: rotation angle along x axis (in radians) -- [B, 1]
    Returns:
        Rotation matrix -- [B, 3, 3]
    """
    B = z.shape[0]

    z = torch.clamp(z, -np.pi, np.pi)
    y = torch.clamp(y, -np.pi, np.pi)
    x = torch.clamp(x, -np.pi, np.pi)

    # Each is [B, 1]
    cosz = torch.cos(z)
    sinz = torch.sin(z)
    cosy = torch.cos(y)
    siny = torch.sin(y)
    cosx = torch.cos(x)
    sinx = torch.sin(x)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)

    # Rotation about z: [B, 3, 3]
    rotz_1 = torch.cat([cosz, -sinz, zeros], dim=1)   # [B, 3]
    rotz_2 = torch.cat([sinz, cosz, zeros], dim=1)
    rotz_3 = torch.cat([zeros, zeros, ones], dim=1)
    zmat = torch.stack([rotz_1, rotz_2, rotz_3], dim=1)  # [B, 3, 3]

    # Rotation about y
    roty_1 = torch.cat([cosy, zeros, siny], dim=1)
    roty_2 = torch.cat([zeros, ones, zeros], dim=1)
    roty_3 = torch.cat([-siny, zeros, cosy], dim=1)
    ymat = torch.stack([roty_1, roty_2, roty_3], dim=1)

    # Rotation about x
    rotx_1 = torch.cat([ones, zeros, zeros], dim=1)
    rotx_2 = torch.cat([zeros, cosx, -sinx], dim=1)
    rotx_3 = torch.cat([zeros, sinx, cosx], dim=1)
    xmat = torch.stack([rotx_1, rotx_2, rotx_3], dim=1)

    rotMat = torch.matmul(torch.matmul(xmat, ymat), zmat)
    return rotMat


def _pose_vec2mat(vec):
    """Converts 6DoF parameters to transformation matrix.

    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    batch_size = vec.shape[0]
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rx = vec[:, 3:4]  # [B, 1]
    ry = vec[:, 4:5]
    rz = vec[:, 5:6]
    rot_mat = _euler2mat(rz, ry, rx)  # [B, 3, 3]
    filler = torch.tensor([0.0, 0.0, 0.0, 1.0], device=vec.device).reshape(
        1, 1, 4).expand(batch_size, -1, -1)
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    transform_mat = torch.cat([transform_mat, filler], dim=1)  # [B, 4, 4]
    return transform_mat


def inverse_warp(depth, pose, intrinsics, intrinsics_inv,
                 pose_mat_inverse=False):
    """Inverse warp a source image to the target image plane.

    Args:
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6] or [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
        pose_mat_inverse: whether to invert the pose matrix
    Returns:
        flow: optical flow induced by depth and pose -- [B, H, W, 2]
        pose_mat: the 4x4 pose matrix -- [B, 4, 4]
    """
    batch_size, img_height, img_width = depth.shape
    device = depth.device

    depth_flat = depth.reshape(batch_size, 1, img_height * img_width)
    grid = _meshgrid_abs(img_height, img_width, device=device)
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 3, H*W]

    cam_coords = _pixel2cam(depth_flat, grid, intrinsics_inv)
    ones = torch.ones(batch_size, 1, img_height * img_width, device=device)
    cam_coords_hom = torch.cat([cam_coords, ones], dim=1)  # [B, 4, H*W]

    if pose.dim() == 3:
        pose_mat = pose
    else:
        pose_mat = _pose_vec2mat(pose)

    if pose_mat_inverse:
        pose_mat = torch.inverse(pose_mat)

    # Build 4x4 intrinsics
    hom_filler = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=device).reshape(1, 1, 4).expand(
        batch_size, -1, -1)
    intrinsics_4 = torch.cat(
        [intrinsics, torch.zeros(batch_size, 3, 1, device=device)], dim=2)
    intrinsics_4 = torch.cat([intrinsics_4, hom_filler], dim=1)

    proj_cam_to_src_pixel = torch.matmul(intrinsics_4, pose_mat)
    src_pixel_coords = _cam2pixel(cam_coords_hom, proj_cam_to_src_pixel)
    src_pixel_coords = src_pixel_coords.reshape(
        batch_size, 2, img_height, img_width)
    src_pixel_coords = src_pixel_coords.permute(0, 2, 3, 1)  # [B, H, W, 2]

    tgt_x, tgt_y = _meshgrid_abs_xy(batch_size, img_height, img_width,
                                     device=device)
    flow_x = src_pixel_coords[:, :, :, 0] - tgt_x
    flow_y = src_pixel_coords[:, :, :, 1] - tgt_y
    flow = torch.stack([flow_x, flow_y], dim=-1)  # [B, H, W, 2]

    return flow, pose_mat


def calculate_pose_basis(cam_coords1, cam_coords2, weights, batch_size=None):
    """Given two point clouds and weights, find the transformation that
    minimizes the distance between the two clouds using SVD.

    Args:
        cam_coords1: point cloud 1 -- [B, 3, N]
        cam_coords2: point cloud 2 -- [B, 3, N]
        weights: weights for alignment -- [B, 1, N]
        batch_size: (unused, kept for API compatibility)
    Returns:
        rigid_pose_mat: transformation matrix -- [B, 4, 4]
    """
    if batch_size is None:
        batch_size = cam_coords1.shape[0]
    device = cam_coords1.device

    # Weighted centroids: [B, 3, 1]
    centroids1 = torch.sum(
        cam_coords1 * weights, dim=2, keepdim=True) / torch.sum(
        weights, dim=2, keepdim=True)
    centroids2 = torch.sum(
        cam_coords2 * weights, dim=2, keepdim=True) / torch.sum(
        weights, dim=2, keepdim=True)

    # Centered coordinates
    # cam_coords1_shifted: [B, N, 3, 1]
    cam_coords1_shifted = (cam_coords1 - centroids1).permute(
        0, 2, 1).unsqueeze(-1)
    # cam_coords2_shifted: [B, N, 1, 3]
    cam_coords2_shifted = (cam_coords2 - centroids2).permute(
        0, 2, 1).unsqueeze(-2)

    # weights_trans: [B, N, 1, 1]
    weights_trans = weights.permute(0, 2, 1).unsqueeze(-1)

    # H = sum over N of (p1 * p2^T * w): [B, 3, 3]
    H = torch.sum(
        torch.matmul(cam_coords1_shifted, cam_coords2_shifted) * weights_trans,
        dim=1)

    # SVD
    U, S, Vh = torch.linalg.svd(H)
    # R = V * U^T
    R = torch.matmul(Vh.transpose(-1, -2), U.transpose(-1, -2))

    # T = -R * centroid1 + centroid2: [B, 3, 1]
    T = -torch.matmul(R, centroids1) + centroids2

    filler = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=device).reshape(1, 1, 4).expand(
        batch_size, -1, -1)
    rigid_pose_mat = torch.cat(
        [torch.cat([R, T], dim=2), filler], dim=1)  # [B, 4, 4]

    return rigid_pose_mat


def inverse_warp_new(depth1, depth2, pose, intrinsics, intrinsics_inv,
                     flow_input, occu_mask, pose_mat_inverse=False):
    """Inverse warp with pose refinement via rigid alignment.

    Args:
        depth1: depth map of the target image -- [B, H, W]
        depth2: depth map of the source image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6] or [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
        flow_input: flow between target and source image -- [B, H, W, 2]
        occu_mask: occlusion mask of target image -- [B, H, W, 1]
        pose_mat_inverse: whether to invert the pose matrix
    Returns:
        flow: optical flow induced by refined pose -- [B, H, W, 2]
        pose_mat2: refined pose matrix -- [B, 4, 4]
        disp1_trans: transformed disparity -- [B, H, W, 1]
        small_mask: mask of points used for rigid alignment -- [B, H, W, 1]
    """
    batch_size, img_height, img_width = depth1.shape
    device = depth1.device

    depth1_flat = depth1.reshape(batch_size, 1, img_height * img_width)
    grid = _meshgrid_abs(img_height, img_width, device=device)
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1)

    # Point Cloud Q_1
    cam_coords1 = _pixel2cam(depth1_flat, grid, intrinsics_inv)
    ones = torch.ones(batch_size, 1, img_height * img_width, device=device)
    cam_coords1_hom = torch.cat([cam_coords1, ones], dim=1)  # [B, 4, H*W]

    if pose.dim() == 3:
        pose_mat = pose
    else:
        pose_mat = _pose_vec2mat(pose)

    if pose_mat_inverse:
        pose_mat = torch.inverse(pose_mat)

    # Point Cloud \hat{Q_1}: transform by initial pose
    cam_coords1_trans = torch.matmul(pose_mat, cam_coords1_hom)[:, 0:3, :]

    # Point Cloud Q_2: build from depth2 and warp by optical flow
    depth2_flat = depth2.reshape(batch_size, 1, img_height * img_width)
    cam_coords2 = _pixel2cam(depth2_flat, grid, intrinsics_inv)
    cam_coords2 = cam_coords2.reshape(batch_size, 3, img_height, img_width)
    # transformer_old expects NHWC
    cam_coords2_nhwc = cam_coords2.permute(0, 2, 3, 1)
    cam_coords2_warped = transformer_old(
        cam_coords2_nhwc, flow_input, [img_height, img_width])
    # Back to [B, 3, H*W]
    cam_coords2_trans = cam_coords2_warped.permute(0, 3, 1, 2).reshape(
        batch_size, 3, -1)

    # Process occlusion mask
    occu_mask_flat = occu_mask.reshape(batch_size, 1, -1)
    occu_mask_weighted = torch.where(
        occu_mask_flat < 0.75,
        torch.ones_like(occu_mask_flat) * 10000.0,
        torch.ones_like(occu_mask_flat))

    # Distance between transformed point clouds
    diff2 = torch.sqrt(
        torch.sum(
            (cam_coords1_trans - cam_coords2_trans) ** 2, dim=1,
            keepdim=True)) * occu_mask_weighted

    # 25th percentile threshold
    percentile_val = torch.quantile(diff2, 0.25, dim=2, keepdim=True)
    small_mask = torch.where(
        diff2 < percentile_val,
        torch.ones_like(diff2),
        torch.zeros_like(diff2))

    # Delta T via SVD alignment
    rigid_pose_mat = calculate_pose_basis(
        cam_coords1_trans, cam_coords2_trans, small_mask, batch_size)

    # T' = deltaT x T
    pose_mat2 = torch.matmul(rigid_pose_mat, pose_mat)

    # Build 4x4 intrinsics
    hom_filler = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=device).reshape(1, 1, 4).expand(
        batch_size, -1, -1)
    intrinsics_4 = torch.cat(
        [intrinsics, torch.zeros(batch_size, 3, 1, device=device)], dim=2)
    intrinsics_4 = torch.cat([intrinsics_4, hom_filler], dim=1)

    proj_cam_to_src_pixel = torch.matmul(intrinsics_4, pose_mat2)
    src_pixel_coords = _cam2pixel(cam_coords1_hom, proj_cam_to_src_pixel)
    src_pixel_coords = src_pixel_coords.reshape(
        batch_size, 2, img_height, img_width)
    src_pixel_coords = src_pixel_coords.permute(0, 2, 3, 1)  # [B, H, W, 2]

    tgt_x, tgt_y = _meshgrid_abs_xy(batch_size, img_height, img_width,
                                     device=device)
    flow_x = src_pixel_coords[:, :, :, 0] - tgt_x
    flow_y = src_pixel_coords[:, :, :, 1] - tgt_y
    flow = torch.stack([flow_x, flow_y], dim=-1)

    # Transformed disparity
    cam_coords1_trans_z = torch.matmul(
        pose_mat2, cam_coords1_hom)[:, 2:3, :]
    cam_coords1_trans_z = cam_coords1_trans_z.reshape(
        batch_size, img_height, img_width, 1)
    disp1_trans = 1.0 / cam_coords1_trans_z

    small_mask_out = small_mask.reshape(batch_size, img_height, img_width, 1)

    return flow, pose_mat2, disp1_trans, small_mask_out
