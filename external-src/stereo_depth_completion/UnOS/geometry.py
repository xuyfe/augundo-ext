"""
Geometric operations for UnOS stereo depth estimation.

Ported from the original TensorFlow utils.py.
Provides Euler angle to rotation matrix conversion, pose vector to
transformation matrix, inverse warping using depth and pose, and
SVD-based rigid pose refinement.

All tensors use PyTorch NCHW format where applicable.
"""

import torch
import torch.nn.functional as F


def euler2mat(z, y, x):
    """Convert Euler angles to a 3x3 rotation matrix (batch).

    Uses the ZYX convention: R = Rz * Ry * Rx.

    Args:
        z: Rotation around z-axis (yaw), shape N.
        y: Rotation around y-axis (pitch), shape N.
        x: Rotation around x-axis (roll), shape N.

    Returns:
        Rotation matrix, shape N x 3 x 3.
    """
    N = z.shape[0]

    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)

    # Rotation matrix around z-axis
    Rz = torch.stack([
        cz, -sz, zeros,
        sz, cz, zeros,
        zeros, zeros, ones
    ], dim=1).reshape(N, 3, 3)

    # Rotation matrix around y-axis
    Ry = torch.stack([
        cy, zeros, sy,
        zeros, ones, zeros,
        -sy, zeros, cy
    ], dim=1).reshape(N, 3, 3)

    # Rotation matrix around x-axis
    Rx = torch.stack([
        ones, zeros, zeros,
        zeros, cx, -sx,
        zeros, sx, cx
    ], dim=1).reshape(N, 3, 3)

    # Combined rotation: R = Rz @ Ry @ Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))

    return R


def pose_vec2mat(vec):
    """Convert a 6-DoF pose vector to a 4x4 transformation matrix.

    Args:
        vec: Pose vector [tx, ty, tz, rx, ry, rz], shape N x 6.
             First 3 elements are translation, last 3 are Euler angles.

    Returns:
        Transformation matrix, shape N x 4 x 4.
    """
    N = vec.shape[0]

    translation = vec[:, :3].unsqueeze(-1)  # N x 3 x 1
    rx = vec[:, 3]
    ry = vec[:, 4]
    rz = vec[:, 5]

    rot = euler2mat(rz, ry, rx)  # N x 3 x 3

    # Build 4x4 matrix
    transform = torch.zeros(N, 4, 4, dtype=vec.dtype, device=vec.device)
    transform[:, :3, :3] = rot
    transform[:, :3, 3:4] = translation
    transform[:, 3, 3] = 1.0

    return transform


def inverse_warp(depth, pose, intrinsics):
    """Compute the flow field induced by depth and camera pose.

    Projects pixels from frame 1 into frame 2 using the depth map
    and relative camera pose, returning the resulting 2D flow field.

    Args:
        depth: Depth map, shape N x 1 x H x W.
        pose: 6-DoF pose vector, shape N x 6.
        intrinsics: Camera intrinsic matrix, shape N x 3 x 3.

    Returns:
        flow: Induced flow field, shape N x 2 x H x W.
        pose_mat: 4x4 transformation matrix, shape N x 4 x 4.
    """
    N, _, H, W = depth.shape

    pose_mat = pose_vec2mat(pose)  # N x 4 x 4

    # Create pixel coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(0, H, dtype=depth.dtype, device=depth.device),
        torch.arange(0, W, dtype=depth.dtype, device=depth.device),
        indexing='ij'
    )

    # Homogeneous pixel coordinates: 3 x (H*W)
    ones = torch.ones_like(x_coords)
    pixel_coords = torch.stack([x_coords, y_coords, ones], dim=0)  # 3 x H x W
    pixel_coords = pixel_coords.reshape(3, -1)  # 3 x (H*W)
    pixel_coords = pixel_coords.unsqueeze(0).expand(N, -1, -1)  # N x 3 x (H*W)

    # Unproject to camera coordinates: K^{-1} * pixel_coords * depth
    intrinsics_inv = torch.inverse(intrinsics)  # N x 3 x 3
    cam_coords = torch.bmm(intrinsics_inv, pixel_coords)  # N x 3 x (H*W)

    depth_flat = depth.reshape(N, 1, -1)  # N x 1 x (H*W)
    cam_coords = cam_coords * depth_flat  # N x 3 x (H*W)

    # Add homogeneous coordinate
    ones_row = torch.ones(N, 1, H * W, dtype=depth.dtype, device=depth.device)
    cam_coords_hom = torch.cat([cam_coords, ones_row], dim=1)  # N x 4 x (H*W)

    # Transform to target frame
    cam_coords_transformed = torch.bmm(pose_mat, cam_coords_hom)  # N x 4 x (H*W)

    # Project to pixel coordinates
    proj_coords = torch.bmm(intrinsics, cam_coords_transformed[:, :3, :])  # N x 3 x (H*W)

    # Normalize by depth (z)
    z = proj_coords[:, 2:3, :].clamp(min=1e-8)
    proj_x = proj_coords[:, 0:1, :] / z
    proj_y = proj_coords[:, 1:2, :] / z

    # Compute flow as displacement from original pixel coordinates
    flow_x = (proj_x - pixel_coords[:, 0:1, :]).reshape(N, 1, H, W)
    flow_y = (proj_y - pixel_coords[:, 1:2, :]).reshape(N, 1, H, W)

    flow = torch.cat([flow_x, flow_y], dim=1)  # N x 2 x H x W

    return flow, pose_mat


def calculate_pose_basis(cam_coords1, cam_coords2, weights):
    """Compute rigid alignment between two sets of 3D points using SVD.

    Finds the rigid transformation (rotation + translation) that best
    aligns cam_coords1 to cam_coords2 in a weighted least-squares sense.

    Args:
        cam_coords1: Source 3D points, shape N x 3 x M.
        cam_coords2: Target 3D points, shape N x 3 x M.
        weights: Per-point weights, shape N x 1 x M.

    Returns:
        Rigid transformation matrix, shape N x 4 x 4.
    """
    N = cam_coords1.shape[0]

    # Normalize weights
    weight_sum = weights.sum(dim=2, keepdim=True).clamp(min=1e-8)  # N x 1 x 1
    w = weights / weight_sum  # N x 1 x M

    # Weighted centroids
    centroid1 = (cam_coords1 * w).sum(dim=2, keepdim=True)  # N x 3 x 1
    centroid2 = (cam_coords2 * w).sum(dim=2, keepdim=True)  # N x 3 x 1

    # Center the points
    p1 = cam_coords1 - centroid1  # N x 3 x M
    p2 = cam_coords2 - centroid2  # N x 3 x M

    # Weighted cross-covariance matrix
    # H = p1 * w @ p2^T  -> N x 3 x 3
    H = torch.bmm(p1 * w, p2.transpose(1, 2))  # N x 3 x 3

    # SVD
    U, S, Vh = torch.linalg.svd(H)  # U: Nx3x3, S: Nx3, Vh: Nx3x3

    # Rotation
    # R = V @ U^T, with correction for reflection
    V = Vh.transpose(1, 2)
    Ut = U.transpose(1, 2)

    # Determinant check to handle reflections
    det = torch.det(torch.bmm(V, Ut))  # N
    sign_mat = torch.eye(3, dtype=cam_coords1.dtype, device=cam_coords1.device)
    sign_mat = sign_mat.unsqueeze(0).expand(N, -1, -1).clone()
    sign_mat[:, 2, 2] = torch.sign(det)

    R = torch.bmm(torch.bmm(V, sign_mat), Ut)  # N x 3 x 3

    # Translation
    t = centroid2 - torch.bmm(R, centroid1)  # N x 3 x 1

    # Build 4x4 matrix
    transform = torch.zeros(N, 4, 4, dtype=cam_coords1.dtype, device=cam_coords1.device)
    transform[:, :3, :3] = R
    transform[:, :3, 3:4] = t
    transform[:, 3, 3] = 1.0

    return transform


def inverse_warp_new(depth1, depth2, pose, intrinsics, flow_input, occu_mask):
    """Refined inverse warp with SVD-based pose refinement.

    Performs inverse warping with an additional SVD-based refinement step
    that uses optical flow and occlusion masks to improve the pose estimate.

    Args:
        depth1: Depth map of frame 1, shape N x 1 x H x W.
        depth2: Depth map of frame 2, shape N x 1 x H x W.
        pose: Initial 6-DoF pose vector, shape N x 6.
        intrinsics: Camera intrinsics, shape N x 3 x 3.
        flow_input: Optical flow from frame 1 to frame 2, shape N x 2 x H x W.
        occu_mask: Occlusion mask, shape N x 1 x H x W. 1=visible, 0=occluded.

    Returns:
        flow: Refined flow field, shape N x 2 x H x W.
        refined_pose_mat: Refined 4x4 pose matrix, shape N x 4 x 4.
        disp1_trans: Transformed disparity from frame 1 to frame 2 coordinates,
                     shape N x 1 x H x W.
        small_mask: Validity mask for regions with consistent depth,
                    shape N x 1 x H x W.
    """
    N, _, H, W = depth1.shape

    pose_mat = pose_vec2mat(pose)  # N x 4 x 4

    # Create pixel coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(0, H, dtype=depth1.dtype, device=depth1.device),
        torch.arange(0, W, dtype=depth1.dtype, device=depth1.device),
        indexing='ij'
    )

    ones = torch.ones_like(x_coords)
    pixel_coords = torch.stack([x_coords, y_coords, ones], dim=0)  # 3 x H x W
    pixel_coords_flat = pixel_coords.reshape(3, -1).unsqueeze(0).expand(N, -1, -1)  # N x 3 x (H*W)

    # Unproject frame 1 pixels to 3D
    intrinsics_inv = torch.inverse(intrinsics)
    cam_coords1 = torch.bmm(intrinsics_inv, pixel_coords_flat)  # N x 3 x (H*W)
    depth1_flat = depth1.reshape(N, 1, -1)
    cam_coords1 = cam_coords1 * depth1_flat  # N x 3 x (H*W)

    # Use optical flow to get corresponding points in frame 2
    flow_x = flow_input[:, 0:1, :, :]  # N x 1 x H x W
    flow_y = flow_input[:, 1:2, :, :]  # N x 1 x H x W

    # Target pixel coordinates from flow
    target_x = (pixel_coords[0:1, :, :].unsqueeze(0) + flow_x)  # N x 1 x H x W
    target_y = (pixel_coords[1:2, :, :].unsqueeze(0) + flow_y)  # N x 1 x H x W

    # Sample depth2 at target locations using backward warping
    from .warping import backward_warp
    depth2_sampled = backward_warp(depth2, flow_input)  # N x 1 x H x W

    # Unproject frame 2 pixels to 3D using sampled depth
    target_ones = torch.ones_like(target_x)
    target_pixel_coords = torch.cat([target_x, target_y, target_ones], dim=1)  # N x 3 x H x W
    target_pixel_flat = target_pixel_coords.reshape(N, 3, -1)  # N x 3 x (H*W)

    cam_coords2 = torch.bmm(intrinsics_inv, target_pixel_flat)  # N x 3 x (H*W)
    depth2_flat = depth2_sampled.reshape(N, 1, -1)
    cam_coords2 = cam_coords2 * depth2_flat  # N x 3 x (H*W)

    # Occlusion mask as weights for SVD
    weights = occu_mask.reshape(N, 1, -1)  # N x 1 x (H*W)

    # SVD-based pose refinement
    refined_pose_mat = calculate_pose_basis(cam_coords1, cam_coords2, weights)  # N x 4 x 4

    # Compute refined flow using the refined pose
    ones_row = torch.ones(N, 1, H * W, dtype=depth1.dtype, device=depth1.device)
    cam_coords1_hom = torch.cat([cam_coords1, ones_row], dim=1)  # N x 4 x (H*W)

    cam_coords1_transformed = torch.bmm(refined_pose_mat, cam_coords1_hom)  # N x 4 x (H*W)

    # Project to pixel coordinates
    proj_coords = torch.bmm(intrinsics, cam_coords1_transformed[:, :3, :])  # N x 3 x (H*W)

    z = proj_coords[:, 2:3, :].clamp(min=1e-8)
    proj_x = proj_coords[:, 0:1, :] / z
    proj_y = proj_coords[:, 1:2, :] / z

    # Flow
    flow_x_refined = (proj_x - pixel_coords_flat[:, 0:1, :]).reshape(N, 1, H, W)
    flow_y_refined = (proj_y - pixel_coords_flat[:, 1:2, :]).reshape(N, 1, H, W)
    flow = torch.cat([flow_x_refined, flow_y_refined], dim=1)  # N x 2 x H x W

    # Compute transformed depth/disparity
    # Depth of transformed points (z-coordinate)
    depth1_transformed = cam_coords1_transformed[:, 2:3, :].reshape(N, 1, H, W)
    # Convert to disparity (inverse depth)
    disp1_trans = 1.0 / depth1_transformed.clamp(min=1e-8)

    # Small mask: check for consistency between transformed depth and depth2
    depth2_at_proj = backward_warp(depth2, flow)
    depth_diff = torch.abs(depth1_transformed - depth2_at_proj)
    small_mask = (depth_diff < 0.1 * depth1_transformed).float() * occu_mask

    return flow, refined_pose_mat, disp1_trans, small_mask
