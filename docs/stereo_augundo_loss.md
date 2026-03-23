# Stereo AugUndo Loss Computation

This document describes the total loss computation in the stereo depth AugUndo pipeline for UnOS and BDF, and how it differs from the original monocular AugUndo loss used with VOICED.

---

## 1. Background: The AugUndo Framework

AugUndo is a data augmentation strategy for self-supervised depth estimation. The core idea is:

1. **Augment** the input images with geometric and photometric transforms
2. **Forward** the augmented images through the model to produce predictions in the augmented frame
3. **Undo** the geometric transform on the predictions, bringing them back to the original frame
4. **Compute loss** in the original (un-augmented) coordinate frame, using the original images as the photometric reconstruction target

This ensures the model learns augmentation-invariant representations while the loss signal remains anchored to the original image geometry.

---

## 2. Original Monocular AugUndo Loss (VOICED)

**File:** `depth_completion/src/depth_completion.py` (lines 493-594)

### Pipeline

```
Input: (image0, image1, image2, sparse_depth0, intrinsics)
  |
  v
[1] Crop-to-shape transform (align all inputs to target resolution)
  |
  v
[2] Geometric augmentation T_ge on (image0, sparse_depth0, validity_map0)
    -> (input_image0, input_sparse_depth0, input_validity_map0, input_intrinsics)
    Records: transform_performed_geometric
  |
  v
[3] Optional sparse depth erosion (if resize was applied)
  |
  v
[4] Point cloud augmentation (random point removal from sparse depth)
  |
  v
[5] Photometric augmentation T_ph on input_image0
    -> augmented input_image0
  |
  v
[6] Normalization of all images
  |
  v
[7] Forward pass: depth_model(input_image0, input_sparse_depth0, input_validity_map0, input_intrinsics)
    -> output_depth0 (in augmented frame)
  |
  v
[8] Undo geometric transform: T_ge^{-1}(output_depth0)
    -> output_depth0 (now in original frame)
  |
  v
[9] Pose estimation: pose_model(image0, image1), pose_model(image0, image2)
  |
  v
[10] Compute loss on un-augmented predictions vs. original images
```

### Loss Function

The monocular loss is computed by the VOICED model (`external_src/depth_completion/voiced/src/voiced_model.py`, lines 157-303). It operates entirely in the **original coordinate frame** after the undo step:

```
L_total = w_color * L_color
        + w_structure * L_structure
        + w_sparse_depth * L_sparse_depth
        + w_smoothness * L_smoothness
        + w_pose * L_pose
```

| Component | Description | Default Weight |
|-----------|-------------|----------------|
| `L_color` | L1 photometric reconstruction loss. Reprojects image1 and image2 onto image0 using predicted depth + estimated pose, then measures pixel-wise color difference. | 0.20 |
| `L_structure` | SSIM structural similarity loss between reprojected images and image0. | 0.80 |
| `L_sparse_depth` | Consistency between predicted depth and available sparse LiDAR depth. | 1.00 |
| `L_smoothness` | Edge-aware depth smoothness regularization. | 0.15 |
| `L_pose` | Forward-backward pose consistency (pose0to1 vs pose1to0). | 0.10 |

### Key Characteristics

- **Single-path computation**: Only one forward pass. Loss is computed once in the original frame.
- **Requires pose estimation**: Uses a separate pose network for ego-motion between frames.
- **Uses sparse depth supervision**: LiDAR points provide direct depth supervision via `L_sparse_depth`.
- **Intrinsics-dependent**: Camera intrinsics are used for reprojection (backproject to 3D, then project to pixel coordinates).

---

## 3. Stereo AugUndo Loss (Unified for BDF and UnOS)

**Files:**
- Training loop: `stereo_depth_completion/stereo_depth_completion.py`
- Stereo loss: `stereo_depth_completion/stereo_losses.py`
- BDF model wrapper: `stereo_depth_completion/bdf_model.py` (`forward_stereo_disparity`)
- UnOS model wrapper: `stereo_depth_completion/unos_model.py` (`forward_disparity`)

### Pipeline

Both BDF and UnOS follow the same AugUndo pipeline, matching the monocular pattern:

```
Input: (left_t, right_t, left_t1, right_t1) from native dataloader (augmentation disabled)
  |
  v
[1] Photometric augmentation T_ph on all 4 images jointly (same random params)
  |
  v
[2] Stereo-safe geometric augmentation T_ge on all 4 images jointly
    - Only horizontal flip allowed (with left<->right swap)
    - Rotation and vertical transforms prohibited (break epipolar geometry)
    Records: transform_performed
  |
  v
[3] Forward pass: model.forward_disparity(aug_left_t, aug_right_t)
    -> Produces multi-scale (disp_left, disp_right) in the augmented frame
    -> No loss computed inside the model
  |
  v
[4] Undo: T_ge^{-1} applied to disp_left and disp_right
    - Spatial layout reversed (e.g. horizontal flip undone)
    - Disparity values scaled by 1/s for resize augmentations
  |
  v
[5] Left-right swap correction for flipped batch elements
    - When T_ge included a horizontal flip, left and right images were swapped
    - The model predicted "left disp" for what was originally the right view
    - Swap disp_left and disp_right to restore original assignment
  |
  v
[6] Compute stereo loss in original frame: compute_stereo_loss(left_t, right_t, disp_left, disp_right)
    - Reconstruction targets are ORIGINAL un-augmented images
    - Disparity predictions have been undone to original geometry
```
Note: UnOS and BDF output only disparity, so depth estimation happens only during evaluation via `depth = 1 / disparity`.

### Loss Function

The stereo AugUndo loss (`stereo_losses.py`) is computed **in the original frame** using undone disparity and original images:

```
L_total = L_rec
        + w_smooth * L_smooth
        + w_lr * L_lr
        + w_temporal * L_temporal   (BDF only)
```

| Component | Description | Default Weight |
|-----------|-------------|----------------|
| `L_rec` | Bidirectional photometric reconstruction with **occlusion masking**. Warps original right image to left (and vice versa) using undone disparity, then computes `alpha * SSIM + (1-alpha) * L1`. Occluded pixels (detected via forward-warped ones) are excluded and loss is normalized by visibility. Summed over 4 scales. | 1.0 (alpha=0.85) |
| `L_smooth` | 2nd-order image-edge-aware disparity smoothness. Weights disparity curvature by `exp(-10 * \|image_gradient\|)`. Applied to both left and right disparity across 4 scales, scaled by `1/(2^s)`. | 10.0 |
| `L_lr` | Left-right disparity consistency with **occlusion masking**. Warps right disparity to the left view (using left disparity) and checks that it equals left disparity, and vice versa. Occluded pixels are excluded. Summed over 4 scales. | 1.0 |
| `L_temporal` | (BDF only) Temporal photometric reconstruction. Predicts 2D flow from `image_t` to `image_t1`, warps `image_t1` to reconstruct `image_t`, computes `alpha * SSIM + (1-alpha) * L1`. Computed at finest scale only for left and right views. | 0.1 |

### Occlusion Masking

Matches the forward-warp approach from UnOS (`monodepth_model.py`):

1. Forward-warp a ones tensor from left to right using left disparity → `right_occ_mask`
2. Forward-warp a ones tensor from right to left using right disparity → `left_occ_mask`
3. Clamp masks to [0, 1]. Pixels with mask ≈ 0 are occluded (not visible from the other view).
4. Apply masks to L1, SSIM, and LR consistency losses; normalize by `mean(mask) + 1e-12`.

The forward warp uses bilinear splatting (`scatter_add_`) and is non-differentiable (masks are detached). Gradients flow through the loss terms themselves, not through the masks.

### Model-Specific Forward Methods

**BDF** (`forward_stereo_disparity` + `forward_temporal_flow`):
- `forward_stereo_disparity`: Runs the stereo pair `(left, right)` and its reverse through MonodepthNet. Extracts horizontal flow component (ch0) as stereo disparity. Negates forward ch0 for positive left disparity; reverse ch0 is already positive right disparity.
- `forward_temporal_flow`: Runs `cat(image_t, image_t1)` through the same network to predict 2D temporal flow. Returns normalized flow (ch0 = fraction of width, ch1 = fraction of height) used for temporal photometric loss.

**UnOS** (`forward_disparity`):
- Calls `feature_pyramid_disp` and `pwc_disp` (shared with `Model_stereo`) via `disp_godard(is_training=False)`
- Skips internal loss computation but preserves gradient flow through shared weights
- Splits the 2-channel output (ch0 = left disp, ch1 = right disp)

### Warping Implementation

The stereo loss uses `F.grid_sample`-based backward warping (`stereo_losses.py`), independent of the model-specific warping used by BDF (`Resample2d`) and UnOS (`transformer_old`). For left reconstruction from right:

```
grid_x_warped = grid_x - disp_left * 2W/(W-1)
left_est = F.grid_sample(right, grid, align_corners=True)
```

For temporal reconstruction (BDF), `warp_with_flow_2d` uses full 2D flow:
```
grid_x_warped = grid_x + flow_u * 2W/(W-1)
grid_y_warped = grid_y + flow_v * 2H/(H-1)
image_t_est = F.grid_sample(image_t1, grid, align_corners=True)
```

### Key Differences from Model-Native Losses

| Aspect | Native BDF/UnOS Loss | Stereo AugUndo Loss |
|--------|---------------------|---------------------|
| **Frame** | Augmented | Original (un-augmented) |
| **Undo step** | None | T_ge^{-1} on disparity + left-right swap |
| **Reconstruction targets** | Augmented images | Original images |
| **Occlusion masking** | Forward-backward or forward-warp masks | Forward-warp masks (matching UnOS) |
| **Temporal signal (BDF)** | 4 directional pairs (stereo + temporal) | Stereo pair + temporal flow loss |
| **Warping** | Model-specific (Resample2d, transformer_old) | Unified F.grid_sample |

---

## 4. Summary of Differences

### Monocular (VOICED) vs. Stereo (BDF/UnOS) AugUndo

| Aspect | Monocular (VOICED) | Stereo AugUndo |
|--------|-------------------|----------------|
| **Undo step** | Yes -- `T_ge^{-1}` applied to depth predictions | Yes -- `T_ge^{-1}` applied to disparity + left-right swap |
| **Loss frame** | Original (un-augmented) frame | Original (un-augmented) frame |
| **Pose network** | Yes (separate PoseNet) | No |
| **Reconstruction signal** | Temporal reprojection (t-1, t+1 -> t) using depth + pose | Stereo warping (right <-> left) using disparity + temporal flow loss (BDF) |
| **Sparse depth** | Yes (LiDAR supervision) | No |
| **Number of scales** | 1 (full resolution) | 4 (multi-scale pyramid) |
| **Smoothness** | 1st-order edge-aware | 2nd-order edge-aware |
| **Intrinsics used in loss** | Yes (for 3D reprojection) | No |

### Stereo-Specific Augmentation Handling

When horizontal flip is applied in the stereo pipeline, two additional steps are required that have no analog in the monocular case:

1. **Left-right image swap during augmentation**: `apply_stereo_geometric_augmentation` swaps the left and right images when a horizontal flip is applied, because flipping reverses the stereo baseline direction.

2. **Left-right disparity swap during undo**: After applying `T_ge^{-1}` to the predicted disparity, the left and right disparity assignments must be swapped back for flipped elements. This is because the model predicted "left disparity" for what was originally the right view (and vice versa) due to the image swap in step 1.

### Augmentation Constraints

Stereo depth models impose strict constraints on geometric augmentation to preserve epipolar geometry:

- **Allowed:** Horizontal flip (with mandatory left-right image swap), resize
- **Prohibited:** Rotation, vertical flip, vertical translation

These constraints ensure that after augmentation, the left and right images remain horizontally rectified, which is the assumption underlying disparity-based stereo matching.

### Dataloader Augmentation

An important implementation detail: each model's native dataloader handles augmentation differently.

- **BDF** (`myCycleImageFolder`): Accepts a `training` flag. When `training=False` (as used by AugUndo), all built-in augmentation (random flip, gamma, brightness, color) is disabled. AugUndo then handles all augmentation on the GPU.

- **UnOS** (`MonodepthDataloader`): Originally had no `training` flag and applied random left-right flip and front-back swap unconditionally. This was fixed by adding a `training=False` parameter to gate these augmentations, preventing double augmentation when used with AugUndo.
