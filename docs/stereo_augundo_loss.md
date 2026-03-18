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

## 3. Stereo AugUndo Loss: BDF

**Files:**
- Training loop: `stereo_depth_completion/stereo_depth_completion.py` (lines 429-494)
- Model + loss: `stereo_depth_completion/bdf_model.py` (lines 86-300)

### Pipeline

```
Input: (left_t, right_t, left_t1, right_t1) from myCycleImageFolder(training=False)
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
[3] Forward pass: BDFModel.forward(aug_left_t, aug_right_t, aug_left_t1, aug_right_t1)
    Internally builds 4 directional pairs:
      former = [left_t1, left_t, right_t, left_t]
      latter = [right_t1, left_t1, right_t1, right_t]
    Runs MonodepthNet on cat(former, latter) and its reverse
    -> Returns multi-scale disparity estimates (disp_est, disp_est_2)
  |
  v
[4] Compute loss: BDFModel.compute_loss(output, original_batch)
    Loss uses the AUGMENTED images (former/latter from the forward output)
    as reconstruction targets -- NOT the un-augmented originals.
```

### Loss Function

The BDF loss (`bdf_model.py`, lines 153-300) is computed **in the augmented frame** using the augmented images that were fed to the model:

```
L_total = L_image + L_image_2
        + 10 * (L_disp_gradient + L_disp_gradient_2)
        + w_lr * L_lr
        + [optional L_2warp]
```

| Component | Description | Weight |
|-----------|-------------|--------|
| `L_image` | Forward photometric reconstruction (right-to-left warp). Combines `alpha * SSIM + (1-alpha) * L1`, masked by forward-backward occlusion masks. | 1.0 |
| `L_image_2` | Reverse photometric reconstruction (left-to-right warp). Same formulation as `L_image`. | 1.0 |
| `L_disp_gradient` | Forward 2nd-order image-edge-aware disparity smoothness (`cal_grad2_error`). | 10.0 |
| `L_disp_gradient_2` | Reverse 2nd-order disparity smoothness. | 10.0 |
| `L_lr` | Left-right disparity consistency: the left disparity warped to the right view should match the right disparity, and vice versa. Applied only to stereo pair indices `[0,1,6,7]`. | 0.5 (default) |
| `L_2warp` | Optional two-warp consistency loss (controlled by `type_of_2warp`). Chains two warps to enforce temporal-stereo consistency. Types 1, 2, 3 represent different warp orderings. | 0.1 |

### Multi-Directional Architecture

BDF processes **4 directional pairs** simultaneously in a single forward pass, yielding 8 disparity maps per scale (4 forward + 4 reverse):

| Index | Pair | Direction |
|-------|------|-----------|
| 0-1 | left_t1 -> right_t1 | Stereo (time t+1) |
| 2-3 | left_t -> left_t1 | Temporal (left) |
| 4-5 | right_t -> right_t1 | Temporal (right) |
| 6-7 | left_t -> right_t | Stereo (time t) |

The forward-backward occlusion masks (`get_mask`) are computed from forward and reverse disparity to handle occluded regions. Stereo pairs (indices 0,1,6,7) use unmasked loss (mask set to 1) since stereo occlusions are handled by LR consistency.

### Key Difference from Monocular

**No undo step and no pose network.** BDF's loss operates entirely in the augmented frame. The `compute_loss` method receives the augmented `former`/`latter` images from `forward()` and computes photometric reconstruction by warping between these augmented views. Since the loss is self-contained within the augmented frame (warp right to reconstruct left, and vice versa), geometric augmentation affects all views equally and the loss remains valid without needing to undo.

The `batch` dict containing original images is passed to `compute_loss` but is only used for determining batch dimensions (indexing occlusion masks), not for reconstruction targets.

---

## 4. Stereo AugUndo Loss: UnOS

**Files:**
- Training loop: `stereo_depth_completion/stereo_depth_completion.py` (lines 465-492)
- Model wrapper: `stereo_depth_completion/unos_model.py`
- Core model: `external_src/stereo_depth_completion/UnOS/models.py` (`Model_stereo`, lines 63-96)
- Loss computation: `external_src/stereo_depth_completion/UnOS/monodepth_model.py` (`MonodepthModel.build_losses`, lines 274-366)

### Pipeline

```
Input: (left_t, right_t, left_t1, right_t1, cam2pix, pix2cam)
       from MonodepthDataloader(training=False)
  |
  v
[1] Photometric augmentation T_ph on all 4 images jointly
  |
  v
[2] Stereo-safe geometric augmentation T_ge on all 4 images jointly
    (same constraints as BDF: only horizontal flip + left<->right swap)
    Records: transform_performed
  |
  v
[3] Forward pass: UnOSModel.forward(aug_left_t, aug_right_t, ..., cam2pix, pix2cam)
    -> Model_stereo computes disparity AND loss internally
    -> Returns (loss, loss_info)
  |
  v
[4] Backpropagate loss directly (no undo step)
```

### Loss Function

The UnOS stereo loss is computed inside `MonodepthModel.build_losses` (lines 274-366). It operates **in the augmented frame**, using the augmented stereo pair that was fed to the model:

```
L_total = L_image
        + w_smooth * L_disp_gradient
        + w_lr * L_lr
```

| Component | Description | Weight |
|-----------|-------------|--------|
| `L_image` | Bidirectional photometric reconstruction. Warps right image to left (and vice versa) using predicted disparity, then computes `alpha * SSIM + (1-alpha) * L1`. Masked by forward-warped occlusion masks with normalization. Summed over 4 scales. | 1.0 |
| `L_disp_gradient` | 2nd-order image-edge-aware disparity smoothness. Applied to both left and right disparities across 4 scales (8 terms each). Averaged and scaled by `1/(2^s)` per scale. | `depth_smooth_weight` (default: 10.0) |
| `L_lr` | Left-right disparity consistency. The right disparity warped to the left view should equal the left disparity, and vice versa. Masked by occlusion masks. Summed over 4 scales. | `lr_loss_weight` (fixed: 1.0) |

### Model Architecture Detail

`Model_stereo` calls `disp_godard()` which:
1. Creates a temporary `MonodepthModel` instance (sharing the `pwc_disp` network)
2. Runs `build_outputs()`: PWCDisp forward pass producing 4-scale disparity maps, warped images, occlusion masks, LR consistency maps, and smoothness terms
3. Runs `build_losses()`: computes the total loss from those outputs

The `PWCDispDecoder` outputs **normalized disparity** (fraction of image width). The normalization happens at the decoder level: `flow[:, 0:1, :, :] / (W / 4.0)`. This normalized disparity is converted to pixel-space flow for warping via `generate_flow_left`: `flow_x = -disp * W`.

### Key Difference from Monocular

Like BDF, **no undo step.** The loss is computed entirely within `Model_stereo.forward()` in the augmented frame. The stereo pair provides its own reconstruction signal (warp right to left and vice versa) without needing a pose network or camera intrinsics for the loss. Camera intrinsics (`cam2pix`, `pix2cam`) are passed through the wrapper but are **not used** by `Model_stereo` in stereo mode -- they would only be needed for optical flow / ego-motion estimation in `Model_depthflow`.

---

## 5. Summary of Differences

### Monocular (VOICED) vs. Stereo (BDF/UnOS) AugUndo

| Aspect | Monocular (VOICED) | Stereo (BDF) | Stereo (UnOS) |
|--------|-------------------|--------------|---------------|
| **Undo step** | Yes -- `T_ge^{-1}` applied to predictions before loss | No | No |
| **Loss frame** | Original (un-augmented) frame | Augmented frame | Augmented frame |
| **Pose network** | Yes (separate PoseNet) | No | No |
| **Reconstruction signal** | Temporal reprojection (t-1, t+1 -> t) using depth + pose | Stereo warping (right <-> left) using disparity | Stereo warping (right <-> left) using disparity |
| **Sparse depth** | Yes (LiDAR supervision) | No | No |
| **Multi-directional** | Single reference view (image0) | 4 directional pairs (stereo + temporal, both directions) | Single stereo pair |
| **Number of scales** | 1 (full resolution) | 4 (multi-scale pyramid) | 4 (multi-scale pyramid) |
| **Occlusion handling** | Validity maps from sparse depth | Forward-backward consistency masks | Forward-warped occlusion masks |
| **Smoothness** | 1st-order edge-aware | 2nd-order edge-aware | 2nd-order edge-aware |
| **Intrinsics used in loss** | Yes (for 3D reprojection) | No | No |

### Why No Undo Step in Stereo?

In the monocular pipeline, the loss requires computing reprojection from temporal neighbors (image1, image2) onto the reference frame (image0). These neighboring images are **not geometrically augmented** in the same way as the reference -- only the reference image and its sparse depth pass through `T_ge`. Therefore, the predicted depth must be undone (`T_ge^{-1}`) before it can be used for reprojection against the original neighbors.

In the stereo pipeline, **all images are augmented identically** with the same geometric transform (including left-right swap for horizontal flip). The loss is computed by warping between images that share the same augmented coordinate frame. Since the disparity is predicted and evaluated within this shared frame, no undo is needed -- the stereo reconstruction loss is inherently equivariant to the joint augmentation.

### Augmentation Constraints

Stereo depth models impose strict constraints on geometric augmentation to preserve epipolar geometry:

- **Allowed:** Horizontal flip (with mandatory left-right image swap), resize
- **Prohibited:** Rotation, vertical flip, vertical translation

These constraints ensure that after augmentation, the left and right images remain horizontally rectified, which is the assumption underlying disparity-based stereo matching.

### Dataloader Augmentation

An important implementation detail: each model's native dataloader handles augmentation differently.

- **BDF** (`myCycleImageFolder`): Accepts a `training` flag. When `training=False` (as used by AugUndo), all built-in augmentation (random flip, gamma, brightness, color) is disabled. AugUndo then handles all augmentation on the GPU.

- **UnOS** (`MonodepthDataloader`): Originally had no `training` flag and applied random left-right flip and front-back swap unconditionally. This was fixed by adding a `training=False` parameter to gate these augmentations, preventing double augmentation when used with AugUndo.
