# Stereo vs. Monocular Depth Completion Pipelines

This document compares the stereo pipeline (`stereo_depth_completion/`) with the monocular depth completion pipeline (`depth_completion/`) to highlight the fundamental differences in problem formulation, architecture, augmentation constraints, and loss design.

---

## Problem Formulation

### Monocular Depth Completion
- **Input**: A single RGB image + a sparse depth map (e.g., from LiDAR) + camera intrinsics
- **Output**: A dense depth map
- **Core idea**: Fill in the gaps in a sparse depth measurement using image appearance cues
- **Supervision**: Can be supervised (ground truth dense depth) or unsupervised (temporal photometric loss using a pose network)

### Stereo Depth Estimation
- **Input**: A rectified stereo pair (left + right images), optionally with temporal frames
- **Output**: Dense disparity maps (left and right), which can be converted to depth via $z = f \cdot b / d$
- **Core idea**: Estimate depth by finding horizontal correspondences between stereo views
- **Supervision**: Self-supervised only (photometric reconstruction between stereo views)

### Key Distinction
The monocular pipeline **completes** a sparse signal (LiDAR points) with help from image context. The stereo pipeline **estimates** depth purely from image correspondences, with no sparse depth input at all.

---

## Model Architectures

### Monocular Models
The monocular pipeline supports several depth completion architectures:

| Model | Input | Approach |
|-------|-------|----------|
| **KBNet** | Image + sparse depth + validity map | Multi-scale calibrated backprojection with learned kernels |
| **ScaffNet** | Sparse depth only | Scaffold network for depth-only completion |
| **FusionNet** | Image + ScaffNet output | Fuses image features with scaffold depth |
| **VOICED** | Image + sparse depth | Sparse-to-dense depth completion |

All models have a `forward_depth()` method that takes `(image, sparse_depth, validity_map, intrinsics)` and returns a dense depth prediction. For unsupervised training, they also have a `forward_pose()` method that estimates the relative camera pose between two frames.

### Stereo Models
The stereo pipeline uses two external architectures:

| Model | Input | Approach |
|-------|-------|----------|
| **BDF** (MonodepthNet) | Concatenated stereo pair (6ch) | ResNet-50 encoder-decoder with skip connections |
| **BDF** (PWCDCNet) | Concatenated stereo pair (6ch) | PWC-Net with correlation volumes and dense connections |
| **UnOS** (PWCDisp) | Separate left + right images | PWC-Net with 1-D cost volumes and coarse-to-fine warping |

Stereo models predict **disparity** (horizontal pixel shift between views), not depth directly. They inherently predict both left and right disparity maps, enabling self-consistency checks.

---

## Data and Inputs

### Monocular
```
Per sample:
  - image:          (B, 3, H, W)    RGB image
  - sparse_depth:   (B, 1, H, W)    Sparse depth from LiDAR
  - validity_map:   (B, 1, H, W)    Binary mask of valid sparse points
  - intrinsics:     (B, 3, 3)       Camera intrinsics
  - [optional] image_t-1, image_t+1: Temporal neighbors for unsupervised training
```

Sparse depth maps undergo **erosion** after geometric augmentation to remove invalid points that may have been created by interpolation near depth boundaries.

### Stereo
```
Per sample:
  - left_t:    (B, 3, H, W)    Left image at time t
  - right_t:   (B, 3, H, W)    Right image at time t
  - left_t1:   (B, 3, H, W)    Left image at time t+1
  - right_t1:  (B, 3, H, W)    Right image at time t+1
  - [UnOS only] cam2pix, pix2cam: Multi-scale intrinsic matrices
```

No sparse depth input. The stereo geometry provides the depth signal entirely through correspondence matching.

---

## Augmentation Constraints

This is one of the most important differences between the two pipelines. The stereo pipeline imposes strict constraints on geometric augmentations to preserve **epipolar geometry** (the assumption that corresponding points lie on the same horizontal scanline in rectified stereo images).

### Monocular — Allowed Augmentations
| Augmentation | Allowed? | Notes |
|-------------|----------|-------|
| Horizontal flip | Yes | |
| Vertical flip | Yes | |
| Rotation | Yes | Any angle |
| Random crop | Yes | Any position |
| Resize | Yes | |
| Horizontal translate | Yes | |
| Vertical translate | Yes | |
| Color jitter | Yes | |
| Occlusion patches | Yes | Random patch removal on image and/or depth |

The monocular pipeline treats each image independently, so any spatial transform is valid as long as the depth map, sparse depth, and image are transformed consistently.

### Stereo — Allowed Augmentations
| Augmentation | Allowed? | Why? |
|-------------|----------|------|
| Horizontal flip | Yes | But must **swap left↔right** images |
| Vertical flip | **No** | Breaks vertical alignment of epipolar lines |
| Rotation | **No** | Rotated images are no longer rectified — epipolar lines are no longer horizontal |
| Random crop | Yes | Same crop on both images preserves alignment |
| Resize | Yes | Same resize preserves rectification |
| Horizontal translate | Yes | Same shift on both images (equivalent to changing baseline) |
| Vertical translate | **No** | Misaligns vertical correspondence |
| Color jitter | Yes | Same jitter on all 4 images |

### Horizontal Flip Handling (Stereo)
When a horizontal flip is applied in the stereo pipeline, the following must happen:
1. Apply the flip to all 4 images
2. **Swap** left and right images (a flipped left image looks like a right image)
3. After disparity prediction and augmentation undo:
   - Swap `disp_left ↔ disp_right` back for flipped elements
   - For BDF's 2-channel flow: also negate the horizontal component

### Augmentation Undo and Disparity Scaling
When geometric augmentation involves resizing, the predicted disparity must be rescaled during undo:

$$d_{original} = d_{augmented} / s$$

where $s$ is the scale factor of the resize. This is because disparity (in normalized units, as a fraction of image width) is proportional to the image width. If the image is scaled up, the disparity in pixel units scales proportionally, but we need to undo this to get the correct disparity for the original resolution.

---

## Loss Functions

### Monocular Loss
The monocular pipeline supports two supervision modes:

**Supervised**:
$$\mathcal{L} = w_1 \cdot L1(d_{pred}, d_{gt}) + w_2 \cdot L2(d_{pred}, d_{gt}) + w_{sm} \cdot \mathcal{L}_{smooth}$$

with optional sparse depth supervision and weight decay regularization.

**Unsupervised**:
$$\mathcal{L} = w_{photo} \cdot \mathcal{L}_{photometric} + w_{sm} \cdot \mathcal{L}_{smooth} + w_{sparse} \cdot \mathcal{L}_{sparse}$$

The photometric loss warps temporal neighbors ($I_{t-1}$, $I_{t+1}$) to the current frame using predicted depth + predicted pose:
1. Project current frame pixels to 3D using predicted depth and intrinsics
2. Transform 3D points using predicted relative pose
3. Reproject to the source frame to get sampling coordinates
4. Compare reconstructed image with actual image

### Stereo Loss
The stereo pipeline is always self-supervised:

$$\mathcal{L} = \mathcal{L}_{rec} + w_{sm} \cdot \mathcal{L}_{smooth} + w_{lr} \cdot \mathcal{L}_{lr} + [w_{temp} \cdot \mathcal{L}_{temporal}]$$

The photometric loss warps one stereo view to the other using predicted disparity:
1. For each pixel in the left image at position $x$, look up the right image at position $x - d_L(x)$
2. Compare the reconstructed left image with the actual left image
3. Do the same for right→left

Key stereo-specific loss terms:
- **Left-right consistency**: The left disparity warped to the right view should match the right disparity, and vice versa
- **Occlusion masking**: Occluded pixels (visible in one view but not the other) are down-weighted or excluded
- **Temporal loss** (optional, BDF): Uses optical flow between consecutive frames for additional photometric supervision

### Loss Computation Location
- **Monocular**: Loss is computed inside the model wrapper (`compute_loss()`)
- **Stereo (AugUndo)**: Loss is computed *outside* the model, in `stereo_losses.py`, after augmentation undo. This separation is essential for the AugUndo approach—the model only predicts disparity in the augmented frame, and all loss computation happens in the original frame.

---

## Pose Network Role

### Monocular
The pose network is a **core component** for unsupervised training. It predicts the relative camera pose between temporal frames ($I_{t-1} \to I_t$ and $I_t \to I_{t+1}$). This pose, combined with predicted depth and known intrinsics, enables temporal photometric loss. Without the pose network, unsupervised monocular depth completion cannot work.

### Stereo
The pose network is **optional** and only used in UnOS's `depth` and `depthflow` modes. The primary depth signal comes from stereo correspondence, not from temporal motion. When present, the pose network provides supplementary supervision through rigid flow consistency.

---

## Pipeline Architecture Summary

### Monocular AugUndo Pipeline
```
Input: (image, sparse_depth, validity_map, intrinsics)
  1. Apply photometric augmentation to image
  2. Apply geometric augmentation to (image, sparse_depth, validity_map)
  3. Erode sparse depth to remove interpolation artifacts
  4. Forward pass: model.forward_depth(aug_image, aug_sparse, aug_valid, intrinsics)
  5. Undo geometric augmentation on predicted depth
  6. Compute loss in original frame (supervised or unsupervised)
  7. Backpropagate
```

### Stereo AugUndo Pipeline
```
Input: (left_t, right_t, left_t1, right_t1)
  1. Apply photometric augmentation to all 4 images
  2. Apply geometric augmentation to all 4 images (swap L/R if flipped)
  3. Forward pass: model.forward_stereo_disparity(aug_left, aug_right)
  4. Undo geometric augmentation on predicted disparity (with scale correction)
  5. Swap L/R disparity back for flipped elements
  6. Compute stereo loss in original frame (photometric + smoothness + LR)
  7. [Optional] Compute temporal loss using flow prediction
  8. Backpropagate
```

### Key Structural Differences
1. **No sparse depth**: Stereo pipeline has no sparse depth input, so no erosion step
2. **Bilateral output**: Stereo predicts both left and right disparity; mono predicts a single depth map
3. **L/R swap**: Stereo must handle the left-right swap that comes with horizontal flipping
4. **Disparity vs depth**: Stereo works in disparity space (horizontal pixel shift), mono works in depth space (meters)
5. **Augmentation restrictions**: Stereo forbids rotation and vertical transforms
6. **Temporal data**: Stereo has 4 input images (2 stereo × 2 temporal); mono has 1-3 images

---

## Evaluation Metrics

Both pipelines evaluate on KITTI benchmarks but with different metrics:

### Monocular Depth Completion
- **MAE**: Mean Absolute Error (mm)
- **RMSE**: Root Mean Squared Error (mm)
- **iMAE**: Inverse MAE (1/km)
- **iRMSE**: Inverse RMSE (1/km)
- Evaluated against semi-dense ground truth depth maps

### Stereo Depth Estimation
- **D1-all**: Percentage of pixels with disparity error > 3px and > 5% of ground truth
- **abs_rel**: Absolute relative depth error
- **sq_rel**: Squared relative depth error
- **rms**: Root mean squared depth error
- **log_rms**: Root mean squared log depth error
- **a1, a2, a3**: Accuracy thresholds ($\delta < 1.25^k$ for $k=1,2,3$)
- Evaluated on KITTI 2015 Scene Flow and KITTI 2012 Stereo benchmarks

---

## Summary Table

| Aspect | Monocular | Stereo |
|--------|-----------|--------|
| **Input** | Image + sparse depth | Stereo pair (+ temporal) |
| **Output** | Dense depth (meters) | Disparity (pixels / normalized) |
| **Supervision** | Supervised or unsupervised | Self-supervised only |
| **Sparse depth** | Yes (LiDAR) | No |
| **Pose network** | Required for unsupervised | Optional (UnOS depth/depthflow) |
| **Rotation augmentation** | Allowed | Forbidden |
| **Vertical flip** | Allowed | Forbidden |
| **L/R swap on flip** | N/A | Required |
| **Erosion after resize** | Yes (on sparse depth) | No |
| **Bilateral prediction** | No (single depth map) | Yes (left + right disparity) |
| **LR consistency loss** | N/A | Yes |
| **Occlusion masking** | Not used | Forward-warp or FB consistency |
| **Models** | KBNet, ScaffNet, FusionNet, VOICED | BDF (MonodepthNet/PWCDCNet), UnOS |
| **Loss location** | Inside model wrapper | Outside model (stereo_losses.py) |
