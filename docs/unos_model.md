# UnOS (Unified Optical flow and Stereo) Model

## Overview

UnOS is a unified framework that jointly learns stereo disparity, optical flow, and camera pose estimation from unlabeled stereo video sequences. The key idea is that stereo matching (disparity), temporal correspondence (optical flow), and ego-motion (pose) are all related geometric problems that can share representations and supervise each other through geometric consistency constraints.

At its core, UnOS uses a **PWC-Net** (Pyramid, Warping, Cost volume) architecture adapted separately for stereo disparity and optical flow. These two branches share the same architectural blueprint but have independent weights and differ in one critical way: the stereo disparity branch searches for correspondences only along the horizontal axis (since rectified stereo pairs differ only horizontally), while the optical flow branch searches in 2D.

---

## Network Architecture

### Feature Pyramid (Shared Blueprint)

Both the disparity and flow branches begin with a **feature pyramid encoder** that extracts multi-scale features from an input image. The architecture is identical between `FeaturePyramidDisp` and `FeaturePyramidFlow` (though they have independent weights):

```
Input image (3 channels)
  → Conv(3→16, stride 2) + LeakyReLU(0.1)
  → Conv(16→16, stride 1) + LeakyReLU(0.1)    → Level 1 features (16 ch, 1/2 res)
  → Conv(16→32, stride 2) + LeakyReLU(0.1)
  → Conv(32→32, stride 1) + LeakyReLU(0.1)    → Level 2 features (32 ch, 1/4 res)
  → Conv(32→64, stride 2) + LeakyReLU(0.1)
  → Conv(64→64, stride 1) + LeakyReLU(0.1)    → Level 3 features (64 ch, 1/8 res)
  → Conv(64→96, stride 2) + LeakyReLU(0.1)
  → Conv(96→96, stride 1) + LeakyReLU(0.1)    → Level 4 features (96 ch, 1/16 res)
  → Conv(96→128, stride 2) + LeakyReLU(0.1)
  → Conv(128→128, stride 1) + LeakyReLU(0.1)  → Level 5 features (128 ch, 1/32 res)
  → Conv(128→192, stride 2) + LeakyReLU(0.1)
  → Conv(192→192, stride 1) + LeakyReLU(0.1)  → Level 6 features (192 ch, 1/64 res)
```

Each level consists of a stride-2 conv (halves spatial resolution) followed by a stride-1 conv (refines features), both with LeakyReLU(0.1) activation. All convolutions use 3×3 kernels with padding 1. The pyramid produces 6 feature levels at resolutions from 1/2 to 1/64 of the input.

### PWC-Disp: Stereo Disparity Network

The stereo disparity network (`PWCDisp`) estimates left-to-right and right-to-left disparities using a coarse-to-fine strategy across 6 decoder levels. It uses a **1-D cost volume** because stereo disparity is purely horizontal.

#### 1-D Cost Volume

The cost volume measures feature similarity between the left and right images at different horizontal offsets. For a maximum displacement `d=4`, the cost volume has `2d+1 = 9` channels:

$$CV(x, y, j) = \frac{1}{C} \sum_{c=1}^{C} f_1(x, y, c) \cdot f_2(x + j - d, y, c), \quad j \in [0, 2d]$$

where $f_1$, $f_2$ are the feature maps from the left and right images, $C$ is the number of feature channels, and the right image features are zero-padded by $d$ on each side. This produces a 9-channel tensor where each channel encodes the matching cost at a specific horizontal offset.

#### Coarse-to-Fine Decoder

The decoder processes from Level 6 (coarsest) to Level 2 (finest). At each level:

1. **Warp**: Upsample the flow from the previous level (×2 spatial + ×2 magnitude), then warp the right image's features to align with the left image using the current flow estimate.
2. **Cost Volume**: Compute a new 1-D cost volume between the left features and the warped right features. This cost volume captures the *residual* displacement not yet accounted for.
3. **Decode**: A dense-connection decoder block processes the concatenation of [cost volume, left features, upsampled flow] and predicts a residual flow.
4. **Accumulate**: Add the residual to the upsampled flow to get the current level's flow estimate.
5. **Clamp**: Enforce sign consistency (left-to-right disparity is negative, right-to-left is positive) via `ReLU` clamping.

The decoder block at each level uses dense connections (DenseNet-style):
```
Input
  → Conv(in→128) + LeakyReLU  → c1
  → Conv(128→128) + LeakyReLU → c2
  → Conv(cat[c1,c2]→96) + LeakyReLU  → c3
  → Conv(cat[c2,c3]→64) + LeakyReLU  → c4
  → Conv(cat[c3,c4]→32) + LeakyReLU  → c5 (feature output)
  → Conv(cat[c4,c5]→1)               → flow_x (disparity, no activation)
```

The output is 1-channel disparity (horizontal only). A zero y-channel is concatenated to form a 2-channel tensor for warping compatibility.

#### Context Refinement

At the finest decoder level (Level 2), an additional **context network** refines the disparity using dilated convolutions that expand the receptive field without losing resolution:

```
Conv(in→128, dilation=1)  → LeakyReLU
Conv(128→128, dilation=2) → LeakyReLU
Conv(128→128, dilation=4) → LeakyReLU
Conv(128→96,  dilation=8) → LeakyReLU
Conv(96→64,  dilation=16) → LeakyReLU
Conv(64→32,  dilation=1)  → LeakyReLU
Conv(32→1,   dilation=1)  → disparity residual (no activation)
```

The context network output is added to the Level 2 flow as a final refinement.

#### Output Normalization

The raw decoder outputs are in pixel units at the decoder's resolution. They are **normalized to fraction-of-width** and upsampled to 4 output scales:

- Scale 0: full resolution (H × W), from Level 2 flow divided by (W/4)
- Scale 1: half resolution (H/2 × W/2), from Level 3 flow divided by (W/8)
- Scale 2: quarter resolution (H/4 × W/4), from Level 4 flow divided by (W/16)
- Scale 3: eighth resolution (H/8 × W/8), from Level 5 flow divided by (W/32)

A small epsilon (`1e-6`) is added to ensure positive disparity values.

#### Bidirectional Estimation

`PWCDisp` contains **two independent decoder instances** (`left_disp` and `right_disp`) with separate weights:
- `left_disp`: computes left-to-right disparity (negative sign, meaning "look left in the right image to find the match")
- `right_disp`: computes right-to-left disparity (positive sign)

Both share the same `FeaturePyramidDisp` for feature extraction. The output at each scale is a 2-channel tensor: channel 0 = left disparity, channel 1 = right disparity.

### PWC-Flow: Optical Flow Network

The optical flow network (`PWCFlow`) has the same coarse-to-fine PWC structure as the disparity network, but with key differences:

#### 2-D Cost Volume

Since optical flow can be in any direction (not just horizontal), the cost volume searches over a 2D patch:

$$CV(x, y, i, j) = \frac{1}{C} \sum_{c=1}^{C} f_1(x, y, c) \cdot f_2(x + j - d, y + i - d, c)$$

For `d=4`, this produces $(2d+1)^2 = 81$ channels—much larger than the 9-channel stereo cost volume.

#### 2-Channel Output

The decoder predicts 2-channel flow (horizontal + vertical) instead of 1-channel disparity. The context network also outputs 2 channels.

#### Output Scaling

The flow outputs are scaled by `×4` (to convert from 1/4 resolution pixel units to full-resolution pixel units) and upsampled to 4 scales, producing flows in **pixel units** at the target resolution.

### PoseNet: Camera Pose Estimation

The `PoseExpNet` estimates the relative 6-DOF camera pose between two frames. It takes a concatenation of two images (6 channels) and outputs a 6-element vector `[tx, ty, tz, rx, ry, rz]`.

Architecture:
```
Input: cat(target_image, source_image) → (B, 6, H, W)

7 encoder blocks of paired convolutions with progressive downsampling:
  Block 1: Conv(6→16, k=7, s=2) + ReLU, Conv(16→16, k=7, s=1) + ReLU
  Block 2: Conv(16→32, k=5, s=2) + ReLU, Conv(32→32, k=5, s=1) + ReLU
  Block 3: Conv(32→64, k=3, s=2) + ReLU, Conv(64→64, k=3, s=1) + ReLU
  Block 4: Conv(64→128, k=3, s=2) + ReLU, Conv(128→128, k=3, s=1) + ReLU
  Block 5: Conv(128→256, k=3, s=2) + ReLU, Conv(256→256, k=3, s=1) + ReLU
  Block 6: Conv(256→256, k=3, s=2) + ReLU, Conv(256→256, k=3, s=1) + ReLU
  Block 7: Conv(256→256, k=3, s=2) + ReLU, Conv(256→256, k=3, s=1) + ReLU

Prediction head: Conv(256→6, k=1, s=1) → spatial average → (B, 6)
```

The translations (first 3 elements) are scaled by 0.01 as an empirical training trick to keep translations in a reasonable numerical range relative to rotations.

Note that PoseNet uses **ReLU** activations (not LeakyReLU), and uses larger kernels (7×7, 5×5) in early layers—reflecting the different nature of pose estimation compared to dense matching.

---

## Training Modes

UnOS supports four training modes with increasing complexity:

### 1. `stereo` — Stereo Disparity Only

**Networks**: FeaturePyramidDisp + PWCDisp

This is the simplest mode. Given a stereo pair $(I_L, I_R)$, the model predicts left and right disparities and is trained with:
- Photometric reconstruction loss (warp right→left and left→right)
- Disparity smoothness loss
- Left-right consistency loss

### 2. `flow` — Optical Flow Only

**Networks**: FeaturePyramidFlow + PWCFlow

Given a temporal pair $(I_t, I_{t+1})$, the model predicts optical flow and is trained with:
- Photometric reconstruction loss (warp $I_{t+1}$→$I_t$ using predicted flow)
- Flow smoothness loss
- Occlusion masking via forward-warping the reverse flow

### 3. `depth` — Stereo + Pose

**Networks**: FeaturePyramidDisp + PWCDisp + FeaturePyramidFlow + PWCFlow + PoseNet

Adds pose estimation to stereo training. The depth (from stereo disparity) and pose are used to compute a **rigid flow** (the motion field implied by the scene geometry and camera motion). This rigid flow is used for an additional temporal photometric loss. The key benefit is that temporal information provides supervision for depth estimation in regions where stereo matching alone may be ambiguous.

### 4. `depthflow` — Full Joint Training

**Networks**: All five sub-networks

The most complete mode. Jointly trains depth, optical flow, and pose with:
- Stereo photometric + smoothness + LR consistency (from stereo pairs)
- Temporal photometric loss using rigid flow (from depth + pose)
- Temporal photometric loss using optical flow
- **Flow consistency loss**: penalizes differences between rigid flow and optical flow in regions where they should agree (static regions). A mask identifies dynamic regions where optical flow and rigid flow are expected to differ.
- Pose refinement via `inverse_warp_new` using both depth maps and optical flow

---

## Loss Functions

### Photometric Reconstruction Loss

The photometric loss measures how well the predicted disparity or flow can reconstruct one image from another:

$$\mathcal{L}_{rec} = \alpha \cdot \text{SSIM}(\hat{I}, I) + (1 - \alpha) \cdot ||\hat{I} - I||_1$$

where $\hat{I}$ is the reconstructed image (via warping), $I$ is the original image, and $\alpha = 0.85$ by default.

**SSIM** (Structural Similarity Index) is computed with 3×3 average pooling (VALID padding, no zero-padding) and constants $C_1 = 0.01^2$, $C_2 = 0.03^2$:

$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

The loss is $(1 - \text{SSIM})/2$, clamped to $[0, 1]$.

### Occlusion Masking

Occluded pixels (visible in one view but not the other) should not contribute to the photometric loss because there is no valid correspondence to reconstruct. UnOS handles this via **forward-warping of ones tensors**:

1. Create a tensor of all ones at the source resolution
2. Forward-warp it using the reverse disparity/flow (bilinear splatting)
3. Clamp to [0, 1]

Pixels in the target view that receive contributions from the source will have values near 1.0 (visible). Occluded pixels receive no contribution and remain near 0.0. This mask is used as a per-pixel weight on the loss:

$$\mathcal{L}_{rec}^{masked} = \frac{\sum_{p} M(p) \cdot \mathcal{L}_{rec}(p)}{\sum_{p} M(p) + \epsilon}$$

where $M(p)$ is the occlusion mask and $\epsilon = 10^{-12}$ prevents division by zero. The masks are **detached** from the computation graph so that gradients do not flow through the occlusion computation.

### Disparity Smoothness Loss (2nd Order)

The smoothness loss encourages spatially smooth disparity maps while allowing sharp discontinuities at image edges:

$$\mathcal{L}_{sm} = \frac{1}{2} \sum_{s=0}^{3} \frac{1}{2^s} \left[ \text{mean}\left( |d_{xx}^s| \cdot w_x^s \right) + \frac{1}{16} \cdot \text{mean}\left( |d_{yy}^s| \cdot w_y^s \right) \right]$$

where:
- $d_{xx}^s = \nabla_x(\nabla_x(d^s))$ is the **second-order** horizontal gradient of disparity at scale $s$
- $d_{yy}^s = \nabla_y(\nabla_y(d^s))$ is the second-order vertical gradient
- $w_x^s = \exp(-10 \cdot \text{mean}_c |\nabla_x(I^s)|)$ is the edge-aware weight (exponentially down-weighted at image edges)
- $w_y^s$ is the corresponding vertical weight
- $1/2^s$ provides **per-scale weighting** (finer scales contribute more)
- $1/16$ is a **y-scale factor** (a quirk of the UnOS implementation where vertical smoothness is 16× weaker than horizontal)

Second-order smoothness is preferred over first-order because it penalizes changes in the *gradient* of disparity, allowing linear depth ramps (like slanted surfaces) while still penalizing noisy oscillations.

### Left-Right Consistency Loss

This loss enforces geometric consistency between the left and right disparity maps:

$$\mathcal{L}_{lr} = \sum_{s=0}^{3} \left[ \frac{\text{mean}(|d_R^{s \to L} - d_L^s| \cdot M_L^s)}{\text{mean}(M_L^s)} + \frac{\text{mean}(|d_L^{s \to R} - d_R^s| \cdot M_R^s)}{\text{mean}(M_R^s)} \right]$$

where $d_R^{s \to L}$ is the right disparity warped to the left view using the left disparity, and vice versa. The occlusion masks $M_L^s$, $M_R^s$ ensure that consistency is only enforced at visible pixels.

### Flow Consistency Loss (depthflow mode only)

In `depthflow` mode, the model enforces consistency between rigid flow (from depth + pose) and optical flow:

$$\mathcal{L}_{fc} = w_{fc} \cdot \text{Charbonnier}\left(\text{sg}(F_{rigid}) - F_{optical}, M_{ref}\right)$$

where $\text{sg}(\cdot)$ is stop-gradient (rigid flow is detached), $M_{ref}$ is a mask that identifies static regions, and Charbonnier is a robust loss function $\sqrt{x^2 + \epsilon}$.

The mask $M_{ref}$ is computed by combining:
- Occlusion mask (where reverse flow indicates the pixel is not visible)
- Flow difference mask: $||F_{rigid} - F_{optical}||_2 < \tau / 2^s$ (regions where rigid and optical flow agree are considered static)

### Total Loss

For each mode, the total loss is a weighted combination:

**stereo**: $\mathcal{L} = \mathcal{L}_{rec} + w_{sm} \cdot \mathcal{L}_{sm} + w_{lr} \cdot \mathcal{L}_{lr}$

**flow**: $\mathcal{L} = \mathcal{L}_{rec}^{optical} + w_{fsm} \cdot \mathcal{L}_{flow\_smooth}$

**depth**: $\mathcal{L} = 10 \cdot \mathcal{L}_{rec}^{depth} + \mathcal{L}_{stereo\_smooth}$

**depthflow**: $\mathcal{L} = 10 \cdot \mathcal{L}_{rec}^{depth} + \mathcal{L}_{stereo\_smooth} + \mathcal{L}_{rec}^{optical} + \mathcal{L}_{flow\_smooth} + \mathcal{L}_{flow\_consist}$

Default weights: $w_{sm} = $ `depth_smooth_weight`, $w_{lr} = 1.0$, $\alpha = $ `ssim_weight`.

---

## External Source Files

The UnOS external source code lives in `external_src/stereo_depth_completion/UnOS/`:

| File | Purpose |
|------|---------|
| `models.py` | High-level model classes (`Model_stereo`, `Model_flow`, `Model_depth`, `Model_depthflow`) and their eval counterparts. Orchestrates sub-networks and loss computation. |
| `monodepth_model.py` | `MonodepthModel` class: wraps `PWCDisp` with multi-scale loss computation (photometric + smoothness + LR consistency). `disp_godard()` helper function. |
| `nets/pwc_disp.py` | `FeaturePyramidDisp`, `CostVolumeDisp` (1-D), `OpticalFlowDecoderDisp`, `ContextNetDisp`, `PWCDispDecoder`, `PWCDisp`. The stereo disparity network. |
| `nets/pwc_flow.py` | `FeaturePyramidFlow`, `CostVolume` (2-D), `OpticalFlowDecoder`, `ContextNet`, `PWCFlow`. The optical flow network. |
| `nets/pose_net.py` | `PoseExpNet`. Camera pose estimation network. |
| `optical_flow_warp_old.py` | `transformer_old()`: backward warping using grid_sample (NHWC interface). |
| `optical_flow_warp_fwd.py` | `transformerFwd()`: forward warping via bilinear splatting (for occlusion masks). |
| `monodepth_dataloader.py` | Data loading and multi-scale intrinsics computation. |
| `utils.py` | `inverse_warp()`, `inverse_warp_new()`: project depth to 3D, transform by pose, reproject to compute rigid flow. |
| `loss_utils.py` | `SSIM()`, `cal_grad2_error()`, `cal_grad2_error_mask()`, `charbonnier_loss()`, image preprocessing. |

---

## AugUndo Wrapper Usage

The AugUndo framework wraps UnOS via `UnOSModel` in `stereo_depth_completion/unos_model.py`. The wrapper:

1. **Initializes** the appropriate UnOS model class (`Model_stereo` or `Model_depthflow`) with configuration parameters.
2. **`forward_disparity()`**: Runs just the feature pyramid + PWCDisp to get disparity predictions (no loss). This is used during AugUndo training where loss is computed externally after augmentation undo.
3. **`forward()`**: Runs the full UnOS forward pass with internal loss computation. Used for native (non-AugUndo) training.
4. Provides utilities for checkpoint save/restore, device management, and mode switching.

In AugUndo training:
- Augmented stereo pair → `forward_disparity()` → augmented disparity
- Undo augmentation on disparity (reverse transforms, apply scale factors)
- Compute loss in original frame using `stereo_losses.py` with UnOS-specific settings:
  - Occlusion masking enabled
  - Per-scale smoothness weighting ($1/2^s$)
  - Y-smoothness scale of $1/16$
  - Smoothness on disparity directly (not 2-channel flow)
