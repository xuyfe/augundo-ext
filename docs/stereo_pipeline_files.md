# Stereo Depth Completion Pipeline — File Reference

This document describes what each file in `augundo-ext/stereo_depth_completion/` does, how they relate to each other, and how data flows through the pipeline.

---

## File Overview

```
stereo_depth_completion/
├── __init__.py                        # Package init
├── stereo_depth_completion_model.py   # Model factory / registry
├── bdf_model.py                       # BDF model wrapper
├── unos_model.py                      # UnOS model wrapper
├── stereo_losses.py                   # Model-agnostic loss computation
├── stereo_depth_completion.py         # Core training & inference pipeline
├── train_stereo_depth_completion.py   # Training CLI entrypoint
└── run_stereo_depth_completion.py     # Evaluation CLI entrypoint
```

---

## `stereo_depth_completion_model.py` — Model Factory

**Purpose**: Maps model name strings to the appropriate wrapper class with model-specific configuration.

**Key function**: `get_stereo_model(model_name, **kwargs)`

This is a simple registry that:
- `'bdf'` → creates a `BDFModel` with BDF-specific config (backbone name, input dimensions, loss weights including `lr_loss_weight`, `alpha_image_loss`, `disp_gradient_loss_weight`, `type_of_2warp`)
- `'unos'` → creates a `UnOSModel` with UnOS-specific config (mode, smoothness weights, flow settings including `depth_smooth_weight`, `ssim_weight`, `flow_smooth_weight`, `flow_consist_weight`, `flow_diff_threshold`)

The factory extracts model-specific parameters from `kwargs` and passes them to the appropriate constructor. This allows the rest of the pipeline to be model-agnostic.

---

## `bdf_model.py` — BDF Model Wrapper

**Purpose**: Wraps the external BDF network (MonodepthNet or PWCDCNet) with a unified interface that the AugUndo pipeline can use.

**Class**: `BDFModel`

### Constructor
- Takes `model_name` ('monodepth' or 'pwc'), loss weight parameters, and device
- Instantiates either `MonodepthNet` (input: 256×512, 6-channel concatenated pair) or `PWCDCNet` (input: 256×832, 6-channel concatenated pair)
- Stores loss hyperparameters as attributes

### Key Methods

**`forward(left_t, left_t1, right_t, right_t1)`**
- Constructs the 4-directional pair batch:
  - `former = [left_t1, left_t, right_t, left_t]`
  - `latter = [right_t1, left_t1, right_t1, right_t]`
- Concatenates each pair along channel dim (6 channels)
- Runs forward (former→latter) and reverse (latter→former) through the network
- Returns dict with `disp_est`, `disp_est_scale`, and reverse predictions
- This produces 8 flow estimates from 4 directional pairs

**`forward_stereo_disparity(left_img, right_img)`**
- Takes a single stereo pair
- Concatenates `cat(left, right)` and `cat(right, left)` → batch of 2
- Runs through network to get left→right and right→left flow
- Extracts horizontal component as disparity, normalizes to fraction-of-width
- Returns: `disp_left` (4 scales), `disp_right` (4 scales), `flow_left` (4 scales), `flow_right` (4 scales)
- Disparity values are **positive** and **normalized** (fraction of width)

**`forward_temporal_flow(image_t, image_t1)`**
- Takes a temporal pair
- Concatenates and runs through network
- Returns: list of 4-scale normalized flow tensors (ch0 = fraction of width, ch1 = fraction of height)

**`compute_loss(forward_outputs, reverse_outputs, ...)`**
- Computes the full native BDF loss from the 4-directional pair outputs
- Builds image pyramids, border masks, and forward-backward occlusion masks
- Computes: reconstruction loss, smoothness loss, LR consistency, optional 2-warp loss
- Returns total loss and loss info dict

### Utility Methods
- `parameters()`, `train()`, `eval()`, `to()`, `data_parallel()`
- `restore_model(checkpoint_path)`, `save_model(checkpoint_path, step, optimizer)`

---

## `unos_model.py` — UnOS Model Wrapper

**Purpose**: Wraps the external UnOS model with the same unified interface.

**Class**: `UnOSModel`

### Constructor
- Takes `mode` ('stereo' or 'depthflow'), loss weight parameters, and device
- Creates an `opt`-like namespace object to match UnOS's expected configuration format
- Instantiates `Model_stereo` or `Model_depthflow` from the external source

### Key Methods

**`forward(image1, image1r, image2, image2r, cam2pix, pix2cam)`**
- Passes all inputs directly to the internal UnOS model
- Loss is computed *inside* the model (unlike BDF where loss is computed externally)
- Returns: `loss` (scalar), `info` (dict of sub-loss components)

**`forward_disparity(left_img, right_img)`**
- Runs just the FeaturePyramidDisp + PWCDisp for inference
- Does NOT compute loss
- Returns: `disp_left` (4 scales), `disp_right` (4 scales)
- Disparity values are positive and normalized (fraction of width)

**`compute_loss()`**
- Simply returns the loss stored from the last `forward()` call

### Utility Methods
Same as BDFModel: `parameters()`, `train()`, `eval()`, `to()`, `data_parallel()`, `restore_model()`, `save_model()`

---

## `stereo_losses.py` — Model-Agnostic Loss Computation

**Purpose**: Provides loss functions that work independently of BDF or UnOS specifics. This is used during AugUndo training where loss is computed in the original (un-augmented) coordinate frame.

### Warping Functions

**`warp_right_to_left(right_img, disp_left)`**
- Backward-warps right image to the left view using left disparity
- `disp_left` is positive, normalized (fraction of width)
- Uses `grid_sample` with bilinear interpolation and zero padding

**`warp_left_to_right(left_img, disp_right)`**
- Backward-warps left image to the right view using right disparity

**`warp_with_flow_2d(source, flow_norm)`**
- General 2D warping using normalized flow (ch0 = fraction of width, ch1 = fraction of height)
- Used for temporal photometric loss

### Occlusion Masking

**`_forward_warp_ones(pixel_disp, direction, H, W)`**
- Implements bilinear splatting: each source pixel at position $x$ is scattered to target position $x \pm d$ with bilinear weights
- Uses `scatter_add_` for efficient GPU accumulation
- Output clamped to [0, 1]: 1.0 = visible, 0.0 = occluded
- Matches the `transformerFwd` approach from UnOS

**`compute_occlusion_masks(disp_left, disp_right)`**
- Computes visibility masks for both left and right views
- Left mask: forward-warp ones from right to left using right disparity
- Right mask: forward-warp ones from left to right using left disparity
- Masks are **detached** (no gradient flow)

### Loss Components

**`ssim(x, y)`**
- Returns $(1 - \text{SSIM})/2$ in $[0, 1]$
- 3×3 average pooling with VALID padding (no zero-padding)
- Constants: $C_1 = 0.01^2$, $C_2 = 0.03^2$

**`smoothness_loss_2nd_order(disp, image, y_scale=1.0)`**
- 2nd-order edge-aware smoothness
- Computes $\nabla_{xx}(d)$ and $\nabla_{yy}(d)$
- Weights: $\exp(-10 \cdot \text{mean}_c |\nabla I|)$
- `y_scale` controls relative weight of vertical vs horizontal smoothness

**`scale_pyramid(img, num_scales)`**
- Creates multi-scale image pyramid using area interpolation

### Main Loss Function

**`compute_stereo_loss(left_img, right_img, disp_left, disp_right, ...)`**

Orchestrates all loss components across multiple scales. Parameters control model-specific behavior:

| Parameter | BDF Setting | UnOS Setting | Effect |
|-----------|-------------|-------------|--------|
| `use_occlusion_mask` | `False` | `True` | Whether to use forward-warp occlusion masks |
| `smooth_per_scale` | `False` | `True` | Whether to weight smoothness by $1/2^s$ |
| `smooth_flow_left/right` | 2-ch flow tensors | `None` | What to compute smoothness on |
| `smooth_pixel_divisor` | `20.0` | `0.0` | Convert normalized→pixel scale and divide |
| `smooth_y_scale` | `1.0` | `1/16` | Y-smoothness relative weight |

Returns: `(total_loss, loss_info_dict)`

### Temporal Loss Functions

**`compute_temporal_photometric_loss(image_t, image_t1, flow_norm, alpha)`**
- Warps `image_t1` to `image_t` using 2D flow
- Computes $\alpha \cdot \text{SSIM} + (1-\alpha) \cdot L1$ at finest scale only

**`compute_flow_smoothness(flow_list, image, ...)`**
- Multi-scale 2nd-order smoothness on 2D flow tensors
- Same parameterization as stereo smoothness

---

## `stereo_depth_completion.py` — Core Training & Inference Pipeline

**Purpose**: The main training loop implementing the AugUndo approach for stereo depth estimation. This is the heart of the pipeline.

### Helper Functions

**`create_stereo_geometric_transforms(augmentation_config)`**
- Creates stereo-safe geometric augmentations
- **Allowed**: horizontal flip, resize, horizontal translate, crop
- **Forbidden**: rotation, vertical flip, vertical translate (these break epipolar rectification)

**`apply_stereo_geometric_augmentation(images, transforms)`**
- Applies the same geometric transform to all 4 images
- **Critical**: if horizontal flip is applied, left and right images are **swapped** after the transform (because flipping a left image makes it look like a right image and vice versa)
- Returns augmented images + `transform_performed` dict recording what was done

**`apply_stereo_photometric_augmentation(images, transforms)`**
- Applies the same color jitter (brightness, contrast, gamma, hue, saturation) to all 4 images uniformly

**`undo_stereo_geometric_augmentation(disp_left, disp_right, transform_performed, ...)`**
- Reverses geometric augmentation on predicted disparity maps
- For multi-scale + resize: upsamples all scales → undoes at full res → downsamples back
- **Disparity scaling**: if the image was resized by factor $s$, the disparity is scaled by $1/s$ (because disparity in normalized units is proportional to image width)
- Extracts the scale factor from the transform record (varies by augmentation type: `resize_to_shape`, `resize_and_crop`, `resize_and_pad`)

### Data Loaders

**`create_bdf_dataloader(data_path, filenames_file, batch_size, ...)`**
- Uses BDF's native `myCycleImageFolder` dataset
- Returns batches of: `(left_t, left_t1, right_t, right_t1)` as tensors in [0, 1]

**`create_unos_dataloader(data_path, filenames_file, batch_size, ...)`**
- Uses UnOS's native `MonodepthDataloader`
- Returns batches of: `(left, right, next_left, next_right, cam2pix, pix2cam)`
- Includes multi-scale camera intrinsics
- 50% random temporal swap (uses $t+1$ as the "current" frame for additional diversity)

### Training Loop: `train()`

The main training function. Accepts all configuration via arguments and orchestrates the full AugUndo pipeline. Supports both epoch-based and iteration-based training with learning rate scheduling.

#### Single Training Step (`_run_one_step()`)

The core 8-step pipeline executed on each batch:

**Step 1 — Load batch**
- BDF: unpack `(left_t, left_t1, right_t, right_t1)`
- UnOS: unpack `(left, right, next_left, next_right, cam2pix, pix2cam)` with 50% temporal swap

**Step 2 — Photometric augmentation**
- Apply random color jitter uniformly to all 4 images

**Step 3 — Geometric augmentation**
- Apply random crop/resize/flip/translate to all 4 images identically
- If horizontal flip: swap left↔right images
- Record which transforms were applied

**Step 4 — Forward pass (disparity only)**
- BDF: `model.forward_stereo_disparity(aug_left, aug_right)` → `(disp_left_aug, disp_right_aug)` at 4 scales + 2-channel flows
- UnOS: `model.forward_disparity(aug_left, aug_right)` → `(disp_left_aug, disp_right_aug)` at 4 scales

**Step 5 — Undo geometric augmentation**
- Reverse crop/resize on disparity maps
- Apply disparity scale factor for resize ($1/s$)

**Step 6 — Swap left/right for flipped elements**
- For each batch element that was horizontally flipped: swap `disp_left ↔ disp_right` at all scales
- For BDF 2-channel flow: also negate the horizontal component (flipping reverses the sign of horizontal displacement)

**Step 7 — Compute loss in original frame**
- Call `compute_stereo_loss()` with model-specific settings
- BDF: 2-channel flow smoothness, pixel divisor=20, no per-scale weighting, no occlusion masks, y_scale=1.0
- UnOS: disparity smoothness, no pixel divisor, per-scale weighting, occlusion masks, y_scale=1/16
- Optional (BDF): compute temporal photometric + flow smoothness loss

**Step 8 — Backpropagation**
- `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`

### Inference: `run()`

The evaluation function:
- Loads model from checkpoint
- Runs inference on test images
- Computes evaluation metrics (depth/disparity) if ground truth is available

---

## `train_stereo_depth_completion.py` — Training Entrypoint

**Purpose**: CLI script that parses arguments and calls `train()`.

### Key Argument Groups

**Model selection**: `--model_name` ('bdf' or 'unos')

**Data**: `--data_path`, `--filenames_file`, `--n_height`, `--n_width`

**Training**: `--n_batch`, `--n_epoch` / `--n_step`, `--learning_rates`, `--learning_schedule`

**Augmentation**:
- Geometric: `--augmentation_random_flip_type` (only 'horizontal' allowed), `--augmentation_random_resize_to_shape`, `--augmentation_random_resize_and_crop`, `--augmentation_random_resize_and_pad`
- Photometric: `--augmentation_random_brightness`, `--augmentation_random_contrast`, `--augmentation_random_gamma`, `--augmentation_random_hue`, `--augmentation_random_saturation`

**BDF-specific**: `--bdf_model_name` ('monodepth'/'pwc'), `--lr_loss_weight`, `--alpha_image_loss`, `--disp_gradient_loss_weight`, `--type_of_2warp`, `--bdf_temporal_weight`

**UnOS-specific**: `--unos_mode` ('stereo'/'depthflow'), `--depth_smooth_weight`, `--ssim_weight`, `--flow_smooth_weight`, `--flow_consist_weight`, `--flow_diff_threshold`

### Validation
- Explicitly prohibits rotation and vertical flip for stereo (raises error if specified)
- Warns if no augmentation is configured

---

## `run_stereo_depth_completion.py` — Evaluation Entrypoint

**Purpose**: CLI script for model evaluation on KITTI benchmarks.

### BDF Evaluation (`eval_bdf()`)
- Loads MonodepthNet/PWCDCNet from checkpoint
- Tests on 200 KITTI 2015 stereo pairs
- Runs inference with original + horizontally-flipped images (post-processing trick from Monodepth)
- Reports: abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3

### UnOS Evaluation (`eval_unos()`)
- Loads `Model_eval_stereo` and transfers weights from checkpoint
- Tests on KITTI 2015 (200 pairs) and KITTI 2012 (194 pairs)
- Per-image inference with per-image calibration and intrinsic scaling
- Reports depth metrics (KITTI 2015) and disparity D1-all (both datasets)
