# BDF (BridgeDepthFlow) Model

## Overview

BDF (BridgeDepthFlow) is a self-supervised stereo depth and optical flow estimation framework that uses **bidirectional consistency** across four images: a stereo pair at time $t$ $(I_L^t, I_R^t)$ and either the previous or next frame's stereo pair $(I_L^{t+1}, I_R^{t+1})$. The key idea is to construct multiple directional pairs from these four images—stereo pairs, temporal pairs, and cross-temporal-stereo pairs—and enforce consistency among all of them.

BDF provides two backbone options:
- **MonodepthNet**: A ResNet-style encoder-decoder that takes a concatenated stereo pair (6 channels) and directly predicts multi-scale disparity maps.
- **PWCDCNet**: A PWC-Net variant using correlation volumes and dense connections for optical flow / disparity estimation.

Both backbones produce 2-channel outputs (horizontal + vertical flow) at 4 scales, which can represent either stereo disparity or optical flow depending on the input pair.

---

## Network Architecture

### MonodepthNet

MonodepthNet is an encoder-decoder architecture inspired by ResNet-50 with skip connections, designed for stereo disparity estimation from a concatenated image pair.

#### Encoder

The encoder is a ResNet-50-like architecture with Bottleneck residual blocks:

```
Input: cat(left_image, right_image) → (B, 6, H, W)

conv1: Conv(6→64, k=7, s=2) + ELU                              → 1/2 res (skip1)
maxpool: MaxPool(k=3, s=2)                                      → 1/4 res (skip2)
layer1: 3 Bottleneck blocks (64→256), last block stride 2       → 1/8 res (skip3)
layer2: 4 Bottleneck blocks (128→512), last block stride 2      → 1/16 res (skip4)
layer3: 6 Bottleneck blocks (256→1024), last block stride 2     → 1/32 res (skip5)
layer4: 3 Bottleneck blocks (512→2048), last block stride 2     → 1/64 res
```

**Bottleneck Block**: Each block has expansion=4 and contains:
```
Input (inplanes)
  → Conv(inplanes→planes, k=1) + ELU
  → Conv(planes→planes, k=3, stride) + ELU
  → Conv(planes→planes×4, k=1)
  + Shortcut: Conv(inplanes→planes×4, k=1, stride)
  → ELU
```

Note that BDF uses **ELU** activations throughout, unlike UnOS which uses LeakyReLU(0.1). The stride-2 downsampling is applied on the *last* block of each layer (not the first, as in standard ResNet).

#### Decoder

The decoder progressively upsamples and merges with skip connections:

```
DecoderBlock(2048→512): upsample(conv5) + cat(skip5)  → conv → 1/32 res
DecoderBlock(512→256):  upsample + cat(skip4)          → conv → 1/16 res
DecoderBlock(256→128):  upsample + cat(skip3)          → conv → 1/8 res
  → DispBlock(128): predict disp4 (1/8 res)
DecoderBlock(128→64):   upsample + cat(skip2, udisp4)  → conv → 1/4 res
  → DispBlock(64):  predict disp3 (1/4 res)
DecoderBlock(64→32):    upsample + cat(skip1, udisp3)  → conv → 1/2 res
  → DispBlock(32):  predict disp2 (1/2 res)
DecoderBlock(32→16):    upsample + cat(udisp2)         → conv → full res
  → DispBlock(16):  predict disp1 (full res)
```

**DecoderBlock**: Upsample(×2) → Conv(in→out, k=3) + ELU → Concatenate with skip → Conv(mid→out, k=3) + ELU

**DispBlock**: Produces 2-channel disparity output:
```
Conv(in→2, k=3) → 0.3 × tanh(·) → normalized disparity in [-0.3, 0.3]
```

The normalized disparity is then scaled to pixel units:
- Channel 0 (horizontal): `disp_norm[:, 0] × W` (disparity in pixels)
- Channel 1 (vertical): `disp_norm[:, 1] × H` (should be near zero for stereo)

Each DispBlock also produces an upsampled version of the disparity that is fed as an additional input to the next decoder block—this provides the decoder with explicit knowledge of the current disparity estimate at each stage.

#### Output Format

MonodepthNet returns two lists:
- `disp_scale`: 4 tensors of pixel-scale disparity, `(B, 2, H_s, W_s)`
- `disp`: 4 tensors of normalized disparity (fraction of spatial dimensions)

### PWCDCNet

PWCDCNet is BDF's alternative backbone based on PWC-Net with dense connections and dilated convolutions. It is designed for general optical flow but is also used for stereo.

#### Encoder

The encoder is a 6-level feature pyramid, similar in spirit to UnOS but with 3 convolutions per level instead of 2:

```
Level 1: Conv(3→16, s=2) → Conv(16→16) → Conv(16→16)     → c1 (1/2 res)
Level 2: Conv(16→32, s=2) → Conv(32→32) → Conv(32→32)    → c2 (1/4 res)
Level 3: Conv(32→64, s=2) → Conv(64→64) → Conv(64→64)    → c3 (1/8 res)
Level 4: Conv(64→96, s=2) → Conv(96→96) → Conv(96→96)    → c4 (1/16 res)
Level 5: Conv(96→128, s=2) → Conv(128→128) → Conv(128→128)  → c5 (1/32 res)
Level 6: Conv(128→196, s=2) → Conv(196→196) → Conv(196→196) → c6 (1/64 res)
```

All convolutions use 3×3 kernels with LeakyReLU(0.1). Unlike MonodepthNet, PWCDCNet processes each image *independently* through the encoder (shared weights), rather than concatenating them.

#### Correlation Layer

At each decoder level, PWCDCNet computes a **2-D correlation volume** between the left features and the warped right features:

$$\text{Corr}(f_1, f_2)(x, y, i, j) = \sum_{c} f_1(x, y, c) \cdot f_2(x+i, y+j, c)$$

for displacement offsets $(i, j) \in [-d, d]^2$ with $d=4$, producing $(2d+1)^2 = 81$ channels. This uses a custom CUDA `Correlation` kernel for efficiency.

#### Coarse-to-Fine Decoder

The decoder follows the same coarse-to-fine principle as UnOS's PWC networks:

```
Level 6: correlation(c16, c26) → DenseDecoder → flow6
Level 5: warp(c25, flow6×0.625) → correlation(c15, warped) → DenseDecoder → flow5
Level 4: warp(c24, flow5×1.25) → correlation(c14, warped) → DenseDecoder → flow4
Level 3: warp(c23, flow4×2.5) → correlation(c13, warped) → DenseDecoder → flow3
Level 2: warp(c22, flow3×5.0) → correlation(c12, warped) → DenseDecoder → flow2
```

The warp scaling factors (0.625, 1.25, 2.5, 5.0) convert the flow from the decoder level's pixel units to the target level's pixel units. For example, at level 5 (1/32 res), the flow from level 6 (1/64 res) is ×2 in spatial size and the flow values are ×0.625 of the full-res scale.

Each **DenseDecoder** block uses DenseNet-style concatenations:
```
Input (corr + features + upsampled flow + upsampled feature)
  → Conv(od→128) → cat with input     → x
  → Conv(→128)   → cat with x         → x
  → Conv(→96)    → cat with x         → x
  → Conv(→64)    → cat with x         → x
  → Conv(→32)    → cat with x         → x
  → predict_flow: Conv(→2)            → 2-channel flow
```

#### Context Network (Dilated Convolutions)

At the finest level, a context network with progressively increasing dilation rates refines the flow:
```
Conv(→128, dilation=1)  → LeakyReLU
Conv(→128, dilation=2)  → LeakyReLU
Conv(→128, dilation=4)  → LeakyReLU
Conv(→96,  dilation=8)  → LeakyReLU
Conv(→64,  dilation=16) → LeakyReLU
Conv(→32,  dilation=1)  → LeakyReLU
Conv(→2,   dilation=1)  → residual flow
```

#### Output

The final flows are upsampled to 4 scales with ×4 magnitude scaling:
```
flow0 = interpolate(flow2 × 4.0, full_res)      → (B, 2, H, W)
flow1 = interpolate(flow3 × 4.0, half_res)      → (B, 2, H/2, W/2)
flow2 = interpolate(flow4 × 4.0, quarter_res)   → (B, 2, H/4, W/4)
flow3 = interpolate(flow5 × 4.0, eighth_res)    → (B, 2, H/8, W/8)
```

All outputs are in **pixel units** at full resolution.

---

## 4-Directional Pair Construction

A distinctive feature of BDF is how it constructs training pairs. Given 4 input images $(I_L^t, I_R^t, I_L^{t+1}, I_R^{t+1})$, BDF creates 4 directional pairs:

| Index | Former | Latter | Relationship |
|-------|--------|--------|-------------|
| 0 | $I_L^{t+1}$ | $I_R^{t+1}$ | Stereo at $t+1$ |
| 1 | $I_L^t$ | $I_L^{t+1}$ | Temporal left |
| 2 | $I_R^t$ | $I_R^{t+1}$ | Temporal right (cross) |
| 3 | $I_L^t$ | $I_R^t$ | Stereo at $t$ |

Each pair is run through the network in both directions (forward and reverse), producing 8 flow estimates total. This allows BDF to enforce consistency across stereo, temporal, and cross-view directions simultaneously.

---

## Loss Functions

### Photometric Reconstruction Loss

Same formulation as UnOS:

$$\mathcal{L}_{rec} = \alpha \cdot \text{SSIM}(\hat{I}, I) + (1 - \alpha) \cdot ||\hat{I} - I||_1$$

with $\alpha = 0.85$ and SSIM computed with 3×3 average pooling (VALID, no padding).

### Occlusion Masking (Forward-Backward Consistency)

Unlike UnOS which uses forward-warp-of-ones, BDF natively uses **forward-backward consistency** to detect occlusions:

1. Warp the backward flow using the forward flow: $F_{bw}^{warped} = \text{warp}(F_{bw}, F_{fw})$
2. Compute the residual: $\Delta F = F_{fw} + F_{bw}^{warped}$
3. A pixel is considered occluded if: $||\Delta F||^2 > 0.01 \cdot (||F_{fw}||^2 + ||F_{bw}||^2) + 0.5$

This produces an occlusion mask $M_{fb}$, combined with a border mask to exclude image boundaries. The final mask is: $M = M_{border} \cdot (1 - M_{fb})$.

**Note**: In the AugUndo wrapper, stereo pairs (indices 0, 1, 6, 7) use **all-ones masks** (no occlusion masking), matching the native BDF behavior for stereo pairs. Only temporal pairs receive occlusion masking.

### Disparity/Flow Smoothness Loss (2nd Order)

BDF uses the same 2nd-order edge-aware smoothness as UnOS, but with important differences in how it's applied:

$$\mathcal{L}_{sm} = \frac{1}{2}\left[ \text{mean}\left( \beta \cdot w_x \cdot |d_{xx}| \right) + \text{mean}\left( \beta \cdot w_y \cdot |d_{yy}| \right) \right]$$

where:
- $d_{xx}, d_{yy}$ are second-order gradients of the disparity/flow
- $w_x = \exp(-10 \cdot \text{mean}_c |\nabla_x I|)$, $w_y = \exp(-10 \cdot \text{mean}_c |\nabla_y I|)$
- $\beta$ is a scaling factor (typically 1.0)

**Key differences from UnOS:**
1. **Input**: BDF computes smoothness on **2-channel flow** (horizontal disparity + vertical component), not just 1-channel disparity
2. **No per-scale weighting**: All 4 scales contribute equally (no $1/2^s$ factor)
3. **Pixel-scale division**: The disparity is converted to pixel units and divided by 20 before smoothness: `cal_grad2_error(disp_scale / 20, image, 1.0)`
4. **Equal y-scale**: Vertical and horizontal smoothness are weighted equally ($y\_scale = 1.0$, vs UnOS's $1/16$)

### Left-Right Consistency Loss

Applied only to stereo pairs (not temporal pairs):

$$\mathcal{L}_{lr} = \text{mean}(|d_L - d_{R \to L}|) + \text{mean}(|d_R - d_{L \to R}|)$$

where the warping is done using the `Resample2d` module. This is applied at all 4 scales.

### 2-Warp Loss (Optional Temporal Consistency)

BDF supports four variants of a "2-warp" loss that enforces temporal consistency. The 2-warp loss uses the flow from one temporal pair to warp a third frame, then compares it with the reconstruction from a different flow path. The variant is selected by `type_of_2warp`:

- **Type 1**: Warp right image at $t+1$ using temporal flow, compare with stereo reconstruction
- **Type 2**: Similar but using different flow combination
- **Type 3**: Cross-view temporal warp
- **Type 4**: Combined cross-temporal consistency

Each variant uses the forward-backward occlusion mask from the temporal pair.

### Total Loss (Native BDF)

$$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{rec}^{reverse} + 10 \cdot \mathcal{L}_{sm} + w_{lr} \cdot \mathcal{L}_{lr}$$

The forward and reverse reconstruction losses are computed separately (from the 8 directional flow predictions), and the smoothness weight is 10.

---

## Data Loading

BDF uses a custom `myCycleImageFolder` dataset class that loads 4-frame sequences:
- Format: `(I_L^t, I_L^{t+1}, I_R^t, I_R^{t+1})`
- File paths specified by a filenames file via `get_kitti_cycle_data()`
- Built-in augmentations: random horizontal flip, gamma adjustment, brightness adjustment, color shift
- Images are normalized to [0, 1]

---

## External Source Files

The BDF external source code lives in `external_src/stereo_depth_completion/BDF/`:

| File | Purpose |
|------|---------|
| `models/MonodepthModel.py` | `MonodepthNet`: ResNet-50 encoder-decoder for stereo disparity. Includes `Bottleneck`, `DecoderBlock`, `DispBlock`. |
| `models/PWC_net.py` | `PWCDCNet`: PWC optical flow network with correlation, dense connections, and dilated context network. |
| `models/networks/submodules.py` | Helper modules: `conv`, `deconv`, `predict_flow`, `i_conv`. |
| `models/networks/correlation_package/` | Custom CUDA correlation kernel. |
| `models/networks/resample2d_package/` | Custom CUDA 2D resampling (warping) module. |
| `models/networks/FlowNetS.py` | FlowNetS architecture (simple sequential). |
| `models/networks/FlowNetC.py` | FlowNetC architecture (correlation-based). |
| `models/networks/FlowNetSD.py` | FlowNetSD architecture (simplified deeper). |
| `models/networks/FlowNetFusion.py` | Lightweight fusion network. |
| `train.py` | Native BDF training loop with all loss computation. |
| `utils/utils.py` | Loss functions: `SSIM`, `cal_grad2_error`, `get_mask`, `warp_2`, `make_pyramid`, `create_border_mask`. |
| `utils/scene_dataloader.py` | Data loading: `myCycleImageFolder`, `get_kitti_cycle_data`. |
| `utils/evaluation_utils.py` | Evaluation metrics: depth errors, KITTI evaluation, Velodyne projection. |

---

## AugUndo Wrapper Usage

The AugUndo framework wraps BDF via `BDFModel` in `stereo_depth_completion/bdf_model.py`. The wrapper:

1. **Initializes** either `MonodepthNet` or `PWCDCNet` based on `model_name` ('monodepth' or 'pwc').
2. **`forward_stereo_disparity()`**: Runs a single stereo pair through the network in both directions (left→right and right→left). Returns multi-scale disparity and flow lists. Disparity is normalized (fraction of width).
3. **`forward_temporal_flow()`**: Runs a temporal pair to predict optical flow (used for optional temporal loss).
4. **`forward()`**: Constructs the full 4-directional pair batch (8 images) and runs them through the network. Used for native BDF training.
5. **`compute_loss()`**: Computes the full BDF loss in the original frame with all components (reconstruction, smoothness, LR consistency, optional 2-warp).

In AugUndo training:
- Augmented stereo pair → `forward_stereo_disparity()` → augmented disparity + flow
- Undo augmentation on disparity (reverse transforms, apply scale factors, swap left/right if flipped)
- For 2-channel flow: also negate the horizontal component if horizontally flipped
- Compute loss in original frame using `stereo_losses.py` with BDF-specific settings:
  - No occlusion masking for stereo pairs (all-ones mask)
  - No per-scale smoothness weighting (all scales equal)
  - Smoothness on 2-channel flow (not just disparity)
  - Pixel-scale division by 20 for smoothness
  - Equal y-scale ($y\_scale = 1.0$)
- Optional: compute temporal photometric + flow smoothness loss using `forward_temporal_flow()`
