# Stereo Depth Completion Integration Notes

## Overview

Two stereo depth completion models have been integrated into the AugUndo framework:

1. **UnOS** - Ported from TensorFlow to PyTorch. PWC-Net based stereo/depth/flow model.
2. **BridgeDepthFlow** - Already PyTorch. ResNet50-based MonodepthNet for stereo disparity.

Both produce depth from stereo image pairs via disparity estimation.

## File Structure

### External Source (model implementations)

```
external-src/stereo_depth_completion/
├── UnOS/
│   ├── __init__.py
│   ├── warping.py           # backward_warp (F.grid_sample), forward_warp (scatter)
│   ├── loss_utils.py         # SSIM, charbonnier, gradient losses
│   ├── feature_pyramid.py    # Feature extraction (6 levels)
│   ├── pwc_disp.py           # PWC stereo disparity (1D cost volume)
│   ├── pwc_flow.py           # PWC optical flow (2D cost volume)
│   ├── pose_net.py           # 7-layer pose network (6DoF output)
│   ├── geometry.py           # euler2mat, inverse_warp, SVD pose refinement
│   ├── monodepth_model.py    # MonodepthModel with stereo losses
│   └── unos_stereo.py        # UnOSStereo, UnOSDepth, UnOSDepthFlow
├── BridgeDepthFlow/
│   ├── __init__.py
│   ├── monodepth_model.py    # ResNet50 encoder-decoder (MonodepthNet)
│   ├── loss_utils.py         # resample2d, SSIM, smoothness, masks
│   └── bridge_depth_flow.py  # BridgeDepthFlowModel wrapper
```

### AugUndo Integration

```
depth_completion/src/
├── unos_model.py                    # UnOSModel wrapper (template interface)
├── bridgedepthflow_model.py         # BridgeDepthFlowModelWrapper (template interface)
├── depth_completion_model.py        # Updated dispatcher (added 'unos', 'bridgedepthflow')
├── stereo_dataloader.py             # StereoDepthCompletionTrainingDataset, InferenceDataset
├── stereo_depth_completion.py       # train(), validate(), run() for stereo AugUndo
├── train_stereo_depth_completion.py # CLI entry point for training
├── run_stereo_depth_completion.py   # CLI entry point for inference
└── test_stereo_smoke.py             # Smoke tests
```

## Key Design Decisions

### Stereo AugUndo Augmentation Strategy

1. **Geometric augmentations** are applied consistently to BOTH left and right images
2. **Horizontal flip** swaps left and right images (preserves stereo geometry)
3. **Photometric augmentations** applied only to the input left image (model sees augmented left + clean right)
4. **AugUndo cycle**: augment inputs -> forward -> reverse-transform output depth -> compute loss on originals

### Data Format

Training data uses UnOS 4-frame format (one line per sample):
```
left_t.png right_t.png left_t+1.png right_t+1.png calib_cam_to_cam.txt
```
All paths relative to KITTI raw data root. See `data/unos_train_4frames.txt`.

### Model Variants

**UnOS** supports three modes via `network_modules`:
- `['stereo']` - Pure stereo disparity (default)
- `['depth']` - Stereo + temporal pose network
- `['depthflow']` - Full model: stereo + optical flow + pose + consistency

**BridgeDepthFlow** only supports `['stereo']` (pure stereo).

### Stereo Models vs Monocular

Key differences from monocular pipeline:
- `forward_depth()` takes `right_image` parameter
- `compute_loss()` takes `right_image0` parameter
- No separate pose network (stereo-only models)
- Loss is self-supervised: photometric reconstruction + LR consistency + smoothness
- Depth = focal_length / disparity (not learned directly)

## Usage

### Training

UnOS with AugUndo framework
```bash
cd augundo-ext/depth_completion/src

python train_stereo_depth_completion.py \
    --train_data_file ../../data/unos_train_4frames.txt \
    --train_data_root ../../data/kitti_raw_data \
    --model_name unos \
    --network_modules stereo \
    --n_batch 4 \
    --n_height 256 \
    --n_width 832 \
    --learning_rates 1e-4 5e-5 \
    --learning_schedule 10 20 \
    --checkpoint_path ../../checkpoints/unos_stereo \
    --augmentation_random_flip_type horizontal \
    --augmentation_random_brightness 0.8 1.2 \
    --augmentation_random_contrast 0.8 1.2
```

UnOS without AugUndo framework
```bash
cd augundo-ext/depth_completion/src

python train_stereo_depth_completion.py \
    --train_data_file ../../data/unos_train_4frames.txt \
    --train_data_root ../../data/kitti_raw_data \
    --model_name unos \
    --network_modules stereo \
    --n_batch 4 \
    --n_height 256 \
    --n_width 832 \
    --learning_rates 1e-4 5e-5 \
    --learning_schedule 10 20 \
    --checkpoint_path ../../checkpoints/unos_stereo \
    --no_augment
```
### Inference

Before inference, run:

```bash
cd augundo-ext

python data/generate_stereo_val_paths.py
```

This will generate the left, right, intrinsics, and sparse depths paths needed for validation.

```bash
python run_stereo_depth_completion.py \
    --left_image_path  ../../data/stereo_val_left_image.txt \
    --right_image_path ../../data/stereo_val_right_image.txt \
    --sparse_depth_path ../../data/stereo_val_sparse_depth.txt \
    --intrinsics_path  ../../data/stereo_val_intrinsics.txt \
    --ground_truth_path ../../data/stereo_val_ground_truth.txt \
    --model_name unos \
    --restore_paths /path/to/checkpoint.pth \
    --output_path /path/to/output \
    --save_outputs
```

### Smoke Test

```bash
cd augundo-ext/depth_completion/src
python test_stereo_smoke.py
```

## TF-to-PyTorch Porting Notes (UnOS)

- NHWC -> NCHW tensor format throughout
- `slim.conv2d` -> `nn.Conv2d` with manual padding
- `tf.image.resize_bilinear` -> `F.interpolate(mode='bilinear', align_corners=True)`
- `Resample2d()` CUDA op -> `F.grid_sample` based `resample2d()`
- TF `tf.nn.avg_pool` 'VALID' padding -> PyTorch `F.avg_pool2d` with no padding
- Forward warping uses scatter-based splatting (custom implementation)
- Cost volume: horizontal-only 1D search for disparity, 2D search for flow
