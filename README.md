# Extending AugUndo to Stereo Depth Estimation and Completion

A PyTorch Implementation Extending AugUndo to Stereo Depth Estimation and Completion Problems.

Based on the paper published in the European Conference on Computer Vision (ECCV) 2024

[[arxiv]](https://arxiv.org/pdf/2310.09739) [[publication]](https://link.springer.com/chapter/10.1007/978-3-031-73039-9_16)

Authors: [Yangchao Wu](https://www.linkedin.com/in/yangchaowu/), [Tian Yu Liu](http://web.cs.ucla.edu/~tianyu), [Hyoungseob Park](https://vision.cs.yale.edu/members/hyoungseob-park.html), [Stefano Soatto](https://web.cs.ucla.edu/~soatto/), [Dong Lao](https://www.linkedin.com/in/dong-lao-97b338b0/), [Alex Wong](https://vision.cs.yale.edu/members/alex-wong.html)

Models have been tested using Python 3.10 and CUDA 12.6

```
@inproceedings{wu2025augundo,
  title={Augundo: Scaling up augmentations for monocular depth completion and estimation},
  author={Wu, Yangchao and Liu, Tian Yu and Park, Hyoungseob and Soatto, Stefano and Lao, Dong and Wong, Alex},
  booktitle={European Conference on Computer Vision},
  pages={274--293},
  year={2025},
  organization={Springer}
}
```



# Setup

## Download Stereo 2012 and Scene Flow 2015

These data sets are used during evaluation for UnOS and BDF.

```bash
# 1. Create and enter the directory
cd /home/ox4/scratch_pi_aw989/ox4/data/

# 2. Download Stereo/Flow 2012 (approx. 2GB)
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip

# 3. Download Scene Flow 2015 (approx. 12GB)
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip

# then, unzip and establish sym links
unzip data_stereo_flow.zip
unzip data_scene_flow.zip

# assuming you're working in augundo-ext
ln -s /path/to/scene_flow_2015 data/
ln -s /path/to/stereo_2012 data/
```

Note, UnOS and BDF only use the `training/` subset of KITTI 2015 for depth evaluation. UnOS additionally uses the KITTI 2012 dataset for disparity evaluation. We renamed the 2015 folder names to `image_0`and `image_1` to match the folder names of the 2012 folder. Also, the calibration files for the 2015 folder have to be downloaded separately.

## Clone OpticalFlowToolKit

UnOS uses the OpticalFlowToolKit by Ruoteng Li during their eval phase. 

```bash
cd augundo-ext/external_src
git clone https://github.com/liruoteng/OpticalFlowToolkit.git
```

# Training and evaluation without AugUndo

Make sure you're working in the home (root) directory. The parameters are set according to the ones used in the original papers.

### UnOS

```bash
sbatch augundo-ext/original_slurm_jobs/train_unos.sh
```

Note: UnOS training also runs evaluation on the *training* sets of KITTI 2012 and KITTI 2015. But, if you want to run only an inference test:

```bash
sbatch augundo-ext/original_slurm_jobs/eval_unos.sh
```

### BridgeDepthFlow

```bash
sbatch augundo-ext/original_slurm_jobs/train_bdf.sh
```

To evaluate BDF:

```bash
sbatch augundo-ext/original_slurm_jobs/eval_bdf.sh
```

# Training and evaluation with AugUndo

### UnOS

```bash
sbatch augundo-ext/slurm_jobs/unos/augundo/train_augundo_unos.sh

sbatch augundo-ext/slurm_jobs/unos/augundo/eval_augundo_unos.sh
```

### BridgeDepthFlow

```bash
sbatch augundo-ext/slurm_jobs/bdf/augundo/train_augundo_bdf.sh

sbatch augundo-ext/slurm_jobs/bdf/augundo/eval_augundo_bdf.sh
```

# Key Implementation Details and Modifications

## Dataloaders and Training Pipeline

We use the native dataloaders from UnOS and BDF during training, instead of the `datasets.py` script used for monocular depth. The stereo models expect 4-frame input batches (left_t, right_t, left_t+1, right_t+1), whereas the monocular `datasets.py` produces a triplet. Using the native dataloaders also allows us to maintain the same training pipeline without architectural redesign.

In `external_src/stereo_depth_completion/UnOS/monodepth_dataloader.py` we modify the `MonodepthDataloader` class to allow setting `training=False` because the original UnOS model already performs some data augmentations to the data with a certain probability (50%). When AugUndo is applied, the native augmentations are disabled to prevent double-augmentation, since the "undoing" step only inverts augmentations applied by the AugUndo pipeline.

The model wrappers for UnOS and BDF are both found under `stereo_depth_completion/`. The new PyTorch implementations of the UnOS and BDF models are found under `external_src/stereo_depth_completion`—UnOS was originally developed with TensorFlow 1.x, while BDF was developed with an older version of Python and CUDA. All network architectures and data loading pipelines were reimplemented in PyTorch, preserving the exact architecture specifications (layer dimensions, activation functions, initialization) and loss formulations of the originals.

## New Scripts

We add various new scripts:

```
augundo-ext/stereo_depth_completion/
  ├── __init__.py
  ├── bdf_model.py                          # BDF wrapper
  ├── unos_model.py                         # UnOS wrapper
  ├── stereo_depth_completion_model.py      # Model registry with get_stereo_model()
  ├── stereo_depth_completion.py            # Core stereo AugUndo training loop
  ├── stereo_losses.py                      # Model-agnostic stereo loss module
  ├── train_stereo_depth_completion.py      # Training CLI entrypoint
  ├── run_stereo_depth_completion.py        # Inference/evaluation CLI entrypoint
  ├── template_model.py                     # Template for implementing new stereo models
  └── template_dataloader.py               # Template dataloader for new models
```

The stereo depth completion scripts are based on the scripts under `depth_completion/`. The pipeline operates directly on disparity predictions: the model predicts disparity in the augmented frame, augmentation undo is applied to the disparity, and all losses are computed on the un-augmented disparity against the original images.

## Key Design Decisions

Stereo augmentation constraints enforced:
  - Rotation is **forbidden** — destroys epipolar alignment
  - Vertical flip is **forbidden** — breaks vertical correspondence between left and right views
  - Resize (crop/pad) is **forbidden** — changes effective focal length, introducing disparity scale mismatch
  - Vertical translation is **forbidden** — misaligns scanline correspondence

Permitted augmentations:
  - Horizontal flip (with left-right image swap to maintain non-negative disparity convention)
  - Horizontal translation (preserves epipolar geometry and absolute disparity values)
  - Color jitter (brightness, contrast, saturation — applied identically to both views with shared parameters)
  - Gaussian blur (applied identically to both views)
  - Gaussian noise (applied to left image only)


## Results

All models are trained on KITTI raw data and evaluated on the held-out KITTI 2015 and KITTI 2012 scene flow test sets. Lower is better for all metrics except a1, a2, a3.

### UnOS - stereo-only mode, 100K iterations

**Depth Metrics (KITTI 2015)**

| Metric  | Original | AugUndo (ours) |
|---------|----------|----------------|
| abs_rel | 0.0956   | **0.0628**     |
| sq_rel  | 1.1856   | **0.8905**     |
| rms     | 5.465    | **4.299**      |
| log_rms | 0.183    | **0.139**      |
| d1_all  | 15.764   | **7.386**      |
| a1      | 0.910    | **0.952**      |
| a2      | 0.965    | **0.980**      |
| a3      | 0.983    | **0.989**      |

**Disparity Metrics (KITTI 2015)**

| Metric   | Original | AugUndo (ours) |
|----------|----------|----------------|
| epe      | 2.8083   | **1.3637**     |
| noc_rate | 0.1499   | **0.0693**     |
| occ_rate | 0.8717   | **0.2652**     |
| err_rate | 0.1666   | **0.0739**     |

**Disparity Metrics (KITTI 2012)**

| Metric   | Original | AugUndo (ours) |
|----------|----------|----------------|
| epe      | 2.3621   | **1.3006**     |
| noc_rate | 0.1456   | **0.0618**     |
| occ_rate | 0.8286   | **0.4780**     |
| err_rate | 0.1576   | **0.0719**     |

AugUndo improves every metric uniformly. The largest gains are in occluded-region error rate (occ_rate drops by 69% on KITTI 2015), consistent with AugUndo's geometric augmentations providing supervision in regions where the photometric loss is unreliable.

### BDF - MonodepthNet backbone, 15 epochs

**Depth Metrics (KITTI 2015)**

| Metric  | Original     | AugUndo (ours) |
|---------|--------------|----------------|
| abs_rel | **0.0754**   | 0.0792         |
| sq_rel  | **0.9312**   | 1.1503         |
| rms     | **4.480**    | 4.767          |
| log_rms | **0.162**    | 0.164          |
| d1_all  | 10.781       | **10.622**     |
| a1      | 0.932        | **0.934**      |
| a2      | 0.975        | **0.976**      |
| a3      | 0.988        | 0.988          |

**Disparity Metrics (KITTI 2015)**

| Metric   | Original     | AugUndo (ours) |
|----------|--------------|----------------|
| epe      | 1.7168       | **1.6668**     |
| noc_rate | **0.0935**   | 0.0998         |
| occ_rate | 0.8858       | **0.4142**     |
| err_rate | 0.1078       | **0.1062**     |

**Disparity Metrics (KITTI 2012)**

| Metric   | Original     | AugUndo (ours) |
|----------|--------------|----------------|
| epe      | 1.9370       | **1.7911**     |
| noc_rate | **0.0950**   | 0.0998         |
| occ_rate | 0.9222       | **0.5755**     |
| err_rate | 0.1143       | **0.1113**     |

BDF presents a more nuanced picture: AugUndo dramatically improves occluded-region error rates (occ_rate reduced by 53% on KITTI 2015, 38% on KITTI 2012) and overall EPE, while the baseline retains slightly better pixel-level depth accuracy in non-occluded regions. This trade-off arises because BDF already includes aggressive photometric augmentation natively; the additional photometric jitter from AugUndo introduces noise that slightly degrades per-pixel precision in well-observed regions, while the geometric augmentations (horizontal translation) provide a strong training signal in occluded regions where the standard photometric loss is uninformative.
