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
sbatch augundo-ext/train_unos.sh
```

Note: UnOS training also runs evaluation on the *training* sets of KITTI 2012 and KITTI 2015. But, if you want to run only an inference test:

```bash
sbatch augundo-ext/eval_unos.sh
```

### BridgeDepthFlow

```bash
sbatch augundo-ext/train_bdf.sh
```

To evaluate BDF:

```bash
sbatch augundo-ext/eval_bdf.sh
```

# Training and evaluation with AugUndo

### UnOS

```bash
sbatch augundo-ext/slurm_jobs/train_augundo_unos.sh

sbatch augundo-ext/slurm_jobs/eval_augundo_unos.sh
```

### BridgeDepthFlow

```bash
sbatch augundo-ext/slurm_jobs/train_augundo_bdf.sh

sbatch augundo-ext/slurm_jobs/eval_augundo_bdf.sh
```

# Key Implementation Details and Modifications

## Dataloaders and Training Pipeline

We use the original dataloaders from UnOS and BDF during training, instead of the `datasets.py` script used for monocular depth. We decided to do this primarily because the inputs that UnOS and BDF expect are of a different format as compared to the processed data from `datasets.py` (UnOS and BDF expect two pairs, whereas `datasets.py` produces a triplet). This also allows us to maintain the same training pipeline without architectural redesign.

In `external_src/stereo_depth_completion/UnOS/monodepth_dataloader.py` we modify the `MonodepthDataloader` class to allow setting `training=False` because the original UnOS model already performs some data augmentations to the data with a certain probability (50\%). So, once AugUndo is implemented to UnOS, some augmentations might be performed twice, so the model will train and evaluate on corrupted data since the "undoing" is only performed on the AugUndo pipeline.

The model wrappers for UnOS and BDF are both found under `stereo_depth_completion/`. The new PyTorch implementations of the UnOS and BDF models are found under `external_src/stereo_depth_completion`—UnOS was originally developed with Tensorflow, while BDF was developed with an older version of Python and CUDA.

## New Scripts

We add various new scripts:

augundo-ext/stereo_depth_completion/                                                                                                             
  ├── __init__.py                                                                                                                 
  ├── bdf_model.py                          # BDF wrapper                            
  ├── unos_model.py                         # UnOS wrapper                        
  ├── stereo_depth_completion_model.py      # Model registry with get_stereo_model()                          
  ├── stereo_depth_completion.py            # Core stereo AugUndo loop                        
  ├── train_stereo_depth_completion.py      # Training CLI entrypoint
  ├── stereo_losess.py                      # Contains helper functions for stereo                          
  └── run_stereo_depth_completion.py        # Inference CLI entrypoint

The stereo depth completion scripts are based on the scripts under `depth_completion/`.

  ## Key Design Decisions

Stereo augmentation constraints enforced:                                                  
  - Rotation is disabled (random_rotate_max=-1) — would break epipolar rectification
  - Vertical flip is excluded from flip types  
  - Crop-and-pad is disabled (can introduce vertical translation)  
  - Only horizontal flip, resize, and horizontal translation are permitted 
  - Horizontal flip triggers left-right image swap to maintain non-negative disparity convention

  Every other augmentation is extended to stereo pairs based on the original AugUndo framework. 

## Loss Computation