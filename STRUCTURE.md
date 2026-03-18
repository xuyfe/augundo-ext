# `augundo-ext/` folder structure (high-level)

This document summarizes the **overall layout** of `augundo-ext/` and where to look for training/evaluation entrypoints.

> Note: Large/generated folders like `data/`, `checkpoints/`, and the local Python environment (`augundo-py310env/`) are intentionally not expanded here.

## Top-level entrypoints

- **Training scripts**
  - `train_bdf.sh`: train BDF (BridgeDepthFlow) model in `external_src/stereo_depth_completion/BDF/`
  - `train_unos.sh`, `train_unos_small.sh`: train UnOS stereo depth completion models in `external_src/stereo_depth_completion/UnOS/`
- **Evaluation scripts**
  - `eval_bdf.sh`: evaluate BDF (stereo / optionally depth metrics)
  - `eval_unos.sh`: evaluate UnOS
- **Docs / metadata**
  - `README.md`: repo usage notes
  - `Implementation_details.md`: implementation notes (project-specific)
  - `requirements.txt`: Python dependencies for this repo
  - `augundo.pdf`: paper/reference included with the project

## Code and assets (key folders)

- **`external_src/`**: vendored research code (third-party or lightly adapted)
  - `external_src/stereo_depth_completion/BDF/`: BridgeDepthFlow (BDF) training/eval code
    - `train.py`: BDF training
    - `test_stereo.py`, `test_flow.py`: inference scripts producing predictions (e.g., `disparities.npy`)
    - `models/`, `utils/`: model definitions + dataloaders + metrics
  - `external_src/stereo_depth_completion/UnOS/`: UnOS training/eval code
    - `main.py`, `test.py`: entrypoints
    - `nets/`, `eval/`, `filenames/`: networks, evaluation, and filelists
  - `external_src/depth_completion/`: third-party depth completion baselines (e.g., KBNet / ScaffNet / VOiD)
  - `external_src/OpticalFlowToolkit/`: optical flow utilities/toolkit

- **`depth_completion/`**: local depth completion implementation / wrappers
  - `depth_completion/src/`: datasets, models, training, inference utilities

- **`utils/`**: shared utilities used by local code
  - `utils/src/`: data/eval/log/net utilities and transforms

- **`setup/`**: dataset/model setup scripts
  - `setup/kitti/`: KITTI split files and setup scripts
  - `setup/nyu_v2/`, `setup/scannet/`, `setup/make3d/`, `setup/void/`: other datasets

- **`training/`**: generated training filelists (primarily KITTI)
- **`testing/`**: generated test filelists (primarily KITTI)
- **`trained_models/`**: saved pretrained weights (where present)
- **`results/`**: saved qualitative outputs + `results.txt` summaries

- **`bash/`**: convenience launchers (often mirror `train/` and `run/` workflows)
- **`torchinductor_owenxuli/`**: TorchInductor cache/artifacts (generated)

## Condensed tree (filtered, depth-limited)

```text
augundo-ext/
  bash/
    depth_completion/
      run/ (fusionnet, kbnet, voiced)
      train/ (fusionnet, kbnet, voiced)
    stereo_depth/
      run/ (lai_pwc)
      train/ (lai_pwc, unos)
  depth_completion/
    src/ ...
  external_src/
    depth_completion/ (kbnet, scaffnet, voiced)
    OpticalFlowToolkit/
    stereo_depth_completion/
      BDF/ (train.py, test_stereo.py, test_flow.py, models/, utils/)
      UnOS/ (main.py, test.py, nets/, eval/, filenames/)
  results/
    bridgedepthflow_stereo/
    unos_stereo/
  setup/
    kitti/ ...
    nyu_v2/ ...
    scannet/ ...
    make3d/ ...
    void/ ...
  testing/
    kitti/ ...
  trained_models/
    stereo_depth/unos/kitti/ ...
  training/
    kitti/ ...
  utils/
    src/ ...
  .gitignore
  README.md
  requirements.txt
  train_bdf.sh
  eval_bdf.sh
  train_unos.sh
  eval_unos.sh
  train_unos_small.sh
  Implementation_details.md
  augundo.pdf
```

