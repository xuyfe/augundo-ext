# AugUndo: Scaling Up Augmentations for Monocular Depth Completion and Estimation
PyTorch Implementation of AugUndo: Scaling Up Augmentations for Monocular Depth Completion and Estimation

Published in the European Conference on Computer Vision (ECCV) 2024

[[arxiv]](https://arxiv.org/pdf/2310.09739) [[publication]](https://link.springer.com/chapter/10.1007/978-3-031-73039-9_16)

Authors: [Yangchao Wu](https://www.linkedin.com/in/yangchaowu/), [Tian Yu Liu](http://web.cs.ucla.edu/~tianyu), [Hyoungseob Park](https://vision.cs.yale.edu/members/hyoungseob-park.html), [Stefano Soatto](https://web.cs.ucla.edu/~soatto/), [Dong Lao](https://www.linkedin.com/in/dong-lao-97b338b0/), [Alex Wong](https://vision.cs.yale.edu/members/alex-wong.html)

Model have been tested on Ubuntu 20.04 using Python 3.8 and PyTorch 1.11.0 (CUDA 11.1)

If this work is useful to you, please cite our paper:

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

**Code and pretrained models for monocular depth estimation are coming soon!**

## Table of Contents
1. [Setting up your virtual environment](#setting-up-virtual-environment)
2. [Setting up your datasets](#setting-up-datasets)
3. [Downloading and running pretrained models](#downloading-running-pretrained-models)
4. [Training with AugUndo](#training-with-augundo)
5. [Adding your own models to AugUndo](#adding-models-to-augundo)
6. [Related projects](#related-projects)
7. [License and disclaimer](#license-disclaimer)

## Setting up your virtual environment <a name="setting-up-virtual-environment"></a>
We will create a virtual environment using virtualenv with dependencies for training and testing our models.
```
virtualenv -p /usr/bin/python3.8 augundo-py38env
source augundo-py38env/bin/activate

export TMPDIR=./

pip install -r requirements.txt
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Setting up your datasets <a name="setting-up-datasets"></a>

For training datasets, we will use [KITTI][kitti_dataset] for outdoors and [VOID][void_github] for indoors. We will additionally use [NYUv2][nyu_v2_dataset], [ScanNet][scannet_dataset], and [Make3D][make3d_dataset] for evaluation. Below are instructions to run our setup script for each dataset. The setup script will (1) store images as sequential temporal triplets and (2) produce paths for training, validation and testing splits.

If you already have the above datasets downloaded, link them to a data directory:
```
mkdir -p data
ln -s /path/to/kitti_raw_data data/
ln -s /path/to/kitti_depth_completion data/
ln -s /path/to/void_release data/
ln -s /path/to/nyu_v2 data/
ln -s /path/to/scannet data/
ln -s /path/to/make3d data/
```

### Downloading datasets

If you do not already have datasets downloaded, please follow the instructions below.

<details>
<summary> <b> VOID dataset </b> </summary>

Assuming you are in the root of the repository, you can download the dataset via commandline using wget.
To construct the same dataset structure use for the setup (data pre-processing) scripts:

```
wget -O void_release.zip 'https://yaleedu-my.sharepoint.com/:u:/g/personal/alex_wong_yale_edu/Ebwvk0Ji8HhNinmAcKI5vSkBEjJTIWlA8PXwKNQX_FvB7g?e=0Zqe7g&download=1'

unzip void_release.zip -d data/
mv void_release.zip data/
```

If you encounter `error: invalid zip file with overlapped components (possible zip bomb)`. Please do the following
```
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
```
and run the above again. This should extract the dataset into your data directory. You should expect to see a directory structure that matches the following:

```
data
|-- void_release
    |-- void_1500
        |-- data
            |-- <sequence>
                |-- image
                |-- sparse_depth
                |-- validity_map
                |-- ground_truth
                |-- absolute_pose
                |-- K.txt
    ...
    |-- void_500
    ...
    |-- void_150
    ...
```

For more detailed instructions on downloading and using VOID and obtaining the raw rosbags, you may visit the [VOID][void_github] dataset webpage.
</details>

<details>
<summary> <b> KITTI dataset </b> </summary>

The `bash/kitti/setup_dataset_kitti.sh` script will download and set up `kitti_raw_data` and `kitti_depth_completion` for you in your data folder.

```
bash bash/kitti/setup_dataset_kitti.sh
```

Note that KITTI may require login for download, so you will need to follow their instructions on their [website][kitti_dataset].

Once you have downloaded KITTI (data_depth_velodyne.zip, data_depth_selection.zip, data_depth_annotated.zip) from their Amazon S3 bucket, you can use the commands from L186-209 from `bash/kitti/setup_dataset_kitti.sh` to complete the remaining steps to set up your data directory.

Once completed, you should expect to see a directory structure that matches the following:

```
data
|-- kitti_raw_data
    |-- 2011_09_26
        |-- 2011_09_26_drive_0001_sync
            |-- image_00
            |-- image_01
            |-- image_02
            |-- image_03
            ...
            |-- velodyne_points
    |-- 2011_09_28
    ...
    |-- 2011_10_03
    ...
|-- kitti_depth_completion
    |-- train_val_split
        |-- ground_truth
            |-- train
            |-- val
        |-- sparse_depth
            |-- train
            |-- val
    |-- testing
        |-- image
        |-- intrinsics
        |-- sparse_depth
    |-- validation
        |-- ground_truth
        |-- image
        |-- intrinsics
        |-- sparse_depth
```
</details>

<details>
<summary> <b> NYUv2 dataset </b> </summary>

For convenience, we have pre-processed the dataset into a subset following [KBNet][kbnet_github], you may download from the browser via:
```
https://yaleedu-my.sharepoint.com/:u:/g/personal/alex_wong_yale_edu/EfTcbD932KBPk81etE7JGRkBEaPpQRTnn4BricoI2ohHNQ?e=LkgTsw
```

or using the following wget command:

```
wget -O nyu_v2_subset.zip 'https://yaleedu-my.sharepoint.com/:u:/g/personal/alex_wong_yale_edu/EfTcbD932KBPk81etE7JGRkBEaPpQRTnn4BricoI2ohHNQ?e=LkgTsw&download=1'

unzip nyu_v2_subset.zip -d data/
mv nyu_v2_subset.zip data/
```

This will download the `nyu_v2_subset.zip` file and unzip it to the data directory.

Note that we will only need the testing split for the dataset.

The zip file is already preprocessed with image and depth frames aligned and synchronized. Alternatively you may want to download the raw data using `setup/nyu_v2/setup_dataset_nyu_v2.sh` or directly from their [website][nyu_v2_dataset], but will need to process the frames using their MATLAB toolbox. We recommend using the pre-processed data above.
</details>

<details>
<summary> <b> ScanNet dataset </b> </summary>

To download ScanNet, please follow the instructions in the [ScanNet][scannet_dataset] website. The authors require signing a Terms of Use. Additionally, you will need to use their toolkit from thier github repository

```
https://github.com/ScanNet/ScanNet
```

to extract the frames as images and depth maps for each scene into the data directory. Note that for each scene, our data processing scripts assume that the exported color images, depth maps, intrinsics, and camera pose are structured as

```
data
|-- scannet
    |-- scans
        |-- sceneXXXX_XX
            |-- export
                |-- color
                |-- depth
                |-- intrinsic
                |-- pose
        ...
    |-- scans_test
        |-- sceneXXXX_XX
            |-- export
                |-- color
                |-- depth
                |-- intrinsic
                |-- pose
```

</details>

<details>
<summary> <b> Make3D dataset </b> </summary>

To download Make3D, please follow the instructions in their [website][make3d_dataset] under Make3D Laser+Image data. For the purpose of our experiments, we will only use the test set.

Note that the depth maps are stored as .mat format, which you can load/convert to PNG using MatLAB (or its python equivalent via `scipy.io.loadmat`).

For convenience, we have assembled the dataset, you may download from the browser via:
```
https://yaleedu-my.sharepoint.com/:u:/g/personal/alex_wong_yale_edu/EUN77bJcgPdKnPOn68AsezQBgzvIaawTXEUd4j31pJhICw?e=QKLvI2
```

or using the following wget command:

```
wget -O make3d.zip 'https://yaleedu-my.sharepoint.com/:u:/g/personal/alex_wong_yale_edu/EUN77bJcgPdKnPOn68AsezQBgzvIaawTXEUd4j31pJhICw?e=QKLvI2&download=1'

unzip make3d.zip -d data/
mv make3d.zip data/
```

```
data
|-- make3d
    |-- images
        |-- img-060705-17.10.14-p-049t000.jpg
        ...
    |-- laser_data
        |-- depth_sph_corr-060705-17.10.14-p-049t000.mat
        ...
```

</details>

### Setting up datasets

Once you have extracted or linked the above datasets to your data directory, you can set them up using
```
python setup/kitti/setup_dataset_kitti.py
python setup/kitti/setup_dataset_kitti_eigen.py
python setup/void/setup_dataset_void.py
python setup/nyu_v2/setup_dataset_nyu_v2.py
python setup/scannnet/setup_dataset_scannet.py
python setup/make3d/setup_dataset_make3d.py
```

which will create RGB camera image triplets (for training), sparse depth maps (if not already provided by the original dataset), and intrinsics files. All will be stored in `<dataset>_derived` directory within your data directory. Each example will be synchronized across different inputs as separate text (`.txt`) files, which contains their relative paths, and will be stored within the training, validation and testing directories with the naming convention of `<dataset>_<split>_<data_type>.txt`.

```
training
|-- void
    |-- supervised
        |-- void_train_ground_truth_1500.txt
        |-- void_train_image_1500.txt
        |-- void_train_intrinsics_1500.txt
        |-- void_train_sparse_depth_1500.txt
        ...
    |-- unsupervised
        |-- void_train_ground_truth_1500.txt
        |-- void_train_image_1500.txt
        |-- void_train_intrinsics_1500.txt
        |-- void_train_sparse_depth_1500.txt
        ...
|-- nyu_v2
    ...
|-- kitti
    ...
|-- scannet
    ...
testing
|-- void
    |-- void_test_ground_truth_1500.txt
    |-- void_test_image_1500.txt
    |-- void_test_intrinsics_1500.txt
    |-- void_test_sparse_depth_1500.txt
|-- nyu_v2
    ...
|-- kitti
    ...
|-- scannet
    ...
|-- make3d
    ...
validation
|-- kitti
    |-- kitti_val_ground_truth.txt
    |-- kitti_val_image.txt
    |-- kitti_val_intrinsics.txt
    |-- kitti_val_sparse_depth.txt
```

If you would like to create your own custom dataset. We recommend following the setup scripts above as examples. We do not assume anything regarding the directory structure of your dataset. The only expectation is that your setup script outputs should be text files with paths aligned across all inputs. Paths pointing to RGB images can be JPG, JPEG, or PNG, depth maps should be 16/32-bit PNGs (where values are scaled by a multiplier, e.g., 256, to preserve precision), and intrinsics should be numpy arrays. Given the paths to them, the inputs can be directly read by our dataloaders.

## Downloading and running pretrained models <a name="downloading-running-pretrained-models"></a>

We provide pretrained models for KBNet, FusionNet (with ScaffNet), and VOICED for monocular depth completion and Monodepth2, HRDepth, and LiteMono for monocular depth estimation. Models are pretrained on KITTI for outdoors and VOID for indoors.

### Downloading pretrained models

To use our pretrained models trained on KITTI and VOID models, you can download them from Google Drive
```
gdown https://drive.google.com/uc?id=1QswQO_W-AhqIEjWPpjG21OboVuW0iQ5o
unzip pretrained_models.zip
```

Note: `gdown` fails intermittently and complains about permission. If that happens, you may also download the models via:
```
https://drive.google.com/file/d/1QswQO_W-AhqIEjWPpjG21OboVuW0iQ5o/view?usp=sharing
```

Once you unzip the file, you will find a directory called `pretrained_models` containing the following file structure:
```
pretrained_models
|-- depth_completion
    |-- kbnet
        |-- kitti
            |-- kbnet-kitti.pth
            |-- posenet-kitti.pth
        |-- void
            |-- kbnet-void1500.pth
            |-- posenet-void1500.pth
    |-- fusionnet
        ...
    |-- scaffnet
        ...
    |-- voiced
        ...
|-- monocular
    |-- hrdepth
        ...
    |-- litemono
        ...
    |-- monodepth2
        ...
```

**AugUndo pretrained models for monocular depth estimation are coming soon!**

### Running pretrained models

We provide run (bash) scripts for each set of pretrained weights on KITTI and VOID; note that we reported the average over 4 runs in our [paper](https://arxiv.org/pdf/2310.09739).

The bash scripts follow the naming convention of `run_<model>-<dataset>.sh` and reside in the following directory structure:

```
bash
|-- depth_completion
    |-- run
        |-- fusionnet
            |-- run_fusionnet-kitti.sh
            |-- run_fusionnet-nyu_v2.sh
            |-- run_fusionnet-scannet.sh
            |-- run_fusionnet-void150.sh
            |-- run_fusionnet-void500.sh
            |-- run_fusionnet-void1500.sh
            |-- run_scaffnet-kitti.sh
            |-- run_scaffnet-void1500.sh
        |-- kbnet
            ...
        |-- voiced
            ...
|-- monocular
    |-- run
        |-- hrdepth
            |-- run_hrdepth-kitti.sh
            |-- run_hrdepth-make3d.sh
            |-- run_hrdepth-nyu_v2.sh
            |-- run_hrdepth-scannet.sh
            |-- run_hrdepth-void1500.sh
        |-- litemono
            ...
        |-- monodepth2
            ...
```

You can run a particular pretrained model by the bash command `bash bash/<task>/run/<model>/run_<model>-<dataset>.sh`. For example:

```
bash bash/depth_completion/run/kbnet/run_kbnet-void1500.sh
bash bash/depth_completion/run/kbnet/run_kbnet-nyu_v2.sh

bash bash/monocular/run/litemono/run_litemono-kitti.sh
bash bash/monocular/run/litemono2/run_litemono-make3d.sh
```

To run these models on your own custom dataset, please set up your data according to the instructions in [Setting up your own dataset](#setting-up-datasets). You can replace the input paths in the run scripts to run inference on your own dataset. For example,

```
--image_path testing/void/void_test_image_500.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_500.txt \
--intrinsics_path testing/void/void_test_intrinsics_500.txt \
--ground_truth_path testing/void/void_test_ground_truth_500.txt \
```
as in `bash/depth_completion/run/kbnet/run_kbnet-void500.sh` which uses KBNet pretrained weights from Void1500 for inference on VOID500.

## Training with AugUndo <a name="training-with-augundo"></a>

We provide bash scripts for training the models with AugUndo on KITTI and on VOID. Note: you must have completed the steps above in order to train. You may train from scratch or finetune an existing model.

### Training from scratch

The bash scripts follow the naming convention of `train_<model>-<dataset>.sh` and reside in the following directory structure:

```
bash
|-- depth_completion
    |-- train
        |-- fusionnet
            |-- train_fusionnet-kitti.sh
            |-- train_fusionnet-void1500.sh
        |-- kbnet
            |-- train_kbnet-kitti.sh
            |-- train_kbnet-void1500.sh
        |-- voiced
            |-- train_voiced-kitti.sh
            |-- train_voiced-void1500.sh
|-- monocular
    |-- train
        |-- hrdepth
            |-- train_hrdepth-kitti.sh
            |-- train_hrdepth-void1500.sh
        |-- litemono
            |-- train_litemono-kitti.sh
            |-- train_litemono-void1500.sh
        |-- monodepth2
            |-- train_monodepth2-kitti.sh
            |-- train_monodepth2-void1500.sh
```

You can run a particular pretrained model by the bash command `bash bash/<task>/train/<model>/train_<model>-<dataset>.sh`. For example:

```
bash bash/depth_completion/train/kbnet/train_kbnet-void1500.sh
bash bash/depth_completion/train/kbnet/train_kbnet-kitti.sh

bash bash/monocular/train/litemono/train_litemono-void1500.sh
bash bash/monocular/train/litemono/train_litemono-kitti.sh
```

### Restoring and finetuning

If you would like to restore a set of pretrained weights to continue training or finetune on another dataset, please follow the directions below.

To restore a set of weights, you can add/modify the `--restore_paths` argument in the training bash script. For example,

```
--checkpoint_path \
    trained_models/depth_completion/kbnet/void1500/kbnet_augundo \
--restore_paths \
    pretrained_models/depth_completion/kbnet/void/kbnet-void1500.pth \
    pretrained_models/depth_completion/kbnet/void/posenet-void1500.pth \

```
will restore the weights of KBNet and PoseNet pretrained on VOID1500. You will find a similar use of `--restore_paths` in the run scripts [above](downloading-running-pretrained-models).

The order of the paths are set based on the (feedforward) conventions of the model. A special case is for FusionNet, which require loading a set of weights for an auxiliary network like ScaffNet if trained from scratch, but otherwise would be contained within the weights of FusionNet.

Train from scratch:
```
--checkpoint_path \
    trained_models/depth_completion/fusionnet/void1500/fusionnet_augundo \
--restore_path \
    pretrained_models/depth_completion/scaffnet/scenenet/scaffnet-scenenet.pth \
```

Restore from pretrained:
```
--checkpoint_path \
    trained_models/depth_completion/fusionnet/void1500/fusionnet_augundo \
--restore_path \
    pretrained_models/depth_completion/fusionnet/void/fusionnet-void1500.pth \
    pretrained_models/depth_completion/fusionnet/void/posenet-void1500.pth \
```

Note that you may omit the path for PoseNet if you would like to finetune the depth model, but train from scratch on the pose model.

Additionally, you may want to take a set of pretrained weights and fintune them on your own dataset.
To finetune on your own custom dataset, please set up your data according to the instructions in [Setting up your own dataset](#setting-up-datasets) and add your unique dataset name into if-else conditions in `depth_completion_model.py` so that it can be selected. You can replace the input paths in the training scripts. For example,

on VOID500
```
--train_image_path training/void/void_train_image_500.txt \
--train_sparse_depth_path training/voidvoid_train_sparse_depth_500.txt \
--train_intrinsics_path training/void/void_train_intrinsics_500.txt \

...

--checkpoint_path \
    trained_models/depth_completion/kbnet/void500/kbnet_augundo \
--restore_paths \
    pretrained_models/depth_completion/kbnet/void/kbnet-void1500.pth \
    pretrained_models/depth_completion/kbnet/void/posenet-void1500.pth \
```

or on VOID 150:
```
--train_image_path training/void/void_train_image_150.txt \
--train_sparse_depth_path training/void/void_train_sparse_depth_150.txt \
--train_intrinsics_path training/void/void_train_intrinsics_150.txt \

...

--checkpoint_path \
    trained_models/depth_completion/kbnet/void150/kbnet_augundo \
--restore_paths \
    pretrained_models/depth_completion/kbnet/void/kbnet-void1500.pth \
    pretrained_models/depth_completion/kbnet/void/posenet-void1500.pth \
```
To monitor your training progress, you may use Tensorboard
```
tensorboard --logdir trained_models/depth_completion/kbnet/void500/kbnet_augundo
tensorboard --logdir trained_models/depth_completion/kbnet/void150/kbnet_augundo
```

## Adding your own models to AugUndo <a name="adding-models-to-augundo"></a>

If you would like to add your own model (outside of those we listed above) to train with AugUndo, you can implement the TemplateModel class in `template_models.py`. You will need to implement the following functions:

```
__init__
transform_inputs
forward_depth
forward_pose
compute_loss
parameters
parameters_depth
parameters_pose
train
eval
to
data_parallel
restore_model
save_model
```

This can be done as a wrapper by importing your own code based into `external_src`,  making a copy of `template_models.py` (e.g., `your_models.py`), and following the instructions within it. Most of the functions can be implemented simply by calling those that you have written in your own model (see `depth/completion/src/kbnet_models.py` for an example.) Alternatively, you may make a copy of `template_models.py` and directly implement your model based on the functions.

Once you have implemented your own model, you can add your unique model name into the if-else conditions in `depth_completion_model.py` so that it can be selected. Enjoy the performance boost!

## Related projects <a name="related-projects"></a>
You may also find the following projects useful:

- [RSA][rsa_github]: *RSA: Resolving Scale Ambiguities in Monocular Depth Estimators through Language Descriptions*. The first work to utilize language for relative- to metric-scale alignment of monocular depth estimators by predicting the transformation parameters from text descriptions of 3D scenes. This work is published in the proceedings of Neural Information Processing Systems (NeurIPS) 2024.
- [ProxyTTA][proxytta_github]: *Test-Time Adaptation for Depth Completion*. The first test-time adaptation method for sparse-to-dense depth completion. ProxyTTA can adapt pretrained models to novel environments by deploying and updating a single adaptation layer with a single optimization step for high-fidelity inference. This work is published in the proceedings of Computer Vision and Pattern Recognition (CVPR) 2024.
- [WorDepth][wordepth_github]: *WorDepth: Variational Language Prior for Monocular Depth Estimation*. The first work to utilize text descriptions of 3D scenes for monocular depth estimation by learning the distribution of 3D scenes for a given description and choosing one that is most compatible with the image. This work is published in the proceedings of Computer Vision and Pattern Recognition (CVPR) 2024.
- [MonDi][mondi_github]: *Monitored Distillation for Positive Congruent Depth Completion*. A positive-congruent (monitored) distillation strategy for training sparse-to-dense depth completion models. The method facilitates unsupervised distillation from a heterogeneous blind ensemble to obtain high-fidelity, real-time models. This work is published in the proceedings of European Conference on Computer Vision (ECCV) 2022.
- [KBNet][kbnet_github]: *Unsupervised Depth Completion with Calibrated Backprojection Layers*. A fast (15 ms/frame) and accurate unsupervised sparse-to-dense depth completion method that introduces a calibrated backprojection layer that improves generalization across sensor platforms. This work is published as an oral paper in the International Conference on Computer Vision (ICCV) 2021.
- [ScaffNet][scaffnet_github]: *Learning Topology from Synthetic Data for Unsupervised Depth Completion*. An unsupervised sparse-to-dense depth completion method that first learns a map from sparse geometry to an initial dense topology from synthetic data (where ground truth comes for free) and amends the initial estimation by validating against the image. This work is published in the Robotics and Automation Letters (RA-L) 2021 and the International Conference on Robotics and Automation (ICRA) 2021.
- [AdaFrame][adaframe_github]: *An Adaptive Framework for Learning Unsupervised Depth Completion*. An adaptive framework for learning unsupervised sparse-to-dense depth completion that balances data fidelity and regularization objectives based on model performance on the data. This work is published in the Robotics and Automation Letters (RA-L) 2021 and the International Conference on Robotics and Automation (ICRA) 2021.
- [VOICED][voiced_github]: *Unsupervised Depth Completion from Visual Inertial Odometry*. An unsupervised sparse-to-dense depth completion method, developed by the authors. The paper introduces Scaffolding for depth completion and a light-weight network to refine it. This work is published in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [VOID][void_github]: from *Unsupervised Depth Completion from Visual Inertial Odometry*. A dataset, developed by the authors, containing indoor and outdoor scenes with non-trivial 6 degrees of freedom. The dataset is published along with this work in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [XIVO][xivo_github]: The Visual-Inertial Odometry system developed at UCLA Vision Lab. This work is built on top of XIVO. The VOID dataset used by this work also leverages XIVO to obtain sparse points and camera poses.
- [GeoSup][geosup_github]: *Geo-Supervised Visual Depth Prediction*. A single image depth prediction method developed by the authors, published in the Robotics and Automation Letters (RA-L) 2019 and the International Conference on Robotics and Automation (ICRA) 2019. This work was awarded **Best Paper in Robot Vision** at ICRA 2019.
- [AdaReg][adareg_github]: *Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction.* A single image depth prediction method that introduces adaptive regularization. This work was published in the proceedings of Computer Vision and Pattern Recognition (CVPR) 2019.

[kitti_dataset]: http://www.cvlibs.net/datasets/kitti/
[make3d_dataset]: http://make3d.cs.cornell.edu/data.html
[nyu_v2_dataset]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
[scannet_dataset]: http://www.scan-net.org/
[void_github]: https://github.com/alexklwong/void-dataset
[voiced_github]: https://github.com/alexklwong/unsupervised-depth-completion-visual-inertial-odometry
[scaffnet_github]: https://github.com/alexklwong/learning-topology-synthetic-data
[adaframe_github]: https://github.com/alexklwong/adaframe-depth-completion
[kbnet_github]: https://github.com/alexklwong/calibrated-backprojection-network
[mondi_github]: https://github.com/alexklwong/mondi-python
[proxytta_github]: https://github.com/seobbro/TTA-depth-completion
[rsa_github]: https://github.com/Adonis-galaxy/RSA
[wordepth_github]: https://github.com/Adonis-galaxy/WorDepth
[xivo_github]: https://github.com/ucla-vision/xivo
[geosup_github]: https://github.com/feixh/GeoSup
[adareg_github]: https://github.com/alexklwong/adareg-monodispnet

## License and disclaimer <a name="license-disclaimer"></a>
This software is property of Yale University, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied.
