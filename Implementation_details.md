# Download Stereo 2012 and Scene Flow 2015

These data sets are used during evaluation for UnOS and BDF

```bash
# the following are used in UnOS and BDF during training and evaluation

# 1. Create and enter the directory
cd /home/ox4/scratch_pi_aw989/ox4/data/

# 2. Download Stereo/Flow 2012 (approx. 2GB)
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip

# 3. Download Scene Flow 2015 (approx. 12GB)
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip

# then, unzip and establish sym links
unzip data_stereo_flow.zip
unzip data_scene_flow.zip

# assuming you're working in augundo-ext,
ln -s /path/to/scene_flow_2015 data/
ln -s /path/to/stereo_2012 data/
```

Note, UnOS only uses the `training/` subset. We renamed the 2015 folder names to `image_0`and `image_1` to match the folder names of the 2012 folder. Also, the calibration files for the 2015 folder have to be downloaded separately. So, we downloaded the calibration files and matched the structure of the 2012 folder.

# Clone OpticalFlowToolKit

UnOS uses the OpticalFlowToolKit by Ruoteng Li during their eval phase. 

```bash
cd augundo-ext/external_src
git clone https://github.com/liruoteng/OpticalFlowToolkit.git
```

# Training, with no AugUndo framework

Make sure you're working in the home (root) directory. The parameters are set according to the ones used in the original papers.

To train (and evaluate) UnOS:

```bash
sbatch augundo-ext/train_unos.sh
```

Note: UnOS training also runs evaluation on the *training* sets of KITTI 2012 and KITTI 2015. But, if you want to run only an inference test:

```bash
sbatch augundo-ext/eval_unos.sh
```

To train BDF:

```bash
sbatch augundo-ext/train_bdf.sh
```

To evaluate BDF:

```bash
sbatch augundo-ext/eval_bdf.sh
```