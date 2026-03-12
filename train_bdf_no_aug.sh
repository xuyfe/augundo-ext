#!/bin/bash
#SBATCH --job-name=train_bdf
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --partition=gpu
#SBATCH --chdir=/home/ox4

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA
module load cuDNN

# virtual environment under /home/ox4/
source augundo-ext/augundo-py310env/bin/activate

# SLURM_SUBMIT_DIR = the directory where you ran "sbatch" from (use repo root)
SENIOR_THESIS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DEPTH_SRC="$SENIOR_THESIS/augundo-ext/depth_completion/src"

echo "Data dir: $SENIOR_THESIS/augundo-ext/data/kitti_raw_data"
echo "CWD:      $DEPTH_SRC"

cd "$DEPTH_SRC" || exit 1
# BDF paper (bdf.pdf §4.1): batch 2, 512×256, Adam 1e-4, LR halved every 3 epochs for 5 times (3,6,9,12,15).
# Augmentations: left-right flip, gamma [0.8,1.2], brightness [0.5,2.0], color shifts [0.8,1.2], each 50% chance.
python train_stereo_depth_completion.py \
    --model_name bridgedepthflow \
    --network_modules stereo \
    --n_batch 2 \
    --n_thread 4 \
    --n_height 256 \
    --n_width 512 \
    --learning_rates 1e-4 5e-5 2.5e-5 1.25e-5 6.25e-6 \
    --learning_schedule 3 6 9 12 15 \
    --train_data_file "$SENIOR_THESIS/augundo-ext/data/unos_train_4frames.txt" \
    --train_data_root "$SENIOR_THESIS/augundo-ext/data/kitti_raw_data" \
    --checkpoint_path "$SENIOR_THESIS/augundo-ext/checkpoints/bridgedepthflow_stereo" \
    --augmentation_random_flip_type horizontal \
    --augmentation_random_gamma 0.8 1.2 \
    --augmentation_random_brightness 0.5 2.0 \
    --augmentation_random_saturation 0.8 1.2

echo "Training completed"