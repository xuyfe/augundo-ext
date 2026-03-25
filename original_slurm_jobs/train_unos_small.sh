#!/bin/bash
#SBATCH --job-name=train_unos_small
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

source augundo-ext/augundo-py310env/bin/activate

SENIOR_THESIS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
UNOS_SRC="$SENIOR_THESIS/augundo-ext/external_src/stereo_depth_completion/UnOS"

DATA_PATH="${DATA_PATH:-$SENIOR_THESIS/augundo-ext/data/kitti_raw_data}"
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/unos"
mkdir -p "$CHECKPOINT_DIR"

echo "Data dir:       $DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "CWD:            $(pwd)"

# Run from augundo-ext so "external_src" resolves; use absolute path for train_file
cd "$SENIOR_THESIS/augundo-ext" || exit 1

# we're only training the stereo model for now, 100k iterations should be enough according to the paper.
python -m external_src.stereo_depth_completion.UnOS.main \
    --data_dir "$DATA_PATH" \
    --train_file "$UNOS_SRC/filenames/kitti_train_files_png_4frames_small.txt" \
    --gt_2012_dir "$SENIOR_THESIS/augundo-ext/data/stereo_2012" \
    --gt_2015_dir "$SENIOR_THESIS/augundo-ext/data/scene_flow_2015" \
    --trace "$CHECKPOINT_DIR" \
    --mode stereo \
    --train_test train \
    --retrain True \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --num_iterations 10 \
    --img_height 256 \
    --img_width 832 \
    --num_scales 4 \
    --ssim_weight 0.85 \
    --depth_smooth_weight 10.0 \
    --flow_smooth_weight 10.0 \
    --flow_consist_weight 0.01 \
    --flow_diff_threshold 4.0

echo "Training completed"
