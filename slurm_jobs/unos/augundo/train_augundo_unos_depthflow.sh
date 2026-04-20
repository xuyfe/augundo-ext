#!/bin/bash
#SBATCH --job-name=train_unos_df
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
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/augundo_unos_depthflow"
mkdir -p "$CHECKPOINT_DIR"

# Initialize from the trained stereo checkpoint (disp weights transfer, flow/pose init random)
STEREO_CHECKPOINT="$SENIOR_THESIS/augundo-ext/checkpoints/augundo_unos_full/final/unos_model.pth"

echo "Data dir:           $DATA_PATH"
echo "Checkpoint dir:     $CHECKPOINT_DIR"
echo "Stereo checkpoint:  $STEREO_CHECKPOINT"
echo "CWD:                $(pwd)"

# Run from augundo-ext so imports resolve
cd "$SENIOR_THESIS/augundo-ext" || exit 1

# Train UnOS depthflow mode, initializing from the stereo-trained checkpoint.
# This follows the UnOS paper's two-stage training: stereo first, then depthflow.
python -u -m external_src.stereo_depth_completion.UnOS.main \
    --mode depthflow \
    --trace "$CHECKPOINT_DIR" \
    --data_dir "$DATA_PATH" \
    --train_file "$UNOS_SRC/filenames/kitti_train_files_png_4frames.txt" \
    --pretrained_model "$STEREO_CHECKPOINT" \
    --retrain True \
    --num_iterations 100000 \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --img_height 256 \
    --img_width 832 \
    --num_scales 4 \
    --depth_smooth_weight 10.0 \
    --ssim_weight 0.85 \
    --flow_smooth_weight 10.0 \
    --flow_consist_weight 0.01 \
    --flow_diff_threshold 4.0 \
    --disp_freeze_iters 10000 \
    --gt_2015_dir "$SENIOR_THESIS/augundo-ext/data/scene_flow_2015/training" \
    --gt_2012_dir "$SENIOR_THESIS/augundo-ext/data/stereo_2012/training"

echo "Training completed"
