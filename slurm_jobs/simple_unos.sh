#!/bin/bash
#SBATCH --job-name=train_augundo_unos
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
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/augundo_unos_new"
mkdir -p "$CHECKPOINT_DIR"

echo "Data dir:       $DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"

# Run from augundo-ext so imports resolve
cd "$SENIOR_THESIS/augundo-ext" || exit 1

echo "CWD:            $(pwd)"

python -u -m stereo_depth_completion.train_stereo_depth_completion \
    --model unos \
    --data_path "$DATA_PATH" \
    --filenames_file "$UNOS_SRC/filenames/kitti_train_files_png_4frames.txt" \
    --checkpoint_path "$CHECKPOINT_DIR/" \
    --unos_mode stereo \
    --input_height 256 \
    --input_width 832 \
    --batch_size 4 \
    --num_iterations 100000 \
    --learning_rate 0.0001 \
    --ssim_weight 0.85 \
    --depth_smooth_weight 10.0 \
    --flow_smooth_weight 10.0 \
    --flow_consist_weight 0.01 \
    --flow_diff_threshold 4.0 \
    --num_scales 4 \
    --augmentation_types horizontal_flip \
    --augmentation_probability 1.0 \
    --n_step_per_checkpoint 10000 \
    --n_step_per_summary 1000 \
    --n_thread 4

echo "Training completed"
