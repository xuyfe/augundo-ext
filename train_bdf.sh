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

source augundo-ext/augundo-py310env/bin/activate

export TF_CPP_MIN_LOG_LEVEL=3

SENIOR_THESIS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
BDF_SRC="$SENIOR_THESIS/augundo-ext/external_src/stereo_depth_completion/BDF"

DATA_PATH="${DATA_PATH:-$SENIOR_THESIS/augundo-ext/data/kitti_raw_data}"
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/bdf"
mkdir -p "$CHECKPOINT_DIR"

echo "Data dir:       $DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "CWD:            $(pwd)"

# Run from augundo-ext so "external_src" resolves
cd "$SENIOR_THESIS/augundo-ext" || exit 1

python -u -m external_src.stereo_depth_completion.BDF.train \
    --data_path "$DATA_PATH" \
    --filenames_file "$BDF_SRC/utils/filenames/kitti_train_files_png_4frames_png.txt" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --model_name pwc \
    --input_height 256 \
    --input_width 512 \
    --batch_size 2 \
    --num_epochs 15 \
    --learning_rate 1e-4 \
    --alpha_image_loss 0.85 \
    --disp_gradient_loss_weight 0.1 \
    --type_of_2warp 0 \
    --num_threads 4

echo "Training completed"
