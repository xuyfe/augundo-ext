#!/bin/bash
#SBATCH --job-name=eval_unos
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

echo "Data dir:       $DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "CWD:            $(pwd)"

# Run from augundo-ext so "external_src" resolves
cd "$SENIOR_THESIS/augundo-ext" || exit 1

python -m external_src.stereo_depth_completion.UnOS.main \
    --data_dir "$DATA_PATH" \
    --train_file "$UNOS_SRC/filenames/kitti_train_files_png_4frames.txt" \
    --gt_2012_dir "$SENIOR_THESIS/augundo-ext/data/stereo_2012/training" \
    --gt_2015_dir "$SENIOR_THESIS/augundo-ext/data/scene_flow_2015/training" \
    --trace "$CHECKPOINT_DIR" \
    --mode stereo \
    --train_test test \
    --pretrained_model "$CHECKPOINT_DIR/model-final.pt" \
    --batch_size 4 \
    --img_height 256 \
    --img_width 832 \
    --num_scales 4

echo "Evaluation completed"
