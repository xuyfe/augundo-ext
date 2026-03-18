#!/bin/bash
#SBATCH --job-name=eval_augundo_unos
#SBATCH --time=1-00:00:00
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
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/augundo_unos"
RESULTS_DIR="$SENIOR_THESIS/augundo-ext/results/augundo_unos"
mkdir -p "$RESULTS_DIR"

# Use CHECKPOINT_FILE env var, or default to the final checkpoint
if [[ -z "${CHECKPOINT_FILE}" ]]; then
    CHECKPOINT_FILE="$CHECKPOINT_DIR/final/unos_model.pth"
fi

if [[ ! -f "$CHECKPOINT_FILE" ]]; then
    echo "Error: checkpoint not found at $CHECKPOINT_FILE"
    exit 1
fi

echo "Data dir:       $DATA_PATH"
echo "Checkpoint:     $CHECKPOINT_FILE"
echo "Results dir:    $RESULTS_DIR"
echo "CWD:            $(pwd)"

# Run from augundo-ext so imports resolve
cd "$SENIOR_THESIS/augundo-ext" || exit 1

python -u -m stereo_depth_completion.run_stereo_depth_completion \
    --model unos \
    --data_path "$DATA_PATH" \
    --filenames_file "$UNOS_SRC/filenames/kitti_train_files_png_4frames.txt" \
    --restore_path "$CHECKPOINT_FILE" \
    --output_path "$RESULTS_DIR" \
    --unos_mode stereo \
    --input_height 256 \
    --input_width 832 \
    --batch_size 1 \
    --num_scales 4 \
    --save_outputs \
    --gt_path "$SENIOR_THESIS/augundo-ext/data/scene_flow_2015" \
    --min_evaluate_depth 0.001 \
    --max_evaluate_depth 80.0

echo "Evaluation completed"
