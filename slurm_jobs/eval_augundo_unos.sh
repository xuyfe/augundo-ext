#!/bin/bash
#SBATCH --job-name=eval_augundo_unos_simple
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

CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/augundo_unos_new"

# Use CHECKPOINT_FILE env var, or default to the final checkpoint
if [[ -z "${CHECKPOINT_FILE}" ]]; then
    CHECKPOINT_FILE="$CHECKPOINT_DIR/final/unos_model.pth"
fi

if [[ ! -f "$CHECKPOINT_FILE" ]]; then
    echo "Error: checkpoint not found at $CHECKPOINT_FILE"
    exit 1
fi

echo "Checkpoint:     $CHECKPOINT_FILE"
echo "CWD:            $(pwd)"

# Run from augundo-ext so imports resolve
cd "$SENIOR_THESIS/augundo-ext" || exit 1

python -u -m stereo_depth_completion.run_stereo_depth_completion \
    --model unos \
    --restore_path "$CHECKPOINT_FILE" \
    --gt_2015_path "$SENIOR_THESIS/augundo-ext/data/scene_flow_2015/training" \
    --gt_2012_path "$SENIOR_THESIS/augundo-ext/data/stereo_2012/training" \
    --unos_mode stereo \
    --input_height 256 \
    --input_width 832 \
    --num_scales 4

echo "Evaluation completed"
