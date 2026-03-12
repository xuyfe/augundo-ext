#!/bin/bash
#SBATCH --job-name=eval_unos
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

# virtual environment under /home/ox4/
source augundo-ext/augundo-py310env/bin/activate

# SLURM_SUBMIT_DIR = the directory where you ran "sbatch" from (use repo root)
SENIOR_THESIS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# Run from depth_completion/src so imports resolve; paths below are absolute from repo root
DEPTH_SRC="$SENIOR_THESIS/augundo-ext/depth_completion/src"
DATA_DIR="$SENIOR_THESIS/augundo-ext/data"
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/unos_stereo"
OUTPUT_DIR="$SENIOR_THESIS/augundo-ext/results/unos_stereo"

# restore_paths must point to a .pth file. Checkpoints live under checkpoints/bridgedepthflow_stereo/checkpoints-*/bridgedepthflow-*.pth
if [[ -d "$CHECKPOINT_DIR" ]]; then
  RESTORE_PTH=$(ls -t "$CHECKPOINT_DIR"/checkpoints-*/unos-*.pth 2>/dev/null | head -1)
  if [[ -z "$RESTORE_PTH" ]]; then
    echo "Error: no unos-*.pth found under $CHECKPOINT_DIR/checkpoints-*/"
    exit 1
  fi
elif [[ -f "$CHECKPOINT_DIR" ]]; then
  RESTORE_PTH="$CHECKPOINT_DIR"
else
  echo "Error: CHECKPOINT_DIR is not a directory or .pth file: $CHECKPOINT_DIR"
  exit 1
fi

echo "CWD:      $DEPTH_SRC"
echo "Data:     $DATA_DIR"
echo "Restore:  $RESTORE_PTH"
echo "Output:   $OUTPUT_DIR"

cd "$DEPTH_SRC" || exit 1
python run_stereo_depth_completion.py \
    --left_image_path "$DATA_DIR/stereo_val_left_image.txt" \
    --right_image_path "$DATA_DIR/stereo_val_right_image.txt" \
    --sparse_depth_path "$DATA_DIR/stereo_val_sparse_depth.txt" \
    --intrinsics_path "$DATA_DIR/stereo_val_intrinsics.txt" \
    --ground_truth_path "$DATA_DIR/stereo_val_ground_truth.txt" \
    --model_name unos \
    --n_height 256 \
    --n_width 832 \
    --restore_paths "$RESTORE_PTH" \
    --output_path "$OUTPUT_DIR" \
    --save_outputs

echo "Evaluation completed"