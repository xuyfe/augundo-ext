#!/bin/bash
#SBATCH --job-name=eval_bdf_eigen
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
BDF_SRC="$SENIOR_THESIS/augundo-ext/external_src/stereo_depth_completion/BDF"

DATA_PATH="${DATA_PATH:-$SENIOR_THESIS/augundo-ext/data/scene_flow_2015}"
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/bdf_monodepth"

# BDF saves checkpoints as model_epoch0, model_epoch1, ... (no .pt extension).
# Use CHECKPOINT_FILE for a specific file, or leave unset to use latest model_epoch* in CHECKPOINT_DIR.
if [[ -z "${CHECKPOINT_FILE}" ]]; then
  CHECKPOINT_FILE=$(ls -t "$CHECKPOINT_DIR"/model_epoch* 2>/dev/null | head -1)
  if [[ -z "$CHECKPOINT_FILE" ]]; then
    echo "Error: no model_epoch* found in $CHECKPOINT_DIR"
    exit 1
  fi
fi

echo "Data dir:       $DATA_PATH"
echo "Checkpoint:     $CHECKPOINT_FILE"
echo "CWD:            $(pwd)"

# Run from augundo-ext so "external_src" resolves
cd "$SENIOR_THESIS/augundo-ext" || exit 1

# Evaluate stereo matching
python -u -m external_src.stereo_depth_completion.BDF.test_stereo \
    --data_path "$DATA_PATH" \
    --filenames_file "$BDF_SRC/utils/filenames/eigen_test_files.txt" \
    --checkpoint_path "$CHECKPOINT_FILE"

# Evaluate stereo depth metrics (run after test_stereo.py).
python -u -m external_src.stereo_depth_completion.BDF.utils.evaluate_kitti \
    --split eigen \
    --predicted_disp_path ./disparities.npy \
    --gt_path "$SENIOR_THESIS/augundo-ext/data/kitti_raw_data"

echo "Evaluation completed"
