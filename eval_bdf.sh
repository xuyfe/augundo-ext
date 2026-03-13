#!/bin/bash
#SBATCH --job-name=eval_bdf
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
BDF_SRC="$SENIOR_THESIS/augundo-ext/external_src/stereo_depth_completion/BDF"

DATA_PATH="${DATA_PATH:-$SENIOR_THESIS/augundo-ext/data/scene_flow_2015}"
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/bdf"

echo "Data dir:       $DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "CWD:            $(pwd)"

# Run from augundo-ext so "external_src" resolves
cd "$SENIOR_THESIS/augundo-ext" || exit 1

# Evaluate optical flow
python -u -m external_src.stereo_depth_completion.BDF.test_flow \
    --data_path "$DATA_PATH" \
    --filenames_file "$BDF_SRC/utils/filenames/kitti_flow_val_files_occ_200.txt" \
    --checkpoint_path "$CHECKPOINT_DIR/latest.pt"

# Evaluate stereo matching
python -u -m external_src.stereo_depth_completion.BDF.test_stereo \
    --data_path "$DATA_PATH" \
    --filenames_file "$BDF_SRC/utils/filenames/kitti_stereo_2015_test_files.txt" \
    --checkpoint_path "$CHECKPOINT_DIR/latest.pt"

# Evaluate stereo depth metrics
python -u -m external_src.stereo_depth_completion.BDF.utils.evaluate_kitti \
    --split kitti \
    --predicted_disp_path ./disparities.npy \
    --gt_path "$SENIOR_THESIS/augundo-ext/data/kitti_raw_data"

echo "Evaluation completed"
