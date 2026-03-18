#!/bin/bash
#SBATCH --job-name=eval_augundo_bdf_native
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

CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/augundo_bdf"
RESULTS_DIR="$SENIOR_THESIS/augundo-ext/results/augundo_bdf"
mkdir -p "$RESULTS_DIR"

# Use CHECKPOINT_FILE env var, or default to the final checkpoint
if [[ -z "${CHECKPOINT_FILE}" ]]; then
    CHECKPOINT_FILE="$CHECKPOINT_DIR/final/bdf_model.pth"
fi

if [[ ! -f "$CHECKPOINT_FILE" ]]; then
    echo "Error: checkpoint not found at $CHECKPOINT_FILE"
    exit 1
fi

echo "Checkpoint:     $CHECKPOINT_FILE"
echo "Results dir:    $RESULTS_DIR"
echo "CWD:            $(pwd)"

# Run from augundo-ext so imports resolve
cd "$SENIOR_THESIS/augundo-ext" || exit 1

# Step 1: Run inference on 200 KITTI 2015 stereo pairs using BDF's test_stereo.py
python -u -m external_src.stereo_depth_completion.BDF.test_stereo \
    --data_path "$SENIOR_THESIS/augundo-ext/data/scene_flow_2015" \
    --filenames_file "$BDF_SRC/utils/filenames/kitti_stereo_2015_test_files_image_01.txt" \
    --checkpoint_path "$CHECKPOINT_FILE" \
    --model_name monodepth \
    --input_height 256 \
    --input_width 512

# Move disparities to results dir
mv ./disparities.npy "$RESULTS_DIR/disparities.npy"

# Step 2: Evaluate against GT
python -u -m external_src.stereo_depth_completion.BDF.utils.evaluate_kitti \
    --split kitti \
    --predicted_disp_path "$RESULTS_DIR/disparities.npy" \
    --gt_path "$SENIOR_THESIS/augundo-ext/data/scene_flow_2015"

echo "Evaluation completed"
