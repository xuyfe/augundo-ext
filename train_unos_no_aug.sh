#!/bin/bash
#SBATCH --job-name=train_unos_no_aug
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=2
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

DATA_PATH="${DATA_PATH:-$SENIOR_THESIS/augundo-ext/data/kitti_raw_data}"
echo "Data dir: $DATA_PATH"
echo "CWD:      $DEPTH_SRC"

cd "$DEPTH_SRC" || exit 1
python train_stereo_depth_completion.py \
    --train_data_file "$SENIOR_THESIS/augundo-ext/data/unos_train_4frames.txt" \
    --train_data_root "$SENIOR_THESIS/augundo-ext/data/kitti_raw_data" \
    --model_name unos \
    --network_modules stereo \
    --n_thread 2 \
    --n_batch 4 \
    --n_height 256 \
    --n_width 832 \
    --learning_rates 1e-4 5e-5 \
    --learning_schedule 10 20 \
    --checkpoint_path "$SENIOR_THESIS/augundo-ext/checkpoints/unos_stereo" \
    --no_augment

echo "Training completed"