#!/bin/bash
#SBATCH --job-name=train_unos_no_aug
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
# learning rates and schedule are from the original paper, as well as image and batch sizes
python train_stereo_depth_completion.py \
    --model_name bridgedepthflow \
    --network_modules stereo \
    --n_batch 2 \
    --n_thread 4 \
    --n_height 256 \
    --n_width 512 \
    --learning_rates 1e-4 5e-5 2.5e-5 1.25e-5 6.25e-6 \
    --learning_schedule 3 6 9 12 15 \
    --train_data_file "$SENIOR_THESIS/augundo-ext/data/unos_train_4frames.txt" \
    --train_data_root "$SENIOR_THESIS/augundo-ext/data/kitti_raw_data" \
    --checkpoint_path "$SENIOR_THESIS/augundo-ext/checkpoints/bridgedepthflow_stereo" \
    --no_augment

echo "Training completed"