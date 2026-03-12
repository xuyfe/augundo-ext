#!/bin/bash
#SBATCH --job-name=eval_bdf
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --chdir=/home/ox4

module load Python/3.10.8-GCCcore-12.2.0

# virtual environment under /home/ox4/
source augundo-ext/augundo-py310env/bin/activate

cd /home/ox4/augundo-ext/depth_completion/src

# bdf
python run_stereo_depth_completion.py \
  --left_image_path  ../../data/stereo_val_left_image_small.txt \
  --right_image_path ../../data/stereo_val_right_image_small.txt \
  --sparse_depth_path ../../data/stereo_val_sparse_depth_small.txt \
  --intrinsics_path ../../data/stereo_val_intrinsics_small.txt \
  --ground_truth_path ../../data/stereo_val_ground_truth_small.txt \
  --model_name bridgedepthflow \
  --n_height 256 --n_width 512 \
  --restore_paths ../../checkpoints/bridgedepthflow_stereo/checkpoints-358245/bridgedepthflow-358245.pth \
  --output_path ../../results/bdf_stereo_debug \
  --save_outputs