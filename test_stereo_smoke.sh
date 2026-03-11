#!/bin/bash
#SBATCH --job-name=test_stereo_smoke
#SBATCH --time=00:10:00
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

# SLURM_SUBMIT_DIR = the directory where you ran "sbatch" from (/home/ox4)

cd "$SLURM_SUBMIT_DIR/augundo-ext"

python depth_completion/src/test_stereo_smoke.py

echo "Stereo smoke test complete"