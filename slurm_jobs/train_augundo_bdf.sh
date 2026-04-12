#!/bin/bash
#SBATCH --job-name=train_augundo_bdf
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

export TF_CPP_MIN_LOG_LEVEL=3

SENIOR_THESIS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
BDF_SRC="$SENIOR_THESIS/augundo-ext/external_src/stereo_depth_completion/BDF"

DATA_PATH="${DATA_PATH:-$SENIOR_THESIS/augundo-ext/data/kitti_raw_data}"
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/augundo_bdf_full"
mkdir -p "$CHECKPOINT_DIR"

echo "Data dir:       $DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "CWD:            $(pwd)"

# Run from augundo-ext so imports resolve
cd "$SENIOR_THESIS/augundo-ext" || exit 1

python -u -m stereo_depth_completion.train_stereo_depth_completion \
    --model bdf \
    --data_path "$DATA_PATH" \
    --filenames_file "$BDF_SRC/utils/filenames/kitti_train_files_png_4frames_png.txt" \
    --checkpoint_path "$CHECKPOINT_DIR/" \
    --bdf_model_name monodepth \
    --input_height 256 \
    --input_width 512 \
    --batch_size 2 \
    --num_epochs 15 \
    --learning_rate 1e-4 \
    --learning_rates 1e-4 5e-5 2.5e-5 1.25e-5 6.25e-6 \
    --learning_schedule 3 6 9 12 \
    --alpha_image_loss 0.85 \
    --disp_gradient_loss_weight 10.0 \
    --temporal_loss_weight 0.1 \
    --lr_loss_weight 0.5 \
    --type_of_2warp 0 \
    --augmentation_types horizontal_flip horizontal_translate color_jitter gaussian_blur noise remove_patch \
    --augmentation_probability 1.0 \
    --augmentation_random_brightness 0.6 1.4 \
    --augmentation_random_contrast 0.6 1.4 \
    --augmentation_random_saturation 0.6 1.4 \
    --augmentation_random_gamma 0.8 1.2 \
    --augmentation_random_hue -0.1 0.1 \
    --augmentation_random_horizontal_translate -0.2 0.2 \
    --augmentation_random_gaussian_blur_kernel_size 3 5 7 9 \
    --augmentation_random_gaussian_blur_sigma_range 0.1 3.0 \
    --augmentation_random_noise_type gaussian \
    --augmentation_random_noise_spread 0.05 \
    --augmentation_random_remove_patch_percent_range 0.01 0.05 \
    --augmentation_random_remove_patch_size 10 10 \
    --checkpoint_every_epoch \
    --n_step_per_summary 100 \
    --n_thread 4

echo "Training completed"
