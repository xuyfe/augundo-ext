#!/bin/bash
# Train UnOS (PWC-Disp) stereo depth with UnOS-style KITTI data.
# Run from augundo-ext repo root.
# Prerequisite: generate list files with generate_unos_stereo_lists.py (see UNOS_KITTI_AND_AUGUNDO.md).

if [[ "$(uname)" == "Darwin" ]]; then
  export DEVICE=cpu
else
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  export DEVICE=gpu
fi

# Paths below are relative to repo root. Create them with:
#   python bash/stereo_depth/generate_unos_stereo_lists.py \
#     --unos_train_file /path/to/UnOS/.../kitti_train_files_png_4frames.txt \
#     --data_root /path/to/kitti_raw --out_dir training/kitti/stereo
LEFT_LIST="${LEFT_LIST:-training/kitti/stereo/train_left.txt}"
RIGHT_LIST="${RIGHT_LIST:-training/kitti/stereo/train_right.txt}"
INTRINSICS_LIST="${INTRINSICS_LIST:-training/kitti/stereo/train_intrinsics.txt}"

python depth_completion/src/train_depth_completion.py \
--train_images_path "$LEFT_LIST" \
--train_stereo_right_path "$RIGHT_LIST" \
--train_intrinsics_path "$INTRINSICS_LIST" \
--n_batch 8 \
--n_height 256 \
--n_width 832 \
--model_name unos_kitti \
--network_modules depth \
--input_channels_image 6 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--learning_rates 1e-4 5e-5 \
--learning_schedule 16 24 \
--augmentation_probabilities 1.0 \
--augmentation_schedule -1 \
--augmentation_random_brightness 0.50 1.50 \
--augmentation_random_contrast 0.50 1.50 \
--augmentation_random_gamma -1 -1 \
--augmentation_random_hue -0.1 0.1 \
--augmentation_random_saturation 0.50 1.50 \
--augmentation_random_noise_type none \
--augmentation_random_noise_spread -1 \
--augmentation_padding_mode edge \
--augmentation_random_crop_type horizontal bottom anchored \
--augmentation_random_crop_to_shape -1 -1 -1 -1 \
--augmentation_random_flip_type horizontal \
--augmentation_random_rotate_max 20 \
--augmentation_random_crop_and_pad 0.90 1.00 \
--augmentation_random_resize_and_pad -1 -1 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_resize_to_shape -1 -1 \
--augmentation_random_remove_patch_percent_range_image 1e-3 5e-3 \
--augmentation_random_remove_patch_size_image 5 5 \
--augmentation_random_remove_patch_percent_range_depth 0.60 0.70 \
--augmentation_random_remove_patch_size_depth 1 1 \
--supervision_type unsupervised \
--w_losses w_color=0.20 w_structure=0.80 w_sparse_depth=0.10 w_smoothness=0.01 w_prior_depth=0.01 threshold_prior_depth=0.30 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--n_step_per_summary 5000 \
--n_image_per_summary 8 \
--n_step_per_checkpoint 5000 \
--start_step_validation 100000 \
--checkpoint_path trained_models/stereo_depth/unos/kitti/unos_stereo \
--device ${DEVICE} \
--n_thread 8
