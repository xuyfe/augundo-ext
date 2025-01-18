#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/train_depth_completion.py \
--train_images_path training/kitti/unsupervised/kitti_train_nonstatic_images.txt \
--train_sparse_depth_path training/kitti/unsupervised/kitti_train_nonstatic_sparse_depth.txt \
--train_intrinsics_path training/kitti/unsupervised/kitti_train_nonstatic_intrinsics.txt \
--val_image_path validation/kitti/kitti_val_image.txt \
--val_sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--val_intrinsics_path validation/kitti/kitti_val_intrinsics.txt \
--val_ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--n_batch 8 \
--n_height 320 \
--n_width 768 \
--model_name voiced \
--network_modules depth pose \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--learning_rates 2e-4 6e-5 3e-5 \
--learning_schedule 16 24 30 \
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
--augmentation_random_rotate_max 5 \
--augmentation_random_crop_and_pad 0.90 1.00 \
--augmentation_random_resize_and_pad -1 -1 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_resize_to_shape -1 -1 \
--augmentation_random_remove_patch_percent_range_image 0.001 0.005 \
--augmentation_random_remove_patch_size_image 5 5 \
--augmentation_random_remove_patch_percent_range_depth -1 -1 \
--augmentation_random_remove_patch_size_depth 1 1 \
--supervision_type unsupervised \
--w_losses w_color=0.20 w_structure=0.80 w_sparse_depth=0.20 w_smoothness=0.05 w_pose=0.01 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--n_step_per_summary 5000 \
--n_image_per_summary 8 \
--n_step_per_checkpoint 5000 \
--start_step_validation 100000 \
--checkpoint_path \
    trained_models/depth_completion/voiced/kitti/voiced_augundo \
--device gpu \
--n_thread 8
