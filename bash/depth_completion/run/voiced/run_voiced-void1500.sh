#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/run_depth_completion.py \
--image_path testing/void/void_test_image_1500.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--intrinsics_path testing/void/void_test_intrinsics_1500.txt \
--ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--restore_paths \
    pretrained_models/depth_completion/voiced/void/voiced-void1500.pth \
--model_name voiced_void \
--network_modules depth \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--output_path \
    pretrained_models/depth_completion/voiced/void/evaluation_results/void1500 \
--device gpu

