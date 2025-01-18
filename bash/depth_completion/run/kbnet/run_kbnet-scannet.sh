#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/run_depth_completion.py \
--image_path testing/scannet/scannet_test_image_corner.txt \
--sparse_depth_path testing/scannet/scannet_test_sparse_depth_corner.txt \
--intrinsics_path testing/scannet/scannet_test_intrinsics_corner.txt \
--ground_truth_path testing/scannet/scannet_test_ground_truth_corner.txt \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--restore_paths \
    pretrained_models/depth_completion/kbnet/void/kbnet-void1500.pth \
--model_name kbnet_void \
--network_modules depth \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--output_path \
    pretrained_models/depth_completion/kbnet/void/evaluation_results/scannet \
--device gpu
