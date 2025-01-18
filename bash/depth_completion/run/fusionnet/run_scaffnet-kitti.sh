#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/run_depth_completion.py \
--image_path validation/kitti/kitti_val_image.txt \
--sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--intrinsics_path validation/kitti/kitti_val_intrinsics.txt \
--ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--input_channels_image 3 \
--input_channels_depth 2 \
--normalized_image_range 0 1 \
--restore_paths \
    pretrained_models/depth_completion/scaffnet/vkitti/scaffnet-vkitti.pth \
--model_name scaffnet_vkitti \
--network_modules vggnet08 spatial_pyramid_pool \
--min_predict_depth 1.5 \
--max_predict_depth 80.0 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--output_path \
    pretrained_models/depth_completion/scaffnet/vkitti/evaluation_results/kitti-val \
--save_outputs \
--device gpu
