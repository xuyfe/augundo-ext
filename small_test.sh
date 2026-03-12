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

  # unos

python run_stereo_depth_completion.py \
  --left_image_path  ../../data/stereo_val_left_image_small.txt \
  --right_image_path ../../data/stereo_val_right_image_small.txt \
  --sparse_depth_path ../../data/stereo_val_sparse_depth_small.txt \
  --intrinsics_path ../../data/stereo_val_intrinsics_small.txt \
  --ground_truth_path ../../data/stereo_val_ground_truth_small.txt \
  --model_name unos \
  --n_height 256 --n_width 832 \
  --restore_paths ../../checkpoints/unos_stereo/checkpoints-238820/unos-238820.pth \
  --output_path ../../results/unos_stereo_debug \
  --save_outputs