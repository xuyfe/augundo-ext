"""
Train stereo depth completion (UnOS / BridgeDepthFlow) on 4-frame UnOS-format data.

Augmentations (AugUndo framework):
  - With augmentations (default): use default args or e.g. --augmentation_probabilities 1.0
  - Without augmentations: add --no_augment or set --augmentation_probabilities 0
"""
import argparse
import torch
from stereo_depth_completion import train


class ParseStrFloatKeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split("=")
                getattr(namespace, self.dest)[key] = float(value)
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))


parser = argparse.ArgumentParser()

# Training input paths (stereo 4-frame format)
parser.add_argument('--train_data_file',
    type=str, required=True, help='Path to training file (UnOS format: left_t right_t left_t+1 right_t+1 calib)')
parser.add_argument('--train_data_root',
    type=str, required=True, help='Root directory for KITTI raw data')
parser.add_argument('--train_sparse_depth_path',
    type=str, default=None, help='Path to list of training sparse depth paths')

# Validation input paths
parser.add_argument('--val_left_image_path',
    type=str, default=None, help='Path to list of validation left image paths')
parser.add_argument('--val_right_image_path',
    type=str, default=None, help='Path to list of validation right image paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, default=None, help='Path to list of validation sparse depth paths')
parser.add_argument('--val_intrinsics_path',
    type=str, default=None, help='Path to list of validation camera intrinsics paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default=None, help='Path to list of validation ground truth depth paths')

# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=4, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=256, help='Height of each sample')
parser.add_argument('--n_width',
    type=int, default=832, help='Width of each sample')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=3, help='Number of input image channels')
parser.add_argument('--input_channels_depth',
    type=int, default=2, help='Number of input depth channels')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range of image intensities after normalization')

# Depth network settings
parser.add_argument('--model_name',
    type=str, default='unos', help='Stereo depth completion model name: unos, bridgedepthflow')
parser.add_argument('--network_modules',
    nargs='+', type=str, default=['stereo'], help='Modules to build: stereo, depth, depthflow')
parser.add_argument('--min_predict_depth',
    type=float, default=1.50, help='Minimum value of predicted depth')
parser.add_argument('--max_predict_depth',
    type=float, default=100.00, help='Maximum value of predicted depth')

# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[1e-4, 5e-5], help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[10, 20], help='Space delimited list to change learning rate')

# Augmentation settings
parser.add_argument('--no_augment',
    action='store_true', help='Disable all augmentations (AugUndo pipeline off). Same as --augmentation_probabilities 0.')
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[1.00], help='Probabilities to use data augmentation. Use 0 to disable (or --no_augment).')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[-1], help='Schedule to change augmentation probability')

# Photometric data augmentations
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_gamma',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_hue',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_gaussian_blur_kernel_size',
    nargs='+', type=int, default=[-1, -1])
parser.add_argument('--augmentation_random_gaussian_blur_sigma_range',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_noise_type',
    type=str, default='none')
parser.add_argument('--augmentation_random_noise_spread',
    type=float, default=-1)

# Geometric data augmentations
parser.add_argument('--augmentation_padding_mode',
    type=str, default='constant')
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=['none'])
parser.add_argument('--augmentation_random_crop_to_shape',
    nargs='+', type=int, default=[-1, -1])
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'])
parser.add_argument('--augmentation_random_rotate_max',
    type=float, default=-1)
parser.add_argument('--augmentation_random_crop_and_pad',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_resize_to_shape',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_resize_and_pad',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_resize_and_crop',
    nargs='+', type=float, default=[-1, -1])

# Occlusion data augmentations
parser.add_argument('--augmentation_random_remove_patch_percent_range_image',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_remove_patch_size_image',
    nargs='+', type=int, default=[-1, -1])
parser.add_argument('--augmentation_random_remove_patch_percent_range_depth',
    nargs='+', type=float, default=[-1, -1])
parser.add_argument('--augmentation_random_remove_patch_size_depth',
    nargs='+', type=int, default=[-1, -1])

# Loss function settings
parser.add_argument('--w_losses',
    nargs='+', type=str, action=ParseStrFloatKeyValueAction, help='Loss weights as key=value pairs')
parser.add_argument('--w_weight_decay_depth',
    type=float, default=0.00)

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.00, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.00, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=5000)
parser.add_argument('--n_step_per_summary',
    type=int, default=1000)
parser.add_argument('--n_image_per_summary',
    type=int, default=4)
parser.add_argument('--start_step_validation',
    type=int, default=5000)
parser.add_argument('--restore_paths',
    nargs='+', type=str, default=[])

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu')
parser.add_argument('--n_thread',
    type=int, default=8)


args = parser.parse_args()

if __name__ == '__main__':

    args.model_name = args.model_name.lower()

    assert len(args.learning_rates) == len(args.learning_schedule)

    args.augmentation_random_crop_type = [
        crop_type.lower() for crop_type in args.augmentation_random_crop_type
    ]
    args.augmentation_random_flip_type = [
        flip_type.lower() for flip_type in args.augmentation_random_flip_type
    ]

    args.device = args.device.lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda' if args.device == 'gpu' else args.device

    if args.w_losses is None:
        args.w_losses = {}

    if args.no_augment:
        args.augmentation_probabilities = [0.0]

    train(train_data_file=args.train_data_file,
          train_data_root=args.train_data_root,
          train_sparse_depth_path=args.train_sparse_depth_path,
          val_left_image_path=args.val_left_image_path,
          val_right_image_path=args.val_right_image_path,
          val_sparse_depth_path=args.val_sparse_depth_path,
          val_intrinsics_path=args.val_intrinsics_path,
          val_ground_truth_path=args.val_ground_truth_path,
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          input_channels_image=args.input_channels_image,
          input_channels_depth=args.input_channels_depth,
          normalized_image_range=args.normalized_image_range,
          model_name=args.model_name,
          network_modules=args.network_modules,
          min_predict_depth=args.min_predict_depth,
          max_predict_depth=args.max_predict_depth,
          w_losses=args.w_losses,
          w_weight_decay_depth=args.w_weight_decay_depth,
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_brightness=args.augmentation_random_brightness,
          augmentation_random_contrast=args.augmentation_random_contrast,
          augmentation_random_gamma=args.augmentation_random_gamma,
          augmentation_random_hue=args.augmentation_random_hue,
          augmentation_random_saturation=args.augmentation_random_saturation,
          augmentation_random_gaussian_blur_kernel_size=args.augmentation_random_gaussian_blur_kernel_size,
          augmentation_random_gaussian_blur_sigma_range=args.augmentation_random_gaussian_blur_sigma_range,
          augmentation_random_noise_type=args.augmentation_random_noise_type,
          augmentation_random_noise_spread=args.augmentation_random_noise_spread,
          augmentation_padding_mode=args.augmentation_padding_mode,
          augmentation_random_crop_type=args.augmentation_random_crop_type,
          augmentation_random_crop_to_shape=args.augmentation_random_crop_to_shape,
          augmentation_random_flip_type=args.augmentation_random_flip_type,
          augmentation_random_rotate_max=args.augmentation_random_rotate_max,
          augmentation_random_crop_and_pad=args.augmentation_random_crop_and_pad,
          augmentation_random_resize_to_shape=args.augmentation_random_resize_to_shape,
          augmentation_random_resize_and_pad=args.augmentation_random_resize_and_pad,
          augmentation_random_resize_and_crop=args.augmentation_random_resize_and_crop,
          augmentation_random_remove_patch_percent_range_image=args.augmentation_random_remove_patch_percent_range_image,
          augmentation_random_remove_patch_size_image=args.augmentation_random_remove_patch_size_image,
          augmentation_random_remove_patch_percent_range_depth=args.augmentation_random_remove_patch_percent_range_depth,
          augmentation_random_remove_patch_size_depth=args.augmentation_random_remove_patch_size_depth,
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          checkpoint_path=args.checkpoint_path,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          n_step_per_summary=args.n_step_per_summary,
          n_image_per_summary=args.n_image_per_summary,
          start_step_validation=args.start_step_validation,
          restore_paths=args.restore_paths,
          device=args.device,
          n_thread=args.n_thread)
