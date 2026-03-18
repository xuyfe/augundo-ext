'''
Training entrypoint for stereo AugUndo.

Usage:
    python -m stereo_depth_completion.train_stereo_depth_completion \
        --model bdf \
        --data_path data/kitti_raw_data \
        --filenames_file external_src/stereo_depth_completion/BDF/utils/filenames/kitti_train_files_png_4frames_png.txt \
        --checkpoint_path checkpoints/stereo_augundo_bdf \
        --augmentation_types horizontal_flip resize color_jitter
'''

import os
import sys
import argparse
import torch

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stereo_depth_completion.stereo_depth_completion_model import get_stereo_model
from stereo_depth_completion.stereo_depth_completion import (
    train, create_bdf_dataloader, create_unos_dataloader)


def get_args():
    parser = argparse.ArgumentParser(description='Stereo AugUndo Training')

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['bdf', 'unos'],
                        help='stereo depth model to train')

    # Data paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to KITTI raw data')
    parser.add_argument('--filenames_file', type=str, required=True,
                        help='path to filenames text file')

    # Input dimensions
    parser.add_argument('--input_height', type=int, default=256,
                        help='input image height')
    parser.add_argument('--input_width', type=int, default=512,
                        help='input image width (BDF default: 512, UnOS default: 832)')

    # Training
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='number of training epochs (use --num_epochs or --num_iterations, not both)')
    parser.add_argument('--num_iterations', type=int, default=None,
                        help='number of training iterations (use --num_epochs or --num_iterations, not both)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=None,
                        help='learning rate schedule values')
    parser.add_argument('--learning_schedule', type=int, nargs='+', default=None,
                        help='learning rate schedule milestones (epochs)')

    # Augmentation
    parser.add_argument('--augmentation_types', type=str, nargs='+',
                        default=['horizontal_flip', 'color_jitter'],
                        help='augmentation types to enable (horizontal_flip, resize, color_jitter). '
                             'Rotation is NEVER allowed for stereo.')
    parser.add_argument('--augmentation_probability', type=float, default=1.0,
                        help='probability of applying augmentation per sample')
    parser.add_argument('--augmentation_random_brightness', type=float, nargs=2,
                        default=[-1, -1],
                        help='brightness augmentation range')
    parser.add_argument('--augmentation_random_contrast', type=float, nargs=2,
                        default=[-1, -1],
                        help='contrast augmentation range')
    parser.add_argument('--augmentation_random_gamma', type=float, nargs=2,
                        default=[-1, -1],
                        help='gamma augmentation range')
    parser.add_argument('--augmentation_random_hue', type=float, nargs=2,
                        default=[-1, -1],
                        help='hue augmentation range')
    parser.add_argument('--augmentation_random_saturation', type=float, nargs=2,
                        default=[-1, -1],
                        help='saturation augmentation range')
    parser.add_argument('--augmentation_random_resize_and_crop', type=float, nargs=2,
                        default=[-1, -1],
                        help='resize-and-crop scale range [min, max]')
    parser.add_argument('--augmentation_random_resize_and_pad', type=float, nargs=2,
                        default=[-1, -1],
                        help='resize-and-pad scale range [min, max]')
    parser.add_argument('--augmentation_random_resize_to_shape', type=float, nargs=2,
                        default=[-1, -1],
                        help='resize-to-shape scale range [min, max]')
    parser.add_argument('--augmentation_padding_mode', type=str, default='edge',
                        help='padding mode for geometric augmentation')

    # BDF-specific hyperparameters
    parser.add_argument('--bdf_model_name', type=str, default='monodepth',
                        choices=['monodepth', 'pwc'],
                        help='BDF backbone model')
    parser.add_argument('--lr_loss_weight', type=float, default=0.5,
                        help='BDF left-right consistency loss weight')
    parser.add_argument('--alpha_image_loss', type=float, default=0.85,
                        help='BDF weight between SSIM and L1 in image loss')
    parser.add_argument('--disp_gradient_loss_weight', type=float, default=0.1,
                        help='BDF disparity smoothness weight')
    parser.add_argument('--type_of_2warp', type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help='BDF two-warp loss type')

    # UnOS-specific hyperparameters
    parser.add_argument('--unos_mode', type=str, default='depthflow',
                        choices=['stereo', 'depthflow'],
                        help='UnOS training mode')
    parser.add_argument('--depth_smooth_weight', type=float, default=10.0,
                        help='UnOS depth smoothness weight')
    parser.add_argument('--ssim_weight', type=float, default=0.85,
                        help='UnOS SSIM weight in pixel loss')
    parser.add_argument('--flow_smooth_weight', type=float, default=10.0,
                        help='UnOS flow smoothness weight')
    parser.add_argument('--flow_consist_weight', type=float, default=0.01,
                        help='UnOS flow consistency weight')
    parser.add_argument('--flow_diff_threshold', type=float, default=4.0,
                        help='UnOS threshold for comparing optical and rigid flow')
    parser.add_argument('--num_scales', type=int, default=4,
                        help='UnOS number of multi-scale levels')

    # Logging
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='path for saving checkpoints')
    parser.add_argument('--n_step_per_checkpoint', type=int, default=1000,
                        help='steps between checkpoints (ignored if --checkpoint_every_epoch is set)')
    parser.add_argument('--checkpoint_every_epoch', action='store_true', default=False,
                        help='save checkpoint at end of each epoch instead of every N steps')
    parser.add_argument('--n_step_per_summary', type=int, default=100,
                        help='steps between log summaries')

    # Restore
    parser.add_argument('--restore_path', type=str, default='',
                        help='path to checkpoint to restore from')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (cuda or cpu)')
    parser.add_argument('--n_thread', type=int, default=4,
                        help='number of data loader threads')

    return parser.parse_args()


def main():
    args = get_args()

    # Validate: rotation must never be in augmentation_types for stereo
    if 'rotation' in args.augmentation_types:
        raise ValueError(
            'Rotation augmentation is NOT valid for stereo models. '
            'It breaks epipolar rectification. Remove "rotation" from --augmentation_types.')

    if 'vertical_flip' in args.augmentation_types:
        raise ValueError(
            'Vertical flip is NOT valid for stereo models. '
            'Remove "vertical_flip" from --augmentation_types.')

    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Build model config
    if args.model == 'bdf':
        config = {
            'model_name': args.bdf_model_name,
            'input_height': args.input_height,
            'input_width': args.input_width,
            'lr_loss_weight': args.lr_loss_weight,
            'alpha_image_loss': args.alpha_image_loss,
            'disp_gradient_loss_weight': args.disp_gradient_loss_weight,
            'type_of_2warp': args.type_of_2warp,
            'device': device,
        }
    elif args.model == 'unos':
        config = {
            'mode': args.unos_mode,
            'img_height': args.input_height,
            'img_width': args.input_width,
            'depth_smooth_weight': args.depth_smooth_weight,
            'ssim_weight': args.ssim_weight,
            'flow_smooth_weight': args.flow_smooth_weight,
            'flow_consist_weight': args.flow_consist_weight,
            'flow_diff_threshold': args.flow_diff_threshold,
            'num_scales': args.num_scales,
            'device': device,
        }

    # Instantiate model
    model_wrapper = get_stereo_model(args.model, config)

    # Restore checkpoint if specified
    if args.restore_path:
        step, _ = model_wrapper.restore_model(args.restore_path)
        print('Restored from checkpoint: {} (step {})'.format(args.restore_path, step))

    # Create dataloader
    if args.model == 'bdf':
        dataloader = create_bdf_dataloader(
            data_path=args.data_path,
            filenames_file=args.filenames_file,
            input_height=args.input_height,
            input_width=args.input_width,
            batch_size=args.batch_size,
            num_threads=args.n_thread)
    elif args.model == 'unos':
        dataloader = create_unos_dataloader(
            data_dir=args.data_path,
            train_file=args.filenames_file,
            img_height=args.input_height,
            img_width=args.input_width,
            batch_size=args.batch_size,
            num_scales=args.num_scales)

    # Determine augmentation config from --augmentation_types
    flip_type = ['horizontal'] if 'horizontal_flip' in args.augmentation_types else []
    resize_and_crop = args.augmentation_random_resize_and_crop if 'resize' in args.augmentation_types else [-1, -1]
    resize_and_pad = args.augmentation_random_resize_and_pad if 'resize' in args.augmentation_types else [-1, -1]
    resize_to_shape = args.augmentation_random_resize_to_shape if 'resize' in args.augmentation_types else [-1, -1]

    brightness = args.augmentation_random_brightness if 'color_jitter' in args.augmentation_types else [-1, -1]
    contrast = args.augmentation_random_contrast if 'color_jitter' in args.augmentation_types else [-1, -1]
    gamma = args.augmentation_random_gamma if 'color_jitter' in args.augmentation_types else [-1, -1]
    hue = args.augmentation_random_hue if 'color_jitter' in args.augmentation_types else [-1, -1]
    saturation = args.augmentation_random_saturation if 'color_jitter' in args.augmentation_types else [-1, -1]

    # Resolve training duration: either num_iterations or num_epochs
    num_epochs = args.num_epochs
    num_iterations = args.num_iterations
    if num_epochs is None and num_iterations is None:
        # Default: epochs for BDF, iterations for UnOS (matching original scripts)
        if args.model == 'bdf':
            num_epochs = 80
        else:
            num_iterations = 100000

    # Train
    train(
        model_name=args.model,
        model_wrapper=model_wrapper,
        dataloader=dataloader,
        augmentation_probability=args.augmentation_probability,
        augmentation_random_flip_type=flip_type,
        augmentation_random_resize_and_crop=resize_and_crop,
        augmentation_random_resize_and_pad=resize_and_pad,
        augmentation_random_resize_to_shape=resize_to_shape,
        augmentation_random_brightness=brightness,
        augmentation_random_contrast=contrast,
        augmentation_random_gamma=gamma,
        augmentation_random_hue=hue,
        augmentation_random_saturation=saturation,
        augmentation_padding_mode=args.augmentation_padding_mode,
        learning_rate=args.learning_rate,
        learning_rates=args.learning_rates,
        learning_schedule=args.learning_schedule,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        checkpoint_path=args.checkpoint_path,
        n_step_per_checkpoint=args.n_step_per_checkpoint,
        checkpoint_every_epoch=args.checkpoint_every_epoch,
        n_step_per_summary=args.n_step_per_summary,
        device=device,
        n_thread=args.n_thread)


if __name__ == '__main__':
    main()
