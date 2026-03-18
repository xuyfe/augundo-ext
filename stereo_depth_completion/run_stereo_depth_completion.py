'''
Inference entrypoint for stereo AugUndo.

Usage:
    python -m stereo_depth_completion.run_stereo_depth_completion \
        --model bdf \
        --data_path data/kitti_raw_data \
        --filenames_file external_src/stereo_depth_completion/BDF/utils/filenames/kitti_test_files.txt \
        --restore_path checkpoints/stereo_augundo_bdf/final/bdf_model.pth \
        --output_path results/stereo_augundo_bdf
'''

import os
import sys
import importlib.util
import argparse
import numpy as np
import torch

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stereo_depth_completion.stereo_depth_completion_model import get_stereo_model
from stereo_depth_completion.stereo_depth_completion import (
    run, create_bdf_dataloader, create_unos_dataloader)


def get_args():
    parser = argparse.ArgumentParser(description='Stereo AugUndo Inference')

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['bdf', 'unos'],
                        help='stereo depth model to run')

    # Data paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to KITTI raw data')
    parser.add_argument('--filenames_file', type=str, required=True,
                        help='path to filenames text file')

    # Input dimensions
    parser.add_argument('--input_height', type=int, default=256,
                        help='input image height')
    parser.add_argument('--input_width', type=int, default=512,
                        help='input image width')

    # Model checkpoint
    parser.add_argument('--restore_path', type=str, required=True,
                        help='path to model checkpoint')

    # Output
    parser.add_argument('--output_path', type=str, required=True,
                        help='directory for saving output depth/disparity maps')
    parser.add_argument('--save_outputs', action='store_true', default=True,
                        help='save output depth/disparity maps to disk')

    # Evaluation
    parser.add_argument('--min_evaluate_depth', type=float, default=0.001,
                        help='minimum depth for evaluation')
    parser.add_argument('--max_evaluate_depth', type=float, default=80.0,
                        help='maximum depth for evaluation')
    parser.add_argument('--gt_path', type=str, default='',
                        help='path to ground truth for evaluation')

    # BDF-specific
    parser.add_argument('--bdf_model_name', type=str, default='monodepth',
                        choices=['monodepth', 'pwc'],
                        help='BDF backbone model')
    parser.add_argument('--lr_loss_weight', type=float, default=0.5)
    parser.add_argument('--alpha_image_loss', type=float, default=0.85)
    parser.add_argument('--disp_gradient_loss_weight', type=float, default=0.1)
    parser.add_argument('--type_of_2warp', type=int, default=0)

    # UnOS-specific
    parser.add_argument('--unos_mode', type=str, default='depthflow',
                        choices=['stereo', 'depthflow'])
    parser.add_argument('--depth_smooth_weight', type=float, default=10.0)
    parser.add_argument('--ssim_weight', type=float, default=0.85)
    parser.add_argument('--flow_smooth_weight', type=float, default=10.0)
    parser.add_argument('--flow_consist_weight', type=float, default=0.01)
    parser.add_argument('--flow_diff_threshold', type=float, default=4.0)
    parser.add_argument('--num_scales', type=int, default=4)

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for inference')

    return parser.parse_args()


def evaluate_depth_metrics(pred_depths, gt_depths, min_depth, max_depth):
    '''
    Compute standard stereo depth metrics.

    Arg(s):
        pred_depths : np.ndarray
            predicted depth maps
        gt_depths : np.ndarray
            ground truth depth maps
        min_depth : float
            minimum depth for evaluation
        max_depth : float
            maximum depth for evaluation

    Returns:
        dict : metric name -> value
    '''

    mask = (gt_depths > min_depth) & (gt_depths < max_depth) & (pred_depths > 0)

    pred = pred_depths[mask]
    gt = gt_depths[mask]

    if len(pred) == 0:
        return {}

    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'delta_1.25': a1,
        'delta_1.25^2': a2,
        'delta_1.25^3': a3,
    }


def main():
    args = get_args()

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

    # Instantiate and restore model
    model_wrapper = get_stereo_model(args.model, config)
    step, _ = model_wrapper.restore_model(args.restore_path)
    print('Restored model from: {} (step {})'.format(args.restore_path, step))

    # Create dataloader
    if args.model == 'bdf':
        dataloader = create_bdf_dataloader(
            data_path=args.data_path,
            filenames_file=args.filenames_file,
            input_height=args.input_height,
            input_width=args.input_width,
            batch_size=args.batch_size,
            num_threads=4,
            shuffle=False)
    elif args.model == 'unos':
        dataloader = create_unos_dataloader(
            data_dir=args.data_path,
            train_file=args.filenames_file,
            img_height=args.input_height,
            img_width=args.input_width,
            batch_size=args.batch_size,
            num_scales=args.num_scales,
            shuffle=False)

    # Run inference
    run(
        model_name=args.model,
        model_wrapper=model_wrapper,
        dataloader=dataloader,
        output_path=args.output_path,
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        save_outputs=args.save_outputs,
        device=device)

    # Evaluate against ground truth if provided
    if args.gt_path:
        print('\nEvaluating against ground truth...')

        # Load BDF evaluation_utils via importlib to avoid utils name collision
        _bdf_eval_path = os.path.join(
            _project_root, 'external_src', 'stereo_depth_completion', 'BDF',
            'utils', 'evaluation_utils.py')
        _eval_spec = importlib.util.spec_from_file_location('_bdf_eval_utils', _bdf_eval_path)
        _eval_utils = importlib.util.module_from_spec(_eval_spec)
        _eval_spec.loader.exec_module(_eval_utils)

        compute_errors = _eval_utils.compute_errors
        load_gt_disp_kitti = _eval_utils.load_gt_disp_kitti
        convert_disps_to_depths_kitti = _eval_utils.convert_disps_to_depths_kitti

        # Load predicted disparities
        pred_files = sorted([
            f for f in os.listdir(args.output_path) if f.endswith('.npy')
        ])
        pred_disps = [np.load(os.path.join(args.output_path, f)) for f in pred_files]

        if pred_disps:
            pred_disps_arr = np.array(pred_disps)
            print('Loaded {} predicted disparity maps'.format(len(pred_disps)))

            gt_disps = load_gt_disp_kitti(args.gt_path)
            gt_depths, pred_depths, _ = convert_disps_to_depths_kitti(gt_disps, pred_disps_arr)

            all_metrics = []
            for i in range(len(gt_depths)):
                gt_d = gt_depths[i]
                pred_d = pred_depths[i]
                mask = gt_d > 0
                if mask.sum() > 0:
                    metrics = compute_errors(gt_d[mask], pred_d[mask])
                    all_metrics.append(metrics)

            if all_metrics:
                mean_metrics = np.mean(all_metrics, axis=0)
                metric_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
                print('\nEvaluation Results:')
                print('-' * 60)
                for name, val in zip(metric_names, mean_metrics):
                    print('  {:>12s}: {:.4f}'.format(name, val))
                print('-' * 60)


if __name__ == '__main__':
    main()
