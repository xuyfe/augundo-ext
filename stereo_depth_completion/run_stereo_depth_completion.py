'''
Inference and evaluation entrypoint for stereo AugUndo.

Evaluation follows each model's native pipeline:
  - BDF: loads 200 KITTI 2015 stereo pairs from scene_flow_2015,
         runs MonodepthNet/PWCDCNet inference, evaluates depth+disparity metrics.
  - UnOS: loads images from scene_flow_2015 (200) and stereo_2012 (194),
          runs Model_eval_stereo inference, evaluates depth+disparity metrics.

Usage (BDF):
    python -m stereo_depth_completion.run_stereo_depth_completion \
        --model bdf \
        --restore_path checkpoints/augundo_bdf/final/bdf_model.pth \
        --gt_path data/scene_flow_2015

Usage (UnOS):
    python -m stereo_depth_completion.run_stereo_depth_completion \
        --model unos \
        --restore_path checkpoints/augundo_unos/final/unos_model.pth \
        --gt_2015_path data/scene_flow_2015/training \
        --gt_2012_path data/stereo_2012/training
'''

import os
import sys
import importlib.util
import argparse
import numpy as np
import cv2
import torch

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _import_from_file(module_name, file_path):
    '''Import a module directly by file path to avoid sys.path name collisions.'''
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_args():
    parser = argparse.ArgumentParser(description='Stereo AugUndo Evaluation')

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['bdf', 'unos'],
                        help='stereo depth model to evaluate')

    # Model checkpoint
    parser.add_argument('--restore_path', type=str, required=True,
                        help='path to model checkpoint')

    # Input dimensions
    parser.add_argument('--input_height', type=int, default=256,
                        help='input image height')
    parser.add_argument('--input_width', type=int, default=512,
                        help='input image width')

    # Output
    parser.add_argument('--output_path', type=str, default='',
                        help='directory for saving output disparities (optional)')

    # GT paths for evaluation
    parser.add_argument('--gt_path', type=str, default='',
                        help='path to scene_flow_2015 root (for BDF eval)')
    parser.add_argument('--gt_2015_path', type=str, default='',
                        help='path to scene_flow_2015/training (for UnOS eval)')
    parser.add_argument('--gt_2012_path', type=str, default='',
                        help='path to stereo_2012/training (for UnOS eval)')

    # BDF-specific
    parser.add_argument('--bdf_model_name', type=str, default='monodepth',
                        choices=['monodepth', 'pwc'],
                        help='BDF backbone model')

    # UnOS-specific
    parser.add_argument('--unos_mode', type=str, default='stereo',
                        choices=['stereo', 'depthflow'])
    parser.add_argument('--num_scales', type=int, default=4)
    parser.add_argument('--depth_smooth_weight', type=float, default=10.0)
    parser.add_argument('--ssim_weight', type=float, default=0.85)
    parser.add_argument('--flow_smooth_weight', type=float, default=10.0)
    parser.add_argument('--flow_consist_weight', type=float, default=0.01)
    parser.add_argument('--flow_diff_threshold', type=float, default=4.0)

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (cuda or cpu)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# BDF evaluation
# ---------------------------------------------------------------------------

def eval_bdf(args, device):
    '''
    Evaluate BDF following the original pipeline:
      1. Load MonodepthNet/PWCDCNet and restore AugUndo checkpoint
      2. Load 200 KITTI 2015 stereo pairs from scene_flow_2015
      3. Run inference (original + horizontally-flipped, take left disparity)
      4. Evaluate depth and disparity metrics against GT
    '''

    _bdf_root = os.path.join(_project_root, 'external_src',
                             'stereo_depth_completion', 'BDF')

    # Import BDF modules via importlib to avoid utils name collision
    if _bdf_root not in sys.path:
        sys.path.insert(0, _bdf_root)

    from models.MonodepthModel import MonodepthNet
    from models.PWC_net import pwc_dc_net

    _bdf_utils = _import_from_file(
        '_bdf_utils', os.path.join(_bdf_root, 'utils', 'utils.py'))
    make_pyramid = _bdf_utils.make_pyramid

    _bdf_eval_utils = _import_from_file(
        '_bdf_eval_utils',
        os.path.join(_bdf_root, 'utils', 'evaluation_utils.py'))
    compute_errors = _bdf_eval_utils.compute_errors
    load_gt_disp_kitti = _bdf_eval_utils.load_gt_disp_kitti
    convert_disps_to_depths_kitti = _bdf_eval_utils.convert_disps_to_depths_kitti

    from utils.scene_dataloader import get_data, myImageFolder, get_transform

    # Build model
    if args.bdf_model_name == 'monodepth':
        net = MonodepthNet().to(device)
    elif args.bdf_model_name == 'pwc':
        net = pwc_dc_net().to(device)
        args.input_width = 832

    # Restore checkpoint
    checkpoint = torch.load(args.restore_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
        step = checkpoint.get('epoch', 0)
    else:
        net.load_state_dict(checkpoint)
        step = 0
    print('Restored BDF model from: {} (step {})'.format(args.restore_path, step))

    net.eval()

    # Load test images (200 KITTI 2015 stereo pairs)
    filenames_file = os.path.join(
        _bdf_root, 'utils', 'filenames',
        'kitti_stereo_2015_test_files_image_01.txt')

    if not args.gt_path:
        print('Error: --gt_path is required for BDF evaluation '
              '(path to scene_flow_2015 root)')
        return

    left_images, right_images = get_data(filenames_file, args.gt_path)
    num_test = len(left_images)
    print('Loaded {} test stereo pairs'.format(num_test))

    class _Param:
        pass
    param = _Param()
    param.input_height = args.input_height
    param.input_width = args.input_width

    test_loader = torch.utils.data.DataLoader(
        myImageFolder(left_images, right_images, None, param),
        batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    # Run inference -- mirrors BDF test_stereo.py exactly
    disparities = np.zeros((num_test, args.input_height, args.input_width),
                           dtype=np.float32)

    with torch.no_grad():
        for batch_idx, (left, right) in enumerate(test_loader):
            # Concatenate original + horizontally flipped
            left_batch = torch.cat(
                (left, torch.from_numpy(
                    np.flip(left.numpy(), 3).copy())), 0)
            right_batch = torch.cat(
                (right, torch.from_numpy(
                    np.flip(right.numpy(), 3).copy())), 0)

            model_input = torch.cat((left_batch, right_batch), 1).to(device)

            if args.bdf_model_name == 'monodepth':
                disp_est_scale, disp_est = net(model_input)
            elif args.bdf_model_name == 'pwc':
                disp_est_scale = net(model_input)
                disp_est = [torch.cat((
                    disp_est_scale[i][:, 0, :, :].unsqueeze(1) /
                    disp_est_scale[i].shape[3],
                    disp_est_scale[i][:, 1, :, :].unsqueeze(1) /
                    disp_est_scale[i].shape[2]), 1) for i in range(4)]

            # Take left disparity at finest scale (negate to get positive values)
            disparities[batch_idx] = \
                -disp_est[0][0, 0, :, :].detach().cpu().numpy()

    print('Inference complete. {} samples'.format(num_test))

    # Save disparities if output path specified
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, 'disparities.npy')
        np.save(save_path, disparities)
        print('Disparities saved to: {}'.format(save_path))

    # Evaluate -- mirrors BDF evaluate_kitti.py (split=kitti)
    print('\nEvaluating against ground truth...')

    gt_disparities = load_gt_disp_kitti(args.gt_path)
    gt_depths, pred_depths, pred_disparities_resized = \
        convert_disps_to_depths_kitti(gt_disparities, disparities)

    num_samples = 200
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80] = 80

        gt_disp = gt_disparities[i]
        mask = gt_disp > 0

        # D1-all metric (standard KITTI stereo benchmark)
        pred_disp = pred_disparities_resized[i]
        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(
            disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        # Depth metrics
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    print('\n{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}'.format(
        'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print('{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}'.format(
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(),
        d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))


# ---------------------------------------------------------------------------
# UnOS evaluation
# ---------------------------------------------------------------------------

def eval_unos(args, device):
    '''
    Evaluate UnOS following the original pipeline:
      1. Load Model_eval_stereo and transfer weights from AugUndo checkpoint
      2. Load images from scene_flow_2015/training (200) and stereo_2012/training (194)
      3. Run inference per-image with calibration
      4. Evaluate depth and disparity metrics against GT
    '''

    _unos_root = os.path.join(_project_root, 'external_src',
                              'stereo_depth_completion', 'UnOS')

    from external_src.stereo_depth_completion.UnOS.models import (
        Model_stereo, Model_eval_stereo)
    from external_src.stereo_depth_completion.UnOS.eval.evaluate_flow import (
        get_scaled_intrinsic_matrix)
    from external_src.stereo_depth_completion.UnOS.eval.evaluate_depth import (
        load_depths, eval_depth)
    from external_src.stereo_depth_completion.UnOS.eval.evaluate_disp import (
        eval_disp_avg)

    # Build opt object for Model_eval_stereo
    class _Opt:
        pass

    opt = _Opt()
    opt.img_height = args.input_height
    opt.img_width = args.input_width
    opt.num_scales = args.num_scales
    opt.depth_smooth_weight = args.depth_smooth_weight
    opt.ssim_weight = args.ssim_weight
    opt.flow_smooth_weight = args.flow_smooth_weight
    opt.flow_consist_weight = args.flow_consist_weight
    opt.flow_diff_threshold = args.flow_diff_threshold

    # Load training model to get weights, then transfer to eval model
    train_model = Model_stereo(opt).to(device)

    checkpoint = torch.load(args.restore_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        train_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        step = checkpoint.get('iteration', 0)
    else:
        train_model.load_state_dict(checkpoint, strict=False)
        step = 0
    print('Restored UnOS model from: {} (step {})'.format(args.restore_path, step))

    # Create eval model and copy weights from training model
    eval_model = Model_eval_stereo(opt).to(device)
    eval_state = eval_model.state_dict()
    train_state = train_model.state_dict()
    for key in eval_state:
        if key in train_state:
            eval_state[key] = train_state[key]
    eval_model.load_state_dict(eval_state)
    eval_model.eval()

    # Free training model memory
    del train_model

    # Determine which eval datasets to run
    eval_datasets = []
    if args.gt_2015_path:
        eval_datasets.append(('kitti_2015', args.gt_2015_path, 200))
    if args.gt_2012_path:
        eval_datasets.append(('kitti_2012', args.gt_2012_path, 194))

    if not eval_datasets:
        print('Error: at least one of --gt_2015_path or --gt_2012_path is '
              'required for UnOS evaluation')
        return

    for eval_name, gt_dir, total_img_num in eval_datasets:
        print('\n--- Evaluating on {} ({} images) ---'.format(
            eval_name, total_img_num))

        test_result_disp = []

        for i in range(total_img_num):
            # Load 4 images: left_t, left_t1, right_t, right_t1
            img1 = cv2.imread(os.path.join(
                gt_dir, 'image_0', '{:06d}_10.png'.format(i)))
            if img1 is None:
                print('Warning: could not load image {} from {}'.format(
                    i, gt_dir))
                continue

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            orig_H, orig_W = img1.shape[0:2]
            img1 = cv2.resize(img1, (args.input_width, args.input_height))

            img2 = cv2.imread(os.path.join(
                gt_dir, 'image_0', '{:06d}_11.png'.format(i)))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img2, (args.input_width, args.input_height))

            imgr = cv2.imread(os.path.join(
                gt_dir, 'image_1', '{:06d}_10.png'.format(i)))
            imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
            imgr = cv2.resize(imgr, (args.input_width, args.input_height))

            img2r = cv2.imread(os.path.join(
                gt_dir, 'image_1', '{:06d}_11.png'.format(i)))
            img2r = cv2.cvtColor(img2r, cv2.COLOR_BGR2RGB)
            img2r = cv2.resize(img2r, (args.input_width, args.input_height))

            # To tensor (B, 3, H, W)
            img1_t = torch.from_numpy(img1).unsqueeze(0).permute(
                0, 3, 1, 2).to(device)
            img2_t = torch.from_numpy(img2).unsqueeze(0).permute(
                0, 3, 1, 2).to(device)
            imgr_t = torch.from_numpy(imgr).unsqueeze(0).permute(
                0, 3, 1, 2).to(device)
            img2r_t = torch.from_numpy(img2r).unsqueeze(0).permute(
                0, 3, 1, 2).to(device)

            # Load calibration
            calib_file = os.path.join(
                gt_dir, 'calib', '{:06d}.txt'.format(i))
            input_intrinsic = get_scaled_intrinsic_matrix(
                calib_file,
                zoom_x=1.0 * args.input_width / orig_W,
                zoom_y=1.0 * args.input_height / orig_H)
            intrinsic_t = torch.from_numpy(
                input_intrinsic).float().to(device)

            # Run inference
            with torch.no_grad():
                eval_model(img1_t, imgr_t, img2_t, img2r_t,
                           intrinsic=intrinsic_t)

            # Collect disparity prediction
            pred_disp = eval_model.pred_disp
            if isinstance(pred_disp, torch.Tensor) and pred_disp.numel() > 1:
                test_result_disp.append(
                    pred_disp.squeeze().cpu().numpy())
            else:
                test_result_disp.append(0.0)

        if len(test_result_disp) == 0:
            print('No images found for {}, skipping'.format(eval_name))
            continue

        # Save disparities if output path specified
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            save_path = os.path.join(
                args.output_path, 'disparities_{}.npy'.format(eval_name))
            np.save(save_path, np.array(test_result_disp, dtype=object))
            print('Disparities saved to: {}'.format(save_path))

        # Depth evaluation (KITTI 2015 only, matching original UnOS test.py)
        if eval_name == 'kitti_2015':
            try:
                gt_depths, pred_depths, gt_disparities, pred_disp_resized = \
                    load_depths(test_result_disp, gt_dir, eval_occ=True)
                abs_rel, sq_rel, rms, log_rms, a1, a2, a3, d1_all = \
                    eval_depth(gt_depths, pred_depths,
                               gt_disparities, pred_disp_resized)
                print('\nDepth metrics (KITTI 2015):')
                print('{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}'.format(
                    'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all',
                    'a1', 'a2', 'a3'))
                print('{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}'.format(
                    abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3))
            except Exception as e:
                print('Depth eval error: {}'.format(e))

            # Disparity evaluation (KITTI 2015)
            try:
                disp_err = eval_disp_avg(
                    test_result_disp, gt_dir, disp_num=0)
                print('\nDisparity metrics (KITTI 2015):')
                print(disp_err)
            except Exception as e:
                print('Disp eval error: {}'.format(e))

        # Disparity evaluation (KITTI 2012)
        if eval_name == 'kitti_2012':
            try:
                disp_err = eval_disp_avg(test_result_disp, gt_dir)
                print('\nDisparity metrics (KITTI 2012):')
                print(disp_err)
            except Exception as e:
                print('Disp eval 2012 error: {}'.format(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.model == 'bdf':
        eval_bdf(args, device)
    elif args.model == 'unos':
        eval_unos(args, device)


if __name__ == '__main__':
    main()
