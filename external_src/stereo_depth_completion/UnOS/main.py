"""
Training entry point for UnOS -- PyTorch reimplementation.
Mirrors the original TF main.py with identical CLI flags.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .monodepth_dataloader import MonodepthDataloader
from .models import (Model_stereo, Model_flow, Model_depth, Model_depthflow,
                     Model_eval_stereo, Model_eval_flow,
                     Model_eval_depth, Model_eval_depthflow)
from .test import test

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100
# How often to run a batch through the validation model.
VAL_INTERVAL = 2500
# How often to save a model checkpoint
SAVE_INTERVAL = 2500


def get_args():
    parser = argparse.ArgumentParser(description='UnOS Training')
    parser.add_argument('--trace', type=str, default='./',
                        help='directory for model checkpoints.')
    parser.add_argument('--num_iterations', type=int, default=300000,
                        help='number of training iterations.')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='filepath of a pretrained model to initialize from.')
    parser.add_argument('--mode', type=str, default='',
                        help='selection from ["flow", "depth", "depthflow", "stereo"]')
    parser.add_argument('--train_test', type=str, default='train',
                        help='whether to train or test')
    parser.add_argument('--retrain', type=bool, default=True,
                        help='whether to reset the iteration counter')
    parser.add_argument('--data_dir', type=str,
                        default='augundo-ext/data/kitti_raw_data',
                        help='root filepath of data.')
    parser.add_argument('--train_file', type=str,
                        default='./filenames/kitti_train_files_png_4frames.txt',
                        help='training file')
    parser.add_argument('--gt_2012_dir', type=str,
                        default='augundo-ext/data/stereo_2012',
                        help='directory of ground truth of kitti 2012')
    parser.add_argument('--gt_2015_dir', type=str,
                        default='augundo-ext/data/scene_flow_2015',
                        help='directory of ground truth of kitti 2015')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='the base learning rate of the generator')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='the number of gpu to use')
    parser.add_argument('--img_height', type=int, default=256,
                        help='Image height')
    parser.add_argument('--img_width', type=int, default=832,
                        help='Image width')
    parser.add_argument('--depth_smooth_weight', type=float, default=10.0,
                        help='Weight for depth smoothness')
    parser.add_argument('--ssim_weight', type=float, default=0.85,
                        help='Weight for using ssim loss in pixel loss')
    parser.add_argument('--flow_smooth_weight', type=float, default=10.0,
                        help='Weight for flow smoothness')
    parser.add_argument('--flow_consist_weight', type=float, default=0.01,
                        help='Weight for flow consistent')
    parser.add_argument('--flow_diff_threshold', type=float, default=4.0,
                        help='threshold when comparing optical flow and rigid flow')
    parser.add_argument('--eval_pose', type=str, default='',
                        help='pose seq to evaluate')
    parser.add_argument('--num_scales', type=int, default=4,
                        help='number of multi-scale levels')
    return parser.parse_args()


def main():
    opt = get_args()

    if opt.trace == '':
        raise Exception('--trace must be specified')

    print('Constructing models and inputs.')

    # Select model/eval classes and eval flags
    if opt.mode == 'depthflow':
        Model = Model_depthflow
        Model_eval = Model_eval_depthflow
        opt.eval_flow = True
        opt.eval_depth = True
        opt.eval_mask = True
    elif opt.mode == 'depth':
        Model = Model_depth
        Model_eval = Model_eval_depth
        opt.eval_flow = True
        opt.eval_depth = True
        opt.eval_mask = False
    elif opt.mode == 'flow':
        Model = Model_flow
        Model_eval = Model_eval_flow
        opt.eval_flow = True
        opt.eval_depth = False
        opt.eval_mask = False
    elif opt.mode == 'stereo':
        Model = Model_stereo
        Model_eval = Model_eval_stereo
        opt.eval_flow = False
        opt.eval_depth = True
        opt.eval_mask = False
    else:
        raise ValueError('mode must be one of flow, depth, depthflow or stereo')

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Create dataset and dataloader
    dataset = MonodepthDataloader(opt)
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    # Create training and eval models
    model = Model(opt).to(device)
    eval_model = Model_eval(opt).to(device)

    # Use DataParallel for multi-GPU
    if opt.num_gpus > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    # Load pretrained if specified
    start_itr = 0
    if opt.pretrained_model:
        checkpoint = torch.load(opt.pretrained_model, map_location=device,
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint
            if opt.train_test == 'test' or (not opt.retrain):
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_itr = checkpoint.get('iteration', 0)
            else:
                # Load only model weights (skip optimizer state)
                model.load_state_dict(checkpoint['model_state_dict'],
                                      strict=False)
        else:
            # Plain state dict
            model.load_state_dict(checkpoint, strict=False)

        if opt.retrain:
            start_itr = 0

    # Copy model weights to eval model
    def _sync_eval():
        raw_model = model.module if hasattr(model, 'module') else model
        eval_state = eval_model.state_dict()
        train_state = raw_model.state_dict()
        for key in eval_state:
            if key in train_state:
                eval_state[key] = train_state[key]
        eval_model.load_state_dict(eval_state)

    # Lazy load GT for evaluation
    gt_flows_2012, noc_masks_2012 = None, None
    gt_flows_2015, noc_masks_2015 = None, None
    gt_masks = None

    if opt.eval_flow:
        from .eval.evaluate_flow import load_gt_flow_kitti
        from .eval.evaluate_mask import load_gt_mask
        gt_flows_2012, noc_masks_2012 = load_gt_flow_kitti('kitti_2012', opt)
        gt_flows_2015, noc_masks_2015 = load_gt_flow_kitti('kitti', opt)
        if opt.eval_mask:
            gt_masks = load_gt_mask(opt)

    # TensorBoard (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(opt.trace)
    except ImportError:
        writer = None

    os.makedirs(opt.trace, exist_ok=True)

    # ---- Training loop ----
    if opt.train_test == 'test':
        _sync_eval()
        eval_model.eval()
        test(eval_model, 0, gt_flows_2012, noc_masks_2012,
             gt_flows_2015, noc_masks_2015, gt_masks, opt, device)
        return

    model.train()
    data_iter = iter(dataloader)
    for itr in range(start_itr, opt.num_iterations):
        # Get next batch, restart dataloader if exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        image1, image_r, image2, image2_r, cam2pix, pix2cam = [
            b.to(device) for b in batch]

        optimizer.zero_grad()
        loss, info = model(image1, image_r, image2, image2_r, cam2pix, pix2cam)

        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        optimizer.step()

        # Logging
        if writer and itr % SUMMARY_INTERVAL == 2:
            for key, val in info.items():
                writer.add_scalar(key, val, itr)

        if itr % 100 == 0:
            sys.stderr.write(
                f'iter {itr}: total_loss = {info.get("total_loss", loss.item()):.4f}\n')

        # Save checkpoint
        if itr % SAVE_INTERVAL == 2 and itr > 0:
            raw_model = model.module if hasattr(model, 'module') else model
            ckpt = {
                'iteration': itr,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_path = os.path.join(opt.trace, f'model-{itr}.pt')
            torch.save(ckpt, save_path)
            print(f'Saved checkpoint to {save_path}')

        # Validation
        if itr % VAL_INTERVAL == 2 and itr > 0:
            _sync_eval()
            eval_model.eval()
            test(eval_model, itr, gt_flows_2012, noc_masks_2012,
                 gt_flows_2015, noc_masks_2015, gt_masks, opt, device)
            model.train()

    # Final save
    raw_model = model.module if hasattr(model, 'module') else model
    torch.save({
        'iteration': opt.num_iterations,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(opt.trace, 'model-final.pt'))
    print('Training completed')


if __name__ == '__main__':
    main()
