import sys
import os
import time
# Resolve local packages (models, utils) when run as python -m external_src.stereo_depth_completion.BDF.train
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter
from models.MonodepthModel import *
from models.PWC_net import *
from models.PWC_net import PWCDCNet
from utils.scene_dataloader import *
from utils.utils import *
from models.networks.submodules import *
from models.networks.resample2d_package import Resample2d

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', default='augundo-ext/data/kitti_raw_data')
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=80)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=0.5)
    parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
    parser.add_argument('--type_of_2warp',             type=int,   help='2warp type', default=0)
    parser.add_argument('--summary_freq',              type=int,   help='print/log every N iterations', default=100)
    args = parser.parse_args()
    return args

args = get_args()

# Ensure checkpoint path ends with separator so files go inside the directory
if args.checkpoint_path and not args.checkpoint_path.endswith(os.sep):
    args.checkpoint_path = args.checkpoint_path + os.sep

print(f"[BDF] Model: {args.model_name}", flush=True)
print(f"[BDF] Data path: {args.data_path}", flush=True)
print(f"[BDF] Checkpoint path: {args.checkpoint_path}", flush=True)
print(f"[BDF] Input size: {args.input_height}x{args.input_width}", flush=True)
print(f"[BDF] Batch size: {args.batch_size}, Epochs: {args.num_epochs}, LR: {args.learning_rate}", flush=True)
print(f"[BDF] type_of_2warp: {args.type_of_2warp}", flush=True)

if args.model_name == 'monodepth':
    net = MonodepthNet().cuda()
elif args.model_name == 'pwc':
    net = pwc_dc_net().cuda()
    args.input_width = 832

print(f"[BDF] Final input size (after model override): {args.input_height}x{args.input_width}", flush=True)

left_image_1, left_image_2, right_image_1, right_image_2 = get_kitti_cycle_data(args.filenames_file, args.data_path)
print(f"[BDF] Dataset size: {len(left_image_1)} samples", flush=True)

CycleLoader = torch.utils.data.DataLoader(
         myCycleImageFolder(left_image_1, left_image_2, right_image_1, right_image_2, True, args),
         batch_size = args.batch_size, shuffle = True, num_workers = args.num_threads, drop_last = False)

num_batches = len(CycleLoader)
print(f"[BDF] Batches per epoch: {num_batches}", flush=True)

optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7, 10, 13], gamma=0.5)

# TensorBoard
tb_dir = os.path.join(os.path.dirname(args.checkpoint_path.rstrip(os.sep)), "tb_bdf")
os.makedirs(tb_dir, exist_ok=True)
writer = SummaryWriter(tb_dir)
print(f"[BDF] TensorBoard log dir: {tb_dir}", flush=True)

global_step = 0

print(f"[BDF] Starting training...", flush=True)

for epoch in range(args.num_epochs):
    epoch_start = time.time()
    epoch_loss_sum = 0.0
    epoch_count = 0

    for batch_idx, (left_image_1, left_image_2, right_image_1, right_image_2) in enumerate(CycleLoader, 0):

        optimizer.zero_grad()

        former = torch.cat((left_image_2, left_image_1, right_image_1, left_image_1), 0)
        latter = torch.cat((right_image_2, left_image_2, right_image_2, right_image_1), 0)

        left_pyramid = make_pyramid(former, 4)
        right_pyramid = make_pyramid(latter, 4)

        model_input = torch.cat((former, latter), 1).cuda()
        model_input_2 = torch.cat((latter, former), 1).cuda()

        if args.model_name == 'monodepth':
            disp_est_scale, disp_est = net(model_input)
            disp_est_scale_2, disp_est_2 = net(model_input_2)

        elif args.model_name == 'pwc':
            disp_est_scale = net(model_input)
            disp_est = [torch.cat((disp_est_scale[i][:,0,:,:].unsqueeze(1) / disp_est_scale[i].shape[3],
                                   disp_est_scale[i][:,1,:,:].unsqueeze(1) / disp_est_scale[i].shape[2]), 1) for i in range(4)]
            disp_est_scale_2 = net(model_input_2)
            disp_est_2 = [torch.cat((disp_est_scale_2[i][:,0,:,:].unsqueeze(1) / disp_est_scale_2[i].shape[3],
                                     disp_est_scale_2[i][:,1,:,:].unsqueeze(1) / disp_est_scale_2[i].shape[2]), 1) for i in range(4)]

        border_mask = [create_border_mask(left_pyramid[i], 0.1) for i in range(4)]

        fw_mask = []
        bw_mask = []
        for i in range(4):
            fw, bw, diff_fw, diff_bw = get_mask(disp_est_scale[i], disp_est_scale_2[i], border_mask[i])
            fw += 1e-3
            bw += 1e-3
            fw[[0,1,6,7]] = fw[[0,1,6,7]] * 0 + 1
            bw[[0,1,6,7]] = bw[[0,1,6,7]] * 0 + 1
            fw_detached = fw.clone().detach()
            bw_detached = bw.clone().detach()
            fw_mask.append(fw_detached)
            bw_mask.append(bw_detached)

        #reconstruction from right to left
        left_est = [Resample2d()(right_pyramid[i], disp_est_scale[i]) for i in range(4)]
        l1_left = [torch.abs(left_est[i] - left_pyramid[i]) * fw_mask[i] for i in range(4)]
        l1_reconstruction_loss_left = [torch.mean(l1_left[i]) / torch.mean(fw_mask[i]) for i in range(4)]
        ssim_left = [SSIM(left_est[i] * fw_mask[i], left_pyramid[i] * fw_mask[i]) for i in range(4)]
        ssim_loss_left = [torch.mean(ssim_left[i]) / torch.mean(fw_mask[i]) for i in range(4)]
        image_loss_left  = [args.alpha_image_loss * ssim_loss_left[i] +
                            (1 - args.alpha_image_loss) * l1_reconstruction_loss_left[i]  for i in range(4)]
        image_loss = image_loss_left[0] + image_loss_left[1] + image_loss_left[2] + image_loss_left[3]

        disp_loss = [cal_grad2_error(disp_est_scale[i] / 20, left_pyramid[i], 1.0) for i in range(4)]
        disp_gradient_loss = disp_loss[0] + disp_loss[1] + disp_loss[2] + disp_loss[3]

        #reconstruction from left to right
        right_est = [Resample2d()(left_pyramid[i], disp_est_scale_2[i]) for i in range(4)]
        l1_right = [torch.abs(right_est[i] - right_pyramid[i]) * bw_mask[i] for i in range(4)]
        l1_reconstruction_loss_right = [torch.mean(l1_right[i]) / torch.mean(bw_mask[i]) for i in range(4)]
        ssim_right = [SSIM(right_est[i] * bw_mask[i], right_pyramid[i] * bw_mask[i]) for i in range(4)]
        ssim_loss_right = [torch.mean(ssim_right[i]) / torch.mean(bw_mask[i]) for i in range(4)]
        image_loss_right  = [args.alpha_image_loss * ssim_loss_right[i] +
                             (1 - args.alpha_image_loss) * l1_reconstruction_loss_right[i]  for i in range(4)]
        image_loss_2 = image_loss_right[0] + image_loss_right[1] + image_loss_right[2] + image_loss_right[3]

        disp_loss_2 = [cal_grad2_error(disp_est_scale_2[i] / 20, right_pyramid[i], 1.0) for i in range(4)]
        disp_gradient_loss_2 = disp_loss_2[0] + disp_loss_2[1] + disp_loss_2[2] + disp_loss_2[3]

        #LR consistency
        right_to_left_disp = [- Resample2d()(disp_est_2[i], disp_est_scale[i]) for i in range(4)]
        left_to_right_disp = [- Resample2d()(disp_est[i], disp_est_scale_2[i]) for i in range(4)]

        lr_left_loss  = [torch.mean(torch.abs(right_to_left_disp[i][[0,1,6,7]] - disp_est[i][[0,1,6,7]]))  for i in range(4)]
        lr_right_loss = [torch.mean(torch.abs(left_to_right_disp[i][[0,1,6,7]] - disp_est_2[i][[0,1,6,7]])) for i in range(4)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        loss = image_loss + image_loss_2 + 10 * (disp_gradient_loss + disp_gradient_loss_2) + args.lr_loss_weight * lr_loss

        """
        ##########################################################################################
        #                                                                                        #
        #   batch              7,8                mask for the direction of the reconstruction   #
        #   forward   L_t ------------> R_t                                                      #
        #              |                 |        mask   : L_t+1 ---> L_t   ---> R_t             #
        #          3,4 |                 | 5,6    mask_2 : L_t+1 ---> R_t+1 ---> R_t             #
        #              |                 |        mask_3 : R_t+1 ---> R_t   ---> L_t             #
        #              v                 v        mask_4 : R_t+1 ---> L_t+1 ---> L_t             #
        #             L_t+1 ----------> R_t+1     mask_5 : R_t   ---> L_t   ---> L_t+1           #
        #                      1,2                                                               #
        #                                                                                        #
        ##########################################################################################
        """

        if args.type_of_2warp == 1:
            mask_4 = [fw_mask[i][[2,3]] for i in range(4)]
            warp2_est_4 = [Resample2d()(left_est[i][[0,1]], disp_est_scale[i][[2,3]]) for i in range(4)]
            loss += 0.1 * sum([warp_2(warp2_est_4[i], left_pyramid[i][[6,7]], mask_4[i], args) for i in range(4)])
            mask_5 = [bw_mask[i][[2,3]] for i in range(4)]
            warp2_est_5 = [Resample2d()(left_est[i][[6,7]], disp_est_scale_2[i][[2,3]]) for i in range(4)]
            loss += 0.1 * sum([warp_2(warp2_est_5[i], left_pyramid[i][[0,1]], mask_5[i], args) for i in range(4)])

        elif args.type_of_2warp == 2:
            mask = [Resample2d()(fw_mask[i][[2,3]], disp_est_scale_2[i][[0,1]]) for i in range(4)]
            warp2_est = [Resample2d()(left_est[i][[2,3]], disp_est_scale_2[i][[6,7]]) for i in range(4)]
            loss += 0.1 * sum([warp_2(warp2_est[i], right_pyramid[i][[6,7]], mask[i], args) for i in range(4)])
            mask_3 = [Resample2d()(fw_mask[i][[4,5]], disp_est_scale[i][[0,1]]) for i in range(4)]
            warp2_est_3 = [Resample2d()(left_est[i][[4,5]], disp_est_scale[i][[6,7]]) for i in range(4)]
            loss += 0.1 * sum([warp_2(warp2_est_3[i], left_pyramid[i][[6,7]], mask_3[i], args) for i in range(4)])

        elif args.type_of_2warp == 3:
            mask = [Resample2d()(fw_mask[i][[2,3]], disp_est_scale_2[i][[0,1]]) for i in range(4)]
            warp2_est = [Resample2d()(left_est[i][[2,3]], disp_est_scale_2[i][[6,7]]) for i in range(4)]
            loss += 0.1 * sum([warp_2(warp2_est[i], right_pyramid[i][[6,7]], mask[i], args) for i in range(4)])
            mask_2 = [fw_mask[i][[4,5]] for i in range(4)]
            warp2_est_2 = [Resample2d()(right_est[i][[0,1]], disp_est_scale[i][[4,5]]) for i in range(4)]
            loss += 0.1 * sum([warp_2(warp2_est_2[i], right_pyramid[i][[6,7]], mask_2[i], args) for i in range(4)])

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        epoch_loss_sum += loss_val
        epoch_count += 1
        global_step += 1

        # Log every summary_freq iterations
        if global_step % args.summary_freq == 0:
            img_loss_val = image_loss.item() + image_loss_2.item()
            disp_grad_val = disp_gradient_loss.item() + disp_gradient_loss_2.item()
            lr_loss_val = lr_loss.item()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"[BDF] Epoch {epoch}/{args.num_epochs} | "
                  f"Batch {batch_idx}/{num_batches} | "
                  f"Step {global_step} | "
                  f"Loss: {loss_val:.4f} | "
                  f"Image: {img_loss_val:.4f} | "
                  f"Disp Smooth: {disp_grad_val:.4f} | "
                  f"LR Consist: {lr_loss_val:.4f} | "
                  f"lr: {current_lr:.6f}",
                  flush=True)

            writer.add_scalar('loss/total', loss_val, global_step)
            writer.add_scalar('loss/image', img_loss_val, global_step)
            writer.add_scalar('loss/disp_gradient', disp_grad_val, global_step)
            writer.add_scalar('loss/lr_consistency', lr_loss_val, global_step)
            writer.add_scalar('params/learning_rate', current_lr, global_step)

    scheduler.step()

    epoch_elapsed = time.time() - epoch_start
    avg_loss = epoch_loss_sum / max(epoch_count, 1)
    print(f"[BDF] Epoch {epoch} complete | "
          f"Avg Loss: {avg_loss:.4f} | "
          f"Time: {epoch_elapsed:.1f}s | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}",
          flush=True)

    # Save checkpoint every epoch
    ckpt_path = os.path.join(args.checkpoint_path, f"model_epoch{epoch}")
    state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}
    torch.save(state, ckpt_path)
    print(f"[BDF] Checkpoint saved: {ckpt_path}", flush=True)

writer.close()
print("[BDF] Training finished.", flush=True)
