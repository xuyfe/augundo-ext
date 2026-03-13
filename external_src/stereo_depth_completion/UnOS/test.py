"""
Evaluation / testing function -- PyTorch reimplementation.
Mirrors the original TF test.py.
"""
import numpy as np
import os
import sys
import cv2

import torch

from .eval.evaluate_depth import load_depths, eval_depth
from .eval.evaluate_flow import get_scaled_intrinsic_matrix, eval_flow_avg
from .eval.evaluate_mask import eval_mask
from .eval.evaluate_disp import eval_disp_avg


def test(eval_model, itr, gt_flows_2012, noc_masks_2012,
         gt_flows_2015, noc_masks_2015, gt_masks, opt, device):
    """Run evaluation on KITTI 2012 and 2015 validation sets.

    Args:
        eval_model: Model_eval_* instance (already on device, in eval mode).
        itr: current training iteration (for logging).
        gt_flows_*: preloaded ground-truth flows (or None).
        noc_masks_*: preloaded non-occluded masks (or None).
        gt_masks: preloaded object masks (or None).
        opt: parsed arguments.
        device: torch device.
    """
    sys.stderr.write("Evaluation at iter [%d]: \n" % itr)

    # Pose evaluation (optional)
    if opt.eval_pose and opt.eval_pose != "":
        try:
            from .eval.pose_evaluation_utils import pred_pose
            from .eval.eval_pose import eval_snippet, kittiEvalOdom

            seqs = opt.eval_pose.split(",")
            odom_eval = kittiEvalOdom("./pose_gt_data/")
            odom_eval.eval_seqs = seqs
            pred_pose(eval_model, opt, device, seqs)

            for seq_no in seqs:
                sys.stderr.write("pose seq %s: \n" % seq_no)
                eval_snippet(
                    os.path.join(opt.trace, "pred_poses", seq_no),
                    os.path.join("./pose_gt_data/", seq_no))
            odom_eval.eval(opt.trace + "/pred_poses/")
            sys.stderr.write("pose_prediction_finished \n")
        except Exception as e:
            sys.stderr.write("Pose evaluation error: %s\n" % str(e))

    for eval_data in ["kitti_2012", "kitti_2015"]:
        test_result_disp = []
        test_result_flow_rigid = []
        test_result_flow_optical = []
        test_result_mask = []
        test_result_disp2 = []
        test_image1 = []

        if eval_data == "kitti_2012":
            total_img_num = 194
            gt_dir = opt.gt_2012_dir
        else:
            total_img_num = 200
            gt_dir = opt.gt_2015_dir

        for i in range(total_img_num):
            img1 = cv2.imread(
                os.path.join(gt_dir, "image_0",
                             str(i).zfill(6) + "_10.png"))
            if img1 is None:
                sys.stderr.write(
                    "Warning: could not load image %d from %s\n" % (i, gt_dir))
                continue
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1_orig = img1.copy()
            orig_H, orig_W = img1.shape[0:2]
            img1 = cv2.resize(img1, (opt.img_width, opt.img_height))

            img2 = cv2.imread(
                os.path.join(gt_dir, "image_0",
                             str(i).zfill(6) + "_11.png"))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img2, (opt.img_width, opt.img_height))

            imgr = cv2.imread(
                os.path.join(gt_dir, "image_1",
                             str(i).zfill(6) + "_10.png"))
            imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
            imgr = cv2.resize(imgr, (opt.img_width, opt.img_height))

            img2r = cv2.imread(
                os.path.join(gt_dir, "image_1",
                             str(i).zfill(6) + "_11.png"))
            img2r = cv2.cvtColor(img2r, cv2.COLOR_BGR2RGB)
            img2r = cv2.resize(img2r, (opt.img_width, opt.img_height))

            # To tensor (B, H, W, 3) uint8 for eval model interface
            img1_t = torch.from_numpy(img1).unsqueeze(0).to(device)
            img2_t = torch.from_numpy(img2).unsqueeze(0).to(device)
            imgr_t = torch.from_numpy(imgr).unsqueeze(0).to(device)
            img2r_t = torch.from_numpy(img2r).unsqueeze(0).to(device)

            # Load calibration
            calib_file = os.path.join(gt_dir, "calib",
                                      str(i).zfill(6) + ".txt")
            input_intrinsic = get_scaled_intrinsic_matrix(
                calib_file,
                zoom_x=1.0 * opt.img_width / orig_W,
                zoom_y=1.0 * opt.img_height / orig_H)
            intrinsic_t = torch.from_numpy(input_intrinsic).float().to(device)

            # Run inference
            with torch.no_grad():
                eval_model(img1_t, imgr_t, img2_t, img2r_t,
                           intrinsic=intrinsic_t)

            # Collect results
            def _np(x):
                if isinstance(x, torch.Tensor) and x.numel() > 1:
                    return x.squeeze().cpu().numpy()
                return 0.0

            test_result_flow_rigid.append(_np(eval_model.pred_flow_rigid))
            test_result_flow_optical.append(_np(eval_model.pred_flow_optical))
            test_result_disp.append(_np(eval_model.pred_disp))
            test_result_disp2.append(_np(eval_model.pred_disp2))
            test_result_mask.append(_np(eval_model.pred_mask))
            test_image1.append(img1_orig)

        if len(test_result_disp) == 0:
            sys.stderr.write("No images found for %s, skipping\n" % eval_data)
            continue

        # Depth evaluation
        if hasattr(opt, 'eval_depth') and opt.eval_depth and eval_data == "kitti_2015":
            print("Evaluate depth at iter [%d] %s" % (itr, eval_data))
            try:
                gt_depths, pred_depths, gt_disparities, pred_disp_resized = \
                    load_depths(test_result_disp, gt_dir, eval_occ=True)
                abs_rel, sq_rel, rms, log_rms, a1, a2, a3, d1_all = eval_depth(
                    gt_depths, pred_depths, gt_disparities, pred_disp_resized)
                sys.stderr.write(
                    "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} \n".format(
                        'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all',
                        'a1', 'a2', 'a3'))
                sys.stderr.write(
                    "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n".format(
                        abs_rel, sq_rel, rms, log_rms, d1_all, a1, a2, a3))
            except Exception as e:
                sys.stderr.write("Depth eval error: %s\n" % str(e))

            try:
                disp_err = eval_disp_avg(
                    test_result_disp, gt_dir, disp_num=0,
                    moving_masks=gt_masks)
                sys.stderr.write("disp err 2015 is \n")
                sys.stderr.write(disp_err + "\n")
            except Exception as e:
                sys.stderr.write("Disp eval error: %s\n" % str(e))

            if opt.mode == "depthflow":
                try:
                    disp_err = eval_disp_avg(
                        test_result_disp2, gt_dir, disp_num=1,
                        moving_masks=gt_masks)
                    sys.stderr.write("disp2 err 2015 is \n")
                    sys.stderr.write(disp_err + "\n")
                except Exception as e:
                    sys.stderr.write("Disp2 eval error: %s\n" % str(e))

        if hasattr(opt, 'eval_depth') and opt.eval_depth and eval_data == "kitti_2012":
            try:
                disp_err = eval_disp_avg(test_result_disp, gt_dir)
                sys.stderr.write("disp err 2012 is \n")
                sys.stderr.write(disp_err + "\n")
            except Exception as e:
                sys.stderr.write("Disp eval 2012 error: %s\n" % str(e))

        # Flow evaluation
        if hasattr(opt, 'eval_flow') and opt.eval_flow and eval_data == "kitti_2012":
            if gt_flows_2012 is not None:
                if opt.mode in ["depth", "depthflow"]:
                    epe = eval_flow_avg(gt_flows_2012, noc_masks_2012,
                                        test_result_flow_rigid, opt)
                    sys.stderr.write("epe 2012 rigid is \n")
                    sys.stderr.write(epe + "\n")

                epe = eval_flow_avg(gt_flows_2012, noc_masks_2012,
                                    test_result_flow_optical, opt)
                sys.stderr.write("epe 2012 optical is \n")
                sys.stderr.write(epe + "\n")

        if hasattr(opt, 'eval_flow') and opt.eval_flow and eval_data == "kitti_2015":
            if gt_flows_2015 is not None:
                if opt.mode in ["depth", "depthflow"]:
                    epe = eval_flow_avg(
                        gt_flows_2015, noc_masks_2015,
                        test_result_flow_rigid, opt,
                        moving_masks=gt_masks)
                    sys.stderr.write("epe 2015 rigid is \n")
                    sys.stderr.write(epe + "\n")

                epe = eval_flow_avg(
                    gt_flows_2015, noc_masks_2015,
                    test_result_flow_optical, opt,
                    moving_masks=gt_masks)
                sys.stderr.write("epe 2015 optical is \n")
                sys.stderr.write(epe + "\n")

        # Mask evaluation
        if hasattr(opt, 'eval_mask') and opt.eval_mask and eval_data == "kitti_2015":
            if gt_masks is not None:
                mask_err = eval_mask(test_result_mask, gt_masks, opt)
                sys.stderr.write("mask_err is %s \n" % str(mask_err))
