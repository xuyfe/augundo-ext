"""Optical flow evaluation -- PyTorch reimplementation.
Removes TF dependency; uses opt parameter instead of global FLAGS.
"""
import os
import numpy as np
import cv2
import multiprocessing
import functools

from ..flowlib import read_flow_png, flow_to_image


def get_scaled_intrinsic_matrix(calib_file, zoom_x, zoom_y):
    intrinsics = load_intrinsics_raw(calib_file)
    intrinsics = scale_intrinsics(intrinsics, zoom_x, zoom_y)
    intrinsics[0, 1] = 0.0
    intrinsics[1, 0] = 0.0
    intrinsics[2, 0] = 0.0
    intrinsics[2, 1] = 0.0
    return intrinsics


def load_intrinsics_raw(calib_file):
    filedata = read_raw_calib_file(calib_file)
    if "P_rect_02" in filedata:
        P_rect = filedata['P_rect_02']
    else:
        P_rect = filedata['P2']
    P_rect = np.reshape(P_rect, (3, 4))
    intrinsics = P_rect[:3, :3]
    return intrinsics


def read_raw_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def scale_intrinsics(mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def read_flow_gt_worker(dir_gt, i):
    flow_true = read_flow_png(
        os.path.join(dir_gt, "flow_occ", str(i).zfill(6) + "_10.png"))
    flow_noc_true = read_flow_png(
        os.path.join(dir_gt, "flow_noc", str(i).zfill(6) + "_10.png"))
    return flow_true, flow_noc_true[:, :, 2]


def load_gt_flow_kitti(mode, opt):
    """Load ground-truth optical flow for KITTI.

    Args:
        mode: 'kitti_2012' or 'kitti'.
        opt: options namespace with gt_2012_dir and gt_2015_dir.
    """
    if mode == "kitti_2012":
        num_gt = 194
        dir_gt = opt.gt_2012_dir
    elif mode == "kitti":
        num_gt = 200
        dir_gt = opt.gt_2015_dir
    else:
        return [], []

    gt_flows = []
    noc_masks = []

    fun = functools.partial(read_flow_gt_worker, dir_gt)
    pool = multiprocessing.Pool(5)
    results = pool.imap(fun, range(num_gt), chunksize=10)
    pool.close()
    pool.join()

    for result in results:
        gt_flows.append(result[0])
        noc_masks.append(result[1])

    return gt_flows, noc_masks


def calculate_error_rate(epe_map, gt_flow, mask):
    bad_pixels = np.logical_and(
        epe_map * mask > 3,
        epe_map * mask / np.maximum(
            np.sqrt(np.sum(np.square(gt_flow), axis=2)), 1e-10) > 0.05)
    return bad_pixels.sum() / max(mask.sum(), 1.0)


def eval_flow_avg(gt_flows, noc_masks, pred_flows, opt,
                  moving_masks=None, write_img=False):
    error, error_noc, error_occ = 0.0, 0.0, 0.0
    error_move, error_static, error_rate = 0.0, 0.0, 0.0
    error_move_rate, error_static_rate = 0.0, 0.0

    num = len(gt_flows)
    for gt_flow, noc_mask, pred_flow, i in zip(
            gt_flows, noc_masks, pred_flows, range(num)):
        H, W = gt_flow.shape[0:2]

        pred_flow = np.copy(pred_flow)
        pred_flow[:, :, 0] = pred_flow[:, :, 0] / opt.img_width * W
        pred_flow[:, :, 1] = pred_flow[:, :, 1] / opt.img_height * H

        flo_pred = cv2.resize(pred_flow, (W, H),
                              interpolation=cv2.INTER_LINEAR)

        if not os.path.exists(os.path.join(opt.trace, "pred_flow")):
            os.makedirs(os.path.join(opt.trace, "pred_flow"), exist_ok=True)

        if write_img:
            cv2.imwrite(
                os.path.join(opt.trace, "pred_flow",
                             str(i).zfill(6) + "_10.png"),
                cv2.cvtColor(flow_to_image(flo_pred), cv2.COLOR_RGB2BGR))

        epe_map = np.sqrt(
            np.sum(np.square(flo_pred[:, :, 0:2] - gt_flow[:, :, 0:2]),
                   axis=2))
        error += np.sum(epe_map * gt_flow[:, :, 2]) / np.sum(gt_flow[:, :, 2])
        error_noc += np.sum(epe_map * noc_mask) / np.sum(noc_mask)
        error_occ += np.sum(epe_map * (gt_flow[:, :, 2] - noc_mask)) / max(
            np.sum(gt_flow[:, :, 2] - noc_mask), 1.0)
        error_rate += calculate_error_rate(epe_map, gt_flow[:, :, 0:2],
                                           gt_flow[:, :, 2])

        if moving_masks:
            move_mask = moving_masks[i]
            error_move_rate += calculate_error_rate(
                epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2] * move_mask)
            error_static_rate += calculate_error_rate(
                epe_map, gt_flow[:, :, 0:2],
                gt_flow[:, :, 2] * (1.0 - move_mask))
            error_move += np.sum(
                epe_map * gt_flow[:, :, 2] * move_mask) / max(
                    np.sum(gt_flow[:, :, 2] * move_mask), 1.0)
            error_static += np.sum(
                epe_map * gt_flow[:, :, 2] * (1.0 - move_mask)) / max(
                    np.sum(gt_flow[:, :, 2] * (1.0 - move_mask)), 1.0)

    if moving_masks:
        result = "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} \n".format(
            'epe', 'epe_noc', 'epe_occ', 'epe_move', 'epe_static',
            'move_err_rate', 'static_err_rate', 'err_rate')
        result += "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f} \n".format(
            error / num, error_noc / num, error_occ / num, error_move / num,
            error_static / num, error_move_rate / num, error_static_rate / num,
            error_rate / num)
        return result
    else:
        result = "{:>10}, {:>10}, {:>10}, {:>10} \n".format(
            'epe', 'epe_noc', 'epe_occ', 'err_rate')
        result += "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f} \n".format(
            error / num, error_noc / num, error_occ / num, error_rate / num)
        return result
