# Some of the code are from the TUM evaluation toolkit:
# https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#absolute_trajectory_error_ate

import math
import re
import os
import functools
import numpy as np
import cv2
import torch

from .evaluate_flow import get_scaled_intrinsic_matrix


# ---------------------------------------------------------------------------
# KITTI odometry sequence metadata
# ---------------------------------------------------------------------------

EVAL_SEQS_START_END = {
    "00": [0, 4540],
    "01": [0, 1100],
    "02": [0, 4660],
    "04": [0, 270],
    "05": [0, 2760],
    "06": [0, 1100],
    "07": [0, 1100],
    "08": [1100, 5170],
    "09": [0, 1590],
    "10": [0, 1200],
}

EVAL_SEQS_PATH = {
    "00": "2011_10_03/2011_10_03_drive_0027_sync",
    "01": "2011_10_03/2011_10_03_drive_0042_sync",
    "02": "2011_10_03/2011_10_03_drive_0034_sync",
    "04": "2011_09_30/2011_09_30_drive_0016_sync",
    "05": "2011_09_30/2011_09_30_drive_0018_sync",
    "06": "2011_09_30/2011_09_30_drive_0020_sync",
    "07": "2011_09_30/2011_09_30_drive_0027_sync",
    "08": "2011_09_30/2011_09_30_drive_0028_sync",
    "09": "2011_09_30/2011_09_30_drive_0033_sync",
    "10": "2011_09_30/2011_09_30_drive_0034_sync",
}


def pred_pose(model, opt, seqs):
    """Run pose prediction on KITTI odometry sequences.

    Replaces the TF ``pred_pose(eval_model, opt, sess, seqs)`` function.
    ``model`` must expose a ``pose_net`` attribute (a PoseExpNet) and be in
    eval mode.  Inference is done under ``torch.no_grad()``.

    Args:
        model: the training/eval model wrapper that has a ``pose_net`` attribute.
        opt: argparse namespace with at least ``data_dir``, ``trace``,
             ``img_height``, ``img_width``.
        seqs: list of sequence id strings, e.g. ["09", "10"].
    """
    output_dir = opt.trace
    device = next(model.pose_net.parameters()).device

    for seq_no in seqs:
        root_img_path = os.path.join(
            opt.data_dir, EVAL_SEQS_PATH[seq_no], "image_02", "data")
        frame_start, frame_end = EVAL_SEQS_START_END[seq_no]
        date = EVAL_SEQS_PATH[seq_no].split("/")[0]

        curr_pose_mat = np.identity(4)
        test_result_pose_mat_full = [curr_pose_mat.copy()]
        test_result_pose_mat = [
            np.reshape(curr_pose_mat[0:3, 0:4], [1, -1])
        ]

        with torch.no_grad():
            for i in range(frame_start, frame_end):
                img1 = cv2.imread(
                    os.path.join(root_img_path,
                                 str(i).zfill(10) + ".png"))
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2 = cv2.imread(
                    os.path.join(root_img_path,
                                 str(i + 1).zfill(10) + ".png"))
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

                imgr_path = re.sub(
                    'image_02',
                    'image_03',
                    os.path.join(root_img_path,
                                 str(i).zfill(10) + ".png"))
                imgr = cv2.imread(imgr_path)
                imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)

                img2r_path = re.sub(
                    'image_02',
                    'image_03',
                    os.path.join(root_img_path,
                                 str(i + 1).zfill(10) + ".png"))
                img2r = cv2.imread(img2r_path)
                img2r = cv2.cvtColor(img2r, cv2.COLOR_BGR2RGB)

                orig_H, orig_W = img1.shape[0:2]
                img1 = cv2.resize(img1, (opt.img_width, opt.img_height))
                img2 = cv2.resize(img2, (opt.img_width, opt.img_height))
                imgr = cv2.resize(imgr, (opt.img_width, opt.img_height))
                img2r = cv2.resize(img2r, (opt.img_width, opt.img_height))

                # Convert to float tensors [0, 1], NCHW
                img1_t = torch.from_numpy(
                    img1.astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)
                img2_t = torch.from_numpy(
                    img2.astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)

                # Pose prediction
                pose_vec = model.pose_net(img1_t, img2_t)  # (1, 6)

                # Convert pose vector to matrix
                from ..utils import _pose_vec2mat
                pred_pose_mat_t = _pose_vec2mat(pose_vec)  # (1, 4, 4)
                pred_pose_mat_np = pred_pose_mat_t[0].cpu().numpy()

                # Scale back to absolute scale
                pred_pose_mat_np[0:3, 3] = pred_pose_mat_np[0:3, 3] * 0.3087

                curr_pose_mat = np.matmul(pred_pose_mat_np, curr_pose_mat)
                test_result_pose_mat_full.append(curr_pose_mat.copy())
                test_result_pose_mat.append(
                    np.reshape(
                        np.linalg.inv(curr_pose_mat)[0:3, 0:4], [1, -1]))

        # Save results
        pred_poses_dir = os.path.join(output_dir, "pred_poses")
        os.makedirs(pred_poses_dir, exist_ok=True)
        os.makedirs(os.path.join(pred_poses_dir, seq_no), exist_ok=True)

        np.savetxt(
            os.path.join(pred_poses_dir, seq_no + '.txt'),
            np.concatenate(test_result_pose_mat, axis=0),
            fmt='%1.4e')

        if seq_no in ["09", "10"]:
            gt_file = os.path.join("./pose_gt_data/", seq_no + "_full.txt")
            if os.path.exists(gt_file):
                with open(gt_file) as f:
                    times = f.readlines()
                times = [float(t.split(" ")[0]) for t in times]

                for i in range(len(test_result_pose_mat_full) - 4):
                    curr_snippet = []
                    for j in range(5):
                        curr_snippet.append(
                            np.matmul(
                                test_result_pose_mat_full[i],
                                np.linalg.inv(
                                    test_result_pose_mat_full[i + j])))
                    dump_pose_seq_TUM(
                        os.path.join(pred_poses_dir, seq_no,
                                     str(i).zfill(6) + ".txt"),
                        curr_snippet,
                        times[i:(i + 5)])


# ---------------------------------------------------------------------------
# ATE computation (adopted from SfMLearner)
# ---------------------------------------------------------------------------

def compute_ate(gtruth_file, pred_file):
    gtruth_list = read_file_list(gtruth_file)
    pred_list = read_file_list(pred_file)
    matches = associate(gtruth_list, pred_list, 0, 0.01)
    if len(matches) < 2:
        return False

    gtruth_xyz = np.array(
        [[float(value) for value in gtruth_list[a][0:3]] for a, b in matches])
    pred_xyz = np.array(
        [[float(value) for value in pred_list[b][0:3]] for a, b in matches])

    offset = gtruth_xyz[0] - pred_xyz[0]
    pred_xyz += offset[None, :]

    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / len(matches)
    return rmse


def read_file_list(filename):
    """Read a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp
    and "d1 d2 d3.." is arbitrary data associated to this timestamp.

    Returns:
        dict of (stamp, data) tuples
    """
    with open(filename) as file:
        data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    lst = [[v.strip() for v in line.split(" ") if v.strip() != ""]
           for line in lines if len(line) > 0 and line[0] != "#"]
    lst = [(float(l[0]), l[1:]) for l in lst if len(l) > 1]
    return dict(lst)


def associate(first_list, second_list, offset, max_difference):
    """Associate two dictionaries of (stamp, data).

    As the time stamps never match exactly, we aim to find the closest match
    for every input tuple.
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


# ---------------------------------------------------------------------------
# Rotation / quaternion / euler utilities
# ---------------------------------------------------------------------------

_FLOAT_EPS_4 = np.finfo(float).eps * 4


def rot2quat(R):
    rz, ry, rx = mat2euler(R)
    qw, qx, qy, qz = euler2quat(rz, ry, rx)
    return qw, qx, qy, qz


def quat2mat(q):
    """Calculate rotation matrix from quaternion.

    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    """
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                     [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                     [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


def mat2euler(M, cy_thresh=None, seq='zyx'):
    """Discover Euler angle vector from 3x3 matrix.

    Taken from:
    http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    """
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    cy = math.sqrt(r33 * r33 + r23 * r23)
    if seq == 'zyx':
        if cy > cy_thresh:
            z = math.atan2(-r12, r11)
            y = math.atan2(r13, cy)
            x = math.atan2(-r23, r33)
        else:
            z = math.atan2(r21, r22)
            y = math.atan2(r13, cy)
            x = 0.0
    elif seq == 'xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi / 2
                x = math.atan2(r12, r13)
            else:
                y = -np.pi / 2
    else:
        raise Exception('Sequence not recognized')
    return z, y, x


def euler2mat(z=0, y=0, x=0, isRadian=True):
    """Return matrix for rotations around z, y and x axes.

    Uses the z, then y, then x convention.
    """
    if not isRadian:
        z = ((np.pi) / 180.) * z
        y = ((np.pi) / 180.) * y
        x = ((np.pi) / 180.) * x
    assert z >= (-np.pi) and z < np.pi, 'Inappropriate z: %f' % z
    assert y >= (-np.pi) and y < np.pi, 'Inappropriate y: %f' % y
    assert x >= (-np.pi) and x < np.pi, 'Inappropriate x: %f' % x

    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]]))
    if Ms:
        return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)


def euler2quat(z=0, y=0, x=0, isRadian=True):
    """Return quaternion corresponding to these Euler angles.

    Uses the z, then y, then x convention.
    Returns [qw, qx, qy, qz].
    """
    if not isRadian:
        z = ((np.pi) / 180.) * z
        y = ((np.pi) / 180.) * y
        x = ((np.pi) / 180.) * x
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
        cx * cy * cz - sx * sy * sz,
        cx * sy * sz + cy * cz * sx,
        cx * cz * sy - sx * cy * sz,
        cx * cy * sz + sx * cz * sy
    ])


def pose_vec_to_mat(vec):
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3, 1))
    rot = euler2mat(vec[5], vec[4], vec[3])
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat


def dump_pose_seq_TUM(out_file, poses, times):
    """Save poses in TUM format."""
    with open(out_file, 'w') as f:
        for p in range(len(times)):
            this_pose = poses[p]
            tx = this_pose[0, 3]
            ty = this_pose[1, 3]
            tz = this_pose[2, 3]
            rot = this_pose[:3, :3]
            qw, qx, qy, qz = rot2quat(rot)
            f.write('%f %f %f %f %f %f %f %f\n' %
                    (times[p], tx, ty, tz, qx, qy, qz, qw))
