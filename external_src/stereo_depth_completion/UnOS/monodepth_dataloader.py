"""
KITTI data loader -- PyTorch reimplementation of UnOS / UnDepthflow.
Adopted from https://github.com/mrharicot/monodepth
"""
from __future__ import absolute_import, division, print_function

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Intrinsics helpers
# ---------------------------------------------------------------------------

def rescale_intrinsics(raw_cam_mat, opt, orig_height, orig_width):
    """Rescale a 3x3 intrinsic matrix to match opt.img_height / img_width."""
    fx = raw_cam_mat[0, 0] * opt.img_width / orig_width
    fy = raw_cam_mat[1, 1] * opt.img_height / orig_height
    cx = raw_cam_mat[0, 2] * opt.img_width / orig_width
    cy = raw_cam_mat[1, 2] * opt.img_height / orig_height
    out = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float32)
    return out


def get_multi_scale_intrinsics(raw_cam_mat, num_scales):
    """Compute cam2pix and pix2cam matrices at *num_scales* resolutions.

    Args:
        raw_cam_mat: (3, 3) numpy array **or** torch tensor.
        num_scales: int.
    Returns:
        cam2pix: (num_scales, 3, 3) tensor.
        pix2cam: (num_scales, 3, 3) tensor.
    """
    if isinstance(raw_cam_mat, np.ndarray):
        raw_cam_mat = torch.from_numpy(raw_cam_mat).float()

    cam2pix_list = []
    for s in range(num_scales):
        factor = 2 ** s
        K = raw_cam_mat.clone()
        K[0, 0] /= factor
        K[0, 2] /= factor
        K[1, 1] /= factor
        K[1, 2] /= factor
        cam2pix_list.append(K)

    cam2pix = torch.stack(cam2pix_list, dim=0)  # (S, 3, 3)
    pix2cam = torch.linalg.inv(cam2pix)          # (S, 3, 3)
    return cam2pix, pix2cam


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MonodepthDataloader(Dataset):
    """PyTorch Dataset that mirrors the original TF MonodepthDataloader.

    Each sample consists of four images (left, right, next_left, next_right)
    plus multi-scale camera intrinsics.
    """

    def __init__(self, opt, training=True):
        super().__init__()
        self.data_path = opt.data_dir
        self.opt = opt
        self.img_height = opt.img_height
        self.img_width = opt.img_width
        self.num_scales = opt.num_scales
        self.training = training

        # Read file list
        with open(opt.train_file, 'r') as f:
            lines = f.read().splitlines()
        self.samples = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                self.samples.append(parts)

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # Image I/O
    # ------------------------------------------------------------------

    def _load_image(self, rel_path):
        """Load an image, convert to float32 [0, 1], resize to target dims."""
        full_path = os.path.join(self.data_path, rel_path)
        img = Image.open(full_path).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        return img, orig_h, orig_w

    def _load_intrinsics(self, rel_path):
        """Parse camera intrinsic file (KITTI format: last line P2 = 3x4)."""
        full_path = os.path.join(self.data_path, rel_path)
        with open(full_path, 'r') as f:
            lines = f.read().splitlines()
        last_line = lines[-1]
        vals = last_line.split()
        # skip the key at position 0 if present
        try:
            float(vals[0])
            float_vals = [float(v) for v in vals]
        except ValueError:
            float_vals = [float(v) for v in vals[1:]]
        mat = np.array(float_vals, dtype=np.float32).reshape(3, 4)
        return mat[:3, :3]

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def augment_image_list(image_list):
        """Apply random colour jitter to a list of images (numpy HWC)."""
        # Random gamma
        gamma = random.uniform(0.8, 1.2)
        image_list = [np.clip(img ** gamma, 0, 1) for img in image_list]

        # Random brightness
        brightness = random.uniform(0.5, 2.0)
        image_list = [np.clip(img * brightness, 0, 1) for img in image_list]

        # Random colour shift
        colors = np.array([random.uniform(0.8, 1.2) for _ in range(3)],
                          dtype=np.float32)
        image_list = [np.clip(img * colors[np.newaxis, np.newaxis, :], 0, 1)
                      for img in image_list]
        return image_list

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        parts = self.samples[index]
        left_path, right_path, next_left_path, next_right_path, calib_path = parts[:5]

        left, orig_h, orig_w = self._load_image(left_path)
        right, _, _ = self._load_image(right_path)
        next_left, _, _ = self._load_image(next_left_path)
        next_right, _, _ = self._load_image(next_right_path)

        raw_cam_mat = self._load_intrinsics(calib_path)
        raw_cam_mat = rescale_intrinsics(raw_cam_mat, self.opt,
                                         float(orig_h), float(orig_w))

        # Random left-right flip (swap left <-> right)
        if self.training and random.random() > 0.5:
            left, right = np.fliplr(right).copy(), np.fliplr(left).copy()
            next_left, next_right = (np.fliplr(next_right).copy(),
                                     np.fliplr(next_left).copy())

        # Random front-back swap (swap current <-> next)
        if self.training and random.random() > 0.5:
            left, next_left = next_left, left
            right, next_right = next_right, right

        # Multi-scale intrinsics
        cam2pix, pix2cam = get_multi_scale_intrinsics(raw_cam_mat,
                                                       self.num_scales)

        # Convert HWC float32 numpy → CHW float32 tensor
        left = torch.from_numpy(left.transpose(2, 0, 1)).float()
        right = torch.from_numpy(right.transpose(2, 0, 1)).float()
        next_left = torch.from_numpy(next_left.transpose(2, 0, 1)).float()
        next_right = torch.from_numpy(next_right.transpose(2, 0, 1)).float()

        return left, right, next_left, next_right, cam2pix, pix2cam
