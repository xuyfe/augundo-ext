'''
Stereo dataloader for AugUndo stereo depth completion.

Both UnOS and BridgeDepthFlow use stereo image pairs. This shared dataloader handles:
- Left and right RGB image pairs
- Sparse depth maps (corresponding to left camera)
- Camera intrinsics (3x3 matrix)
- Stereo baseline (from KITTI calibration)
- Optional temporal frames for methods that use temporal consistency
- Optional ground truth depth for supervised training/evaluation

Data format notes:
- Sparse depth maps correspond to the LEFT camera (standard KITTI convention)
- Both models expect images in [0, 1] float32 range
- Images are in CHW format (PyTorch convention)

Both models share the same data requirements so a single dataloader suffices.
'''

import torch
import numpy as np
from utils.src import data_utils
from depth_completion.src.datasets import random_crop


def load_stereo_pair(left_path, right_path, data_format='CHW'):
    '''
    Load a stereo pair of images

    Arg(s):
        left_path : str
            path to left camera image
        right_path : str
            path to right camera image
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : left image
        numpy[float32] : right image
    '''

    left = data_utils.load_image(left_path, normalize=False, data_format=data_format)
    right = data_utils.load_image(right_path, normalize=False, data_format=data_format)
    return left, right


def parse_kitti_calibration(calib_path, cam_id=2):
    '''
    Parse KITTI calibration file to get intrinsics and baseline.

    Reads calib_cam_to_cam.txt and extracts the 3x3 intrinsics matrix for the
    specified camera and the stereo baseline between camera 2 (left color) and
    camera 3 (right color).

    Arg(s):
        calib_path : str
            path to calib_cam_to_cam.txt
        cam_id : int
            camera id (2 for left color, 3 for right color)
    Returns:
        numpy[float32] : 3x3 intrinsics matrix
        float : stereo baseline in meters
    '''

    calib_data = data_utils.load_calibration(calib_path)

    # Extract 3x4 projection matrices for left and right color cameras
    P_rect_02 = calib_data['P_rect_02'].reshape(3, 4)
    P_rect_03 = calib_data['P_rect_03'].reshape(3, 4)

    # Intrinsics from the upper-left 3x3 of the requested camera's projection matrix
    P_rect = calib_data['P_rect_0{}'.format(cam_id)].reshape(3, 4)
    intrinsics = P_rect[0:3, 0:3].astype(np.float32)

    # Baseline = |t_x_right - t_x_left| where t_x = P[0,3] / P[0,0]
    # P[0,3] = -f_x * b_x, so b_x = -P[0,3] / P[0,0]
    t_x_left = P_rect_02[0, 3] / P_rect_02[0, 0]
    t_x_right = P_rect_03[0, 3] / P_rect_03[0, 0]
    baseline = np.float32(np.abs(t_x_right - t_x_left))

    return intrinsics, baseline


def derive_right_image_path(left_path):
    '''
    Derive the right camera image path from a left camera image path.

    For KITTI, replaces image_02 with image_03 in the path.

    Arg(s):
        left_path : str
            path to left camera image (must contain 'image_02')
    Returns:
        str : path to corresponding right camera image
    '''

    return left_path.replace('image_02', 'image_03')


class StereoDepthCompletionTrainingDataset(torch.utils.data.Dataset):
    '''
    Training dataset for stereo depth completion.

    Fetches:
        (1) Left image at time t
        (2) Right image at time t
        (3) Sparse depth map at time t (left camera)
        (4) Camera intrinsics (3x3)
        (5) Stereo baseline (float scalar)

    For temporal methods (UnOS depth/depthflow mode), also fetches:
        (6) Left image at time t+1
        (7) Right image at time t+1

    Arg(s):
        left_image_paths : list[str]
            paths to left camera images at time t
        right_image_paths : list[str]
            paths to right camera images at time t
        sparse_depth_paths : list[str]
            paths to sparse depth maps at time t (left camera)
        intrinsics_paths : list[str]
            paths to 3x3 intrinsic camera calibration matrices (.npy files)
            or paths to KITTI calib_cam_to_cam.txt files
        left_image_next_paths : list[str] or None
            paths to left camera images at time t+1 (for temporal methods)
        right_image_next_paths : list[str] or None
            paths to right camera images at time t+1 (for temporal methods)
        ground_truth_paths : list[str] or None
            paths to ground truth depth maps at time t
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
        load_intrinsics_from_calib : bool
            if True, intrinsics_paths point to KITTI calib_cam_to_cam.txt files
            and intrinsics/baseline are parsed from them; otherwise intrinsics_paths
            point to .npy files and baseline must be provided separately
        default_baseline : float
            default stereo baseline in meters (used when load_intrinsics_from_calib=False)
    '''

    def __init__(self,
                 left_image_paths,
                 right_image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 left_image_next_paths=None,
                 right_image_next_paths=None,
                 ground_truth_paths=None,
                 random_crop_shape=None,
                 random_crop_type=None,
                 load_intrinsics_from_calib=False,
                 default_baseline=0.54):

        self.n_sample = len(left_image_paths)

        # Validate required paths have consistent length
        for paths in [right_image_paths, sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.left_image_paths = left_image_paths
        self.right_image_paths = right_image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        # Optional temporal frames
        self.has_temporal = left_image_next_paths is not None
        if self.has_temporal:
            assert right_image_next_paths is not None
            assert len(left_image_next_paths) == self.n_sample
            assert len(right_image_next_paths) == self.n_sample
        self.left_image_next_paths = left_image_next_paths
        self.right_image_next_paths = right_image_next_paths

        # Optional ground truth
        self.has_ground_truth = \
            ground_truth_paths is not None and None not in ground_truth_paths
        if self.has_ground_truth:
            assert len(ground_truth_paths) == self.n_sample
        self.ground_truth_paths = ground_truth_paths

        # Crop settings
        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape
        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        # Intrinsics loading mode
        self.load_intrinsics_from_calib = load_intrinsics_from_calib
        self.default_baseline = np.float32(default_baseline)

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load left and right images at time t
        left_image, right_image = load_stereo_pair(
            self.left_image_paths[index],
            self.right_image_paths[index],
            data_format=self.data_format)

        # Load sparse depth (left camera)
        if self.sparse_depth_paths[index] is not None:
            sparse_depth = data_utils.load_depth(
                path=self.sparse_depth_paths[index],
                data_format=self.data_format)
        else:
            _, h, w = left_image.shape
            sparse_depth = np.zeros((1, h, w), dtype=np.float32)

        # Load intrinsics and baseline
        if self.load_intrinsics_from_calib:
            intrinsics, baseline = parse_kitti_calibration(
                self.intrinsics_paths[index], cam_id=2)
        else:
            intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)
            baseline = self.default_baseline

        # Collect all spatial inputs for consistent cropping
        inputs = [left_image, right_image, sparse_depth]

        # Load temporal frames if available
        if self.has_temporal:
            left_image_next, right_image_next = load_stereo_pair(
                self.left_image_next_paths[index],
                self.right_image_next_paths[index],
                data_format=self.data_format)
            inputs.extend([left_image_next, right_image_next])

        # Load ground truth if available
        if self.has_ground_truth:
            ground_truth = data_utils.load_depth(
                path=self.ground_truth_paths[index],
                data_format=self.data_format)
            inputs.append(ground_truth)

        # Apply random crop consistently to ALL spatial inputs and adjust intrinsics
        if self.do_random_crop:
            inputs, [intrinsics] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert all to float32
        inputs = [T.astype(np.float32) for T in inputs]
        intrinsics = intrinsics.astype(np.float32)

        # Unpack inputs
        idx = 0
        left_image = inputs[idx]; idx += 1
        right_image = inputs[idx]; idx += 1
        sparse_depth = inputs[idx]; idx += 1

        # Build output dictionary
        output = {
            'left_image': left_image,
            'right_image': right_image,
            'sparse_depth': sparse_depth,
            'intrinsics': intrinsics,
            'baseline': baseline,
        }

        if self.has_temporal:
            output['left_image_next'] = inputs[idx]; idx += 1
            output['right_image_next'] = inputs[idx]; idx += 1

        if self.has_ground_truth:
            output['ground_truth'] = inputs[idx]; idx += 1

        return output

    def __len__(self):
        return self.n_sample


class StereoDepthCompletionInferenceDataset(torch.utils.data.Dataset):
    '''
    Inference/validation dataset for stereo depth completion.
    No augmentation is applied.

    Fetches:
        (1) Left image at time t
        (2) Right image at time t
        (3) Sparse depth map at time t (left camera)
        (4) Camera intrinsics (3x3)
        (5) Stereo baseline (float scalar)
        (6) Optional: ground truth depth map

    Arg(s):
        left_image_paths : list[str]
            paths to left camera images
        right_image_paths : list[str]
            paths to right camera images
        sparse_depth_paths : list[str]
            paths to sparse depth maps (left camera)
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrices (.npy files)
            or paths to KITTI calib_cam_to_cam.txt files
        ground_truth_paths : list[str] or None
            paths to ground truth depth maps
        load_intrinsics_from_calib : bool
            if True, parse intrinsics from KITTI calib_cam_to_cam.txt files
        default_baseline : float
            default stereo baseline in meters (used when load_intrinsics_from_calib=False)
    '''

    def __init__(self,
                 left_image_paths,
                 right_image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 ground_truth_paths=None,
                 load_intrinsics_from_calib=False,
                 default_baseline=0.54):

        self.n_sample = len(left_image_paths)

        for paths in [right_image_paths, sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.left_image_paths = left_image_paths
        self.right_image_paths = right_image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.has_ground_truth = \
            ground_truth_paths is not None and all([x is not None for x in ground_truth_paths])
        if self.has_ground_truth:
            self.ground_truth_paths = ground_truth_paths

        self.load_intrinsics_from_calib = load_intrinsics_from_calib
        self.default_baseline = np.float32(default_baseline)

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load left and right images
        left_image, right_image = load_stereo_pair(
            self.left_image_paths[index],
            self.right_image_paths[index],
            data_format=self.data_format)

        # Load sparse depth (left camera)
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load intrinsics and baseline
        if self.load_intrinsics_from_calib:
            intrinsics, baseline = parse_kitti_calibration(
                self.intrinsics_paths[index], cam_id=2)
        else:
            intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)
            baseline = self.default_baseline

        output = {
            'left_image': left_image.astype(np.float32),
            'right_image': right_image.astype(np.float32),
            'sparse_depth': sparse_depth.astype(np.float32),
            'intrinsics': intrinsics.astype(np.float32),
            'baseline': baseline,
        }

        # Load ground truth if available
        if self.has_ground_truth:
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
            output['ground_truth'] = ground_truth.astype(np.float32)

        return output

    def __len__(self):
        return self.n_sample


class StereoFromMonocularPaths(torch.utils.data.Dataset):
    '''
    Adapter that derives right image paths from existing monocular path lists.

    For KITTI, if the left image is at .../image_02/data/XXXXX.png,
    the right image is at .../image_03/data/XXXXX.png.
    This allows reusing existing monocular path list files with the stereo pipeline.

    The monocular images_paths are expected to be triplet images (t-1, t, t+1
    concatenated along width). This adapter loads the center frame (time t) as
    the left image and derives the right image path from it.

    Arg(s):
        images_paths : list[str]
            paths to monocular image triplets (left camera / image_02)
        sparse_depth_paths : list[str]
            paths to sparse depth maps at time t
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrices (.npy files)
        ground_truth_paths : list[str] or None
            paths to ground truth depth maps
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
        default_baseline : float
            default stereo baseline in meters
    '''

    def __init__(self,
                 images_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 ground_truth_paths=None,
                 random_crop_shape=None,
                 random_crop_type=None,
                 default_baseline=0.54):

        self.n_sample = len(images_paths)

        for paths in [sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.images_paths = images_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        # Derive right image triplet paths from left by replacing image_02 -> image_03
        self.right_images_paths = [
            derive_right_image_path(p) for p in images_paths
        ]

        self.has_ground_truth = \
            ground_truth_paths is not None and None not in ground_truth_paths
        if self.has_ground_truth:
            assert len(ground_truth_paths) == self.n_sample
        self.ground_truth_paths = ground_truth_paths

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape
        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.default_baseline = np.float32(default_baseline)

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load left image triplet and extract center frame (time t)
        left_triplet = data_utils.load_image(
            self.images_paths[index],
            normalize=False,
            data_format=self.data_format)
        _, left_image, _ = np.split(left_triplet, indices_or_sections=3, axis=-1)

        # Load right image triplet and extract center frame (time t)
        right_triplet = data_utils.load_image(
            self.right_images_paths[index],
            normalize=False,
            data_format=self.data_format)
        _, right_image, _ = np.split(right_triplet, indices_or_sections=3, axis=-1)

        # Load sparse depth (left camera)
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load intrinsics
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)
        baseline = self.default_baseline

        # Collect all spatial inputs for consistent cropping
        inputs = [left_image, right_image, sparse_depth]

        if self.has_ground_truth:
            ground_truth = data_utils.load_depth(
                path=self.ground_truth_paths[index],
                data_format=self.data_format)
            inputs.append(ground_truth)

        # Apply random crop consistently to all spatial inputs and adjust intrinsics
        if self.do_random_crop:
            inputs, [intrinsics] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert to float32
        inputs = [T.astype(np.float32) for T in inputs]
        intrinsics = intrinsics.astype(np.float32)

        # Unpack
        idx = 0
        left_image = inputs[idx]; idx += 1
        right_image = inputs[idx]; idx += 1
        sparse_depth = inputs[idx]; idx += 1

        output = {
            'left_image': left_image,
            'right_image': right_image,
            'sparse_depth': sparse_depth,
            'intrinsics': intrinsics,
            'baseline': baseline,
        }

        if self.has_ground_truth:
            output['ground_truth'] = inputs[idx]; idx += 1

        return output

    def __len__(self):
        return self.n_sample


def build_stereo_datasets_from_unos_file(
        train_file_path,
        kitti_raw_root,
        sparse_depth_paths=None,
        ground_truth_paths=None,
        random_crop_shape=None,
        random_crop_type=None):
    '''
    Build a StereoDepthCompletionTrainingDataset from a UnOS-format training file.

    Each line in the file has format:
        left_t.png right_t.png left_t+1.png right_t+1.png calib_cam_to_cam.txt

    All paths are relative to kitti_raw_root.

    Arg(s):
        train_file_path : str
            path to the training file (e.g. unos_train_4frames.txt)
        kitti_raw_root : str
            root directory for KITTI raw data
        sparse_depth_paths : list[str] or None
            paths to sparse depth maps; if None, no sparse depth is loaded
            (a dummy zero tensor will be produced)
        ground_truth_paths : list[str] or None
            paths to ground truth depth maps
        random_crop_shape : tuple[int] or None
            shape (height, width) to crop inputs
        random_crop_type : list[str] or None
            crop type specification
    Returns:
        StereoDepthCompletionTrainingDataset : training dataset with temporal frames
    '''

    import os

    left_t_paths = []
    right_t_paths = []
    left_next_paths = []
    right_next_paths = []
    calib_paths = []

    with open(train_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            parts = line.split()
            assert len(parts) == 5, \
                'Expected 5 fields per line, got {}: {}'.format(len(parts), line)

            left_t_paths.append(os.path.join(kitti_raw_root, parts[0]))
            right_t_paths.append(os.path.join(kitti_raw_root, parts[1]))
            left_next_paths.append(os.path.join(kitti_raw_root, parts[2]))
            right_next_paths.append(os.path.join(kitti_raw_root, parts[3]))
            calib_paths.append(os.path.join(kitti_raw_root, parts[4]))

    # When no sparse depth is available, pass None paths so the dataset
    # creates zero-filled dummy depth tensors
    if sparse_depth_paths is None:
        sparse_depth_paths = [None] * len(left_t_paths)

    dataset = StereoDepthCompletionTrainingDataset(
        left_image_paths=left_t_paths,
        right_image_paths=right_t_paths,
        sparse_depth_paths=sparse_depth_paths,
        intrinsics_paths=calib_paths,
        left_image_next_paths=left_next_paths,
        right_image_next_paths=right_next_paths,
        ground_truth_paths=ground_truth_paths,
        random_crop_shape=random_crop_shape,
        random_crop_type=random_crop_type,
        load_intrinsics_from_calib=True)

    return dataset
