import torch
import numpy as np
from utils.src import data_utils


def load_image_triplet(path, normalize=True, data_format='CHW'):
    '''
    Load in triplet frames from path

    Arg(s):
        path : str
            path to image triplet
        normalize : bool
            if set, normalize to [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : image at t-1
        numpy[float32] : image at t
        numpy[float32] : image at t+1
    '''

    # Load image triplet and split into images at t-1, t, t+1
    images = data_utils.load_image(
        path,
        normalize=normalize,
        data_format=data_format)

    # Split along width
    image1, image0, image2 = np.split(images, indices_or_sections=3, axis=-1)

    return image1, image0, image2

# new code begin
def load_kitti_calib_intrinsics(calib_path):
    '''
    Load 3x3 intrinsics from KITTI calib_cam_to_cam.txt (UnOS-style).
    Uses the last line of the file, 12 values, reshape to 3x4 and takes 3x3.
    '''
    with open(calib_path) as f:
        lines = f.readlines()
    last_line = lines[-1].strip().split()
    # values after the key (e.g. "P_rect_02:")
    values = np.array([float(x) for x in last_line[1:13]], dtype=np.float32)
    P = values.reshape(3, 4)
    return P[0:3, 0:3].astype(np.float32)
# new code end

def load_depth(depth_path, data_format='CHW'):
    '''
    Load depth

    Arg(s):
        depth_path : str
            path to depth map
        data_format : str
            'CHW', or 'HWC'
    Return:
        numpy[float32] : depth map (1 x H x W)
    '''

    return data_utils.load_depth(depth_path, data_format=data_format)

def load_depth_with_validity_map(depth_path, data_format='CHW'):
    '''
    Load depth and validity maps

    Arg(s):
        depth_path : str
            path to depth map
        data_format : str
            'CHW', or 'HWC'
        Returns:
            numpy[float32] : depth and validity map (2 x H x W)
    '''

    depth_map, validity_map = data_utils.load_depth_with_validity_map(
        depth_path,
        data_format=data_format)

    if data_format == 'CHW':
        return np.concatenate([depth_map, validity_map], axis=0)
    elif data_format == 'HWC':
        return np.concatenate([depth_map, validity_map], axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width+1)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height+1)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width

    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    # Adjust intrinsics
    if intrinsics is not None:
        offset_principal_point = [[0.0, 0.0, -x_start],
                                  [0.0, 0.0, -y_start],
                                  [0.0, 0.0, 0.0     ]]

        intrinsics = [
            in_ + offset_principal_point for in_ in intrinsics
        ]

        return outputs, intrinsics
    else:
        return outputs


class DepthCompletionMonocularTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) camera image at t
        (2) camera image at t-1
        (3) camera image at t+1
        (4) sparse depth map at t
        (5) intrinsic camera calibration matrix

    Arg(s):
        images_paths : list[str]
            paths to camera images t-1, t, t+1
        sparse_depth_paths : list[str]
            paths to sparse depth maps at t
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 images_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 random_crop_shape=None,
                 random_crop_type=None):

        self.n_sample = len(images_paths)

        input_paths = [
            sparse_depth_paths,
            intrinsics_paths
        ]

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.images_paths = images_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image1, image0, image2 = load_image_triplet(
            self.images_paths[index],
            normalize=False)

        # Load sparse depth
        sparse_depth0 = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image0, image1, image2, sparse_depth0], [intrinsics] = random_crop(
                inputs=[image0, image1, image2, sparse_depth0],
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert to float32
        image0, image1, image2, sparse_depth0, intrinsics = [
            T.astype(np.float32)
            for T in [image0, image1, image2, sparse_depth0, intrinsics]
        ]

        return image0, image1, image2, sparse_depth0, intrinsics

    def __len__(self):
        return self.n_sample

# new code begin
class DepthCompletionStereoTrainingDataset(torch.utils.data.Dataset):
    '''
    UnOS-style stereo dataset: left + right paths and calib path per sample.
    Returns (stereo_6ch, stereo_6ch, stereo_6ch, sparse_dummy, intrinsics) so the
    depth_completion training loop receives 6-channel input for PWCModel/UnOS.
    '''

    def __init__(self,
                 images_paths,
                 right_images_paths,
                 intrinsics_paths,
                 random_crop_shape=None,
                 random_crop_type=None,
                 data_format='CHW'):

        self.n_sample = len(images_paths)
        assert len(right_images_paths) == self.n_sample
        assert len(intrinsics_paths) == self.n_sample

        self.images_paths = images_paths
        self.right_images_paths = right_images_paths
        self.intrinsics_paths = intrinsics_paths
        self.random_crop_shape = random_crop_shape
        self.random_crop_type = random_crop_type
        self.data_format = data_format
        self.do_random_crop = (
            random_crop_shape is not None
            and all(x > 0 for x in random_crop_shape)
        )

    def __getitem__(self, index):
        left = data_utils.load_image(
            self.images_paths[index],
            normalize=False,
            data_format=self.data_format)
        right = data_utils.load_image(
            self.right_images_paths[index],
            normalize=False,
            data_format=self.data_format)
        # left/right: CHW
        stereo_6ch = np.concatenate([left, right], axis=0)

        if self.intrinsics_paths[index].endswith('.txt'):
            intrinsics = load_kitti_calib_intrinsics(self.intrinsics_paths[index])
        else:
            intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        if self.do_random_crop:
            n_height, n_width = self.random_crop_shape
            _, o_height, o_width = stereo_6ch.shape
            d_height = o_height - n_height
            d_width = o_width - n_width
            y_start = d_height // 2
            x_start = d_width // 2
            if 'horizontal' in (self.random_crop_type or []):
                x_start = np.random.randint(0, d_width + 1)
            if 'bottom' in (self.random_crop_type or []):
                y_start = d_height
            stereo_6ch = stereo_6ch[:, y_start:y_start + n_height, x_start:x_start + n_width]
            intrinsics = intrinsics + np.array(
                [[0, 0, -x_start], [0, 0, -y_start], [0, 0, 0]], dtype=np.float32)

        sparse_dummy = np.zeros((1, stereo_6ch.shape[1], stereo_6ch.shape[2]), dtype=np.float32)
        stereo_6ch = stereo_6ch.astype(np.float32)
        return stereo_6ch, stereo_6ch.copy(), stereo_6ch.copy(), sparse_dummy, intrinsics.astype(np.float32)

    def __len__(self):
        return self.n_sample
# new code end

class DepthCompletionSupervisedTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) camera image at t
        (2) sparse depth map at t
        (3) intrinsic camera calibration matrix
        (4) ground truth depth map at t

    Arg(s):
        image_paths : list[str]
            paths to camera images t-1, t, t+1
        sparse_depth_paths : list[str]
            paths to sparse depth maps at t
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        ground_truth_paths : list[str]
            paths to ground truth depth maps at time t
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 ground_truth_paths,
                 random_crop_shape=None,
                 random_crop_type=None,
                 load_image_triplet=True):

        self.n_sample = len(image_paths)

        assert ground_truth_paths is not None and None not in ground_truth_paths

        input_paths = [
            sparse_depth_paths,
            intrinsics_paths,
            ground_truth_paths
        ]

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths
        self.ground_truth_paths = ground_truth_paths

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.load_image_triplet = load_image_triplet

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
            path=self.image_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        if self.intrinsics_paths[index] is not None:
            intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)
        else:
            intrinsics = np.eye(3)

        # Load ground truth
        ground_truth = data_utils.load_depth(
            path=self.ground_truth_paths[index],
            data_format=self.data_format)

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image, sparse_depth, ground_truth], [intrinsics] = random_crop(
                inputs=[image, sparse_depth, ground_truth],
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Convert to float32
        image, sparse_depth, intrinsics, ground_truth = [
            T.astype(np.float32)
            for T in [image, sparse_depth, intrinsics, ground_truth]
        ]

        return image, sparse_depth, intrinsics, ground_truth

    def __len__(self):
        return self.n_sample


class DepthCompletionInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) intrinsic camera calibration matrix

    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        ground_truth_paths : list[str]
            paths to ground truth depth maps
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 ground_truth_paths=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        for paths in [sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.is_available_ground_truth = \
           ground_truth_paths is not None and all([x is not None for x in ground_truth_paths])

        if self.is_available_ground_truth:
            self.ground_truth_paths = ground_truth_paths

        self.data_format = 'CHW'
        self.load_image_triplets = load_image_triplets

    def __getitem__(self, index):

        # Load image
        if self.load_image_triplets:
            _, image, _ = load_image_triplet(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)
        else:
            image = data_utils.load_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index])

        inputs = [
            image,
            sparse_depth,
            intrinsics
        ]

        # Load ground truth if available
        if self.is_available_ground_truth:
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
            inputs.append(ground_truth)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        # Return image, sparse_depth, intrinsics, and if available, ground_truth
        return inputs

    def __len__(self):
        return self.n_sample
