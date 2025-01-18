import sys, os, glob, cv2, argparse
import numpy as np
import multiprocessing as mp
sys.path.insert(0, os.getcwd())
from utils.src import data_utils


parser = argparse.ArgumentParser()
parser.add_argument('--paths_only', action='store_true')
parser.add_argument('--n_thread',  type=int, default=8)

args = parser.parse_args()


KITTI_ROOT_DIRPATH = os.path.join('data', 'kitti_raw_data')

KITTI_CALIBRATION_FILENAME = 'calib_cam_to_cam.txt'

EIGEN_TRAIN_PATHS_FILE = os.path.join('setup', 'kitti', 'kitti_eigen_train.txt')
EIGEN_VAL_PATHS_FILE = os.path.join('setup', 'kitti', 'kitti_eigen_validation.txt')
EIGEN_TEST_PATHS_FILE = os.path.join('setup', 'kitti', 'kitti_eigen_test.txt')
ZHOU_TRAIN_PATHS_FILE = os.path.join('setup', 'kitti', 'kitti_zhou_train.txt')
STATIC_FRAME_PATHS_FILE = os.path.join('setup', 'kitti', 'kitti_static_frames.txt')

TRAIN_REFS_DIRPATH = os.path.join('training', 'kitti_eigen')
VAL_REFS_DIRPATH = os.path.join('validation', 'kitti_eigen')
TEST_REFS_DIRPATH = os.path.join('testing', 'kitti_eigen')

# Output file paths for training
TRAIN_IMAGES_LEFT_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_images_left.txt')
TRAIN_IMAGES_RIGHT_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_images_right.txt')
TRAIN_IMAGES_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_images.txt')

TRAIN_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_intrinsics_left.txt')
TRAIN_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_intrinsics_right.txt')
TRAIN_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_intrinsics.txt')

TRAIN_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_focal_length_baseline_left.txt')
TRAIN_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_focal_length_baseline_right.txt')
TRAIN_FOCAL_LENGTH_BASELINE_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_focal_length_baseline.txt')

TRAIN_GROUND_TRUTH_LEFT_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_ground_truth_left.txt')
TRAIN_GROUND_TRUTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_ground_truth_right.txt')
TRAIN_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_ground_truth.txt')

# Output file paths for training (nonstatic)
TRAIN_IMAGES_LEFT_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_images_left-nonstatic.txt')
TRAIN_IMAGES_RIGHT_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_images_right-nonstatic.txt')
TRAIN_IMAGES_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_images-nonstatic.txt')

TRAIN_INTRINSICS_LEFT_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_intrinsics_left-nonstatic.txt')
TRAIN_INTRINSICS_RIGHT_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_intrinsics_right-nonstatic.txt')
TRAIN_INTRINSICS_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_intrinsics-nonstatic.txt')

TRAIN_FOCAL_LENGTH_BASELINE_LEFT_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_focal_length_baseline_left-nonstatic.txt')
TRAIN_FOCAL_LENGTH_BASELINE_RIGHT_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_focal_length_baseline_right-nonstatic.txt')
TRAIN_FOCAL_LENGTH_BASELINE_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_focal_length_baseline-nonstatic.txt')

TRAIN_GROUND_TRUTH_LEFT_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_ground_truth_left-nonstatic.txt')
TRAIN_GROUND_TRUTH_RIGHT_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_ground_truth_right-nonstatic.txt')
TRAIN_GROUND_TRUTH_NONSTATIC_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_eigen_train_ground_truth-nonstatic.txt')

# Output file paths for Zhou monocular
ZHOU_IMAGES_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_zhou_train_images.txt')
ZHOU_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_zhou_train_intrinsics.txt')
ZHOU_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, 'kitti_zhou_train_ground_truth.txt')

# Output file paths for validation
VAL_IMAGE_FILEPATH = os.path.join(
    VAL_REFS_DIRPATH, 'kitti_eigen_val_image.txt')
VAL_INTRINSICS_FILEPATH = os.path.join(
    VAL_REFS_DIRPATH, 'kitti_eigen_val_intrinsics.txt')
VAL_FOCAL_LENGTH_BASELINE_FILEPATH = os.path.join(
    VAL_REFS_DIRPATH, 'kitti_eigen_val_focal_length_baseline.txt')
VAL_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REFS_DIRPATH, 'kitti_eigen_val_ground_truth.txt')

# Output file paths for testing
TEST_IMAGE_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, 'kitti_eigen_test_image.txt')
TEST_INTRINSICS_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, 'kitti_eigen_test_intrinsics.txt')
TEST_FOCAL_LENGTH_BASELINE_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, 'kitti_eigen_test_focal_length_baseline.txt')
TEST_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, 'kitti_eigen_test_ground_truth.txt')

# Output directory
OUTPUT_DIRPATH = os.path.join('data', 'kitti_raw_data_eigen')


def process_frame(args):
    '''
    Generate data for a single frame

    Arg(s):
        args : tuple[str]
            image path at time t=0,
            image path at time t=1,
            image path at time t=-1,
            boolean flag if set then create paths only
    Returns:
        str : image triplet output path
        str : ground truth (velodyne) output path
    '''

    image0_path, image1_path, image2_path, paths_only = args

    if not paths_only:
        # Load image at time t
        image0 = cv2.imread(image0_path)
    else:
        image0 = None

    if image1_path is not None and image2_path is not None and not paths_only:
        # Load images at t-1 and t+1
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        images = np.concatenate([image1, image0, image2], axis=1)
    else:
        images = image0

    images_path = image0_path.replace(KITTI_ROOT_DIRPATH, OUTPUT_DIRPATH)

    # Extract useful components from paths
    _, _, date, sequence, camera, _, filename = image0_path.split(os.sep)
    camera_id = np.int32(camera[-1])
    file_id, ext = os.path.splitext(filename)

    velodyne_path = os.path.join(
        KITTI_ROOT_DIRPATH, date, sequence, 'velodyne_points', 'data', file_id + '.bin')

    calibration_dirpath = os.path.join(KITTI_ROOT_DIRPATH, date)

    # Get groundtruth depth from velodyne
    velodyne_path_exists = os.path.exists(velodyne_path)

    if not paths_only:
        if velodyne_path_exists:
            shape = image0.shape[0:2]
            ground_truth = data_utils.velodyne2depth(
                calibration_dirpath,
                velodyne_path,
                shape,
                camera_id=camera_id)
        else:
            print('WARNING: unable to locate velodyne path: {}'.format(velodyne_path))

    # Construct ground truth output path
    ground_truth_path = image0_path \
        .replace('image', 'ground_truth') \
        .replace(KITTI_ROOT_DIRPATH, OUTPUT_DIRPATH)
    ground_truth_path = ground_truth_path[:-3] + 'png'

    if not paths_only:
        os.makedirs(os.path.dirname(images_path), exist_ok=True)
        os.makedirs(os.path.dirname(ground_truth_path), exist_ok=True)

        cv2.imwrite(images_path, images)

        if velodyne_path_exists:
            data_utils.save_depth(ground_truth, ground_truth_path)

    return images_path, ground_truth_path


'''
Create camara intrinsics as numpy
'''
# Build a mapping between the camera intrinsics to the directories
intrinsics_files = sorted(glob.glob(os.path.join(
    KITTI_ROOT_DIRPATH, '*', KITTI_CALIBRATION_FILENAME)))

intrinsics_dkeys = {}
focal_length_baseline_dkeys = {}

for intrinsics_file in intrinsics_files:
    # Example: data/kitti_raw_data_eigen/2011_09_26/intrinsics2.npy
    intrinsics2_path = intrinsics_file \
        .replace(KITTI_ROOT_DIRPATH, os.path.join(OUTPUT_DIRPATH)) \
        .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics2.npy')
    intrinsics3_path = intrinsics_file \
        .replace(KITTI_ROOT_DIRPATH, os.path.join(OUTPUT_DIRPATH)) \
        .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics3.npy')

    # Example: data/kitti_raw_data_eigen/2011_09_26/focal_length_baseline2.npy
    focal_length_baseline2_path = intrinsics_file \
        .replace(KITTI_ROOT_DIRPATH, os.path.join(OUTPUT_DIRPATH)) \
        .replace(KITTI_CALIBRATION_FILENAME, 'focal_length_baseline2.npy')
    focal_length_baseline3_path = intrinsics_file \
        .replace(KITTI_ROOT_DIRPATH, os.path.join(OUTPUT_DIRPATH)) \
        .replace(KITTI_CALIBRATION_FILENAME, 'focal_length_baseline3.npy')

    sequence_dirpath = os.path.split(intrinsics2_path)[0]

    if not os.path.exists(sequence_dirpath):
        os.makedirs(sequence_dirpath)

    calib = data_utils.load_calibration(intrinsics_file)
    intrinsics2 = np.reshape(calib['P_rect_02'], [3, 4])
    intrinsics3 = np.reshape(calib['P_rect_03'], [3, 4])

    # camera2 is left of camera0 (-6cm) camera3 is right of camera2 (+53.27cm)
    b2 = intrinsics2[0, 3] / -intrinsics2[0, 0]
    b3 = intrinsics3[0, 3] / -intrinsics3[0, 0]
    baseline = b3 - b2

    # Focal length of the cameras
    focal_length2 = intrinsics2[0, 0]
    focal_length3 = intrinsics3[0, 0]

    intrinsics2 = intrinsics2[:3, :3].astype(np.float32)
    intrinsics3 = intrinsics3[:3, :3].astype(np.float32)

    focal_length_baseline2 = np.concatenate([
        np.expand_dims(focal_length2, axis=-1),
        np.expand_dims(baseline, axis=-1)], axis=-1)

    focal_length_baseline3 = np.concatenate([
        np.expand_dims(focal_length3, axis=-1),
        np.expand_dims(baseline, axis=-1)], axis=-1)

    # Store as numpy
    if not args.paths_only:
        np.save(intrinsics2_path, intrinsics2)
        np.save(intrinsics3_path, intrinsics3)
        np.save(focal_length_baseline2_path, focal_length_baseline2)
        np.save(focal_length_baseline3_path, focal_length_baseline3)

    # Add as keys to instrinsics dictionary
    sequence_date = intrinsics_file.split(os.sep)[2]
    intrinsics_dkeys[(sequence_date, 'image_02')] = intrinsics2_path
    intrinsics_dkeys[(sequence_date, 'image_03')] = intrinsics3_path
    focal_length_baseline_dkeys[(sequence_date, 'image_02')] = focal_length_baseline2_path
    focal_length_baseline_dkeys[(sequence_date, 'image_03')] = focal_length_baseline3_path


'''
Split file paths into training, validation and testing
'''
eigen_train_paths = data_utils.read_paths(EIGEN_TRAIN_PATHS_FILE)
eigen_val_paths = data_utils.read_paths(EIGEN_VAL_PATHS_FILE)
eigen_test_paths = data_utils.read_paths(EIGEN_TEST_PATHS_FILE)
zhou_train_paths = data_utils.read_paths(ZHOU_TRAIN_PATHS_FILE)

for dirpath in [TRAIN_REFS_DIRPATH, VAL_REFS_DIRPATH, TEST_REFS_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

eigen_train_image_left_paths = []
eigen_train_image_right_paths = []

eigen_val_image_left_paths = []
eigen_val_image_right_paths = []

eigen_test_image_left_paths = []
eigen_test_image_right_paths = []

zhou_train_image_left_paths = []
zhou_train_image_right_paths = []

data_paths = [
    [
        'training',
        eigen_train_paths,
        eigen_train_image_left_paths,
        eigen_train_image_right_paths
    ], [
        'validation',
        eigen_val_paths,
        eigen_val_image_left_paths,
        eigen_val_image_right_paths
    ], [
        'testing',
        eigen_test_paths,
        eigen_test_image_left_paths,
        eigen_test_image_right_paths
    ], [
        'training',
        zhou_train_paths,
        zhou_train_image_left_paths,
        zhou_train_image_right_paths
    ]
]

'''
Load eigen split paths
'''
for paths in data_paths:

    # Unpack data paths
    data_split, \
        image_paths, \
        image_left_paths, \
        image_right_paths = paths

    # Split each line into left and right image paths
    for idx in range(len(image_paths)):

        print('Reading {}/{} {} image file paths...'.format(idx + 1, len(image_paths), data_split), end='\r')
        image_left_path, image_right_path = image_paths[idx].split()
        image_left_paths.append(os.path.join(KITTI_ROOT_DIRPATH, image_left_path)[:-3] + 'png')
        image_right_paths.append(os.path.join(KITTI_ROOT_DIRPATH, image_right_path)[:-3] + 'png')

    print('Completed reading {}/{} {} image file paths'.format(
        idx + 1,
        len(image_paths),
        data_split),
        end='\r')

# Join left and right paths for monocular full set
eigen_train_image_paths = eigen_train_image_left_paths + eigen_train_image_right_paths
eigen_val_image_paths = eigen_val_image_left_paths + eigen_val_image_right_paths
eigen_test_image_paths = eigen_test_image_left_paths + eigen_test_image_right_paths
zhou_train_image_paths = zhou_train_image_left_paths + zhou_train_image_right_paths

eigen_train_dirpaths = set([
    path.split(os.sep)[-4] for path in eigen_train_image_paths
])
eigen_val_dirpaths = set([
    path.split(os.sep)[-4] for path in eigen_val_image_paths
])
eigen_test_dirpaths = set([
    path.split(os.sep)[-4] for path in eigen_test_image_paths
])
zhou_train_dirpaths = set([
    path.split(os.sep)[-4] for path in zhou_train_image_paths
])

'''
Create image triplets for structure from motion
'''
images_output_paths = []
ground_truth_output_paths = []

# Example: data/kitti_raw_data/2011_09_26/
sequence_date_dirpaths = glob.glob(os.path.join(KITTI_ROOT_DIRPATH, '*/'))

for sequence_date_dirpath in sequence_date_dirpaths:

    # Example: data/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/
    sequence_dirpaths = glob.glob(os.path.join(sequence_date_dirpath, '*/'))

    for sequence_dirpath in sequence_dirpaths:

        sequence = sequence_dirpath.split(os.sep)[-2]

        # Example: data/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data
        for camera_dirpath in ['image_02', 'image_03']:
            sequence_image_paths = sorted(glob.glob(
                os.path.join(sequence_dirpath, camera_dirpath, 'data', '*.png')))

            is_train = False
            is_val_test = False

            if sequence in eigen_train_dirpaths or sequence in zhou_train_dirpaths:
                is_train = True
                indices = range(1, len(sequence_image_paths) - 1)
            elif sequence in eigen_val_dirpaths or sequence in eigen_test_dirpaths:
                is_val_test = True
                indices = range(len(sequence_image_paths))
            else:
                is_train = True
                indices = range(1, len(sequence_image_paths) - 1)

            pool_inputs = []

            for idx in indices:
                # Concatenate image at time t-1, t, and t+1
                image0_path = sequence_image_paths[idx]

                if is_train:
                    image1_path = sequence_image_paths[idx-1]
                    image2_path = sequence_image_paths[idx+1]
                elif is_val_test:
                    image1_path = None
                    image2_path = None

                pool_inputs.append((image0_path, image1_path, image2_path, args.paths_only))

            print('Processing {} {} file paths for {}'.format(
                idx + 1, camera_dirpath, sequence_dirpath))

            with mp.Pool(args.n_thread) as pool:
                pool_results = pool.map(process_frame, pool_inputs)

                for result in pool_results:
                    images_path, ground_truth_path = result

                    images_output_paths.append(images_path)
                    ground_truth_output_paths.append(ground_truth_path)

            print('Completed processing {} {} file path paths for {}'.format(
                idx + 1, camera_dirpath, sequence_dirpath))

'''
Load the list of static paths
'''
kitti_static_frames_paths = data_utils.read_paths(STATIC_FRAME_PATHS_FILE)

kitti_static_frames_parts = []
for path in kitti_static_frames_paths:
    parts = path.split(' ')
    kitti_static_frames_parts.append((parts[1], parts[2]))

'''
Split the paths into training, training (nonstatic), validation and testing
'''
# Training
train_images_left_output_paths = []
train_images_right_output_paths = []
train_images_output_paths = []

train_intrinsics_left_output_paths = []
train_intrinsics_right_output_paths = []
train_intrinsics_output_paths = []

train_focal_length_baseline_left_output_paths = []
train_focal_length_baseline_right_output_paths = []
train_focal_length_baseline_output_paths = []

train_ground_truth_left_output_paths = []
train_ground_truth_right_output_paths = []
train_ground_truth_output_paths = []

# Training (nonstatic)
train_images_left_nonstatic_output_paths = []
train_images_right_nonstatic_output_paths = []
train_images_nonstatic_output_paths = []

train_intrinsics_left_nonstatic_output_paths = []
train_intrinsics_right_nonstatic_output_paths = []
train_intrinsics_nonstatic_output_paths = []

train_focal_length_baseline_left_nonstatic_output_paths = []
train_focal_length_baseline_right_nonstatic_output_paths = []
train_focal_length_baseline_nonstatic_output_paths = []

train_ground_truth_left_nonstatic_output_paths = []
train_ground_truth_right_nonstatic_output_paths = []
train_ground_truth_nonstatic_output_paths = []

# Training (Zhou)
zhou_images_output_paths = []
zhou_intrinsics_output_paths = []
zhou_ground_truth_output_paths = []

# Validation
val_image_output_paths = []
val_intrinsics_output_paths = []
val_focal_length_baseline_output_paths = []
val_ground_truth_output_paths = []

# Testing
test_image_output_paths = []
test_intrinsics_output_paths = []
test_focal_length_baseline_output_paths = []
test_ground_truth_output_paths = []

for images_output_path, ground_truth_output_path in zip(images_output_paths, ground_truth_output_paths):
    # Get key for camera intrinsics
    _, _, date, _, camera, _, _ = images_path.split(os.sep)

    intrinsics_output_path = intrinsics_dkeys[(date, camera)]
    focal_length_baseline_output_path = focal_length_baseline_dkeys[(date, camera)]

    image_path = images_output_path.replace(OUTPUT_DIRPATH, KITTI_ROOT_DIRPATH)

    if image_path in zhou_train_image_paths:
        # Add to all training (Zhou)
        zhou_images_output_paths.append(images_output_path)
        zhou_intrinsics_output_paths.append(intrinsics_output_path)
        zhou_ground_truth_output_paths.append(ground_truth_output_path)

    if image_path in eigen_train_image_paths:

        # Check if image path for right or left camera
        is_left_camera = False
        is_right_camera = False

        if image_path in eigen_train_image_left_paths:
            is_left_camera = True
        elif image_path in eigen_train_image_right_paths:
            is_right_camera = True
        else:
            raise ValueError('Eigen train left and right paths are inconsistent')

        # Add to all training paths
        train_images_output_paths.append(images_output_path)
        train_intrinsics_output_paths.append(intrinsics_output_path)
        train_focal_length_baseline_output_paths.append(focal_length_baseline_output_path)
        train_ground_truth_output_paths.append(ground_truth_output_path)

        # Add to left and right camera paths
        if is_left_camera:
            train_images_left_output_paths.append(images_output_path)
            train_intrinsics_left_output_paths.append(intrinsics_output_path)
            train_focal_length_baseline_left_output_paths.append(focal_length_baseline_output_path)
            train_ground_truth_left_output_paths.append(ground_truth_output_path)

        if is_right_camera:
            train_images_right_output_paths.append(images_output_path)
            train_intrinsics_right_output_paths.append(intrinsics_output_path)
            train_focal_length_baseline_right_output_paths.append(focal_length_baseline_output_path)
            train_ground_truth_right_output_paths.append(ground_truth_output_path)

        # Check if path is for nonstatic camera
        is_static = False

        for parts in kitti_static_frames_parts:
            if parts[0] in image_path and parts[1] in image_path:
                is_static = True
                break

        if not is_static:
            # Add to nonstatic training paths
            train_images_nonstatic_output_paths.append(images_output_path)
            train_intrinsics_nonstatic_output_paths.append(intrinsics_output_path)
            train_focal_length_baseline_nonstatic_output_paths.append(focal_length_baseline_output_path)
            train_ground_truth_nonstatic_output_paths.append(ground_truth_output_path)

            # Check if image path for right or left camera
            if is_left_camera:
                train_images_left_nonstatic_output_paths.append(images_output_path)
                train_intrinsics_left_nonstatic_output_paths.append(intrinsics_output_path)
                train_focal_length_baseline_left_nonstatic_output_paths.append(focal_length_baseline_output_path)
                train_ground_truth_left_nonstatic_output_paths.append(ground_truth_output_path)

            if is_right_camera:
                train_images_right_nonstatic_output_paths.append(images_output_path)
                train_intrinsics_right_nonstatic_output_paths.append(intrinsics_output_path)
                train_focal_length_baseline_right_nonstatic_output_paths.append(focal_length_baseline_output_path)
                train_ground_truth_right_nonstatic_output_paths.append(ground_truth_output_path)

    elif image_path in eigen_val_image_left_paths:
        val_image_output_paths.append(images_output_path)
        val_intrinsics_output_paths.append(intrinsics_output_path)
        val_focal_length_baseline_output_paths.append(focal_length_baseline_output_path)
        val_ground_truth_output_paths.append(ground_truth_output_path)

    elif image_path in eigen_test_image_left_paths:
        test_image_output_paths.append(images_output_path)
        test_intrinsics_output_paths.append(intrinsics_output_path)
        test_focal_length_baseline_output_paths.append(focal_length_baseline_output_path)
        test_ground_truth_output_paths.append(ground_truth_output_path)

'''
Write training paths to file
'''
# Training paths for left camera
print('Storing {} left training image triplet file paths into: {}'.format(
    len(train_images_left_output_paths),
    TRAIN_IMAGES_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGES_LEFT_FILEPATH,
    train_images_left_output_paths)

print('Storing {} left training camera intrinsics file paths into: {}'.format(
    len(train_intrinsics_left_output_paths),
    TRAIN_INTRINSICS_LEFT_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_LEFT_FILEPATH,
    train_intrinsics_left_output_paths)

print('Storing {} left training focal length and baseline file paths into: {}'.format(
    len(train_focal_length_baseline_left_output_paths),
    TRAIN_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH))
data_utils.write_paths(
    TRAIN_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH,
    train_focal_length_baseline_left_output_paths)

print('Storing {} left training ground truth file paths into: {}'.format(
    len(train_ground_truth_left_output_paths),
    TRAIN_GROUND_TRUTH_LEFT_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_LEFT_FILEPATH,
    train_ground_truth_left_output_paths)

# Training paths for right camera
print('Storing {} right training image triplet file paths into: {}'.format(
    len(train_images_right_output_paths),
    TRAIN_IMAGES_RIGHT_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGES_RIGHT_FILEPATH,
    train_images_right_output_paths)

print('Storing {} right training camera intrinsics file paths into: {}'.format(
    len(train_intrinsics_right_output_paths),
    TRAIN_INTRINSICS_RIGHT_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_RIGHT_FILEPATH,
    train_intrinsics_right_output_paths)

print('Storing {} right training focal length and baseline file paths into: {}'.format(
    len(train_focal_length_baseline_right_output_paths),
    TRAIN_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH))
data_utils.write_paths(
    TRAIN_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH,
    train_focal_length_baseline_right_output_paths)

print('Storing {} right training ground truth file paths into: {}'.format(
    len(train_ground_truth_right_output_paths),
    TRAIN_GROUND_TRUTH_RIGHT_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_RIGHT_FILEPATH,
    train_ground_truth_right_output_paths)

# Training paths for both cameras
print('Storing {} training image triplet file paths into: {}'.format(
    len(train_images_output_paths),
    TRAIN_IMAGES_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGES_FILEPATH,
    train_images_output_paths)

print('Storing {} training camera intrinsics file paths into: {}'.format(
    len(train_intrinsics_output_paths),
    TRAIN_INTRINSICS_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_FILEPATH,
    train_intrinsics_output_paths)

print('Storing {} training focal length and baseline file paths into: {}'.format(
    len(train_focal_length_baseline_output_paths),
    TRAIN_FOCAL_LENGTH_BASELINE_FILEPATH))
data_utils.write_paths(
    TRAIN_FOCAL_LENGTH_BASELINE_FILEPATH,
    train_focal_length_baseline_output_paths)

print('Storing {} training ground truth file paths into: {}'.format(
    len(train_ground_truth_output_paths),
    TRAIN_GROUND_TRUTH_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_FILEPATH,
    train_ground_truth_output_paths)

'''
Write nonstatic training paths to file
'''
# Training paths for nonstatic left camera
print('Storing {} nonstatic left training image triplet file paths into: {}'.format(
    len(train_images_left_nonstatic_output_paths),
    TRAIN_IMAGES_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGES_LEFT_NONSTATIC_FILEPATH,
    train_images_left_nonstatic_output_paths)

print('Storing {} nonstatic left training camera intrinsics file paths into: {}'.format(
    len(train_intrinsics_left_nonstatic_output_paths),
    TRAIN_INTRINSICS_LEFT_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_LEFT_NONSTATIC_FILEPATH,
    train_intrinsics_left_nonstatic_output_paths)

print('Storing {} nonstatic left training focal length and baseline file paths into: {}'.format(
    len(train_focal_length_baseline_left_nonstatic_output_paths),
    TRAIN_FOCAL_LENGTH_BASELINE_LEFT_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_FOCAL_LENGTH_BASELINE_LEFT_NONSTATIC_FILEPATH,
    train_focal_length_baseline_left_nonstatic_output_paths)

print('Storing {} nonstatic left training ground truth file paths into: {}'.format(
    len(train_ground_truth_left_nonstatic_output_paths),
    TRAIN_GROUND_TRUTH_LEFT_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_LEFT_NONSTATIC_FILEPATH,
    train_ground_truth_left_nonstatic_output_paths)

# Training paths for nonstatic right camera
print('Storing {} nonstatic right training image triplet file paths into: {}'.format(
    len(train_images_right_nonstatic_output_paths),
    TRAIN_IMAGES_RIGHT_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGES_RIGHT_NONSTATIC_FILEPATH,
    train_images_right_nonstatic_output_paths)

print('Storing {} nonstatic right training camera intrinsics file paths into: {}'.format(
    len(train_intrinsics_right_nonstatic_output_paths),
    TRAIN_INTRINSICS_RIGHT_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_RIGHT_NONSTATIC_FILEPATH,
    train_intrinsics_right_nonstatic_output_paths)

print('Storing {} nonstatic right training focal length and baseline file paths into: {}'.format(
    len(train_focal_length_baseline_right_nonstatic_output_paths),
    TRAIN_FOCAL_LENGTH_BASELINE_RIGHT_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_FOCAL_LENGTH_BASELINE_RIGHT_NONSTATIC_FILEPATH,
    train_focal_length_baseline_right_nonstatic_output_paths)

print('Storing {} nonstatic right training ground truth file paths into: {}'.format(
    len(train_ground_truth_right_nonstatic_output_paths),
    TRAIN_GROUND_TRUTH_RIGHT_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_RIGHT_NONSTATIC_FILEPATH,
    train_ground_truth_right_nonstatic_output_paths)

# Training paths for both nonstatic cameras
print('Storing {} nonstatic training image triplet file paths into: {}'.format(
    len(train_images_nonstatic_output_paths),
    TRAIN_IMAGES_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_IMAGES_NONSTATIC_FILEPATH,
    train_images_nonstatic_output_paths)

print('Storing {} nonstatic training camera intrinsics file paths into: {}'.format(
    len(train_intrinsics_nonstatic_output_paths),
    TRAIN_INTRINSICS_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_INTRINSICS_NONSTATIC_FILEPATH,
    train_intrinsics_nonstatic_output_paths)

print('Storing {} nonstatic training focal length and baseline file paths into: {}'.format(
    len(train_focal_length_baseline_nonstatic_output_paths),
    TRAIN_FOCAL_LENGTH_BASELINE_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_FOCAL_LENGTH_BASELINE_NONSTATIC_FILEPATH,
    train_focal_length_baseline_nonstatic_output_paths)

print('Storing {} nonstatic training ground truth file paths into: {}'.format(
    len(train_ground_truth_nonstatic_output_paths),
    TRAIN_GROUND_TRUTH_NONSTATIC_FILEPATH))
data_utils.write_paths(
    TRAIN_GROUND_TRUTH_NONSTATIC_FILEPATH,
    train_ground_truth_nonstatic_output_paths)

'''
Write training (Zhou) paths to file
'''
print('Storing {} training (Zhou) image file paths into: {}'.format(
    len(zhou_images_output_paths),
    ZHOU_IMAGES_FILEPATH))
data_utils.write_paths(
    ZHOU_IMAGES_FILEPATH,
    zhou_images_output_paths)

print('Storing {} training (Zhou) camera intrinsics file paths into: {}'.format(
    len(zhou_intrinsics_output_paths),
    ZHOU_INTRINSICS_FILEPATH))
data_utils.write_paths(
    ZHOU_INTRINSICS_FILEPATH,
    zhou_intrinsics_output_paths)

print('Storing {} training (Zhou) ground truth file paths into: {}'.format(
    len(zhou_ground_truth_output_paths),
    ZHOU_GROUND_TRUTH_FILEPATH))
data_utils.write_paths(
    ZHOU_GROUND_TRUTH_FILEPATH,
    zhou_ground_truth_output_paths)

'''
Write validation paths to file
'''
print('Storing {} validation image file paths into: {}'.format(
    len(val_image_output_paths),
    VAL_IMAGE_FILEPATH))
data_utils.write_paths(
    VAL_IMAGE_FILEPATH,
    val_image_output_paths)

print('Storing {} validation camera intrinsics file paths into: {}'.format(
    len(val_intrinsics_output_paths),
    VAL_INTRINSICS_FILEPATH))
data_utils.write_paths(
    VAL_INTRINSICS_FILEPATH,
    val_intrinsics_output_paths)

print('Storing {} validation focal length and baseline file paths into: {}'.format(
    len(val_focal_length_baseline_output_paths),
    VAL_FOCAL_LENGTH_BASELINE_FILEPATH))
data_utils.write_paths(
    VAL_FOCAL_LENGTH_BASELINE_FILEPATH,
    val_focal_length_baseline_output_paths)

print('Storing {} validation ground truth file paths into: {}'.format(
    len(val_ground_truth_output_paths),
    VAL_GROUND_TRUTH_FILEPATH))
data_utils.write_paths(
    VAL_GROUND_TRUTH_FILEPATH,
    val_ground_truth_output_paths)

'''
Write testing paths to file
'''
print('Storing {} testing image file paths into: {}'.format(
    len(test_image_output_paths),
    TEST_IMAGE_FILEPATH))
data_utils.write_paths(
    TEST_IMAGE_FILEPATH,
    test_image_output_paths)

print('Storing {} testing camera intrinsics file paths into: {}'.format(
    len(test_intrinsics_output_paths),
    TEST_INTRINSICS_FILEPATH))
data_utils.write_paths(
    TEST_INTRINSICS_FILEPATH,
    test_intrinsics_output_paths)

print('Storing {} testing focal length and baseline file paths into: {}'.format(
    len(test_focal_length_baseline_output_paths),
    TEST_FOCAL_LENGTH_BASELINE_FILEPATH))
data_utils.write_paths(
    TEST_FOCAL_LENGTH_BASELINE_FILEPATH,
    test_focal_length_baseline_output_paths)

print('Storing {} testing ground truth file paths into: {}'.format(
    len(test_ground_truth_output_paths),
    TEST_GROUND_TRUTH_FILEPATH))
data_utils.write_paths(
    TEST_GROUND_TRUTH_FILEPATH,
    test_ground_truth_output_paths)
