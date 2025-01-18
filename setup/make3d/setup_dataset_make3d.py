import sys, os, cv2, glob, argparse
import scipy.io as sio

sys.path.insert(0, './')
from utils.src import data_utils


'''
Paths for Make3d dataset
'''
MAKE3D_ROOT_DIRPATH = os.path.join('data', 'make3d')
MAKE3D_LASER_DIRPATH = os.path.join(MAKE3D_ROOT_DIRPATH, 'laser_data')
MAKE3D_IMAGE_DIRPATH = os.path.join(MAKE3D_ROOT_DIRPATH, 'images')

'''
Output paths
'''
MAKE3D_DERIVED_DIRPATH = os.path.join('data', 'make3d_derived')

TEST_REF_DIRPATH = os.path.join('testing', 'make3d')

# Paths to files for testing
TEST_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'make3d_test_image.txt')
TEST_GROUND_TRUTH_SMALL_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'make3d_test_ground_truth_small.txt')
TEST_GROUND_TRUTH_LARGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'make3d_test_ground_truth_large.txt')

'''
Helper functions
'''
def load_depth(path):
    '''
    Loads depth from Make3d .mat files

    Arg(s):
        path : str
            path to .mat file from laser scanner
    Returns:
        numpy[float32] : depth map
    '''

    mat = sio.loadmat(path)['Position3DGrid']
    return mat[..., 3]


def setup_dataset_make3d_testing(paths_only):
    '''
    Fetch image and ground truth paths for testing

    Arg(s):
        paths_only : bool
            if set, then only produces paths
    '''

    test_image_paths = []
    test_ground_truth_large_paths = []
    test_ground_truth_small_paths = []

    # Create output directories
    image_output_dirpath = MAKE3D_IMAGE_DIRPATH \
        .replace(MAKE3D_ROOT_DIRPATH, MAKE3D_DERIVED_DIRPATH)
    ground_truth_large_output_dirpath = MAKE3D_LASER_DIRPATH \
        .replace(MAKE3D_ROOT_DIRPATH, MAKE3D_DERIVED_DIRPATH) \
        .replace('laser_data', 'laser_data_large')
    ground_truth_small_output_dirpath = MAKE3D_LASER_DIRPATH \
        .replace(MAKE3D_ROOT_DIRPATH, MAKE3D_DERIVED_DIRPATH) \
        .replace('laser_data', 'laser_data_small')

    os.makedirs(image_output_dirpath, exist_ok=True)
    os.makedirs(ground_truth_large_output_dirpath, exist_ok=True)
    os.makedirs(ground_truth_small_output_dirpath, exist_ok=True)

    # Grab all image and laser (ground truth) data
    image_paths = sorted(glob.glob(os.path.join(MAKE3D_IMAGE_DIRPATH, '*.jpg')))
    ground_truth_paths = sorted(glob.glob(os.path.join(MAKE3D_LASER_DIRPATH, '*.mat')))

    n_sample = len(image_paths)

    assert n_sample == len(ground_truth_paths), \
        'Number of images ({}) does not match the number of laser scans ({})'.format(
            n_sample, len(ground_truth_paths))

    for image_path, ground_truth_path in zip(image_paths, ground_truth_paths):

        # Crop image to match laser scanner
        # Based on https://github.com/nianticlabs/monodepth2/issues/89
        image = cv2.imread(image_path)
        image_center_y = image.shape[0] // 2
        image_width_y = image.shape[1] // 4

        image_crop_start_y = image_center_y - image_width_y
        image_crop_end_y = image_center_y + image_width_y

        image_crop = image[image_crop_start_y:image_crop_end_y, :, :]

        image_output_path = os.path.join(
            image_output_dirpath,
            os.path.basename(image_path))

        if not paths_only:
            cv2.imwrite(image_output_path, image_crop)

        # Create ground truth
        ground_truth = load_depth(ground_truth_path)

        # Create large ground truth matching image size
        ground_truth_large = cv2.resize(
            ground_truth,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST)

        ground_truth_center_y = image_center_y
        ground_truth_width_y = image.shape[1] // 3.93

        ground_truth_large_crop_start_y = int(ground_truth_center_y - ground_truth_width_y)
        ground_truth_large_crop_end_y = int(ground_truth_center_y + ground_truth_width_y)

        ground_truth_large_crop = ground_truth_large[ground_truth_large_crop_start_y:ground_truth_large_crop_end_y, :]

        # Create small ground truth for evaluation (the large and small are almost the same)
        ground_truth_small_crop_start_y = int((55 - 21) // 2)
        ground_truth_small_crop_end_y = int((55 + 21) // 2)

        ground_truth_small_crop = ground_truth[ground_truth_small_crop_start_y:ground_truth_small_crop_end_y, :]

        ground_truth_large_output_path = os.path.join(
            ground_truth_large_output_dirpath,
            os.path.splitext(os.path.basename(ground_truth_path))[0] + '.png')

        ground_truth_small_output_path = os.path.join(
            ground_truth_small_output_dirpath,
            os.path.splitext(os.path.basename(ground_truth_path))[0] + '.png')

        if not paths_only:
            data_utils.save_depth(ground_truth_large_crop, ground_truth_large_output_path, multiplier=256.0)
            data_utils.save_depth(ground_truth_small_crop, ground_truth_small_output_path, multiplier=256.0)

        # Store paths
        test_image_paths.append(image_output_path)
        test_ground_truth_large_paths.append(ground_truth_large_output_path)
        test_ground_truth_small_paths.append(ground_truth_small_output_path)

    print('Storing {} image file paths into: {}'.format(
        len(test_image_paths), TEST_IMAGE_FILEPATH))
    data_utils.write_paths(TEST_IMAGE_FILEPATH, image_paths)

    print('Storing {} large ground truth file paths into: {}'.format(
        len(test_ground_truth_large_paths), TEST_GROUND_TRUTH_LARGE_FILEPATH))
    data_utils.write_paths(TEST_GROUND_TRUTH_LARGE_FILEPATH, test_ground_truth_large_paths)

    print('Storing {} small ground truth file paths into: {}'.format(
        len(test_ground_truth_small_paths), TEST_GROUND_TRUTH_SMALL_FILEPATH))
    data_utils.write_paths(TEST_GROUND_TRUTH_SMALL_FILEPATH, test_ground_truth_small_paths)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--paths_only', action='store_true', help='If set, then generate paths only')

    args = parser.parse_args()

    dirpaths = [
        TEST_REF_DIRPATH
    ]

    # Create directories for output files
    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    # Set up dataset for testing
    setup_dataset_make3d_testing(args.paths_only)
