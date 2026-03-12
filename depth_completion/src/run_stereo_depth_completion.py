import argparse
import torch
from stereo_depth_completion import run


parser = argparse.ArgumentParser()

# Input paths
parser.add_argument('--left_image_path',
    type=str, required=True, help='Path to list of left image paths')
parser.add_argument('--right_image_path',
    type=str, required=True, help='Path to list of right image paths')
parser.add_argument('--sparse_depth_path',
    type=str, required=True, help='Path to list of sparse depth paths')
parser.add_argument('--intrinsics_path',
    type=str, required=True, help='Path to list of camera intrinsics paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth depth paths')

# Restore path settings
parser.add_argument('--restore_paths',
    nargs='+', type=str, required=True, help='Path to restore model from checkpoint')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=3)
parser.add_argument('--input_channels_depth',
    type=int, default=2)
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1])

# Inference size (optional). If set, inputs are resized to this before forward (avoids UnOS pyramid shape mismatch).
parser.add_argument('--n_height', type=int, default=None, help='Inference height (e.g. 256 for UnOS)')
parser.add_argument('--n_width', type=int, default=None, help='Inference width (e.g. 832 for UnOS)')

# Depth network settings
parser.add_argument('--model_name',
    type=str, default='unos', help='Stereo depth completion model name: unos, bridgedepthflow')
parser.add_argument('--network_modules',
    nargs='+', type=str, default=['stereo'])
parser.add_argument('--min_predict_depth',
    type=float, default=1.50)
parser.add_argument('--max_predict_depth',
    type=float, default=100.00)

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.00)
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.00)

# Output settings
parser.add_argument('--output_path',
    type=str, required=True, help='Path to save outputs')
parser.add_argument('--save_outputs',
    action='store_true', help='Save output depth maps')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='Use input filenames for outputs')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu')


args = parser.parse_args()

if __name__ == '__main__':

    args.model_name = args.model_name.lower()

    args.device = args.device.lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda' if args.device == 'gpu' else args.device

    run(left_image_path=args.left_image_path,
        right_image_path=args.right_image_path,
        sparse_depth_path=args.sparse_depth_path,
        intrinsics_path=args.intrinsics_path,
        ground_truth_path=args.ground_truth_path,
        restore_paths=args.restore_paths,
        n_height=args.n_height,
        n_width=args.n_width,
        input_channels_image=args.input_channels_image,
        input_channels_depth=args.input_channels_depth,
        normalized_image_range=args.normalized_image_range,
        model_name=args.model_name,
        network_modules=args.network_modules,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        output_path=args.output_path,
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        device=args.device)
