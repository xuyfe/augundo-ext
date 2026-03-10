#!/usr/bin/env python3
"""
Generate UnOS 5-column train file from augundo data/kitti_raw_data.

UnOS format: each line is
  left_path right_path next_left_path next_right_path calib_path
all relative to data_dir (kitti_raw_data root). Only emits rows where
the next frame (frame+1) exists so temporal pairs are valid.

Usage (run from augundo-ext repo root):
  python bash/stereo_depth/generate_unos_train_file_from_augundo_data.py \\
    --raw_dir data/kitti_raw_data \\
    --out_file data/unos_train_4frames.txt
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description='Build UnOS 5-column train file from augundo raw KITTI')
    ap.add_argument('--raw_dir', type=str, default='data/kitti_raw_data',
                    help='Path to KITTI raw root (same as UnOS --data_dir)')
    ap.add_argument('--out_file', type=str, default='data/unos_train_4frames.txt',
                    help='Output UnOS-format train file')
    ap.add_argument('--max_samples', type=int, default=None,
                    help='Max number of lines (default: all)')
    args = ap.parse_args()

    raw_dir = os.path.normpath(args.raw_dir)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError('Raw dir not found: {}'.format(raw_dir))

    lines = []

    for date_name in sorted(os.listdir(raw_dir)):
        date_path = os.path.join(raw_dir, date_name)
        if not os.path.isdir(date_path):
            continue
        calib_rel = os.path.join(date_name, 'calib_cam_to_cam.txt')
        if not os.path.isfile(os.path.join(raw_dir, calib_rel)):
            continue

        for drive_name in sorted(os.listdir(date_path)):
            drive_path = os.path.join(date_path, drive_name)
            if not os.path.isdir(drive_path) or '_sync' not in drive_name:
                continue
            left_dir = os.path.join(drive_path, 'image_02', 'data')
            right_dir = os.path.join(drive_path, 'image_03', 'data')
            if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
                continue

            left_frames = sorted(f for f in os.listdir(left_dir) if f.endswith('.png'))
            right_frames = set(f for f in os.listdir(right_dir) if f.endswith('.png'))
            prefix = os.path.join(date_name, drive_name)

            for j, frame in enumerate(left_frames):
                if frame not in right_frames:
                    continue
                if j + 1 >= len(left_frames):
                    continue
                next_frame = left_frames[j + 1]
                if next_frame not in right_frames:
                    continue

                left = os.path.join(prefix, 'image_02', 'data', frame)
                right = os.path.join(prefix, 'image_03', 'data', frame)
                next_left = os.path.join(prefix, 'image_02', 'data', next_frame)
                next_right = os.path.join(prefix, 'image_03', 'data', next_frame)
                lines.append('{} {} {} {} {}'.format(left, right, next_left, next_right, calib_rel))

                if args.max_samples and len(lines) >= args.max_samples:
                    break
            if args.max_samples and len(lines) >= args.max_samples:
                break
        if args.max_samples and len(lines) >= args.max_samples:
            break

    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)
    with open(args.out_file, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print('Wrote {} ({} lines). Use with UnOS: --data_dir=<path_to_{}> --train_file=<path_to_{}>'.format(
        args.out_file, len(lines), os.path.basename(raw_dir.rstrip('/')), os.path.basename(args.out_file)))


if __name__ == '__main__':
    main()
