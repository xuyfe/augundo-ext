#!/usr/bin/env python3
"""
Build UnOS-style stereo list files from augundo's data folder (raw KITTI only).

Walks raw_dir (e.g. data/kitti_raw_data), finds every (date, drive, frame) that
has both image_02 and image_03, and writes:
  - <out_prefix>_left.txt   : paths to .../image_02/data/FRAME.png
  - <out_prefix>_right.txt  : paths to .../image_03/data/FRAME.png
  - <out_prefix>_intrinsics.txt : paths to .../DATE/calib_cam_to_cam.txt

Paths are relative to cwd (run from augundo-ext repo root).

Usage:
  python bash/stereo_depth/generate_stereo_lists_from_augundo_data.py \\
    --raw_dir data/kitti_raw_data \\
    --out_dir training/kitti/stereo \\
    --max_samples 5000
"""

import argparse
import os
import glob


def main():
    ap = argparse.ArgumentParser(description='Build stereo left/right/calib lists from augundo raw KITTI')
    ap.add_argument('--raw_dir', type=str, default='data/kitti_raw_data',
                    help='Path to KITTI raw root (contains DATE/DRIVE_sync/image_02/data, image_03/data)')
    ap.add_argument('--out_dir', type=str, default='training/kitti/stereo',
                    help='Output directory for list files')
    ap.add_argument('--out_prefix', type=str, default='train',
                    help='Prefix for output files: <prefix>_left.txt, etc.')
    ap.add_argument('--max_samples', type=int, default=None,
                    help='Max number of pairs to emit (default: all)')
    ap.add_argument('--path_prefix', type=str, default='',
                    help='Prefix to prepend to every path (e.g. data/ if paths should be data/kitti_raw_data/...)')
    args = ap.parse_args()

    raw_dir = os.path.normpath(args.raw_dir)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError('Raw dir not found: {}'.format(raw_dir))

    left_list, right_list, calib_list = [], [], []

    # Expect: raw_dir / DATE / DRIVE_sync / image_02/data/*.png and image_03/data/*.png
    #         raw_dir / DATE / calib_cam_to_cam.txt
    for date_name in sorted(os.listdir(raw_dir)):
        date_path = os.path.join(raw_dir, date_name)
        if not os.path.isdir(date_path):
            continue
        calib_path = os.path.join(date_path, 'calib_cam_to_cam.txt')
        if not os.path.isfile(calib_path):
            continue
        # Paths relative to cwd (run from repo root)
        calib_rel = os.path.join(raw_dir, date_name, 'calib_cam_to_cam.txt')
        if args.path_prefix:
            calib_rel = args.path_prefix.rstrip('/') + '/' + calib_rel

        for drive_name in sorted(os.listdir(date_path)):
            drive_path = os.path.join(date_path, drive_name)
            if not os.path.isdir(drive_path) or '_sync' not in drive_name:
                continue
            left_dir = os.path.join(drive_path, 'image_02', 'data')
            right_dir = os.path.join(drive_path, 'image_03', 'data')
            if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
                continue
            left_frames = set(f for f in os.listdir(left_dir) if f.endswith('.png'))
            right_frames = set(f for f in os.listdir(right_dir) if f.endswith('.png'))
            common = left_frames & right_frames
            for frame in sorted(common):
                left_path = os.path.join(raw_dir, date_name, drive_name, 'image_02', 'data', frame)
                right_path = os.path.join(raw_dir, date_name, drive_name, 'image_03', 'data', frame)
                if args.path_prefix:
                    left_path = args.path_prefix.rstrip('/') + '/' + left_path
                    right_path = args.path_prefix.rstrip('/') + '/' + right_path
                left_list.append(left_path)
                right_list.append(right_path)
                calib_list.append(calib_rel)
                if args.max_samples and len(left_list) >= args.max_samples:
                    break
        if args.max_samples and len(left_list) >= args.max_samples:
            break

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in [('left', left_list), ('right', right_list), ('intrinsics', calib_list)]:
        out_path = os.path.join(args.out_dir, '{}_{}.txt'.format(args.out_prefix, name))
        with open(out_path, 'w') as f:
            for r in rows:
                f.write(r + '\n')
        print('Wrote {} ({} lines)'.format(out_path, len(rows)))
    print('Total stereo pairs: {}'.format(len(left_list)))


if __name__ == '__main__':
    main()
