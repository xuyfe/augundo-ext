#!/usr/bin/env python3
"""
Generate augundo stereo train/val list files from an UnOS-format train file.

UnOS format: each line has 5 columns (space-separated):
  left_path right_path next_left_path next_right_path calib_path
All paths are relative to a data root (e.g. KITTI raw).

Usage:
  python generate_unos_stereo_lists.py \\
    --unos_train_file /path/to/UnOS/UnDepthflow/filenames/kitti_train_files_png_4frames.txt \\
    --data_root /path/to/kitti_raw \\
    --out_dir training/kitti/stereo

This creates under out_dir:
  train_left.txt   (column 0)
  train_right.txt  (column 1)
  train_intrinsics.txt (column 4, calib path)

For validation, use --unos_val_file and --out_prefix val (creates val_*.txt).
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description='Generate stereo list files from UnOS train file')
    ap.add_argument('--unos_train_file', type=str, required=True,
                    help='Path to UnOS-format train file (5 columns per line)')
    ap.add_argument('--unos_val_file', type=str, default=None,
                    help='Optional UnOS-format val file')
    ap.add_argument('--data_root', type=str, default='',
                    help='Prefix to prepend to each path (default: none)')
    ap.add_argument('--out_dir', type=str, default='.',
                    help='Output directory for list files')
    ap.add_argument('--out_prefix', type=str, default='train',
                    help='Prefix for output filenames: <prefix>_left.txt, etc.')
    args = ap.parse_args()

    def process(unos_file, prefix):
        if not os.path.isfile(unos_file):
            raise FileNotFoundError(unos_file)
        left_list, right_list, calib_list = [], [], []
        with open(unos_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                left_list.append(args.data_root + parts[0] if args.data_root else parts[0])
                right_list.append(args.data_root + parts[1] if args.data_root else parts[1])
                calib_list.append(args.data_root + parts[4] if args.data_root else parts[4])
        os.makedirs(args.out_dir, exist_ok=True)
        for name, rows in [('left', left_list), ('right', right_list), ('intrinsics', calib_list)]:
            out_path = os.path.join(args.out_dir, '{}_{}.txt'.format(prefix, name))
            with open(out_path, 'w') as o:
                for r in rows:
                    o.write(r + '\n')
            print('Wrote {} ({} lines)'.format(out_path, len(rows)))
        return len(left_list)

    n = process(args.unos_train_file, args.out_prefix)
    print('Train samples: {}'.format(n))
    if args.unos_val_file:
        process(args.unos_val_file, 'val')


if __name__ == '__main__':
    main()
