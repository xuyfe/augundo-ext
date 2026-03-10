#!/bin/bash
# Train BridgeDepthFlow (monodepth) on augundo KITTI-style data.
# Run from SeniorThesis. If TRAIN_FILE is set and exists, use it; else pairs are auto-discovered from RAW_DIR.
# Mac: MPS or CPU. Linux: set CUDA_VISIBLE_DEVICES if needed.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUGUNDO_EXT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SENIOR_THESIS="$(cd "$AUGUNDO_EXT/../.." && pwd)"
RAW_DIR="${RAW_DIR:-$AUGUNDO_EXT/data/kitti_raw_data}"
TRAIN_FILE="${TRAIN_FILE:-}"
TRACE_DIR="${TRACE_DIR:-$SENIOR_THESIS/BridgeDepthFlow_augundo/checkpoints}"

if [[ ! -d "$RAW_DIR" ]]; then
  echo "Data dir not found: $RAW_DIR (set RAW_DIR or place KITTI raw under augundo-ext/data/kitti_raw_data)"
  exit 1
fi

CMD=(python -m BridgeDepthFlow_augundo.main --model_name monodepth --data_path "$RAW_DIR" --trace "$TRACE_DIR" --input_height 256 --input_width 512 --batch_size 2 --num_epochs 80)
if [[ -n "$TRAIN_FILE" && -f "$TRAIN_FILE" ]]; then
  CMD+=(--filenames_file "$TRAIN_FILE")
else
  echo "No TRAIN_FILE (or file missing); auto-discovering pairs from $RAW_DIR"
fi

cd "$SENIOR_THESIS"
"${CMD[@]}" "$@"
