#!/usr/bin/env bash
# Run the TensorFlow UnOS model (SeniorThesis/UnOS/UnDepthflow) using data under augundo-ext/data.
#
# Prerequisites:
#   - TensorFlow 1.x (e.g. tensorflow-gpu 1.x) and Python 2/3 as required by UnOS
#   - augundo-ext/data/kitti_raw_data populated with KITTI raw (date/drive_sync/image_02,03/data/*.png, date/calib_cam_to_cam.txt)
#
# Usage (from augundo-ext repo root):
#   bash bash/stereo_depth/run_unos_with_augundo_data.sh [--mode stereo] [--trace /path/to/checkpoints] [--max_samples N]
#
# Or from SeniorThesis:
#   bash augundo-ext/bash/stereo_depth/run_unos_with_augundo_data.sh
#
# Options:
#   --mode       One of: stereo (default), flow, depth, depthflow
#   --trace      Directory for UnOS checkpoints and logs (default: augundo-ext/data/unos_trace)
#   --max_samples  Cap train file to N lines (default: all)
#   --no-gen     Skip regenerating the train file (use existing data/unos_train_4frames.txt)
#   --gt_2012    Path to KITTI 2012 stereo GT (optional)
#   --gt_2015    Path to KITTI 2015 scene flow GT (optional)

set -e

# Script lives in augundo-ext/bash/stereo_depth; go to augundo-ext
AUGUNDO_EXT="$(cd "$(dirname "$0")/../.." && pwd)"
# UnOS code lives in SeniorThesis/UnOS/UnDepthflow (sibling of augundo-ext under SeniorThesis)
UNOS_DIR="${AUGUNDO_EXT}/../UnOS/UnDepthflow"
RAW_DIR="${AUGUNDO_EXT}/data/kitti_raw_data"
TRAIN_FILE="${AUGUNDO_EXT}/data/unos_train_4frames.txt"
TRACE_DIR="${AUGUNDO_EXT}/data/unos_trace"
MODE=stereo
MAX_SAMPLES=
NO_GEN=
GT_2012=
GT_2015=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)      MODE="$2"; shift 2 ;;
    --trace)     TRACE_DIR="$2"; shift 2 ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
    --no-gen)    NO_GEN=1; shift ;;
    --gt_2012)   GT_2012="$2"; shift 2 ;;
    --gt_2015)   GT_2015="$2"; shift 2 ;;
    *)           echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ ! -d "$RAW_DIR" ]]; then
  echo "Raw data dir not found: $RAW_DIR"
  echo "Populate augundo-ext/data/kitti_raw_data with KITTI raw (e.g. 2011_09_26/..., calib_cam_to_cam.txt)."
  exit 1
fi

if [[ -z "$NO_GEN" ]]; then
  echo "Generating UnOS train file from $RAW_DIR -> $TRAIN_FILE"
  GEN_ARGS=(--raw_dir "$RAW_DIR" --out_file "$TRAIN_FILE")
  [[ -n "$MAX_SAMPLES" ]] && GEN_ARGS+=(--max_samples "$MAX_SAMPLES")
  python3 "${AUGUNDO_EXT}/bash/stereo_depth/generate_unos_train_file_from_augundo_data.py" "${GEN_ARGS[@]}"
else
  if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Train file missing: $TRAIN_FILE (run without --no-gen first)"
    exit 1
  fi
  echo "Using existing train file: $TRAIN_FILE"
fi

if [[ ! -d "$UNOS_DIR" ]] || [[ ! -f "$UNOS_DIR/main.py" ]]; then
  echo "UnOS code not found: $UNOS_DIR"
  echo "Expected SeniorThesis/UnOS/UnDepthflow (TensorFlow UnOS repo)."
  exit 1
fi

mkdir -p "$TRACE_DIR"
echo "Running UnOS (mode=$MODE) with data_dir=$RAW_DIR, train_file=$TRAIN_FILE, trace=$TRACE_DIR"
cd "$UNOS_DIR"

RUN_CMD=(
  python main.py
  --data_dir "$RAW_DIR"
  --train_file "$TRAIN_FILE"
  --mode "$MODE"
  --train_test train
  --retrain True
  --trace "$TRACE_DIR"
  --batch_size 4
)
[[ -n "$GT_2012" ]] && RUN_CMD+=(--gt_2012_dir "$GT_2012")
[[ -n "$GT_2015" ]] && RUN_CMD+=(--gt_2015_dir "$GT_2015")

exec "${RUN_CMD[@]}"
