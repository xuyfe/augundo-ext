#!/usr/bin/env bash
# Run the PyTorch UnOS model (SeniorThesis/UnOS_torch) using data under augundo-ext/data.
#
# Prerequisites:
#   - Python 3 and PyTorch (pip install torch torchvision numpy Pillow)
#   - augundo-ext/data/kitti_raw_data populated with KITTI raw
#
# Usage (from augundo-ext repo root):
#   bash bash/stereo_depth/run_unos_torch_with_augundo_data.sh [--trace /path/to/checkpoints] [--max_samples N]
#
# Options:
#   --trace        Directory for checkpoints (default: augundo-ext/data/unos_trace_torch)
#   --max_samples  Cap train file to N lines
#   --no-gen       Use existing data/unos_train_4frames.txt
#   --batch_size   Batch size (default: 4)
#   --resume       Resume from latest.pt in trace dir
#   --tensorboard  Write TensorBoard logs to trace/tensorboard (view with: tensorboard --logdir <trace>)

set -e

# On Mac (MPS): fall back to CPU for ops not yet implemented on MPS (e.g. grid_sampler backward)
export PYTORCH_ENABLE_MPS_FALLBACK=1

AUGUNDO_EXT="$(cd "$(dirname "$0")/../.." && pwd)"
UNOS_TORCH_DIR="${AUGUNDO_EXT}/../UnOS_torch"
RAW_DIR="${AUGUNDO_EXT}/data/kitti_raw_data"
TRAIN_FILE="${AUGUNDO_EXT}/data/unos_train_4frames.txt"
TRACE_DIR="${AUGUNDO_EXT}/data/unos_trace_torch"
MAX_SAMPLES=
NO_GEN=
BATCH_SIZE=4
RESUME=
TENSORBOARD=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace)       TRACE_DIR="$2"; shift 2 ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
    --no-gen)      NO_GEN=1; shift ;;
    --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
    --resume)      RESUME=1; shift ;;
    --tensorboard) TENSORBOARD=1; shift ;;
    *)             echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ ! -d "$RAW_DIR" ]]; then
  echo "Raw data dir not found: $RAW_DIR"
  echo "Populate augundo-ext/data/kitti_raw_data with KITTI raw."
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

if [[ ! -d "$UNOS_TORCH_DIR" ]] || [[ ! -f "$UNOS_TORCH_DIR/main.py" ]]; then
  echo "UnOS_torch not found: $UNOS_TORCH_DIR"
  echo "Expected SeniorThesis/UnOS_torch (PyTorch UnOS)."
  exit 1
fi

mkdir -p "$TRACE_DIR"
echo "Running UnOS_torch with data_dir=$RAW_DIR, train_file=$TRAIN_FILE, trace=$TRACE_DIR"
cd "$UNOS_TORCH_DIR" || exit 1

RUN_CMD=(
  python main.py
  --data_dir "$RAW_DIR"
  --train_file "$TRAIN_FILE"
  --trace "$TRACE_DIR"
  --batch_size "$BATCH_SIZE"
)
[[ -n "$RESUME" ]] && RUN_CMD+=(--resume)
[[ -n "$TENSORBOARD" ]] && RUN_CMD+=(--tensorboard)

exec "${RUN_CMD[@]}"
