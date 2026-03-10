#!/usr/bin/env bash
# Run UnOS_torch evaluation using a trained checkpoint and augundo-ext data.
#
# Usage (from augundo-ext repo root):
#   bash bash/stereo_depth/run_unos_torch_eval_with_augundo_data.sh --checkpoint <path_to_latest.pt> [--output_dir ./eval_out] [--gt_list path/to/gt_paths.txt]
#
# Options:
#   --checkpoint   Path to checkpoint (e.g. data/unos_trace_torch/latest.pt)
#   --output_dir  Where to save pred disparity (default: data/unos_torch_eval)
#   --list_file   UnOS 5-col list (default: data/unos_train_4frames.txt)
#   --gt_list     Optional: one GT disparity path per line for metrics

set -e

AUGUNDO_EXT="$(cd "$(dirname "$0")/../.." && pwd)"
UNOS_TORCH_DIR="${AUGUNDO_EXT}/../UnOS_torch"
RAW_DIR="${AUGUNDO_EXT}/data/kitti_raw_data"
LIST_FILE="${AUGUNDO_EXT}/data/unos_train_4frames.txt"
OUTPUT_DIR="${AUGUNDO_EXT}/data/unos_torch_eval"
CHECKPOINT=
GT_LIST=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
    --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --list_file)   LIST_FILE="$2"; shift 2 ;;
    --gt_list)     GT_LIST="$2"; shift 2 ;;
    *)             echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]] || [[ ! -f "$CHECKPOINT" ]]; then
  echo "Usage: $0 --checkpoint <path_to_latest.pt> [--output_dir ...] [--gt_list ...]"
  echo "Example: $0 --checkpoint data/unos_trace_torch/latest.pt"
  exit 1
fi

if [[ ! -d "$UNOS_TORCH_DIR" ]] || [[ ! -f "$UNOS_TORCH_DIR/eval.py" ]]; then
  echo "UnOS_torch not found: $UNOS_TORCH_DIR"
  exit 1
fi

if [[ ! -f "$LIST_FILE" ]]; then
  echo "List file not found: $LIST_FILE. Generate it first with generate_unos_train_file_from_augundo_data.py"
  exit 1
fi

cd "$UNOS_TORCH_DIR"
RUN_CMD=(
  python eval.py
  --checkpoint "$CHECKPOINT"
  --data_dir "$RAW_DIR"
  --list_file "$LIST_FILE"
  --output_dir "$OUTPUT_DIR"
)
[[ -n "$GT_LIST" ]] && RUN_CMD+=(--gt_list "$GT_LIST")

exec "${RUN_CMD[@]}"
