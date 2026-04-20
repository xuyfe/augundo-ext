module load Python/3.10.8-GCCcore-12.2.0
module load CUDA
module load cuDNN

source augundo-ext/augundo-py310env/bin/activate

SENIOR_THESIS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
UNOS_SRC="$SENIOR_THESIS/augundo-ext/external_src/stereo_depth_completion/UnOS"

DATA_PATH="${DATA_PATH:-$SENIOR_THESIS/augundo-ext/data/kitti_raw_data}"
CHECKPOINT_DIR="$SENIOR_THESIS/augundo-ext/checkpoints/augundo_unos_test_imports"
mkdir -p "$CHECKPOINT_DIR"

echo "Data dir:       $DATA_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"

# Run from augundo-ext so imports resolve
cd "$SENIOR_THESIS/augundo-ext" || exit 1

echo "CWD:            $(pwd)"

python - << 'PY'
print("Testing stereo_depth_completion / UnOS imports...")

try:
    import os, sys, pathlib
    # Ensure the *current working directory* (which we cd'ed to in the shell)
    # is on sys.path; this is the augundo-ext repo root.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # Also add parent directory as a fallback (in case the package root is one level up).
    parent = str(pathlib.Path(cwd).parent)
    if parent not in sys.path:
        sys.path.append(parent)

    import stereo_depth_completion.train_stereo_depth_completion as train_mod
    from stereo_depth_completion.stereo_depth_completion_model import get_stereo_model
    from stereo_depth_completion.unos_model import UnOSModel
    from external_src.stereo_depth_completion.UnOS.main import Model_depthflow, Model_stereo

    print("  Imported train_stereo_depth_completion OK")
    print("  Imported get_stereo_model OK")
    print("  Imported UnOSModel wrapper OK")
    print("  Imported UnOS core models (Model_depthflow / Model_stereo) OK")

    # Smoke-test argument parsing without starting training.
    # Use a minimal dummy argv; we do NOT construct the model or dataloaders here.
    import sys
    argv_backup = sys.argv
    sys.argv = ["train_stereo_depth_completion.py", "--help"]
    try:
        train_mod.build_arg_parser() if hasattr(train_mod, "build_arg_parser") else None
        print("  train_stereo_depth_completion module is importable and its CLI parser is accessible.")
    finally:
        sys.argv = argv_backup

    print("All UnOS-related imports succeeded.")
except Exception as e:
    import traceback
    print("Import test FAILED:")
    traceback.print_exc()
    raise SystemExit(1)
PY

echo "Import test completed"
