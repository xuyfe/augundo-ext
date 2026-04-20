#!/bin/bash
#SBATCH --job-name=test_augundo_unos_imports
#SBATCH --time=0-00:10:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --chdir=/home/ox4

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
    # Ensure the *current working directory* (augundo-ext repo root) is on sys.path.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # Also add parent directory as a fallback (in case the package root is one level up).
    parent = str(pathlib.Path(cwd).parent)
    if parent not in sys.path:
        sys.path.append(parent)

    # Import the UnOS-facing pieces we care about. We intentionally avoid
    # importing the full training CLI module here, since that pulls in
    # additional AugUndo utilities that are not needed just to verify the
    # UnOS wiring.
    from stereo_depth_completion.stereo_depth_completion_model import get_stereo_model
    from stereo_depth_completion.unos_model import UnOSModel
    from external_src.stereo_depth_completion.UnOS.main import Model_depthflow, Model_stereo

    print("  Imported get_stereo_model OK")
    print("  Imported UnOSModel wrapper OK")
    print("  Imported UnOS core models (Model_depthflow / Model_stereo) OK")

    print("All UnOS-related imports succeeded.")
except Exception as e:
    import traceback
    print("Import test FAILED:")
    traceback.print_exc()
    raise SystemExit(1)
PY

echo "Import test completed"

