# UnOS Model, KITTI Setup, and Running augundo’s UnOS (PyTorch) the Same Way

## 1. How the SeniorThesis/UnOS model works

UnOS (Unified Unsupervised Optical-flow and Stereo-depth) lives under **`SeniorThesis/UnOS/UnDepthflow`**. It is a **TensorFlow 1.x** codebase that jointly learns:

- **Stereo depth** (disparity) from left–right image pairs (PWC-Disp).
- **Optical flow** between frames (PWC-Flow).
- **Pose** (ego-motion) and optional **motion segmentation**.

### 1.1 Modes and training stages

Training is split into modes (see `main.py`):

| Mode         | What is trained                    | Typical use              |
|-------------|------------------------------------|---------------------------|
| `stereo`    | PWC-Disp only (stereo depth)       | Stereo-only depth         |
| `flow`      | PWC-Flow only                      | Stage 1: optical flow     |
| `depth`     | PWC-Disp + pose (flow frozen)      | Stage 2: depth + pose     |
| `depthflow` | Flow + depth + pose + segmentation | Stage 3: full joint      |

For **stereo depth only**, the relevant mode is **`stereo`**.

### 1.2 Stereo depth in UnOS

- **Input:** Left image and right image (same time step).
- **Network:** PWC-Disp (Pyramid, Warping, Cost volume, from PWC-Net), outputs disparity at 4 scales.
- **Loss (in `monodepth_model.py`):**
  - **Image reconstruction:** warp right→left (and left→right) with predicted disparity; L1 + SSIM (with occlusion masking).
  - **Disparity smoothness:** edge-aware 2nd-order smoothness.
  - **Left–right consistency:** disparity should be consistent when warped across views.
- **Output:** Disparity (and depth = 1/disparity, scaled by baseline and focal length).

### 1.3 Data flow in UnOS

- **Dataloader:** `MonodepthDataloader` in `monodepth_dataloader.py`.
- Each batch needs: **left**, **right**, **next_left**, **next_right**, and **camera intrinsics** (for stereo mode only left/right and intrinsics are used for the depth net; next frames are for flow/pose in other modes).
- Images are resized to `img_height`×`img_width` (default 256×832). Intrinsics are rescaled to that resolution.
- Augmentation: random horizontal flip (L↔R), random “front/back” (current vs next frame swap).

---

## 2. How KITTI is set up for UnOS

### 2.1 Directory layout (KITTI raw)

UnOS expects **KITTI raw** under a single root (e.g. `data_dir`):

- **Left camera:** `image_02/data/XXXXXX.png`
- **Right camera:** `image_03/data/XXXXXX.png`
- **Calibration:** one `calib_cam_to_cam.txt` per **date** (e.g. `2011_09_26/calib_cam_to_cam.txt`). The code uses the **last line** of the “P_rect_xx” block to get the 3×4 projection; it then takes the 3×3 part as intrinsics.

Example:

```text
data_dir/
  2011_09_26/
    calib_cam_to_cam.txt
    2011_09_26_drive_XXXX_sync/
      image_02/data/0000000000.png   # left
      image_03/data/0000000000.png   # right
  2011_09_30/
    calib_cam_to_cam.txt
    2011_09_30_drive_XXXX_sync/
      image_02/data/...
      image_03/data/...
```

You need the [KITTI raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php) and the calibration files. For validation, UnOS also uses [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php) and [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php) ground truth (optional for stereo-only).

### 2.2 Train file format (UnOS)

The train file is a **text file**; each line has **5 space-separated paths** (all relative to `data_dir`):

```text
left_path right_path next_left_path next_right_path calib_path
```

Example (from `UnOS/UnDepthflow/filenames/kitti_train_files_png_4frames.txt`):

```text
2011_09_30/2011_09_30_drive_0028_sync/image_02/data/0000001562.png 2011_09_30/2011_09_30_drive_0028_sync/image_03/data/0000001562.png 2011_09_30/2011_09_30_drive_0028_sync/image_02/data/0000001563.png 2011_09_30/2011_09_30_drive_0028_sync/image_03/data/0000001563.png 2011_09_30/calib_cam_to_cam.txt
```

- **Columns 1–2:** left and right image for the **current** frame (stereo pair).
- **Columns 3–4:** left and right for the **next** frame (used in flow/depth/depthflow modes).
- **Column 5:** path to `calib_cam_to_cam.txt` for that date.

For **stereo-only** training, only columns 1, 2, and 5 are needed by the depth loss; the dataloader still reads all five.

### 2.3 Running UnOS (TensorFlow) with this setup

From the **UnOS/UnDepthflow** directory (with TensorFlow 1.x and KITTI raw available):

**Stereo-only (PWC-Disp):**

```bash
python main.py \
  --data_dir=/path/to/kitti_raw \
  --batch_size=4 \
  --mode=stereo \
  --train_test=train \
  --retrain=True \
  --train_file=./filenames/kitti_train_files_png_4frames.txt \
  --gt_2012_dir=/path/to/kitti_2012_gt \
  --gt_2015_dir=/path/to/kitti_2015_gt \
  --trace=/path/to/output/checkpoints_and_logs
```

Important flags: `--data_dir`, `--train_file`, `--trace`. Image size defaults to 256×832.

---

## 3. augundo’s UnOS model (`external_src/stereo_model/unos_model.py`)

The **augundo** repo wraps the same **PWC-Disp** idea in PyTorch:

- **Class:** `PWCModel` in `augundo-ext/external_src/stereo_model/unos_model.py`.
- **Input:** It expects **6-channel “image”** = **left (3 ch) concatenated with right (3 ch)** along the channel dimension: `[B, 6, H, W]`.
- **Output:** Depth from disparity: `depth = (fx * baseline) / disparity`, with optional clamping.
- **No pose:** `forward_pose` returns identity; `parameters_pose()` is empty.

So to “run the augundo UnOS model the same way” as UnOS, you need:

1. **Same data semantics:** stereo pairs (left + right) and intrinsics.
2. **Batch format:** each batch element is one 6-channel tensor (left | right).

The **augundo depth_completion** training pipeline is built around:

- **Image triplets** (t-1, t, t+1) from a **single path** that points to one wide image (three images concatenated along width), and
- Separate lists: `train_images_path`, `train_sparse_depth_path`, `train_intrinsics_path`.

It does **not** by default read UnOS-style “left right next_left next_right calib” lines or produce 6-channel stereo input. So you have two ways to run the augundo UnOS model in an UnOS-like way.

---

## 4. Option A: Run original UnOS (TensorFlow) with KITTI

- **Setup:** KITTI raw as above; train file with 5 columns per line.
- **Run:** From `UnOS/UnDepthflow`, use the `python main.py ... --mode=stereo` command above.
- **Result:** TensorFlow checkpoints and stereo depth in the original UnOS setup. No augundo code involved.

---

## 5. Option B: Run augundo’s UnOS (PyTorch) with UnOS-style KITTI data

To use **augundo’s** `unos_model.py` with the **same data setup** (KITTI raw + UnOS-format train file), the pipeline must:

1. **Read UnOS-style lines:** for each sample, (left_path, right_path, calib_path) — we can ignore next frame paths for stereo-only.
2. **Load left and right images** and build **one 6-channel image** (left | right) per sample.
3. **Load intrinsics** (from the calib file or from a separate list that matches the train file).
4. **Feed 6-channel batches** into the depth_completion trainer so that `forward_depth(image=..., ...)` receives `[B, 6, H, W]`.

Concretely, that means:

- **Stereo train file (UnOS-style):**  
  One path per line, where each “path” is actually **one line** of the UnOS train file (5 columns), or you use a **3-column** variant:  
  `left_path right_path calib_path`  
  and the dataloader reads that.

- **Dataset:** A small **stereo dataset** that:
  - Parses that file (either 5-column or 3-column),
  - Loads left and right images,
  - Loads or builds a 3×3 intrinsics matrix (from calib),
  - Returns `(stereo_6ch, sparse_dummy, intrinsics)` so the rest of the depth_completion code sees a single “image” that is 6-channel. You can use a dummy sparse depth (e.g. zeros) and a dummy validity map if the pipeline requires them.

- **Training script:** Use the existing `train_depth_completion.py` with:
  - `model_name=unos_kitti`,
  - `network_modules depth`,
  - `input_channels_image 6`,
  - Your **new** train/val lists and the **stereo dataset** plugged in where the dataloader is built (or a flag that selects “stereo dataset” when `model_name` is unos/pwc).

Once that dataset and train list are in place, you run the same bash script you use for UnOS in augundo, but with:

- `--train_images_path` (and val/calib) pointing to the **UnOS-style** list(s),
- The dataloader using the stereo dataset so that the “image” is 6-channel.

So “running the augundo UnOS model in that way” means: **same KITTI layout and same logical data (stereo pairs + calib), with a stereo dataloader that produces 6-channel input for the PyTorch UnOS model.**

---

## 6. Summary

| Topic | UnOS (SeniorThesis) | augundo UnOS (PyTorch) |
|-------|----------------------|-------------------------|
| **Code** | `UnOS/UnDepthflow` (TF 1.x) | `augundo-ext/external_src/stereo_model/unos_model.py` |
| **Input** | Left + right (separate tensors) | Single 6-ch image (left \| right) |
| **KITTI** | Raw + 5-col train file | Same raw; need UnOS-style list + stereo dataset |
| **Run “the same way”** | `main.py --mode=stereo` with `data_dir` + `train_file` | Use stereo dataset + 6-ch; same KITTI data and train file format |

---

## 7. How augundo’s data setup differs from UnOS

From the **augundo-ext README** and your `data/` folder:

### 7.1 Augundo data layout (after setup)

- **Raw:** `data/kitti_raw_data/DATE/DRIVE_sync/image_02/data/*.png` (left), `image_03/data/*.png` (right). Same as UnOS/KITTI raw.
- **Processed (derived):**  
  - **Training:** `data/kitti_depth_completion_derived/train_val_split/image_triplet/train/.../image_02/FRAME.png` — each file is **one PNG** containing **three images concatenated horizontally** (t−1, t, t+1) for **one camera (image_02)**. So augundo training is **monocular temporal triplets**, not stereo pairs.  
  - **Intrinsics:** Per-date `data/kitti_depth_completion_derived/data/DATE/intrinsics_left.npy` (training) or per-frame `.npy` in validation.  
  - **Validation:** `data/kitti_depth_completion/validation/image/` — one image per file; filenames encode drive, frame, and camera (`image_02` or `image_03`).

So:

| | UnOS | Augundo (default) |
|---|------|-------------------|
| **Training image** | Separate left + right paths per frame | One path = one file = **triplet** (3 frames concatenated, **one camera**) |
| **Per sample** | left, right, next_left, next_right, calib | image_triplet path, sparse_depth path, intrinsics path |
| **Intrinsics** | One `calib_cam_to_cam.txt` per date | Per-date or per-frame `.npy` |

To feed **UnOS-style** (left + right pairs) into **unos_model.py** you must **bypass** the triplet setup and provide **stereo pairs** from the **raw** data (and optionally match augundo’s intrinsics).

### 7.2 Feeding UnOS-style data into unos_model using augundo’s `data/` folder

**unos_model.py** expects **6-channel input** (left | right). The pipeline expects:

1. **Left image path** per sample  
2. **Right image path** per sample (same frame, other camera)  
3. **Intrinsics** per sample (either KITTI `calib_cam_to_cam.txt` or augundo’s `.npy`)

Use **raw** for images: `data/kitti_raw_data/DATE/DRIVE_sync/image_02/data/FRAME.png` and `image_03/data/FRAME.png`. For intrinsics you can use either:

- **UnOS-style:** one calib path per sample, e.g. `data/kitti_raw_data/DATE/calib_cam_to_cam.txt` (our stereo dataset loads it and parses the last line to 3×3), or  
- **Augundo-style:** one `.npy` path per sample if you have per-frame intrinsics (e.g. from validation or from a script that exports them for training).

**Concrete options:**

- **Option A – Build stereo lists from augundo raw only**  
  Run the script below to walk `data/kitti_raw_data` and write `train_left.txt`, `train_right.txt`, `train_intrinsics.txt` (calib per row). Paths are relative to repo root (e.g. `data/kitti_raw_data/...`). Then train with `train_unos_kitti_stereo.sh` (or equivalent) with `--train_images_path`, `--train_stereo_right_path`, `--train_intrinsics_path` pointing to those files.

- **Option B – Use an UnOS-format train file**  
  If you have a 5-column UnOS train file, use `generate_unos_stereo_lists.py` with `--data_root data/kitti_raw_data` and `--out_dir training/kitti/stereo` so that the three list files use paths under your augundo `data/` folder.

- **Option C – Use augundo validation images for stereo**  
  Validation has both `image_02` and `image_03` in the same folder; filenames encode camera. You can write a small script to pair rows by (drive, frame) and output left/right/intrinsics lists, then run inference or a small training run with the stereo dataset.

---

## 8. Running augundo UnOS with UnOS-style KITTI data (step-by-step)

Augundo now supports a **stereo training dataset** that reads left/right/calib lists (same semantics as UnOS). Use it as follows.

### 8.1 Generate list files from UnOS train file

From the **augundo-ext** repo root:

```bash
python bash/stereo_depth/generate_unos_stereo_lists.py \
  --unos_train_file /path/to/SeniorThesis/UnOS/UnDepthflow/filenames/kitti_train_files_png_4frames.txt \
  --data_root /path/to/kitti_raw \
  --out_dir training/kitti/stereo
```

This creates under `training/kitti/stereo/`:

- `train_left.txt` – left image paths
- `train_right.txt` – right image paths  
- `train_intrinsics.txt` – paths to `calib_cam_to_cam.txt` (one per row)

Paths in the files are `data_root + column`; use `--data_root ""` if your UnOS file already has absolute paths.

### 8.2 Generate stereo lists from augundo’s `data/` folder (no UnOS file)

If you only have augundo’s data (e.g. `data/kitti_raw_data` and optionally `data/kitti_depth_completion_derived`), run:

```bash
python bash/stereo_depth/generate_stereo_lists_from_augundo_data.py \
  --raw_dir data/kitti_raw_data \
  --out_dir training/kitti/stereo \
  --max_samples 5000
```

This walks `raw_dir`, finds every (date, drive, frame) that has both `image_02` and `image_03`, and writes:

- `train_left.txt`  — paths to `.../image_02/data/FRAME.png`
- `train_right.txt` — paths to `.../image_03/data/FRAME.png`
- `train_intrinsics.txt` — paths to `.../DATE/calib_cam_to_cam.txt`

Paths are relative to the **current working directory** (run from repo root so that paths match the rest of augundo). Then run `train_unos_kitti_stereo.sh` with default list paths, or set `LEFT_LIST`, `RIGHT_LIST`, `INTRINSICS_LIST`.

### 8.3 Train the PyTorch UnOS model (stereo)

Use the same resolution as UnOS (256×832), 6-channel input, and the stereo dataset:

```bash
bash bash/stereo_depth/train_unos_kitti_stereo.sh
```

Or call `train_depth_completion.py` directly with:

- `--train_images_path training/kitti/stereo/train_left.txt`
- `--train_stereo_right_path training/kitti/stereo/train_right.txt`
- `--train_intrinsics_path training/kitti/stereo/train_intrinsics.txt`
- `--model_name unos_kitti`
- `--network_modules depth`
- `--input_channels_image 6`
- `--n_height 256 --n_width 832`
- (Do **not** pass `--train_sparse_depth_path` when using stereo; the stereo dataset uses dummy sparse depth.)

So: **same KITTI layout and same logical data as UnOS**, with a stereo dataloader that produces 6-channel input for `augundo-ext/external_src/stereo_model/unos_model.py`.

**Note:** `unos_model.py`’s `compute_loss()` is currently a placeholder (returns 0). For real stereo training you must either implement a stereo loss there (e.g. UnOS-style image reconstruction + smoothness + left–right consistency) or train with the TensorFlow UnOS code (Option A) and use the PyTorch model only for inference/fine-tuning.

---

## 9. Running the TensorFlow UnOS (SeniorThesis/UnOS) with augundo-ext/data

To run the **original TensorFlow UnOS** model using data under **augundo-ext/data** (no separate KITTI download for the train file):

### 9.1 Generate the UnOS 5-column train file from augundo raw

From the **augundo-ext** repo root:

```bash
python bash/stereo_depth/generate_unos_train_file_from_augundo_data.py \
  --raw_dir data/kitti_raw_data \
  --out_file data/unos_train_4frames.txt
```

This walks `data/kitti_raw_data` (same layout as §2.1), and for each (date, drive, frame) where the **next** frame exists, writes one line: `left right next_left next_right calib_path` (all paths **relative to** `raw_dir`). So UnOS's `--data_dir` must point at that same root.

Optional: `--max_samples N` caps the number of lines (e.g. for a quick test).

### 9.2 Run UnOS with the generated train file

Use the provided wrapper script (it generates the train file if needed, then runs UnOS from `UnOS/UnDepthflow`):

From **augundo-ext** repo root:

```bash
bash bash/stereo_depth/run_unos_with_augundo_data.sh
```

Defaults: `--mode stereo`, `--trace augundo-ext/data/unos_trace`. To train flow/depth/depthflow or change the trace directory:

```bash
bash bash/stereo_depth/run_unos_with_augundo_data.sh --mode stereo --trace /path/to/checkpoints
bash bash/stereo_depth/run_unos_with_augundo_data.sh --mode flow --trace data/unos_trace_flow
```

Optional flags:

- `--mode` — `stereo` (default), `flow`, `depth`, or `depthflow`
- `--trace` — directory for checkpoints and logs (default: `augundo-ext/data/unos_trace`)
- `--max_samples N` — cap the train file to N lines
- `--no-gen` — skip regenerating the train file (use existing `data/unos_train_4frames.txt`)
- `--gt_2012`, `--gt_2015` — paths to KITTI 2012/2015 ground truth (optional, for validation)

**Prerequisites:** TensorFlow 1.x (and CUDA if using GPU), and `augundo-ext/data/kitti_raw_data` populated with KITTI raw (date/drive_sync/image_02, image_03, and date/calib_cam_to_cam.txt). The script resolves paths so it works when run from **augundo-ext** or from **SeniorThesis**.
