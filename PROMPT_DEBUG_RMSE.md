# Prompt: Debug inflated stereo depth evaluation RMSE

Use this prompt with Claude Code (or another AI) to analyze why evaluation RMSE is orders of magnitude too large and to compare data/transform flows to the original models.

---

## Context

I trained and evaluated two stereo depth completion models in the augundo-ext pipeline:

1. **UnOS** (stereo-only) via `train_unos_no_aug.sh` → eval via `eval_unos.sh`
2. **BridgeDepthFlow** (stereo) via `train_bdf_no_aug.sh` → eval via `eval_bdf.sh`

Results are written to:
- `augundo-ext/results/unos_stereo/results.txt`
- `augundo-ext/results/bridgedepthflow_stereo/results.txt`

**Observed metrics (problem):**
- UnOS: MAE ≈ 84367, RMSE ≈ 85159, iMAE ≈ 79.7, iRMSE ≈ 90.5
- BDF:  MAE ≈ 14122, RMSE ≈ 18179, iMAE ≈ 573,  iRMSE ≈ 576

An example of the ground truth, image, output_depth and sparse_depth for each model is also contained in augundo-ext/results.

The papers report **RMSE on the order of &lt;5** (in meters) for their stereo models:
- UnOS: see `augundo-ext/unos.pdf` (e.g. Table 2: depth metrics on KITTI, RMSE ~3.4–4.2 m)
- BDF:  see `augundo-ext/bdf.pdf` (Lai et al. CVPR 2019; depth evaluation, RMSE in similar range)

So the current evaluation is producing RMSE values that are **thousands of times too large**, suggesting a bug in scale/units, alignment, or how predictions/ground truth are produced or compared during **inference/evaluation** (not necessarily training).

---

## Your tasks

Please analyze the codebase and identify the causes. Do the following in order.

### 1. Trace how evaluation metrics are computed (inference path)

- Find where MAE, RMSE, iMAE, iRMSE are computed for the **stereo evaluation** (the one used by `eval_unos.sh` / `eval_bdf.sh`).
- Key files to inspect:
  - `augundo-ext/depth_completion/src/run_stereo_depth_completion.py` (entry point for eval scripts)
  - `augundo-ext/depth_completion/src/stereo_depth_completion.py` (likely contains the `run()` path and any metric aggregation)
  - Any `eval_utils` or similar used for `mean_abs_err`, `root_mean_sq_err`, `inv_mean_abs_err`, `inv_root_mean_sq_err`
- For the **same evaluation path**, determine:
  - In what **units** are (i) the model’s depth output and (ii) the ground-truth depth assumed to be (e.g. meters vs mm)?
  - Whether any scaling (e.g. `1000.0 *` or `0.001 *`) is applied to depth **before** calling the metric functions, and whether that matches the units the metric functions expect.
  - Whether the reported MAE/RMSE are in **meters** or **millimeters** in the log, and what the papers report (meters vs mm).
- Check for **spatial alignment** between predicted depth and ground truth (same resolution, same crop/region, no resize that could change scale or invalidate metrics).

**Goal:** Find any unit mismatch, double scaling, or wrong assumption about depth scale that would inflate RMSE (e.g. treating meter-scale depth as mm, or vice versa).

### 2. Trace how ground truth and model outputs are loaded and preprocessed in the eval pipeline

- For **ground truth**:
  - Where are GT depth maps loaded (e.g. `stereo_dataloader.py`, `data_utils.load_depth`, or similar)?
  - What **multiplier** or conversion is used (e.g. 16-bit PNG: value/256 = meters)? Is that consistent with what the metric computation assumes?
- For **model output**:
  - In the eval path, what does the model return (e.g. depth in meters, disparity, or something else)?
  - In `unos_model.py` and `bridgedepthflow_model.py` (or their forward paths), how is **depth** derived from the network output (e.g. from disparity: depth = (focal * baseline) / disparity)? Is the output in **meters**?
- For **inference**:
  - Is the same **image size** (e.g. `n_height`, `n_width`) and the same **preprocessing** (normalization, crop, etc.) used in eval as in training? Note: `eval_unos.sh` uses `--n_width 832`, `eval_bdf.sh` uses `--n_width 512`; training scripts may differ (e.g. BDF trained at 512).

**Goal:** Ensure GT and prediction are in the same units and that no extra scaling or misinterpretation happens only at eval time.

### 3. Compare data loading and transforms: training vs evaluation vs original implementations

- **Training (augundo):**
  - How does `train_stereo_depth_completion.py` / `stereo_depth_completion.py` build the training dataloader (e.g. from `stereo_dataloader.py` or similar)?
  - What transforms are applied to images and depth (normalization, crop, augmentations when not using `--no_augment`)? What **depth units** does the training loss assume?
- **Evaluation (augundo):**
  - How does `run_stereo_depth_completion.py` load validation/GT and feed the model (batch size, image size, normalization)?
  - Are validation images and GT preprocessed in the **same way** as training (aside from augmentation), especially normalization and crop/resize?
- **Original models:**
  - **UnOS:** Under `SeniorThesis/UnOS/` (and the paper `augundo-ext/unos.pdf`), how do they define depth (from disparity?), in what units, and how do they evaluate (crop, cap depth range, metric definition)?
  - **BridgeDepthFlow:** Under `SeniorThesis/BridgeDepthFlow/` (and the paper `augundo-ext/bdf.pdf`), how do they define depth and in what units, and how do they compute evaluation metrics?
- List any differences between (a) augundo training vs augundo eval, and (b) augundo vs original UnOS/BDF in:
  - Input normalization
  - Image/depth resolution or crop
  - Depth units and scale (meters vs mm, formula from disparity)
  - Depth capping (min/max) for evaluation

**Goal:** Find inconsistencies between training and evaluation or between our pipeline and the original papers that could explain the inflated RMSE.

### 4. Summarize findings and suggest concrete fixes

- List **specific bugs or mismatches** that would inflate RMSE (e.g. “GT loaded in mm but compared to model output in meters”, “model outputs disparity but code treats it as depth in meters”, “eval uses different n_width than training for BDF”).
- For each, suggest a **concrete code or config change** (file, function, and what to change) so that:
  - Predicted and GT depth are in the same units before computing MAE/RMSE.
  - Eval preprocessing and resolution match training (and, where applicable, the original paper).
- If the fix is non-obvious (e.g. need to convert disparity to depth with the correct fx and baseline), specify the formula and where to get fx/baseline in the eval path.

---

## File reference (augundo-ext)

- **Scripts:** `train_unos_no_aug.sh`, `train_bdf_no_aug.sh`, `eval_unos.sh`, `eval_bdf.sh` (repo root or under augundo-ext)
- **Training entry:** `depth_completion/src/train_stereo_depth_completion.py`
- **Eval entry:** `depth_completion/src/run_stereo_depth_completion.py`
- **Training/eval logic:** `depth_completion/src/stereo_depth_completion.py` (train + run + any validation metric code)
- **Dataloaders / data:** `depth_completion/src/stereo_dataloader.py`; `utils/src/data_utils.py` (e.g. `load_depth`, `load_image`)
- **Model wrappers:** `depth_completion/src/unos_model.py`, `depth_completion/src/bridgedepthflow_model.py`
- **Eval helpers:** wherever `mean_abs_err`, `root_mean_sq_err`, `inv_mean_abs_err`, `inv_root_mean_sq_err` are defined (e.g. `utils/src/eval_utils.py` or similar)
- **Original implementations:** `external-src/stereo_depth_completion/UnOS/`, `external-src/stereo_depth_completion/BridgeDepthFlow/`
- **Papers:** `augundo-ext/unos.pdf`, `augundo-ext/bdf.pdf` (or paths you use for UnOS and BDF papers)

---

## Expected output format

1. **Metrics computation:** Short description of the eval code path and the exact place MAE/RMSE are computed; units of prediction and GT at that point; whether the reported numbers are in m or mm.
2. **GT and model output:** How GT is loaded (and scaled) and how model depth is obtained (formula if from disparity); any mismatch in units or scale.
3. **Data/transform comparison:** Bullet list of differences (training vs eval, augundo vs original) that could affect depth scale or alignment.
4. **Root causes:** Numbered list of bugs/mismatches that explain the inflated RMSE.
5. **Fixes:** Numbered list of concrete changes (file + location + change) to get RMSE in the expected range (&lt;5 in meters, or equivalent in mm if we choose to report in mm consistently).

Feel free to ask me any questions or clarifications.

# WHERE THE MISTAKE WAS:

I now have the complete picture. Here's the full analysis:

  ---
  Debug Analysis: Inflated Stereo Depth Evaluation RMSE

  1. Metric Computation (confirmed correct)

  The metric computation in stereo_depth_completion.py is standard and correct:
  - MAE = mean(|1000 * pred - 1000 * gt|) → result in mm
  - RMSE = sqrt(mean((1000 * pred - 1000 * gt)²)) → result in mm
  - GT loaded via data_utils.load_depth() divides 16-bit PNG by 256 → meters ✓
  - The 1000x multiplier converts to mm ✓

  The eval functions in eval_utils.py are trivial and correct.

  2. Root Cause: UnOS — Missing baseline + wrong width scale

  File: unos_model.py:156

  Current code:
  depth = fx / (disp_left.clamp(min=1e-6) * dw)   # dw = inference_width / 4

  Original UnOS evaluation (UnOS/UnDepthflow/eval/evaluation_utils.py:53-60):
  pred_disp = width * cv2.resize(pred_disp, (width, height))  # normalize→pixels at FULL width
  pred_depth = width_to_focal[width] * 0.54 / pred_disp       # depth = fx * baseline / pixel_disp

  Two bugs:
  1. Missing KITTI baseline (0.54m) — the stereo formula is depth = fx × baseline / disp_pixels, not
  fx / disp_pixels
  2. Wrong width — PWC outputs normalized disparity; converting to pixel-space requires multiplying
  by the full inference width (orig_w), not the 1/4-scale feature width (dw = orig_w/4)

  Combined error factor: orig_w / (baseline × dw) = 832 / (0.54 × 208) = 7.4×

  With depth predictions ~7.4× too large, a scene at true mean depth ~11m would predict ~81m → MAE ≈
  70,000mm. Observed MAE = 84,367mm. Consistent.

  3. Root Cause: BridgeDepthFlow — 1/disp instead of fx*baseline/disp

  File: external-src/.../BridgeDepthFlow/bridge_depth_flow.py:84

  Current code:
  depth = 1.0 / (torch.abs(horiz_disp) + 1e-6)   # horiz_disp is pixel-scale

  Original BDF evaluation (BridgeDepthFlow/utils/evaluation_utils.py:53-60):
  pred_disp = width * cv2.resize(pred_disp, (width, height))   # normalized → pixel-scale
  pred_depth = width_to_focal[width] * 0.54 / pred_disp        # fx * baseline / disp

  Original test_stereo.py:68 saves normalized disparity (disp_est[0]), and the eval script later
  scales to pixel-space and applies focal × baseline. Our wrapper uses pixel-scale disparity
  (disp_est_scale[0]) but just does 1/disp — missing the focal_length × baseline factor entirely.

  Error factor: fx_scaled × 0.54 ≈ 297 × 0.54 ≈ 160× too small. Predictions get clamped to min 1.5m,
  so typical error for a 20m GT point ≈ 18.5m = 18,500mm. Observed MAE = 14,122mm. Consistent.

  4. Training Loss Impact

  The training losses are not affected — both models use self-supervised photometric reconstruction
  loss computed on disparity/warped images, not on metric depth. The 1/disp proxy and fx/(disp*dw)
  formulas were only used in forward_depth for evaluation, not in compute_loss.

  5. Concrete Fixes

  Fix A — UnOS (unos_model.py:149-156):

  # Current (WRONG):
  depth = fx / (disp_left.clamp(min=1e-6) * dw)

  # Fixed:
  KITTI_BASELINE = 0.54  # meters
  _, _, dh, dw = disp_left.shape
  _, _, orig_h, orig_w = image.shape
  fx = intrinsics[:, 0, 0].reshape(-1, 1, 1, 1)
  # Normalized disp → pixel disp uses full width; stereo formula includes baseline
  depth = (fx * KITTI_BASELINE) / (disp_left.clamp(min=1e-6) * orig_w)

  Fix B — BridgeDepthFlow (bridgedepthflow_model.py:111-117):

  The wrapper's forward_depth already receives intrinsics. Change the depth conversion to use proper
  stereo formula:

  # Current (WRONG) — in bridge_depth_flow.py forward():
  depth = 1.0 / (torch.abs(horiz_disp) + 1e-6)

  # Fix: do conversion in the wrapper instead, using intrinsics
  depth_raw, disp_est_scale, disp_est = self.model_depth(image, right_image)

  # Proper stereo depth from pixel-scale disparity
  KITTI_BASELINE = 0.54
  horiz_disp = disp_est_scale[0][:, 0:1, :, :]  # pixel-scale
  fx = intrinsics[:, 0, 0].reshape(-1, 1, 1, 1)
  output_depth = (fx * KITTI_BASELINE) / (torch.abs(horiz_disp).clamp(min=1e-6))
  output_depth = torch.clamp(output_depth, min=self.min_predict_depth, max=self.max_predict_depth)
