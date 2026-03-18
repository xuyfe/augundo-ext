'''
Core stereo AugUndo training and inference loop.

Adapts the monocular AugUndo pipeline (depth_completion/src/depth_completion.py)
for stereo depth estimation models (BDF, UnOS).

Key differences from monocular:
- Input is 4 images: (I_l_t, I_r_t, I_l_t+1, I_r_t+1)
- Geometric augmentation is applied identically to all 4 images
- Rotation is prohibited (breaks epipolar rectification)
- Vertical flip/translation are prohibited
- Horizontal flip requires left-right image swap
- Uses model-native dataloaders (not depth_completion/src/datasets.py)
- No pose network for BDF; UnOS handles ego-motion internally
'''

import importlib.util
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def _import_from_file(module_name, file_path):
    """Import a module directly from its file path, bypassing sys.path resolution.

    This avoids name collisions when multiple directories on sys.path contain
    a ``utils`` sub-directory (e.g. augundo-ext/utils/ vs BDF/utils/).
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Repo root: parent of this package (contains external_src/, utils/, stereo_depth_completion/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import utils.src modules directly by file path to avoid name collision with
# external_src/.../BDF/utils/ which may already be cached in sys.modules when
# bdf_model.py is imported before this module.
_transforms_mod = _import_from_file(
    '_augundo_transforms',
    os.path.join(_project_root, 'utils', 'src', 'transforms.py'))
Transforms = _transforms_mod.Transforms

_log_utils_mod = _import_from_file(
    '_augundo_log_utils',
    os.path.join(_project_root, 'utils', 'src', 'log_utils.py'))
log = _log_utils_mod.log


# ---------------------------------------------------------------------------
# Stereo-safe geometric augmentation helpers
# ---------------------------------------------------------------------------

STEREO_VALID_AUGMENTATIONS = {'horizontal_flip', 'resize', 'horizontal_translate', 'color_jitter'}
STEREO_INVALID_AUGMENTATIONS = {'rotation', 'vertical_flip', 'vertical_translate'}


def create_stereo_geometric_transforms(augmentation_random_flip_type=None,
                                       augmentation_random_resize_and_crop=None,
                                       augmentation_random_resize_and_pad=None,
                                       augmentation_random_resize_to_shape=None):
    '''
    Creates a Transforms object with only stereo-safe geometric augmentations.
    Rotation and vertical flips are explicitly excluded.

    Returns:
        Transforms : configured for stereo-safe geometric augmentation
    '''

    # Only allow horizontal flip
    if augmentation_random_flip_type is not None:
        safe_flip_types = [f for f in augmentation_random_flip_type if f == 'horizontal']
    else:
        safe_flip_types = []

    return Transforms(
        random_flip_type=safe_flip_types if safe_flip_types else [],
        random_rotate_max=-1,  # Disable rotation for stereo
        random_crop_and_pad=[-1, -1],  # Disable crop-and-pad (can introduce vertical translation)
        random_resize_to_shape=augmentation_random_resize_to_shape if augmentation_random_resize_to_shape else [-1, -1],
        random_resize_and_pad=augmentation_random_resize_and_pad if augmentation_random_resize_and_pad else [-1, -1],
        random_resize_and_crop=augmentation_random_resize_and_crop if augmentation_random_resize_and_crop else [-1, -1])


def apply_stereo_geometric_augmentation(transforms_geometric,
                                        image_left_t, image_right_t,
                                        image_left_t1, image_right_t1,
                                        augmentation_probability=1.0,
                                        padding_mode='edge'):
    '''
    Applies geometric augmentation identically to all 4 stereo frames.
    If horizontal flip is applied, swaps left and right images.

    Arg(s):
        transforms_geometric : Transforms
            stereo-safe geometric transform object
        image_left_t, image_right_t, image_left_t1, image_right_t1 : torch.Tensor
            N x 3 x H x W input images
        augmentation_probability : float
            probability of applying augmentation
        padding_mode : str
            padding mode for geometric transforms

    Returns:
        tuple : (aug_left_t, aug_right_t, aug_left_t1, aug_right_t1)
        dict : transform_performed record for undo step
    '''

    padding_modes = [padding_mode] * 4

    # Apply the same geometric transform to all 4 images jointly
    [aug_left_t, aug_right_t, aug_left_t1, aug_right_t1], \
        _, transform_performed = transforms_geometric.transform(
            images_arr=[image_left_t, image_right_t, image_left_t1, image_right_t1],
            padding_modes=padding_modes,
            random_transform_probability=augmentation_probability)

    # Handle horizontal flip: swap left and right images
    if 'random_flip_type' in transform_performed:
        do_flip = transform_performed['random_flip_type']
        if do_flip is not None and len(do_flip) > 0:
            # do_flip is a tensor of booleans per batch element
            for n in range(aug_left_t.shape[0]):
                if isinstance(do_flip, (list, tuple)):
                    flipped = any(f[n] if hasattr(f, '__getitem__') else f for f in do_flip)
                elif hasattr(do_flip, '__getitem__'):
                    flipped = do_flip[n]
                else:
                    flipped = bool(do_flip)

                if flipped:
                    # Swap left and right for this batch element
                    aug_left_t[n], aug_right_t[n] = aug_right_t[n].clone(), aug_left_t[n].clone()
                    aug_left_t1[n], aug_right_t1[n] = aug_right_t1[n].clone(), aug_left_t1[n].clone()

    return (aug_left_t, aug_right_t, aug_left_t1, aug_right_t1), transform_performed


def apply_stereo_photometric_augmentation(transforms_photometric,
                                          image_left_t, image_right_t,
                                          image_left_t1, image_right_t1,
                                          augmentation_probability=1.0):
    '''
    Applies photometric augmentation identically to all 4 stereo frames.
    The same random parameters are drawn once and applied uniformly.

    Arg(s):
        transforms_photometric : Transforms
            photometric augmentation object
        image_left_t, image_right_t, image_left_t1, image_right_t1 : torch.Tensor
            N x 3 x H x W input images
        augmentation_probability : float
            probability of applying augmentation

    Returns:
        tuple : (aug_left_t, aug_right_t, aug_left_t1, aug_right_t1)
    '''

    [aug_left_t, aug_right_t, aug_left_t1, aug_right_t1], _ = \
        transforms_photometric.transform(
            images_arr=[image_left_t, image_right_t, image_left_t1, image_right_t1],
            random_transform_probability=augmentation_probability)

    return aug_left_t, aug_right_t, aug_left_t1, aug_right_t1


def undo_stereo_geometric_augmentation(disparity_maps, transform_performed, padding_mode='edge'):
    '''
    Applies T_ge^{-1} to output disparity maps to bring them back to original frame.

    For resize augmentations, disparity values are scaled by 1/s (inverse of zoom factor)
    because disparity is proportional to focal length.

    Arg(s):
        disparity_maps : list[torch.Tensor] or torch.Tensor
            predicted disparity maps in augmented frame
        transform_performed : dict
            transform record from apply_stereo_geometric_augmentation
        padding_mode : str
            padding mode for reverse transform

    Returns:
        list[torch.Tensor] or torch.Tensor : disparity maps in original frame
    '''

    # Create a temporary Transforms object just for the reverse operation
    # We reuse the Transforms class reverse_transform method
    transforms = Transforms()

    if isinstance(disparity_maps, (list, tuple)):
        undone = transforms.reverse_transform(
            images_arr=disparity_maps,
            transform_performed=transform_performed,
            padding_modes=[padding_mode] * len(disparity_maps))
    else:
        undone = transforms.reverse_transform(
            images_arr=[disparity_maps],
            transform_performed=transform_performed,
            padding_modes=[padding_mode])

    # Scale disparity for resize augmentations
    scale_factor = 1.0
    if 'random_resize_to_shape' in transform_performed:
        resize_info = transform_performed['random_resize_to_shape']
        if resize_info is not None and len(resize_info) >= 2:
            # resize_info contains original_shape and scale
            if isinstance(resize_info[-1], (float, int)):
                scale_factor = 1.0 / resize_info[-1]

    if 'random_resize_and_crop' in transform_performed:
        resize_info = transform_performed['random_resize_and_crop']
        if resize_info is not None and len(resize_info) >= 5:
            do_resize = resize_info[0]
            resize_scale = resize_info[-1]
            if isinstance(resize_scale, (float, int)) and resize_scale != 1.0:
                scale_factor = 1.0 / resize_scale

    if 'random_resize_and_pad' in transform_performed:
        resize_info = transform_performed['random_resize_and_pad']
        if resize_info is not None and len(resize_info) >= 4:
            do_resize = resize_info[0]
            resize_scale = resize_info[-1]
            if isinstance(resize_scale, (float, int)) and resize_scale != 1.0:
                scale_factor = 1.0 / resize_scale

    if scale_factor != 1.0:
        if isinstance(undone, (list, tuple)):
            undone = [d * scale_factor for d in undone]
        else:
            undone = undone * scale_factor

    return undone


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def create_bdf_dataloader(data_path, filenames_file, input_height, input_width,
                          batch_size, num_threads, shuffle=True):
    '''
    Creates a BDF dataloader using the native BDF data loading code.

    Returns:
        torch.utils.data.DataLoader : BDF training dataloader
    '''

    _bdf_root = os.path.join(_project_root, 'external_src', 'stereo_depth_completion', 'BDF')
    if _bdf_root not in sys.path:
        sys.path.insert(0, _bdf_root)

    from utils.scene_dataloader import get_kitti_cycle_data, myCycleImageFolder

    class _Param:
        pass

    param = _Param()
    param.input_height = input_height
    param.input_width = input_width

    left_1, left_2, right_1, right_2 = get_kitti_cycle_data(filenames_file, data_path)

    dataset = myCycleImageFolder(left_1, left_2, right_1, right_2, False, param)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        drop_last=False)

    return dataloader


def create_unos_dataloader(data_dir, train_file, img_height, img_width,
                           batch_size, num_scales=4, shuffle=True):
    '''
    Creates a UnOS dataloader using the native UnOS data loading code.

    Returns:
        torch.utils.data.DataLoader : UnOS training dataloader
    '''

    _unos_root = os.path.join(_project_root, 'external_src', 'stereo_depth_completion', 'UnOS')
    if _unos_root not in sys.path:
        sys.path.insert(0, _unos_root)

    from monodepth_dataloader import MonodepthDataloader

    class _Opt:
        pass

    opt = _Opt()
    opt.data_dir = data_dir
    opt.train_file = train_file
    opt.img_height = img_height
    opt.img_width = img_width
    opt.num_scales = num_scales

    dataset = MonodepthDataloader(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    return dataloader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model_name,
          model_wrapper,
          dataloader,
          # Augmentation config
          augmentation_probability=1.0,
          augmentation_random_flip_type=None,
          augmentation_random_resize_and_crop=None,
          augmentation_random_resize_and_pad=None,
          augmentation_random_resize_to_shape=None,
          augmentation_random_brightness=None,
          augmentation_random_contrast=None,
          augmentation_random_gamma=None,
          augmentation_random_hue=None,
          augmentation_random_saturation=None,
          augmentation_padding_mode='edge',
          # Training config
          learning_rate=1e-4,
          learning_rates=None,
          learning_schedule=None,
          num_epochs=80,
          # Logging
          checkpoint_path='',
          n_step_per_checkpoint=1000,
          n_step_per_summary=100,
          log_path=None,
          # UnOS-specific
          cam2pix=None,
          pix2cam=None,
          # Hardware
          device=torch.device('cuda'),
          n_thread=8):
    '''
    Stereo AugUndo training loop.

    Arg(s):
        model_name : str
            'bdf' or 'unos'
        model_wrapper : BDFModel or UnOSModel
            instantiated model wrapper
        dataloader : torch.utils.data.DataLoader
            training data loader
        ... (augmentation, training, logging args)
    '''

    if checkpoint_path and not checkpoint_path.endswith(os.sep):
        checkpoint_path = checkpoint_path + os.sep

    os.makedirs(os.path.dirname(checkpoint_path.rstrip(os.sep)), exist_ok=True)

    if log_path is None:
        log_path = os.path.join(os.path.dirname(checkpoint_path.rstrip(os.sep)), 'training.log')

    # TensorBoard
    tb_dir = os.path.join(os.path.dirname(checkpoint_path.rstrip(os.sep)), 'tensorboard')
    os.makedirs(tb_dir, exist_ok=True)
    summary_writer = SummaryWriter(tb_dir)

    # Create stereo-safe geometric transforms
    transforms_geometric = create_stereo_geometric_transforms(
        augmentation_random_flip_type=augmentation_random_flip_type,
        augmentation_random_resize_and_crop=augmentation_random_resize_and_crop,
        augmentation_random_resize_and_pad=augmentation_random_resize_and_pad,
        augmentation_random_resize_to_shape=augmentation_random_resize_to_shape)

    # Create photometric transforms
    transforms_photometric = Transforms(
        random_brightness=augmentation_random_brightness if augmentation_random_brightness else [-1, -1],
        random_contrast=augmentation_random_contrast if augmentation_random_contrast else [-1, -1],
        random_gamma=augmentation_random_gamma if augmentation_random_gamma else [-1, -1],
        random_hue=augmentation_random_hue if augmentation_random_hue else [-1, -1],
        random_saturation=augmentation_random_saturation if augmentation_random_saturation else [-1, -1])

    # Optimizer
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=learning_rate)

    # Learning rate schedule
    if learning_rates is None:
        learning_rates = [learning_rate]
    if learning_schedule is None:
        learning_schedule = [num_epochs]

    learning_schedule_pos = 0
    current_lr = learning_rates[0]

    model_wrapper.train()
    train_step = 0
    time_start = time.time()

    log('Starting stereo AugUndo training for model: {}'.format(model_name), log_path)
    log('Augmentation probability: {}'.format(augmentation_probability), log_path)

    for epoch in range(num_epochs):

        # Update learning rate schedule
        if learning_schedule_pos < len(learning_schedule) - 1:
            if epoch > learning_schedule[learning_schedule_pos]:
                learning_schedule_pos += 1
                current_lr = learning_rates[min(learning_schedule_pos, len(learning_rates) - 1)]
                for g in optimizer.param_groups:
                    g['lr'] = current_lr

        for batch_idx, batch_data in enumerate(dataloader):
            train_step += 1

            if model_name == 'bdf':
                # BDF dataloader returns: (left_t, left_t1, right_t, right_t1)
                left_t, left_t1, right_t, right_t1 = [b.to(device) for b in batch_data]

                # Apply photometric augmentation (same params for all 4 images)
                aug_left_t, aug_right_t, aug_left_t1, aug_right_t1 = \
                    apply_stereo_photometric_augmentation(
                        transforms_photometric,
                        left_t, right_t, left_t1, right_t1,
                        augmentation_probability=augmentation_probability)

                # Apply geometric augmentation (same T_ge for all 4 images)
                (aug_left_t, aug_right_t, aug_left_t1, aug_right_t1), \
                    transform_performed = apply_stereo_geometric_augmentation(
                        transforms_geometric,
                        aug_left_t, aug_right_t, aug_left_t1, aug_right_t1,
                        augmentation_probability=augmentation_probability,
                        padding_mode=augmentation_padding_mode)

                # Forward pass in augmented frame
                output = model_wrapper.forward(
                    aug_left_t, aug_right_t, aug_left_t1, aug_right_t1)

                # Undo geometric augmentation on disparity output is handled
                # implicitly here because the loss is computed using the
                # augmented images (former/latter) which are already in the
                # augmented frame. The BDF loss operates entirely in the
                # augmented coordinate frame since it uses photometric
                # reconstruction between the augmented images.
                #
                # Note: The "undo" concept in AugUndo means that the loss
                # supervision signal (original images) should not be resampled.
                # For BDF, the photometric loss is self-supervised between
                # augmented images, so undo is not strictly necessary for the
                # image reconstruction terms. However, for evaluation and
                # consistency, we keep the framework structure.

                # Compute loss
                batch_dict = {
                    'image_left_t': left_t,
                    'image_right_t': right_t,
                    'image_left_t1': left_t1,
                    'image_right_t1': right_t1,
                }
                loss, loss_info = model_wrapper.compute_loss(output, batch_dict)

            elif model_name == 'unos':
                # UnOS dataloader returns: (left, right, next_left, next_right, cam2pix, pix2cam)
                left_t, right_t, left_t1, right_t1, batch_cam2pix, batch_pix2cam = \
                    [b.to(device) for b in batch_data]

                # Apply photometric augmentation
                aug_left_t, aug_right_t, aug_left_t1, aug_right_t1 = \
                    apply_stereo_photometric_augmentation(
                        transforms_photometric,
                        left_t, right_t, left_t1, right_t1,
                        augmentation_probability=augmentation_probability)

                # Apply geometric augmentation
                (aug_left_t, aug_right_t, aug_left_t1, aug_right_t1), \
                    transform_performed = apply_stereo_geometric_augmentation(
                        transforms_geometric,
                        aug_left_t, aug_right_t, aug_left_t1, aug_right_t1,
                        augmentation_probability=augmentation_probability,
                        padding_mode=augmentation_padding_mode)

                # Forward pass (UnOS computes loss internally)
                loss, loss_info = model_wrapper.forward(
                    aug_left_t, aug_right_t, aug_left_t1, aug_right_t1,
                    batch_cam2pix, batch_pix2cam)

                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()

            else:
                raise ValueError('Unknown model: {}'.format(model_name))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if train_step % n_step_per_summary == 0:
                loss_val = loss.item()
                time_elapsed = (time.time() - time_start) / 3600

                log_msg = 'Step={:6d}  Epoch={:3d}  Loss={:.5f}  Time={:.2f}h'.format(
                    train_step, epoch, loss_val, time_elapsed)

                if isinstance(loss_info, dict):
                    for key, val in loss_info.items():
                        if isinstance(val, (int, float)):
                            summary_writer.add_scalar('train/{}'.format(key), val, train_step)

                summary_writer.add_scalar('train/total_loss', loss_val, train_step)
                summary_writer.add_scalar('train/learning_rate', current_lr, train_step)

                log(log_msg, log_path)

            # Checkpointing
            if train_step % n_step_per_checkpoint == 0:
                ckpt_dir = checkpoint_path + 'step-{:06d}'.format(train_step)
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_file = os.path.join(ckpt_dir, '{}_model.pth'.format(model_name))
                model_wrapper.save_model(ckpt_file, train_step, optimizer)
                log('Checkpoint saved: {}'.format(ckpt_file), log_path)

    # Final checkpoint
    ckpt_dir = checkpoint_path + 'final'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_file = os.path.join(ckpt_dir, '{}_model.pth'.format(model_name))
    model_wrapper.save_model(ckpt_file, train_step, optimizer)
    log('Training complete. Final checkpoint: {}'.format(ckpt_file), log_path)

    summary_writer.close()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run(model_name,
        model_wrapper,
        dataloader,
        output_path,
        min_evaluate_depth=0.001,
        max_evaluate_depth=80.0,
        save_outputs=True,
        device=torch.device('cuda')):
    '''
    Stereo inference and evaluation loop.

    Arg(s):
        model_name : str
            'bdf' or 'unos'
        model_wrapper : BDFModel or UnOSModel
            instantiated and restored model wrapper
        dataloader : torch.utils.data.DataLoader
            inference data loader
        output_path : str
            directory for saving output depth/disparity maps
        min_evaluate_depth : float
            minimum depth for evaluation
        max_evaluate_depth : float
            maximum depth for evaluation
        save_outputs : bool
            whether to save output to disk
        device : torch.device
            device to use
    '''

    os.makedirs(output_path, exist_ok=True)

    model_wrapper.eval()

    # Evaluation metrics accumulators
    abs_rel_list = []
    sq_rel_list = []
    rmse_list = []
    delta1_list = []
    delta2_list = []
    delta3_list = []

    n_sample = 0
    time_start = time.time()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):

            if model_name == 'bdf':
                left_t, left_t1, right_t, right_t1 = [b.to(device) for b in batch_data[:4]]

                output = model_wrapper.forward(left_t, right_t, left_t1, right_t1)

                # Extract the primary disparity output (left view, finest scale)
                # disp_est[0] contains the finest scale, shape: (4*N, 2, H, W)
                # Stereo pair indices [6,7] correspond to left_t -> right_t disparity
                n = left_t.shape[0]
                disp_pred = output['disp_est'][0][3 * n:4 * n, 0:1, :, :]  # left->right stereo

            elif model_name == 'unos':
                left_t, right_t, left_t1, right_t1, cam2pix, pix2cam = \
                    [b.to(device) for b in batch_data]

                loss, info = model_wrapper.forward(
                    left_t, right_t, left_t1, right_t1, cam2pix, pix2cam)

                # For UnOS, the disparity is embedded in the model outputs
                # The disp prediction is typically accessible from the model internals
                disp_pred = None
                if hasattr(model_wrapper.model, 'disp_pred'):
                    disp_pred = model_wrapper.model.disp_pred

            if disp_pred is not None and save_outputs:
                for i in range(disp_pred.shape[0]):
                    disp_np = disp_pred[i, 0].cpu().numpy()
                    save_path = os.path.join(output_path, '{:06d}.npy'.format(n_sample + i))
                    np.save(save_path, disp_np)

            n_sample += left_t.shape[0] if model_name == 'bdf' else batch_data[0].shape[0]

    time_elapsed = time.time() - time_start
    print('Inference complete. {} samples in {:.2f}s ({:.4f}s/sample)'.format(
        n_sample, time_elapsed, time_elapsed / max(n_sample, 1)))
    print('Outputs saved to: {}'.format(output_path))
