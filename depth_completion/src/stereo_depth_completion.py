'''
Stereo depth completion with AugUndo augmentation.

Extends the monocular AugUndo pipeline to stereo models (UnOS, BridgeDepthFlow).

Key stereo-specific AugUndo considerations:
- Geometric augmentations are applied consistently to BOTH left and right images
- Horizontal flip swaps left and right images (reverses stereo geometry)
- Photometric augmentations are applied only to the input left image
- AugUndo: augment inputs, forward through model, reverse-transform output, compute loss
- Stereo models compute their own loss internally using left+right pairs
'''

import os, time, sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, os.getcwd())
from utils.src import data_utils, eval_utils
from utils.src.log_utils import log
from depth_completion_model import DepthCompletionModel
from utils.src.transforms import Transforms
from stereo_dataloader import (
    StereoDepthCompletionTrainingDataset,
    StereoDepthCompletionInferenceDataset,
    build_stereo_datasets_from_unos_file
)


def train(train_data_file,
          train_data_root,
          train_sparse_depth_path,
          # Validation paths
          val_left_image_path,
          val_right_image_path,
          val_sparse_depth_path,
          val_intrinsics_path,
          val_ground_truth_path,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          # Input settings
          input_channels_image,
          input_channels_depth,
          normalized_image_range,
          # Depth network settings
          model_name,
          network_modules,
          min_predict_depth,
          max_predict_depth,
          # Loss function settings
          w_losses,
          w_weight_decay_depth,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          # Photometric data augmentations
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_gamma,
          augmentation_random_hue,
          augmentation_random_saturation,
          augmentation_random_gaussian_blur_kernel_size,
          augmentation_random_gaussian_blur_sigma_range,
          augmentation_random_noise_type,
          augmentation_random_noise_spread,
          # Geometric data augmentations
          augmentation_padding_mode,
          augmentation_random_crop_type,
          augmentation_random_crop_to_shape,
          augmentation_random_flip_type,
          augmentation_random_rotate_max,
          augmentation_random_crop_and_pad,
          augmentation_random_resize_to_shape,
          augmentation_random_resize_and_pad,
          augmentation_random_resize_and_crop,
          # Occlusion data augmentations
          augmentation_random_remove_patch_percent_range_image,
          augmentation_random_remove_patch_size_image,
          augmentation_random_remove_patch_percent_range_depth,
          augmentation_random_remove_patch_size_depth,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path,
          n_step_per_checkpoint,
          n_step_per_summary,
          n_image_per_summary,
          start_step_validation,
          restore_paths,
          # Hardware settings
          device='cuda',
          n_thread=8):

    # Select device
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    # Set up checkpoint and event paths
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_dirpath = os.path.join(checkpoint_path, 'checkpoints-{}')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty
    }

    '''
    Load input paths and set up dataloaders
    '''
    # Read sparse depth paths if provided
    if train_sparse_depth_path is not None:
        train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)
    else:
        train_sparse_depth_paths = None

    # Build training dataset from UnOS-format 4-frame file
    train_dataset = build_stereo_datasets_from_unos_file(
        train_file_path=train_data_file,
        kitti_raw_root=train_data_root,
        sparse_depth_paths=train_sparse_depth_paths,
        random_crop_shape=(n_height, n_width),
        random_crop_type=augmentation_random_crop_type)

    n_train_sample = len(train_dataset)

    n_train_step = \
        learning_schedule[-1] * np.floor(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=True)

    # Set up normalization
    train_transforms_normalization = Transforms(
        normalized_image_range=normalized_image_range)

    # Set up geometric augmentations
    # For stereo: disable horizontal flip in geometric transforms
    # We handle horizontal flip separately since it swaps left/right
    stereo_flip_type = []
    do_stereo_horizontal_flip = False
    for ft in augmentation_random_flip_type:
        if ft == 'horizontal':
            do_stereo_horizontal_flip = True
        else:
            stereo_flip_type.append(ft)
    if not stereo_flip_type:
        stereo_flip_type = ['none']

    train_transforms_geometric = Transforms(
        random_flip_type=stereo_flip_type,
        random_rotate_max=augmentation_random_rotate_max,
        random_crop_and_pad=augmentation_random_crop_and_pad,
        random_resize_to_shape=augmentation_random_resize_to_shape,
        random_resize_and_pad=augmentation_random_resize_and_pad,
        random_resize_and_crop=augmentation_random_resize_and_crop)

    train_transforms_crop_to_shape = Transforms(
        random_crop_to_shape=augmentation_random_crop_to_shape)

    train_transforms_photometric = Transforms(
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_gamma=augmentation_random_gamma,
        random_hue=augmentation_random_hue,
        random_saturation=augmentation_random_saturation,
        random_gaussian_blur_kernel_size=augmentation_random_gaussian_blur_kernel_size,
        random_gaussian_blur_sigma_range=augmentation_random_gaussian_blur_sigma_range,
        random_noise_type=augmentation_random_noise_type,
        random_noise_spread=augmentation_random_noise_spread,
        random_remove_patch_percent_range=augmentation_random_remove_patch_percent_range_image,
        random_remove_patch_size=augmentation_random_remove_patch_size_image)

    train_transforms_point_cloud = Transforms(
        random_remove_patch_percent_range=augmentation_random_remove_patch_percent_range_depth,
        random_remove_patch_size=augmentation_random_remove_patch_size_depth)

    # Load validation data if available
    is_available_validation = \
        val_left_image_path is not None and \
        val_right_image_path is not None and \
        val_sparse_depth_path is not None and \
        val_intrinsics_path is not None and \
        val_ground_truth_path is not None

    if is_available_validation:
        val_left_image_paths = data_utils.read_paths(val_left_image_path)
        val_right_image_paths = data_utils.read_paths(val_right_image_path)
        val_sparse_depth_paths = data_utils.read_paths(val_sparse_depth_path)
        val_intrinsics_paths = data_utils.read_paths(val_intrinsics_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        n_val_sample = len(val_left_image_paths)

        for paths in [val_right_image_paths, val_sparse_depth_paths,
                      val_intrinsics_paths, val_ground_truth_paths]:
            assert len(paths) == n_val_sample

        val_dataloader = torch.utils.data.DataLoader(
            StereoDepthCompletionInferenceDataset(
                left_image_paths=val_left_image_paths,
                right_image_paths=val_right_image_paths,
                sparse_depth_paths=val_sparse_depth_paths,
                intrinsics_paths=val_intrinsics_paths,
                ground_truth_paths=val_ground_truth_paths),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        val_transforms = Transforms(
            normalized_image_range=normalized_image_range)

    '''
    Set up the model
    '''
    depth_completion_model = DepthCompletionModel(
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    parameters_depth_model = depth_completion_model.parameters_depth()

    depth_completion_model.train()

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    '''
    Log settings
    '''
    log('Stereo depth completion training', log_path)
    log('Model: {}'.format(model_name), log_path)
    log('n_train_sample={}  n_train_step={}'.format(n_train_sample, n_train_step), log_path)
    log('', log_path)

    '''
    Train model
    '''
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    optimizer_depth = torch.optim.Adam([
        {
            'params': parameters_depth_model,
            'weight_decay': w_weight_decay_depth
        }],
        lr=learning_rate)

    # Split along batch across multiple GPUs
    if torch.cuda.device_count() > 1:
        depth_completion_model.data_parallel()

    # Start training
    train_step = 0

    if len(restore_paths) > 0:
        try:
            train_step, optimizer_depth, _ = depth_completion_model.restore_model(
                restore_paths,
                optimizer_depth=optimizer_depth)
        except Exception:
            print('Failed to restore optimizer: Ignoring...')
            train_step, _, _ = depth_completion_model.restore_model(restore_paths)

        for g in optimizer_depth.param_groups:
            g['lr'] = learning_rate

        n_train_step = n_train_step + train_step

    time_start = time.time()

    # Padding modes for: image, sparse_depth, validity_map
    padding_modes = [augmentation_padding_mode, 'constant', 'constant']

    interpolation_modes = \
        train_transforms_geometric.map_interpolation_mode_names_to_enums(['nearest'])

    log('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            for g in optimizer_depth.param_groups:
                g['lr'] = learning_rate

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        for batch in train_dataloader:

            train_step = train_step + 1

            # Fetch data from dict-based dataloader
            left_image0 = batch['left_image'].to(device)
            right_image0 = batch['right_image'].to(device)
            sparse_depth0 = batch['sparse_depth'].to(device)
            intrinsics = batch['intrinsics'].to(device)

            # Validity map is where sparse depth is available
            validity_map0 = torch.where(
                sparse_depth0 > 0,
                torch.ones_like(sparse_depth0),
                sparse_depth0)

            # ---- Stereo horizontal flip: swap left/right with 50% probability ----
            if do_stereo_horizontal_flip and np.random.random() < 0.5 * augmentation_probability:
                # Flip both images horizontally and swap left/right
                left_image0_flipped = torch.flip(right_image0, dims=[-1])
                right_image0_flipped = torch.flip(left_image0, dims=[-1])
                left_image0 = left_image0_flipped
                right_image0 = right_image0_flipped

                # Also flip sparse depth and validity map
                sparse_depth0 = torch.flip(sparse_depth0, dims=[-1])
                validity_map0 = torch.flip(validity_map0, dims=[-1])

                # Adjust intrinsics: cx = width - cx
                _, _, _, w = left_image0.shape
                intrinsics = intrinsics.clone()
                intrinsics[:, 0, 2] = w - intrinsics[:, 0, 2]

            # ---- Crop to shape (applied to all images consistently) ----
            [left_image0, right_image0, sparse_depth0, validity_map0], \
                [intrinsics], _ = train_transforms_crop_to_shape.transform(
                    images_arr=[left_image0, right_image0, sparse_depth0, validity_map0],
                    intrinsics_arr=[intrinsics],
                    interpolation_modes=interpolation_modes,
                    random_transform_probability=augmentation_probability)

            # ---- Geometric augmentation for AugUndo ----
            # Apply same geometric transform to left, right, sparse_depth, validity_map
            # AugUndo: augment inputs, forward, undo on output, loss on originals
            [input_left_image0, input_right_image0, input_sparse_depth0, input_validity_map0], \
                [input_intrinsics], \
                transform_performed_geometric = train_transforms_geometric.transform(
                    images_arr=[left_image0, right_image0, sparse_depth0, validity_map0],
                    intrinsics_arr=[intrinsics],
                    padding_modes=[augmentation_padding_mode, augmentation_padding_mode, 'constant', 'constant'],
                    random_transform_probability=augmentation_probability)

            # Perform point removal from sparse depth
            [input_sparse_depth0], _ = train_transforms_point_cloud.transform(
                images_arr=[input_sparse_depth0],
                random_transform_probability=augmentation_probability)

            # Photometric augmentation on input left image only
            [input_left_image0], _ = train_transforms_photometric.transform(
                images_arr=[input_left_image0],
                random_transform_probability=augmentation_probability)

            # Normalize all images
            [left_image0, right_image0, input_left_image0, input_right_image0], _ = \
                train_transforms_normalization.transform(
                    images_arr=[left_image0, right_image0, input_left_image0, input_right_image0])

            # ---- Forward through the network ----
            output_depth0 = depth_completion_model.forward_depth(
                image=input_left_image0,
                sparse_depth=input_sparse_depth0,
                validity_map=input_validity_map0,
                intrinsics=input_intrinsics,
                right_image=input_right_image0,
                return_all_outputs=True)

            # ---- AugUndo: reverse geometric transform on output depth ----
            output_depth0 = train_transforms_geometric.reverse_transform(
                images_arr=output_depth0,
                transform_performed=transform_performed_geometric,
                padding_modes=[padding_modes[0]])

            # ---- Compute loss on original (unaugmented) images ----
            w_losses['epoch'] = epoch

            loss, loss_info = depth_completion_model.compute_loss(
                image0=left_image0,
                image1=left_image0,  # No temporal frames for stereo-only
                image2=left_image0,
                output_depth0=output_depth0,
                sparse_depth0=sparse_depth0,
                validity_map0=validity_map0,
                intrinsics=intrinsics,
                pose0to1=None,
                pose0to2=None,
                supervision_type='unsupervised',
                w_losses=w_losses,
                right_image0=right_image0)

            # ---- Backpropagate ----
            optimizer_depth.zero_grad()
            loss.backward()
            optimizer_depth.step()

            if (train_step % n_step_per_summary) == 0:
                # Log summary
                depth_completion_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    image0=left_image0,
                    output_depth0=output_depth0[0].detach().clone(),
                    sparse_depth0=sparse_depth0,
                    validity_map0=validity_map0,
                    scalars=loss_info,
                    n_image_per_summary=min(n_batch, n_image_per_summary))

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                if train_step >= start_step_validation and is_available_validation:
                    depth_completion_model.eval()

                    with torch.no_grad():
                        best_results = validate(
                            depth_model=depth_completion_model,
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_image_per_summary=n_image_per_summary,
                            log_path=log_path)

                    depth_completion_model.train()

                # Save checkpoints
                depth_completion_model.save_model(
                    checkpoint_dirpath.format(train_step),
                    train_step,
                    optimizer_depth)

    # Final validation and checkpoint
    depth_completion_model.eval()

    if is_available_validation:
        with torch.no_grad():
            best_results = validate(
                depth_model=depth_completion_model,
                dataloader=val_dataloader,
                transforms=val_transforms,
                step=train_step,
                best_results=best_results,
                min_evaluate_depth=min_evaluate_depth,
                max_evaluate_depth=max_evaluate_depth,
                device=device,
                summary_writer=val_summary_writer,
                n_image_per_summary=n_image_per_summary,
                log_path=log_path)

    depth_completion_model.save_model(
        checkpoint_dirpath.format(train_step),
        train_step,
        optimizer_depth)


def validate(depth_model,
             dataloader,
             transforms,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             summary_writer,
             n_image_per_summary=4,
             n_interval_per_summary=250,
             log_path=None):

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    image_summary = []
    output_depth_summary = []
    sparse_depth_summary = []
    validity_map_summary = []
    ground_truth_summary = []

    for idx, batch in enumerate(dataloader):

        left_image = batch['left_image'].to(device)
        right_image = batch['right_image'].to(device)
        sparse_depth = batch['sparse_depth'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        ground_truth = batch['ground_truth'].to(device)

        with torch.no_grad():
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            [left_image_norm, right_image_norm], _ = transforms.transform(
                images_arr=[left_image, right_image],
                random_transform_probability=0.0)

            output_depth = depth_model.forward_depth(
                image=left_image_norm,
                sparse_depth=sparse_depth,
                validity_map=validity_map,
                intrinsics=intrinsics,
                right_image=right_image_norm,
                return_all_outputs=False)

        if (idx % n_interval_per_summary) == 0 and summary_writer is not None:
            image_summary.append(left_image)
            output_depth_summary.append(output_depth)
            sparse_depth_summary.append(sparse_depth)
            validity_map_summary.append(validity_map)
            ground_truth_summary.append(ground_truth)

        # Convert to numpy to validate
        output_depth = np.squeeze(output_depth.cpu().numpy())
        ground_truth_np = np.squeeze(ground_truth.cpu().numpy())

        validity_map_np = np.where(ground_truth_np > 0, 1, 0)

        validity_mask = np.where(validity_map_np > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth_np > min_evaluate_depth,
            ground_truth_np < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        output_depth = output_depth[mask]
        ground_truth_np = ground_truth_np[mask]

        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth_np)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth_np)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth_np)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth_np)

    # Compute mean metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    # Log to tensorboard
    if summary_writer is not None and len(image_summary) > 0:
        depth_model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image0=torch.cat(image_summary, dim=0),
            output_depth0=torch.cat(output_depth_summary, dim=0),
            sparse_depth0=torch.cat(sparse_depth_summary, dim=0),
            validity_map0=torch.cat(validity_map_summary, dim=0),
            ground_truth0=torch.cat(ground_truth_summary, dim=0),
            scalars={'mae': mae, 'rmse': rmse, 'imae': imae, 'irmse': irmse},
            n_image_per_summary=n_image_per_summary)

    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'), log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse, imae, irmse), log_path)

    n_improve = 0
    if np.round(mae, 2) <= np.round(best_results['mae'], 2):
        n_improve += 1
    if np.round(rmse, 2) <= np.round(best_results['rmse'], 2):
        n_improve += 1
    if np.round(imae, 2) <= np.round(best_results['imae'], 2):
        n_improve += 1
    if np.round(irmse, 2) <= np.round(best_results['irmse'], 2):
        n_improve += 1

    if n_improve > 2:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse

    log('Best results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'), log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse'],
        best_results['imae'],
        best_results['irmse']), log_path)

    return best_results


def run(left_image_path,
        right_image_path,
        sparse_depth_path,
        intrinsics_path,
        ground_truth_path,
        # Restore path settings
        restore_paths,
        # Input settings
        input_channels_image,
        input_channels_depth,
        normalized_image_range,
        # Depth network settings
        model_name,
        network_modules,
        min_predict_depth,
        max_predict_depth,
        # Evaluation settings
        min_evaluate_depth,
        max_evaluate_depth,
        # Output settings
        output_path,
        save_outputs,
        keep_input_filenames,
        # Hardware settings
        device):

    # Select device
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    '''
    Set up output paths
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_path = os.path.join(output_path, 'results.txt')
    output_dirpath = os.path.join(output_path, 'outputs')

    if save_outputs:
        image_dirpath = os.path.join(output_dirpath, 'image')
        output_depth_dirpath = os.path.join(output_dirpath, 'output_depth')
        sparse_depth_dirpath = os.path.join(output_dirpath, 'sparse_depth')
        ground_truth_dirpath = os.path.join(output_dirpath, 'ground_truth')

        for dirpath in [output_dirpath, image_dirpath, output_depth_dirpath,
                        sparse_depth_dirpath, ground_truth_dirpath]:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

    '''
    Load input paths and set up dataloader
    '''
    left_image_paths = data_utils.read_paths(left_image_path)
    right_image_paths = data_utils.read_paths(right_image_path)
    sparse_depth_paths = data_utils.read_paths(sparse_depth_path)
    intrinsics_paths = data_utils.read_paths(intrinsics_path)

    is_available_ground_truth = False
    if ground_truth_path is not None and ground_truth_path != '':
        is_available_ground_truth = True
        ground_truth_paths = data_utils.read_paths(ground_truth_path)
    else:
        ground_truth_paths = None

    n_sample = len(left_image_paths)

    dataloader = torch.utils.data.DataLoader(
        StereoDepthCompletionInferenceDataset(
            left_image_paths=left_image_paths,
            right_image_paths=right_image_paths,
            sparse_depth_paths=sparse_depth_paths,
            intrinsics_paths=intrinsics_paths,
            ground_truth_paths=ground_truth_paths),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    transforms = Transforms(
        normalized_image_range=normalized_image_range)

    '''
    Set up the model
    '''
    depth_model = DepthCompletionModel(
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    depth_model.restore_model(restore_paths)
    depth_model.eval()

    log('Stereo depth completion inference', log_path)
    log('Model: {}'.format(model_name), log_path)
    log('n_sample={}'.format(n_sample), log_path)
    log('', log_path)

    '''
    Run model
    '''
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    time_elapse = 0.0

    from PIL import Image

    for idx, batch in enumerate(dataloader):

        left_image = batch['left_image'].to(device)
        right_image = batch['right_image'].to(device)
        sparse_depth = batch['sparse_depth'].to(device)
        intrinsics = batch['intrinsics'].to(device)

        if dataloader.dataset.has_ground_truth:
            ground_truth = batch['ground_truth'].to(device)

        time_start = time.time()

        with torch.no_grad():
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            [left_image_norm, right_image_norm], _ = transforms.transform(
                images_arr=[left_image, right_image],
                random_transform_probability=0.0)

            output_depth = depth_model.forward_depth(
                image=left_image_norm,
                sparse_depth=sparse_depth,
                validity_map=validity_map,
                intrinsics=intrinsics,
                right_image=right_image_norm,
                return_all_outputs=False)

        time_elapse = time_elapse + (time.time() - time_start)

        output_depth_np = np.squeeze(output_depth.detach().cpu().numpy())

        if save_outputs:
            image_np = np.transpose(np.squeeze(left_image.cpu().numpy()), (1, 2, 0))
            sparse_depth_np = np.squeeze(sparse_depth.cpu().numpy())

            if keep_input_filenames:
                filename = os.path.splitext(os.path.basename(left_image_paths[idx]))[0] + '.png'
            else:
                filename = '{:010d}.png'.format(idx)

            image_save_path = os.path.join(image_dirpath, filename)
            image_save = (255 * image_np).astype(np.uint8)
            Image.fromarray(image_save).save(image_save_path)

            output_depth_save_path = os.path.join(output_depth_dirpath, filename)
            data_utils.save_depth(output_depth_np, output_depth_save_path)

            sparse_depth_save_path = os.path.join(sparse_depth_dirpath, filename)
            data_utils.save_depth(sparse_depth_np, sparse_depth_save_path)

        if is_available_ground_truth:
            ground_truth_np = np.squeeze(ground_truth.cpu().numpy())

            if save_outputs:
                ground_truth_save_path = os.path.join(ground_truth_dirpath, filename)
                data_utils.save_depth(ground_truth_np, ground_truth_save_path)

            validity_map_np = np.where(ground_truth_np > 0, 1, 0)
            validity_mask = np.where(validity_map_np > 0, 1, 0)
            min_max_mask = np.logical_and(
                ground_truth_np > min_evaluate_depth,
                ground_truth_np < max_evaluate_depth)
            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            output_depth_masked = output_depth_np[mask]
            ground_truth_masked = ground_truth_np[mask]

            mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth_masked, 1000.0 * ground_truth_masked)
            rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth_masked, 1000.0 * ground_truth_masked)
            imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth_masked, 0.001 * ground_truth_masked)
            irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth_masked, 0.001 * ground_truth_masked)

    time_elapse = time_elapse * 1000.0

    if is_available_ground_truth:
        mae_mean = np.mean(mae)
        rmse_mean = np.mean(rmse)
        imae_mean = np.mean(imae)
        irmse_mean = np.mean(irmse)

        log('Evaluation results:', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}'.format('MAE', 'RMSE', 'iMAE', 'iRMSE'), log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_mean, rmse_mean, imae_mean, irmse_mean), log_path)

    log('Total time: {:.2f} ms  Average time per sample: {:.2f} ms'.format(
        time_elapse, time_elapse / float(n_sample)), log_path)
