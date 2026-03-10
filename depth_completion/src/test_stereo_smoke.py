'''
Smoke test for stereo depth completion models (UnOS and BridgeDepthFlow).

Verifies that:
1. Models can be instantiated
2. Forward pass produces correct output shapes
3. Loss computation runs without errors
4. Save/restore cycle works

Usage:
    cd augundo-ext/depth_completion/src
    python test_stereo_smoke.py
'''

import os, sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Change to augundo-ext directory so relative imports work
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))


def test_model(model_name, network_modules, device):
    '''Test a stereo model wrapper end-to-end'''

    print('=' * 60)
    print('Testing: {} with modules {}'.format(model_name, network_modules))
    print('=' * 60)

    from depth_completion_model import DepthCompletionModel

    # 1. Instantiate model
    print('[1] Instantiating model...')
    model = DepthCompletionModel(
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=1.5,
        max_predict_depth=100.0,
        device=device)
    print('    OK')

    # 2. Forward pass
    print('[2] Testing forward_depth...')
    n_batch = 2
    h, w = 256, 832

    left_image = torch.rand(n_batch, 3, h, w, device=device)
    right_image = torch.rand(n_batch, 3, h, w, device=device)
    sparse_depth = torch.zeros(n_batch, 1, h, w, device=device)
    validity_map = torch.zeros(n_batch, 1, h, w, device=device)
    intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(n_batch, 1, 1)
    intrinsics[:, 0, 0] = 721.5  # fx
    intrinsics[:, 1, 1] = 721.5  # fy
    intrinsics[:, 0, 2] = 609.5  # cx
    intrinsics[:, 1, 2] = 172.8  # cy

    model.eval()

    with torch.no_grad():
        output = model.forward_depth(
            image=left_image,
            sparse_depth=sparse_depth,
            validity_map=validity_map,
            intrinsics=intrinsics,
            right_image=right_image,
            return_all_outputs=False)

    print('    Output shape: {}'.format(output.shape))
    assert output.shape == (n_batch, 1, h, w), \
        'Expected ({}, 1, {}, {}), got {}'.format(n_batch, h, w, output.shape)
    print('    Depth range: [{:.3f}, {:.3f}]'.format(output.min().item(), output.max().item()))
    print('    OK')

    # 3. Forward with return_all_outputs
    print('[3] Testing forward_depth with return_all_outputs=True...')
    with torch.no_grad():
        outputs = model.forward_depth(
            image=left_image,
            sparse_depth=sparse_depth,
            validity_map=validity_map,
            intrinsics=intrinsics,
            right_image=right_image,
            return_all_outputs=True)
    assert isinstance(outputs, list), 'Expected list output'
    print('    Got {} outputs'.format(len(outputs)))
    print('    OK')

    # 4. Loss computation
    print('[4] Testing compute_loss...')
    model.train()
    output_all = model.forward_depth(
        image=left_image,
        sparse_depth=sparse_depth,
        validity_map=validity_map,
        intrinsics=intrinsics,
        right_image=right_image,
        return_all_outputs=True)

    loss, loss_info = model.compute_loss(
        image0=left_image,
        image1=left_image,
        image2=left_image,
        output_depth0=output_all,
        sparse_depth0=sparse_depth,
        validity_map0=validity_map,
        intrinsics=intrinsics,
        pose0to1=None,
        pose0to2=None,
        supervision_type='unsupervised',
        w_losses={},
        right_image0=right_image)

    print('    Loss: {:.6f}'.format(loss.item()))
    print('    Loss info keys: {}'.format(list(loss_info.keys())))

    # Check loss is finite and positive
    assert torch.isfinite(loss), 'Loss is not finite: {}'.format(loss.item())
    print('    OK')

    # 5. Backward pass
    print('[5] Testing backward pass...')
    loss.backward()
    print('    OK')

    # 6. Save/restore
    print('[6] Testing save/restore...')
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_model(tmpdir, step=100, optimizer_depth=None)

        # Check files were created
        saved_files = os.listdir(tmpdir)
        print('    Saved files: {}'.format(saved_files))
        assert len(saved_files) > 0, 'No files saved'

        # Restore
        restore_paths = [os.path.join(tmpdir, f) for f in saved_files if f.endswith('.pth')]
        step, _, _ = model.restore_model(restore_paths)
        print('    Restored at step: {}'.format(step))
    print('    OK')

    # 7. Parameters
    print('[7] Testing parameters...')
    params = model.parameters_depth()
    n_params = sum(p.numel() for p in params)
    print('    Number of depth parameters: {:,}'.format(n_params))
    print('    OK')

    print()
    print('All tests passed for {}!'.format(model_name))
    print()

    return True


def test_dataloader():
    '''Test stereo dataloader with sample data'''

    print('=' * 60)
    print('Testing: Stereo Dataloader')
    print('=' * 60)

    from stereo_dataloader import parse_kitti_calibration

    calib_path = os.path.join('data', 'kitti_raw_data', '2011_09_26', 'calib_cam_to_cam.txt')
    if os.path.exists(calib_path):
        print('[1] Testing parse_kitti_calibration...')
        intrinsics, baseline = parse_kitti_calibration(calib_path)
        print('    Intrinsics shape: {}'.format(intrinsics.shape))
        print('    fx={:.2f}  fy={:.2f}  cx={:.2f}  cy={:.2f}'.format(
            intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]))
        print('    Baseline: {:.4f} m'.format(baseline))
        assert intrinsics.shape == (3, 3)
        assert baseline > 0
        print('    OK')
    else:
        print('[1] Skipping calibration test (no KITTI data found)')

    train_file = os.path.join('data', 'unos_train_4frames.txt')
    if os.path.exists(train_file) and os.path.exists(os.path.join('data', 'kitti_raw_data')):
        print('[2] Testing build_stereo_datasets_from_unos_file...')
        from stereo_dataloader import build_stereo_datasets_from_unos_file

        dataset = build_stereo_datasets_from_unos_file(
            train_file_path=train_file,
            kitti_raw_root=os.path.join('data', 'kitti_raw_data'))

        print('    Dataset size: {}'.format(len(dataset)))

        if len(dataset) > 0:
            sample = dataset[0]
            print('    Sample keys: {}'.format(list(sample.keys())))
            print('    left_image shape: {}'.format(sample['left_image'].shape))
            print('    right_image shape: {}'.format(sample['right_image'].shape))
            print('    sparse_depth shape: {}'.format(sample['sparse_depth'].shape))
            print('    intrinsics shape: {}'.format(sample['intrinsics'].shape))
            print('    baseline: {:.4f}'.format(sample['baseline']))
        print('    OK')
    else:
        print('[2] Skipping dataset test (no training data found)')

    print()
    print('Dataloader tests passed!')
    print()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    print()

    all_passed = True

    # Test dataloader
    try:
        test_dataloader()
    except Exception as e:
        print('DATALOADER TEST FAILED: {}'.format(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test BridgeDepthFlow (simpler model, test first)
    try:
        test_model('bridgedepthflow', ['stereo'], device)
    except Exception as e:
        print('BRIDGEDEPTHFLOW TEST FAILED: {}'.format(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test UnOS stereo mode
    try:
        test_model('unos', ['stereo'], device)
    except Exception as e:
        print('UNOS TEST FAILED: {}'.format(e))
        import traceback
        traceback.print_exc()
        all_passed = False

    if all_passed:
        print('=' * 60)
        print('ALL SMOKE TESTS PASSED')
        print('=' * 60)
    else:
        print('=' * 60)
        print('SOME TESTS FAILED')
        print('=' * 60)
        sys.exit(1)
