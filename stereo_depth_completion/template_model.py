import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add the necessary paths for your model
# Note that if you import your model, your code should be stored in external_src/stereo_depth_completion/
# sys.path.insert(0, os.path.join('external_src', 'stereo_depth_completion', 'YourModel'))
# TODO: Import necessary classes or packages for your model


class TemplateStereoModel(object):
    '''
    Template for interfacing a new stereo depth estimation model with the
    stereo AugUndo pipeline.

    The stereo AugUndo training loop calls the following methods:
        - forward_stereo_disparity(left, right) : predict multi-scale disparity
        - forward_temporal_flow(image_t, image_t1) : predict temporal optical flow (optional)
        - forward(left_t, right_t, left_t1, right_t1) : full forward pass for inference
        - compute_loss(output, batch) : compute model-native loss (used in non-AugUndo mode)
        - parameters() : return all trainable parameters
        - train() / eval() : set training/evaluation mode
        - to(device) : move model to device
        - data_parallel() : wrap model in DataParallel
        - restore_model(path, optimizer) : load checkpoint
        - save_model(path, step, optimizer) : save checkpoint

    Arg(s):
        input_height : int
            input image height
        input_width : int
            input image width
        num_scales : int
            number of multi-scale output levels (typically 4)
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 input_height=256,
                 input_width=512,
                 num_scales=4,
                 device=torch.device('cuda')):

        self.input_height = input_height
        self.input_width = input_width
        self.num_scales = num_scales
        self.device = device

        # TODO: Instantiate your stereo depth estimation model
        # self.net = YourStereoNetwork(...)

        # TODO: Move your model to device
        # self.net = self.net.to(device)

    def forward_stereo_disparity(self, left, right):
        '''
        Predict stereo disparity for a single left-right pair.

        This is the primary method called during stereo AugUndo training.
        It runs inference on the augmented stereo pair and returns multi-scale
        disparity predictions. The training loop then undoes the geometric
        augmentation on these predictions before computing the loss in the
        original coordinate frame.

        Arg(s):
            left : torch.Tensor[float32]
                N x 3 x H x W left image
            right : torch.Tensor[float32]
                N x 3 x H x W right image

        Returns:
            disp_left : list[torch.Tensor[float32]]
                num_scales tensors each (N, 1, H_s, W_s) positive normalised left disparity
                (disparity as a fraction of image width, ranging [0, 1])
            disp_right : list[torch.Tensor[float32]]
                num_scales tensors each (N, 1, H_s, W_s) positive normalised right disparity
            flow_left : list[torch.Tensor[float32]] or None
                (Optional) num_scales tensors each (N, 2, H_s, W_s) full 2-channel normalised
                flow for left view (ch0 = horizontal disparity, ch1 = vertical).
                Used for flow-aware smoothness loss. Return None if not applicable.
            flow_right : list[torch.Tensor[float32]] or None
                (Optional) num_scales tensors each (N, 2, H_s, W_s) full 2-channel normalised
                flow for right view. Return None if not applicable.
        '''

        # TODO: Run your model on the stereo pair and extract multi-scale disparity
        # Example:
        #   features_left = self.net.encoder(left)
        #   features_right = self.net.encoder(right)
        #   raw_disp = self.net.decoder(features_left, features_right)

        # TODO: Convert raw model output to positive normalised disparity
        # Normalised means disparity is expressed as a fraction of image width [0, 1].
        # If your model outputs pixel disparity, divide by image width.
        # If your model outputs inverse depth, convert to disparity using baseline & focal length.

        disp_left = [None] * self.num_scales   # TODO: Replace with actual predictions
        disp_right = [None] * self.num_scales  # TODO: Replace with actual predictions

        # Optional: 2-channel flow for smoothness computation
        # Set to None if your model does not produce 2-channel flow
        flow_left = None
        flow_right = None

        return disp_left, disp_right, flow_left, flow_right

    def forward_temporal_flow(self, image_t, image_t1):
        '''
        Predict temporal 2D optical flow from image_t to image_t1.

        This method is OPTIONAL. It is called during training only if
        temporal_loss_weight > 0. The predicted flow is used to compute a
        temporal photometric consistency loss by backward-warping image_t1
        to reconstruct image_t.

        If your model does not support temporal flow prediction, you can
        either leave this method as-is (returning zeros) and set
        temporal_loss_weight=0, or raise NotImplementedError.

        Arg(s):
            image_t : torch.Tensor[float32]
                N x 3 x H x W image at time t
            image_t1 : torch.Tensor[float32]
                N x 3 x H x W image at time t+1

        Returns:
            list[torch.Tensor[float32]] :
                num_scales tensors each (N, 2, H_s, W_s) normalized flow
                (ch0 = horizontal as fraction of width, ch1 = vertical as fraction of height)
        '''

        # TODO: Implement temporal flow prediction if your model supports it
        # If not supported, set temporal_loss_weight=0 in the training config
        raise NotImplementedError(
            'Temporal flow prediction not implemented for this model. '
            'Set temporal_loss_weight=0 to disable temporal loss.')

    def forward(self, image_left_t, image_right_t, image_left_t1, image_right_t1):
        '''
        Full forward pass for inference/evaluation.

        This method is called during evaluation (run_stereo_depth_completion.py).
        It should produce the complete model output including all disparity
        predictions needed for evaluation metrics.

        Arg(s):
            image_left_t : torch.Tensor[float32]
                N x 3 x H x W left image at time t
            image_right_t : torch.Tensor[float32]
                N x 3 x H x W right image at time t
            image_left_t1 : torch.Tensor[float32]
                N x 3 x H x W left image at time t+1
            image_right_t1 : torch.Tensor[float32]
                N x 3 x H x W right image at time t+1

        Returns:
            dict : dictionary containing model outputs, must include at minimum:
                'disp_left' : list[torch.Tensor] - multi-scale left disparity (normalised)
                'disp_right' : list[torch.Tensor] - multi-scale right disparity (normalised)
        '''

        # TODO: Implement full forward pass
        # For many models, this is equivalent to forward_stereo_disparity
        # but may include additional outputs (e.g., optical flow, pose)
        disp_left, disp_right, flow_left, flow_right = \
            self.forward_stereo_disparity(image_left_t, image_right_t)

        output = {
            'disp_left': disp_left,
            'disp_right': disp_right,
        }

        # TODO: Add additional outputs as needed (flow, depth, etc.)

        return output

    def compute_loss(self, output, batch):
        '''
        Computes the model-native loss.

        NOTE: In stereo AugUndo training mode, the loss is computed externally
        by stereo_losses.compute_stereo_loss() in the original (un-augmented)
        coordinate frame. This method is used only when training WITHOUT
        AugUndo (i.e., the model's own training objective).

        Arg(s):
            output : dict
                Output from forward()
            batch : dict
                Batch data with keys like 'image_left_t', 'image_right_t', etc.

        Returns:
            torch.Tensor[float32] : total scalar loss
            dict[str, float] : dictionary of individual loss components for logging
        '''

        # TODO: Implement your model's native loss computation
        loss = None
        loss_info = {
            'loss': 0.0,
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of all trainable parameters in the model.

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        # TODO: Return your model's parameters
        # return list(self.net.parameters())
        return []

    def train(self):
        '''
        Sets model to training mode.
        '''

        # TODO: Set your model into training mode
        # self.net.train()
        pass

    def eval(self):
        '''
        Sets model to evaluation mode.
        '''

        # TODO: Set your model into evaluation mode
        # self.net.eval()
        pass

    def to(self, device):
        '''
        Move model to a device.

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device

        # TODO: Move your model to device
        # self.net = self.net.to(device)

    def data_parallel(self):
        '''
        Wraps model in DataParallel for multi-GPU training.
        '''

        # TODO: Wrap your network in DataParallel
        # self.net = torch.nn.DataParallel(self.net)
        pass

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint.

        Arg(s):
            restore_path : str
                path to model checkpoint
            optimizer : torch.optim
                optimizer to restore

        Returns:
            int : training step/iteration from checkpoint
            torch.optim : restored optimizer (or None)
        '''

        # TODO: Implement checkpoint loading for your model
        # Example:
        #   checkpoint = torch.load(restore_path, map_location=self.device)
        #   self.net.load_state_dict(checkpoint['state_dict'])
        #   if optimizer is not None and 'optimizer' in checkpoint:
        #       optimizer.load_state_dict(checkpoint['optimizer'])
        #   step = checkpoint.get('step', 0)
        step = 0

        return step, optimizer

    def save_model(self, checkpoint_path, step, optimizer=None):
        '''
        Save model checkpoint.

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer to save
        '''

        # TODO: Implement checkpoint saving for your model
        # Example:
        #   os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        #   state = {
        #       'step': step,
        #       'state_dict': self.net.state_dict(),
        #   }
        #   if optimizer is not None:
        #       state['optimizer'] = optimizer.state_dict()
        #   torch.save(state, checkpoint_path)
        pass
