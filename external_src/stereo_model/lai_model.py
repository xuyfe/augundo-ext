import os
import torch
import torch.nn as nn
from .nets.lai_stereo import pwc_dc_net

class LaiModel(object):
    '''
    Wrapper for the BridgeDepthFlow PWC-Net model (Lai et al.)
    '''

    def __init__(self,
                 dataset_name,
                 network_modules,
                 min_predict_depth,
                 max_predict_depth,
                 device=torch.device('cuda')):

        self.model_name = 'lai_stereo'
        self.device = device
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        
        # Instantiate model
        self.model = pwc_dc_net()
        self.model.to(self.device)

    def forward_depth(self, image, sparse_depth, validity_map, intrinsics, return_all_outputs=False):
        '''
        Forwards stereo pair through network.
        Expects 'image' to be N x 6 x H x W (concatenated left and right images).
        '''
        
        # The model expects [B, 6, H, W]
        if image.shape[1] == 3:
            # If we only got 3 channels, we might be in trouble unless sparse_depth is actually the right image?
            # Or maybe we just can't run. 
            # Assuming the inputs are correct for a stereo model wrapper.
            # If strictly depth completion, this might fail.
            # But usually for stereo tasks reused as depth completion, we pack L+R into image.
            pass

        # Forward pass
        # returns [flow0, flow1, flow2, flow3]
        outputs = self.model(image)
        
        # High resolution flow (scale 0)
        flow_est = outputs[0]
        
        # Calculate disparity: -flow_x
        # flow is [B, 2, H, W]. flow[:,0] is x-component (u).
        disparity = -flow_est[:, 0:1, :, :]
        
        # Convert to depth
        # depth = fx * baseline / disparity
        # Check if intrinsics provided
        if intrinsics is not None:
             # intrinsics: [B, 3, 3]
             fx = intrinsics[:, 0, 0].view(-1, 1, 1, 1)
             
             # Hardcoded baseline for KITTI if not provided? 
             # Ideally should be passed in. Assuming 0.54m for now as standard.
             baseline = 0.54 
             
             depth = (fx * baseline) / (disparity + 1e-8)
             
             # Clamp depth
             depth = torch.clamp(depth, self.min_predict_depth, self.max_predict_depth)
             
             # Mask out invalid disparities (<=0)
             depth[disparity <= 0] = 0
             
             output = depth
        else:
             # Just return disparity if no intrinsics (though function is named forward_depth)
             output = disparity

        if return_all_outputs:
            return [output]
        else:
            return output

    def forward_pose(self, image0, image1):
        # This model does not predict pose
        return None

    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     sparse_depth0,
                     validity_map0,
                     intrinsics,
                     pose0to1,
                     pose0to2,
                     w_losses):
        # Simple placeholder for loss computation
        return None, {}

    def parameters_depth(self):
        return self.model.parameters()

    def parameters_pose(self):
        return None

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.to(device)

    def data_parallel(self):
        self.model = nn.DataParallel(self.model)

    def restore_model(self, restore_path, optimizer=None):
        if restore_path is not None:
            checkpoint = torch.load(restore_path)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        return optimizer

    def save_model(self, checkpoint_path, step, optimizer):
        state = {'state_dict': self.model.state_dict()}
        torch.save(state, checkpoint_path)
