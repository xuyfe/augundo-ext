import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Ensure nets is reachable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nets.pwc_disp import PWCDisp

class PWCModel(object):
    def __init__(self,
                 dataset_name=None,
                 network_modules=[],
                 min_predict_depth=-1.0,
                 max_predict_depth=-1.0,
                 device=torch.device('cuda')):
        
        self.model = PWCDisp()
        self.device = device
        self.model.to(device)
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        
        # Default baseline for KITTI if needed, but we output depth=1/disp
        self.baseline = 0.54 

    # if return_all_outputs is True, run the model twice with swapped images to get rl_disparities
    def forward_depth(self, image, sparse_depth, validity_map, intrinsics, return_all_outputs=False):
        # Image Handling: Check if 6 channels (Stereo L+R concatenated)
        img1 = image
        img2 = None
        
        if image.shape[1] == 6:
            img1 = image[:, :3, :, :]
            img2 = image[:, 3:, :, :]
        elif image.shape[1] == 3:
            # Maybe sparse_depth is the second image?
            # Or maybe we are in a mode where img2 is unavailable?
            # PWCDisp NEEDS 2 images.
            # If provided 3 channels, we cannot run stereo matching. 
            # We assume inputs are correct.
             pass
        
        if img2 is None:
             # Fallback: maybe sparse_depth is passed as img2 if it has 3 channels?
            if sparse_depth is not None and sparse_depth.shape[1] == 3:
                img2 = sparse_depth
            else:
                # Critical failure for stereo
                # Return zeros or raise error
                raise ValueError("PWCModel requires stereo input (6 channel image or image+image pair)")

        # Run Model
        # output is now [disp0, disp1, disp2, disp3]
        # disp0 is Full Resolution, Normalized (0-1 relative to width)
        
        # Left-to-Right (Standard)
        # neg=True because traditionally L->R disparity is considered "negative flow" (moving left) 
        # but output is made positive by ReLU.
        lr_disparities = self.model(img1, img2, neg=True) # [L0, L1, L2, L3]
        
        # Finest disparity (Full res)
        disp0_norm = lr_disparities[0]
        
        # Input image size
        H, W = img1.shape[2], img1.shape[3]
        
        # Denormalize to get Pixel Disparity
        disp = disp0_norm * float(W)
        
        # Enforce positive disparity
        disp = F.relu(disp) + 1e-6
        
        # Convert to Depth
        if intrinsics is not None:
             fx = intrinsics[:, 0, 0].view(-1, 1, 1, 1)
             depth = (1.0 / disp) * fx * self.baseline
        else:
             depth = 1.0 / disp
        
        # Clamping
        if self.max_predict_depth > 0:
            depth = torch.clamp(depth, self.min_predict_depth, self.max_predict_depth)
            
        if return_all_outputs:
            # Run Right-to-Left for consistency loss
            rl_disparities = self.model(img2, img1, neg=False) # [R0, R1, R2, R3]
            
            # Pack outputs for loss computation
            # The original TF output was [concat(L0, R0), concat(L1, R1), ...]
            # We should probably return something similar or a dictionary.
            # Let's return a list of concatenated tensors to match the structural expectation
            # if the caller expects the same format as UnDepthflow loss.
            
            combined_outputs = []
            min_disp = 1e-6
            for ltr, rtl in zip(lr_disparities, rl_disparities):
                 # ltr and rtl are [B, 1, H, W]
                 # concat -> [B, 2, H, W]
                 combined = torch.cat([ltr + min_disp, rtl + min_disp], dim=1)
                 combined_outputs.append(combined)
            
            return depth, combined_outputs
        
        return depth
    
    def forward_pose(self, image0, image1):
        # PWC is not a pose net. Return Identity or None?
        # wrapper expects N x 4 x 4
        B = image0.shape[0]
        pose = torch.eye(4).view(1, 4, 4).repeat(B, 1, 1).to(self.device)
        return pose

    def compute_loss(self, **kwargs):
        # Placeholder
        return torch.tensor(0.0).to(self.device), {}
        
    def parameters_depth(self):
        return list(self.model.parameters())

    def parameters_pose(self):
        return []
    
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()
        
    def to(self, device):
        self.device = device
        self.model.to(device)

    def restore_model(self, restore_path, optimizer=None):
        if restore_path is not None:
            checkpoint = torch.load(restore_path, map_location=self.device)
            # Handle if state_dict is inside a key
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        return 0, optimizer

    def save_model(self, checkpoint_path, step, optimizer):
        torch.save({
            'state_dict': self.model.state_dict(),
            'step': step
        }, checkpoint_path)
