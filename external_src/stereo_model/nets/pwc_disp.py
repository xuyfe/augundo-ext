import torch
import torch.nn as nn
import torch.nn.functional as F

def leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.1)

# since optical flow is 2d line search and stereo depth is 1d serach (since pixels can only move left/right)

class FeaturePyramid(nn.Module):
    def __init__(self):
        super(FeaturePyramid, self).__init__()
        # Input channels: 3 (RGB image)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1)
        self.conv12 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        c1 = leaky_relu(self.conv1(x))
        c2 = leaky_relu(self.conv2(c1))
        c3 = leaky_relu(self.conv3(c2))
        c4 = leaky_relu(self.conv4(c3))
        c5 = leaky_relu(self.conv5(c4))
        c6 = leaky_relu(self.conv6(c5))
        c7 = leaky_relu(self.conv7(c6))
        c8 = leaky_relu(self.conv8(c7))
        c9 = leaky_relu(self.conv9(c8))
        c10 = leaky_relu(self.conv10(c9))
        c11 = leaky_relu(self.conv11(c10))
        c12 = leaky_relu(self.conv12(c11))
        
        return [c2, c4, c6, c8, c10, c12]

class CostVolume(nn.Module):
    def __init__(self, d=4):
        super(CostVolume, self).__init__()
        self.d = d

    def forward(self, feat1, feat2):
        # feat1, feat2: [B, C, H, W]
        B, C, H, W = feat1.size()
        cost_list = []
        
        # padding, left and right
        feat2_padded = F.pad(feat2, (self.d, self.d, 0, 0), mode='constant', value=0)

        for i in range(2 * self.d + 1):
            # slice the padded feature2
            feat2_slice = feat2_padded[:, :, :, i : i + W]
            
            # calculate cost
            cost = torch.mean(feat1 * feat2_slice, dim=1, keepdim=True)
            cost_list.append(cost)
            
        return torch.cat(cost_list, dim=1)

# corresponding to transformer_old
def warp(x, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    
    vgrid = grid + flow

    # scale grid to [-1,1] 
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1) # [B, H, W, 2]
    
    output = F.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda() if x.is_cuda else torch.autograd.Variable(torch.ones(x.size()))
    mask = F.grid_sample(mask, vgrid, align_corners=True)

    # if W==128:
        # print('debug', mask[0,0,0,:])
        
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    
    return output * mask


class OpticalFlowDecoder(nn.Module):
    def __init__(self, in_channels):
        super(OpticalFlowDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128+128, 96, 3, 1, 1)
        self.conv4 = nn.Conv2d(128+96, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(96+64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(64+32, 1, 3, 1, 1) # Output is 1 channel for Disparity (X flow)

    def forward(self, x):
        c1 = leaky_relu(self.conv1(x))
        c2 = leaky_relu(self.conv2(c1))
        c3 = leaky_relu(self.conv3(torch.cat((c1, c2), dim=1)))
        c4 = leaky_relu(self.conv4(torch.cat((c2, c3), dim=1)))
        c5 = leaky_relu(self.conv5(torch.cat((c3, c4), dim=1)))
        flow_x = self.conv6(torch.cat((c4, c5), dim=1))
        
        # In TF code: flow_y = tf.zeros_like(flow_x). PWC-Disp only predicts x.
        flow_y = torch.zeros_like(flow_x)
        flow = torch.cat((flow_x, flow_y), dim=1)
        
        return flow, c5

class ContextNet(nn.Module):
    def __init__(self, in_channels):
        super(ContextNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, 1, 1, dilation=1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 2, dilation=2)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 4, dilation=4)
        self.conv4 = nn.Conv2d(128, 96, 3, 1, 8, dilation=8)
        self.conv5 = nn.Conv2d(96, 64, 3, 1, 16, dilation=16)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1, dilation=1)
        self.conv7 = nn.Conv2d(32, 1, 3, 1, 1, dilation=1)

    def forward(self, x):
        c1 = leaky_relu(self.conv1(x))
        c2 = leaky_relu(self.conv2(c1))
        c3 = leaky_relu(self.conv3(c2))
        c4 = leaky_relu(self.conv4(c3))
        c5 = leaky_relu(self.conv5(c4))
        c6 = leaky_relu(self.conv6(c5))
        flow_x = self.conv7(c6)
        
        flow_y = torch.zeros_like(flow_x)
        flow = torch.cat((flow_x, flow_y), dim=1)
        
        return flow

class PWCDisp(nn.Module):
    def __init__(self):
        super(PWCDisp, self).__init__()
        self.feature_pyramid = FeaturePyramid()
        self.cost_volume = CostVolume(d=4)
        
        # Decoders for each level
        # Input channels calculation:
        # cv (9) + feat (C) + upsampled_flow (2)
        # However, note in TF code `tf.concat([cv5, feature1_5, flow6to5], axis=3)`
        # cv is 2*d + 1 = 9 channels.
        
        # Level 6
        self.decoder6 = OpticalFlowDecoder(9)
        
        # Level 5: cv(9) + feat(128) + flow(2) = 139
        self.decoder5 = OpticalFlowDecoder(9 + 128 + 2)
        
        # Level 4: cv(9) + feat(96) + flow(1) = 106
        self.decoder4 = OpticalFlowDecoder(9 + 96 + 1)
        
        # Level 3: cv(9) + feat(64) + flow(1) = 74
        self.decoder3 = OpticalFlowDecoder(9 + 64 + 1)
        
        # Level 2: cv(9) + feat(32) + flow(1) = 42
        self.decoder2 = OpticalFlowDecoder(9 + 32 + 1)
        
        self.context_net = ContextNet(1 + 32) # flow2 (1) + feat from decoder (32)

    def forward(self, img1, img2, neg=False):
        # Feature Extraction
        f1 = self.feature_pyramid(img1)
        f2 = self.feature_pyramid(img2)
        
        # Unpack features 
        feature1_1, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = f1
        feature2_1, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = f2
        
        # Level 6
        cv6 = self.cost_volume(feature1_6, feature2_6)
        flow6, _ = self.decoder6(cv6)
        if neg: flow6 = -F.relu(-flow6)
        else: flow6 = F.relu(flow6)
            
        # Level 5
        flow6_up = F.interpolate(flow6, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        feat2_5_w = warp(feature2_5, flow6_up) 
        cv5 = self.cost_volume(feature1_5, feat2_5_w)
        flow5, _ = self.decoder5(torch.cat([cv5, feature1_5, flow6_up], dim=1))
        flow5 = flow5 + flow6_up
        if neg: flow5 = -F.relu(-flow5)
        else: flow5 = F.relu(flow5)
            
        # Level 4
        flow5_up = F.interpolate(flow5, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        feat2_4_w = warp(feature2_4, flow5_up)
        cv4 = self.cost_volume(feature1_4, feat2_4_w)
        flow5_up_x = flow5_up[:, 0:1, :, :]
        flow4, _ = self.decoder4(torch.cat([cv4, feature1_4, flow5_up_x], dim=1))
        flow4 = flow4 + flow5_up
        if neg: flow4 = -F.relu(-flow4)
        else: flow4 = F.relu(flow4)
        
        # Level 3
        flow4_up = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        feat2_3_w = warp(feature2_3, flow4_up)
        cv3 = self.cost_volume(feature1_3, feat2_3_w)
        flow4_up_x = flow4_up[:, 0:1, :, :]
        flow3, _ = self.decoder3(torch.cat([cv3, feature1_3, flow4_up_x], dim=1))
        flow3 = flow3 + flow4_up
        if neg: flow3 = -F.relu(-flow3)
        else: flow3 = F.relu(flow3)

        # Level 2
        flow3_up = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
        feat2_2_w = warp(feature2_2, flow3_up)
        cv2 = self.cost_volume(feature1_2, feat2_2_w)
        flow3_up_x = flow3_up[:, 0:1, :, :]
        flow2_raw, feat_det = self.decoder2(torch.cat([cv2, feature1_2, flow3_up_x], dim=1))
        flow2_raw = flow2_raw + flow3_up
        if neg: flow2_raw = -F.relu(-flow2_raw)
        else: flow2_raw = F.relu(flow2_raw)
            
        # Context Net
        flow2_raw_x = flow2_raw[:, 0:1, :, :]
        # tf.concat([flow2_raw[:, :, :, 0:1], f2], axis=3)
        flow2 = self.context_net(torch.cat([flow2_raw_x, feat_det], dim=1)) + flow2_raw
        if neg: flow2 = -F.relu(-flow2)
        else: flow2 = F.relu(flow2)
            
        # Scale and Upsample to simulate the behavior of the TF `construct_model_pwc_full_disp`
        # TF Code:
        # disp0 = tf.image.resize_bilinear(flow2[:, :, :, 0:1] / (W / (2**2)), [H, W])
        # disp1 = tf.image.resize_bilinear(flow3[:, :, :, 0:1] / (W / (2**3)), [H // 2, W // 2])
        # ...
        
        # In PyTorch, img1.size() is [B, C, H, W]
        # We need H and W from the original input. Since features are passed, we don't naturally have H, W here.
        # However, feature1_2 is 1/4 resolution.
        H_2, W_2 = feature1_2.shape[2], feature1_2.shape[3]
        H_full, W_full = H_2 * 4, W_2 * 4
        
        # flow2 is [B, 2, H/4, W/4]. We only care about X component for disparity.
        # flow2[:, 0:1, :, :]
        
        # disp0 (Full Res)
        # Normalize by feature width: flow_x / W_feature
        # TF: flow2 / (W_full / 4)
        disp0_norm = flow2[:, 0:1, :, :] / (W_full / 4.0)
        disp0 = F.interpolate(disp0_norm, size=(H_full, W_full), mode='bilinear', align_corners=True)
        
        # disp1 (1/2 Res) - Derived from flow3 (1/8 Res)
        # TF: disp1 = resize(flow3 / (W/8), [H/2, W/2])
        # Wait, TF `disp1` uses `flow3` (from level 3, 1/8 res).
        disp1_norm = flow3[:, 0:1, :, :] / (W_full / 8.0)
        disp1 = F.interpolate(disp1_norm, size=(H_full//2, W_full//2), mode='bilinear', align_corners=True)
        
        # disp2 (1/4 Res) - Derived from flow4 (1/16 Res)
        # TF: disp2 = resize(flow4 / (W/16), [H/4, W/4])
        disp2_norm = flow4[:, 0:1, :, :] / (W_full / 16.0)
        disp2 = F.interpolate(disp2_norm, size=(H_full//4, W_full//4), mode='bilinear', align_corners=True)

        # disp3 (1/8 Res) - Derived from flow5 (1/32 Res)
        # TF: disp3 = resize(flow5 / (W/32), [H/8, W/8])
        disp3_norm = flow5[:, 0:1, :, :] / (W_full / 32.0)
        disp3 = F.interpolate(disp3_norm, size=(H_full//8, W_full//8), mode='bilinear', align_corners=True)
        
        if neg:
            return -disp0, -disp1, -disp2, -disp3
        else:
            return disp0, disp1, disp2, disp3


# to get rl_disparities, run forward_depth twice with swapped images (return_all_outputs=True)