"""
BridgeDepthFlow integration for augundo-ext.

Adapted from BridgeDepthFlow (https://github.com/sxyu/BridgeDepthFlow)
which performs self-supervised monocular depth estimation from stereo pairs.

This package provides:
- MonodepthNet: The encoder-decoder network for stereo disparity estimation
- BridgeDepthFlowModel: A clean nn.Module wrapper with forward and training loss
- Loss utilities: SSIM, smoothness, occlusion masking, and warping functions
"""

from .monodepth_model import MonodepthNet
from .bridge_depth_flow import BridgeDepthFlowModel

__all__ = ["MonodepthNet", "BridgeDepthFlowModel"]
