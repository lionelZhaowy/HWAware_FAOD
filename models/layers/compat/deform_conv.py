"""FAOD's unmodulated MMCV DeformConv2d contract on torchvision 0.20.

MMCV 2.1 reference: mmcv/ops/deform_conv.py (Apache-2.0, OpenMMLab).
Keeps parameter names/shape, ReLU Kaiming initialization, no bias, interleaved
(dy, dx) offsets and offset-controlled AMP dtype. This is NOT modulated DCNv2.
No custom CUDA build or MMCV runtime dependency is needed.
"""
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deform_groups=1,
                 bias=False, im2col_step=32):
        super().__init__()
        if bias:
            raise ValueError('The legacy FAOD operator has no bias')
        if in_channels % groups or out_channels % groups or in_channels % deform_groups:
            raise ValueError('Invalid convolution/deformable groups')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = _pair(kernel_size), _pair(stride)
        self.padding, self.dilation = _pair(padding), _pair(dilation)
        self.groups, self.deform_groups = groups, deform_groups
        self.im2col_step = im2col_step  # Legacy attribute; torchvision manages its own batching.
        self.transposed, self.output_padding = False, (0,)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x, offset):
        expected = 2 * self.deform_groups * self.kernel_size[0] * self.kernel_size[1]
        if offset.ndim != 4 or offset.shape[1] != expected:
            raise ValueError(f'Expected {expected} interleaved offset channels')
        # Match MMCV's offset-driven dtype conversion during mixed precision.
        # Disable autocast here, so torchvision does not silently choose another dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return deform_conv2d(x.to(offset.dtype), offset, self.weight.to(offset.dtype),
                                 bias=None, stride=self.stride, padding=self.padding,
                                 dilation=self.dilation, mask=None)
