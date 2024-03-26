import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import *


class ResNet(torchvision.models.ResNet):
    def __init__(
        self, 
        block: type[BasicBlock] | type[Bottleneck], 
        layers: torch.List[int], 
        num_classes: int = 1000, 
        zero_init_residual: bool = False, 
        groups: int = 1, 
        width_per_group: int = 64, 
        replace_stride_with_dilation: torch.List[bool] | None = None, 
        norm_layer: Callable[..., nn.Module] | None = None,
        in_channels: int = 3
    ) -> None:
        super(ResNet, self).__init__(
            block, 
            layers, 
            num_classes, 
            zero_init_residual, 
            groups, 
            width_per_group, 
            replace_stride_with_dilation, 
            norm_layer
        )
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
