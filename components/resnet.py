from opacus.validators import ModuleValidator
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from typing import *


class ResNet(torchvision.models.ResNet):
    def __init__(
        self, 
        block: type[BasicBlock] | type[Bottleneck], 
        layers: List[int], 
        num_classes: int = 1000, 
        zero_init_residual: bool = False, 
        groups: int = 1, 
        width_per_group: int = 64, 
        replace_stride_with_dilation: List[bool] | None = None, 
        norm_layer: Callable[..., nn.Module] | None = None,
        in_channels: int = 3
    ) -> None:
        """
        ResNet model from torchvision.models.resnet
        
        :param block: Type of block to build the model with
        :type block: type[BasicBlock] | type[Bottleneck]
        :param layers: Number of layers of each block in the model
        :type layers: torch.List[int]
        :param num_classes: Number of output classes, defaults to 1000
        :type num_classes: int, optional
        :param zero_init_residual: Whether to zero initialize the last BN in each residual block, defaults to False
        :type zero_init_residual: bool, optional
        :param groups: Number of groups to use in the 3x3 convolutions, defaults to 1
        :type groups: int, optional
        :param width_per_group: Width of each group in the 3x3 convolutions, defaults to 64
        :type width_per_group: int, optional
        :param replace_stride_with_dilation: Whether to replace stride with dilation in the first convolution of each block, defaults to None
        :type replace_stride_with_dilation: torch.List[bool] | None, optional
        :param norm_layer: What type of normalization layer to use, defaults to None
        :type norm_layer: Callable[..., nn.Module] | None, optional
        :param in_channels: Number of channels in input data, defaults to 3
        :type in_channels: int, optional
        """
        super(ResNet, self).__init__(
            block, 
            layers, 
            num_classes, 
            zero_init_residual, 
            groups, 
            width_per_group, 
            replace_stride_with_dilation, 
            norm_layer,
        )
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.prefc_norm = nn.Identity() # this will be modified by the CR mechanism later

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.prefc_norm(x)
        x = self.fc(x)
        
        return x


def clean_state_dict(loaded_state: Mapping[str, Any], curr_state: Mapping[str, Any]) -> Mapping[str, Any]:
    state_dict = {}
    for k in loaded_state.keys():
        if 'fc' in k.split('.') or 'prefc_norm' in k.split('.'):
            state_dict[k] = curr_state[k]
        else:
            state_dict[k] = loaded_state[k]
    
    return state_dict


def build_resnet50(num_classes: int, in_channels: int, pretrained: bool = True) -> ResNet:
    """
    Build ResNet50 model

    :param num_classes: Number of output classes
    :type num_classes: int
    :param in_channels: Number of channels in input data
    :type in_channels: int
    :return: ResNet50 model
    :rtype: ResNet
    """
    model = ResNet(
        block=Bottleneck, 
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes
    )
    
    if pretrained:
        loaded_state = ResNet50_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
        curr_state = model.state_dict()
        state_dict = clean_state_dict(loaded_state, curr_state)
        model.load_state_dict(state_dict)
    
    return model


def build_resnet18(num_classes: int, in_channels: int, pretrained: bool = True, use_differential_privacy: bool = False) -> ResNet:
    """
    Build ResNet18 model

    :param num_classes: Number of output classes
    :type num_classes: int
    :param in_channels: Number of channels in input data
    :type in_channels: int
    :return: ResNet18 model
    :rtype: ResNet
    """
    
    model = ResNet(
        block=BasicBlock, 
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        norm_layer=nn.BatchNorm2d
    )
    
    if pretrained:
        loaded_state = ResNet18_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
        curr_state = model.state_dict()
        state_dict = clean_state_dict(loaded_state, curr_state)
        model.load_state_dict(state_dict)
    
    if use_differential_privacy:
        model = ModuleValidator.fix(model)
    
    return model
