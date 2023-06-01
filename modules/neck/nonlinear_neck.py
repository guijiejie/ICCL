import torch
import torch.nn as nn
import torchvision
import math
from mmcv.cnn import (normal_init, constant_init, kaiming_init)
from ..utils import init_parameters

from .build import NECK_REGISTERY

def build_linear_layer(in_features, out_features, norm=False, bias=True, relu=True):
    ret = [nn.Linear(in_features, out_features, bias=bias and not norm)]
    if norm:
        ret += [nn.BatchNorm1d(out_features)]
    if relu:
        ret += [nn.ReLU(inplace=True)]

    return ret

@NECK_REGISTERY.register
class NonlinearNeck(nn.Module):
    def __init__(self, layer_info, avgpool=False, initial=dict()):
        super().__init__()
        layer_list = []

        self.out_channels = None
        for l in layer_info:
            layer_list.extend(build_linear_layer(**l))
            self.out_channels = l['out_features']

        self.layer = nn.Sequential(*layer_list)
        self.initial = initial
        self.avgpool = None
        if avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                init_parameters(m, **self.initial)

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.avgpool is not None:
            x = self.avgpool(x).view(x.size(0), -1)

        x = self.layer(x)
        return x

class NonlinearNeckMultiView(nn.Module):
    def __init__(self, layer_info, layer_info_text, avgpool=False, preRelu=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        layer_list = []
        for l in layer_info:
            layer_list.extend(self.__build_layer(**l))
        self.imglayer = nn.Sequential(*layer_list)

        layer_list = []
        for l in layer_info_text:
            layer_list.extend(self.__build_layer(**l))
        self.textlayer = nn.Sequential(*layer_list)

        if avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = None
        
        self.preRelu = preRelu
    
    def __build_layer(self, in_features, out_features, linear_cfg=dict(type="linear"), norm=False, relu=False):
        ret = build_linear_layer(linear_cfg, in_features, out_features, norm)
        if relu:
            ret.append(self.relu)
        return ret
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, PWSLinear)):
                m.reset_parameters()

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, text=None):
        if self.preRelu:
            x = self.relu(x)

        if self.avgpool is not None:
            x = self.avgpool(x).view(x.size(0), -1)

        x = self.imglayer(x)
        if text is None:
            return x
        else:
            text_features = self.textlayer(text)
            return x, text_features
