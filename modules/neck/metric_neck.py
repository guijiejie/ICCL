import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn import (build_norm_layer, constant_init)

from mmcv.runner import load_checkpoint
from .nonlinear_neck import NonlinearNeck
from ..utils import init_parameters
from .build import NECK_REGISTERY

@NECK_REGISTERY.register
class MetricNeck(nn.Module):
    def __init__(self, layer_info, prototypes, initial=dict()):
        super().__init__()

        self.proj = NonlinearNeck(layer_info)

        self.weight = nn.Parameter(torch.Tensor(self.proj.out_channels, prototypes))
        
        self.initial = initial

    def init_weights(self, pretrained=None):
        nn.init.kaiming_uniform_(self.weight, a=1)

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
        outs = self.proj(x)
        outs = F.normalize(outs, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        outs = outs @ weight
        return outs

