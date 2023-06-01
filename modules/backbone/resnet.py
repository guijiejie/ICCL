import torchvision
import torch
import torch.nn as nn

from ..utils import init_parameters
from .build import BACKBONE_REGISTERY

@BACKBONE_REGISTERY.register("ResNetRaw")
class ResNetRaw(nn.Module):
    def __init__(self, depth=50, zero_init_residual=True, frozen=False, initial=dict()):
        super().__init__()
        depth2model = {
            50:torchvision.models.resnet50,
            18:torchvision.models.resnet18,
            34:torchvision.models.resnet34,
        }
        self.model = depth2model[depth](num_classes=2048, zero_init_residual=zero_init_residual)
        self.model.fc = nn.Identity()
        self.frozen = frozen
        self.initial = initial
        self.zero_init_residual = zero_init_residual
        self._freeze_stages()
    
    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen:
            for m in self.model.modules():
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_parameters(m, **self.initial)

    def forward(self, x):
        return self.model(x)
