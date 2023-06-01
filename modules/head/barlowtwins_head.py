import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import init_parameters
from ..apis import train
from ..neck import NonlinearNeck

from .build import HEAD_REGISTERY
from .utilsfns import *

class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = F.normalize(x, dim=0)
        return x

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

@HEAD_REGISTERY.register
class BarlowTwinsHead(nn.Module):
    def __init__(self, in_channels=2048, out_channels=8192, proj_layers=3, lambd=0.005, crossgpu=True, initial=dict()):
        super().__init__()
        self.linear_cfg = dict(type="linear")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        proj_list = []
        for _ in range(proj_layers-1):
            proj_list.append(nn.Linear(in_channels, out_channels, bias=False))
            proj_list.append(nn.BatchNorm1d(out_channels))
            proj_list.append(nn.ReLU(inplace=True))

            in_channels = out_channels
        proj_list.append(nn.Linear(out_channels, out_channels, bias=False))
        proj_list.append(nn.BatchNorm1d(out_channels, affine=False))
        # proj_list.append(L2Norm())
        self.proj = nn.Sequential(*proj_list)

        self.lambd = lambd
        self.crossgpu = crossgpu

        self.initial = initial
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                init_parameters(m, **self.initial)

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        def __cal_sim(p1, p2):
            c = p1.t() @ p2

            if self.crossgpu and torch.distributed.is_initialized():
                total_batch_size = torch.distributed.get_world_size() * p1.size(0)
                c.div_(total_batch_size)
                torch.distributed.all_reduce(c)
            else:
                c.div_(p1.size(0))

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + self.lambd * off_diag
            return loss

        outputs = {}

        if isinstance(x1, torch.Tensor):
            x1 = self.proj(x1)
            x2 = self.proj(x2)
            loss_sim = __cal_sim(x1, x2)
        else:
            x1 = list(map(lambda xi:self.proj(xi), x1))
            x2 = list(map(lambda xi:self.proj(xi), x2))
            loss_sim = 0
            cnts = 0
            if len(x1) > 1:
                for i in range(len(x1)-1):
                    for j in range(i+1, len(x1)):
                        loss_sim += __cal_sim(x1[i], x1[j])
                        cnts += 1
            
            for xi in x1:
                for yi in x2:
                    loss_sim += __cal_sim(xi, yi)
                    cnts += 1
            
            loss_sim = loss_sim / cnts

        outputs['barlowtwins'] = loss_sim
        outputs['loss'] = loss_sim
        
        return outputs