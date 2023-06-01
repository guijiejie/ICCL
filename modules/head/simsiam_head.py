import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import init_parameters
from ..apis import train
from ..neck import NonlinearNeck
from .build import HEAD_REGISTERY
from .utilsfns import *

@HEAD_REGISTERY.register
class SimSiamHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, proj_layers=2, initial=dict()):
        super().__init__()
        
        self.proj = None
        if proj_layers > 0:
            proj_list = []
            for _ in range(proj_layers-1):
                proj_list.append(nn.Linear(in_channels, out_channels, bias=False))
                proj_list.append(nn.BatchNorm1d(out_channels))
                proj_list.append(nn.ReLU(inplace=True))
                in_channels = out_channels

            proj_list.append(nn.Linear(out_channels, out_channels, bias=False))
            proj_list.append(nn.BatchNorm1d(out_channels, affine=False))
            self.proj = nn.Sequential(*proj_list)

        self.pred = nn.Sequential(nn.Linear(out_channels, hidden_channels, bias=False),
                                nn.BatchNorm1d(hidden_channels),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_channels, out_channels))
        
        self.initial = initial

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_parameters(m, **self.initial)

            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        outputs = {}

        def __cal_sim(z1, z2):
            if self.proj is not None:
                outputs['pre-sim'] = F.cosine_similarity(z1, z2, dim=-1).mean()
                z1 = self.proj(z1)
                z2 = self.proj(z2)

            p1 = self.pred(z1)
            p2 = self.pred(z2)

            q1 = z1.detach()
            q2 = z2.detach()
            
            outputs['p_norm'] = p1.norm(dim=1).mean()
            outputs['q_norm'] = q1.norm(dim=1).mean()

            loss_sim = D(p1, q2) / 2 + D(p2, q1) / 2
            return loss_sim

        if isinstance(x1, torch.Tensor):
            loss_sim = __cal_sim(x1, x2)
        else:
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

        outputs['sim'] = loss_sim
        outputs['loss'] = loss_sim
        
        return outputs