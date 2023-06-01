import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import init_parameters
from ..apis import train
from ..neck import NonlinearNeck
from .build import HEAD_REGISTERY
from .utilsfns import *

# All modules which is required to use in config file should register. 
# Register TestHead by '@HEAD_REGISTERY.register' and import in __init__.py (e.g., from .test.py import TestHead)
# Then you can set Head['type'] = "TestHead" in config file to use this module.
@HEAD_REGISTERY.register
class TestHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, proj_layers=2, initial=dict()):
        super().__init__()
        
        self.proj = None
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
        '''
        All registered module of backbone, neck, and head should implement init_weights function.
        We provide init_parameters for quick init. Details can be found in modules/utils/initializer.py
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_parameters(m, **self.initial)

            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        '''
        The return value of forward methods in backbone and neck will be processed in structure.
        Details Model Structure can be seen in modules/structure/ImgModel.py:Model.

        For head modules, the return value of forward methods is a dict.
        One can put tensor in dict to log this value.
        The training process will get outputs['loss'] to backward the gradients.
        '''
        outputs = {}

        def __cal_sim(z1, z2):
            # Here we put the cosin_similarity of z1 and z2 into outputs. The log name of this value is pre-sim.
            outputs['pre-sim'] = F.cosine_similarity(z1, z2, dim=-1).mean()
            z1 = self.proj(z1)
            z2 = self.proj(z2)

            p1 = self.pred(z1)
            p2 = self.pred(z2)

            q1 = z1.detach()
            q2 = z2.detach()
            
            # Here we put the l2-norm of p1 and q1 into outputs.
            outputs['p_norm'] = p1.norm(dim=1).mean()
            outputs['q_norm'] = q1.norm(dim=1).mean()

            loss_sim = D(p1, q2) / 2 + D(p2, q1) / 2
            return loss_sim

        loss_sim = __cal_sim(x1, x2)

        outputs['sim'] = loss_sim

        # We use loss_sim as the final loss. The backward will be applied on outputs['loss']
        outputs['loss'] = loss_sim
        
        return outputs