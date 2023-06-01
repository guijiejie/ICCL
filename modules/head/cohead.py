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
class CoTrainHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, initial=dict()):
        super().__init__()

        self.pred1 = nn.Sequential(nn.Linear(in_channels, hidden_channels, bias=False),
                                nn.BatchNorm1d(hidden_channels),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_channels, out_channels))

        self.pred2 = nn.Sequential(nn.Linear(in_channels, hidden_channels, bias=False),
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
        assert isinstance(x1, list)
        outputs = {}

        def __cal_sim(z1, z2, mlp1, mlp2):
            p1 = mlp1(z1)
            p2 = mlp2(z2)

            q1 = z1.detach()
            q2 = z2.detach()

            loss_sim = D(p1, q2) / 2 + D(p2, q1) / 2
            return loss_sim

        if len(x1) == 1:
            loss_sim = __cal_sim(x1[0], x2[0], self.pred1, self.pred2)
        else:
            loss_sim = 0

            loss_sim = __cal_sim(x1[0], x1[1], self.pred1, self.pred1) + __cal_sim(x2[0], x2[1], self.pred2, self.pred2)

            for i in range(2):
                for j in range(2):
                    if i == j:
                        loss_sim += __cal_sim(x1[i], x2[j], self.pred1, self.pred2)
           
            loss_sim = loss_sim / 4

        outputs['sim'] = loss_sim
        outputs['loss'] = loss_sim
        
        return outputs