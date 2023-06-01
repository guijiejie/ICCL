from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from ..utils import init_parameters
from ..apis import train
from .build import HEAD_REGISTERY
from .utilsfns import *

@HEAD_REGISTERY.register
class BYOLHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=2048, out_channels=128, initial=dict()):
        super().__init__()

        self.pred = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels, bias=False),
        )

        self.initial = initial

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

    def __loss(self, p1, q2):
        pred_norm = nn.functional.normalize(p1, dim=1)
        target_norm = nn.functional.normalize(q2, dim=1)

        loss = ((pred_norm - target_norm) ** 2).sum(dim=-1).mean()
        return loss

    def forward(self, p, q, select_num=None):
        p1, p2 = p[0:2]
        q1, q2 = q[0:2]
        outputs = {}
        outputs['loss'] = 0.0

        if isinstance(p1, list):
            CLIP_NUM = len(p1)
            select_num = CLIP_NUM if select_num is None else select_num

            assert p1[0].requires_grad

            for i in range(select_num):
                pred1 = self.pred(p1[i])
                pred2 = self.pred(p2[i])

                for j in range(CLIP_NUM):
                    outputs['loss'] += (self.__loss(pred1, q2[j]) + self.__loss(pred2, q1[j]))
            
            outputs['loss'] /= (select_num * CLIP_NUM)
        else:
            p1 = self.pred(p1)
            p2 = self.pred(p2)

            outputs['loss'] = self.__loss(p1, q2) + self.__loss(p2, q1)

        outputs['sim'] = F.cosine_similarity(p1, p2, dim=-1).mean()

        return outputs 
    
    def evaluate(self, x):
        pass
