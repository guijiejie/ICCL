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
class MoCoHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=2048, out_channels=128, moco_buffer=dict(queue_size=4096), tau=0.2, initial=dict()):
        super().__init__()

        self.pred = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        if moco_buffer is None:
            self.queue1 = None
            self.queue2 = None
        else:
            self.add_module("queue1", BufferQueue(out_channels, **moco_buffer))
            self.add_module("queue2", BufferQueue(out_channels, **moco_buffer))
        
        self.tau = tau
        self.initial = initial

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
    
    def __loss(self, p1, q2, queue, outputs=None):
        T = self.tau
        p1 = nn.functional.normalize(p1, dim=1, eps=1e-6)

        if torch.distributed.is_initialized():
            bs_per_gpu = p1.size(0)
            q2 = concat_all_gather(q2)

            label = bs_per_gpu * torch.distributed.get_rank() + torch.arange(bs_per_gpu, dtype=torch.long).cuda()
        else:
            label = torch.arange(p1.size(0), dtype=torch.long).cuda()
        
        if queue is not None:
            q2 = queue(q2)

        q2 = nn.functional.normalize(q2, dim=1, eps=1e-6)

        sim = p1 @ q2.t()

        loss = 2 * T * cross_entropy_shootinfs(sim / T, label)

        if outputs is not None:
            outputs['sim'] = sim[torch.arange(p1.size(0)), label].mean()

        return loss

    def forward(self, p, q):
        outputs = {}
        outputs['loss'] = 0.0

        p1 = self.pred(p[0])
        p2 = self.pred(p[1])

        q1, q2 = q

        outputs['loss'] = self.__loss(p1, q2, self.queue1, outputs) + self.__loss(p2, q1, self.queue2)
        outputs['moco'] = outputs['loss']
        return outputs 
    
    def evaluate(self, x):
        pass
