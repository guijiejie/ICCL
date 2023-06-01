import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from ..utils import init_parameters
from ..apis import train
from .build import HEAD_REGISTERY
from .utilsfns import *

class BufferQueue(nn.Module):
    def __init__(self, num_features, queue_size):
        super().__init__()
        assert queue_size % world_size == 0
        self.N = queue_size // world_size
        self.C = num_features

        self.register_buffer("queue", torch.rand(self.N, self.C).cuda())

    def forward(self, embedding, x, prototypes):
        with torch.no_grad():
            bs = x.size(0)

            out = x.detach()
            self.queue = nn.functional.normalize(self.queue, dim=1, p=2)
            out = torch.cat((prototypes(self.queue), out))

            self.queue[bs:] = self.queue[:-bs].clone()
            self.queue[:bs] = embedding

        return out

    def extra_repr(self):
        return 'num_features={}, num_sample={}'.format(self.C, self.N)

@HEAD_REGISTERY.register
class SwAVHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=2048, out_channels=128, nmb_prototypes=3000, temperature=0.1, 
                    freeze_prototypes_niters=313, sinkhorn_cfg=dict(iterations=3, epsilon=0.05), queue_size=None, initial=dict()):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_cfg = sinkhorn_cfg
        self.freeze_prototypes_niters = freeze_prototypes_niters

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
                        nn.Linear(in_channels, hidden_channels),
                        nn.BatchNorm1d(hidden_channels),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_channels, out_channels))

        self.prototypes = nn.Linear(out_channels, nmb_prototypes, bias=False)

        self.prototypes.requires_grad_(False)

        self.queue_size = queue_size
        if queue_size is not None:
            for i in range(2):
                self.add_module("queue1_{}".format(i), BufferQueue(num_features=out_channels, queue_size=queue_size))
                self.add_module("queue2_{}".format(i), BufferQueue(num_features=out_channels, queue_size=queue_size))

        self.register_buffer("iteration", torch.zeros(1, dtype=torch.long))
        self.initial = initial

    def __reset_prototypes(self, outputs):
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

            if self.iteration < self.freeze_prototypes_niters:
                outputs["frozen"] = ['prototypes']

    def __forward_head(self, x):
        x = self.proj(x)
        x = nn.functional.normalize(x, dim=1, p=2)

        out = self.prototypes(x)
        return x, out
    
    def __forwardfeature(self, x, basename, idx):
        bs = x.size(0)
        r, z = self.__forward_head(x)
            
        p = shoot_infs(F.log_softmax(z / self.temperature, dim=1))
        if self.queue_size is not None:
            q = getattr(self, "{}_{}".format(basename, idx))(r, z, self.prototypes)
        else:
            q = z.detach()

        q = distributed_sinkhorn(q, **self.sinkhorn_cfg)[-bs:]
        return p, q


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

        outputs = {}
        
        self.__reset_prototypes(outputs)

        if not isinstance(x1, list):
            x1 = [x1]
            x2 = [x2]
        
        CLIP_NUM = len(x1)

        feats1 = []
        feats2 = []
        for i, f in enumerate(x1):
            feats1.append(self.__forwardfeature(f, "queue1", i))

        for i, f in enumerate(x2):
            feats2.append(self.__forwardfeature(f, "queue2", i))

        loss = 0.0
        loss_cnts = 0
        for xi in feats1:
            for yi in feats2:
                loss -= torch.mean(torch.sum(xi[1].detach() * yi[0], dim=1))
                loss -= torch.mean(torch.sum(yi[1].detach() * xi[0], dim=1))
                loss_cnts += 2
        loss /= loss_cnts

        outputs['loss'] = loss

        self.iteration[0] = self.iteration[0] + 1

        return outputs 
    
    def evaluate(self, x):
        pass