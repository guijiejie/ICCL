import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import init_parameters
from ..apis import train
from ..neck import NonlinearNeck
from .build import HEAD_REGISTERY
from .utilsfns import *
from torch import einsum
from einops import rearrange

@HEAD_REGISTERY.register
class RCCHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, proj_layers=2, queue_size=4096,
                    tau1=0.1, tau2=0.05, adaptive=False, proj_bn=True, pca=None, warmup_epoch=[0, 0], initial=dict()):
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
            if proj_bn:
                proj_list.append(nn.BatchNorm1d(out_channels, affine=False))
            self.proj = nn.Sequential(*proj_list)

        self.pred = nn.Sequential(nn.Linear(out_channels, hidden_channels, bias=False),
                                nn.BatchNorm1d(hidden_channels),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_channels, out_channels))

        self.tau1 = tau1
        self.tau2 = tau2
        self.adaptive = adaptive
        self.initial = initial

        self.buffer_queue = BufferQueue(out_channels, queue_size=queue_size)
        if pca is not None:
            self.pca = PCA(**pca)
        else:
            self.pca = None

        self.warmup_epoch = warmup_epoch

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_parameters(m, **self.initial)

            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def mce_loss(self, p, q, outputs=None):
        t1 = self.tau1
        t2 = self.tau2

        # l2 norm
        p = F.normalize(p, dim=1)
        q = F.normalize(q, dim=1)

        q = torch.softmax(q / t2, dim=1).detach()

        if self.adaptive:
            tmpt1 = q.norm(dim=1, keepdim=True)
            t1 = tmpt1

        loss = - (q * p / t1).sum(dim=1).mean()
        if outputs is not None:
            outputs['qnorm'] = q.norm(dim=1).mean()
        return loss
    
    def sim_loss(self, p, q, outputs=None):
        # l2 norm
        p = F.normalize(p, dim=1)
        q = F.normalize(q, dim=1)

        return - F.cosine_similarity(p, q.detach(), dim=-1, eps=1e-8).mean()

    def forward(self, x1, x2):
        outputs = {}
        if not isinstance(x1, list):
            x1 = [x1]
            x2 = [x2]
        
        assert len(x1) == 1 or len(x1) == 2

        if len(x1) == 1:
            x = x1 + x2
            if self.proj is not None:
                x = list(map(self.proj, x))

            q = list(map(lambda xi:xi.detach(), x))
            p = list(map(self.pred, x))
        else:
            if self.proj is not None:
                x1 = list(map(self.proj, x1))
                x2 = list(map(self.proj, x2))
            
            q = list(map(lambda xi:xi.detach(), x2))
            p = list(map(self.pred, x1))

        p1, p2 = p
        q1, q2 = q

        loss_sim = D(p1, q2) / 2 + D(p2, q1) / 2

        if self.pca is not None:
            pca_feats = torch.cat([q1, q2], dim=0)
            pca_feats = concat_all_gather(pca_feats)
            pca_feats = self.buffer_queue(pca_feats)
            self.pca.fit(pca_feats)
            q1 = self.pca.transform(q1)
            q2 = self.pca.transform(q2)

            p1 = self.pca.transform(p1)
            p2 = self.pca.transform(p2)

        loss_mce = self.mce_loss(p1, q2, outputs) / 2 + self.mce_loss(p2, q1) / 2

        now_epoch = train.global_epoch
        if now_epoch < self.warmup_epoch[0]:
            scale = 0.0
        elif now_epoch < self.warmup_epoch[1]:
            scale = (now_epoch - self.warmup_epoch[0]) / float(self.warmup_epoch[1] - self.warmup_epoch[0])
        else:
            scale = 1.0
        loss = scale * loss_mce + (1 - scale) * loss_sim

        outputs['sim'] = F.cosine_similarity(p1, q2, dim=-1).mean()
        outputs['loss'] = loss
        
        return outputs