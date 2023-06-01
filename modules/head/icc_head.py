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
class ICCHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, proj_layers=2, 
                    warmup_epoch=[0, 0], tau1=0.1, tau2=0.07, ratio=1.0, adaptive=False, pca=None, initial=dict()):
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
        
        self.warmup_epoch = warmup_epoch
        self.tau1 = tau1
        self.tau2 = tau2
        self.ratio = ratio
        self.initial = initial
        self.adaptive = adaptive

        if pca is not None:
            self.pca = PCA(**pca)
        else:
            self.pca = None

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
            
            loss_sim = D(p1, q2) / 2 + D(p2, q1) / 2

            if self.pca is not None:
                self.pca.fit(torch.cat([q1, q2], dim=0))
                q1 = self.pca.transform(q1)
                q2 = self.pca.transform(q2)

                p1 = self.pca.transform(p1)
                p2 = self.pca.transform(p2)

            m1 = torch.softmax(F.normalize(q1, dim=1) / self.tau2, dim=1).detach()
            m2 = torch.softmax(F.normalize(q2, dim=1) / self.tau2, dim=1).detach()

            if self.adaptive:
                tmptau1 = m2.norm(dim=1)
                # tmptau1[tmptau1 > self.tau1] = self.tau1

                tmptau2 = m1.norm(dim=1)
                # tmptau2[tmptau2 > self.tau1] = self.tau1

                outputs['tau_norm'] = (m2.norm(dim=1).mean() + m1.norm(dim=1).mean()) / 2.0

                n1 = torch.softmax(F.normalize(p1, dim=1) / tmptau1.unsqueeze(1).detach(), dim=1)
                n2 = torch.softmax(F.normalize(p2, dim=1) / tmptau2.unsqueeze(1).detach(), dim=1)
            else:
                n1 = torch.softmax(F.normalize(p1, dim=1) / self.tau1, dim=1)
                n2 = torch.softmax(F.normalize(p2, dim=1) / self.tau1, dim=1)

            logn1 = shoot_infs(n1.log())
            logn2 = shoot_infs(n2.log())
            loss_aug = - ((m1 * logn2).sum(dim=1).mean() + (m2 * logn1).sum(dim=1).mean()) / 2.0

            prob_sum = (n1 + n2).sum(dim=0)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(prob_sum)
                prob_sum = prob_sum / (m1.size(0) * torch.distributed.get_world_size())
            else:
                prob_sum = prob_sum / m1.size(0)
            prob_sum /= 2.0

            loss_uniform = -(math.log(prob_sum.size(-1)) + prob_sum.log().mean())

            now_epoch = train.global_epoch
            if now_epoch < self.warmup_epoch[0]:
                scale = 0.0
            elif now_epoch < self.warmup_epoch[1]:
                scale = (now_epoch - self.warmup_epoch[0]) / float(self.warmup_epoch[1] - self.warmup_epoch[0])
            else:
                scale = 1.0
            loss = scale * (self.ratio * loss_uniform + loss_aug) + (1 - scale) * loss_sim

            outputs['scale'] = torch.tensor(scale)
            outputs['loss_sim'] = loss_sim
            outputs['loss_uniform'] = loss_uniform
            outputs['loss_aug'] = loss_aug
            return loss

        if isinstance(x1, torch.Tensor):
            total_loss = __cal_sim(x1, x2, outputs)
        else:
            total_loss = 0
            cnts = 0
            if len(x1) > 1:
                for i in range(len(x1)-1):
                    for j in range(i+1, len(x1)):
                        total_loss += __cal_sim(x1[i], x1[j])
                        cnts += 1
            
            for xi in x1:
                for yi in x2:
                    total_loss += __cal_sim(xi, yi)
                    cnts += 1
            
            total_loss = total_loss / cnts

        outputs['loss'] = total_loss
        
        return outputs

@HEAD_REGISTERY.register
class MomentumICCHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, warmup_epoch=[0, 0], 
                    tau1=0.1, tau2=0.07, ratio=5.0, adaptive=False, initial=dict()):
        super().__init__()
        self.pred = nn.Sequential(nn.Linear(in_channels, hidden_channels, bias=False),
                                nn.BatchNorm1d(hidden_channels),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_channels, out_channels, bias=False))
        
        self.warmup_epoch = warmup_epoch
        self.tau1 = tau1
        self.tau2 = tau2
        self.ratio = ratio
        self.initial = initial
        self.adaptive = adaptive

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_parameters(m, **self.initial)

            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __loss_sim(self, p1, q2):
        pred_norm = nn.functional.normalize(p1, dim=1)
        target_norm = nn.functional.normalize(q2, dim=1)

        loss = ((pred_norm - target_norm) ** 2).sum(dim=-1).mean()
        return loss

    def forward(self, p, q):
        outputs = {}
        p1, p2 = p[0:2]
        q1, q2 = q[0:2]

        p1 = self.pred(p1)
        p2 = self.pred(p2)
            
        loss_sim = self.__loss_sim(p1, q2) + self.__loss_sim(p2, q1)

        m1 = torch.softmax(F.normalize(q1, dim=1) / self.tau2, dim=1).detach()
        m2 = torch.softmax(F.normalize(q2, dim=1) / self.tau2, dim=1).detach()

        outputs['tau_norm'] = (m2.norm(dim=1).mean() + m1.norm(dim=1).mean()) / 2.0
        
        if self.adaptive:
            tmptau1 = m2.norm(dim=1)
            tmptau1[tmptau1 > self.tau1] = self.tau1

            tmptau2 = m1.norm(dim=1)
            tmptau2[tmptau2 > self.tau1] = self.tau1

            n1 = torch.softmax(F.normalize(p1, dim=1) / tmptau1.unsqueeze(1).detach(), dim=1)
            n2 = torch.softmax(F.normalize(p2, dim=1) / tmptau2.unsqueeze(1).detach(), dim=1)
        else:
            n1 = torch.softmax(F.normalize(p1, dim=1) / self.tau1, dim=1)
            n2 = torch.softmax(F.normalize(p2, dim=1) / self.tau1, dim=1)

        logn1 = shoot_infs(n1.log())
        logn2 = shoot_infs(n2.log())
        loss_aug = - ((m1 * logn2).sum(dim=1).mean() + (m2 * logn1).sum(dim=1).mean()) / 2.0

        reg_uniformity = (n1 + n2).sum(dim=0)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(reg_uniformity)
            reg_uniformity = reg_uniformity / torch.distributed.get_world_size()

        reg_uniformity = reg_uniformity / 2.0 / m1.size(0)
        reg_uniformity = -(math.log(reg_uniformity.size(-1)) + reg_uniformity.log().mean())

        now_epoch = train.global_epoch
        if now_epoch < self.warmup_epoch[0]:
            scale = 0.0
        elif now_epoch < self.warmup_epoch[1]:
            scale = (now_epoch - self.warmup_epoch[0]) / float(self.warmup_epoch[1] - self.warmup_epoch[0])
        else:
            scale = 1.0

        total_loss = scale * (self.ratio * reg_uniformity + loss_aug) + (1 - scale) * loss_sim

        outputs['scale'] = torch.tensor(scale)
        outputs['loss_sim'] = loss_sim
        outputs['loss_uniform'] = reg_uniformity
        outputs['loss_aug'] = loss_aug

        outputs['loss'] = total_loss
        
        return outputs