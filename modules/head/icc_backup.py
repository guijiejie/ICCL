import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils import init_parameters
from ..apis import train
from ..neck import NonlinearNeck

def shoot_infs(inp_tensor):
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.min(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

@torch.no_grad()
def distributed_sinkhorn(out, iterations=3, epsilon=0.05):
    Q = torch.exp(out / epsilon).t()
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    B = Q.shape[1] * world_size # total batch size
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def D(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1, eps=1e-8).mean()

class ICCHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, proj_layers=2, 
                    warmup_epoch=[0, 0], tau1=0.1, tau2=0.05, ratio=1.0, maxscale=1.0, swav=False, dino=False, normp=True, adaptive=False, initial=dict()):
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
        self.maxscale = maxscale
        self.normp = normp
        self.swav = swav
        self.dino = dino
        if self.dino:
            self.center_momentum = 0.9
            self.register_buffer("center", torch.zeros(1, out_channels))
        self.adaptive = adaptive

    @torch.no_grad()
    def update_center(self, q):
        q = torch.cat(q, dim=0)
        batch_center = torch.sum(q, dim=0, keepdim=True)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            batch_center = batch_center / (len(q) * torch.distributed.get_world_size())
        else:
            batch_center = batch_center / len(q)

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

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
        ## 可以用weight decay的形式来保证uniform
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

            if self.swav:
                m1 = distributed_sinkhorn(F.normalize(q1, dim=1))
                m2 = distributed_sinkhorn(F.normalize(q2, dim=1))
            elif self.dino:
                m1 = F.softmax((q1 - self.center) / self.tau2, dim=1)
                m2 = F.softmax((q2 - self.center) / self.tau2, dim=1)
            else:
                m1 = torch.softmax(F.normalize(q1, dim=1) / self.tau2, dim=1).detach()
                m2 = torch.softmax(F.normalize(q2, dim=1) / self.tau2, dim=1).detach()
            

            if self.adaptive:
                tmptau1 = m2.norm(dim=1)
                tmptau1[tmptau1 > self.tau1] = self.tau1

                tmptau2 = m1.norm(dim=1)
                tmptau2[tmptau2 > self.tau1] = self.tau1

                outputs['tau_norm'] = (m2.norm(dim=1).mean() + m1.norm(dim=1).mean()) / 2.0

                n1 = torch.softmax(F.normalize(p1, dim=1) / tmptau1.unsqueeze(1).detach(), dim=1)
                n2 = torch.softmax(F.normalize(p2, dim=1) / tmptau2.unsqueeze(1).detach(), dim=1)
            else:
                if self.normp:
                    n1 = torch.softmax(F.normalize(p1, dim=1) / self.tau1, dim=1)
                    n2 = torch.softmax(F.normalize(p2, dim=1) / self.tau1, dim=1)
                else:
                    n1 = torch.softmax(p1 / self.tau1, dim=1)
                    n2 = torch.softmax(p2 / self.tau1, dim=1)

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
                loss = loss_sim
            elif now_epoch < self.warmup_epoch[1]:
                scale = (now_epoch - self.warmup_epoch[0]) / float(self.warmup_epoch[1] - self.warmup_epoch[0])
                loss = scale * (self.ratio * loss_uniform + loss_aug) + (1 - scale) * loss_sim
            else:
                scale = 1.0
                loss = scale * (self.ratio * loss_uniform + loss_aug) + (1 - scale) * loss_sim

            if self.dino:
                self.update_center([q1, q2])

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

class MomentumICCHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, out_channels=2048, warmup_epoch=[0, 0], 
                        maxscale=1.0, tau1=0.1, tau2=0.07, ratio=5.0, frozen=False, adaptive=False, initial=dict()):
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
        self.maxscale = maxscale
        self.frozen = frozen
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

    def __loss(self, p1, q2):
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
            
        loss_sim = self.__loss(p1, q2) + self.__loss(p2, q1)

        now_epoch = train.global_epoch
        if now_epoch < self.warmup_epoch[0]:
            scale = 0.0
        elif now_epoch < self.warmup_epoch[1]:
            scale = (now_epoch - self.warmup_epoch[0]) / float(self.warmup_epoch[1] - self.warmup_epoch[0])
        else:
            scale = 1.0
        
        scale = self.maxscale * scale
        m1 = torch.softmax(F.normalize(q1, dim=1) / self.tau2, dim=1).detach()
        m2 = torch.softmax(F.normalize(q2, dim=1) / self.tau2, dim=1).detach()

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

        prob_sum = (n1 + n2).sum(dim=0)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(prob_sum)
            prob_sum = prob_sum / (m1.size(0) * torch.distributed.get_world_size())
        else:
            prob_sum = prob_sum / m1.size(0)
        prob_sum /= 2.0

        loss_uniform = -(math.log(prob_sum.size(-1)) + prob_sum.log().mean())

        total_loss = scale * (self.ratio * loss_uniform + loss_aug) + (1 - scale) * loss_sim

        if self.frozen:
            total_loss += scale * loss_sim

        outputs['scale'] = torch.tensor(scale)
        outputs['loss_sim'] = loss_sim
        outputs['loss_uniform'] = loss_uniform
        outputs['loss_aug'] = loss_aug

        outputs['loss'] = total_loss
        
        return outputs