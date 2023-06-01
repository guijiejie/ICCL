import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Accuracy
from ..apis import train
from ..neck import NonlinearNeck
from ..utils import init_parameters
from .build import HEAD_REGISTERY
from .utilsfns import *

@HEAD_REGISTERY.register
class DinoHead(nn.Module):
    def __init__(self, prototypes=65536, tau1=0.1, tau2=0.05, center_momentum=0.9, initial=dict()):
        super().__init__()

        self.tau1 = tau1
        self.tau2 = tau2
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, prototypes))

        self.initial = initial

    def init_weights(self):
        pass
    
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

    def forward(self, p, q):
        if not isinstance(p, list):
            p = [p]
        if not isinstance(q, list):
            q = [q]

        q = q[0:2]

        students = list(map(lambda xi:F.log_softmax(xi / self.tau1, dim=1), p))
        teachers = list(map(lambda xi:F.softmax((xi - self.center) / self.tau2, dim=1), q))

        outputs = {}
        outputs['local_cnts'] = torch.tensor(len(students))
        outputs['global_cnts'] = torch.tensor(len(teachers))

        total_loss = 0
        n_loss_terms = 0

        for id_q, feat_q in enumerate(teachers):
            for id_p, feat_p in enumerate(students):
                if id_q == id_p:
                    continue
                
                total_loss -= torch.sum(feat_q.detach() * feat_p, dim=-1).mean()
                n_loss_terms += 1
                
        total_loss /= n_loss_terms
        outputs['loss'] = total_loss

        self.update_center(q)

        return outputs
