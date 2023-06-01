import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from math import cos, pi
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .build import STRUCTURE_REGISTERY
from ..head import build as build_head
from ..neck import build as build_neck
from ..backbone import build as build_backbone
build_func = {
    'backbone':build_backbone,
    'neck':build_neck,
    'head':build_head,
}

@STRUCTURE_REGISTERY.register
class Model(nn.Module):
    '''
    A simple example of a Model Structure.
    '''
    def __init__(self, cfg):
        super(Model, self).__init__()
        build_dict = {
            "backbone":"backbone",
            "neck":"neck"
        }

        # Here we build the backbone and the neck for Model.
        # v is the name in build_func.
        # k is the name in config file.
        # The registered module is named as self.backbone if k is 'backbone' and alias is None
        # If the config file do not have k (neck), then self.neck = None
        for k, v in build_dict.items():
            build_func[v](self, cfg, k, alias=None)

        self.backbone.init_weights()
        if self.neck is not None:
            self.neck.init_weights()
        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                print("load from {}".format(pretrained))
                load_checkpoint(self, pretrained, strict=True, map_location="cpu")

        build_func['head'](self, cfg, 'head', alias=None)
        if self.head is not None:
            self.head.init_weights()
        
    def forward(self, x, label=None, forward_knn=False):
        '''
        In test mode, the test process may pass forward_knn or any other flag to be distingushed from training process. 
        '''
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        out = self.head(out, label)
        return out

    
    def forward_knn(self, x):
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        
        out = self.head.inference(out)
        return out

    def forward_test(self, x):
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        
        out = self.head.inference(out)
        return out

@STRUCTURE_REGISTERY.register
class MomentumSSModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        build_dict = {
            "backbone":"backbone",
            "neck":"neck"
        }

        for k, v in build_dict.items():
            build_func[v](self, cfg, k, alias=None)
            build_func[v](self, cfg, k, alias=f'target_{k}')

        self.online_net = nn.Sequential(
            self.backbone, self.neck)
        self.target_net = nn.Sequential(
            self.target_backbone, self.target_neck)

        build_func['head'](self, cfg, 'head', alias=None)

        for param in self.target_net.parameters():
            param.requires_grad = False

        model_cfg = cfg['Model']
        self.base_momentum = model_cfg['base_momentum']
        self.momentum = model_cfg['base_momentum']
        self.end_momentum = model_cfg['end_momentum']

        self.init_weights(cfg)

        self.total_epochs = cfg['total_epochs']
        self.update_interval = cfg['update_interval']
    
    def init_weights(self, cfg):
        self.backbone.init_weights()
        if self.neck is not None:
            self.neck.init_weights()

        self.head.init_weights()
        
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)

        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                load_checkpoint(self, pretrained, strict=False, map_location="cpu")

    def forward(self, x1, x2=None, forward_knn=False):
        if forward_knn:
            return self.forward_knn(x1)
        if not isinstance(x1, list):
            x = [x1, x2]
        else:
            x = x1 + x2

        p = list(map(lambda xi:self.online_net(xi), x))

        with torch.no_grad():
            q = list(map(lambda xi:self.target_net(xi).clone().detach(), x))

        out = self.head(p, q)
        out['momentum'] = torch.Tensor([self.momentum])
        return out
    
    def forward_knn(self, x):
        out = self.backbone(x)
        return out

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    def before_train_iter(self, now_iter, total_iter, now_epoch):
        if now_iter % self.update_interval == 0:
            cur_iter = now_epoch * total_iter + now_iter
            max_iter = self.total_epochs * total_iter
            self.momentum = self.end_momentum - (self.end_momentum - self.base_momentum) * (cos(pi * cur_iter / float(max_iter)) + 1) / 2

    def after_train_iter(self, now_iter, total_iter, now_epoch):
        if now_iter % self.update_interval == 0:
            self._momentum_update()

@STRUCTURE_REGISTERY.register
class SSModel(nn.Module):
    def __init__(self, cfg):
        super(SSModel, self).__init__()
        build_dict = {
            "backbone":"backbone",
            "neck":"neck",
            "head":"head"
        }

        for k, v in build_dict.items():
            build_func[v](self, cfg, k, alias=None)

        self.backbone.init_weights()
        if self.neck is not None:
            self.neck.init_weights()
        self.head.init_weights()
        
        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                load_checkpoint(self, pretrained, strict=True, map_location="cpu")
    
    def forward(self, x1, x2=None, forward_knn=False):
        if forward_knn:
            return self.forward_knn(x1)

        if not isinstance(x1, list):
            x1 = [x1]
            x2 = [x2]

        p1 = list(map(lambda xi:self.backbone(xi), x1))
        p2 = list(map(lambda xi:self.backbone(xi), x2))
        
        if self.neck is not None:
            p1 = list(map(lambda xi:self.neck(xi), p1))
            p2 = list(map(lambda xi:self.neck(xi), p2))

        out = self.head(p1, p2)
        return out
    
    def forward_knn(self, x):
        out = self.backbone(x)
        return out

@STRUCTURE_REGISTERY.register
class TwoSSModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        build_dict = {
            "backbone":"backbone",
            "neck":"neck",
            "backbone2":"backbone",
            "neck2":"neck",
            "head":"head"
        }

        for k, v in build_dict.items():
            build_func[v](self, cfg, k, alias=None)

        self.backbone.init_weights()
        self.backbone2.init_weights()
        self.neck.init_weights()
        self.neck2.init_weights()
        self.head.init_weights()
        
        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                load_checkpoint(self, pretrained, strict=True, map_location="cpu")
        
        model_cfg = cfg['Model']
        self.swap_transform = model_cfg.get("SwapTrans", False)
    
    def forward(self, x1, x2=None, forward_knn=False):
        if forward_knn:
            return self.forward_knn(x1)

        if not isinstance(x1, list):
            x1 = [x1]
            x2 = [x2]
        if self.swap_transform:
            x1 = x1 + x2
            x2 = x1

        p1 = list(map(lambda xi:self.backbone(xi), x1))
        p2 = list(map(lambda xi:self.backbone2(xi), x2))
        
        if self.neck is not None:
            p1 = list(map(lambda xi:self.neck(xi), p1))
            p2 = list(map(lambda xi:self.neck2(xi), p2))

        out = self.head(p1, p2)
        return out
    
    def forward_knn(self, x):
        out = self.backbone(x)
        return out