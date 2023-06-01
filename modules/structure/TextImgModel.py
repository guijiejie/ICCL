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
class MultiModalModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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
        if self.head is not None:
            self.head.init_weights()
        
        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                load_checkpoint(self, pretrained, strict=False, map_location="cpu")
    
    def forward(self, x1, x2=None, forward_knn=False):
        if forward_knn:
            return self.forward_knn(x1)

        if isinstance(x1, list):
            p1, q = self.backbone(x1[0], x2)
            p2 = list(map(lambda xi:self.backbone(xi), x1[1:]))
            out = self.head(p1, p2, q)
        else:
            p, q = self.backbone(x1, x2)
            out = self.head(p, q)
        return out
    
    def forward_knn(self, x):
        x = rearrange(x, "n i c d h w -> (c i) n d h w")

        out = torch.stack(list(map(lambda a:self.backbone(a), x)))

        out = rearrange(out, "i n d -> n i d")
        return out
    
    def encode_img(self, x):
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        if self.head is not None:
            out = self.head.encode_img(out)
        return out
    
    def encode_text(self, x):
        text_feature = self.backbone.encode_text(x)
        if self.head is not None:
            text_feature = self.head.encode_text(text_feature)
        return text_feature
    
    def infer_sim(self, x1, x2):
        p, q = self.backbone(x1, x2)
        out = self.head.infer_sim(p, q)
        return out

@STRUCTURE_REGISTERY.register
class MomentumMultiModalModel(nn.Module):
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
    
    def __mmforward(self, net, img, text=None):
        p1, t = net[0](img[0], text)
        p1, t = net[1](p1, t)
        p_rest = list(map(lambda xi:net(xi), img[1:]))

        p = [p1] + p_rest

        return p, t

    def forward(self, x1, x2=None, forward_knn=False):
        if forward_knn:
            return self.forward_knn(x1)

        assert isinstance(x1, list)

        p, t = self.__mmforward(self.online_net, x1, x2)

        with torch.no_grad():
            momentum_p = list(map(lambda xi:self.target_net(xi).clone().detach(), x1))
            momentum_t = self.target_net[0].encode_text(x2)
            momentum_t = self.target_net[1].textlayer(momentum_t).clone().detach()

        out = self.head(p, momentum_p, t, momentum_t)
        return out
    
    def forward_knn(self, x):
        x = rearrange(x, "n i c d h w -> (c i) n d h w")

        out = torch.stack(list(map(lambda a:self.backbone(a), x)))

        out = rearrange(out, "i n d -> n i d")
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