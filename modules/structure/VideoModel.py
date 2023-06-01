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
class VideoModel(nn.Module):
    def __init__(self, cfg):
        super(VideoModel, self).__init__()
        build_dict = {
            "backbone":"backbone"
        }

        for k, v in build_dict.items():
            build_func[v](self, cfg, k, alias=None)

        self.backbone.init_weights()
        if "pretrained" in cfg:
            pretrained = cfg.pop("pretrained")
            if pretrained is not None:
                load_checkpoint(self, pretrained, strict=True, map_location="cpu")

        build_func['neck'](self, cfg, 'neck', alias=None)
        if self.neck is not None:
            self.neck.init_weights()

        build_func['head'](self, cfg, 'head', alias=None)
        self.head.init_weights()
        
    def forward(self, x, label=None, aggregate=False):
        if aggregate:
            return self.forward_aggregate(x, label)

        N, I, C, D, H, W = x.size()

        x = rearrange(x, "n i c d h w -> (i c) n d h w")

        CLIP_NUM = I * C
        # label = repeat(label, "n d -> (b n) d", b = I * C)
        total_out = None

        for i in range(CLIP_NUM):
            out = self.backbone(x[i])
            if self.neck is not None:
                out = self.neck(out)
            out = self.head(out, label)

            if total_out is None:
                total_out = out
            else:
                for k, v in out.items():
                    if isinstance(v, dict):
                        for k1, v1 in v.items():
                            total_out[k][k1] += v1
                    else:
                        total_out[k] += v
        
        for k, v in total_out.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    total_out[k][k1] = v1 / float(CLIP_NUM)
            else:
                total_out[k] = v / float(CLIP_NUM)
        return total_out
    
    def forward_aggregate(self, x, label):
        N, I, C, D, H, W = x.size()

        x = rearrange(x, "n i c d h w -> (n i c) d h w")
        CLIP_NUM = I * C
        # label = repeat(label, "n d -> (b n) d", b = I * C)

        out = self.backbone(x)
        
        if self.neck is not None:
            out = self.neck(out)
        
        out = rearrange(out, "(n N) C -> n N C", N=CLIP_NUM).mean(dim=1)
        
        out = self.head(out, label)
        
        return out