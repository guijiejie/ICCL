from .modellib.sbert import SBERT
from .modellib.swin_transformer import SwinTransformer
from .modellib.clip import CLIP
from .modellib.convit import ConViT
import torch.nn as nn
import torch
from torch import Tensor
import math
from .build import BACKBONE_REGISTERY

def build_linear_layer(in_features, out_features, norm=False, bias=True, relu=True):
    ret = [nn.Linear(in_features, out_features, bias=bias and not norm)]
    if norm:
        ret += [nn.BatchNorm1d(out_features)]
    if relu:
        ret += [nn.ReLU(inplace=True)]

    return ret

class NonlinearNeck(nn.Module):
    def __init__(self, layer_info, avgpool=False):
        super().__init__()
        layer_list = []

        for l in layer_info:
            layer_list.extend(build_linear_layer(**l))

        self.layer = nn.Sequential(*layer_list)
        self.avgpool = None
        if avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                m.reset_parameters()

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.avgpool is not None:
            x = self.avgpool(x).view(x.size(0), -1)

        x = self.layer(x)
        return x

@BACKBONE_REGISTERY.register
class MultiModalBackbone(nn.Module):
    def __init__(self, embed_dim=512, frozen=False, vision_model="SwinTransformer", vision_params=dict(), text_model=None, text_params=dict(), img_projector=False):
        super().__init__()

        vision_model_dict = {
            "SwinTransformer":SwinTransformer,
            "CLIP":CLIP,
            "ConViT":ConViT,
        }
        text_model_dict = {
            "SBERT":SBERT,
        }

        self.visual_encoder = vision_model_dict.get(vision_model, SwinTransformer)(**vision_params)
        if self.visual_encoder.output_dim == embed_dim:
            self.visual_projection = None
        else:
            self.visual_projection = nn.Parameter(torch.empty(self.visual_encoder.output_dim, embed_dim))
        
        self.img_projector = None
        # if img_projector:
        #     self.img_projector = NonlinearNeck(layer_info=[
        #         dict(in_features=embed_dim, out_features=embed_dim, norm=True, relu=True),
        #         dict(in_features=embed_dim, out_features=embed_dim, norm=True, relu=True),
        #         dict(in_features=embed_dim, out_features=embed_dim, norm=True, relu=False),
        #     ])

        if text_model is None:
            self.text_encoder = None
            self.text_projection = None
        else:
            self.text_encoder = text_model_dict.get(text_model, SBERT)(**text_params)

            if self.text_encoder.width != embed_dim:
                self.text_projection = nn.Parameter(torch.empty(self.text_encoder.width, embed_dim))
            else:
                self.text_projection = None
        
        self.initialize_parameters()

        self.frozen = frozen
        if frozen:
            self.__frozen()

    def initialize_parameters(self):
        if self.visual_projection is not None:
            border = math.sqrt(3) / math.sqrt(self.visual_encoder.output_dim)
            nn.init.uniform_(self.visual_projection, a=-border, b=border)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.text_encoder.width ** -0.5)

    def __frozen(self):
        for name in ['visual_encoder', 'text_encoder', 'img_projector']:
            if hasattr(self, name) and getattr(self, name) is not None:
                getattr(self, name).eval()
                
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.__frozen()
    
    def forward(self, x, text=None):
        if text is None:
            return self.encode_img(x)
        else:
            img_features = self.encode_img(x)
            text_features = self.encode_text(text)
            return img_features, text_features

    def encode_img(self, img):
        out = self.visual_encoder(img)
        if self.visual_projection is not None:
            out = out @ self.visual_projection
        # if self.img_projector is not None:
        #     out = torch.cat([out, self.img_projector(out)], dim=1)
        return out

    def encode_text(self, text):
        assert self.text_encoder is not None
        out = self.text_encoder(text)

        if self.text_projection is not None:
            out = out @ self.text_projection

        return out

    def init_weights(self, pretrained=None):
        self.initialize_parameters()
