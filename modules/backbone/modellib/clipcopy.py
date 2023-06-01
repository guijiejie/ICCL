from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
import math
import os
import torchvision.transforms as T
from PIL import Image

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def resize_pos_embed(self, posemb, embsize=3):
        posemb_tok, posemb_grid = posemb[:1], posemb[1:]
        gs_old = int(math.sqrt(len(posemb_grid)))

        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(embsize, embsize), mode='bilinear')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(embsize * embsize, -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=0)
        return posemb

    def forward(self, x):
        if x.size(3) == 7:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        else:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
            posemb_new = self.resize_pos_embed(self.positional_embedding, 3)
            x = x + posemb_new[:, None, :].to(x.dtype)

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        # self.attnpool = None

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj 

        return x

class CLIP(nn.Module):
    # param: ckpt_path, str, ckpt路径
    # param: frozen, bool, 是否冻结所有权重
    # param: centercrop, bool, 图片预处理是否采用centercrop
    '''
    e.g., 
    from clip import CLIP
    from PIL import Image
    m = CLIP(ckpt_path = "clip_rn50_v2.pth", frozen=True)
    pipeline = m.getTransform() # 获得图片预处理

    img = Image.open("demo.jpg").convert("RGB")
    preprocessed_img = pipeline(img)
    preprocessed_img = preprocessed_img.unsqueeze(0)

    feats = m.encode_img(preprocessed_img) # 这种形式preprocessed_img 需要满足shape为 [N, 3, 224, 224] 的Tensor, 其中图片已经预处理好均值化过。 或者可以用下面那种形式
    feats = m.encode_img(img, need_preprocess=True) # 传入PIL.image图片，或者可以传路径
    feats = m.encode_img("demo.jpg", need_preprocess=True) # 或者可以传list
    feats = m.encode_img(["demo.jpg", "demo2.jpg"], need_preprocess=True)
    '''
    def __init__(self, ckpt_path=None, frozen=True, centercrop=False):
        super().__init__()
        
        assert ckpt_path is not None and os.path.exists(ckpt_path), f"Please check ckpt path: [{ckpt_path}]"
        
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

            isvit = 'visual.positional_embedding' in state_dict

            self.output_dim = 512 if isvit else 1024

            if isvit:
                self.vision_width = 768
                self.vision_layers = 12
                self.vision_patch_size = 32
                self.grid_size = 7
                self.image_resolution = 224
                self.vision_heads = 12

                self.visual = VisualTransformer(
                    input_resolution=self.image_resolution,
                    patch_size=self.vision_patch_size,
                    width=self.vision_width,
                    layers=self.vision_layers,
                    heads=self.vision_heads,
                    output_dim=self.output_dim
                )
                
            else:
                self.vision_layers = (3, 4, 6, 3)
                self.vision_width = 64
                self.output_width = 7
                self.vision_patch_size = None
                assert self.output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
                self.image_resolution = 224
                self.vision_heads = 32

                self.visual = ModifiedResNet(
                    layers=self.vision_layers,
                    output_dim=self.output_dim,
                    heads=self.vision_heads,
                    input_resolution=self.image_resolution,
                    width=self.vision_width
                )
            
        except Exception as e:
            print(e, flush=True)
            raise AssertionError("Failed to Load ckpt for clip.")
        
        self.load_state_dict(state_dict, strict=True)
        self.frozen = frozen
        self.__frozen()

        img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if centercrop:
            transforms = [T.Resize(224), T.CenterCrop(224)]
        else:
            transforms = [T.Resize((224, 224))]
        transforms += [T.ToTensor(), T.Normalize(**img_norm_cfg)]
        self.pipeline = T.Compose(transforms)
    
    def getTransform(self):
        return self.pipeline

    def open_img(self, img, pipeline):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        assert isinstance(img, Image.Image), f"Need type PIL.Image.Image, but the img type is {type(img)}"

        img = pipeline(img)
        return img

    def encode_img(self, img, need_preprocess=False):
        if need_preprocess:
            if isinstance(img, list):
                img = list(map(lambda x:self.open_img(x, self.pipeline), img))
                img = torch.stack(img)
            else:
                img = self.open_img(img, self.pipeline)
                img = img.unsqueeze(0)
                
        return self.visual(img.type(self.dtype))

    def forward(self, img, need_preprocess=False):
        return self.encode_img(img)

    def __frozen(self):
        if self.frozen:
            self.visual.eval()
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.__frozen()

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype