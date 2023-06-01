import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.runner import load_checkpoint

from ..utils import init_parameters, init_constant

from .build import BACKBONE_REGISTERY

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(self.mid_channels)

        self.conv2 = nn.Conv2d(self.mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(self.mid_channels)

        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(self.mid_channels)

        self.conv3 = nn.Conv2d(self.mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out
        
        out = _inner_forward(x)
        out = self.relu(out)
        return out


def get_expansion(block, expansion=None):
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:          
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                **kwargs))

        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)

@BACKBONE_REGISTERY.register("ResNet")
class ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 frozen=False,
                 zero_init_residual=False,
                 avgpool=True,
                 initial=dict()):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.frozen = frozen
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)
        self.initial = initial

        self._make_stem_layer(in_channels, base_channels)

        self.res_layers = []
        _in_channels = base_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride
                )
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        if avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.avgpool = None

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(stem_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen:
            for m in [self.conv1, self.norm1]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

            for i in range(4):
                m = getattr(self, f'layer{i+1}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_parameters(m, **self.initial)
                
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init_constant(m, 1.0)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    init_constant(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    init_constant(m.norm2, 0)
        
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)

        x = self.relu(x)

        if self.maxpool is not None:
            x = self.maxpool(x)
        
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        
        if self.avgpool is not None:
            bs = x.size(0)
            out = self.avgpool(x).view(bs, -1)
        
        return out

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()

@BACKBONE_REGISTERY.register("ResNet_Cifar")
class ResNet_Cifar(ResNet):
    def __init__(self, **kwargs):
        super(ResNet_Cifar, self).__init__(**kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(stem_channels)

        self.maxpool = None

