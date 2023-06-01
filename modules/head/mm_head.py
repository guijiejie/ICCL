import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Accuracy
from mmcv.cnn import normal_init, kaiming_init
import numpy as np
from ..apis import train

from ..neck import NonlinearNeck
from .build import HEAD_REGISTERY
from .utilsfns import *

@HEAD_REGISTERY.register
class MultiModalHead(nn.Module):
    def __init__(self, in_channels=1024, ssType="simsiam", ssParams=dict(), projector=True, extra_projector=False):
        super().__init__()
        self.has_projector = projector
        self.extra_projector = extra_projector

        if projector:
            ssParams['proj_layers'] = 0
            out_channels = ssParams['out_channels']
            self.img_projector = NonlinearNeck(layer_info=[
                dict(in_features=in_channels, out_features=out_channels, norm=True, relu=True),
                dict(in_features=out_channels, out_features=out_channels, norm=True, relu=True),
                dict(in_features=out_channels, out_features=out_channels, norm=True, relu=False),
            ])

            if self.extra_projector:
                self.extra_proj = NonlinearNeck(layer_info=[
                                        dict(in_features=out_channels, out_features=2*out_channels, norm=True, relu=True),
                                        dict(in_features=2*out_channels, out_features=2*out_channels, norm=True, relu=True),
                                        dict(in_features=2*out_channels, out_features=out_channels, norm=True, relu=False),
                                    ])

            self.text_projector = NonlinearNeck(layer_info=[
                dict(in_features=in_channels, out_features=out_channels, norm=True, relu=True),
                dict(in_features=out_channels, out_features=out_channels, norm=True, relu=True),
                dict(in_features=out_channels, out_features=out_channels, norm=False, relu=False),
            ])

        self.multiview_head = HEAD_REGISTERY.load(ssType)(in_channels=in_channels, **ssParams)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.maxbound = (torch.tensor(1.) * np.log(100)).cuda()

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

    def infer_sim(self, p, q):
        if self.has_projector:
            p = self.img_projector(p)
            q = self.text_projector(q)
        
        p = p / p.norm(dim=-1, keepdim=True)
        q = q / q.norm(dim=-1, keepdim=True)

        return p, q

    def forward(self, p1, p2, q):
        listp = [p1] + p2
        assert isinstance(listp, list)
        if self.has_projector:
            presim = F.cosine_similarity(listp[0], listp[1], dim=-1).mean()
            listp = list(map(self.img_projector, listp))

            q = self.text_projector(q)

        outputs = self.multiview_head(listp[0], listp[1])

        for i in range(2):
            for j in range(2, len(listp)):
                tmpoutputs = self.multiview_head(listp[i], listp[j])
                for k, v in outputs.items():
                    outputs[k] = v + tmpoutputs[k]
        
        for k, v in outputs.items():
            outputs[k] = v / (2 * len(listp) - 3)
        
        if self.extra_projector:
            listp = list(map(self.extra_proj, listp))

        listp = listp[0:2]
        logit_scale = torch.max(self.logit_scale.exp(), self.maxbound.exp())

        image_features = list(map(lambda x:x/x.norm(dim=-1, keepdim=True), listp))
        
        text_features = q / q.norm(dim=-1, keepdim=True)

        logits_per_image = list(map(lambda x:logit_scale * x @ text_features.t(), image_features))
        logits_per_text = logit_scale * text_features @ image_features[0].t()

        label = torch.arange(q.size(0), dtype=torch.long).cuda()
        
        loss = 0.
        cnts = len(logits_per_image)
        for i in range(cnts):
            loss += (cross_entropy_shootinfs(logits_per_image[i], label) + cross_entropy_shootinfs(logits_per_text, label))

        loss /= cnts
        loss /= 2.

        outputs['mm_sim'] = (logits_per_image[0] / logit_scale)[torch.arange(q.size(0)), label].mean()
        outputs['multimodal'] = loss
        if self.has_projector:
            outputs['pre-sim'] = presim
        outputs['loss'] = outputs['loss'] + loss
        return outputs

    def encode_text(self, text):
        if self.has_projector:
            text = self.text_projector(text)
        
        return text

    def encode_img(self, img):
        if self.has_projector:
            img = self.img_projector(img)
        
        return img

@HEAD_REGISTERY.register
class MomentumMultiModalHead(nn.Module):
    def __init__(self, in_channels=1024, ssType="MoCo", ssParams=dict(), momentum=False, buffer_params=dict(queue_size=4096)):
        super().__init__()
        self.multiview_head = HEAD_REGISTERY.load(ssType)(**ssParams)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.maxbound = (torch.tensor(1.) * np.log(100)).cuda()
        self.momentum = momentum
        if self.momentum:
            self.add_module("img_queue1", BufferQueue(in_channels, **buffer_params))
            self.add_module("img_queue2", BufferQueue(in_channels, **buffer_params))
            self.add_module("text_queue", BufferQueue(in_channels, **buffer_params))

            self.img_predictor = NonlinearNeck(layer_info=[
                dict(in_features=in_channels, out_features=in_channels * 2, norm=True, relu=True),
                dict(in_features=in_channels * 2, out_features=in_channels * 2, norm=True, relu=True),
                dict(in_features=in_channels * 2, out_features=in_channels, norm=True, relu=False),
            ])

            self.text_predictor = NonlinearNeck(layer_info=[
                dict(in_features=in_channels, out_features=in_channels * 2, norm=True, relu=True),
                dict(in_features=in_channels * 2, out_features=in_channels * 2, norm=True, relu=True),
                dict(in_features=in_channels * 2, out_features=in_channels, norm=True, relu=False),
            ])

    def init_weights(self):
        self.multiview_head.init_weights()
    
    def __enqueue(self, feats, queue):
        if torch.distributed.is_initialized():
            feats = concat_all_gather(feats)

        feats = queue(feats)
        
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.detach()

        return feats

    def forward(self, p, momentum_p, t, momentum_t):
        outputs = self.multiview_head(p, momentum_p)
        
        logit_scale = torch.max(self.logit_scale.exp(), self.maxbound.exp())
        
        if self.momentum:
            p = list(map(lambda x:self.img_predictor(x), p))
            t = self.text_predictor(t)

        image_features = list(map(lambda x:x/x.norm(dim=-1, keepdim=True), p))
        
        text_features = t / t.norm(dim=-1, keepdim=True)

        if self.momentum:
            bs_per_gpu = momentum_t.size(0)
            momentum_image_features = [self.__enqueue(momentum_p[0], self.img_queue1), self.__enqueue(momentum_p[1], self.img_queue2)]
            momentum_text_features = self.__enqueue(momentum_t, self.text_queue)
            
            logits_per_image = list(map(lambda x:logit_scale * x @ momentum_text_features.t(), image_features))
            logits_per_text = logit_scale * text_features @ momentum_image_features[0].t()

            if torch.distributed.is_initialized():
                label = bs_per_gpu * torch.distributed.get_rank() + torch.arange(bs_per_gpu, dtype=torch.long).cuda()
            else:
                label = torch.arange(bs_per_gpu, dtype=torch.long).cuda()

        else:
            logits_per_image = list(map(lambda x:logit_scale * x @ text_features.t(), image_features))
            logits_per_text = logit_scale * text_features @ image_features[0].t()

            label = torch.arange(t.size(0), dtype=torch.long).cuda()
        
        loss = 0.
        cnts = len(logits_per_image)
        for i in range(cnts):
            loss += (cross_entropy_shootinfs(logits_per_image[i], label) + cross_entropy_shootinfs(logits_per_text, label))

        loss /= cnts
        loss /= 2.

        outputs['mm_sim'] = (logits_per_image[0] / logit_scale)[torch.arange(t.size(0)), label].mean()
        outputs['multimodal'] = loss
        outputs['loss'] = outputs['loss'] + loss
        
        return outputs

    def encode_text(self, text):
        feature = self.text_proj(text)
        
        return feature