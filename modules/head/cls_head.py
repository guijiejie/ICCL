import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Accuracy
from mmcv.cnn import normal_init
from .build import HEAD_REGISTERY
from .utilsfns import *

def cross_entropy(pred, label):
    loss = F.cross_entropy(pred, label, reduction='none')
    loss = loss.mean()
    return loss

@HEAD_REGISTERY.register
class LinearClsHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 topk=(1, )):
        super(LinearClsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.accuracy = Accuracy(topk=self.topk)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        loss = cross_entropy(cls_score, gt_label)
        # compute accuracy
        acc = self.accuracy(cls_score, gt_label)
        assert len(acc) == len(self.topk)
        # losses['scores'] = cls_score
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        return losses

    def forward(self, x, gt_label):
        if len(x.size()) > 2:
            x = self.gap(x).view(x.size(0), -1)

        logits = self.fc(x)
        losses = self.loss(logits, gt_label)

        return losses

    def inference(self, x):
        if len(x.size()) > 2:
            x = self.gap(x).view(x.size(0), -1)

        logits = self.fc(x)
        
        return torch.argmax(logits, dim=1)

    def multicls_ret(self, x, labels):
        if len(x.size()) > 2:
            x = self.gap(x).view(x.size(0), -1)

        logits = self.fc(x)
        preds = torch.argmax(logits, dim=1)
        
        correct = (preds == labels)

        ret = torch.zeros((2, self.num_classes)).cuda()

        match = (preds == labels)
        for i in range(self.num_classes):
            mask = (labels == i)

            ret[0][i] += mask.sum()
            ret[1][i] += (match[mask]).sum()

        return ret
