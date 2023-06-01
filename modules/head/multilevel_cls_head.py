import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Accuracy
from mmcv.cnn import normal_init
from .build import HEAD_REGISTERY

def cross_entropy(pred, label):
    loss = F.cross_entropy(pred, label, reduction='none')
    loss = loss.mean()
    return loss

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

class MetricLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        normal_init(self, mean=0, std=0.01, bias=0)

    def forward(self, x):
        weight = F.normalize(self.weight, dim=1)
        out = F.linear(x, weight, self.bias)

        return out

@HEAD_REGISTERY.register
class MLLinearClsHead(nn.Module):
    def __init__(self, num_classes, in_channels, metric=False, topk=(1, ), ratio=None):
        super(MLLinearClsHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_levels = len(self.num_classes)
        if ratio is None:
            self.ratio = [1.0/self.num_levels] * self.num_levels
        else:
            self.ratio = ratio

        self.metric = metric

        for num_class in self.num_classes:
            if num_class < 0:
                raise ValueError(
                    f'num_classes={num_class} must be a positive integer')

        self._init_layers()

        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.accuracy = Accuracy(topk=self.topk)

    def _init_layers(self):
        self.classifier = []
        for i, num_class in enumerate(self.num_classes):
            name = "classifier{}".format(i)
            self.classifier.append(name)
            if self.metric:
                self.add_module(name, MetricLinear(self.in_channels, num_class))
            else:
                self.add_module(name, nn.Linear(self.in_channels, num_class))

    def init_weights(self):
        for c in self.classifier:
            normal_init(getattr(self, c), mean=0, std=0.01, bias=0)

    def loss(self, logits, gt_label, aggregate=None):
        num_samples = logits[0].size()

        outputs = dict(accuracy=dict())
        total_loss = 0

        for i in range(len(logits)):
            labels = gt_label[:,i]
            loss = cross_entropy(logits[i], labels)

            total_loss = total_loss + self.ratio[i] * loss
            
            acc = self.accuracy(logits[i], labels)

            outputs['{}-loss'.format(i)] = loss

            outputs['accuracy']['{}-level'.format(i)] = acc[0]

        outputs['loss'] = total_loss
        return outputs

    def forward(self, x, gt_label, aggregate=None):
        logits = list(map(lambda f:getattr(self, f)(x), self.classifier))
        outputs = self.loss(logits, gt_label, aggregate)

        return outputs

    def getlabels(self, x):
        logits = list(map(lambda f:getattr(self, f)(x), self.classifier))
        return logits

    def extra_repr(self):
        return 'metric={}, ratio={}'.format(
            self.metric, self.ratio)

    def inference(self, x):
        logits = list(map(lambda f:getattr(self, f)(x), self.classifier))[0]
        return torch.argmax(logits, dim=1)

    def multicls_ret(self, x, labels):
        logits = list(map(lambda f:getattr(self, f)(x), self.classifier))[0]
        preds = torch.argmax(logits, dim=1)

        labels = labels[:, 0]

        correct = (preds == labels)

        ret = torch.zeros((2, self.num_classes[0])).cuda()

        match = (preds == labels)
        for i in range(self.num_classes[0]):
            mask = (labels == i)

            ret[0][i] += mask.sum()
            ret[1][i] += (match[mask]).sum()

        return ret