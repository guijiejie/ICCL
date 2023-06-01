import torch.nn as nn
import torch

def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    #pred => [batchsize, d], pred_label => [batchsize, maxk]
    _, pred_label = pred.topk(maxk, dim=1)
    #pred_label => [maxk, batchsize], target => [batchsize]
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        # res.append(correct_k.mul_(100.0 / pred.size(0)))
        res.append(correct_k)
    return res[0] if return_single else res

class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)