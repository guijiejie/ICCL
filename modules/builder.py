import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import re
import math
import warnings

from .utils import LARC
try:
    from torchlars import LARS
    torchlars = True
except:
    from .utils import LARS
    torchlars = False

def build_optimizer(cfg, model, ckpt=None, logger=None, strict=True):

    cfgoptimizer = cfg.pop("optimizer")
    opt_type = cfgoptimizer.pop("type")
    
    baselr = cfgoptimizer['lr']

    group_list = [dict(params=[])]
    group_name_list = []
    param_groups = cfgoptimizer.pop("param_group", [])
    
    if len(param_groups) == 0:
        group_list = model.parameters()
    else:
        for g in param_groups:
            group_name_list.append(g.pop("name"))
            if 'lr_ratio' in g:
                g['lr'] = g.pop("lr_ratio") * baselr
                
            group_list.append(
                dict(params=[], **g)
            )
        
        for p in model.named_parameters():
            find = False
            for i, n in enumerate(group_name_list):
                if re.search(n, p[0]) is not None:
                    group_list[i+1]["params"].append(p[1])
                    find = True
                    break
            if not find:
                group_list[0]["params"].append(p[1])
        
        for i, g in enumerate(group_list[1:]):
            if logger is not None:
                logger.print('{}:{}'.format(group_name_list[i], len(g['params'])))

    if opt_type == "SGD":
        optimizer = torch.optim.SGD(group_list, **cfgoptimizer)
    elif opt_type == "LARC":
        base_optimizer = torch.optim.SGD(group_list, **cfgoptimizer)
        optimizer = LARC(base_optimizer, 0.001, False)
    elif opt_type == "LARS":
        if torchlars:
            base_optimizer = torch.optim.SGD(group_list, **cfgoptimizer)
            optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
        else:
            optimizer = LARS(group_list, **cfgoptimizer)
    elif opt_type == "Adam":
        optimizer = torch.optim.AdamW(group_list, **cfgoptimizer)
    else:
        raise ValueError(f'optimizer={opt_type} does not support.')

    if ckpt:
        print('==> Resuming...')
        model.load_state_dict(ckpt['state_dict'], strict=strict)
        optimizer.load_state_dict(ckpt['optimizer'])
        # print(optimizer.state_dict())
    return optimizer

class MyScheduler(object):
    def __init__(self, lr_scheduler, lr_type):
        self.scheduler = lr_scheduler
        self.epoch = (lr_type == "epoch")
    def step_epoch(self):
        if self.epoch:
            self.scheduler.step()
    
    def step_iter(self):
        if not self.epoch:
            self.scheduler.step()

class CosineWarmUpLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_ratio=0.001, warmup_iters=10, last_epoch=-1, verbose=False):
        self.T_max = T_max - warmup_iters
        self.eta_min = eta_min

        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters

        super(CosineWarmUpLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_iters:
            return [base_lr * self.last_epoch * (1 - self.warmup_ratio) / self.warmup_iters + base_lr * self.warmup_ratio for base_lr in self.base_lrs]

        if self.last_epoch == self.warmup_iters:
            return self.base_lrs

        # assert (self.last_epoch + self.warmup_iters - 1 - self.T_max) % (2 * self.T_max) != 0
        
        return [(1 + math.cos(math.pi * (self.last_epoch - self.warmup_iters) / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1 - self.warmup_iters) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min if not group.get('fix_lr', False) else group['lr']
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

def apply_consinelr(T_max, now_iter, part):
    if len(part) != 0:
        sub_iter = 0
        for pi in part:
            if now_iter < pi:
                break
            elif now_iter == pi:
                now_ratio = 2
                last_ratio = 1 + math.cos(math.pi * (now_iter - 1 - sub_iter) / T_max)
                return now_ratio / last_ratio
            else:
                sub_iter = pi
        
        now_iter = now_iter - sub_iter

    now_ratio = 1 + math.cos(math.pi * (now_iter) / T_max)
    last_ratio = 1 + math.cos(math.pi * (now_iter - 1) / T_max)

    return now_ratio / last_ratio

class GroupCosineWarmUpLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, part, eta_min=0, warmup_ratio=0.001, warmup_iters=10, last_epoch=-1, verbose=False):
        self.T_max = T_max - warmup_iters
        # self.part = [math.ceil(self.T_max / float(pi)) for pi in part]
        self.part = [part for _ in range(len(optimizer.param_groups))]
        self.part[0] = []
        print(self.part)
        self.eta_min = eta_min

        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters

        super(GroupCosineWarmUpLR, self).__init__(optimizer, last_epoch, verbose)

        assert len(self.optimizer.param_groups) == len(self.part)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        if self.last_epoch < self.warmup_iters:
            return [base_lr * self.last_epoch * (1 - self.warmup_ratio) / self.warmup_iters + base_lr * self.warmup_ratio for base_lr in self.base_lrs]

        if self.last_epoch == self.warmup_iters:
            return self.base_lrs

        # assert (self.last_epoch + self.warmup_iters - 1 - self.T_max) % (2 * self.T_max) != 0
        nowlrs = []
        for i, group in enumerate(self.optimizer.param_groups):
            group_lr = group['lr'] - self.eta_min
            now_iter = self.last_epoch - self.warmup_iters
            ratio = apply_consinelr(self.T_max, now_iter, self.part[i])

            tmplr = ratio * group_lr + self.eta_min
            
            nowlrs.append(tmplr)
        return nowlrs

    def _get_closed_form_lr(self):
        assert False
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

def build_lrscheduler(cfg, optimizer, total_iter, last_epoch=-1):
    lr_cfg = cfg.pop("lr_config")
    lr_policy = lr_cfg.pop("policy")
    lr_type = lr_cfg.pop("type", "epoch")
    if lr_type == 'iter':
        assert 'cosine' in lr_policy
        lr_cfg['T_max'] = lr_cfg['T_max'] * total_iter
        if last_epoch >= 0:
            last_epoch += 1
            last_epoch = last_epoch * total_iter - 1
        if lr_policy == "groupcosinewarmup":
            lr_cfg['part'] = [pi * total_iter for pi in lr_cfg['part']]

    if lr_policy == "step":
        scheduler = lr_scheduler.MultiStepLR(optimizer, last_epoch=last_epoch, **lr_cfg)
    elif lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, last_epoch=last_epoch, **lr_cfg)
    elif lr_policy == "cosinewarmup":
        scheduler = CosineWarmUpLR(optimizer, last_epoch=last_epoch, **lr_cfg)
    elif lr_policy == "groupcosinewarmup":
        scheduler = GroupCosineWarmUpLR(optimizer, last_epoch=last_epoch, **lr_cfg)
    else:
        raise ValueError(f'scheduler={lr_policy} does not support.')

    return MyScheduler(scheduler, lr_type)
