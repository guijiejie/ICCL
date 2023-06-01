import torch
import torch.nn as nn
import time
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast


class emptyenv(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        return False

    def __call__(self, func):
        pass

global_epoch = 0

def frozen_parameters(model, outputs):
    if 'frozen' in outputs:
        for name, p in model.named_parameters():
            for frozenname in outputs['frozen']:
                if frozenname in name:
                    p.grad = None
                    break
    
        outputs['frozen'] = torch.tensor(1.0)

class epoch_train(object):
    def __init__(self, device, logger, saver, update_interval, mixed):
        self.device = device
        self.logger = logger
        self.saver = saver
        self.update_interval = update_interval
        self.mixed = mixed
        if self.mixed:
            self.caster = autocast
            self.scaler = GradScaler()
        else:
            self.caster = emptyenv
    
    def __call__(self, model, epoch, data_loader, optimizer, scheduler):
        model.train()
        global global_epoch
        global_epoch = epoch
        it = 0
        total_iter = data_loader.__len__()
        start_time = time.time()
        self.logger.new_epoch()

        optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            it += 1
            # inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            loginfo = {
                'mode': "train",
                'datatime': round(time.time() - start_time, 4),
                'epoch': epoch,
                'iter': it,
                'lr': optimizer.param_groups[0]['lr'],
                'total': total_iter,
                'batchsize': inputs.size(0) if isinstance(inputs, torch.Tensor) else inputs[0].size(0)
            }
            if hasattr(model, "before_train_iter"):
                model.before_train_iter(it, total_iter, epoch)

            with self.caster():
                outputs = model(inputs, targets)
                loss = outputs['loss']

                if self.update_interval > 1:
                    loss = loss / self.update_interval
            
            if self.mixed:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            frozen_parameters(model, outputs)

            if it % self.update_interval == 0:
                if self.mixed:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            if hasattr(model, "after_train_iter"):
                model.after_train_iter(it, total_iter, epoch)
            
            if len(optimizer.param_groups) > 1:
                outputs['group0_lr'] = torch.tensor(optimizer.param_groups[0]['lr'])
                outputs['group1_lr'] = torch.tensor(optimizer.param_groups[1]['lr'])

            scheduler.step_iter()

            loginfo['time'] = round(time.time() - start_time, 4)
            start_time = loginfo['time'] + start_time

            self.logger.record_train(loginfo, outputs)
            
            self.saver.save(epoch, model, optimizer)
        
        if it % self.update_interval != 0:
            if self.mixed:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step_epoch()
        self.saver.saveepoch(epoch, model, optimizer)

class epoch_train_multigpu(object):
    def __init__(self, device, logger, saver, update_interval, mixed):
        self.device = device
        self.logger = logger
        self.saver = saver
        self.update_interval = update_interval
        self.mixed = mixed

        if self.mixed:
            self.caster = autocast
            self.scaler = GradScaler()
        else:
            self.caster = emptyenv

    
    def __call__(self, model, epoch, data_loader, optimizer, scheduler):
        model.train()
        global global_epoch
        global_epoch = epoch
        it = 0
        total_iter = data_loader.__len__()
        start_time = time.time()
        self.logger.new_epoch()

        optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            it += 1
            
            # inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            loginfo = {
                'mode': "train",
                'datatime': round(time.time() - start_time, 4),
                'epoch': epoch,
                'iter': it,
                'lr': optimizer.param_groups[0]['lr'],
                'total': total_iter,
                'batchsize': inputs.size(0) if isinstance(inputs, torch.Tensor) else inputs[0].size(0)
            }

            if hasattr(model.module, "before_train_iter"):
                model.module.before_train_iter(it, total_iter, epoch)

            if it % self.update_interval == 0 or it == total_iter:
                with self.caster():
                    outputs = model(inputs, targets)
                    loss = outputs['loss']

                    if self.update_interval > 1:
                        loss = loss / self.update_interval

                if self.mixed:
                    self.scaler.scale(loss).backward()
                    frozen_parameters(model, outputs)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    frozen_parameters(model, outputs)
                    optimizer.step()
                optimizer.zero_grad()
            else:
                with model.no_sync():
                    with self.caster():
                        outputs = model(inputs, targets)
                        loss = outputs['loss']

                        if self.update_interval > 1:
                            loss = loss / self.update_interval

                    if self.mixed:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

            if hasattr(model.module, "after_train_iter"):
                model.module.after_train_iter(it, total_iter, epoch)

            if len(optimizer.param_groups) > 1:
                outputs['group0_lr'] = torch.tensor(optimizer.param_groups[0]['lr'])
                outputs['group1_lr'] = torch.tensor(optimizer.param_groups[1]['lr'])
                
            scheduler.step_iter()
            
            loginfo['time'] = round(time.time() - start_time, 4)
            start_time = loginfo['time'] + start_time
            
            self.logger.record_train(loginfo, outputs)

            self.saver.save(epoch, model, optimizer)
            
        scheduler.step_epoch()
        self.saver.saveepoch(epoch, model, optimizer)
        torch.distributed.barrier()