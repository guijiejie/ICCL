import argparse
import os
from mmcv import Config
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="configs/imagenette/icc.py", type=str)
parser.add_argument('--workdir', default="result", type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--mode', default="SSModel", type=str)
parser.add_argument('--multigpu', default=0, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--syncbn', default=1, type=int)
parser.add_argument('--mixed', default=1, type=int)
parser.add_argument('--jizhi_report', default=1, type=int)

def reinit(old_ckpt, args, device, saver, test_loader, testsize):
    usemultigpu = (args.multigpu > 0)

    config = Config.fromfile(args.config)
    model = build_model(config, args.mode)
    if usemultigpu:
        torch.distributed.barrier()
        
        if args.syncbn == 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(device), device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = model.to(device)

    state_dict = {
        'optimizer':old_ckpt['optimizer'],
        'state_dict':{}
    }
    for k, v in old_ckpt['state_dict'].items():
        if "head.multiview_head" not in k:
            state_dict['state_dict'][k] = v
    
    for param_group in state_dict['optimizer']['param_groups'][1:]:
        for pid in param_group['params']:
            del state_dict['optimizer']['state'][pid]
    
    optimizer = build_optimizer(config, model, state_dict, logger, strict=False)
    scheduler = build_lrscheduler(config, optimizer, train_loader.__len__(), old_ckpt['epoch'])
    total_epoch = config["total_epochs"]
    
    evaluate = register_val(config.get("Evaluate", None), model, device, test_loader, testsize, logger, config)

    mixedprecision = (args.mixed == 1) and config.get("mixed_precision", True)
    update_interval = config['update_interval'] if 'update_interval' in config else 1

    train = epoch_train_multigpu(device, logger, saver, update_interval, mixedprecision) if usemultigpu else epoch_train(device, logger, saver, update_interval, mixedprecision)

    return model, optimizer, scheduler, evaluate, train

if __name__ == "__main__":
    args = parser.parse_args()
    
    usemultigpu = (args.multigpu > 0)
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if usemultigpu:
        device=torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=args.multigpu)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.environ['RANK'] = "0"

    config = Config.fromfile(args.config)
    
    if usemultigpu:
        config['dataset']['num_workers'] = int(config['dataset']['num_workers'] / args.multigpu) if config['dataset']['num_workers'] >= args.multigpu else 1
        config['dataset']['batchsize'] = int(config['dataset']['batchsize'] / args.multigpu)

    dataset = build_dataloader(config, usemultigpu)
    testsize = dataset.testsize()
    train_loader, test_loader = dataset.get_loader()

    logger = Logger(args, config)
    logger.print(args.config)
    logger.print(f"{config.text}")
    logger.print(f"{dataset.transform_train}")
    logger.print(f"{dataset.transform_test}")
    saver = Saver(config, logger)

    if args.resume is None:
        print('==> Training from scratch..')
        ckpt = None
        start_epoch = -1
    else:
        print(f'==> Resuming from {args.resume}..')
        ckpt = torch.load(args.resume, map_location='cpu')
        start_epoch = ckpt['epoch'] - 1
        logger.setsteps(ckpt['epoch'], len(train_loader))

    model = build_model(config, args.mode)
    if usemultigpu:
        if args.syncbn == 1:
            logger.print("trans to sync bn.")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(device), device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = model.to(device)
    logger.print(f"{model}")

    optimizer = build_optimizer(config, model, ckpt, logger)

    scheduler = build_lrscheduler(config, optimizer, train_loader.__len__(), start_epoch)
    total_epoch = config["total_epochs"]
    
    evaluate = register_val(config.get("Evaluate", None), model, device, test_loader, testsize, logger, config)
    evaluate_interval = config['evaluate_interval'] if 'evaluate_interval' in config else 1

    update_interval = config['update_interval'] if 'update_interval' in config else 1

    cudnn.benchmark = True

    logger.print(f"sync bn:{args.syncbn==1}")
    mixedprecision = (args.mixed == 1) and config.get("mixed_precision", True)
    logger.print(f"mixed precision:{mixedprecision}")

    train = epoch_train_multigpu(device, logger, saver, update_interval, mixedprecision) if usemultigpu else epoch_train(device, logger, saver, update_interval, mixedprecision)

    parttrain = config.get("parttrain", None)
    logger.print(f"{parttrain}")
    if parttrain is not None:
        for epoch in range(start_epoch + 1, total_epoch):
            if epoch in parttrain:
                state_dict = {
                    "state_dict":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "epoch":epoch-1,
                }
                model, optimizer, scheduler, evaluate, train = reinit(state_dict, args, device, saver, test_loader, testsize)

            dataset.set_epoch(epoch)

            train(model, epoch, train_loader, optimizer, scheduler)

            if (epoch + 1) % evaluate_interval == 0:
                if args.local_rank == 0:
                    evaluate(epoch)
        
        saver.saveend(total_epoch, model, optimizer)
    else:
        for epoch in range(start_epoch + 1, total_epoch):
            dataset.set_epoch(epoch)

            train(model, epoch, train_loader, optimizer, scheduler)

            if (epoch + 1) % evaluate_interval == 0:
                if args.local_rank == 0:
                    evaluate(epoch)
        
        saver.saveend(total_epoch, model, optimizer)
        
        
        
