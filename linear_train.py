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
parser.add_argument('--config', default="configs/imagenette/linear_probe.py", type=str)
parser.add_argument('--workdir', default="", type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--pretrained', default=None, type=str)
parser.add_argument('--mode', default="Model", type=str)
parser.add_argument('--multigpu', default=0, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--syncbn', default=0, type=int)
parser.add_argument('--mixed', default=0, type=int)
parser.add_argument('--jizhi_report', default=1, type=int)
parser.add_argument('--basedir', default="imagenet_all", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    usemultigpu = (args.multigpu > 0)
    
    if args.seed is not None:
        random.seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed_all(args.seed + args.local_rank)

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

    if args.pretrained is not None:
        config['pretrained'] = args.pretrained
    
    dataset = build_dataloader(config, usemultigpu)
    testsize = dataset.testsize()
    train_loader, test_loader = dataset.get_loader()

    logger = Logger(args, config)
    logger.print(f"{config.text}")
    logger.print(f"{dataset.transform_train}")
    logger.print(f"{dataset.transform_test}")
    saver = Saver(config, logger)

    if args.resume is None:
        if config["pretrained"] is not None:
            print('==> Training from {}..'.format(config["pretrained"]))
        else:
            print('==> Training from scratch..')
        ckpt = None
        start_epoch = -1
    else:
        print(f'==> Resuming from {args.resume}..')
        ckpt = torch.load(args.resume)
        start_epoch = ckpt['epoch']

    model = build_model(config, args.mode)
    if usemultigpu:
        if args.syncbn == 1 and config.get("SyncBN", True):
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

    mixedprecision = (args.mixed == 1) and config.get("mixed_precision", True)
    logger.print(f"mixed precision:{mixedprecision}")
    logger.print(f"sync bn:{args.syncbn==1}")

    train = epoch_train_multigpu(device, logger, saver, update_interval, mixedprecision) if usemultigpu else epoch_train(device, logger, saver, update_interval, mixedprecision)

    for epoch in range(start_epoch + 1, total_epoch):
        dataset.set_epoch(epoch)

        train(model, epoch, train_loader, optimizer, scheduler)
        if (epoch + 1) % evaluate_interval == 0 or (epoch + 1) == total_epoch:
            if args.local_rank == 0:
                evaluate(epoch)

    saver.saveend(total_epoch, model, optimizer)

    config_path = args.config
    eval_file_name = os.path.basename(config_path).replace("probe", "knn")
    if config_path.startswith("configs/"):
        config_path = config_path[config_path.find('/')+1:]
        config_path = config_path[:config_path.find('/')]
    
    # os.system(f"python3 linear_val.py --ckpt {args.pretrained} --mode {config_path} --config {eval_file_name}")
    
