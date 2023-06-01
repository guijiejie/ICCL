from thop import profile
import argparse

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv import Config
import torch
import random
import numpy as np

from modules import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="configs/resnet50_pws.py", type=str)
parser.add_argument('--gpu_ids', default=[0], type=list)
parser.add_argument('--seed', default=0, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    config = Config.fromfile(args.config)
    batchsize = config.dataset.batchsize
    
    dataset = data_loader(config)
    testsize = dataset.testsize()
    train_loader, test_loader = dataset.get_loader()

    logger = Logger(args, config, save_file=False)
    logger.print(f"{config.text}")

    model = build_model(config)
    model = model.to(device)
    backbone = model.backbone
    # model = MMDataParallel(model.to(device), device_ids=args.gpu_ids)
    
    optimizer = build_optimizer(config, model)

    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

    num_params = 0
    for param in backbone.parameters():
        num_params += param.numel()
    print("params:{}".format(num_params / 1e6))

    numsize = 100
    model.train()
    torch.cuda.synchronize()
    start = time.time()
    prog_bar = mmcv.ProgressBar(numsize)
    for _ in range(numsize):
        optimizer.zero_grad()

        outputs = model(inputs, targets)
        loss = outputs['loss']

        loss.backward()
        optimizer.step()

        prog_bar.update()
    totaltime = time.time() - start
    torch.cuda.synchronize()

    print("training time:{}".format(totaltime/float(numsize)))

    model.eval()

    prog_bar = mmcv.ProgressBar(numsize)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(numsize):
            outputs = model(inputs, targets)
            prog_bar.update()
    totaltime = time.time() - start
    torch.cuda.synchronize()

    print("inference time:{}".format(totaltime/float(numsize)))
