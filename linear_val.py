import argparse
import sys
from mmcv.runner import load_checkpoint
from mmcv import Config
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T

from modules import *
from tqdm import tqdm
import mmcv
from einops import rearrange
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="linear_knn.py", type=str)
parser.add_argument('--ckpt', default="wechat_video_search_gpu_mixed_clip_mv_sberttrain_notextmv_2021_06_23_17", type=str)
parser.add_argument('--postfix', default="end", type=str)
parser.add_argument('--mode', default="imagenette", type=str)
parser.add_argument('--multigpu', default=0, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--eval', default=1, type=int)

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

t = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(**img_norm_cfg)
            ])

def open_img(path):
    img = Image.open(path).convert("RGB")
    img = t(img).unsqueeze(0)
    return img.cuda()

def f(path, model):
    img = open_img(path)
    v = model(img)
    v = v / v.norm(dim=1, keepdim=True)

    return v

def sim(v1, v2):
    print((v1 * v2).sum())

def testsim(model, dataloader, name):
    model = model.backbone
    model.eval()
    import clip
    clipmodel, t = clip.load(name)
    clipmodel.eval()

    prog_bar = mmcv.ProgressBar(len(dataloader))

    sim = 0.0
    cnts = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            N, I, C, D, H, W = inputs.size()

            inputs = rearrange(inputs, "n i c d h w -> (n i c) d h w")
            out = model(inputs)
            clipout = clipmodel.encode_image(inputs)
            
            out = out / out.norm(dim=1, keepdim=True)
            clipout = clipout / clipout.norm(dim=1, keepdim=True)

            sim += (out * clipout).sum()
            cnts += out.size(0)

            prog_bar.update()
        
        print(sim / cnts)

@torch.no_grad()
def tsne(model, test_loader):
    import matplotlib.pyplot as plt
    from sklearn import manifold,datasets
    from matplotlib.patches import Circle
    X = []
    Y = []
    for inputs, targets in test_loader:
        feats = model.backbone(inputs)
        X.append(feats.detach().cpu().clone().numpy())
        Y.append(targets.detach().cpu().clone().numpy())
    
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    '''X是特征，不包含target;X_tsne是已经降维之后的特征'''
    tsne = manifold.TSNE(n_components=2, init='random', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    for i in range(X_norm.shape[0]):
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(Y[i]), color=plt.cm.Set1(Y[i]), fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(X_norm[i, 0], X_norm[i, 1], c=plt.cm.Set1(Y[i]))
    plt.xticks([])
    plt.yticks([])
    plt.savefig('t-SNE.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config.fromfile(f"configs/{args.mode}/"+args.config)
    
    dataset = build_dataloader(config, usemultigpu=False)
    testsize = dataset.testsize()
    loader = dataset.get_loader()

    logger = Logger(args, config, save_file=False)
    
    model = build_model(config, args.mode)
    print(f'==> evaluate from {args.ckpt}..')
    ckpt_path = args.ckpt
    ckpt_path = ckpt_path.replace("/backbone.pth", "")
    
    load_checkpoint(model, args.ckpt)
    base_dir = f"{ckpt_path}/knnlog.txt"
    
    model = model.to(device)

    if args.eval == 1:
        evaluate = register_val(config.get("Evaluate", None), model, device, loader, testsize, config, logger)
        ans = evaluate()
        with open(base_dir, "w") as f:
            json.dump(ans, f)
        # evaluate.finaleval()