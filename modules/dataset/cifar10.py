import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
from functools import partial
from mmcv.parallel import collate
import copy
import numpy as np
from PIL import Image

from .dataloader import OurDataLoader
from .build import DataPrefetcher, DATASET_REGISTERY

from .pipeline.transforms import build_transforms

class SS_CIFAR(data.Dataset):
    def __init__(self, root, train, transform):
        super(SS_CIFAR, self).__init__()

        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train)

        self.pipeline = transform

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        img, _ = self.dataset.__getitem__(idx)

        if isinstance(self.pipeline, list):
            aug1 = self.pipeline[0](img)
            aug2 = self.pipeline[1](img)
        else:
            aug1, aug2 = self.pipeline(img)
            
        return aug1, aug2

class MIX_CIFAR(data.Dataset):
    def __init__(self, root, train, transform):
        super(MIX_CIFAR, self).__init__()

        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train)

        if isinstance(transform, list):
            self.pipeline1 = transform[0]
            self.pipeline2 = transform[1]
        else:
            self.pipeline1 = copy.deepcopy(transform)
            self.pipeline2 = transform

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        img, target = self.dataset.__getitem__(idx)

        aug1 = self.pipeline1(img)
        aug2 = self.pipeline2(img)
        ret = torch.stack((aug1, aug2), dim=0)
        return ret, target

class LESS_CIFAR(data.Dataset):
    def __init__(self, root, train, transform, maxnum=20):
        super().__init__()

        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train)
        self.rawdata = self.dataset.data
        self.rawtarget = self.dataset.targets

        self.datapool = {k:[] for k in range(10)}
        for i in range(len(self.rawtarget)):
            if len(self.datapool[self.rawtarget[i]]) < maxnum:
                self.datapool[self.rawtarget[i]].append(self.rawdata[i])
        
        self.data = []
        self.targets = []

        for k, v in self.datapool.items():
            for img in v:
                self.data.append(img)
                self.targets.append(k)
        
        self.data = np.array(self.data)

        self.transform = transform

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        target = self.targets[idx]

        img = self.transform(img)
        return img, target

__MODE_DICT__ = {
    "linear":torchvision.datasets.CIFAR10,
    "selfsupervisied":SS_CIFAR,
    "mix":MIX_CIFAR,
    "less":LESS_CIFAR
}

@DATASET_REGISTERY.register
class CIFAR10():
    def __init__(self, train_root, test_root, batchsize, num_workers, usemultigpu=False, num_class=10, 
                    trainmode="linear", transform_train=None, testmode="linear", transform_test=None):
        assert trainmode in __MODE_DICT__ and testmode in __MODE_DICT__
        assert transform_train is not None and transform_test is not None

        self.transform_train = build_transforms(transform_train)
        self.transform_test = build_transforms(transform_test)

        self.usemultigpu = usemultigpu

        self.trainset = __MODE_DICT__[trainmode](root=train_root, train=True, transform=self.transform_train)
        self.testset = __MODE_DICT__[testmode](root=test_root, train=False, transform=self.transform_test)

        if self.usemultigpu:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
        else:
            self.train_sampler = None
        self.test_sampler = None

        self.train_loader = DataPrefetcher(OurDataLoader(self.trainset, batch_size=batchsize, shuffle=not self.usemultigpu, 
                                                        num_workers=num_workers, pin_memory=True, sampler=self.train_sampler))

        self.test_loader = DataPrefetcher(torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=False, 
                                                    num_workers=num_workers, pin_memory=True, sampler=self.test_sampler))

    def get_loader(self):
        return self.train_loader, self.test_loader
    
    def set_epoch(self, epoch):
        if self.usemultigpu:
            self.train_sampler.set_epoch(epoch)

    def testsize(self):
        return self.testset.__len__()