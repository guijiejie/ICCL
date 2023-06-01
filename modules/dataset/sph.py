import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
PROCESS = False
if PROCESS:
    from .dataset_type.sphprocess import SPH_SS, SPH_Linear
else:
    from .dataset_type import SPH_SS, SPH_Linear

from .dataset_type import Base64Dataset

from .build import DataPrefetcher, DataPrefetcherProcess, MultiModalPrefetcher, DATASET_REGISTERY
import copy
from .pipeline.transforms import build_transforms

import random

def collate_function(data):
    if len(data) > 1:
        inputs = []
        targets = []
        for _ in range(len(data)):
            inputs += data[_][0]
            targets += data[_][1]
    else:
        inputs = data[0][0]
        targets = data[0][1]

    shuffle_data = list(zip(inputs, targets))

    random.shuffle(shuffle_data)

    inputs, targets = zip(*shuffle_data)
    return inputs, targets
    
__MODE_DICT__ = {
    "boundingbox":SPH_Linear,
    "linear":SPH_Linear,
    "selfsupervised":SPH_SS,
    "multimodal":Base64Dataset
}
@DATASET_REGISTERY.register
class SPH():
    def __init__(self, train_root, test_root, train_list, test_list, batchsize, num_workers, usemultigpu=False, 
                        trainmode="linear", transform_train=None, testmode="linear", transform_test=None, copy_dockerdata=True, testbatchsize=None, appendmask=False):
        
        assert transform_train is not None and transform_test is not None

        self.train_preload = build_transforms(transform_train[0])
        self.transform_train = build_transforms(transform_train[1])

        self.test_preload = build_transforms(transform_test[0])
        self.transform_test = build_transforms(transform_test[1])
        
        self.usemultigpu = usemultigpu
        
        if appendmask:
            other_params = dict(appendmas =True)
        else:
            other_params = dict()
        
        if PROCESS:
            self.trainset = __MODE_DICT__[trainmode](train_root, self.train_preload, self.transform_train, train_list, copy_dockerdata=copy_dockerdata, batch_size=batchsize, appendmask=appendmask)

            self.testset = SPH_Linear(test_root, self.test_preload, self.transform_test, test_list, copy_dockerdata=copy_dockerdata, batch_size=batchsize, **other_params)
        else:
            self.trainset = __MODE_DICT__[trainmode](train_root, self.train_preload, self.transform_train, train_list, copy_dockerdata=copy_dockerdata, **other_params)

            self.testset = SPH_Linear(test_root, self.test_preload, self.transform_test, test_list, copy_dockerdata=copy_dockerdata)

        if self.usemultigpu:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
        else:
            self.train_sampler = None
        self.test_sampler = None
        
        if PROCESS:
            self.train_loader = DataPrefetcherProcess(self.trainset)
            self.test_loader = DataPrefetcherProcess(self.testset)
        else:
            if trainmode == "multimodal":
                self.train_loader = MultiModalPrefetcher(torch.utils.data.DataLoader(self.trainset, batch_size=2 if isinstance(self.transform_train, list) else 4, 
                                    shuffle=not self.usemultigpu, num_workers=0, pin_memory=False, collate_fn=collate_function, sampler=self.train_sampler), batch_size=batchsize, pickle_size=2048)
            else:
                self.train_loader = DataPrefetcher(torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, 
                                        shuffle=not self.usemultigpu, num_workers=num_workers, pin_memory=True, sampler=self.train_sampler))

            self.test_loader = DataPrefetcher(torch.utils.data.DataLoader(self.testset, batch_size=testbatchsize if testbatchsize is not None else batchsize, 
                                    shuffle=False, num_workers=num_workers, pin_memory=True, sampler=self.test_sampler))

    def get_loader(self):
        return self.train_loader, self.test_loader

    def set_epoch(self, epoch):
        if self.usemultigpu:
            self.train_sampler.set_epoch(epoch)
        
        if hasattr(self.trainset, "assignWorks"):
            self.trainset.assignWorks(self.train_sampler)

    def testsize(self):
        return self.testset.__len__()
    
    def trainsize(self):
        return self.trainset.__len__()
