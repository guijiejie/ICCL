import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import copy
from .pipeline.transforms import build_transforms
import random

import os
from PIL import Image
import torch.utils.data as data
import torch
import copy

import random

from multiprocessing import Process
from .utils import unzipimagenet
from .build import DATASET_REGISTERY

class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)
    
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            if isinstance(self.next_input, list):
                self.next_input = list(map(lambda x:x.cuda(non_blocking=True), self.next_input))
            else:
                self.next_input = self.next_input.cuda(non_blocking=True)
            
            if isinstance(self.next_target, list):
                self.next_target =list(map(lambda x:x.cuda(non_blocking=True), self.next_target))
            else:
                self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break

class ImageList(object):
    def __init__(self, root, list_file, num_class, copy_dockerdata=False):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.has_labels = len(lines[0].split()) == 2

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if self.rank == 0 and os.path.exists("/dockerdata") and copy_dockerdata:
            self.copy_process = Process(target=unzipimagenet)
            self.copy_process.start()
        else:
            self.copy_process = None

        self.mode = True

        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split() for l in lines])
            self.labels = [int(l) for l in self.labels]
            for l in self.labels:
                assert l < num_class
        else:
            self.fns = [l.strip() for l in lines]

        if 'imagenet/train' in root:
            self.copyfns = [os.path.join("/dockerdata/imagenet/train", fn) for fn in self.fns]
        elif 'imagenet/val' in root:
            self.copyfns = [os.path.join("/dockerdata/imagenet/val", fn) for fn in self.fns]

        self.fns = [os.path.join(root, fn) for fn in self.fns]
        
    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if os.path.exists("/dockerdata/imagenet_copy_complete.txt"):
            img = Image.open(self.copyfns[idx])
        else:
            img = Image.open(self.fns[idx])
        img = img.convert('RGB')

        if self.has_labels:
            target = self.labels[idx]
            return img, target
        else:
            return img
    
    def has_labels(self):
        return self.has_labels

class ImageNet_Dataset(data.Dataset):
    def __init__(self, root, transforms, list_file, num_class, copy_dockerdata=False):
        super(ImageNet_Dataset, self).__init__()
        self.root = root
        self.list_file = list_file
        self.num_class = num_class
        
        self.imagelist = ImageList(root, list_file, num_class, copy_dockerdata=copy_dockerdata)

        self.pipeline = transforms

        if self.imagelist.has_labels:
            self.item_func = self.__get_labeled_item
        else:
            self.item_func = self.__get_unlabeled_item

    def __len__(self):
        return self.imagelist.get_length()
    
    def __get_labeled_item(self, idx):
        img, target = self.imagelist.get_sample(idx)
        if self.pipeline is not None:
            img = self.pipeline(img)
        
        return img, target

    def __get_unlabeled_item(self, idx):
        img = self.imagelist.get_sample(idx)

        if isinstance(self.pipeline, list):
            ret = list(map(lambda t:t(img), self.pipeline))
        else:
            ret = self.pipeline(img)

        if len(ret) > 2:
            ret = [ret[0:2], ret[2:]]
        return ret

    def __getitem__(self, idx):
        return self.item_func(idx)

__MODE_DICT__ = {
    "linear":ImageNet_Dataset,
}

@DATASET_REGISTERY.register
class ImageNet():
    def __init__(self, train_root, test_root, train_list, test_list, batchsize, num_workers, usemultigpu=False, num_class=1000, 
                        trainmode="linear", transform_train=None, testmode="linear", transform_test=None, copy_dockerdata=False):
        
        assert transform_train is not None and transform_test is not None

        self.transform_train = build_transforms(transform_train)
        self.transform_test = build_transforms(transform_test)
        
        self.usemultigpu = usemultigpu
        
        self.trainset = ImageNet_Dataset(train_root, self.transform_train, train_list, num_class, copy_dockerdata=copy_dockerdata)

        self.testset = ImageNet_Dataset(test_root, self.transform_test, test_list, num_class)

        if self.usemultigpu:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
        else:
            self.train_sampler = None
        self.test_sampler = None

        self.train_loader = DataPrefetcher(torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, 
                            shuffle=not self.usemultigpu, num_workers=num_workers, pin_memory=True, sampler=self.train_sampler))
        
        self.test_loader = DataPrefetcher(torch.utils.data.DataLoader(self.testset, batch_size=batchsize, 
                                shuffle=False, num_workers=num_workers, pin_memory=True, sampler=self.test_sampler))

    def get_loader(self):
        return self.train_loader, self.test_loader

    def set_epoch(self, epoch):
        if self.usemultigpu:
            self.train_sampler.set_epoch(epoch)

    def testsize(self):
        return self.testset.__len__()
    
    def trainsize(self):
        return self.trainset.__len__()
