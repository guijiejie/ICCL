import os
from PIL import Image
import torch.utils.data as data
import torch
import copy

import random
    
class ImageList(object):
    def __init__(self, root, list_file, num_class, preload=False):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.preload = preload
        self.has_labels = len(lines[0].split()) == 2

        self.copy_dockerdata = os.path.isdir("/dockerdata") and False
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split() for l in lines])
            self.labels = [int(l) for l in self.labels]
            for l in self.labels:
                assert l < num_class
        else:
            self.fns = [l.strip() for l in lines]

        if self.copy_dockerdata:
            if self.rank == 0:
                if not os.path.isdir("/dockerdata/dataset"):
                    os.mkdir("/dockerdata/dataset")

            self.dockerroot = "/dockerdata/dataset"

            self.rawfns = [os.path.join(root, fn) for fn in self.fns]

            self.fns = [os.path.join(self.dockerroot, os.path.basename(fn)) for fn in self.fns]
        else:
            self.fns = [os.path.join(root, fn) for fn in self.fns]

        if self.preload:
            self.data_infos = []
            for idx in range(len(self.fns)):
                img = Image.open(self.fns[idx])
                img = img.convert('RGB')
                target = self.labels[idx]
                self.data_infos.append([img, target])
        
    def __copydata(self, idx):
        if self.copy_dockerdata and not os.path.exists(self.fns[idx]):
            os.system("cp {} {}".format(self.rawfns[idx], self.fns[idx]))

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.preload:
            return self.data_infos[idx]
        self.__copydata(idx)
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')

        if self.has_labels:
            target = self.labels[idx]
            return img, target
        else:
            return img

    def get_sample_vis(self, idx):
        if self.preload:
            return self.data_infos[idx]

        img = Image.open(self.fns[idx])
        img = img.convert('RGB')

        if self.has_labels:
            target = self.labels[idx]
            return img, target, self.fns[idx]
        else:
            return img, self.fns[idx]
    
    
    def has_labels(self):
        return self.has_labels

class ImageNet_Dataset(data.Dataset):
    def __init__(self, root, transforms, list_file, num_class, preload=False, copydockerdata=True):
        super(ImageNet_Dataset, self).__init__()
        self.root = root
        self.list_file = list_file
        self.num_class = num_class
        
        self.imagelist = ImageList(root, list_file, num_class, preload)

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
        if self.pipeline is not None:
            if isinstance(self.pipeline, list):
                multi_crops = list(map(lambda trans: trans(img), ))
            else:
                multi_crops = self.pipeline(img)
            if len(multi_crops) == 2:
                return multi_crops[0], multi_crops[1]
            else:
                _ ,C, H, W = multi_crops[0].size()
                start = 0
                for i in range(len(multi_crops)):
                    if multi_crops[i].size()[2] != H:
                        crops1 = torch.cat(multi_crops[start:i])
                        crops2 = torch.cat(multi_crops[i:])
                        break
                
                return crops1, crops2

    def __getitem__(self, idx):
        return self.item_func(idx)