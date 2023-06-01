from PIL import Image
import torch.utils.data as data
import torch
import os
from .dataloader import OurDataLoader
from .pipeline.transforms import build_transforms
from .build import DataPrefetcher, DATASET_REGISTERY

class ImageNetteDataset(data.Dataset):
    def __init__(self, root, pipeline=None, testmode=True):
        self.name2id = {'n03000684':0, 'n03028079':1, 'n03425413':2, 'n02102040':3, 'n03417042':4, 'n03394916':5, 'n02979186':6, 
                        'n03888257':7, 'n01440764':8, 'n03445777':9}

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.pipeline = pipeline
        
        lines = open("imagenet_label/noisy_imagenette.csv").read().strip().split("\n")[1:]
        self.imgs = {}
        self.fns = []
        self.labels = []
        for l in lines:
            if str(testmode) in l:
                v = l.strip().split(",")
                assert v[1] in v[0]
                self.fns.append(os.path.join(root, v[0]))
                self.labels.append(self.name2id[v[1]])

    def __len__(self):
        return len(self.fns)

    def __open_img(self, idx):
        if idx in self.imgs:
            return self.imgs[idx]
        else:
            img = Image.open(self.fns[idx]).convert("RGB")
            self.imgs[idx] = img

            return img

    def __getitem__(self, idx):
        img = self.__open_img(idx)
        label = self.labels[idx]
        if self.pipeline is None:
            return img, label

        if isinstance(self.pipeline, list):
            img = list(map(lambda t:t(img), self.pipeline))
        else:
            img = self.pipeline(img)
        
        return img, label

def collate_nolabel(data):
    inputs, targets = zip(*data)
    views = zip(*inputs)
    inputs = list(map(lambda view:torch.stack(view), views))
    
    return inputs

def collate_label(data):
    inputs, targets = zip(*data)
    inputs = torch.stack(inputs)
    targets = torch.tensor(targets)

    return inputs, targets

@DATASET_REGISTERY.register
class ImageNette():
    def __init__(self, batchsize, num_workers, usemultigpu=False, trainmode="linear", transform_train=None, testmode="linear", transform_test=None, root=None):
        
        assert transform_train is not None and transform_test is not None

        self.transform_train = build_transforms(transform_train)
        self.transform_test = build_transforms(transform_test)
        
        self.usemultigpu = usemultigpu
        
        self.trainset = ImageNetteDataset(root, self.transform_train, testmode=False)
        self.testset = ImageNetteDataset(root, self.transform_test, testmode=True)

        if self.usemultigpu:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
        else:
            self.train_sampler = None
        self.test_sampler = None

        if trainmode != 'linear':
            collate_fn = collate_nolabel
        else:
            collate_fn = collate_label

        if self.usemultigpu:
            self.train_loader = DataPrefetcher(torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, 
                                        shuffle=not self.usemultigpu, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, sampler=self.train_sampler))
        else:
            self.train_loader = DataPrefetcher(OurDataLoader(self.trainset, batch_size=batchsize, 
                                        shuffle=not self.usemultigpu, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, sampler=self.train_sampler))
        self.test_loader = DataPrefetcher(torch.utils.data.DataLoader(self.testset, batch_size=batchsize, 
                                shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_label, sampler=self.test_sampler))

    def get_loader(self):
        return self.train_loader, self.test_loader

    def set_epoch(self, epoch):
        if self.usemultigpu:
            self.train_sampler.set_epoch(epoch)

    def testsize(self):
        return self.testset.__len__()
    
    def trainsize(self):
        return self.trainset.__len__()