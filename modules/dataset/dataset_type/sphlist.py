import os
from PIL import Image
import torch.utils.data as data
import copy
# from ..pipeline import tarfile
from ..pipeline import lines_detector_np
import tarfile
import torch
import io
import pickle
import random
import numpy as np

class PythonTar(object):
    def __init__(self, path):
        self.t = tarfile.open(path)
    
    def IsValidTarFile(self):
        self.names = [fn for fn in self.t.getnames() if "jpeg" in fn or "jpg" in fn or "png" in fn]
        return True

    def GetFileNames(self):
        self.names.sort(key = (lambda a:int(a[::-1][a[::-1].find(".")+1:a[::-1].find("/")][::-1])))
        return self.names

    def GetContents(self, name):
        return self.t.extractfile(name).read()

class SPHList(object):
    def __init__(self, root, list_file, transforms_preload, copy_dockerdata=True, boundingbox=False):
        if isinstance(root, list):
            if os.path.isdir(root[0]):
                self.root = root[0]
            else:
                self.root = root[1]
        else:
            self.root = root
            
        self.transforms_preload = transforms_preload

        self.copy_dockerdata = copy_dockerdata if os.path.isdir("/dockerdata") else False

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if self.copy_dockerdata:
            if self.rank == 0:
                if not os.path.isdir("/dockerdata/dataset"):
                    os.mkdir("/dockerdata/dataset")

            self.dockerroot = "/dockerdata/dataset"
            
            saveroot = self.dockerroot
        else:
            saveroot = self.root
        
        self.boundingbox = boundingbox
        
        with open(list_file, "r") as f:
            lines = f.readlines()
            self.fns = []
            for l in lines:
                imgtuple = l.strip().split(" ")
                self.fns.append((os.path.join(saveroot, imgtuple[0]), *list(map(int, imgtuple[1:])) ))

    def __copydata(self, fn):
        if self.copy_dockerdata and not os.path.exists(fn):
            if 'supply1' in fn:
                os.system("cp {} {}".format(os.path.join(self.root, "supply1", os.path.basename(fn)),fn))
            else:
                os.system("cp {} {}".format(os.path.join(self.root, os.path.basename(fn)),fn))

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        self.__copydata(self.fns[idx][0])
        # tar = yuvalTar.TarFile(self.fns[idx][0])
        tar = PythonTar(self.fns[idx][0])

        if not tar.IsValidTarFile():
            print(self.fns[idx])
            raise EnvironmentError
            return []
        
        img_list = tar.GetFileNames()
        max_length = len(img_list)

        if max_length <= 0:
            randomid = random.randint(0, self.get_length() - 1)
            return self.get_sample(randomid)

        if self.boundingbox:
            img = Image.open(io.BytesIO(tar.GetContents(img_list[int(max_length // 2)]))).convert("RGB")
            bbox = lines_detector_np(img)

        fids = self.transforms_preload(len(img_list))

        if isinstance(fids[0], list):
            ret = [[], []]

            for ind in fids[0]:
                img = Image.open(io.BytesIO(tar.GetContents(img_list[min(ind, max_length-1)])))
                img = img.convert('RGB')
                if self.boundingbox:
                    img = img.crop(bbox)

                ret[0].append(img)

            for ind in fids[1]:
                img = Image.open(io.BytesIO(tar.GetContents(img_list[min(ind, max_length-1)])))
                img = img.convert('RGB')
                if self.boundingbox:
                    img = img.crop(bbox)
                ret[1].append(img)
        else:
            ret = []
            for ind in fids:
                img = Image.open(io.BytesIO(tar.GetContents(img_list[min(ind, max_length-1)])))
                img = img.convert('RGB')
                if self.boundingbox:
                    img = img.crop(bbox)
                ret.append(img)

        return [ret, *self.fns[idx][1:]]

class SPH_SS(data.Dataset):
    def __init__(self, root, transform_preload, transforms, list_file, boundingbox=False, copy_dockerdata=True):
        super(SPH_SS, self).__init__()
        self.root = root
        self.list_file = list_file

        self.imagelist = SPHList(root, list_file, transform_preload, copy_dockerdata)

        self.pipeline = transforms

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if copy_dockerdata and os.path.isdir("/dockerdata"):
            if self.rank == 0:
                if not os.path.isdir("/dockerdata/dataset"):
                    os.mkdir("/dockerdata/dataset")

                if not os.path.isdir("/dockerdata/dataset/supply1"):
                    os.mkdir("/dockerdata/dataset/supply1")

    def __len__(self):
        return self.imagelist.get_length()

    def __getitem__(self, idx):
        imgs = self.imagelist.get_sample(idx)[0]
        if isinstance(imgs[0], list):
            ret1 = []
            ret2 = []
            ret3 = []
            ret4 = []
            for img in imgs[0]:
                multi_crops = list(map(lambda trans: trans(img), self.pipeline))
                ret1.append(multi_crops[0])
                ret2.append(multi_crops[1])

            for img in imgs[1]:
                multi_crops = list(map(lambda trans: trans(img), self.pipeline))
                ret3.append(multi_crops[0])
                ret4.append(multi_crops[1])
            
            ret1 = torch.stack(ret1)
            ret2 = torch.stack(ret2)
            ret3 = torch.stack(ret3)
            ret4 = torch.stack(ret4)

            return torch.stack((ret1, ret3)), torch.stack((ret2, ret4))
        else:
            ret1 = []
            ret2 = []
            for img in imgs:
                multi_crops = list(map(lambda trans: trans(img), self.pipeline))
                ret1.append(multi_crops[0])
                ret2.append(multi_crops[1])
            ret1 = torch.stack(ret1)
            ret2 = torch.stack(ret2)
            return ret1.unsqueeze(0), ret2.unsqueeze(0)

class SPH_Linear(data.Dataset):
    def __init__(self, root, transform_preload, transforms, list_file, boundingbox=False, copy_dockerdata=True):
        super(SPH_Linear, self).__init__()
        self.root = root
        self.list_file = list_file

        self.boundingbox = boundingbox

        self.imagelist = SPHList(root, list_file, transform_preload, copy_dockerdata, boundingbox=self.boundingbox)

        self.pipeline = transforms

        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        if copy_dockerdata and os.path.isdir("/dockerdata"):
            if self.rank == 0:
                if not os.path.isdir("/dockerdata/dataset"):
                    os.mkdir("/dockerdata/dataset")

                if not os.path.isdir("/dockerdata/dataset/supply1"):
                    os.mkdir("/dockerdata/dataset/supply1")

    def __len__(self):
        return self.imagelist.get_length()
    
    def __getitem__(self, idx):
        ret = self.imagelist.get_sample(idx)
        imgs = ret[0]
        left_ret = ret[1:]

        if isinstance(imgs[0], list):
            ret1 = []
            ret2 = []
            for img in imgs[0]:
                ret1.append(self.pipeline(img))

            for img in imgs[1]:
                ret2.append(self.pipeline(img))
            
            ret1 = torch.stack(ret1)
            ret2 = torch.stack(ret2)
            
            return torch.stack((ret1, ret2)), torch.tensor(left_ret)
        else:
            ret = []
            for img in imgs:
                ret.append(self.pipeline(img))

            ret = torch.stack(ret)

            return ret.unsqueeze(0), torch.tensor(left_ret)
