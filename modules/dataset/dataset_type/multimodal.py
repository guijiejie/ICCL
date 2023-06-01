import os
from PIL import Image
import torch.utils.data as data
import copy
# from ..pipeline import tarfile
from multiprocessing import Pool
from multiprocessing import Manager
import torch
import io
import random
import base64
import json
import numpy as np

def _pipeline_preprocess(worklist, vocab, outputs_queue, pipeline, seed):
    try:
        random.seed(seed)
        torch.manual_seed(seed)

        inputs = []
        targets = []

        for line in worklist:
            if len(line.split("\t")) < 3:
                line += "\t。"
            imgid, imgstr, title = line.split("\t")[0:3]

            img = Image.open(io.BytesIO(base64.b64decode(imgstr))).convert("RGB")
            if isinstance(pipeline, list):
                img = list(map(lambda t:t(img), pipeline))
            else:
                img = pipeline(img)
            # token = torch.tensor(vocab.sentence2id(title))

            inputs.append(img)
            targets.append(title)
        
        outputs_queue.put((inputs, targets))
    except Exception as e:
        print(e)

class PipelinePreprocess:
    def __init__(self, transforms, num_workers=2, rank=0):
        self.pipeline = transforms

        self.pool = Pool(num_workers)
        self.manager = Manager()
        self.outputs_queue = self.manager.Queue(num_workers)
        self.num_workers = num_workers
        self.rank = rank
        self.vocab = None
    
    def assignWorks(self, imgs):
        if len(imgs) % self.num_workers == 0:
            worksize = len(imgs) // self.num_workers
        else:
            worksize = (len(imgs) // self.num_workers) + 1

        base_seed = torch.LongTensor(1).random_()[0]
        for i in range(self.num_workers):
            worklist = imgs[i*worksize:i*worksize + worksize]
            self.pool.apply_async(func=_pipeline_preprocess, args=(worklist, self.vocab, self.outputs_queue, self.pipeline, base_seed + i + self.rank * self.num_workers, ))

    def get(self):
        inputs = []
        targets = []
        for _ in range(self.num_workers):
            t1, t2 = self.outputs_queue.get()
            inputs += t1
            targets += t2

        return inputs, targets

class Base64Dataset(data.Dataset):
    def __init__(self, root, transform_preload, transforms, list_file, copy_dockerdata=True, preprocess=True, appendmask=False):
        super().__init__()

        self.root = root
        self.copy_dockerdata = copy_dockerdata if os.path.isdir("/dockerdata") else False
        
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if self.copy_dockerdata:
            if self.rank == 0:
                if not os.path.isdir("/dockerdata/dataset"):
                    os.mkdir("/dockerdata/dataset")

            self.dockerroot = "/dockerdata/dataset"

            with open(list_file, "r") as f:
                self.rawfns = [os.path.join(self.root, fn.strip()) for fn in f.readlines()]

            with open(list_file, "r") as f:
                self.fns = [os.path.join(self.dockerroot, fn.strip().replace("/","_")) for fn in f.readlines()]
        else:
            with open(list_file, "r") as f:
                self.fns = [os.path.join(root, fn.strip()) for fn in f.readlines()]

        self.pipeline = transforms
        self.appendmask = appendmask
        self.preprocess = preprocess
        if preprocess:
            self.pipeline_preprocess = PipelinePreprocess(self.pipeline, num_workers=4, rank=self.rank)
        else:
            self.pipeline_preprocess = None

    def __len__(self):
        return len(self.fns)

    def __copydata(self, idx):
        if self.copy_dockerdata and not os.path.exists(self.fns[idx]):
            os.system("cp {} {}".format(self.rawfns[idx], self.fns[idx]))

    def addmask(self, fn, targets):
        basename = os.path.basename(fn)
        with open(f"/mnt/ceph/home/yuvalliu/clip_title_img/sim/{basename}.json", "r") as f:
            masks = np.array(json.load(f))
            masks = (masks >= 0.15).astype(np.long).tolist()
        
        targets = list(zip(targets, masks))

        return targets

    def __getitem__(self, idx):
        self.__copydata(idx)
        contents = open(self.fns[idx], "r").read().strip()
        lines = contents.split("\n")

        if self.preprocess:
            if len(lines) < 2048:
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        
            self.pipeline_preprocess.assignWorks(lines)
            inputs, targets = self.pipeline_preprocess.get()

            if self.appendmask:
                targets = self.addmask(self.fns[idx], targets)

            return inputs, targets
        else:
            inputs = []
            targets = []
            for l in lines:
                imgstr = l.split("\t")
                targets.append(int(imgstr[2]))
                imgstr = imgstr[1]

                img = Image.open(io.BytesIO(base64.b64decode(imgstr))).convert("RGB")
                if isinstance(self.pipeline, list):
                    img = list(map(lambda t:t(img), self.pipeline))
                else:
                    img = self.pipeline(img)
                
                inputs.append(img)
            randomimg = inputs[-1]
            while len(inputs) != 2048:
                inputs.append(randomimg)
                targets.append(-1)
            return inputs, targets