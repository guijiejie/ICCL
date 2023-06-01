import torch
from queue import Queue
import threading
from threading import Thread
import time
from multiprocessing import Pool

class dataLoaderThread(threading.Thread):
    def __init__(self, loaditer, maxsize):
        threading.Thread.__init__(self)

        self.loaditer = loaditer
        self.maxsize = maxsize
        self.queue = Queue(self.maxsize)
        self.finished = False

    def run(self):
        while True:
            if self.finished:
                break

            try:
                inputs, targets = next(self.loaditer)
            except StopIteration:
                self.finished = True
                break
                
            self.queue.put([inputs, targets])
    
    def iteration(self):
        if self.finished and self.queue.empty():
            return None, None
        
        inputs, targets = self.queue.get()
        return inputs, targets

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
        # self.next_input, self.next_target = self.loadthread.iteration()
        # if self.next_input is None:
        #     return

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
        # self.loadthread = dataLoaderThread(self.loaditer, 2)
        # self.loadthread.start()
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

class DataPrefetcherProcess():
    def __init__(self, dataset, stop_after=None):
        self.dataset = dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

        self.nowidx = 0

    def __len__(self):
        return self.dataset.iterLen()
    
    def preload(self):
        try:
            self.next_input, self.next_target = self.dataset.__getitem__(self.nowidx)
            self.nowidx += 1
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_input, list):
                self.next_input = torch.cat(self.next_input, dim=1)
            else:
                self.next_input = self.next_input.cuda(non_blocking=True)
            
            if isinstance(self.next_target, list):
                self.next_target = torch.cat(self.next_target, dim=1)
            else:
                self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.nowidx = 0
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

class PickleDataPrefetcher():
    def __init__(self, loader, batch_size, pickle_size, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stop_after = stop_after
        self.stream = torch.cuda.Stream()

        self.next_inputs = None
        self.next_targets = None
        
        self.next_input = None
        self.next_target = None

        self.complete = True

        self.batch_size = batch_size
        self.pickle_size = pickle_size

    def __len__(self):
        return len(self.loader) * int(self.loader.batch_size * self.pickle_size / self.batch_size)
    
    def __pack(self, inputs, targets):
        inputs = torch.stack(inputs, dim=0)
        if isinstance(targets[0], torch.Tensor):
            targets = torch.stack(targets, dim=0)
        else:
            targets = torch.tensor(targets)
        
        return inputs, targets

    def preload(self):
        if self.complete:
            self.next_inputs, self.next_targets = self.loaditer.iteration()
            if self.next_inputs is None:
                self.next_input = None
                self.next_target = None
                return
            
            self.complete = False

        self.next_input, self.next_target = self.__pack(self.next_inputs[0:self.batch_size], self.next_targets[0:self.batch_size])

        self.next_inputs = self.next_inputs[self.batch_size:]
        self.next_targets = self.next_targets[self.batch_size:]

        if len(self.next_inputs) == 0:
            self.complete = True

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = dataLoaderThread(iter(self.loader), 2)
        self.loaditer.start()
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

def mm_pack(inputs, targets):
    if isinstance(inputs[0], torch.Tensor):
        inputs = torch.stack(inputs, dim=0)
    else:
        inputs = list(zip(*inputs))
        inputs = list(map(lambda x:torch.stack(x, dim=0), inputs))
    
    return inputs, targets

class MultiModalPrefetcher():
    def __init__(self, loader, batch_size, pickle_size, stop_after=None, packfn=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stop_after = stop_after
        self.stream = torch.cuda.Stream()

        self.next_inputs = None
        self.next_targets = None
        
        self.next_input = None
        self.next_target = None

        self.complete = True

        self.batch_size = batch_size
        self.pickle_size = pickle_size
        
        if packfn is None:
            self.packfn = mm_pack
        else:
            self.packfn = packfn

    def __len__(self):
        return len(self.loader) * int(self.loader.batch_size * self.pickle_size / self.batch_size)

    def preload(self):
        if self.complete:
            # try:
            #     self.next_inputs, self.next_targets = next(self.loaditer)
            # except StopIteration:
            #     self.next_input = None
            #     self.next_target = None
            #     return
            self.next_inputs, self.next_targets = self.loaditer.iteration()
            if self.next_inputs is None:
                self.next_input = None
                self.next_target = None
                return
            
            self.complete = False

        self.next_input, self.next_target = self.packfn(self.next_inputs[0:self.batch_size], self.next_targets[0:self.batch_size])

        self.next_inputs = self.next_inputs[self.batch_size:]
        self.next_targets = self.next_targets[self.batch_size:]

        if len(self.next_inputs) == 0:
            self.complete = True

        with torch.cuda.stream(self.stream):
            if isinstance(self.next_input, list):
                for i in range(len(self.next_input)):
                    self.next_input[i] = self.next_input[i].cuda(non_blocking=True)
            else:
                self.next_input = self.next_input.cuda(non_blocking=True)
            
            if isinstance(self.next_target, torch.Tensor):
                self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = dataLoaderThread(iter(self.loader), 1)
        self.loaditer.start()
        # self.loaditer = iter(self.loader)
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

class PickleThreadPrefetcher():
    def __init__(self, dataset, batch_size, pickle_size, world_size, stop_after=None):
        self.dataset = dataset
        self.stop_after = stop_after
        self.stream = torch.cuda.Stream()

        self.next_inputs = None
        self.next_targets = None
        
        self.next_input = None
        self.next_target = None

        self.complete = True
        self.index = None

        self.batch_size = batch_size
        self.pickle_size = pickle_size
        self.world_size = world_size

        self.inner_loop_size = int(self.pickle_size / self.batch_size)

    def __len__(self):
        return int(len(self.dataset) * self.inner_loop_size / self.world_size)
    
    def __pack(self, inputs, targets):
        inputs = torch.stack(inputs, dim=0)
        if isinstance(targets[0], torch.Tensor):
            targets = torch.stack(targets, dim=0)
        else:
            targets = torch.tensor(targets)
        
        return inputs, targets

    def preload(self):
        if self.complete:
            ret = self.dataset.get()
            
            if ret is None:
                self.next_input = None
                self.next_target = None
                return
            
            self.next_inputs, self.next_targets = ret

            self.complete = False
            self.nowind = 0

        self.next_input, self.next_target = self.__pack(self.next_inputs[0:self.batch_size], self.next_targets[0:self.batch_size])

        self.next_inputs = self.next_inputs[self.batch_size:]
        self.next_targets = self.next_targets[self.batch_size:]

        if len(self.next_inputs) == 0:
            self.complete = True
            self.nowind = 0

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
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

from ..utils import Registery

DATASET_REGISTERY = Registery("DATASET")

def build_dataloader(config, usemultigpu):
    print('==> Preparing data..')
    dataset_cfg = config.pop("dataset")
    dataset_type = dataset_cfg.pop("type")
    
    return DATASET_REGISTERY.load(dataset_type)(usemultigpu=usemultigpu, **dataset_cfg)