import torch
import torch.utils.data as data

def collate_default(data):
    inputs, targets = zip(*data)
    inputs = torch.stack(inputs)
    targets = torch.tensor(targets)
    
    return inputs, targets

class OurDataLoader(object):
    def __init__(self, 
                dataset : data.Dataset,
                batch_size : int,
                shuffle : bool  = True,
                num_workers : int = 8,
                pin_memory : bool = True,
                collate_fn = None,
                sampler = None) -> None:
        super().__init__()
        assert shuffle == True
        assert sampler == None

        self.dataset = dataset
        self._dataloader = torch.utils.data.DataLoader(
                                dataset, batch_size=batch_size, shuffle=shuffle, 
                                num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, sampler=sampler)
        self.epoch = 0

    def __preload(self):
        if self.epoch == 0:
            self.loaditer1 = iter(self._dataloader)
        else:
            self.loaditer1 = self.loaditer2
        
        self.loaditer2 = iter(self._dataloader)

        self.epoch += 1
        return self.loaditer1

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        loader = self.__preload()
        for _ in range(len(self)):
            ans = next(loader)
            yield ans

class A(data.Dataset):
    def __init__(self, size=10) -> None:
        super().__init__()
        self.list = list(range(size))
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        return self.list[index]