import torch
import os

class Saver():
    def __init__(self, cfg, logger, prefix = "epoch"):
        self.active = logger.local_rank == 0
        self.saver_cfg = cfg.pop("saver")
        if self.active:
            self.interval = self.saver_cfg["interval"]
            self.prefix = prefix
            self.workdir = logger.workdir
            self.total_epochs = cfg["total_epochs"]
            self.steps = 0
            self.maxsize = self.saver_cfg.get("maxsize", 3)
            self.name_list = []
            
            if os.path.exists(self.workdir):
                for fn in os.listdir(self.workdir):
                    if ".pth" in fn and self.prefix in fn and "end" not in fn:
                        self.name_list.append(os.path.join(self.workdir, fn))
            
                self.name_list.sort(key=(lambda x:int(os.path.basename(x).split("_")[-1][:-len(".pth")])))

                while len(self.name_list) > self.maxsize:
                    os.remove(self.name_list[0])
                    self.name_list.remove(self.name_list[0])

        if self.saver_cfg.get("saveType", "epoch") == "epoch":
            self.save_func = self.__save_epoch
        else:
            self.save_func = self.__save_iter

    def __save_epoch(self, epoch, model, optimizer):
        if not self.active:
            return
        epoch = epoch + 1
        if epoch % self.interval == 0 or epoch == self.total_epochs:
            save_name = f"{self.workdir}/{self.prefix}_{epoch}.pth"
            end_name = f"{self.workdir}/{self.prefix}_end.pth"
            print(f'Saving to {save_name}')
            state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, save_name)
            self.name_list.append(save_name)

            if os.path.islink(end_name):
                os.remove(end_name)

            os.symlink(os.path.basename(save_name), end_name)

            if len(self.name_list) > self.maxsize:
                os.remove(self.name_list[0])
                self.name_list.remove(self.name_list[0])

    def __save_iter(self, epoch, model, optimizer):
        if not self.active:
            return
        epoch = epoch + 1
        self.steps = self.steps + 1
        
        if self.steps % self.interval == 0:
            save_name = f"{self.workdir}/{self.prefix}_{self.steps}.pth"
            end_name = f"{self.workdir}/{self.prefix}_end.pth"
            print(f'Saving to {save_name}')
            state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, save_name)
            self.name_list.append(save_name)

            if os.path.islink(end_name):
                os.remove(end_name)

            os.symlink(os.path.basename(save_name), end_name)

            if len(self.name_list) > self.maxsize:
                os.remove(self.name_list[0])
                self.name_list.remove(self.name_list[0])

    def save(self, epoch, model, optimizer):
        if self.save_func == self.__save_iter:
            self.save_func(epoch, model, optimizer)
    
    def saveepoch(self, epoch, model, optimizer):
        if self.save_func == self.__save_epoch:
            self.save_func(epoch, model, optimizer)

    def saveend(self, epoch, model, optimizer):
        if not self.active:
            return
        epoch = epoch + 1
        
        end_name = f"{self.workdir}/{self.prefix}_end.pth"

        if os.path.islink(end_name):
            os.remove(end_name)

        print(f'Saving to {end_name}')
        state = {
            'state_dict': model.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, end_name)