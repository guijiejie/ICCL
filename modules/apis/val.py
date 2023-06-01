import torch
import torch.nn as nn
import numpy as np
import time
import mmcv
from ..utils import KNN
try:
    from sklearn.neighbors import KNeighborsClassifier
except:
    print('doesnt has sklearn')
import copy
from einops import rearrange

def register_val(typename, model, device, data_loader, datasize, logger=None, config=None):
    return __NAME2CLS.get(typename, evaluate_rank)(model, device, data_loader, datasize, logger, config)


class evaluate_cls(object):
    def __init__(self, model, device, data_loader, datasize, logger=None, config=None):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.datasize = datasize
        self.logger = logger

    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        correct = None
        total = 0
        prog_bar = mmcv.ProgressBar(len(self.data_loader))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                # inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets != -1)
                inputs = inputs[mask]
                targets = targets[mask]

                outputs = self.model(inputs, targets)

                total += targets.size(0)
                if correct is None:
                    correct = {}
                    for k in outputs['accuracy']:
                        correct[k] = outputs['accuracy'][k].item()
                else:
                    for k in outputs['accuracy']:
                        correct[k] += outputs['accuracy'][k].item()

                prog_bar.update()

        for k in correct:
            correct[k] = round(correct[k] / float(total) * 100.0, 4)
        
        self.logger.record_eval(epoch, correct)

    def finaleval(self):
        self.model.eval()

        correct = None
        total = 0
        prog_bar = mmcv.ProgressBar(len(self.data_loader))

        multicls_ret = None

        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                
                out = model.backbone(inputs)
                if model.neck is not None:
                    out = model.neck(out)
                out = model.head.multicls_ret(out, targets)

                if multicls_ret is None:
                    multicls_ret = out
                else:
                    multicls_ret += out

                prog_bar.update()

        for i in range(multicls_ret.size(1)):
            self.logger.print(f"CLS {i}:\ttotal {int(multicls_ret[0][i].item())}\tcorrect {int(multicls_ret[1][i].item())}\tratio {round(multicls_ret[1][i].item()/multicls_ret[0][i].item(), 4)}")

class evaluate_videocls(object):
    def __init__(self, model, device, data_loader, datasize, logger=None, config=None):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.datasize = datasize
        self.logger = logger

    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        correct = None
        total = 0
        prog_bar = mmcv.ProgressBar(len(self.data_loader))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                # inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs, targets, aggregate=True)

                total += targets.size(0)
                if correct is None:
                    correct = {}
                    for k in outputs['accuracy']:
                        correct[k] = outputs['accuracy'][k].item()
                else:
                    for k in outputs['accuracy']:
                        correct[k] += outputs['accuracy'][k].item()

                prog_bar.update()

        for k in correct:
            correct[k] = round(correct[k] / float(total) * 100.0, 4)
        
        self.logger.record_eval(epoch, correct)
    
    def finaleval(self):
        self.model.eval()

        correct = None
        total = 0
        prog_bar = mmcv.ProgressBar(len(self.data_loader))

        multicls_ret = None

        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                if isinstance(inputs, list):
                    inputs, bboxfeats, bboxmasks = inputs
                    bboxfeats = rearrange(bboxfeats, "n i c -> i n c")
                else:
                    bboxfeats = None
                N, I, C, D, H, W = inputs.size()

                inputs = rearrange(inputs, "n i c d h w -> (n i c) d h w")
                CLIP_NUM = I * C

                out = model.backbone(inputs)
                
                if model.neck is not None:
                    out = model.neck(out)
                
                out = rearrange(out, "(n N) C -> n N C", N=CLIP_NUM).mean(dim=1)
                
                if bboxfeats is not None:
                    out = model.head.multicls_ret(out, targets, bboxfeats)
                else:
                    out = model.head.multicls_ret(out, targets)

                if multicls_ret is None:
                    multicls_ret = out
                else:
                    multicls_ret += out

                prog_bar.update()

        for i in range(multicls_ret.size(1)):
            self.logger.print(f"CLS {i}:\ttotal {int(multicls_ret[0][i].item())}\tcorrect {int(multicls_ret[1][i].item())}\tratio {round(multicls_ret[1][i].item()/multicls_ret[0][i].item(), 4)}")

class evaluate_knn(object):
    def __init__(self, model, device, data_loader, datasize, logger, config=None):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.datasize = datasize
        self.config = config
        self.logger = logger
        self.outputs_container = None
        self.targets_container = np.zeros([1, datasize], dtype=np.int64)
        
        self.knn_config = config.get('knn', {})
        if "topk" not in self.knn_config:
            if "topk_percent" not in self.knn_config:
                self.topk = int(datasize / config['num_class'] * 0.2)
            else:
                self.topk = int(datasize / config['num_class'] * self.knn_config.pop('topk_percent'))
            self.knn_config['topk'] = self.topk
        else:
            self.topk = self.knn_config['topk']
    
    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        prog_bar = mmcv.ProgressBar(len(self.data_loader))
        self.outputs_container = None
        self.targets_container

        sample_idx = 0
        labels = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                # inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                batchsize = targets.size(0)

                outputs = self.model(inputs, forward_knn=True)
                if len(outputs.size()) > 2:
                    outputs = outputs.mean(axis=[2,3])
                
                mask = (targets != -1)
                outputs = outputs[mask]
                targets = targets[mask]
                if self.outputs_container is None:
                    self.outputs_container = copy.deepcopy(outputs.cpu().numpy())
                else:
                    self.outputs_container = np.concatenate([self.outputs_container, copy.deepcopy(outputs.detach().cpu().clone().numpy())], axis=0)
                labels += targets.cpu().numpy().tolist()

                sample_idx += batchsize
                prog_bar.update()

        self.targets_container = np.array(labels)[np.newaxis,:]
        print('==> Calculating KNN..')
        total_acc = KNN(self.outputs_container, self.targets_container, **self.knn_config)
        correct = {
            f"KNN-{self.topk}":total_acc * 100.0
        }
        self.logger.record_eval(epoch, correct)

class evaluate_videoknn(object):
    def __init__(self, model, device, data_loader, datasize, logger, config=None):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.datasize = datasize
        self.config = config
        self.logger = logger
        self.outputs_container = None
        self.targets_container = np.zeros([1, datasize], dtype=np.int64)
        
        self.knn_config = config.get('knn', {})
        if "topk" not in self.knn_config:
            if "topk_percent" not in self.knn_config:
                self.topk = int(datasize / config['num_class'] * 0.2)
            else:
                self.topk = int(datasize / config['num_class'] * self.knn_config.pop('topk_percent'))
            self.knn_config['topk'] = self.topk
        else:
            self.topk = self.knn_config['topk']
    
    def __cal_inner_score(self, outs):
        score = 0.0
        l = torch.nn.functional.normalize(outs, dim=2)
        r = rearrange(l, "n i c -> n c i")
        score += torch.min(torch.bmm(l, r), dim=2)[0].mean(dim=1).sum()
        
        return score

    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        prog_bar = mmcv.ProgressBar(len(self.data_loader))

        inner_score = 0.0

        sample_idx = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                batchsize = targets.size(0)
                if len(targets.size()) > 1:
                    targets = targets[:,0]

                outputs = self.model(inputs, forward_knn=True)

                if len(outputs.size()) > 4:
                    outputs = outputs.mean(axis=[3,4])

                inner_score += self.__cal_inner_score(outputs)
                
                outputs = outputs.mean(dim=1)
                
                if self.outputs_container is None:
                    self.outputs_container = np.zeros([self.datasize, outputs.size(1)], dtype=np.float32)
                
                self.outputs_container[sample_idx:sample_idx+batchsize] = outputs.cpu().numpy()
                self.targets_container[0, sample_idx:sample_idx+batchsize] = targets.cpu().numpy()

                sample_idx += batchsize
                prog_bar.update()

        print('==> Calculating KNN..')
        total_acc = KNN(self.outputs_container, self.targets_container, **self.knn_config)
        correct = {
            f"KNN-{self.topk}":total_acc * 100.0,
            'inner-score':inner_score.item() / self.datasize * 100.0
        }
        self.logger.record_eval(epoch, correct)

class evaluate_rank(object):
    def __init__(self, model, device, data_loader, datasize, logger, config=None):
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.datasize = datasize
        self.logger = logger
    
    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        prog_bar = mmcv.ProgressBar(self.datasize)

        featurelist = []
        namelist = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs = inputs.to(self.device)
                batchsize = inputs.size(0)

                outputs = self.model(inputs, forward_knn=True)
                if len(outputs.size()) > 2:
                    outputs = outputs.mean(axis=[2,3])
                
                featurelist.append(copy.deepcopy(outputs.detach().cpu().data.numpy().copy()))
                namelist += targets

                for _ in range(batchsize):
                    prog_bar.update()

        print('==> Calculating..')
        featurelist = np.concatenate(featurelist, axis=0)
        featurelist = featurelist / np.linalg.norm(featurelist, axis=1, keepdims=True)
        outputs = self.data_loader.dataset.eval(featurelist, namelist)
        self.logger.record_eval(epoch, outputs)

class evaluate_nocls(object):
    def __init__(self, model, device, data_loader, datasize, config, logger):
        self.model = model
        self.device = device
        self.train_loader, self.test_loader = data_loader
        self.datasize = datasize
        self.config = config
        self.logger = logger
        self.outputs_container = np.zeros([datasize], dtype=np.int64)
        self.targets_container = np.zeros([datasize], dtype=np.int64)
        
        # self.nn_config = 20
    
    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        prog_bar = mmcv.ProgressBar(self.datasize)

        sample_idx = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                # inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                batchsize = targets.size(0)

                outputs = self.model.forward_knn(inputs)
                pred = torch.softmax(outputs, dim=-1)
                _, cls_label = torch.topk(pred, k=1, dim=1, largest=True)
                
                self.outputs_container[sample_idx:sample_idx+batchsize] = cls_label.cpu().squeeze(dim=1).numpy()
                self.targets_container[sample_idx:sample_idx+batchsize] = targets.cpu().numpy()

                sample_idx += batchsize
                for _ in range(batchsize):
                    prog_bar.update()

        print('==> Calculating..')
        num_cls = self.targets_container.max() + 1
        clsset = []
        redirect = dict()
        for c in range(num_cls):
            clsnum = self.targets_container[self.outputs_container == c]
            r = dict()
            for n in clsnum:
                if n not in r:
                    r[n] = 0
                r[n] += 1
            clsset.append(r)

            maxitem = sorted(r.items(), key=lambda x:x[1])[-1]
            redirect[c] = maxitem[0]

            # assert maxitem[1] >= self.nn_config / 2.
            # assert maxitem[0] not in redirect
            # for m in maxitem:
            #     if m not in redirect:
            #         redirect[m] = c
            #         break

        print(clsset)
        print(redirect)

        def mp(entry):
            return redirect[entry]
        mp = np.vectorize(mp)

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.cpu().numpy()
                
                batchsize = targets.shape[0]

                outputs = self.model.forward_knn(inputs)
                pred = torch.softmax(outputs, dim=-1)
                _, cls_label = torch.topk(pred, k=1, dim=1, largest=True)

                total += batchsize
                correct += (mp(cls_label.cpu().squeeze(dim=1).numpy()) == targets).sum()

        total_acc = correct / total
        correct = {
            f"top-1":total_acc * 100.0
        }
        self.logger.record_eval(epoch, correct)

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T=0.07, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5

class traditional_knn(object):
    def __init__(self, model, device, data_loader, datasize, config, logger):
        self.model = model
        self.device = device
        self.train_loader, self.test_loader = data_loader
        self.datasize = datasize
        self.config = config
        self.logger = logger

        self.num_cls = self.config.get('num_class', 10)
    
    def __call__(self, epoch=0, usemultigpu=False):
        self.model.eval()

        prog_bar = mmcv.ProgressBar(len(self.train_loader))

        classifier = KNeighborsClassifier(5, weights="distance")
        X = None
        Y = None
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                batchsize = targets.size(0)

                outputs = self.model.backbone(inputs)
                if self.model.neck is not None:
                    outputs = self.model.neck(outputs)

                if len(outputs.size()) > 2:
                    outputs = outputs.mean(axis=[2,3])
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                if X is None:
                    X = outputs.cpu().detach().clone().numpy()
                    Y = targets.cpu().detach().clone().numpy()
                else:
                    X = np.concatenate([X, outputs.cpu().detach().clone().numpy()], axis=0)
                    Y = np.concatenate([Y, targets.cpu().detach().clone().numpy()], axis=0)
                prog_bar.update()

        print('==> Calculating..{}'.format(X.shape))
        
        train_features = torch.tensor(X).cuda()
        train_labels = torch.tensor(Y).cuda()
        # classifier.fit(X, Y)
        
        M = None
        N = None
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                batchsize = targets.shape[0]
                outputs = self.model.backbone(inputs)
                if self.model.neck is not None:
                    outputs = self.model.neck(outputs)
                if len(outputs.size()) > 2:
                    outputs = outputs.mean(axis=[2,3])
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                
                if M is None:
                    M = outputs.cpu().detach().clone().numpy()
                    N = targets.cpu().detach().clone().numpy()
                else:
                    M = np.concatenate([M, outputs.cpu().detach().clone().numpy()], axis=0)
                    N = np.concatenate([N, targets.cpu().detach().clone().numpy()], axis=0)

                # predicts = classifier.predict(outputs.cpu().numpy())

                # total += batchsize
                # correct += (torch.tensor(predicts, dtype=targets.dtype) == targets.cpu()).sum()

        test_features = torch.tensor(M).cuda()
        test_labels = torch.tensor(N).cuda()

        correct = {}
        for nn in [5, 10, 20]:
            top1, top5 = knn_classifier(train_features, train_labels, test_features, test_labels, nn, num_classes=self.num_cls)
            
            correct[f"nn{nn}-top1"] = top1
            correct[f"nn{nn}-top5"] = top5

        # print(correct)
        self.logger.record_eval(epoch, correct)
        return correct

__NAME2CLS = {
    "CLS":evaluate_cls,
    "KNN":evaluate_knn,
    "RANK":evaluate_rank,
    "VideoCLS":evaluate_videocls,
    "VideoKNN":evaluate_videoknn,
    'EvalKNN':traditional_knn,
    # "QueryNN":evaluate_querynn
}
