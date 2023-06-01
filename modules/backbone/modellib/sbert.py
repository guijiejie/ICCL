from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch
from torch import Tensor

def to_device(data):
    for key in data:
        if isinstance(data[key], Tensor):
            data[key] = data[key].cuda()
    return data

class SBERT(nn.Module):
    def __init__(self, train_mode=False, use_cuda=True, basepath="/mnt/ceph/home/yuyuanzeng/clip-full/preprocess/query_analysis/distiluse-base-multilingual-cased-v1"):
        super().__init__()
        self.model = SentenceTransformer(basepath)
        self.width = 512

        self.use_cuda = use_cuda
        self.train_mode = train_mode

        self.__freeze()
    
    def __freeze(self):
        if not self.train_mode:
            for m in self.parameters():
                m.requires_grad_(False)
            self.model.eval()
            self.training = False
    
    def train(self, mode=True):
        super().train(mode)
        self.__freeze()

    # text是原始句子
    def forward(self, text):
        token = self.model.tokenize(text)
        if self.use_cuda:
            token = to_device(token)

        features = self.model.forward(token)['sentence_embedding']
        return features

    def init_weights(self, ckpt=None):
        pass

    def encode_text(self, text):
        token = self.model.tokenize(text)
        if self.use_cuda:
            token = to_device(token)
        features = self.model.forward(token)['sentence_embedding']
        return features