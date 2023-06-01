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
    def __init__(self, logname="wechat_video_search_gpu_mixed_clip_mv_sberttrain_notextmv_2021_06_23_17", 
                train_mode=True, 
                use_cuda=True
                ):
        super().__init__()

        self.width = 512

        if logname is not None:
            ckpt = torch.load(f"/mnt/ceph/home/yuvalliu/result/{logname}/epoch_end.pth", map_location="cpu")['state_dict']
            state_dict = {}

            if 'module.backbone.model.text_projection' in ckpt:
                output_dim = ckpt['module.backbone.model.text_projection'].size(1)
                self.text_projection = nn.Parameter(torch.empty(self.width, output_dim))
                state_dict['text_projection'] = ckpt['module.backbone.model.text_projection']
            else:
                self.text_projection = None
        
        if logname is None or 'module.backbone.model.textencoder.model.2.linear.bias' in ckpt:
            self.model = SentenceTransformer("/mnt/ceph/home/yuyuanzeng/clip-full/preprocess/query_analysis/distiluse-base-multilingual-cased-v1")
        else:
            self.model = SentenceTransformer("/mnt/ceph/home/yuvalliu/pretrained/sbert.net_models_clip-ViT-B-32-multilingual-v1")

        if logname is not None:
            for k, v in ckpt.items():
                if 'module.backbone.model.textencoder' in k:
                    k = k[len("module.backbone.model.textencoder."):]
                    state_dict[k] = v

            self.load_state_dict(state_dict, strict=True)

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

    # text是原始句子, list
    def forward(self, text):
        token = self.model.tokenize(text)
        if self.use_cuda:
            token = to_device(token)

        features = self.model.forward(token)['sentence_embedding']

        if self.text_projection is not None:
            features = features @ self.text_projection
        return features

    def init_weights(self, ckpt=None):
        pass
    
    # text是原始句子, list
    def encode_text(self, text):
        token = self.model.tokenize(text)
        if self.use_cuda:
            token = to_device(token)
        features = self.model.forward(token)['sentence_embedding']

        if self.text_projection is not None:
            features = features @ self.text_projection
        return features
