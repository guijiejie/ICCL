from .tinybert import BertModel
import torch.nn as nn
import torch

class VBertConfig(object):
    def __init__(self, **kwargs):

        self.vocab_size = 21128
        self.input_size = 512
        self.hidden_size = 512
        self.num_fuse_layers = 3
        self.output_dim = 512
        self.clip_dim = 512
        self.bert_output = 768
        self.bert_config_path = './save/bert-base-chinese-config-l3.json'
        self.bert_save_path = './save/bert-base-chinese-pytorch_model.bin'

        # number of negative samples in queue
        self.memory_size = 10240

        for k in kwargs:
            setattr(self, k, kwargs[k])


class BertEncoder(nn.Module):
    def __init__(self,
                 bert_config_path: str,
                 is_trainning: bool = False,
                 training_params: bool = False):
        super(BertEncoder, self).__init__()
        self.training_params = training_params
        self.bert = BertModel(bert_config_path, is_trainning)
        self.__freeze()

    def forward(self, query):
        batch_size, max_len_q = list(query.size())
        lens = query.argmax(dim=-1)
        query_mask = torch.arange(max_len_q).expand(batch_size, max_len_q).to(device=query.device) >= lens.unsqueeze(1)

        if self.training_params:
            q_encoded_layers, q_atts, pooled_output = self.bert(query,
                                                                token_type_ids=None,
                                                                attention_mask=1-query_mask.long())
        else:
            with torch.no_grad():
                q_encoded_layers, q_atts, pooled_output = self.bert(query,
                                                                    token_type_ids=None,
                                                                    attention_mask=1-query_mask.long())
        return q_encoded_layers[-1]
    
    def __freeze(self):
        if not self.training_params:
            for m in self.parameters():
                m.requires_grad_(False)
    
    def train(self, mode=True):
        super().train(mode)
        if mode == True and not self.training_params:
            self.__freeze()

class BERT(nn.Module):
    def __init__(self, training_bert=False, is_trainning=False, 
                bert_config_path = "/mnt/ceph/home/yuvalliu/bert/bert-base-chinese-config-l3.json", 
                bert_save_path = "/mnt/ceph/home/yuvalliu/bert/bert-chinese-layer11.bin",
                width = 768):
        super().__init__()
        self.width = width
        
        self.bert_encoder = BertEncoder(bert_config_path, is_trainning, training_params=training_bert)
        ckpt = torch.load(bert_save_path)

        self.bert_encoder.load_state_dict(ckpt, strict=True)

    def forward(self, text):
        x = self.bert_encoder(text)
        x = x[:, 0, :]

        return x

    def init_weights(self, ckpt=None):
        pass

    def encode_text(self, text):
        x = self.bert_encoder(text)
        x = x[:, 0, :]

        return x