import os
import tqdm
import torch.nn as nn
import copy
import torch
import math
import torch.nn.functional as F
import numpy as np
from datasets.bert_processors.abstract_processor import BertProcessor, InputExample
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
     "Implements FFN equation."

     def __init__(self, d_model, d_ff, dropout=0.1):
         super(PositionwiseFeedForward, self).__init__()
         self.w_1 = nn.Linear(d_model, d_ff)
         self.w_2 = nn.Linear(d_ff, d_model)
         self.dropout = nn.Dropout(dropout)

     def forward(self, x):
         return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query,key,value:torch.Size([30, 8, 10, 64])
    # decoder mask:torch.Size([30, 1, 9, 9])
    d_k = query.size(-1)
    key_ = key.transpose(-2, -1)  # torch.Size([30, 8, 64, 10])
    # torch.Size([30, 8, 10, 10])
    scores = torch.matmul(query, key_) / math.sqrt(d_k)
    if mask is not None:
        # decoder scores:torch.Size([30, 8, 9, 9]),
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        #Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 48=768//16
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query,key,value:torch.Size([2, 10, 768])
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)    #2
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]  # query,key,value:torch.Size([30, 8, 10, 64])
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                          dropout=self.dropout)
         # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
                  nbatches, -1, self.h * self.d_k)
        ret = self.linears[-1](x)  # torch.Size([2, 10, 768])
        return ret

#layer normalization [(cite)](https://arxiv.org/abs/1607.06450). do on
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl

# That is, the output of each sub-layer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented by the sub-layer itself.  We apply dropout [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the output of each sub-layer, before it is added to the sub-layer input and normalized.
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}}=512$.
class SublayerConnection(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        G = F.sigmoid(self.g_W1(x) + self.g_W2(self.dropout(sublayer(self.norm(x)))) + self.g_b)
        # (batch, m, word_dim)
        ret = G * x + (1 - G) * self.dropout(sublayer(self.norm(x)))
        return ret

# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn      #多头注意力机制
        self.feed_forward = feed_forward    #前向神经网络
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # torch.Size([30, 10, 512])
        ret = self.sublayer[1](x, self.feed_forward)
        return ret

class AAPDProcessor(BertProcessor):
    NAME = 'AAPD'
    NUM_CLASSES = 7
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir,'mag', 'exPFD_train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'mag', 'exPFD_dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'mag', 'exPFD_test.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            text = ['None']*4
            guid = "%s-%s" % (set_type, i)
            number = 0
            for num,(key,value) in enumerate(eval(line[1]).items()):
                if number < 4:
                # if number < 10 and key != 'title' and key != 'Reference':
                    text[number] = value
                    number += 1
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples
