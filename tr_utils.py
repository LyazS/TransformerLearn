import torch
import torch.nn as nn
import copy
import numpy as np


def clones(module, N):
    moduls = [copy.deepcopy(module) for _ in range(N)]
    return nn.ModuleList(moduls)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones((features)))
        self.b_2 = nn.Parameter(torch.zeros((features)))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        std = x.std(axis=-1, keepdim=True)
        out = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return out


class ResidualConnection(nn.Module):
    """
    残差链接

    里边sublayer可以嵌入selfatt或者fc层
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        out_layer = sublayer(self.norm(x))
        out_res = x + self.dropout(out_layer)
        return out_res


def subsequent_mask(size):
    """
    第一行只有第一列是1，意思是时刻1只能attend to输入1，
    第三行说明时刻3可以attend to {1,2,3}而不能attend to{4,5}的输入，
    因为在Decode的时候这是属于Future的信息
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype(np.uint8)
    return 1 - torch.from_numpy(subsequent_mask)


class SeqBatch():
    """
    仅支持词向量int，暂未写成float的多维向量

    src：np.int,shape=(bs,src_len)
    tgt：np.int,shape=(bs,tgt_len)
    """
    def __init__(self, src, tgt=None, pad=0):
        """
        src,tgt:类向量，(bs,seqlen:[0,1,2,...,N]) 有N个词
        """
        self.src = src
        # 输入到encoder的源，不等0则mask为1
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # decoder的输入
            self.tgt = tgt[:, :-1]
            # decoder的输出
            self.tgt_y = tgt[:, 1:]
            # 输入的mask
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).cpu().numpy().sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        ss_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        # 未来信息的位置置0
        tgt_mask = tgt_mask & ss_mask
        return tgt_mask
