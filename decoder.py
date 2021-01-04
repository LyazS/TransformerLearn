import numpy as np
import torch
import torch.nn as nn
from tr_utils import clones, LayerNorm, ResidualConnection


class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        与encoder基本结构
        由N个DecoderLayer的stack组成
        """
        super().__init__()
        self.layers = clones(layer, N)
        self.layernorm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        out = self.layernorm(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, size, attn, src_attn, ff, dropout):

        super().__init__()
        self.size = size
        self.attn = attn
        self.src_attn = src_attn
        self.ff = ff
        self.sublayer = clones(ResidualConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        输入x
        Encoder层的输出memory
        输入Encoder的Mask(src_mask)
        输入Decoder的Mask(tgt_mask)
        
        attn输入：Q,K,V,mask

        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        out = self.sublayer[2](x, self.ff)
        return out
