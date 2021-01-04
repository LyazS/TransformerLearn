import numpy as np
import torch
import torch.nn as nn
from tr_utils import clones, LayerNorm, ResidualConnection


class Encoder(nn.Module):
    """
    编码器是由多层(注意力和ff层)组成

    官方原论文里的计算流程：
    x -> attention(x) -> x+self-attention(x) -> layernorm(x+self-attention(x)) => y
    y -> dense(y) -> y+dense(y) -> layernorm(y+dense(y)) => z(输入下一层)

    这里代码实现的流程：
    x -> layernorm(x) -> attention(layernorm(x)) -> x + attention(layernorm(x)) => y
    y -> layernorm(y) -> dense(layernorm(y)) -> y+dense(layernorm(y))
    """
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.layernorm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        out = self.layernorm(x)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, ff, dropout, N=2):
        """
        size：
        attn：注意力层，可以是多头注意力
        ff：线性层
        dropout：～
        N：2层，分别为注意力层和ff层
        """
        super().__init__()
        self.attn = attn
        self.size = size
        self.ff = ff
        self.sublayer = clones(ResidualConnection(size, dropout), N)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        out = self.sublayer[1](x, self.ff)
        return out
