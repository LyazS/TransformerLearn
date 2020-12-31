import numpy as np
import torch
import torch.nn as nn
from .tr_utils import clones, LayerNorm,ResidualConnection


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        encoder：编码器
        decoder：解码器
        src_embed：源embed层
        tgt_embed：目标embed层
        generator：decoder最后生成层
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        """
        编码输入
        """
        x_embed = self.src_embed(src)
        out_enc = self.encoder(x_embed, src_mask)
        return out_enc

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        解码输入
        memory：编码器的编码输出
        src_mask：源输入mask
        tgt
        tgt_mask
        """
        x_embed = self.tgt_embed(tgt)
        out_enc = self.encoder(x_embed, memory, src_mask, tgt_mask)
        return out_enc

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src：源输入
        tgt：目标输入
        src_mask：源输入mask
        tgt_mask：目标输入mask
        """
        out_enc = self.encode(src, src_mask)
        out_dec = self.decode(tgt, tgt_mask)
        return out_dec


class Encoder(nn.Module):
    """
    编码器是由多层注意力和fc层组成

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
    def __init__(self,size,attn,fc,dropout):
        """
        size：
        attn：注意力层，可以是多头注意力
        fc：线性层
        dropout：～
        """