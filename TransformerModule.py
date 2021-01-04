import copy
import numpy as np
import torch
import torch.nn as nn
from tr_utils import clones, LayerNorm, ResidualConnection
from pos_enc import PositionalEncoding
from AttentionModule import MultiHeadAttention
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_embed=None,
                 tgt_embed=None,
                 generator=None):
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
        self.src_embed = src_embed if src_embed is not None else lambda x: x
        self.tgt_embed = tgt_embed if tgt_embed is not None else lambda x: x
        self.generator = generator if generator is not None else lambda x: x

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
        out_enc = self.decoder(x_embed, memory, src_mask, tgt_mask)
        return out_enc

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src：源输入
        tgt：目标输入
        src_mask：源输入mask
        tgt_mask：目标输入mask
        """
        out_enc = self.encode(src, src_mask)
        out_dec = self.decode(out_enc, src_mask, tgt, tgt_mask)
        return out_dec


class FFSubLayer(nn.Module):
    """
    feedforward层
    即全链接层
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.ff1(x)
        out = self.dropout(self.relu(out))
        out = self.ff2(out)
        return out


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * np.sqrt(self.d_model)


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab,
                 tgt_vocab,
                 N=6,
                 d_model=512,
                 d_ff=2048,
                 heads=8,
                 dropout=0.1):
        super().__init__()
        deepCopy = copy.deepcopy
        attn = MultiHeadAttention(heads, d_model, dropout)
        ff = FFSubLayer(d_model, d_ff, dropout)
        positionEnc = PositionalEncoding(d_model, dropout)
        encoder = Encoder(
            EncoderLayer(d_model, deepCopy(attn), deepCopy(ff), dropout), N)
        decoder = Decoder(
            DecoderLayer(d_model, deepCopy(attn), deepCopy(attn), deepCopy(ff),
                         dropout), N)
        self.model = EncoderDecoder(
            encoder,
            decoder,
            nn.Sequential(Embeddings(d_model, src_vocab), positionEnc),
            nn.Sequential(Embeddings(d_model, tgt_vocab), positionEnc),
            # nn.Sequential(positionEnc),
            # nn.Sequential(positionEnc),
            None,
        )
        self.para_init()

    def para_init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)

