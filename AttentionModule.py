"""
SelfAttention
+------------------------------------------+
|                                          |
|     X    *     W_Q      =    Q           |
|     X    *     W_K      =    K           |
|     X    *     W_V      =    V           |
|                                          |
|  +-----+    +-----+       +-----+        |
|  |  dim     |  dim        |  dim         |
|  |seq       |dim          |seq           |
|  +          +             +              |
|                                          |
|     Q    *     K.t      =    score       |
|                                          |
|  +-----+    +-----+       +-----+        |
|  |  dim     |  seq        |  seq         |
|  |seq       |dim          |seq           |
|  +          +             +              |
|                                          |
|               score w/ mask (in Decoder) |
|                                          |
|  score = softmax(score/sqrt(d_k),axis=1) |
|                                          |
|   score  *      V       =    Z           |
|                                          |
|  +-----+    +-----+       +-----+        |
|  |  seq     |  dim        |  dim         |
|  |seq       |seq          |seq           |
|  +          +             +              |
|                                          |
+------------------------------------------+


Encoder Block
+----------------------------------------------+
|                                              |
| X_embed + Pos_encoding                       |
|         |                                    |
|         v                +-----------------+ |
|        X_embed +-------->|  SelfAttention  | |
|            +             +---+-------------+ |
|            |                 |               |
|            +---------+       |               |
|                      |       |               |
|                      |       |               |
|                  +---v-------v---+           |
|                  |  X_embed, Z1  |           |
|       LayerNorm( |  ResidualADD  | )         |
|            +     +---------------+           |
|            |                                 |
|            |                                 |
|           Z2---------+--------->             |
|                      | Linear( Z2 )          |
|                      |       |               |
|                  +---v-------v---+           |
|                  |   Z2,    Z3   |           |
| out = LayerNorm( |  ResidualADD  | )         |
|                  +---------------+           |
|                                              |
+----------------------------------------------+

X = Embed(X_in) + PosEnc
Z_sa = SelfAttention(Q, K, V)(X)
Z = LayerNorm(X + Z_sa)
Z_lin = LinearReLULinear(Z)
Z_out = LayerNorm(Z + Z_lin)

"""
import numpy as np
import torch
import torch.nn as nn
from tr_utils import clones


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    out = torch.matmul(p_attn, value)
    return out, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, d_model, dropout=0.1):
        """
        head_num：
        d_model：
        """
        super().__init__()
        self.head_num = head_num
        self.d_model = d_model
        assert d_model % head_num == 0
        # 确定模型的总维度大小，然后平均分配给各个head
        self.d_k = d_model // head_num
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 注意力，用于后续可视化
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        输入:
            QKV.shape=(batch_size,seq_len,f_dim)
            mask.shape=(batch_size,1,seq_len)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        bs = query.shape[0]
        # 1. 线性变换获取Q,K,V
        Q, K, V = [
            l(x).view(bs, -1, self.head_num, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears[:3], (query, key, value))
        ]

        # 2. 进行注意力计算
        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        # 3. 将所有head拼接起来，作最后线性变换
        x = x.transpose(1, 2).contiguous().view(bs, -1,
                                                self.head_num * self.d_k)
        out = self.linears[-1](x)
        return out