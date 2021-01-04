import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
输入x.shape=(batch_size,seq_len,embed_dim)
因此对应的位置编码为pos_enc.shape=(seq_len,embed_dim)
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = dropout
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(np.log(1e4) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        out = x + self.pe[:, :x.shape[1]]
        return out


def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度

    pos = np.arange(0, max_seq_len).reshape((-1, 1))
    embed = np.arange(0, embed_dim).reshape((1, -1))
    embed = 1 / np.power(1e4, 2 * embed / embed_dim)
    positional_encoding = np.matmul(pos, embed)

    positional_encoding[1:, 0::2] = np.sin(
        positional_encoding[1:, 0::2])  # dim 2i   偶数
    positional_encoding[1:, 1::2] = np.cos(
        positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    return positional_encoding


if __name__ == "__main__":
    positional_encoding = get_positional_encoding(
        max_seq_len=100,
        embed_dim=16,
    )
    plt.figure(figsize=(10, 10))
    sns.heatmap(positional_encoding)
    plt.title("Sinusoidal Function")
    plt.xlabel("hidden dimension")
    plt.ylabel("sequence length")
    plt.show()