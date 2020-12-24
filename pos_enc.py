import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
输入x.shape=(batch_size,seq_len,embed_dim)
因此对应的位置编码为pos_enc.shape=(seq_len,embed_dim)
"""


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