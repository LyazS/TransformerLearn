import torch
import torch.nn as nn
import copy


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
        self.dropout = dropout

    def forward(self, x, sublayer):
        out_layer = sublayer(self.norm(x))
        out_res = x + self.dropout(out_layer)
        return out_res
