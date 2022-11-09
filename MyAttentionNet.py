import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from ContrastiveLoss import MyContrastiveLoss


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.norm(x)
        return x


class MyAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super(MyAttention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_Q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_K = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_V = nn.Linear(dim, dim_head * heads, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        residual = x
        b, n_x, d_x, h_x = *x.shape, self.heads
        b, n_y, d_y, h_y = *y.shape, self.heads

        q = self.to_Q(x).view(b, -1, self.heads, d_y // self.heads).transpose(1, 2)
        k = self.to_K(y).view(b, -1, self.heads, d_x // self.heads).transpose(1, 2)
        v = self.to_V(y).view(b, -1, self.heads, d_x // self.heads).transpose(1, 2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out + residual)
        out = torch.flatten(out, 1)
        return out


if __name__ == '__main__':
    x = torch.rand(1, 90, 90)  # [batch, tokens, dim]

    # get attention network
    model = MyAttention(90, 1, 90)
    mask = torch.zeros(1, 90, 90)  # tokens X tokens
    mask[:, 5:20, 5:20] = 1
    attention = model(x, mask)

    # get embedding using MLP
    MLP_layer = MLP(8100,256,10,0.2)
    out = MLP_layer(attention)

    # for a simple test
    loss = MyContrastiveLoss(out, out)
    print(loss)
