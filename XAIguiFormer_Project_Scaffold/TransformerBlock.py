import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / math.sqrt(x.shape[-1]))
        return self.scale * x / (norm + self.eps)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # x: [B, F, D]
        attn_output, _ = self.attn(x, x, x)  # Vanilla MHA
        x = self.norm1(x + attn_output)     # Residual + RMSNorm

        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)       # Residual + RMSNorm

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
