# TODO: XAIGuided Transformer (Eq. 13)
import torch
from captum.attr import DeepLift
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / (x.size(-1) ** 0.5))
        return self.weight * x / (norm + self.eps)

class XAIGuidedMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, drofe_fn=None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)

        self.drofe = drofe_fn  # doit être une instance de RotaryFrequencyDemographicEncoding

    def forward(self, x, Qexpl, Kexpl, fl, fu, age, gender):
        B, F_tokens, D = x.shape

        Q = self.W_q(Qexpl)
        K = self.W_k(Kexpl)

        # ✅ Appliquer dRoFE après W_q et W_k
        if self.drofe is not None:
            Q = self.drofe(Q, fl, fu, age, gender)
            K = self.drofe(K, fl, fu, age, gender)

        V = self.W_v(x)

        Q = Q.view(B, F_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, F_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, F_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, F_tokens, D)
        output = self.W_o(attn_output)
        return output


class XAIGuidedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, drofe_fn=None, dropout=0.1):
        super().__init__()
        self.drofe_fn = drofe_fn
        self.attn = XAIGuidedMultiheadAttention(dim, num_heads)
        # self.norm1 = nn.RMSNorm(dim)
        # self.norm2 = nn.RMSNorm(dim)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)


        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )


    def forward(self, x, Qexpl, Kexpl, fl, fu, age, gender):
        # Injecter les encodages démographiques si la fonction est fournie
        if self.drofe_fn is not None:
            Qexpl = self.drofe_fn(Qexpl, age, gender)
            Kexpl = self.drofe_fn(Kexpl, age, gender)

        attn_out = self.attn(x, Qexpl, Kexpl, fl, fu, age, gender)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x
    
    
class XAIGuidedTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1, drofe_fn=None):
        super().__init__()
        self.layers = nn.ModuleList([
            XAIGuidedTransformerBlock(dim, num_heads, drofe_fn=drofe_fn, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, Qexpl, Kexpl, fl, fu, age, gender):
        for layer in self.layers:
            x = layer(x, Qexpl, Kexpl, fl, fu, age, gender)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, d_model=128, num_classes=9):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling sur les 9 tokens
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x: [B, 9, D]
        """
        x = x.transpose(1, 2)                # [B, D, 9] pour pooling sur time/token
        x = self.pool(x).squeeze(-1)         # [B, D]
        return self.classifier(x)            # [B, num_classes]
