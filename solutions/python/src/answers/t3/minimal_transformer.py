"""手写最小 Decoder-only Transformer（沐曦编程题「手写 Transformer 结构」）。

覆盖面试要求答出的全部组件:
  token embedding + 可学习位置编码 → N × [Pre-RMSNorm → 因果 MHA → 残差,
  Pre-RMSNorm → FFN(GELU) → 残差] → 最终 RMSNorm → LM head(共享词表)。
因果掩码用上三角 mask=-inf 实现 decode-only; GQA/权重吸收等变体见
answers/attention/（MHA/GQA/MLA 手写工程）。
测试: tests/test_t3.py 训练 200 步拟合一一映射任务, loss 显著下降。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # W_q/W_k/W_v 一次投影
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B,S,d) → (B,H,S,dh)：把头维拆出来
        q = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, float("-inf"))   # 因果掩码: 看不见未来
        y = F.softmax(att, dim=-1) @ v
        return self.proj(y.transpose(1, 2).reshape(b, s, d))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class Block(nn.Module):
    """Pre-Norm 残差块（现代 LLM 的标准排布: norm 在子层之前）。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1, self.attn = RMSNorm(d_model), CausalSelfAttention(d_model, n_heads)
        self.ln2, self.ffn = RMSNorm(d_model), FFN(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class MinimalTransformer(nn.Module):
    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 64) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)   # 可学习绝对位置编码
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, s = idx.shape
        pos = torch.arange(s, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))
