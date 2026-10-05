"""Multi-Head Attention 手写实现（PyTorch）。

对应面试题:
    - 百度-一面-1 "Attention 的原理"、字节-豆包 "MHA"
    - 综合面经题库-2 PyTorch 编程题 1 "实现 Multi-Head Attention(3 道变体)"
    - 拼多多/联想 "手写 MHA"
    - 京东-实习 "手写 GQA"(见 gqa.py)

要点:
    1. 投影: qkv_proj / out_proj
    2. reshape 到 (B, S, H, D) 再 transpose 到 (B, H, S, D) —— 注意 view 只能用于连续内存
    3. softmax 前按 1/sqrt(d_k) 缩放(scaled dot-product), 数值稳定要减 max
    4. causal mask: 上三角 -inf
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """标准 MHA, 支持 causal mask 与 self/cross attention。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.scaling = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """query/key/value: (B, S_q, E); cross-attention 时 S_q != S_kv。

        返回 (B, S_q, E)。
        """
        bsz, sq, _ = query.shape
        skv = key.shape[1]

        # (B, S, E) -> (B, S, H, D) -> (B, H, S, D)
        q = self.q_proj(query).view(bsz, sq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(bsz, skv, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(bsz, skv, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product: (B, H, S_q, S_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if causal:
            # 上三角(不含对角线)置 -inf, softmax 后为 0
            causal_mask = torch.triu(
                torch.ones(sq, skv, dtype=torch.bool, device=query.device), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # softmax 内部做了 max 减法(数值稳定)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)  # (B, H, S_q, D)
        # 合并多头: transpose 回 (B, S, H, D) 后必须 contiguous 才能 view
        out = out.transpose(1, 2).contiguous().view(bsz, sq, self.d_model)
        return self.out_proj(out)


def sdpa_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """去投影的纯 attention 数学核心, 与 F.scaled_dot_product_attention 对拍用。

    q/k/v: (B, H, S, D)
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if causal:
        s = q.shape[-2]
        scores = scores.masked_fill(
            torch.triu(torch.ones(s, s, dtype=torch.bool, device=q.device), diagonal=1),
            float("-inf"),
        )
    return torch.matmul(F.softmax(scores, dim=-1), v)


def naive_softmax_overflow_demo() -> dict[str, float]:
    """面试追问"Softmax 为什么要减 max": 演示大 logits 直接 exp 会溢出为 inf/nan。

    来源: 三星-二面(Softmax 数值稳定)、太初(手写 softmax 并行化)、原粒(Reduce 计算顺序与数值精度)。
    """
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    naive = torch.exp(x) / torch.exp(x).sum()
    stable = F.softmax(x, dim=0)
    return {
        "naive_max_exp": float(torch.exp(x).max()),  # inf
        "naive_has_nan": bool(torch.isnan(naive).any()),
        "stable_sum": float(stable.sum()),  # 1.0
        "stable": stable.tolist(),
    }
