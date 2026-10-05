"""GQA / MQA 手写实现。

对应面试题:
    - AI-Infra-一面 "手写含 GQA 的 Attention"
    - 阿里-控股-一面 "手写 MQA/GQA"
    - 京东-实习 "手写 GQA"
    - 快手-一面-3 "给定 hidden_size 与 RoPE 维度算 GQA head 数"(理论, 见文档)
    - 小米-校招 "手写 MQA"

核心思想:
    MHA:  n_kv_heads = n_heads            KV Cache 大小 ∝ n_heads
    GQA:  n_kv_heads = n_heads / group_size 一组 query head 共享 kv head
    MQA:  n_kv_heads = 1                  所有 query head 共享一个 kv head
    KV Cache 显存 = 2 * n_layers * n_kv_heads * d_head * seq_len * batch * dtype_bytes
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """GQA/MQA 统一实现: n_kv_heads=1 即 MQA, n_kv_heads=n_heads 即 MHA。"""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "query 头数必须是 kv 头数的整数倍"
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.scaling = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """x: (B, S, E) self-attention。"""
        bsz, seq, _ = x.shape

        q = self.q_proj(x).view(bsz, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.n_kv_heads, self.d_head).transpose(1, 2)

        # 关键一步: 把 kv 头"扩张"给同组的 query 头
        # (B, n_kv, S, D) -> (B, n_kv, group, S, D) -> (B, n_kv*group, S, D)
        if self.group_size > 1:
            k = k.unsqueeze(2).expand(bsz, self.n_kv_heads, self.group_size, seq, self.d_head)
            k = k.reshape(bsz, self.n_heads, seq, self.d_head)
            v = v.unsqueeze(2).expand(bsz, self.n_kv_heads, self.group_size, seq, self.d_head)
            v = v.reshape(bsz, self.n_heads, seq, self.d_head)
            # 等价简写: k = k.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if causal:
            mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), 1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.out_proj(out)


def kv_cache_size(
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    batch: int,
    seq_len: int,
    dtype_bytes: int = 2,
) -> int:
    """KV Cache 显存估算公式(字节)——多家公司考过:

        2 * n_layers * n_kv_heads * d_head * batch * seq_len * dtype_bytes
    其中 2 表示 K 和 V 两个张量。
    例(快手-实习-一面-1 Qwen 显存估算): Qwen-7B, 32 层, 32 头(MHA), d_head=128,
    fp16, batch=1, seq=2048 → 2*32*32*128*1*2048*2 = 1.07 GB
    """
    return 2 * n_layers * n_kv_heads * d_head * batch * seq_len * dtype_bytes


@torch.no_grad()
def gqa_matches_mha_when_full_sharing() -> dict[str, bool]:
    """验证性实验: group_size=1 时 GQA 输出与 MHA 完全一致(同一组权重)。"""
    torch.manual_seed(0)
    d_model, n_heads = 64, 8
    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads=n_heads)
    x = torch.randn(2, 16, d_model)

    q = gqa.q_proj(x).view(2, 16, n_heads, d_model // n_heads).transpose(1, 2)
    k = gqa.k_proj(x).view(2, 16, n_heads, d_model // n_heads).transpose(1, 2)
    v = gqa.v_proj(x).view(2, 16, n_heads, d_model // n_heads).transpose(1, 2)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    mine = gqa(x, causal=True)
    # 手写版多了 out_proj, 对拍前先过同一个 out_proj
    ref = ref.transpose(1, 2).contiguous().view(2, 16, d_model)
    ref = gqa.out_proj(ref)
    return {"max_abs_err": float((ref - mine).abs().max()), "allclose": bool(torch.allclose(ref, mine, atol=1e-5))}
