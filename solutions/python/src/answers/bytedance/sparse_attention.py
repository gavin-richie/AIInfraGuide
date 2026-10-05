"""Sparse Attention 的 PyTorch 实现：动态 top-k 块选择 + 局部窗（Quest/Sparse Transformer 杂交版）。

对应题目：字节-AI-Infra-实习-一二三面 二面 11「Sparse Attention 的 Python kernel 如何实现？」

两条技术路线（面试先讲清楚再写代码）：

1. **静态模式**（Sparse Transformer / BigBird / Longformer）：local / strided / global /
   random 位置是先验固定的，mask 可以离线构建，kernel 无需先算分数。
2. **动态选择**（Quest / H2O / Token Pooling）：先对每个 KV 块打分，再为每个 query
   块保留 top-k 块——本模块实现这条路线，并叠加一个永远保留的局部窗（滑窗是几乎
   所有方案的兜底，防近邻信息丢失）。

块代表分数取「块内真实 token 的 pairwise score 均值」，数学上等于块内均值池化向量的
内积（实现里用求和向量做 masked mean，天然排除 padding token）。Quest 原文用的是
页内 K 的逐维 min/max 与 q 乘出的上下界，思想相同、打分更保守。

复杂度：dense 是 O(Sq·Sk·d)；块稀疏后每行只算 (k_top + 窗口) 个块 →
O(Sq·(k_top+w)·block·d)，长序列下显著省算力与 KV 读取（decode 时省的是 KV 访存）。

运行验证::

    cd solutions/python && python -m pytest tests/test_bytedance.py -q
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _pad_to_blocks(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int, torch.Tensor]:
    """(B,H,S,D) -> (按块求和 (B,H,nb,D), 块数 nb, 每块真实 token 数 (nb,))。"""
    s = x.shape[-2]
    nb = math.ceil(s / block_size)
    padded = F.pad(x, (0, 0, 0, nb * block_size - s))  # pad 值为 0，不参与求和
    blocked = padded.view(*x.shape[:2], nb, block_size, x.shape[-1])
    block_sum = blocked.sum(dim=3)
    g = torch.arange(nb, device=x.device)
    counts = (s - g * block_size).clamp(min=0, max=block_size)
    return block_sum, nb, counts


def block_scores(q: torch.Tensor, k: torch.Tensor, block_size: int) -> torch.Tensor:
    """每个 (query 块, key 块) 的代表分数 = 块内真实 token 的 q·k 分数均值。

    q: (B, H, Sq, D), k: (B, H, Sk, D) -> (B, H, n_qblocks, n_kblocks)，全部有限。
    """
    q_sum, nqb, nq = _pad_to_blocks(q, block_size)
    k_sum, nkb, nk = _pad_to_blocks(k, block_size)
    scores = torch.einsum("bhgd,bhkd->bhgk", q_sum, k_sum)
    denom = nq.view(1, 1, -1, 1) * nk.view(1, 1, 1, -1)  # (1,1,nqb,nkb)
    return scores / denom


def select_blocks(scores: torch.Tensor, top_k: int, n_local: int) -> torch.Tensor:
    """由块分数得到布尔块 mask：top-k 块 ∪ 局部窗（块索引 [i-n_local, i+n_local]）。

    scores: (B, H, nq, nk) -> bool (B, H, nq, nk)
    """
    b, h, nq, nk = scores.shape
    k_eff = min(top_k, nk)
    topk = torch.topk(scores, k_eff, dim=-1).indices  # (B,H,nq,k)
    topk_mask = torch.zeros(b, h, nq, nk, dtype=torch.bool, device=scores.device)
    topk_mask.scatter_(-1, topk, True)

    qi = torch.arange(nq, device=scores.device).view(1, 1, nq, 1)
    ki = torch.arange(nk, device=scores.device).view(1, 1, 1, nk)
    local_mask = (qi - ki).abs() <= n_local
    return topk_mask | local_mask


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 16,
    top_k: int = 2,
    n_local: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (output, token 级 mask)。output 与「dense attention + 同一 mask」数值一致。

    q/k/v: (B, H, S, D)，不带 causal（causal 与稀疏化正交，可叠加）。
    """
    bscores = block_scores(q, k, block_size)
    bmask = select_blocks(bscores, top_k, n_local)

    b, h, sq, d = q.shape
    sk = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    token_mask = bmask.repeat_interleave(block_size, dim=-2).repeat_interleave(
        block_size, dim=-1
    )[..., :sq, :sk]

    attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
    attn = attn.masked_fill(~token_mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    attn = torch.nan_to_num(attn)  # 全 -inf 行（空块行）输出 0
    out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
    return out, token_mask


def dense_attention_with_mask(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """参考实现：dense 计算后套同一个 mask，用于与稀疏版对拍。"""
    d = q.shape[-1]
    attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(d)
    attn = attn.masked_fill(~mask, float("-inf"))
    attn = torch.nan_to_num(F.softmax(attn, dim=-1))
    return torch.einsum("bhqk,bhkd->bhqd", attn, v)


def sparse_flops(
    seq: int, d: int, block_size: int, top_k: int, n_local: int
) -> tuple[int, int]:
    """(sparse 的 QK^T 乘加量, dense 对应量)，用于说明省了多少。

    只算 QK^T（PV 同量级）；每行保留 top-k ∪ 窗口共约 top_k + 2·n_local + 1 个块。
    """
    n_blocks = seq // block_size
    kept = min(top_k + 2 * n_local + 1, n_blocks)
    per_row = kept * block_size
    return seq * per_row * d, seq * seq * d
