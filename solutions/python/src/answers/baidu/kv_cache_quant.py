"""KV Cache INT8 量化 —— 百度实习-一面-3 Q8 的可运行对拍。

对应面试题:
    - 百度-AI-Infra-实习-一面-3 Q8 "KV Cache 量化的原理与实现方式"

实现粒度: per-token per-head(每个 token 每个头的 K 向量一个 absmax scale)——
    细粒度换取精度; 写 cache 前量化, attention 读取时反量化(真实引擎里
    反量化发生在 FA kernel 内部, 这里用"读出后统一 dequant"演示语义)。
同时给出 int4 档(对称 7 bit)对比, 展示精度-显存权衡。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def quantize_kv_per_token_head(
    kv: torch.Tensor, bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """kv: (B, H_kv, S, D) -> (int_quant, scale), scale 形状 (B, H_kv, S, 1)。

    对称量化: q = round(x / s), s = absmax / qmax, qmax = 2^(bits-1) - 1。
    """
    qmax = 2 ** (bits - 1) - 1
    scale = kv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(kv / scale), -qmax - 1, qmax)
    return q, scale


def attention_with_quant_kv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bits: int = 8,
) -> torch.Tensor:
    """q: (B, H, S, D), k/v: (B, H_kv, S, D), GQA 广播由 sdpa 原生处理。"""
    kq, ks = quantize_kv_per_token_head(k, bits)
    vq, vs = quantize_kv_per_token_head(v, bits)
    return F.scaled_dot_product_attention(q, kq * ks, vq * vs)


def kv_quant_metrics(
    batch: int = 2, heads: int = 8, seq: int = 256, dim: int = 64, seed: int = 0,
) -> dict[str, object]:
    """全精度 vs int8/int4 KV 的输出误差与 top-1 logit 一致率(最后一行取 argmax 近似)。"""
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    ref = F.scaled_dot_product_attention(q, k, v)
    res: dict[str, object] = {"ref_abs_mean": float(ref.abs().mean())}
    for bits in (8, 4):
        out = attention_with_quant_kv(q, k, v, bits)
        err = (out - ref).abs()
        # 用输出对最后一个 token 的聚合当作简易"下游 logits"观察
        last_ref = ref[:, :, -1, :].argmax(dim=-1)
        last_out = out[:, :, -1, :].argmax(dim=-1)
        res[f"int{bits}_max_err"] = float(err.max())
        res[f"int{bits}_rel_err"] = float(err.mean() / ref.abs().mean())
        res[f"int{bits}_last_token_top1_agree"] = float((last_ref == last_out).float().mean())
    return res
