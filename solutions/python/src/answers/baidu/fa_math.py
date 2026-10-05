"""FlashAttention 数学推导的逐步轨迹实现 —— 百度实习-一面-1 Q1 的"推导可演示版"。

对应面试题:
    - 百度-AI-Infra-实习-一面-1 Q1 "FlashAttention 的核心原理及其数学推导过程"
    - 百度-AI-Infra-实习-一面-3 Q11 "Bc 块的切分策略与 1-loop 实现"

与 answers/attention/flash_attention.py 的区别: 那边是 v1/v2 对照的工程实现,
这边记录 FA2 单个 Q 块扫 KV 块的**逐步中间量** (m, l, alpha, block 输出),
面试时可直接口述这份轨迹; 另附 causal 分块数公式与 SRAM 容量约束的数值检查。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def fa2_forward_with_trace(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 32,
    causal: bool = True,
) -> tuple[torch.Tensor, list[dict[str, list[float]]]]:
    """FA2(外层 Q、内层 KV, 即 1-loop)带中间量轨迹的前向。

    q/k/v: (S, D)。返回 (output, trace), trace 是第一个 Q 块扫各 KV 块时
    维护的 (m, l, alpha, 未归一化 acc 的范数) —— online rescaling 的现场证据。
    """
    seq, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)
    trace: list[dict[str, list[float]]] = []

    for qs in range(0, seq, block_size):
        qe = min(qs + block_size, seq)
        qb = q[qs:qe]

        m_i = torch.full((qe - qs,), float("-inf"))
        l_i = torch.zeros(qe - qs)
        acc = torch.zeros_like(qb)

        kv_end = qe if causal else seq
        for ks in range(0, kv_end, block_size):
            ke = min(ks + block_size, seq)
            logits = (qb @ k[ks:ke].T) * scale
            if causal and ke > qs:
                qi = torch.arange(qs, qe).unsqueeze(-1)
                kj = torch.arange(ks, ke).unsqueeze(0)
                logits = logits.masked_fill(kj > qi, float("-inf"))

            m_blk = logits.max(dim=-1).values
            m_new = torch.maximum(m_i, m_blk)
            alpha = torch.exp(m_i - m_new)                    # 旧结果的重缩放因子
            p = torch.exp(logits - m_new.unsqueeze(-1))       # 分块 exp, 不上溢
            l_i = l_i * alpha + p.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + p @ v[ks:ke]
            m_i = m_new

            if qs == 0:                                        # 记录第一个 Q 块的轨迹
                trace.append({
                    "kv_block": [ks, ke],
                    "m": m_i.tolist(),                        # 运行最大值(单调不减)
                    "l": l_i.tolist(),                        # 运行指数和(单调不减)
                    "alpha": alpha.tolist(),                  # <=1, 旧贡献的折扣
                    "acc_norm": float(acc.norm()),
                })

        out[qs:qe] = acc / l_i.unsqueeze(-1)                  # 唯一一次除以 l
    return out, trace


def flash_attention_correctness(
    batch: int = 2, heads: int = 4, seq: int = 128, dim: int = 32, block_size: int = 32,
) -> dict[str, object]:
    """与 F.scaled_dot_product_attention 对拍(causal + 非 causal), 并检查轨迹单调性。"""
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    outs, max_err = [], 0.0
    for b in range(batch):
        for h in range(heads):
            o, trace = fa2_forward_with_trace(q[b, h], k[b, h], v[b, h], block_size, causal=True)
            outs.append(o)
            max_err = max(max_err, float((o - ref[b, h]).abs().max()))
            for t0, t1 in zip(trace, trace[1:]):              # 正确的不变量:
                raw0 = [li * math.exp(mi) for li, mi in zip(t0["l"], t0["m"])]
                raw1 = [li * math.exp(mi) for li, mi in zip(t1["l"], t1["m"])]
                assert all(x <= y + 1e-3 for x, y in zip(raw0, raw1))  # 未归一化分母单调增
    return {"max_err_vs_sdpa": max_err, "passed": max_err < 1e-4}


def causal_block_count(seq: int, br: int, bc: int) -> tuple[int, int, int]:
    """分块 causal attention 的块计数(面试白板推导):

    - Q 块数 ceil(seq/br), KV 块数 ceil(seq/bc);
    - 无 causal: 全配对 = nq * nk;
    - causal:   第 i 个 Q 块只扫到第 ceil((i+1)*br/bc) 个 KV 块, 且最后一个
                交叉块只需局部三角 mask —— 完整计算块 + 斜跨对角块的计数。
    """
    nq, nk = -(-seq // br), -(-seq // bc)
    full = nq * nk
    causal_pairs = sum(-(-min((i + 1) * br, seq) // bc) for i in range(nq))
    return full, causal_pairs, full - causal_pairs            # 省掉的块数约一半


def sram_feasibility(br: int, bc: int, d: int, bytes_q: int = 2, bytes_kv: int = 2,
                     sram_bytes: int = 192 * 1024) -> bool:
    """FA 论文的 SRAM 容量约束: br*d*bq + 2*bc*d*bkv <= SRAM(3090 每 SM 192KB 上限档)。

    顺带回答"Bc 切分策略": bc 大 -> GEMM 效率高、重缩放次数(∝ seq/bc)少,
    但挤占 SRAM 降 occupancy; fp16 常取 br=bc=64~128 且 br >= bc。
    """
    need = br * d * bytes_q + 2 * bc * d * bytes_kv
    return need <= sram_bytes
