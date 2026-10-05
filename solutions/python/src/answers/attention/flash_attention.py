"""FlashAttention v1 / v2 手写实现（PyTorch 分块版）。

对应面试题:
    - 综合面经题库-2 PyTorch 编程题 "实现 Flash Attention v1"
    - 快手-校招-2 "Online Softmax 与 FA 伪代码"
    - 百度-实习-一面-1 "FA 原理与数学推导"
    - 字节-校招-一面 "CUDA GEMM 手写与评估指标" + 手写 MLA

v1 vs v2 的区别(v2 面试高频):
    1. 外层循环: v1 外层遍历 K/V 内层 Q(一个 block 的 KV 被多个 Q block 反复读写
       中间结果 Ooj/lij 要写回 HBM); v2 交换内外循环, 外层 Q 内层 KV,
       中间结果留在寄存器/SRAM, 不写回显存 → HBM 访存量降半。
    2. v2 减少非 matmul 计算: 非乘法 FLOPs 从 v1 的占比大幅下降(重缩放合并、
       分母倒数乘法化)。
    3. v2 并行度: v1 在 seq_len 维并行(KV 块数), 长序列小 batch 时 SM 不满;
       v2 增加 Q 块间并行 + 头/批维并行, 解码场景也能打满。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def flash_attention_v1(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """v1: 外层 KV、内层 Q。数学上正确但要物化 Ooj/lij 中间矩阵(HBM 往返)。

    q/k/v: (B, H, S, D)。返回 (B, H, S, D)。
    """
    bsz, n_heads, seq, d = q.shape
    scale = 1.0 / math.sqrt(d)

    out = torch.zeros_like(q)
    lse = torch.zeros(bsz, n_heads, seq, device=q.device)  # 行和 l
    m_i = torch.full((bsz, n_heads, seq), float("-inf"), device=q.device)

    for k_start in range(0, seq, block_size):
        k_end = k_start + block_size
        k_block = k[:, :, k_start:k_end]  # (B,H,Bc,D)
        v_block = v[:, :, k_start:k_end]

        logits = torch.einsum("bhqd,bhkd->bhqk", q, k_block) * scale  # (B,H,S,Bc)
        if causal:
            # 行 q_idx 只能看 <= 自己的列
            q_idx = torch.arange(seq, device=q.device).unsqueeze(-1)
            k_idx = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
            logits = logits.masked_fill(k_idx > q_idx, float("-inf"))

        m_block = logits.max(dim=-1).values                            # (B,H,S)
        m_new = torch.maximum(m_i, m_block)

        # 重缩放旧行和与本块 exp
        alpha = torch.exp(m_i - m_new)
        p = torch.exp(logits - m_new.unsqueeze(-1))          # (B,H,S,Bc)
        l_new = lse * alpha + p.sum(dim=-1)

        # 注意: v1 里 out 的旧值也要乘 alpha, 再累加 p @ v_block
        out = out * alpha.unsqueeze(-1) + torch.einsum("bhqk,bhkd->bhqd", p, v_block)
        m_i, lse = m_new, l_new

    return out / lse.unsqueeze(-1)


def flash_attention_v2(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """v2: 外层 Q、内层 KV。中间状态 (m, l, o) 全程驻留"寄存器"(张量切片),
    一个 Q 块算完才写回一次 —— 这正是"为何外层遍历 Q"的答案。

    causal: 上三角 mask 的分块裁剪 —— 当 k_start > q_end 时整块跳过,
    且最后一个交叉块只需处理局部三角。
    """
    bsz, n_heads, seq, d = q.shape
    scale = 1.0 / math.sqrt(d)

    out = torch.empty_like(q)
    for q_start in range(0, seq, block_size):
        q_end = q_start + block_size
        q_block = q[:, :, q_start:q_end]  # (B,H,Br,D)

        m_i = torch.full((bsz, n_heads, q_block.shape[2]), float("-inf"), device=q.device)
        l_i = torch.zeros_like(m_i)
        acc = torch.zeros_like(q_block)  # (B,H,Br,D) 模拟寄存器里的输出累加器

        # causal 时 KV 只需扫到当前 Q 块末尾
        kv_end = q_end if causal else seq
        for k_start in range(0, kv_end, block_size):
            k_end = k_start + block_size
            k_block = k[:, :, k_start:k_end]
            v_block = v[:, :, k_start:k_end]

            logits = torch.einsum("bhqd,bhkd->bhqk", q_block, k_block) * scale

            if causal and k_end > q_start:
                # 只有"斜跨对角线"的块需要局部 mask(纯下方块不需要)
                q_idx = torch.arange(q_start, q_end, device=q.device).unsqueeze(-1)
                k_idx = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                logits = logits.masked_fill(k_idx > q_idx, float("-inf"))

            m_block = logits.max(dim=-1).values
            m_new = torch.maximum(m_i, m_block)
            alpha = torch.exp(m_i - m_new)                # 旧结果重缩放
            p = torch.exp(logits - m_new.unsqueeze(-1))   # (B,H,Br,Bc)
            l_new = l_i * alpha + p.sum(dim=-1)

            acc = acc * alpha.unsqueeze(-1) + torch.einsum("bhqk,bhkd->bhqd", p, v_block)
            m_i, l_i = m_new, l_new

        out[:, :, q_start:q_end] = acc / l_i.unsqueeze(-1)  # 唯一一次写回

    return out


@torch.no_grad()
def flash_matches_reference() -> dict[str, object]:
    """对拍: v1/v2 与 F.scaled_dot_product_attention(causal) 的数值一致性。"""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 128, 32, dtype=torch.float32)
    k = torch.randn(2, 4, 128, 32, dtype=torch.float32)
    v = torch.randn(2, 4, 128, 32, dtype=torch.float32)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    v1 = flash_attention_v1(q, k, v, causal=True)
    v2 = flash_attention_v2(q, k, v, causal=True)
    return {
        "v1_max_err": float((ref - v1).abs().max()),
        "v2_max_err": float((ref - v2).abs().max()),
        "both_close": bool(torch.allclose(ref, v1, atol=1e-4) and torch.allclose(ref, v2, atol=1e-4)),
    }
