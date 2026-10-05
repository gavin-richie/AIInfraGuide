"""MLA 矩阵吸收（weight absorption）：数值等价验证 + prefill/decode 复杂度模型。

对应面试题:
    - 小厂-实习-1 "MLA 在 prefill 与 decode 阶段的计算复杂度差异，以及矩阵吸收优化的原理"
    - 联想 "DeepSeek R1 采用了什么注意力优化方案"

记号（DeepSeek-V2/V3）:
    h        输入 hidden state (d_model)
    c_kv     = W_DKV h, 潜在 KV 向量 (d_c=512)  ← KV Cache 只存它
    k        = W_UK c_kv, v = W_UV c_kv  (n_h × d_h)
    q        = W_UQ W_DQ h

吸收推导（attention 里的矩阵乘满足结合律）:
    score: q·k^T = (W_UQ q̃)·(W_UK c_kv)^T = q̃·(W_UK^T W_UQ^T)^T … 逐头地
        K 吸收进 Q:  q_h' = W_UK,h @ q_h ∈ R^{d_c}，直接与 c_kv 做点积
    value:  o_h = A_h @ v_h = A_h @ (W_UV,h c_kv) = (A_h @ W_UV,h) @ c_kv
        V 吸收进 A: 先在 latent 空间做注意力加权平均，再一次性升维

decode（S_q=1）吸收赢: 每个缓存 token 从"升维成 n_h·d_h 的 K/V"变成"直接用 d_c 维 c_kv 参与
点积"，投影 FLOPs 降约 d_h 倍，且省去 K/V 激活的读写;
prefill（S_q 大）不吸收赢: K/V 只需物化一次，后续是 (S_q×d_h) 的大 GEMM 满血吃 Tensor Core，
而吸收把 Q 侧变成 (S_q×d_c×d_h) 的更大 GEMM，反而更慢（SGLang/vLLM 的实际实现即
"prefill 非吸收 + decode 吸收"）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def attention_naive(q: torch.Tensor, c_kv: torch.Tensor, w_uk: torch.Tensor,
                    w_uv: torch.Tensor) -> torch.Tensor:
    """朴素路径: 先把 c_kv 升维成 K/V 再做标准 MHA。

    Args:
        q:     (n_h, S_q, d_h)   query（已升维）
        c_kv:  (S_kv, d_c)       缓存的潜在向量
        w_uk:  (n_h, d_h, d_c)   升维权重
        w_uv:  (n_h, d_h, d_c)
    Returns:
        (n_h, S_q, d_h)
    """
    n_h, _, d_h = q.shape
    k = torch.einsum("hdc,sc->hsd", w_uk, c_kv)   # (n_h, S_kv, d_h)
    v = torch.einsum("hdc,sc->hsd", w_uv, c_kv)
    scores = q @ k.transpose(-1, -2) / (d_h**0.5)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def attention_absorbed(q: torch.Tensor, c_kv: torch.Tensor, w_uk: torch.Tensor,
                       w_uv: torch.Tensor) -> torch.Tensor:
    """吸收路径: K 吸收进 Q、V 吸收进注意力权重，全程不物化 K/V。

    q' = W_UK @ q        (n_h, S_q, d_c)     ← K 吸收
    u  = softmax(q' c_kv^T / sqrt(d_h)) @ c_kv  (n_h, S_q, d_c) ← latent 空间加权平均
    o  = u @ W_UV^T      (n_h, S_q, d_h)     ← V 吸收（最后一步升维）
    """
    d_h = q.shape[-1]
    qp = q @ w_uk                                   # (n_h, S_q, d_c)
    scores = qp @ c_kv.transpose(-1, -2) / (d_h**0.5)
    attn = torch.softmax(scores, dim=-1)
    u = attn @ c_kv                                 # (n_h, S_q, d_c)
    return u @ w_uv.transpose(-1, -2)               # (n_h, S_q, d_h)


@dataclass
class MLACosts:
    """两种路径的 MACs（乘加次数）对比。"""

    naive_mac: float
    absorbed_mac: float

    @property
    def speedup(self) -> float:
        return self.naive_mac / max(self.absorbed_mac, 1.0)


def mla_costs(n_h: int, d_h: int, d_c: int, s_q: int, s_kv: int) -> MLACosts:
    """只比较 attention 内核部分的 MACs（投影输入侧 W_DQ/W_DKV 两路相同，不计）。

    naive:    KV 升维 2·S_kv·n_h·d_h·d_c + score/value 2·S_q·S_kv·n_h·d_h
    absorbed: Q 吸收 S_q·n_h·d_h·d_c + score/latent 平均 2·S_q·S_kv·n_h·d_c
              + 最终升维 S_q·n_h·d_c·d_h
    """
    naive = 2 * s_kv * n_h * d_h * d_c + 2 * s_q * s_kv * n_h * d_h
    absorbed = (s_q * n_h * d_h * d_c + 2 * s_q * s_kv * n_h * d_c
                + s_q * n_h * d_c * d_h)
    return MLACosts(naive_mac=naive, absorbed_mac=absorbed)


def demo(deepseek_shapes: bool = True) -> dict[str, float]:
    """DeepSeek-V3 形状 (n_h=128, d_h=128, d_c=512) 下 decode/prefill 的对比。"""
    n_h, d_h, d_c = (128, 128, 512) if deepseek_shapes else (32, 128, 512)
    decode = mla_costs(n_h, d_h, d_c, s_q=1, s_kv=4096)
    prefill = mla_costs(n_h, d_h, d_c, s_q=4096, s_kv=4096)
    return {
        "decode_naive_GMAC": decode.naive_mac / 1e9,
        "decode_absorbed_GMAC": decode.absorbed_mac / 1e9,
        "decode_speedup": decode.speedup,
        "prefill_naive_GMAC": prefill.naive_mac / 1e9,
        "prefill_absorbed_GMAC": prefill.absorbed_mac / 1e9,
        "prefill_speedup(吸收/非吸收)": prefill.absorbed_mac / prefill.naive_mac,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    n_h, d_h, d_c, s_q, s_kv = 4, 64, 32, 3, 10
    q = torch.randn(n_h, s_q, d_h)
    c_kv = torch.randn(s_kv, d_c)
    w_uk = torch.randn(n_h, d_h, d_c)
    w_uv = torch.randn(n_h, d_h, d_c)
    out_n = attention_naive(q, c_kv, w_uk, w_uv)
    out_a = attention_absorbed(q, c_kv, w_uk, w_uv)
    print("max |naive - absorbed| =", (out_n - out_a).abs().max().item())
    for k, v in demo().items():
        print(f"{k}: {v:.3f}")
