"""Online Softmax 手写实现 —— FlashAttention 的数学基础。

对应面试题:
    - 快手-校招-2 "Online Softmax 与 FA 伪代码"
    - 海康 "Online Softmax + FA"
    - 小米-实习 "Online Softmax"
    - 飞腾 "Softmax 稳定与 Online Softmax"

核心思想(三步推导):
    普通数值稳定 softmax 需要"全局 max", 导致必须两遍扫描:
        m = max(x);  l = sum(exp(x - m));  out = exp(x - m) / l
    Online 版本把序列分块, 增量维护 (m, l):
        新块到来时:
            m_new = max(m_old, max(x_block))
            l_new = l_old * exp(m_old - m_new) + sum(exp(x_block - m_new))
    旧贡献乘 exp(m_old - m_new) 做"重归一化", 单遍扫描即可。
"""
from __future__ import annotations

import torch


def softmax_naive(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """未做数值稳定的朴素版: 大输入直接 exp 溢出。"""
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """标准数值稳定版: 减全局 max。两遍扫描(一遍 max + 一遍 exp/sum)。"""
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=dim, keepdim=True)


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """单遍扫描 Online Softmax(逐"行"处理, 行内按块流式更新)。

    x: (N,) 一维向量。返回与 softmax_stable 一致的结果。
    这里按每块 8 个元素模拟流式分块, 展示算法结构。
    核心维护 (m, l): 新块 max 更新时, 历史分子要整体乘 exp(m_old - m_new)
    重归一化 —— 这是 FA 单遍计算的关键一步。
    """
    import math

    n = x.shape[0]
    m = float("-inf")   # 当前见过的最大值
    l = 0.0             # 当前归一化分母(基于 m)
    numerator = torch.zeros_like(x)  # 未归一化分子(以当前 m 为基准)
    block = 8
    for start in range(0, n, block):
        xb = x[start:start + block]
        m_new = max(m, xb.max().item())
        scale = math.exp(m - m_new) if m != float("-inf") else 0.0
        numerator[:start] *= scale                 # 历史块重缩放到新基准
        numerator[start:start + block] = torch.exp(xb - m_new)
        l = l * scale + numerator[start:start + block].sum().item()
        m = m_new
    return numerator / l


def online_attention_row(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """FlashAttention 单 query 行的在线 attention(核心内循环):

        o = o * exp(m_old - m_new) * (l_old / l_new) + exp(qk - m_new) @ v / l_new
    分块流式更新 (m, l, o), 不需要物化完整 attention 矩阵。
    q: (D,), k/v: (S, D) → out: (D,)
    """
    d = q.shape[-1]
    m = float("-inf")
    l = 0.0
    o = torch.zeros_like(v[0])
    block = 16
    for start in range(0, k.shape[0], block):
        kb, vb = k[start:start + block], v[start:start + block]
        logits = (kb @ q) / (d ** 0.5)          # (B,)
        m_b = logits.max().item()
        m_new = max(m, m_b)
        p = torch.exp(logits - m_new)           # (B,) 当前块未归一化权重
        l_new = l * math_exp(m - m_new) + p.sum().item()
        o = o * math_exp(m - m_new) + p @ vb    # 旧输出重缩放后累加新块
        m, l = m_new, l_new
    return o / l


def math_exp(x: float) -> float:
    import math

    return math.exp(x) if x != float("-inf") else 0.0
