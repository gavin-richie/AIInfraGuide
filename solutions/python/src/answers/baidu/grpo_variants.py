"""GRPO 及其变体的损失/优势实现 —— 百度一面-3 Q5/Q6 的配套代码。

对应面试题:
    - 百度-AI-Infra-一面-3 Q5 "GRPO 的损失计算方式, 序列级别损失如何分配到每个 token;
       序列级别平均与批次级别平均的区别"
    - 百度-AI-Infra-一面-3 Q6 "其他 GRPO 变体(DAPO、GSPO 等)的特点"

实现要点:
    - grpo_advantage: 组内基线 (mean/std 归一), 无 critic;
    - grpo_loss: sequence-mean(GRPO 原始) vs token-mean(DAPO) 两种归一化的开关对比;
    - dapo_clip_higher: DAPO 的非对称裁剪 (上界放宽 epsilon_hi);
    - gspo_sequence_ratio: 序列级比率 = token 比率的几何平均 (log 域求和/token 数)。
"""
from __future__ import annotations

import torch


def grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """组内基线: A_i = (r_i - mean(r)) / (std(r) + 1e-6), 同一序列内所有 token 共享。

    rewards: (G,) 一个 prompt 下 G 个回答的奖励。
    返回 (G,) 的优势。组内无区分度(全对/全错)时 std=0 -> 优势恒 0,
    这正是 DAPO dynamic sampling 要过滤掉的"白算组"。
    """
    std = rewards.std()
    return (rewards - rewards.mean()) / (std + 1e-6)


def grpo_loss(
    logp: torch.Tensor,           # 当前策略各 token 的 logprob, 形状 (G, T)
    logp_old: torch.Tensor,       # 采样策略, (G, T)
    advantages_seq: torch.Tensor, # 每条序列的优势(广播到 token), (G,)
    loss_mask: torch.Tensor,      # 回答 token 掩码(-pad), (G, T)
    reduction: str = "token",     # "sequence"(GRPO 原始) | "token"(DAPO)
    eps: float = 0.2,
    eps_hi: float | None = None,  # DAPO clip-higher: 上界用更宽的 eps_hi
) -> torch.Tensor:
    """GRPO/DAPO 策略损失。per-token 比率 clip 后:

    sequence-mean: 先序列内按 token 平均, 再对 G 条序列平均(每条回答权重相等,
                   长回答的单 token 梯度被 1/|o_i| 稀释);
    token-mean:    batch 内所有有效 token 一起平均(每个 token 权重相等,
                   长序列贡献总梯度更大 —— DAPO 的 Token-Level PG)。
    """
    assert reduction in ("sequence", "token")
    eps_hi = eps if eps_hi is None else eps_hi
    r = torch.exp(logp - logp_old)                              # (G, T)
    a = advantages_seq.unsqueeze(-1).expand_as(r)               # (G, T)
    lo = torch.clamp(r, 1.0 - eps, 1.0 + eps_hi)                # 非对称 clip
    per_tok = -torch.min(r * a, lo * a) * loss_mask             # (G, T)

    if reduction == "sequence":
        seq_len = loss_mask.sum(dim=-1).clamp(min=1.0)          # |o_i|
        per_seq = per_tok.sum(dim=-1) / seq_len                 # 序列内 token 平均
        return per_seq.mean()                                   # G 条序列平均
    return per_tok.sum() / loss_mask.sum().clamp(min=1.0)       # 全 batch token 平均


def sequence_vs_token_mean_demo() -> dict[str, float]:
    """长短序列混合时两种归一化的梯度尺度差异(面试"区别"的数值演示)。

    构造: 2 条回答 —— 短(4 token)与长(36 token), 每条序列内所有 token 的
    per-token 损失同为 1。sequence-mean 下两者各贡献 1/2; token-mean 下
    长序列贡献 0.9 —— 即"长序列主导更新", 需配 DAPO 的 overlong 过滤。
    """
    g = torch.tensor([
        [1.0] * 4 + [0.0] * 32,    # 短回答(pad 0)
        [1.0] * 36,                # 长回答
    ])
    mask = (g > 0).float()
    per_tok = g                                                   # 已是逐 token 损失
    seq_mean = (per_tok.sum(-1) / mask.sum(-1)).mean().item()
    tok_mean = (per_tok.sum() / mask.sum()).item()
    long_share = (per_tok[1].sum() / per_tok.sum()).item()
    return {"sequence_mean": seq_mean, "token_mean": tok_mean,
            "long_sequence_token_share": long_share}


def gspo_sequence_ratio(
    logp: torch.Tensor, logp_old: torch.Tensor, loss_mask: torch.Tensor,
) -> torch.Tensor:
    """GSPO 序列级比率: exp( mean_t [log pi - log pi_old] ) —— token 比率的几何平均。

    对比: token 级比率连乘(算术上 exp(sum)) 在长序列上指数级偏离 1、方差爆炸;
    几何平均(除以 token 数)把比率拉回 O(1), 序列整体 clip, 训练更稳。
    返回 (G,)。
    """
    diff = (logp - logp_old) * loss_mask
    return torch.exp(diff.sum(-1) / loss_mask.sum(-1).clamp(min=1.0))
