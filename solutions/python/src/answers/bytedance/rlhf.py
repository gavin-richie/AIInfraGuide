"""RLHF 三代算法的核心损失：PPO（clip + KL）、DPO、GRPO。

对应题目：抖音电商训练优化 2/3/6（PPO 全流程 / DPO 与 PPO 区别 / GRPO 原理）、
豆包基础知识 3（PPO 与 DPO 核心思想）。SFT loss 见 ``answers/torch_impl/sft_loss.py``。

公式速查：

- **PPO clip**：``L = -E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ]``，``r = π(a|s)/π_old(a|s)``。
  A>0 时限制 r ≤ 1+ε（好动作别一步学太猛），A<0 时限制 r ≥ 1-ε。
- **token KL（k3 无偏低方差估计）**：样本取自 q（当前策略），逐 token 计算
  ``k3 = e^{logp-logq} - (logp-logq) - 1 ≥ 0``，取均值即 ``KL(q‖p)`` 的无偏估计
  （期望拆开：``E_q[p/q]=1``、``E_q[logp-logq]=-KL``）。PPO-LLM 里以
  ``reward = r_env - β·KL(π‖π_ref)`` 形式进入 advantage。
- **DPO**：``L = -E[ log σ( β(Δπ - Δref) ) ]``，Δ = logπ(chosen) - logπ(rejected)。
  把 RLHF 的 reward 最大化解出闭式代入 Bradley-Terry 得到，绕开显式 reward model
  与采样，直接在偏好对上优化策略。
- **GRPO**：同一条 prompt 采 G 个回答为一组，advantage = (r - 组均值)/组标准差
  （组内相对排名），去掉 value model；Dr.GRPO 变体再去掉 std 只减均值。

运行验证::

    cd solutions/python && python -m pytest tests/test_bytedance.py -q
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------- PPO


def ppo_clip_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """标准 PPO clip 目标（返回要最小化的 loss）。logp/old_logp/advantages 同形状。"""
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def k3_kl_estimator(logq: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """KL(q‖p) 的 k3 无偏估计（样本来自 q，逐 token 非负）。

    k3(x) = e^{logp(x)-logq(x)} - (logp(x)-logq(x)) - 1；
    E_{x~q}[k3] = 1 + KL(q‖p) - 1 = KL(q‖p)。
    """
    t = logp - logq
    return torch.exp(t) - t - 1.0


def ppo_lm_reward(
    env_reward: torch.Tensor,
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """RLHF 里 LLM 的 per-token reward：r_env（只在序列末 token 非零）- β·KL。"""
    return env_reward - beta * k3_kl_estimator(policy_logp, ref_logp)


# ----------------------------------------------------------------------- DPO


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> dict[str, torch.Tensor]:
    """DPO 损失与诊断量。*_logp 均为序列级 log prob（sum over tokens）。

    返回 loss / implicit_reward（β(Δπ-Δref)，即 Bradley-Terry 隐式奖励差）、
    accuracy（隐式奖励 > 0 的比例，训练监控核心指标）。
    """
    pi_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_margin - ref_margin)
    loss = -F.logsigmoid(logits).mean()
    return {
        "loss": loss,
        "implicit_reward": logits,
        "accuracy": (logits > 0).float().mean(),
    }


# ---------------------------------------------------------------------- GRPO


def grpo_advantages(
    rewards: torch.Tensor, group_size: int, use_std: bool = True
) -> torch.Tensor:
    """组内相对 advantage：每 group_size 条回答一组，(r - mean)/组内总体标准差。

    ``use_std=False`` 即 Dr.GRPO（只减均值），避免小 std 放大噪声与长度偏置。
    组内 std=0（奖励全同）时 advantage 记 0。
    """
    if rewards.numel() % group_size:
        raise ValueError("rewards length must be a multiple of group_size")
    groups = rewards.view(-1, group_size)
    centered = groups - groups.mean(dim=-1, keepdim=True)
    if not use_std:
        return centered.reshape(-1)
    std = centered.std(dim=-1, keepdim=True, unbiased=False)
    adv = centered / std.clamp_min(1e-6)
    adv = torch.where(std > 1e-6, adv, torch.zeros_like(adv))
    return adv.reshape(-1)
