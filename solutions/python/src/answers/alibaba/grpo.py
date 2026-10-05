"""GRPO / PPO 核心实现（阿里巴巴-淘天-AI-Infra-一面-2 Q1-Q5）。

面试考点全覆盖：
1. 重要性采样比率 ratio = π_θ(a|s) / π_θ_old(a|s)，允许用旧策略采集的样本
   多次更新新策略（off-policy 复用样本，提高样本利用率）；
2. PPO-Clip：min(ratio·A, clip(ratio)·A)——正优势时封顶、负优势时托底，
   限制单步更新幅度防止策略崩溃（置信域的廉价替代）；
3. GRPO 组相对优势：同一 prompt 采样 G 个回答，A_i = (r_i - mean)/std，
   去掉 value network（critic），用组内基线代替；
4. k3 无偏低 KL 估计：E_π[exp(logπ_ref - logπ_θ) - (logπ_ref - logπ_θ) - 1]，
   逐点非负、无需求解整个分布的 KL。
"""

from __future__ import annotations

import torch


def importance_ratio(logp_new: torch.Tensor, logp_old: torch.Tensor) -> torch.Tensor:
    """r = exp(logπ_θ - logπ_old)。log-space 相减再取指数，避免数值下溢。"""
    return torch.exp(logp_new - logp_old)


def ppo_clipped_surrogate(
    ratio: torch.Tensor, adv: torch.Tensor, clip_eps: float = 0.2
) -> torch.Tensor:
    """PPO 裁剪后的 surrogate（per-token，未聚合）。

    L = min(r·A, clip(r, 1-ε, 1+ε)·A)：
    - A>0（好动作）：ratio 超过 1+ε 后 loss 不再下降 → 梯度截断，防止贪心过大步；
    - A<0（坏动作）：ratio 低于 1-ε 后 loss 不再下降 → 防止过度惩罚已远离的动作。
    """
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return torch.min(surr1, surr2)


def ppo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    adv: torch.Tensor,
    clip_eps: float = 0.2,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """标准 PPO policy loss（对 token/样本取 masked mean 后取负号做最小化）。"""
    ratio = importance_ratio(logp_new, logp_old)
    per_token = ppo_clipped_surrogate(ratio, adv, clip_eps)
    return -masked_mean(per_token, token_mask)


def grpo_group_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """组相对优势：A_i = (r_i - mean(r)) / (std(r) + eps)。

    同一 prompt 的 G 个 rollout 组成一个组；组内归一化把绝对奖励变成相对优劣，
    省去 critic/value network。可分辨性要求组内奖励不能全相同（全同 → A=0 无梯度）。
    """
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def k3_kl_estimator(logp_pi: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """k3 无偏低 KL 估计（逐点非负），样本来自 π_θ。

    E_π[exp(logπ_ref - logπ_θ) - (logπ_ref - logπ_θ) - 1] = KL(π_θ || π_ref)。
    逐点非负的原因：e^t - t - 1 ≥ 0 对任意 t 成立。
    """
    diff = logp_ref - logp_pi
    return torch.exp(diff) - diff - 1


def grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    adv: torch.Tensor,
    logp_ref: torch.Tensor | None = None,
    beta: float = 0.0,
    clip_eps: float = 0.2,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """DeepSeek-GRPO 目标（含旧策略比率裁剪与 KL 正则的常见实现）：

    L = -E[ min(r·A, clip(r)·A) ] + β · E[ KL(π_θ || π_ref) ]
    其中 A 为组相对优势（grpo_group_advantage），KL 用 k3 估计。
    β=0 时退化为纯组相对策略梯度；logp_old=logp_new 时退化为 on-policy。
    """
    ratio = importance_ratio(logp_new, logp_old)
    pg = -masked_mean(ppo_clipped_surrogate(ratio, adv, clip_eps), token_mask)
    if beta > 0.0:
        if logp_ref is None:
            raise ValueError("beta>0 需要 ref 策略的 logp_ref（冻结的初始模型）")
        kl = masked_mean(k3_kl_estimator(logp_new, logp_ref), token_mask)
        return pg + beta * kl
    return pg


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """按 response token mask 求平均（padding/指令段不参与）。"""
    if mask is None:
        return x.mean()
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def ppo_vs_grpo_demo(seed: int = 0) -> dict[str, float]:
    """一个 batch 的数值演示：PPO 需要 value 网络（GAE），GRPO 用组内基线。

    返回 PPO（critic 给出的 A）与 GRPO（组相对 A）在同一批奖励下的优势相关性、
    clip 前后 loss 的差异，供文档与测试引用。
    """
    torch.manual_seed(seed)
    # 4 个 prompt × 组大小 G=8 的奖励；PPO 风格优势用 (r - V) 近似（V 取组均值）
    rewards = torch.randn(4, 8)
    adv_grpo = torch.stack([grpo_group_advantage(r) for r in rewards]).flatten()
    adv_ppo = (rewards - rewards.mean(dim=1, keepdim=True)).flatten()  # 无 std 归一化的 baseline 版
    corr = torch.corrcoef(torch.stack([adv_grpo, adv_ppo]))[0, 1].item()

    # clip 效果：ratio 偏离 1 越远、优势为正时，clip 把更新幅度限制住
    ratio = torch.tensor([0.5, 0.9, 1.0, 1.1, 2.0], requires_grad=True)
    adv = torch.ones(5)
    clipped = ppo_clipped_surrogate(ratio, adv, 0.2)
    unclipped = ratio * adv
    return {
        "adv_corr_ppo_grpo": corr,
        "clipped_gain": clipped.sum().item(),
        "unclipped_gain": unclipped.sum().item(),
    }
