"""RL 基础与 PPO / DPO / GRPO 的最小可运行实现（面向面试讲解）。

来源: 京东-AI-Infra-实习 Q2/Q6/Q9（PPO/DPO/GRPO 区别、Advantage 计算、Critic 更新、
奖励函数与冷启动）、Q3（熵/交叉熵/KL 的联系、分类为何不用 MSE）。

三个 loss 都写成 batch 张量函数，与论文公式一一对应：
- PPO:     r_t(θ)·A_t 裁剪 surrogate；A_t 用 GAE(λ)；critic 回归 V(s) 目标
- DPO:     -log σ(β·[(π_θ(y_w|x)/π_ref(y_w|x)) - (π_θ(y_l|x)/π_ref(y_l|x))])
- GRPO:    组内去均值（可选去 std）做 advantage，无 critic；加 KL 惩罚项
为什么分类/采样不用 MSE：见 mse_vs_ce_gradient_demo —— MSE+sigmoid 的梯度
在错误且自信时消失（sigmoid 饱和），CE 的梯度残差 (p - y) 不饱和。
"""

import torch
import torch.nn.functional as F


# ---------- 信息论三件套：熵 / 交叉熵 / KL（京东 Q3） ----------


def shannon_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """H(p) = -Σ p log p，输入为概率分布（已归一化，支持 batch）。"""
    return -(p * (p + eps).log()).sum(dim=-1)


def cross_entropy(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """H(p, q) = -Σ p log q。注意 H(p,q) = H(p) + KL(p‖q) >= H(p)。"""
    return -(p * (q + eps).log()).sum(dim=-1)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(p‖q) = Σ p (log p - log q) >= 0，不对称（这是它与距离的区别）。"""
    return (p * ((p + eps).log() - (q + eps).log())).sum(dim=-1)


def k3_kl_estimator(log_pi: torch.Tensor, log_ref: torch.Tensor) -> torch.Tensor:
    """GRPO/DeepSeek 用的无偏低方差 KL 估计 k3：0.5(log(π/ref))^2 - log(π/ref)。

    逐 token 采样估计 E_x~π[KL] 时只需两边 log 概率，不需要全词表 softmax。
    """
    log_ratio = log_pi - log_ref
    return 0.5 * log_ratio.square() - log_ratio


def mse_vs_ce_gradient_demo(z: torch.Tensor, target: float = 1.0) -> dict[str, torch.Tensor]:
    """二元分类：logit=z，目标 y。比较 sigmoid+MSE 与 sigmoid+CE 对 z 的梯度。

    结论（代码可验）：y=1、z=-10（错得很自信）时 MSE 梯度 ≈ σ(z)·(1-σ(z))·(σ(z)-y)
    ≈ 4.5e-5（饱和消失），CE 梯度 = σ(z)-y ≈ -1（不饱和，纠错信号强）。
    """
    p = torch.sigmoid(z)
    grad_mse = p * (1 - p) * (p - target)  # d/dz [0.5(σ(z)-y)^2]
    grad_ce = p - target  # d/dz [CE]（教科书结论：softmax/sigmoid+CE 残差形式）
    return {"grad_mse": grad_mse, "grad_ce": grad_ce}


# ---------- PPO（京东 Q2/Q6） ----------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(λ)：δ_t = r_t + γV(s_{t+1}) - V(s_t)；A_t = Σ (γλ)^l δ_{t+l}。

    rewards/values/next_values/dones: (T,) 或 (B, T)。returns = A + V（critic 回归目标）。
    """
    squeeze = rewards.dim() == 1
    if squeeze:
        rewards, values, next_values, dones = (t.unsqueeze(0) for t in (rewards, values, next_values, dones))
    delta = rewards + gamma * next_values * (1 - dones) - values
    advantage = torch.zeros_like(delta)
    gae = torch.zeros_like(delta[..., 0])
    T = delta.shape[-1]
    for t in reversed(range(T)):
        gae = delta[..., t] + gamma * lam * (1 - dones[..., t]) * gae
        advantage[..., t] = gae
    returns = advantage + values
    if squeeze:
        advantage, returns = advantage.squeeze(0), returns.squeeze(0)
    return advantage, returns


def ppo_clipped_surrogate(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """L_CLIP = E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ]，r = exp(logπ_new - logπ_old)。"""
    ratio = torch.exp(log_probs_new - log_probs_old)
    sur1 = ratio * advantages
    sur2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return torch.min(sur1, sur2).mean()


def critic_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Critic（价值网络）就是回归：MSE(V(s_t), R_t)，配 value clip 更稳。"""
    return F.mse_loss(values, returns)


def ppo_policy_entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """策略熵 bonus（鼓励探索，防止过早坍缩），从 logits 直接算。"""
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1).mean()


# ---------- DPO ----------


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L_DPO = -E[ log σ( β ( Δπ_chosen - Δπ_rejected ) ) ]。

    Δπ = log π_θ(y|x) - log π_ref(y|x)（对整条回答的序列 log 概率求和）。
    返回 (loss, 隐式奖励差)。
    """
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin).mean()
    return loss, margin


# ---------- GRPO ----------


def grpo_group_advantage(
    rewards: torch.Tensor, eps_std: float = 1e-6
) -> torch.Tensor:
    """组内归一化：A_i = (r_i - mean(group)) / (std(group) + eps)。

    rewards: (num_prompts, group_size)。同一 prompt 采样 G 条回答，
    用组内基线替代 PPO 的 critic —— GRPO 去掉价值网络的关键一步。
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps_std)


def grpo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    log_probs_ref: torch.Tensor | None = None,
    kl_coef: float = 0.0,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """GRPO 目标：重要性采样比 + 组优势 + 裁剪，外加对参考策略的 KL 惩罚。

    DeepSeekMath 的 KL 放在 loss（k3 估计）而非 reward 里，避免 reward 尺度污染。
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    sur1 = ratio * advantages
    sur2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.min(sur1, sur2).mean()
    if kl_coef > 0 and log_probs_ref is not None:
        loss = loss + kl_coef * k3_kl_estimator(log_probs_new, log_probs_ref).mean()
    return loss


def ppo_vs_grpo_summary() -> str:
    """两分钟口头版对比（面试直答）。"""
    return (
        "PPO: actor+critic 双网络, GAE(λ) 算优势, 裁剪 surrogate + value clip + "
        "entropy bonus, on-policy, 显存/算力翻倍, 对超参敏感;\n"
        "DPO: 免采样免 reward 模型的离线偏好优化, 直接在 (chosen, rejected) 对上"
        "最大化隐式奖励差, 简单稳定但表达能力受限于偏好数据分布;\n"
        "GRPO: 去掉 critic, 同一 prompt 采样 G 条组内对比出优势, 训练成本介于两者之间, "
        "KL(k3) 直接放 loss; 是 DeepSeek-R1 验证过的大规模 RL 路线。"
    )
