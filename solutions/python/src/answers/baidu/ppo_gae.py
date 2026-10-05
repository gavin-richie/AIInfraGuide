"""PPO / GAE / V-trace 手写实现 —— 百度一面-3 强化学习全卷的核心代码题。

对应面试题:
    - 百度-AI-Infra-一面-3 Q3 "PPO clip 在 A>0/A<0 时分别如何限制上下界"
    - 百度-AI-Infra-一面-3 Q4 "PPO 损失函数 + GAE 计算方式, lambda 对偏差方差的影响"
    - 百度-AI-Infra-实习-1 Q4 "异步 RL 训练场景下如何对损失函数进行修正"

实现要点:
    - ppo_clip_loss: 单向截断语义(A>0 越上界梯度为 0, A<0 越下界梯度为 0, 回头方向始终有梯度);
    - compute_gae:   向量化 GAE(lambda), lambda=0 退化为一步 TD, lambda=1 退化为 MC-基线;
    - vtrace_correction: IMPALA 式截断重要性采样(异步/stale 样本的修正)。
"""
from __future__ import annotations

import torch


def ppo_clip_loss(
    logp: torch.Tensor,          # 当前策略 log pi_theta(a|s), 形状 (N,)
    logp_old: torch.Tensor,      # 采样策略 log pi_old(a|s), 形状 (N,)
    advantages: torch.Tensor,    # 优势 A_t, 形状 (N,)
    eps: float = 0.2,
) -> torch.Tensor:
    """L^CLIP = E[min(r*A, clip(r, 1-eps, 1+eps)*A)],  r = pi_theta / pi_old。"""
    r = torch.exp(logp - logp_old)
    clipped = torch.clamp(r, 1.0 - eps, 1.0 + eps)
    return -torch.min(r * advantages, clipped * advantages).mean()


def clip_gradient_mask(
    logp: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor, eps: float = 0.2,
) -> torch.Tensor:
    """返回 bool 掩码: 哪些样本的 r 对 loss 的梯度被 clip 掉(为 True 表示无梯度)。

    面试题"clip 在 A 正负下的上下界"的数值验证:
        A > 0 且 r > 1+eps  -> 上界截断(梯度 0), "不再往增大方向推";
        A < 0 且 r < 1-eps  -> 下界截断(梯度 0), "不再往减小方向推";
        反方向越界仍有梯度(min 选中未截断分支)。
    """
    r = torch.exp(logp - logp_old)
    upper = (advantages > 0) & (r > 1.0 + eps)
    lower = (advantages < 0) & (r < 1.0 - eps)
    return upper | lower


def compute_gae(
    rewards: torch.Tensor,       # r_t,        形状 (T,)
    values: torch.Tensor,        # V(s_t),     形状 (T,)
    next_values: torch.Tensor,   # V(s_{t+1}), 形状 (T,)（终态传 0）
    dones: torch.Tensor,         # 终止标志,   形状 (T,), 1 表示 s_{t+1} 是新 episode 起点
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """向量化 GAE(lambda):  A_t = sum_i (gamma*lam)^i * delta_{t+i},
    delta_t = r_t + gamma * (1-done) * V(s_{t+1}) - V(s_t)。

    返回 (advantages, returns), returns = advantages + values (critic 的回归目标)。
    """
    T = rewards.shape[0]
    deltas = rewards + gamma * (1.0 - dones) * next_values - values
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype)
    for t in reversed(range(T)):        # 反向递推: A_t = delta_t + gamma*lam*(1-done_t)*A_{t+1}
        gae = deltas[t] + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values


def gae_lambda_extremes(rewards, values, next_values, dones, gamma=0.99):
    """lambda 两端的退化验证(面试"偏差-方差权衡"的数值证据):

    lambda = 0  -> A_t = delta_t (一步 TD: 依赖 V, 低方差高偏差);
    lambda = 1  -> A_t = sum gamma^i r_{t+i} - V(s_t) (MC 回报减基线: 无偏但方差大)。
    """
    td, _ = compute_gae(rewards, values, next_values, dones, gamma, lam=0.0)
    mc, _ = compute_gae(rewards, values, next_values, dones, gamma, lam=1.0)
    return td, mc


def vtrace_correction(
    behavior_logp: torch.Tensor,   # 采样策略(旧/异步 actor) log mu(a|s), (N,)
    target_logp: torch.Tensor,     # 当前策略 log pi(a|s), (N,)
    values: torch.Tensor,          # V(s), (N,)
    next_values: torch.Tensor,     # V(s'), (N,)
    rewards: torch.Tensor,         # r, (N,)
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V-trace 截断重要性采样 (IMPALA), 异步 RL 的损失修正。

    rho_t = min(rho_bar, pi/mu)     # 截断的策略比率, 控制价值目标与策略梯度的偏差
    c_t   = min(c_bar,  pi/mu)      # 截断的轨迹系数, 保证误差不随时间步累积
    v_t   = V(s_t) + sum_i gamma^i (prod_{j<t} c_j) * rho_{t+i} * delta_{t+i}

    返回 (vtrace_returns, rho)。面试答法: 异步样本的 ratio 会随版本差距爆炸,
    无截断的 IS 在长轨迹上方差无界; V-trace 用 rho/c 双截断换取收敛保证。
    """
    log_ratio = (target_logp - behavior_logp).clamp(-10.0, 10.0)
    ratio = torch.exp(log_ratio)
    rho = torch.clamp(ratio, max=rho_bar)
    c = torch.clamp(ratio, max=c_bar)
    delta_raw = rewards + gamma * next_values - values          # 原始 TD 残差

    T = rewards.shape[0]
    vtrace = torch.zeros_like(values)
    acc = torch.zeros((), dtype=values.dtype)                   # A_t = rho_t*delta_t + gamma*c_t*A_{t+1}
    for t in reversed(range(T)):
        acc = rho[t] * delta_raw[t] + gamma * c[t] * acc
        vtrace[t] = values[t] + acc
    return vtrace, rho
