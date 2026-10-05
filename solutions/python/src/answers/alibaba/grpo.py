"""PPO / GRPO 核心组件（阿里巴巴-淘天-AI-Infra-一面-2 Q1-Q5）。

对应面试题：
    Q1 GRPO 与 PPO 的核心区别；
    Q2 PPO Clip 的具体作用；
    Q3 重要性采样的作用（除限制更新幅度外的用途）、与 KL 约束的区别；
    Q4 马尔可夫性质的形式化定义；
    Q5 从早期策略梯度到 GRPO 的演进脉络（保留/舍弃了什么）。

本模块实现四个可测试的核心组件：

1. ``ppo_clip_loss``：L = -E[ min(r·A, clip(r, 1-eps, 1+eps)·A) ]，
   r = π_θ(a|s) / π_θold(a|s)。
2. ``group_relative_advantage``：GRPO 去掉 critic，用同一 prompt 组内
   多个 rollout 回报的均值当 baseline（除以 std 归一，Dr.GRPO 指出
   std 归一会引入长度偏差，可选项）。
3. ``grpo_loss``：clip 目标 - β·k3 KL 估计（KL 直接进 loss，而非像 PPO
   那样 shaping 进 reward）。
4. ``k3_kl_estimator``：Schulman 的 k3 单样本 KL 估计（GRPO 论文采用）：
   目标是估 KL[π_θ || π_ref] = E_{x~π_θ}[logπ_θ - logπ_ref]，样本只能从
   当前策略 π_θ 采。令 r = π_ref/π_θ，
       k1 = -log r                （E = KL[p||q]？——有偏）
       k2 = (1/2)(log r)^2        （逐点非负，但有偏）
       k3 = r - log r - 1         （无偏且逐点非负，r - 1 >= log r）
   GRPO 把 -β·k3 直接加进 loss，约束 π_θ 不偏离冻结的 π_ref。
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------- PPO Clip
def importance_ratio(logp: torch.Tensor, logp_old: torch.Tensor) -> torch.Tensor:
    """重要性采样权重 r = π_θ(a|s) / π_θold(a|s)，log 域相减防下溢。"""
    return torch.exp(logp - logp_old)


def ppo_clip_loss(logp: torch.Tensor, logp_old: torch.Tensor,
                  advantage: torch.Tensor, eps: float = 0.2) -> torch.Tensor:
    """PPO-Clip 目标（取负号变成最小化 loss）。

    clip 的作用（Q2）：把 r 限制在 [1-eps, 1+eps]，超过界后梯度截断——
      A > 0（好动作）：r > 1+eps 不再增奖励 → 防止一步把概率推得过大；
      A < 0（坏动作）：r < 1-eps 不再继续压概率 → 防止摧毁旧策略。
    本质是 trust region 的零阶近似：单样本上限制策略更新幅度，替代 TRPO 的
    二阶 KL 约束，换来实现简单、可 minibatch 复用样本。
    """
    r = importance_ratio(logp, logp_old)
    unclipped = r * advantage
    clipped = torch.clamp(r, 1.0 - eps, 1.0 + eps) * advantage
    return -torch.min(unclipped, clipped).mean()


# ---------------------------------------------------------------- GRPO
def group_relative_advantage(rewards: torch.Tensor,
                             normalize_std: bool = True) -> torch.Tensor:
    """GRPO 组相对优势：A_i = (r_i - mean(group)) [/ std(group)]。

    同一 prompt 采样 G 条回答，组均值是天然的无偏 baseline（组越大方差越低），
    从而省掉整个 value network（PPO 的 critic）——省显存、免 critic 与
    policy 的博弈失衡。normalize_std=False 即 Dr.GRPO 的去偏差版本。
    """
    mean = rewards.mean()
    adv = rewards - mean
    if normalize_std:
        adv = adv / rewards.std().clamp(min=1e-6)
    return adv


def k3_kl_estimator(logp_cur: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """KL[π_θ || π_ref] 的 k3 单样本估计（样本来自当前策略 π_θ）。

    k3 = r - log r - 1，其中 r = π_ref/π_θ = exp(logp_ref - logp_cur)：
    无偏（E_{π_θ}[k3] = KL[π_θ||π_ref]）且逐点非负（r-1 >= log r）。
    GRPO 把 -β·KL 直接加进 loss（约束当前策略不偏离冻结参考 π_ref），
    与 PPO 在 reward 里加 KL shaping 不同：loss-KL 的梯度不会被 clip 截断、
    参考策略固定不会 "reward hacking"。
    """
    logr = logp_ref - logp_cur
    return torch.exp(logr) - logr - 1.0


def grpo_loss(logp: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor,
              rewards: torch.Tensor, eps: float = 0.2, beta: float = 0.04,
              normalize_std: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO 完整目标：clip(重要性采样) 目标 + β·k3 KL 正则。

    Returns:
        (loss, advantage)：advantage 为组相对优势，供测试断言。
    """
    adv = group_relative_advantage(rewards, normalize_std=normalize_std)
    policy_loss = ppo_clip_loss(logp, logp_old, adv, eps)
    kl = k3_kl_estimator(logp, logp_ref).mean()
    return policy_loss + beta * kl, adv


# ---------------------------------------------------------------- 马尔可夫性（Q4）
def markov_property_example() -> dict[str, bool]:
    """对比满足/违反马尔可夫性的转移函数（形式化定义见模块 docstring）。

    定义：状态 s_t 具有马尔可夫性 ⟺
        P(s_{t+1} | s_t, a_t, s_{0:t-1}, a_{0:t-1}) = P(s_{t+1} | s_t, a_t)
    即给定当前状态与动作，未来与历史条件独立。

    Examples:
        - 马尔可夫：next = f(s, a)（一阶马尔可夫链 / MDP）；
        - 非马尔可夫：next 依赖上上步动作（二阶过程）——除非把
          (s_t, a_{t-1}) 拼起来当新状态，才能恢复马尔可夫性（状态增广）。
    """
    def markov_next(state: float, action: float, prev_action: float = 0.0) -> float:
        return 0.9 * state + 0.1 * action  # prev_action 被忽略 → 马尔可夫

    def non_markov_next(state: float, action: float, prev_action: float = 0.0) -> float:
        return 0.9 * state + 0.1 * action + 0.5 * prev_action  # 依赖历史动作

    s, a1, a2 = 0.6, 0.3, -0.4
    return {
        "markov_process": markov_next(s, a1, prev_action=a2) == markov_next(s, a1, prev_action=-a2),
        "non_markov_process": non_markov_next(s, a1, prev_action=a2) != non_markov_next(s, a1, prev_action=-a2),
    }


# ---------------------------------------------------------------- 演进脉络（Q5）
POLICY_GRAD_EVOLUTION: list[tuple[str, str, str]] = [
    # (算法, 保留/引入, 舍弃/痛点)
    ("REINFORCE", "策略梯度定理：∇J = E[∇logπ·G_t]", "Monte Carlo 回报方差极大"),
    ("Vanilla PG + baseline", "减去 baseline b(s) 降方差（不引入偏差）", "baseline 手工设计，非最优"),
    ("Actor-Critic", "用 V(s) 学习 baseline，TD 误差即 advantage", "引入 critic：偏差、训练不稳、显存"),
    ("GAE(λ)", "λ 回归在 TD 与 MC 间折中，平滑 bias/variance", "仍依赖 critic 质量"),
    ("TRPO", "KL 信任域约束单调改进", "二阶优化（共轭梯度+Fisher），实现重"),
    ("PPO", "clip 一阶近似信任域 + 重要性采样支持样本复用", "KL 约束软化，仍需 critic"),
    ("GRPO", "组内均值 baseline 去 critic；KL(k3) 进 loss", "critic 网络与 reward shaping KL"),
]
