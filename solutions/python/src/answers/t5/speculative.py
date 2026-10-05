"""投机采样（Speculative Decoding）draft/target 交互模拟。

对应面试题:
    - 小厂-实习-1 "投机采样推理中草稿模型与主模型的交互流程，比较 vLLM 与 SGLang 的实现差异"

算法（Leviathan et al. 2023 / Chen et al. 2023，无损采样）:
    1. draft 模型自回归地快速采出 γ 个 token（小模型，逐 token 便宜）
    2. target 模型一次并行前向验证这 γ+1 个位置（大 GEMM，摊薄权重读取）
    3. 从左到右逐个接受: token x 以 min(1, p(x)/q(x)) 的概率被接受
       - 全部接受 → 从 target 分布额外采样 bonus token
       - 第 i 个被拒 → 从归一化残差 norm(max(p-q, 0)) 重采样并停止
    4. 输出分布与直接从 target 采样**严格等价**（这是"无损"的含义）
"""
from __future__ import annotations

import torch


def speculative_step(target_logits: torch.Tensor, draft_logits: torch.Tensor,
                     q_draft_tokens: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """执行一次 verify 步。

    Args:
        target_logits: (gamma+1, vocab) target 对草稿序列每个位置的 logits
        draft_logits:  (gamma, vocab)   draft 采样时的 logits（最后一位无草稿）
        q_draft_tokens: (gamma,) 草稿采出的 token id
    Returns:
        (<=gamma+1,) 本步实际产出（被接受前缀 + 重采样/bonus token）
    """
    p = torch.softmax(target_logits.float(), dim=-1)
    q = torch.softmax(draft_logits.float(), dim=-1)
    gamma = q_draft_tokens.numel()
    out: list[int] = []

    for i in range(gamma):
        x = int(q_draft_tokens[i])
        p_x, q_x = float(p[i, x]), float(q[i, x])
        accept_prob = min(1.0, p_x / q_x) if q_x > 0 else 1.0
        if torch.rand(1, generator=generator).item() < accept_prob:
            out.append(x)
        else:
            # 拒绝：从残差分布 norm(max(p - q, 0)) 重采样
            residual = torch.clamp(p[i] - q[i], min=0.0)
            if residual.sum() == 0:
                residual = p[i]
            residual = residual / residual.sum()
            out.append(int(torch.multinomial(residual, 1, generator=generator).item()))
            return torch.tensor(out, dtype=torch.long)

    # 全部接受 → bonus token 从最后一个位置的 target 分布采
    out.append(int(torch.multinomial(p[gamma], 1, generator=generator).item()))
    return torch.tensor(out, dtype=torch.long)


def greedy_speculative_step(target_logits: torch.Tensor,
                           q_draft_tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
    """贪心解码下的投机采样（面试常考的直觉版）。

    draft 与 target 的 argmax 一致 → 接受；不一致 → 停在该位置并采纳 target 的 argmax。
    Returns: (产出 token 序列, 被拒绝位置；-1 表示全部接受)
    """
    p_arg = target_logits.argmax(dim=-1)
    for i, x in enumerate(q_draft_tokens.tolist()):
        if int(p_arg[i]) != int(x):
            return torch.tensor(q_draft_tokens[:i].tolist() + [int(p_arg[i])], dtype=torch.long), i
    # 全部接受，bonus = 最后一个位置的 argmax
    return torch.tensor(q_draft_tokens.tolist() + [int(p_arg[-1])], dtype=torch.long), -1


def expected_accepted_tokens(p: torch.Tensor, q: torch.Tensor, gamma: int) -> float:
    """闭式期望接受数 E[#accepted] = (1 - α^(gamma+1)) / (1 - α)，α = Σ_x min(p_x, q_x)。

    用于回答"投机采样什么时候划算"：α 越接近 1（draft 越准）接受越多。
    """
    alpha = torch.clamp(torch.minimum(p, q), min=0).sum().item()
    if alpha >= 1.0 - 1e-12:
        return float(gamma)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def acceptance_rate(target_logits: torch.Tensor, draft_logits: torch.Tensor,
                    q_draft_tokens: torch.Tensor) -> float:
    """贪心口径的草稿命中率（vLLM/SGLang 的 metrics 里的 accepted_tokens/... 的分子）。"""
    p_arg = target_logits.argmax(dim=-1)
    hit = sum(int(p_arg[i]) == int(x) for i, x in enumerate(q_draft_tokens.tolist()))
    return hit / q_draft_tokens.numel()


def distribution_equivalence_check(n_runs: int = 4000, vocab: int = 8,
                                   gamma: int = 3, seed: int = 0) -> float:
    """统计验证投机采样输出分布 == target 分布（total variation 距离应随 n_runs 收敛到 0）。"""
    g = torch.Generator().manual_seed(seed)
    target_logits = torch.randn(gamma + 1, vocab, generator=g)
    draft_logits = torch.randn(gamma, vocab, generator=g)
    p0 = torch.softmax(target_logits[0].float(), dim=-1)  # 只统计第一个产出 token 的分布

    counts = torch.zeros(vocab, dtype=torch.float64)
    for _ in range(n_runs):
        q_tok = torch.tensor([torch.multinomial(
            torch.softmax(draft_logits[i].float(), -1), 1, generator=g).item()
            for i in range(gamma)])
        out = speculative_step(target_logits, draft_logits, q_tok, g)
        counts[int(out[0])] += 1
    emp = counts / n_runs
    return 0.5 * float((emp - p0.double()).abs().sum())  # TV distance
