"""MoE 路由分析与负载均衡：专家分工度量、aux loss、路由坍缩检测。

对应题目：字节-AI-Infra-实习-一面-1 推理优化 6-8（MoE 推理优化 / 如何判断学到了
专家分工 / RL 场景 reward 导致 routing 退化的处理）。

核心公式：

- **Switch/DeepSeek 负载均衡损失**：``L_aux = N_e · Σ_e f_e · P_e``，
  其中 ``f_e`` = 分到专家 e 的 token 占比（按 top-k 计数），
  ``P_e`` = 全体 token 对专家 e 的平均 router 概率，``N_e`` = 专家数。
  完全均匀（f=P=1/N_e）时取最小值 **1**——这也是测试里的解析校验点。
- **z-loss**（稳定 router logit）：``L_z = mean(logsumexp(logits)²)``，
  防止 logits 爆炸导致 softmax 饱和、梯度消失。
- **分工是否学到的判据**（Q7）：路由熵介于 ln(N_e)（完全随机）与 0（坍缩）之间，
  但"低熵 + 各专家负载均匀 + 按领域聚类的专家激活"才是真分工；
  只看熵低可能只是坍缩到少数专家。

运行验证::

    cd solutions/python && python -m pytest tests/test_bytedance.py -q
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def routing_stats(
    router_probs: torch.Tensor, topk_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (f, P)：每个专家的 token 分配占比与平均 router 概率。

    router_probs: (T, E) 归一化概率；topk_ids: (T, top_k) 每条 token 选中的专家 id。
    """
    num_experts = router_probs.shape[-1]
    f = torch.bincount(topk_ids.reshape(-1), minlength=num_experts).float()
    f = f / max(f.sum(), 1.0)
    p = router_probs.mean(dim=0)
    return f, p


def load_balance_loss(
    router_probs: torch.Tensor, topk_ids: torch.Tensor
) -> torch.Tensor:
    """Switch 风格 aux loss：E · Σ f_e·P_e。均匀负载时 = 1，越失衡越大（上限 E）。"""
    f, p = routing_stats(router_probs, topk_ids)
    return router_probs.shape[-1] * torch.sum(f * p)


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """z-loss = mean(logsumexp(logits)²)，可导，惩罚 router logits 过大。"""
    return torch.mean(torch.logsumexp(logits, dim=-1) ** 2)


def routing_entropy(router_probs: torch.Tensor) -> float:
    """平均 token 路由熵（nats）。uniform 时 = ln(E)，one-hot 时 = 0。"""
    ent = -(router_probs * router_probs.clamp_min(1e-12).log()).sum(dim=-1)
    return ent.mean().item()


def expert_utilization(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """每个专家承接的 token 占比（含共享前缀的可视化口径）。"""
    counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts).float()
    return counts / counts.sum()


def diagnose_routing(
    router_probs: torch.Tensor,
    topk_ids: torch.Tensor,
    collapse_share: float = 0.5,
    entropy_ratio: float = 0.3,
) -> dict[str, float | bool]:
    """一次给出 Q7/Q8 需要的体检指标。

    - ``collapsed``：top-1 集中度超过 collapse_share 的 expert 数 >= 1 且
      平均路由熵显著低于均匀（< entropy_ratio·ln(E)) —— 路由坍缩（退化为 dense 子集）。
    - ``balanced``：负载均衡损失接近下界 1 —— 分工良好（还需人工看专家激活是否语义聚类）。
    """
    num_experts = router_probs.shape[-1]
    f = expert_utilization(topk_ids, num_experts)
    ent = routing_entropy(router_probs)
    uniform_ent = float(torch.log(torch.tensor(float(num_experts))))
    aux = load_balance_loss(router_probs, topk_ids).item()
    collapsed = bool((f > collapse_share).any() and ent < entropy_ratio * uniform_ent)
    return {
        "expert_load": f,
        "routing_entropy": ent,
        "uniform_entropy": uniform_ent,
        "aux_loss": aux,
        "collapsed": collapsed,
        "balanced": aux < 1.0 + 0.05 * num_experts,
    }


def topk_route(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """标准 top-k 路由：softmax 后取 top-k，概率在选中专家内重归一化（DeepSeek 风格）。"""
    probs = F.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs, top_k, dim=-1)
    gates = topv / topv.sum(dim=-1, keepdim=True)
    return gates, topi
