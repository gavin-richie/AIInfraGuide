"""EPLB（Expert Parallelism Load Balancer）冗余专家负载均衡模拟。

对应面试题:
    - 小厂-实习-1 "介绍 DeepSeek V3 中 EPLB 的推理机制"

背景: MoE 推理（EP 部署）下专家负载天然不均衡（DeepSeek-V3 论文观察
不同 expert 的激活频率差异可达数倍），热专家所在 rank 成为短板。
EPLB（DeepSeek 开源的专家编排工具）的做法:
    1. 离线统计各 expert 的历史负载（如 2 小时窗口的 batch 平均负载）
    2. 给高负载专家冗余拷贝（copy count ∝ 负载），训练时同步冗余参数
    3. 把全部"物理专家"（含冗余）重排到各 GPU/rank，最小化最大 rank 负载
    4. 周期性（如每天）在低峰期 rebalance，路由表 (logical→physical) 同步更新

本模块实现核心第 3 步的贪心近似: 反复把"当前最重 expert 的一个拷贝"
放到"当前最轻的 rank"。真实 EPLB 用带冗余层数约束的均衡 min-cost 问题求解器。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import heapq


@dataclass
class EPLBPlan:
    n_rank: int
    expert_to_ranks: dict[int, list[int]] = field(default_factory=dict)
    rank_load: list[float] = field(default_factory=list)

    @property
    def imbalance(self) -> float:
        """最大 rank 负载 / 平均负载，越接近 1 越均衡。"""
        return max(self.rank_load) / max(sum(self.rank_load) / self.n_rank, 1e-12)


def eplb_balance(loads: list[float], n_rank: int, redundancy_factor: float = 1.5,
                 max_copies: int | None = None) -> EPLBPlan:
    """把逻辑专家（含冗余拷贝）均衡编排到 n_rank 个 rank 上。

    Args:
        loads: 每个 expert 的平均负载（如激活 token 数占比）
        n_rank: EP 的 rank（GPU）数
        redundancy_factor: 总物理专家数 = ceil(n_expert × redundancy_factor)，
                           多出的份额按负载分给热专家
        max_copies: 单个 expert 最多拷贝数（防止过度冗余）
    """
    n_expert = len(loads)
    total_phys = int(n_expert * redundancy_factor + 0.5)
    extra = max(total_phys - n_expert, 0)
    if max_copies is None:
        max_copies = max(2, n_rank)

    # 1) 冗余份额按负载比例分配给热专家
    copies = [1] * n_expert
    heap = [(-w, i) for i, w in enumerate(loads)]  # max-heap by load
    heapq.heapify(heap)
    for _ in range(extra):
        _, i = heapq.heappop(heap)
        if copies[i] >= max_copies:      # 该专家拷贝数已达上限，跳过
            continue
        copies[i] += 1
        heapq.heappush(heap, (-(loads[i] / copies[i]), i))  # 按"每拷贝负载"再排

    # 2) 贪心放置: 最重拷贝 → 最轻 rank
    phys = sorted(((loads[i] / copies[i], i) for i in range(n_expert) for _ in range(copies[i])),
                  reverse=True)
    rank_load = [0.0] * n_rank
    min_heap = [(0.0, r) for r in range(n_rank)]  # (load, rank)
    heapq.heapify(min_heap)
    expert_to_ranks: dict[int, list[int]] = {i: [] for i in range(n_expert)}
    for w, i in phys:
        cur, r = heapq.heappop(min_heap)
        expert_to_ranks[i].append(r)
        rank_load[r] += w
        heapq.heappush(min_heap, (rank_load[r], r))

    return EPLBPlan(n_rank=n_rank, expert_to_ranks=expert_to_ranks, rank_load=rank_load)


def without_eplb(loads: list[float], n_rank: int) -> EPLBPlan:
    """基线: 无冗余、round-robin/连续切分放置（EP 默认 consecutive 分组）。"""
    n = len(loads)
    per = (n + n_rank - 1) // n_rank
    rank_load = [sum(loads[r * per:(r + 1) * per]) for r in range(n_rank)]
    return EPLBPlan(n_rank=n_rank, rank_load=rank_load)


if __name__ == "__main__":
    # 模拟 DeepSeek-V3: 256 experts、8 ranks、负载长尾（zipf-ish）
    import random
    random.seed(0)
    loads = [random.paretovariate(1.5) for _ in range(256)]
    base = without_eplb(loads, 8)
    plan = eplb_balance(loads, n_rank=8, redundancy_factor=1.5)
    print(f"无 EPLB   : 最大 rank 负载/均值 = {base.imbalance:.2f}")
    print(f"EPLB 1.5x : 最大 rank 负载/均值 = {plan.imbalance:.2f}")
    top = sorted(range(256), key=lambda i: -loads[i])[:3]
    print(f"最热 3 个 expert 的拷贝数: {[len(plan.expert_to_ranks[i]) for i in top]}")
