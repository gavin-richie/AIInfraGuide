"""流水线并行调度模拟器：GPipe / 1F1B / ZB-H1 / ZB-H2 的气泡与显存对比。

来源: 美团-AI-Infra-实习 Q1（Zero-Bubble 流水线调度的原理）、快手-校招-2 Q5（PP 显存峰值）。

模型（与 Zero Bubble Pipeline Parallelism, arXiv:2401.10241 对齐）：
- 反向拆两段：B（输入梯度，必须回传上一个 stage，是跨卡关键依赖）与
  W（权重梯度，只有 optimizer step 依赖，可任意延后）。
- 依赖：F(i,j) ← F(i,j-1)；B(i,j) ← F(i,j) + B(i,j+1)；
  W(i,j) ← F(i,j) + B(i,j+1)（与 B 同源，但没有下游消费者）。
- 调度（两级实现，模拟论文的"延后 W"机制）：
  1) phase-1 只排 F/B 的贪心表调度（F 优先保持前向供给，到达在途上限后做 B）：
     * gpipe : 无在途上限（全部前向做完再做反向，激活驻留 = m）；
     * 1f1b  : 在途上限 p-j，W 紧跟 B（等价 B/W 合并的传统 1F1B）；
     * zb_h1 : 在途上限与 1F1B 相同（论文：同显存画像），只是 W 全部延后；
     * zb_h2 : 上限放大到 2(p-j)-1（warmup 期多跑 F 把 B 链整体提前）；
  2) phase-2（仅 zb_*）把 W 逐个塞进 stage 时间线的空隙（first-fit，不延误任何
     已排算子），塞不下的排到尾部（冷排空）——这正是"用 W 填气泡 + 尾部 drain"。

本模拟器复现的核心结论（pytest 断言，f=b=w=1 时）：
- 1F1B 气泡（首卡 idle）== (p-1)(f+b)，b 为输入梯度段耗时；B/W 合并的工程实现
  对应 b←b+w，即经典 Megatron 公式 (p-1)(f+b_total)；
- ZB-H1 气泡 == (p-1)(f+b-w)：f=b=w 时恰为 1F1B 的 1/3（论文原话）；
- ZB-H2 所有 stage idle == 0（论文"零气泡"：trapezoid 变 parallelogram），
  总时长 == (m+p-1)f + m(b+w)，代价是激活要等延后的 W 才释放（首卡 ~2p-1）。
"""

from dataclasses import dataclass, field

Op = tuple[str, int, int]  # (kind "F"/"B"/"W", micro i, stage j)


@dataclass
class SimResult:
    policy: str
    p: int
    m: int
    f: float
    b: float
    w: float
    total_time: float
    start: dict[Op, float]
    end: dict[Op, float]
    stage_seq: dict[int, list[Op]]

    @property
    def busy_time(self) -> float:
        """每个 stage 的满负荷时长 = m·(f+b+w)。"""
        return self.m * (self.f + self.b + self.w)

    def stage_span(self, j: int) -> tuple[float, float]:
        times = [self.start[op] for op in self.stage_seq[j]] + [
            self.end[op] for op in self.stage_seq[j]
        ]
        return min(times), max(times)

    def stage_idle(self, j: int) -> float:
        """stage 首末算子之间的空闲（真正的"气泡"口径）。"""
        first, last = self.stage_span(j)
        return last - first - self.busy_time

    @property
    def max_stage_idle(self) -> float:
        return max(self.stage_idle(j) for j in range(self.p))

    @property
    def bubble_ratio(self) -> float:
        """makespan 效率口径：(总时长 - 满负荷) / 总时长。"""
        return (self.total_time - self.busy_time) / self.total_time

    @property
    def peak_activations(self) -> list[int]:
        """每个 stage 峰值驻留微批数（F 完成 +1，W 完成 -1 的最大前缀和）。

        ZB 延后 W → 激活释放更晚，这是它换零气泡的显存代价。
        """
        peaks = []
        for j in range(self.p):
            cur = peak = 0
            for kind, _, _ in self.stage_seq[j]:
                if kind == "F":
                    cur += 1
                    peak = max(peak, cur)
                elif kind == "W":
                    cur -= 1
            peaks.append(peak)
        return peaks


class _Sim:
    def __init__(self, p: int, m: int, f: float, b: float, w: float, policy: str) -> None:
        self.p, self.m, self.f, self.b, self.w, self.policy = p, m, f, b, w, policy
        self.cost = {"F": f, "B": b, "W": w}
        self.start: dict[Op, float] = {}
        self.end: dict[Op, float] = {}
        self.stage_time = [0.0] * p
        self.stage_seq: dict[int, list[Op]] = {j: [] for j in range(p)}
        self.f_started = [0] * p
        self.b_done = [0] * p
        self.f_done = [0] * p

    # ---- 依赖 ----
    def deps(self, op: Op) -> list[Op]:
        kind, i, j = op
        out: list[Op] = []
        if kind == "F":
            if j > 0:
                out.append(("F", i, j - 1))
        else:  # B 与 W 同源：需本卡 F 完成 + 下游梯度
            out.append(("F", i, j))
            if j < self.p - 1:
                out.append(("B", i, j + 1))
        return out

    def ready(self, op: Op) -> bool:
        return all(d in self.end for d in self.deps(op))

    def dep_end(self, op: Op) -> float:
        times = [self.end[d] for d in self.deps(op)]
        return max(times) if times else 0.0

    def schedule(self, op: Op) -> None:
        st = max(self.stage_time[op[2]], self.dep_end(op))
        self.start[op] = st
        self.end[op] = st + self.cost[op[0]]
        self.stage_time[op[2]] = self.end[op]
        self.stage_seq[op[2]].append(op)
        if op[0] == "F":
            self.f_started[op[2]] = op[1] + 1
            self.f_done[op[2]] = op[1] + 1
        elif op[0] == "B":
            self.b_done[op[2]] = op[1] + 1

    # ---- phase-1 策略：返回 stage j 此刻应执行的 F/B（或 None 等待）----
    def choose(self, j: int) -> Op | None:
        i_f = self.f_started[j]
        ready_b = min(
            (i for i in range(self.b_done[j], self.m) if self.ready(("B", i, j))), default=None
        )
        can_f = i_f < self.m and self.ready(("F", i_f, j))
        # 在途微批上限（激活显存约束的代理）
        if self.policy == "gpipe":
            cap = self.m
        elif self.policy == "zb_h2":
            cap = max(1, 2 * (self.p - j) - 1)
        else:  # 1f1b 与 zb_h1 同上限（论文：ZB-H1 与 1F1B 同显存画像）
            cap = self.p - j
        inflight = self.f_done[j] - self.b_done[j]

        if self.policy == "gpipe":
            if can_f:
                return ("F", i_f, j)
            if ready_b is not None:
                return ("B", ready_b, j)
            return self._drain_w(j)

        # 1f1b / zb_h1 / zb_h2 共用 1F1B 的 F/B 结构：F 优先保持前向供给，
        # 到达在途上限后做 B。差异只在 W 的位置与上限大小：
        # 1f1b   : W 紧跟 B（合并反向）；
        # zb_h1  : W 延后（phase-2 填隙），上限同 1F1B；
        # zb_h2  : W 延后 + 上限放大到 ~2(p-j)，warmup 期多跑 F 把 B 链提前。
        if self.policy == "1f1b":
            last = self.stage_seq[j][-1] if self.stage_seq[j] else None
            if last is not None and last[0] == "B" and ("W", last[1], j) not in self.end:
                return ("W", last[1], j)
        if can_f and inflight < cap:
            return ("F", i_f, j)
        if ready_b is not None:
            return ("B", ready_b, j)
        if self.policy == "1f1b":
            return self._drain_w(j)
        return None  # zb_*：W 留给 phase-2 填隙

    def _drain_w(self, j: int) -> Op | None:
        for i in range(self.m):
            op = ("W", i, j)
            if op not in self.end and self.ready(op):
                return op
        return None

    # ---- phase-2：把延后的 W 塞进空隙 / 尾部排空 ----
    def _fill_w_into_gaps(self, j: int) -> None:
        seq = self.stage_seq[j]
        # 收集空隙 [prev_end, next_start]
        gaps: list[list[float]] = []
        for prev, nxt in zip(seq, seq[1:]):
            if self.end[prev] < self.start[nxt]:
                gaps.append([self.end[prev], self.start[nxt]])
        tail = self.stage_time[j]
        for i in range(self.m):
            op = ("W", i, j)
            if op in self.end:
                continue
            # first-fit：找第一个塞得下的空隙（W 依赖 B(i,j+1)，需检查就绪时间）
            for gap in gaps:
                earliest = max(gap[0], self.dep_end(op))
                if earliest + self.w <= gap[1]:
                    self.start[op] = earliest
                    self.end[op] = earliest + self.w
                    gap[0] = self.end[op]
                    break
            else:
                st = max(tail, self.dep_end(op))
                self.start[op] = st
                self.end[op] = st + self.w
                tail = self.end[op]
        # 重排 stage_seq 为时间序（W 插进了中间）
        self.stage_seq[j] = sorted(seq + [("W", i, j) for i in range(self.m)], key=lambda o: self.start[o])

    def run(self) -> SimResult:
        # phase-1：gpipe/1f1b 连 W 一起排完；zb_* 只排 F/B
        while True:
            progress = False
            for j in range(self.p):
                while True:
                    op = self.choose(j)
                    if op is None:
                        break
                    self.schedule(op)
                    progress = True
            if not progress:
                break
        # phase-2：zb 的 W 填隙 + 冷排空
        if self.policy in ("zb_h1", "zb_h2"):
            for j in range(self.p):
                self._fill_w_into_gaps(j)
        total = len(self.end)
        assert total == 3 * self.m * self.p, f"算子数不完整: {total}"
        return SimResult(
            policy=self.policy,
            p=self.p,
            m=self.m,
            f=self.f,
            b=self.b,
            w=self.w,
            total_time=max(self.end.values()),
            start=self.start,
            end=self.end,
            stage_seq=self.stage_seq,
        )


def simulate(p: int, m: int, f: float = 1.0, b: float = 1.0, w: float = 1.0, policy: str = "1f1b") -> SimResult:
    if m < p:
        raise ValueError("模拟假设 m >= p（微批数不少于 stage 数）")
    if policy not in {"gpipe", "1f1b", "zb_h1", "zb_h2"}:
        raise ValueError(f"未知策略: {policy}")
    return _Sim(p, m, f, b, w, policy).run()


def verify_dependencies(res: SimResult) -> None:
    """合法性自检：每个 op 晚于全部依赖结束；同一 stage 上的算子串行不重叠。"""
    ref = _Sim(res.p, res.m, res.f, res.b, res.w, res.policy)
    for op in res.end:
        for d in ref.deps(op):
            assert res.end[d] <= res.start[op] + 1e-9, f"依赖违例: {d} -> {op}"
    for j, seq in res.stage_seq.items():
        for prev, cur in zip(seq, seq[1:]):
            assert res.end[prev] <= res.start[cur] + 1e-9, f"stage {j} 重叠: {prev} / {cur}"


def compare_schedules(p: int, m: int, f: float = 1.0, b: float = 1.0, w: float = 1.0) -> list[SimResult]:
    return [simulate(p, m, f, b, w, pol) for pol in ("gpipe", "1f1b", "zb_h1", "zb_h2")]


def summary_table(p: int = 8, m: int = 32) -> str:
    """打印对比表（文档/教学用）。"""
    lines = [f"p={p} stages, m={m} microbatches, f=1, b=1, w=1（反向拆 B/W 各 1）"]
    lines.append(f"{'policy':8s} {'total':>6s} {'max_idle':>8s} {'makespan%':>9s} {'peak_act':>8s}")
    for r in compare_schedules(p, m):
        lines.append(
            f"{r.policy:8s} {r.total_time:6.0f} {r.max_stage_idle:8.0f} "
            f"{100 * (1 - r.bubble_ratio):8.1f}% {max(r.peak_activations):8d}"
        )
    return "\n".join(lines)
