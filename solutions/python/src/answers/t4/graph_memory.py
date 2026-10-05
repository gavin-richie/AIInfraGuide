"""计算图运行所需最小内存（元戎启行：给定计算图求运行时峰值内存）。

对应面试题:
    - 元戎启行-校招-一面-1 编程题 8: "给定一个计算图，计算运行该计算图所需的最小内存"
      （与 MLIR Bufferization 的 in-place / liveness 分析同源，见 docs/interview-answers/90）

模型:
    计算图是 DAG，节点 = 算子，每个算子产出一张中间 tensor（大小 out_bytes），
    边 = 数据依赖。运行按某个拓扑序逐算子执行：
      1) 分配本算子输出 tensor（+out_bytes）
      2) 读入全部输入
      3) 某输入 tensor 若已无后续消费者 → 释放（-size）

    "最小内存"的两个层次:
    a) 独立分配 + 及时释放：峰值 = 任意时刻活跃 tensor 字节和的最大值
       （liveness 分析，下界；执行顺序会影响它，可用贪心/全排列对拍）；
    b) 内存池复用（arena / bufferization）：活跃区间不重叠的 tensor 共享同一块内存，
       峰值进一步逼近"最大重叠字节数"；区间大小不一时是 bin packing（NP-hard），
       工程上用 first-fit。

API:
    Op(name, inputs, out_bytes)      -- 算子：名字 / 输入 tensor 名 / 输出字节数
    build_graph(ops, weights)        -- 建图：消费者、依赖边
    peak_memory(order, graph)        -- 给定执行顺序的峰值（及时释放）
    peak_no_free(graph)              -- 从不释放的总占用（对照）
    min_peak_greedy(graph)           -- 贪心调度：优先执行"净释放最多"的就绪算子
    min_peak_bruteforce(graph)       -- 小图全排列拓扑序取最优（对拍用）
    assign_arena_offsets(order, g)   -- first-fit 内存池偏移分配（区间不重叠可复用）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations


@dataclass
class Op:
    name: str
    inputs: list[str]
    out_bytes: int


@dataclass
class TensorInfo:
    producer: str
    size: int
    consumers: list[str] = field(default_factory=list)


@dataclass
class Graph:
    ops: dict[str, Op]
    tensors: dict[str, TensorInfo]   # 中间 tensor + 外部输入/权重
    # 想要执行顺序：用 min_peak_greedy(g) 的第二个返回值


def build_graph(ops: list[Op], weights: dict[str, int] | None = None) -> Graph:
    """ops 顺序任意；weights 是常量输入（权重）的字节数，它们全程存活。"""
    op_map = {op.name: op for op in ops}
    tensors: dict[str, TensorInfo] = {
        w: TensorInfo(producer="<weight>", size=sz) for w, sz in (weights or {}).items()
    }
    for op in ops:
        tensors[op.name] = TensorInfo(producer=op.name, size=op.out_bytes)
        for inp in op.inputs:
            info = tensors.setdefault(inp, TensorInfo(producer="<input>", size=0))
            info.consumers.append(op.name)
    return Graph(ops=op_map, tensors=tensors)


def _annotate_last_use(order: list[str], g: Graph) -> None:
    """每个 tensor 的 last_use = 最后一个消费者的执行步（无消费者 = 永不释放）。"""
    for t in g.tensors.values():
        t.last_use = -1  # type: ignore[attr-defined]
    for step, name in enumerate(order):
        for inp in g.ops[name].inputs:
            g.tensors[inp].last_use = step  # type: ignore[attr-defined]
    never = len(order)  # 不会到达的哨兵值 = 永不释放
    for t in g.tensors.values():
        if t.last_use == -1:  # type: ignore[attr-defined]
            t.last_use = never  # type: ignore[attr-defined]


def peak_memory(order: list[str], g: Graph) -> int:
    """给定拓扑序：产出即分配、最后消费者之后释放，返回中间 tensor 的峰值占用。"""
    assert sorted(order) == sorted(g.ops), "order 必须是全部算子的一个排列"
    _annotate_last_use(order, g)
    cur = peak = 0
    for step, name in enumerate(order):
        cur += g.ops[name].out_bytes
        peak = max(peak, cur)
        for inp in g.ops[name].inputs:
            if g.tensors[inp].last_use == step:  # type: ignore[attr-defined]
                cur -= g.tensors[inp].size
    return peak


def peak_no_free(g: Graph) -> int:
    """对照：从不释放（每张中间 tensor 独立内存）的总占用。"""
    return sum(op.out_bytes for op in g.ops.values())


def _is_topological(order: tuple[str, ...] | list[str], g: Graph) -> bool:
    seen: set[str] = set()
    for name in order:
        for inp in g.ops[name].inputs:
            if inp in g.ops and inp not in seen:
                return False
        seen.add(name)
    return True


def min_peak_bruteforce(g: Graph) -> tuple[int, list[str]]:
    """小图（≤8 个算子）对拍：枚举全部拓扑序取峰值最小者。"""
    assert len(g.ops) <= 8, "全排列只用于小图对拍"
    best, best_order = None, None
    for order in permutations(g.ops):
        if _is_topological(order, g):
            p = peak_memory(list(order), g)
            if best is None or p < best:
                best, best_order = p, list(order)
    assert best is not None
    return best, best_order  # type: ignore[return-value]


def min_peak_greedy(g: Graph) -> tuple[int, list[str]]:
    """贪心调度：每步在就绪算子里选（即将释放的输入字节 - 输出字节）最大者。

    直觉：尽快让大 tensor 死掉，压平活跃区间的重叠。不保证最优，
    但对链式/钻石等常见图与最优一致（测试里与全排列对拍）。
    """
    remaining = {t: len(info.consumers) for t, info in g.tensors.items()}
    executed: set[str] = set()
    order: list[str] = []
    cur = peak = 0
    while len(order) < len(g.ops):
        ready = [n for n in g.ops
                 if n not in executed
                 and all(i not in g.ops or i in executed for i in g.ops[n].inputs)]
        assert ready, "图中存在环，不是合法计算图"

        def net_free(n: str) -> int:
            freed = sum(g.tensors[i].size for i in g.ops[n].inputs if remaining[i] == 1)
            return freed - g.ops[n].out_bytes

        pick = max(sorted(ready), key=net_free)  # sorted 保证平局时按名字稳定
        cur += g.ops[pick].out_bytes
        peak = max(peak, cur)
        for inp in g.ops[pick].inputs:
            remaining[inp] -= 1
            if remaining[inp] == 0:
                cur -= g.tensors[inp].size
        executed.add(pick)
        order.append(pick)
    return peak, order


def assign_arena_offsets(order: list[str], g: Graph) -> tuple[int, dict[str, int]]:
    """内存池 first-fit：活跃区间不重叠的中间 tensor 复用同一块内存。

    返回 (arena 总大小, tensor -> 偏移)。等大小时 left-edge 算法最优；
    大小不一是 bin packing（NP-hard），first-fit 是工程常规解。
    """
    _annotate_last_use(order, g)
    # (产生步, 释放步, 字节, 名字)；永不释放的按 [start, +inf) 处理
    big = len(order) * 2 + 1
    intervals = [
        (step, max(g.tensors[name].last_use, step), g.tensors[name].size, name)  # type: ignore[attr-defined]
        for step, name in enumerate(order)
    ]
    intervals.sort(key=lambda x: (x[0], -x[2]))
    placed: list[tuple[int, int, int, int]] = []  # (start, end, offset, size)
    offsets: dict[str, int] = {}
    arena = 0
    for start, end, size, name in intervals:
        if end >= big:  # 永不释放：不能与任何区间复用
            offsets[name] = arena
            arena += size
            placed.append((start, big, offsets[name], size))
            continue
        # 候选偏移：0 与每个已放区间的右端
        candidates = sorted({0} | {off + sz for _, _, off, sz in placed})
        chosen = None
        for cand in candidates:
            ok = True
            for ps, pe, poff, psz in placed:
                time_overlap = ps < end and start < pe
                addr_overlap = poff < cand + size and cand < poff + psz
                if time_overlap and addr_overlap:
                    ok = False
                    break
            if ok:
                chosen = cand
                break
        assert chosen is not None
        offsets[name] = chosen
        arena = max(arena, chosen + size)
        placed.append((start, end, chosen, size))
    return arena, offsets
