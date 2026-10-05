"""区间调度 / 活动选择(贪心)。

对应面试题:
    - 阶跃星辰-实习 编程 7: 电视节目起止时间, 最多完整观看多少个节目

结论: 按结束时间升序排序, 每次选"结束最早且开始时间不与已选冲突"的节目 ——
结束越早, 给后面留的余地越大(交换论证可证最优)。O(n log n)。
与 dp.py::min_rooms_meetings(最多重叠区间数)是同一族题的两面。
"""
from __future__ import annotations

__all__ = ["max_complete_activities"]


def max_complete_activities(shows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """返回最多能完整观看的节目列表(保持选择顺序)。

    shows: (start, end) 列表; 区间 [start, end) 语义, start == 上一结束 不算冲突。
    """
    chosen: list[tuple[int, int]] = []
    last_end = float("-inf")
    for start, end in sorted(shows, key=lambda s: s[1]):
        if start >= last_end:
            chosen.append((start, end))
            last_end = end
    return chosen
