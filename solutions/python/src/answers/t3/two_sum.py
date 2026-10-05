"""两数之和（哈希一遍法）。

来源: 沐曦-AI-Infra-实习-一面（编程 9）。LeetCode 1。
要点: 边遍历边查"target - x 是否出现过", 一遍 O(n) 时间 / O(n) 空间;
对比暴力 O(n²) 与排序+双指针 O(n log n)（会打乱下标，需带下标排序）。
"""
from __future__ import annotations


def two_sum(nums: list[int], target: int) -> tuple[int, int]:
    """返回和为 target 的两个下标 (i, j)，保证 i < j；无解抛 ValueError。

    >>> two_sum([2, 7, 11, 15], 9)
    (0, 1)
    """
    seen: dict[int, int] = {}           # 值 -> 下标
    for j, x in enumerate(nums):
        if (i := seen.get(target - x)) is not None:
            return (i, j)
        seen[x] = j
    raise ValueError("no solution")
