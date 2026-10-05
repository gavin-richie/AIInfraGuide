"""滑动窗口中位数: 大根堆+小根堆双堆 + 延迟删除, O(n log k)。

来源: 腾讯-AI-Infra-实习-一二面(编程 13, 面试要求双堆实现)。LeetCode 480。

设计(见 docs/interview-answers/42-腾讯-编程题.md Q4):
- low 大根堆(Python 存负数)存较小一半, high 小根堆存较大一半;
  不变式 low_size == high_size 或 low_size == high_size + 1。
- 堆不支持任意位置删除 -> 被滑出窗口的元素记入 delayed 计数(逻辑删除),
  等它浮到堆顶时再真正弹出(prune); 平衡判断用有效大小 low_size/high_size。
- 正确性: 堆顶(含脏元素) >= 本堆所有有效元素, 且任一脏元素 <= 对面堆全部
  有效元素, 因此拿堆顶做路由/删除分类是安全的。
"""
from __future__ import annotations

import heapq
from collections import Counter


class DualHeap:
    """双堆维护一组数的中位数(支持带延迟标记的删除)。"""

    def __init__(self) -> None:
        self.low: list[int] = []    # 大根堆(存负数): 较小的一半
        self.high: list[int] = []   # 小根堆: 较大的一半
        self.delayed: Counter[int] = Counter()  # 待删除元素计数
        self.low_size = 0           # low 中有效元素个数
        self.high_size = 0

    def _prune_low(self) -> None:
        while self.low and self.delayed[-self.low[0]]:
            self.delayed[-self.low[0]] -= 1
            heapq.heappop(self.low)

    def _prune_high(self) -> None:
        while self.high and self.delayed[self.high[0]]:
            self.delayed[self.high[0]] -= 1
            heapq.heappop(self.high)

    def _balance(self) -> None:
        # 只在需要搬移堆顶时 prune(搬出的一定是有效堆顶)
        if self.low_size > self.high_size + 1:
            self._prune_low()
            heapq.heappush(self.high, -heapq.heappop(self.low))
            self.low_size -= 1
            self.high_size += 1
        elif self.low_size < self.high_size:
            self._prune_high()
            heapq.heappush(self.low, -heapq.heappop(self.high))
            self.high_size -= 1
            self.low_size += 1

    def insert(self, num: int) -> None:
        if self.low and num > -self.low[0]:
            heapq.heappush(self.high, num)
            self.high_size += 1
        else:
            heapq.heappush(self.low, -num)
            self.low_size += 1
        self._balance()

    def erase(self, num: int) -> None:
        """删除一个已在窗口中的元素(延迟删除)。"""
        self.delayed[num] += 1
        if num <= -self.low[0]:
            self.low_size -= 1
            self._prune_low()
        else:
            self.high_size -= 1
            self._prune_high()
        self._balance()

    def median(self) -> float:
        assert self.low_size + self.high_size > 0
        self._prune_low()
        self._prune_high()
        if (self.low_size + self.high_size) % 2:
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2


def sliding_window_median_dual_heap(nums: list[int], k: int) -> list[float]:
    """双堆解法 O(n log k)。朴素 O(n*klogk) 版见 algorithms/topk_heap.py。"""
    if k <= 0 or k > len(nums):
        raise ValueError("k 需满足 1 <= k <= len(nums)")
    dh = DualHeap()
    out: list[float] = []
    for i, x in enumerate(nums):
        if i >= k:
            dh.erase(nums[i - k])
        dh.insert(x)
        if i >= k - 1:
            out.append(dh.median())
    return out
