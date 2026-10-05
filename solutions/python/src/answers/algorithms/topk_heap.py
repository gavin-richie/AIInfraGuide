"""堆 / TopK 类高频题。

来源: 快手-一面-2（每行有序矩阵第 K 小）、MiniMax-实习（K 个有序数组中位数、滑动窗口
最大值）、网易-校招（前 K 高频）、腾讯-实习-一二面（滑动窗口中位数双堆）、壁仞-一面
（TopK 堆）、阿里-一面-2（第 K 大）、小鹏（uint8 第 K 大）等。
"""
from __future__ import annotations

import heapq
from collections import Counter, deque


def kth_largest(nums: list[int], k: int) -> int:
    """LeetCode 215: 维护大小为 k 的小根堆, O(n log k)。

    面试要点: 求第 K 大用小根堆(堆顶是 k 个数中的最小), 求第 K 小用大根堆;
    也可用快速选择 quickselect 平均 O(n)。
    来源: 阿里-一面-2、壁仞-一面。
    """
    heap: list[int] = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    if len(heap) < k:
        raise ValueError("k is larger than len(nums)")
    return heap[0]


def quickselect_kth_largest(nums: list[int], k: int) -> int:
    """快速选择版第 K 大, 平均 O(n)。"""
    import random

    def partition(lo: int, hi: int, pivot_idx: int) -> int:
        pivot = nums[pivot_idx]
        nums[pivot_idx], nums[hi] = nums[hi], nums[pivot_idx]
        store = lo
        for i in range(lo, hi):
            if nums[i] < pivot:
                nums[i], nums[store] = nums[store], nums[i]
                store += 1
        nums[store], nums[hi] = nums[hi], nums[store]
        return store

    target = len(nums) - k  # 升序后第 target 位即第 k 大
    lo, hi = 0, len(nums) - 1
    while True:
        if lo == hi:
            return nums[lo]
        p = partition(lo, hi, random.randint(lo, hi))
        if p == target:
            return nums[p]
        if p < target:
            lo = p + 1
        else:
            hi = p - 1


def kth_smallest_sorted_matrix(matrix: list[list[int]], k: int) -> int:
    """LeetCode 378: 每行(列)升序矩阵第 K 小, 多路归并小根堆 O(k log n)。

    来源: 快手-一面-2。另一解法: 值域二分 + 统计 <= mid 的个数。
    """
    n = len(matrix)
    heap = [(matrix[r][0], r, 0) for r in range(min(n, k))]
    heapq.heapify(heap)
    val = matrix[0][0]
    for _ in range(k):
        val, r, c = heapq.heappop(heap)
        if c + 1 < n:
            heapq.heappush(heap, (matrix[r][c + 1], r, c + 1))
    return val


def median_of_k_sorted_arrays(arrays: list[list[int]]) -> float:
    """K 个有序数组的中位数: 二分答案值域 + count_less_equal 计数。

    来源: MiniMax-实习-一面-1。K 路归并堆解法在总长很大时是 O(N log K),
    值域二分是 O(K log(range) * log N) 级别, 对超大规模友好。
    这里实现"第 k 小"的二分解, 中位数取两次第 k 小。
    """
    total = sum(len(a) for a in arrays)
    if total == 0:
        raise ValueError("empty input")

    def count_le(x: int) -> int:
        import bisect

        return sum(bisect.bisect_right(a, x) for a in arrays)

    def kth_smallest(k: int) -> int:  # 全局第 k 小(1-indexed)
        lo = min(a[0] for a in arrays)
        hi = max(a[-1] for a in arrays)
        while lo < hi:
            mid = (lo + hi) // 2
            if count_le(mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo

    if total % 2:
        return float(kth_smallest(total // 2 + 1))
    return (kth_smallest(total // 2) + kth_smallest(total // 2 + 1)) / 2


def top_k_frequent(words: list[str], k: int) -> list[str]:
    """LeetCode 692: 前 K 高频字符串, 频次降序、同频字典序升序。

    来源: 网易-校招（前 K 高频元素）、综合面经题库-1（前 K 高频字符串）。
    用堆 O(n log k); 桶排序可 O(n)。
    """
    counter = Counter(words)
    # Python 堆是小根堆: 频次取负, 同频字典序要升序所以 (-freq, word) 直接比较
    return [w for w, _ in heapq.nsmallest(k, counter.items(), key=lambda x: (-x[1], x[0]))]


def top_k_frequent_buckets(nums: list[int], k: int) -> list[int]:
    """LeetCode 347 桶排序版 O(n): 桶下标 = 频次。"""
    counter = Counter(nums)
    buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]
    for num, freq in counter.items():
        buckets[freq].append(num)
    out: list[int] = []
    for freq in range(len(buckets) - 1, 0, -1):
        for num in buckets[freq]:
            out.append(num)
            if len(out) == k:
                return out
    return out


class MedianFinder:
    """LeetCode 295: 数据流中位数, 双堆(大根堆存小一半 + 小根堆存大一半)。

    来源: 腾讯-实习-一二面（滑动窗口中位数的核心数据结构就是双堆+延迟删除）。
    add_num O(log n), find_median O(1)。
    """

    def __init__(self) -> None:
        self.low: list[int] = []   # 大根堆(存负数模拟)
        self.high: list[int] = []  # 小根堆

    def add_num(self, num: int) -> None:
        # 先入大根堆(较小一半), 再把其堆顶平衡到小根堆
        heapq.heappush(self.low, -num)
        heapq.heappush(self.high, -heapq.heappop(self.low))
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def find_median(self) -> float:
        if len(self.low) > len(self.high):
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2


def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """LeetCode 239: 单调递减双端队列, O(n)。

    队列存下标, 队首永远是窗口最大值; 入队前把队尾更小的都弹掉。
    来源: MiniMax-实习-一面-2。堆解法是 O(n log k)。
    """
    dq: deque[int] = deque()  # 存下标, 对应值单调递减
    out: list[int] = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:  # 队首过期
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


def sliding_window_median(nums: list[int], k: int) -> list[float]:
    """LeetCode 480: 滑动窗口中位数。

    简洁解法: 每个窗口 sorted 取中位数, O(n*klogk), 面试先给这个再优化;
    最优解是双堆 + 延迟删除 O(n log k), 思路见 docs/interview-answers/12-算法题.md。
    来源: 腾讯-实习-二面。
    """
    out: list[float] = []
    for i in range(len(nums) - k + 1):
        window = sorted(nums[i:i + k])
        if k % 2:
            out.append(float(window[k // 2]))
        else:
            out.append((window[k // 2 - 1] + window[k // 2]) / 2)
    return out
