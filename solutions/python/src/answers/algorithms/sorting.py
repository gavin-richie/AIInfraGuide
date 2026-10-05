"""排序类题。

来源: 综合面经题库-2（快排）、腾讯-TEG-三面（无库函数大写转小写并排序）、
     三星（回文串判断）、华为-实习（快排/堆排）等。
"""
from __future__ import annotations

import heapq
from collections import Counter


def quick_sort(nums: list[int]) -> list[int]:
    """快排——面试标准手写版(随机化三路取中等优化见 quick_sort_inplace)。

    来源: 综合面经题库-2、百度-校招-一面、腾讯-TEG-三面。
    平均 O(n log n), 最坏 O(n^2), 空间 O(log n) 递归栈。
    """
    if len(nums) <= 1:
        return nums[:]
    import random

    pivot = random.choice(nums)
    less = [x for x in nums if x < pivot]
    equal = [x for x in nums if x == pivot]
    greater = [x for x in nums if x > pivot]
    return quick_sort(less) + equal + quick_sort(greater)


def quick_sort_inplace(nums: list[int], lo: int = 0, hi: "int | None" = None) -> None:
    """原地快排(随机化 + 双指针分区), 展示手写功底的标准版本。"""
    import random

    if hi is None:
        hi = len(nums) - 1
    if lo >= hi:
        return
    # 随机选 pivot 换到末尾, 防止有序数组退化 O(n^2)
    rand_idx = random.randint(lo, hi)
    nums[rand_idx], nums[hi] = nums[hi], nums[rand_idx]
    pivot, store = nums[hi], lo
    for i in range(lo, hi):
        if nums[i] < pivot:
            nums[i], nums[store] = nums[store], nums[i]
            store += 1
    nums[store], nums[hi] = nums[hi], nums[store]
    quick_sort_inplace(nums, lo, store - 1)
    quick_sort_inplace(nums, store + 1, hi)


def heap_sort(nums: list[int]) -> list[int]:
    """堆排序 O(n log n)/O(1): 建堆 + 逐个取出。

    来源: 华为-实习-2、百度-校招（STL 做堆的容器为 priority_queue / make_heap）。
    """
    heapq.heapify(nums)
    return [heapq.heappop(nums) for _ in range(len(nums))]


def merge_sort(nums: list[int]) -> list[int]:
    """归并排序 O(n log n) 稳定排序。来源: 小马智行（归并排序）。"""
    if len(nums) <= 1:
        return nums[:]
    mid = len(nums) // 2
    left, right = merge_sort(nums[:mid]), merge_sort(nums[mid:])
    out: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def insertion_sort(nums: list[int]) -> list[int]:
    """插入排序, 小数据/近乎有序时快于快排(算子排序里小 kernel 常用它收尾)。"""
    for i in range(1, len(nums)):
        key, j = nums[i], i - 1
        while j >= 0 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1
        nums[j + 1] = key
    return nums


# ---------------------------------------------------------------- 字符串
def uppercase_to_lowercase_sorted(s: str) -> str:
    """不用库函数: 手写大写转小写(ord 差 32) + 计数排序 O(n+k)。

    来源: 腾讯-TEG-三面。若只要求排序不要求计数排序, 手写快排即可。
    """
    # 手写 tolower: 'A'=65, 'a'=97
    lowered = [chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s]
    # 计数排序(字符集可见范围)
    min_c, max_c = min(ord(c) for c in lowered), max(ord(c) for c in lowered)
    counts = [0] * (max_c - min_c + 1)
    for c in lowered:
        counts[ord(c) - min_c] += 1
    out: list[str] = []
    for i, cnt in enumerate(counts):
        out.extend([chr(min_c + i)] * cnt)
    return "".join(out)


def is_palindrome(s: str) -> bool:
    """回文判断: 双指针 O(n)。来源: 三星-一面。"""
    i, j = 0, len(s) - 1
    while i < j:
        if s[i] != s[j]:
            return False
        i, j = i + 1, j - 1
    return True


def check_inclusion(s1: str, s2: str) -> bool:
    """LeetCode 567 字符串排列: 滑动窗口 + 计数。来源: 中科类脑（文本替换同族）。"""
    if len(s1) > len(s2):
        return False
    need = Counter(s1)
    window = Counter(s2[: len(s1)])
    if window == need:
        return True
    for i in range(len(s1), len(s2)):
        window[s2[i]] += 1
        left_ch = s2[i - len(s1)]
        window[left_ch] -= 1
        if window[left_ch] == 0:
            del window[left_ch]
        if window == need:
            return True
    return False
