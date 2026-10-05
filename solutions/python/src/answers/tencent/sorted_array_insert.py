"""无序数组元素插入有序数组并保持有序。

来源: 腾讯-AI-Infra-实习-一二面(编程 8), 需分析时间复杂度。

两条路线(有序数组长 n, 无序数组长 m):
1. 逐个二分插入: 每次查找 O(log n) + 搬移后缀 O(n), 总 O(m*n) -- 适合流式(来一个插一个);
2. 先排序再归并: O(m log m) + O(n+m) -- 适合批量, m 大时远优。
"""
from __future__ import annotations

from bisect import bisect_right


def _lower_bound(arr: list[int], x: int) -> int:
    """手写二分: 第一个 >= x 的位置(等价 bisect_left)。"""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def insert_one_binary(sorted_arr: list[int], x: int) -> list[int]:
    """单个元素二分插入, O(log n + n)。返回新数组(不改入参)。"""
    i = _lower_bound(sorted_arr, x)
    return sorted_arr[:i] + [x] + sorted_arr[i:]


def insert_all_one_by_one(sorted_arr: list[int], unsorted: list[int]) -> list[int]:
    """逐个二分插入, 总 O(m * (n+m))。流式场景的正确选择。"""
    arr = list(sorted_arr)
    for x in unsorted:
        arr.insert(_lower_bound(arr, x), x)
    return arr


def insert_all_via_merge(sorted_arr: list[int], unsorted: list[int]) -> list[int]:
    """无序数组排序 + 双指针归并, 总 O(m log m + n + m)。批量场景的选择。"""
    second = sorted(unsorted)
    out: list[int] = []
    i = j = 0
    while i < len(sorted_arr) and j < len(second):
        if sorted_arr[i] <= second[j]:  # <= 保证稳定(原序数组元素在前)
            out.append(sorted_arr[i])
            i += 1
        else:
            out.append(second[j])
            j += 1
    out.extend(sorted_arr[i:])
    out.extend(second[j:])
    return out


def bisect_insert_std(sorted_arr: list[int], unsorted: list[int]) -> list[int]:
    """标准库版对照(bisect.insort 语义), 用于测试对拍。"""
    arr = list(sorted_arr)
    for x in unsorted:
        arr.insert(bisect_right(arr, x), x)
    return arr
