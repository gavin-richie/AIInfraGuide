"""有序二进制字符串中查找第一个 1。

来源: 三星-AI-Infra-一面-1（编程 10）。串形如 "000...0111...1"（前缀全 0 后缀全 1，
即布尔单调序列），二分 O(log n) 找最左的 '1'；全 0 返回 -1。
若输入是整数比特串（0/1 排列任意）则只能 O(n) 扫描——面试先问清单调性。
"""
from __future__ import annotations


def first_one(s: str) -> int:
    """返回第一个 '1' 的下标, 不存在返回 -1。要求 0 在前 1 在后（单调）。

    >>> first_one("000111")
    3
    >>> first_one("0000")
    -1
    """
    if not s or s[-1] == "0":
        return -1
    lo, hi = 0, len(s) - 1
    while lo < hi:                      # 找第一个 '1' = lower_bound
        mid = (lo + hi) // 2
        if s[mid] == "1":
            hi = mid
        else:
            lo = mid + 1
    return lo


def first_one_general(s: str) -> int:
    """不依赖单调性的 O(n) 兜底版（对拍用）。"""
    for i, ch in enumerate(s):
        if ch == "1":
            return i
    return -1
