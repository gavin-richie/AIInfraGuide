"""双指针 / 滑动窗口题。

来源: 百度-实习-一面-1（最长连续序列、至多 K 不同字符最长子串）、综合面经题库-1
（三数之和、最长有效括号）、美团-校招-一面（行列有序矩阵查找）等。
"""
from __future__ import annotations


def longest_consecutive(nums: list[int]) -> int:
    """LeetCode 128: 最长连续序列 O(n)。哈希集合, 只从"序列起点"开始数。

    来源: 百度-实习-一面-1。
    """
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 not in s:  # x 是某段连续序列的起点
            length = 1
            while x + length in s:
                length += 1
            best = max(best, length)
    return best


def length_of_longest_substring_k_distinct(s: str, k: int) -> int:
    """LeetCode 340: 至多 K 个不同字符的最长子串, 滑窗 + 计数哈希 O(n)。

    来源: 百度-实习-一面-1。
    """
    from collections import defaultdict

    count: defaultdict[str, int] = defaultdict(int)
    left = best = 0
    for right, ch in enumerate(s):
        count[ch] += 1
        while len(count) > k:  # 不同字符超限, 收缩左边界
            left_ch = s[left]
            count[left_ch] -= 1
            if count[left_ch] == 0:
                del count[left_ch]
            left += 1
        best = max(best, right - left + 1)
    return best


def length_of_longest_substring(s: str) -> int:
    """LeetCode 3: 无重复字符最长子串（k=1 的特例思想相同）。"""
    last_seen: dict[str, int] = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best


def three_sum(nums: list[int]) -> list[list[int]]:
    """LeetCode 15: 排序 + 固定一个数 + 双指针, O(n^2)。

    来源: 旷视-二面。
    """
    nums.sort()
    out: list[list[int]] = []
    n = len(nums)
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue  # 去重
        if nums[i] + nums[i + 1] + nums[i + 2] > 0:
            break
        if nums[i] + nums[n - 2] + nums[n - 1] < 0:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            if total < 0:
                lo += 1
            elif total > 0:
                hi -= 1
            else:
                out.append([nums[i], nums[lo], nums[hi]])
                while lo < hi and nums[lo] == nums[lo + 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi - 1]:
                    hi -= 1
                lo, hi = lo + 1, hi - 1
    return out


def longest_valid_parentheses(s: str) -> int:
    """LeetCode 32 Hard: 最长有效括号。栈底存"最后一个未匹配位置"。

    来源: 阿里-控股-实习-一面。
    """
    stack = [-1]  # 栈底哨兵: 最后一个无法匹配的右括号下标
    best = 0
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)  # 当前右括号无法匹配, 成为新哨兵
            else:
                best = max(best, i - stack[-1])
    return best


def search_matrix(matrix: list[list[int]], target: int) -> bool:
    """LeetCode 240: 行列均升序的矩阵查找, 从左下/右上角走 O(m+n)。

    来源: 美团-校招-一面。
    """
    if not matrix or not matrix[0]:
        return False
    r, c = len(matrix) - 1, 0  # 左下角
    while r >= 0 and c < len(matrix[0]):
        if matrix[r][c] == target:
            return True
        if matrix[r][c] > target:
            r -= 1
        else:
            c += 1
    return False


def longest_substring_without_repeat_or_k(s: str) -> dict[str, int]:
    """面试追问时演示两种窗口收缩策略的区别, 返回两种解法结果便于对比。"""
    return {
        "unique": length_of_longest_substring(s),
        "k_distinct_2": length_of_longest_substring_k_distinct(s, 2),
    }
