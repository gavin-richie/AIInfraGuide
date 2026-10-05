"""动态规划 / 数学类高频题。

来源: 综合面经题库-1/2（下一个排列、两有序数组中位数、环上走法、两指键盘、n! 尾随零、
字符串相乘、K-Coverage Intervals）、字节-校招-1（环上 10 节点 n 步回原点）、
百度-一面-2（最长公共子序列）、百度-校招-二面（字符串相乘）、字节-实习（下一个排列）、
快手-校招-一面（n! 尾随零、股票最大利润）、蔚来-二面（背包）等。
"""
from __future__ import annotations


# -------------------------------------------------------------------- 子数组/子序列
def max_sub_array(nums: list[int]) -> int:
    """LeetCode 53 最大子数组和: Kadane 算法, dp[i] = max(nums[i], dp[i-1]+nums[i])。

    来源: 腾讯-校招-一二面-1。
    """
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


def longest_common_subsequence(a: str, b: str) -> int:
    """LeetCode 1143: 经典二维 DP, O(mn)。

    来源: 百度-一面-2。滚动数组可优化到 O(min(m,n)) 空间。
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def edit_distance(a: str, b: str) -> int:
    """LeetCode 72 编辑距离, 二维 DP。来源: 虾皮（编辑距离）。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def longest_palindromic_substring(s: str) -> str:
    """LeetCode 5: 中心扩展 O(n^2)/O(1), 也可 Manacher O(n)/DP O(n^2)。

    来源: 小米-一面。
    """
    if not s:
        return ""
    start = end = 0

    def expand(l: int, r: int) -> tuple[int, int]:
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l, r = l - 1, r + 1
        return l + 1, r - 1

    for i in range(len(s)):
        for l, r in (expand(i, i), expand(i, i + 1)):
            if r - l > end - start:
                start, end = l, r
    return s[start:end + 1]


# -------------------------------------------------------------------- 排列/组合数学
def next_permutation(nums: list[int]) -> None:
    """LeetCode 31 原地完成下一个排列:

    1. 从右找第一个升序对 nums[i] < nums[i+1];
    2. 从右找第一个大于 nums[i] 的 nums[j] 并交换;
    3. 反转 i+1 之后的后缀(它当前是降序, 变升序即最小)。
    来源: 字节-实习-三面、综合面经题库-2。
    """
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1:] = reversed(nums[i + 1:])


def find_median_sorted_arrays(a: list[int], b: list[int]) -> float:
    """LeetCode 4 Hard: 二分较短数组的分割线, O(log(min(m,n)))。

    来源: 综合面经题库-2、字节-校招-一面。面试也可先讲归并 O(m+n) 再优化。
    """
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    total_left = (m + n + 1) // 2
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2          # a 的分割线
        j = total_left - i          # b 的分割线
        a_left = a[i - 1] if i > 0 else float("-inf")
        a_right = a[i] if i < m else float("inf")
        b_left = b[j - 1] if j > 0 else float("-inf")
        b_right = b[j] if j < n else float("inf")
        if a_left <= b_right and b_left <= a_right:
            if (m + n) % 2:
                return float(max(a_left, b_left))
            return (max(a_left, b_left) + min(a_right, b_right)) / 2
        if a_left > b_right:
            hi = i - 1
        else:
            lo = i + 1
    raise ValueError("input not sorted")


def trailing_zeroes_of_factorial(n: int) -> int:
    """LeetCode 172: n! 尾随零 = min(2,5 的因子数) = 5 的因子数, 不断除 5。

    来源: 快手-校招-一面。
    """
    count = 0
    while n:
        n //= 5
        count += n
    return count


def circle_return_ways(n_nodes: int, steps: int) -> int:
    """环上 N 个节点走 S 步回到起点的方案数: 矩阵快速幂或 DP。

    dp[s][v] = dp[s-1][(v-1)%N] + dp[s-1][(v+1)%N]。
    来源: 字节-校招-1（10 节点 n 步回到原点的走法数）。
    """
    dp = [0] * n_nodes
    dp[0] = 1  # 第 0 步在起点
    for _ in range(steps):
        ndp = [0] * n_nodes
        for v in range(n_nodes):
            ndp[(v + 1) % n_nodes] += dp[v]
            ndp[(v - 1) % n_nodes] += dp[v]
        dp = ndp
    return dp[0]


def min_distance_two_keys(n: int) -> int:
    """LeetCode 650 两指键盘(只有复制/粘贴): n = 分解质因数之和。

    来源: 综合面经题库-2。DP 解: dp[i] = min(dp[j] + i//j) for j|i。
    """
    dp = [0] * (n + 1)
    for i in range(2, n + 1):
        dp[i] = i  # 全粘贴最坏
        j = 2
        while j * j <= i:
            if i % j == 0:
                dp[i] = min(dp[i], dp[i // j] + j)
            j += 1
    return dp[n]


def string_multiply(num1: str, num2: str) -> str:
    """LeetCode 43 大数乘法: 先乘加到结果数组再统一进位, O(mn)。

    来源: 百度-校招-二面-1。
    """
    if num1 == "0" or num2 == "0":
        return "0"
    m, n = len(num1), len(num2)
    res = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = (ord(num1[i]) - ord("0")) * (ord(num2[j]) - ord("0"))
            p1, p2 = i + j, i + j + 1
            total = mul + res[p2]
            res[p2] = total % 10
            res[p1] += total // 10
    start = 0 if res[0] else 1
    return "".join(map(str, res[start:]))


# -------------------------------------------------------------------- 背包 / 接雨水
def max_profit_stock(prices: list[int]) -> int:
    """LeetCode 121 买卖股票一次: 维护历史最低价。来源: 快手-校招-一面。"""
    lowest = float("inf")
    best = 0
    for p in prices:
        lowest = min(lowest, p)
        best = max(best, p - lowest)
    return best


def knapsack_01(weights: list[int], values: list[int], cap: int) -> int:
    """LeetCode 416 型 01 背包: 一维 DP 容量倒序。来源: 蔚来-二面。"""
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):  # 倒序保证每件物品只用一次
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[cap]


def knapsack_complete(weights: list[int], values: list[int], cap: int) -> int:
    """完全背包: 容量正序即可重复选取。来源: 蔚来-二面。"""
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(w, cap + 1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[cap]


def trap_rain_water(height: list[int]) -> int:
    """LeetCode 42 接雨水: 双指针 O(n)/O(1)。

    来源: 阿里-校招-一面。每格水量 = min(左边最大, 右边最大) - 自身高度。
    """
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    while left < right:
        if height[left] < height[right]:
            left_max = max(left_max, height[left])
            water += left_max - height[left]
            left += 1
        else:
            right_max = max(right_max, height[right])
            water += right_max - height[right]
            right -= 1
    return water


# -------------------------------------------------------------------- 区间题
def min_segment_length(points: list[int], k: int) -> int:
    """给定数轴上若干点和 k 条等长线段, 求能覆盖全部点的最短线段长度。

    二分答案 L, 贪心判定: 从最左未覆盖点开始, 一条线段尽量右放(盖到 p+L)。
    来源: 综合面经题库-2 Q6。
    """
    if not points:
        return 0
    if k <= 0:
        raise ValueError("k must be positive")
    pts = sorted(points)

    def coverable(length: int) -> bool:
        used, seg_end = 0, float("-inf")
        for p in pts:
            if p > seg_end:  # 新开一条线段, 左端贴住 p
                used += 1
                if used > k:
                    return False
                seg_end = p + length
        return True

    lo, hi = 0, pts[-1] - pts[0]
    while lo < hi:  # 整数二分
        mid = (lo + hi) // 2
        if coverable(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def k_coverage_intervals(intervals: list[tuple[int, int]], k: int) -> int:
    """被至少 k 个区间覆盖的整数点个数: 差分数组/扫描线。

    来源: 综合面经题库-1（K-Coverage Intervals）。
    """
    if not intervals:
        return 0
    diff: dict[int, int] = {}
    for l, r in intervals:
        diff[l] = diff.get(l, 0) + 1
        diff[r + 1] = diff.get(r + 1, 0) - 1
    events = sorted(diff.items())
    count = 0
    coverage = 0
    for idx, (pos, delta) in enumerate(events):
        coverage += delta
        if coverage >= k and idx + 1 < len(events):
            count += events[idx + 1][0] - pos
    return count


def min_rooms_meetings(intervals: list[tuple[int, int]]) -> int:
    """会议室 II(最多重叠区间数): 排序 + 最小堆, 或差分扫描线。

    来源: 与"线段覆盖""区间调度"同族, 阶跃星辰-二面考过区间调度。
    """
    import heapq

    if not intervals:
        return 0
    intervals = sorted(intervals)
    heap: list[int] = []  # 结束时间
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)
        else:
            heapq.heappush(heap, end)
    return len(heap)
