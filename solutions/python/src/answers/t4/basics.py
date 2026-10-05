"""T4 基础编程题合集：矩阵旋转 / 两数之和 / 最短路径 DP。

对应面试题:
    - 蔚来-AI-Infra-实习-一面-1 编程题 6: "用 Python 实现矩阵旋转"
    - 理想汽车-云-AI-Infra-实习-一面 编程题 5: "和等于 target 的两个数"（LeetCode 1）
    - 理想汽车-AI-Infra-校招-一面 编程题 10: "左上角到右下角最短路径"（LeetCode 64 同族）
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# 蔚来：矩阵旋转（顺时针 90°）
# ---------------------------------------------------------------------------
def rotate_matrix_pythonic(m: list[list[int]]) -> list[list[int]]:
    """一行版：倒序每行再转置。`[*zip(*m[::-1])]` 是面试口头加分写法。"""
    return [list(row) for row in zip(*m[::-1])]


def rotate_matrix_general(m: list[list[int]]) -> list[list[int]]:
    """M×N 通用版（非方阵）：out[j][N-1-i] = m[i][j]。"""
    if not m:
        return []
    rows, cols = len(m), len(m[0])
    return [[m[rows - 1 - j][i] for j in range(rows)] for i in range(cols)]


def rotate_matrix_inplace(m: list[list[int]]) -> None:
    """方阵原地版（LeetCode 48）：先转置再逐行翻转，O(1) 额外空间。"""
    n = len(m)
    for i in range(n):
        for j in range(i + 1, n):
            m[i][j], m[j][i] = m[j][i], m[i][j]
    for row in m:
        row.reverse()


# ---------------------------------------------------------------------------
# 理想云：两数之和（LeetCode 1）
# ---------------------------------------------------------------------------
def two_sum(nums: list[int], target: int) -> tuple[int, int]:
    """哈希一遍扫：查 target-x 是否见过。O(n) 时间 / O(n) 空间。"""
    seen: dict[int, int] = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return seen[target - x], i
        seen[x] = i
    raise ValueError("no solution")


# ---------------------------------------------------------------------------
# 理想校招：左上 → 右下最短路径（LeetCode 64 最小路径和，只能右/下移动）
# ---------------------------------------------------------------------------
def min_path_sum(grid: list[list[int]]) -> int:
    """1D 滚动数组 DP：dp[j] = grid[i][j] + min(dp[j](上), dp[j-1](左))。"""
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    dp = [0] * cols
    dp[0] = grid[0][0]
    for j in range(1, cols):
        dp[j] = dp[j - 1] + grid[0][j]
    for i in range(1, rows):
        dp[0] += grid[i][0]
        for j in range(1, cols):
            dp[j] = grid[i][j] + min(dp[j], dp[j - 1])
    return dp[-1]
