"""位运算 / 数学题。

来源: 综合面经题库-1（Hamming Weight）、综合题库-5（整数次幂）、字节-校招-一面
（二进制 1 的个数）、字节-实习-一面-4（sqrt(x)）、三星（二进制串找第一个 1）、
腾讯-校招（Hamming Weight）等。
"""
from __future__ import annotations


def hamming_weight(n: int) -> int:
    """LeetCode 191: 二进制中 1 的个数。

    基础解: 逐位 & 1 再右移, O(log n)。
    进阶解: n & (n-1) 每次消去最低位的 1, 循环次数 = 1 的个数, 见下方。
    来源: 综合面经题库-1、腾讯-校招-一二面-1、字节-校招-一面。
    """
    count = 0
    while n:
        n &= n - 1  # 每次去掉最低位的 1
        count += 1
    return count


def hamming_weight_shift(n: int) -> int:
    """逐位判断版本(面试常要求两种写法对比)。"""
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def first_one_index(n: int) -> int:
    """二进制串中第一个 1 的位置(从 0 位起, n=0 返回 -1)。

    来源: 三星-一面。Python 可用 n.bit_length()-1, 也可 x & (-x) 取最低位 1。
    """
    if n == 0:
        return -1
    lowest = n & (-n)          # 只保留最低位的 1
    return lowest.bit_length() - 1


def is_power_of_two(n: int) -> bool:
    """2 的整数次幂判定: n > 0 且 n & (n-1) == 0。"""
    return n > 0 and (n & (n - 1)) == 0


def my_pow(x: float, n: int) -> float:
    """LeetCode 50 快速幂: O(log n), 迭代处理指数二进制位。

    来源: 综合题库-5（整数次幂）。
    """
    if n < 0:
        x, n = 1 / x, -n
    result = 1.0
    while n:
        if n & 1:
            result *= x
        x *= x
        n >>= 1
    return result


def my_sqrt(x: int) -> int:
    """LeetCode 69 sqrt(x): 整数二分或牛顿迭代, 只返回整数部分。

    来源: 字节-实习-一面-4。
    """
    if x < 2:
        return x
    lo, hi = 1, x // 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid * mid == x:
            return mid
        if mid * mid < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi


def newton_sqrt(x: float, iters: int = 50) -> float:
    """牛顿迭代法开方: y_{t+1} = (y_t + x/y_t) / 2, 二次收敛。"""
    if x == 0:
        return 0.0
    y = x / 2 if x > 1 else x
    for _ in range(iters):
        y = (y + x / y) / 2
    return y


def float_equal(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """C++ 面试题"如何比较两个 float 相等"的 Python 演示:

    绝对误差 + 相对误差双阈值(math.isclose 同思想)。
    来源: 综合面经题库-1 算法题 2。
    """
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def nth_char_rotated_concat(s: str, n: int) -> str:
    """LeetCode 1545 变体(面试原题): 初始串 s, 每次将当前串右旋一位后拼到末尾,
    长度每轮翻倍(len*2^k), 求无限扩展后第 N 位字符(1-indexed)。

    递归降维: 设 L_k = len(s) * 2^k 覆盖 N。
    - N 落在前半(L_{k-1}) → 递归 f(k-1, N);
    - N 落在后半: 后半 = rotate_right(前半), 即后半第 i 位(0-indexed)
      = 前半第 (i-1) mod L_{k-1} 位 → 递归 f(k-1, ((i-1) mod L_{k-1}) + 1)。
    O(log N) 时间。
    来源: 综合面经题库-2 算法题 8。
    """
    if n <= 0:
        raise ValueError("n must be positive")
    m = len(s)
    k = 0
    while m * (1 << k) < n:  # 找到覆盖 N 的最小代数
        k += 1

    def f(k: int, n: int) -> str:
        if k == 0:
            return s[n - 1]
        half = m << (k - 1)
        if n <= half:
            return f(k - 1, n)
        i = n - half - 1  # 后半中的 0-indexed 位置
        prev = (i - 1) % half  # rotate_right: 后半[i] = 前半[i-1]
        return f(k - 1, prev + 1)

    return f(k, n)


def nth_char_rotated_concat_bruteforce(s: str, n: int) -> str:
    """暴力展开版, 仅用于对拍验证递归版正确性。"""
    cur = s
    while len(cur) < n:
        cur = cur + (cur[-1] + cur[:-1])
    return cur[n - 1]
