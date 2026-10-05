"""百度批次新写算法题 —— 已有工程覆盖的题见 answers/algorithms/(53 篇附表)。

对应面试题:
    - 百度-AI-Infra-一面-3 Q11   层序遍历并记录层级(102 变体)  -> level_order_with_levels
    - 百度-AI-Infra-实习-2 Q23   用栈实现队列(232)              -> MyQueue
    - 百度-AI-Infra-校招-一面-2 Q6 翻转二叉树(226)              -> invert_tree
    - 百度-AI-Infra-校招-二面-1 Q6 字符串相乘(43)               -> multiply
    - 百度-AI-Infra-校招-二面-2 Q6 删除有序数组重复元素(26)      -> remove_duplicates
"""
from __future__ import annotations

from collections import deque
from typing import Optional


class TreeNode:
    __slots__ = ("val", "left", "right")

    def __init__(self, val: int = 0, left: "Optional[TreeNode]" = None,
                 right: "Optional[TreeNode]" = None) -> None:
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def from_list(vals: list[Optional[int]]) -> "Optional[TreeNode]":
        """层序数组(与 LeetCode 一致, None 为空位)建树。"""
        if not vals or vals[0] is None:
            return None
        root = TreeNode(vals[0])
        queue: deque[TreeNode] = deque([root])
        i = 1
        while queue and i < len(vals):
            node = queue.popleft()
            if i < len(vals) and vals[i] is not None:
                node.left = TreeNode(vals[i])  # type: ignore[arg-type]
                queue.append(node.left)
            i += 1
            if i < len(vals) and vals[i] is not None:
                node.right = TreeNode(vals[i])  # type: ignore[arg-type]
                queue.append(node.right)
            i += 1
        return root


# ---- 一3 Q11: 层序遍历带层级 -------------------------------------------------
def level_order_with_levels(root: Optional[TreeNode]) -> list[tuple[int, int]]:
    """返回 [(val, level)] (level 从 0 计)。

    队列元素带层号; 另有"每轮处理 len(queue) 个"的整层双循环写法
    (algorithms/trees.py::level_order), 两者面试都算对。
    """
    if not root:
        return []
    out: list[tuple[int, int]] = []
    queue: deque[tuple[TreeNode, int]] = deque([(root, 0)])
    while queue:
        node, lv = queue.popleft()
        out.append((node.val, lv))
        if node.left:
            queue.append((node.left, lv + 1))
        if node.right:
            queue.append((node.right, lv + 1))
    return out


# ---- 实2 Q23: 用栈实现队列 ---------------------------------------------------
class MyQueue:
    """双栈队列: in 只管 push, out 空时才把 in 倒入 —— 元素至多被搬运一次, 均摊 O(1)。"""

    def __init__(self) -> None:
        self.in_stack: list[int] = []
        self.out_stack: list[int] = []

    def push(self, x: int) -> None:
        self.in_stack.append(x)

    def _shift(self) -> None:
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())

    def pop(self) -> int:
        self._shift()
        return self.out_stack.pop()

    def peek(self) -> int:
        self._shift()
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack


# ---- 校一2 Q6: 翻转二叉树 ----------------------------------------------------
def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


def invert_tree_iterative(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """显式栈版: 深树防递归爆栈。"""
    stack = [root] if root else []
    while stack:
        node = stack.pop()
        node.left, node.right = node.right, node.left
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return root


# ---- 校二1 Q6: 字符串相乘 ----------------------------------------------------
def multiply(num1: str, num2: str) -> str:
    """大数竖式: 第 i 位与第 j 位的乘积累加到结果数组的 i+j+1 位, 延迟进位。"""
    if num1 == "0" or num2 == "0":
        return "0"
    m, n = len(num1), len(num2)
    res = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            cur = int(num1[i]) * int(num2[j]) + res[i + j + 1]
            res[i + j + 1] = cur % 10
            res[i + j] += cur // 10
    return "".join(map(str, res)).lstrip("0")


# ---- 校二2 Q6: 删除有序数组重复元素 --------------------------------------------
def remove_duplicates(nums: list[int]) -> int:
    """双指针原地去重, 返回新长度; 尾部残值不计。变体 LC 80(保留两次)见测试。"""
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1


def remove_duplicates_at_most_twice(nums: list[int]) -> int:
    """LC 80 变体: 每个元素最多保留两次(面试常见追问, 顺手实现)。"""
    if len(nums) <= 2:
        return len(nums)
    slow = 2
    for fast in range(2, len(nums)):
        if nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1
    return slow
