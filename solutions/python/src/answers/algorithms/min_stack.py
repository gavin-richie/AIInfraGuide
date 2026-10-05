"""O(1) 最小值栈（LeetCode 155）与栈类题。

来源: 字节跳动-实习-三面（O(1) 最小栈）、淘天-一面-2（BST 迭代器多解法）、
     百度-实习-2（栈/队列互实现）、百度-校招-二面-1（验证栈序列）。
"""
from __future__ import annotations

from collections import deque
from typing import Optional


class MinStack:
    """辅助栈同步维护当前最小值, push/pop/get_min 都是 O(1)。

    两种等价写法: 辅助栈与主栈等长(存"当前最小值"), 或只在更小时入栈+计数。
    这里用前者, 更简单且 pop 无分支。
    """

    def __init__(self) -> None:
        self._stack: list[int] = []
        self._min_stack: list[int] = []

    def push(self, val: int) -> None:
        self._stack.append(val)
        if not self._min_stack or val <= self._min_stack[-1]:
            self._min_stack.append(val)
        else:
            self._min_stack.append(self._min_stack[-1])

    def pop(self) -> int:
        self._min_stack.pop()
        return self._stack.pop()

    def top(self) -> int:
        return self._stack[-1]

    def get_min(self) -> int:
        return self._min_stack[-1]


def bst_iterator(root):
    """LeetCode 173 生成器解法: 中序遍历生成器天然支持"暂停/恢复"。

    空间 O(h) 的显式栈版本是标准答案, 生成器版本是 Python 面试加分写法;
    O(1) 空间的 Morris 遍历版本思路见 docs/interview-answers/12-算法题.md。
    来源: 阿里-淘天-一面-2（要求多解法）。

    用法: for val in bst_iterator(root): ...
    """
    stack, node = [], root
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        yield node.val
        node = node.right


class TwoQueuesStack:
    """两个队列实现栈（只用 popleft 模拟队列行为）。

    来源: 百度-实习-2。push O(n)、pop O(1) 的取舍。
    """

    def __init__(self) -> None:
        self.q1: deque[int] = deque()
        self.q2: deque[int] = deque()

    def push(self, x: int) -> None:
        # 入 q2, 再把 q1 全部接上, 交换 => q1 队首即栈顶
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1

    def pop(self) -> int:
        return self.q1.popleft()

    def top(self) -> int:
        return self.q1[0]


def validate_stack_sequences(pushed: list[int], popped: list[int]) -> bool:
    """LeetCode 946: 验证栈序列。贪心模拟入栈, 栈顶等于 popped 当前值则弹。

    来源: 百度-校招-二面-1。
    """
    stack, i = [], 0
    for x in pushed:
        stack.append(x)
        while stack and i < len(popped) and stack[-1] == popped[i]:
            stack.pop()
            i += 1
    return i == len(popped)
