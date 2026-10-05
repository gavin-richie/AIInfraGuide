"""二叉搜索树迭代器三种解法（阿里巴巴-淘天-AI-Infra-一面-2 Q8，LeetCode 173）。

从暴力到最优的演进（面试要求口述思路）：
1. Flatten：构造时一次性中序遍历存数组，next/hasNext O(1)，内存 O(n)；
2. 受控栈（最优解）：中序遍历"走一步算一步"，栈只存从根到下一个最小节点的
   左链，内存 O(h)（h 为树高），next 均摊 O(1)（每条边至多入栈/出栈一次）；
3. 生成器版：Python 用 yield 表达受控遍历，语义同解法 2，代码最短。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

from answers.algorithms.trees import TreeNode


class BSTIteratorFlatten:
    """解法 1（暴力）：构造时全量中序展开。next/hasNext O(1)，空间 O(n)。"""

    def __init__(self, root: Optional[TreeNode]) -> None:
        self._vals: list[int] = []
        self._idx = 0
        stack, node = [], root
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            self._vals.append(node.val)
            node = node.right

    def next(self) -> int:
        val = self._vals[self._idx]
        self._idx += 1
        return val

    def hasNext(self) -> bool:
        return self._idx < len(self._vals)


class BSTIteratorStack:
    """解法 2（最优）：受控栈。空间 O(h)，next 均摊 O(1)。

    栈内始终维护"当前最小未访问节点及其祖先左链"。
    next：弹出栈顶即最小值；若它有右子树，把右子树的整条左链压栈。
    """

    def __init__(self, root: Optional[TreeNode]) -> None:
        self._stack: list[TreeNode] = []
        self._push_left(root)

    def _push_left(self, node: Optional[TreeNode]) -> None:
        while node:
            self._stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self._stack.pop()
        self._push_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return bool(self._stack)


class BSTIteratorGenerator:
    """解法 3：生成器（惰性中序），语义与解法 2 等价。"""

    def __init__(self, root: Optional[TreeNode]) -> None:
        self._gen: Iterator[int] = self._inorder(root)
        self._next_val: int | None = None

    @staticmethod
    def _inorder(node: Optional[TreeNode]) -> Iterator[int]:
        stack, cur = [], node
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            yield cur.val
            cur = cur.right

    def next(self) -> int:
        if self._next_val is None:
            self._next_val = next(self._gen, None)
        val, self._next_val = self._next_val, None
        return val

    def hasNext(self) -> bool:
        if self._next_val is None:
            self._next_val = next(self._gen, None)
        return self._next_val is not None


def iterate_all(root: Optional[TreeNode]) -> list[int]:
    """用受控栈迭代器读出全序，等价于中序遍历结果（测试对拍用）。"""
    it = BSTIteratorStack(root)
    out: list[int] = []
    while it.hasNext():
        out.append(it.next())
    return out
