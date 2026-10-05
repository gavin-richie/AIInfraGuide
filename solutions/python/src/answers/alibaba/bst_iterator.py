"""二叉搜索树迭代器（阿里巴巴-淘天-AI-Infra-一面-2 Q8，LeetCode 173）。

对应面试题："实现二叉搜索树迭代器，要求先阐述思路，给出从暴力到最优的
多种解法。"——考察点不在"会写"，而在按 复杂度阶梯 给出方案并说明取舍。

三种解法（从暴力到最优）：

1. ``BSTIteratorFlatten``：构造时一次中序遍历把全部值拍平进数组，
   next/hasNext O(1)。预处理 O(n)、空间 O(n)——树大但只访问前几个
   元素时浪费。
2. ``BSTIteratorStack``：显式栈模拟受控中序，懒推进：构造时只把
   根到最左的路径压栈（O(h)）；next 弹出栈顶并缓存，同时把它右子树的
   左脊柱压栈。next 均摊 O(1)（n 次 next 总共每节点进出栈一次），
   空间 O(h)。**面试的标准最优解**。
3. ``BSTIteratorMorris``：Morris 线索化——临时把中序前驱的 right 指回
   当前节点实现"回边"，遍历完恢复；空间 O(1)、next 均摊 O(1)，
   代价是遍历中临时修改树结构（并发读不安全、代码易错）。

基础版实现在《算法题》篇 min_stack.py::bst_iterator（栈解法），
本模块补齐多解法对比与均摊分析。
"""
from __future__ import annotations

from typing import Iterator, Optional


class TreeNode:
    def __init__(self, val: int = 0,
                 left: "Optional[TreeNode]" = None, right: "Optional[TreeNode]" = None):
        self.val = val
        self.left = left
        self.right = right


def _push_left(node: Optional[TreeNode], stack: list[TreeNode]) -> None:
    while node is not None:
        stack.append(node)
        node = node.left


class BSTIteratorFlatten:
    """解法 1（暴力）：构造时全量中序拍平。"""

    def __init__(self, root: Optional[TreeNode]):
        self._vals: list[int] = []
        self._inorder(root)
        self._i = 0

    def _inorder(self, node: Optional[TreeNode]) -> None:
        if node is None:
            return
        self._inorder(node.left)
        self._vals.append(node.val)
        self._inorder(node.right)

    def next(self) -> int:  # O(1)
        val = self._vals[self._i]
        self._i += 1
        return val

    def hasNext(self) -> bool:  # O(1)
        return self._i < len(self._vals)


class BSTIteratorStack:
    """解法 2（标准最优）：受控中序 + 显式栈，next 均摊 O(1)，空间 O(h)。"""

    def __init__(self, root: Optional[TreeNode]):
        self._stack: list[TreeNode] = []
        _push_left(root, self._stack)

    def next(self) -> int:
        node = self._stack.pop()          # 栈顶 = 当前最小未访问节点
        _push_left(node.right, self._stack)  # 右子树的左脊柱接上
        return node.val

    def hasNext(self) -> bool:
        return bool(self._stack)


class BSTIteratorMorris:
    """解法 3（空间最优）：Morris 线索化，O(1) 额外空间，均摊 O(1) next。

    维护"游标" cur：每次 next 先沿左脊柱走，过程中把每个节点的中序前驱
    （左子树最右节点）的 right 临时指向该节点形成回边；再次到达该节点时
    说明左子树已访问完 → 输出并沿回边上行，随后拆除线索恢复树结构。
    """

    def __init__(self, root: Optional[TreeNode]):
        self._cur = root
        self._pending: Optional[int] = None  # 惰性计算缓存的下一个值

    def _advance(self) -> Optional[int]:
        while self._cur is not None:
            if self._cur.left is None:
                val = self._cur.val
                self._cur = self._cur.right  # 回边（若已建）或右子树
                return val
            prev = self._cur.left
            while prev.right is not None and prev.right is not self._cur:
                prev = prev.right  # 找中序前驱
            if prev.right is None:
                prev.right = self._cur  # 建线索
                self._cur = self._cur.left
            else:
                prev.right = None  # 拆线索，恢复结构
                val = self._cur.val
                self._cur = self._cur.right
                return val
        return None

    def next(self) -> int:
        if self._pending is not None:
            val, self._pending = self._pending, None
            return val
        assert self._cur is not None, "next() called when exhausted"
        return self._advance()  # type: ignore[return-value]

    def hasNext(self) -> bool:
        if self._pending is not None:
            return True
        if self._cur is None:
            return False
        self._pending = self._advance()  # 惰性探一步
        return self._pending is not None


def iter_values(it) -> Iterator[int]:
    while it.hasNext():
        yield it.next()
