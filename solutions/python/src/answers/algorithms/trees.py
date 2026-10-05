"""二叉树 / 图遍历高频题。

来源: 百度-一面-3（层序遍历带层级）、百度-实习-1（右视图）、腾讯-实习-一二面（滑动
窗口中位数见 topk_heap）、字节-实习-一二三面（棋盘 BFS）、综合面经题库-2（二叉树最大
路径和、Path Sum III）、小米-实习（层序遍历）、元戎启行（拓扑排序）、腾讯 TEG（Trie）。
"""
from __future__ import annotations

from collections import deque
from typing import Optional


class TreeNode:
    __slots__ = ("val", "left", "right")

    def __init__(self, val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None) -> None:
        self.val, self.left, self.right = val, left, right

    @staticmethod
    def from_level_order(vals: list[Optional[int]]) -> Optional["TreeNode"]:
        """按 LeetCode 层序数组建树, None 表示空节点。"""
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


def level_order(root: Optional[TreeNode]) -> list[list[int]]:
    """LeetCode 102: BFS 按层分组。每层先记录 len(queue) 再弹出该数量节点。

    来源: 百度-一面-3（要求输出带层级）、小米-实习-一二面、阿里-一面-1。
    """
    if root is None:
        return []
    out: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        level = [node.val for node in queue]  # 当前队列恰为一层
        out.append(level)
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return out


def right_side_view(root: Optional[TreeNode]) -> list[int]:
    """LeetCode 199: 每层最后一个节点。来源: 百度-实习-1。"""
    if root is None:
        return []
    view: list[int] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        view.append(node.val)  # 循环结束后 node 即该层最后弹出的节点
    return view


def inorder_nonrecursive(root: Optional[TreeNode]) -> list[int]:
    """LeetCode 94 非递归中序: 显式栈。来源: 字节-抖音（非递归中序遍历）。"""
    out, stack, node = [], [], root
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.val)
        node = node.right
    return out


def max_path_sum(root: Optional[TreeNode]) -> int:
    """LeetCode 124 Hard: 二叉树最大路径和。

    递归返回"单边最大贡献", 全局答案在节点处合并左右两条边。
    来源: 综合面经题库-2。
    """
    best = -float("inf")

    def gain(node: Optional[TreeNode]) -> int:
        nonlocal best
        if node is None:
            return 0
        left = max(gain(node.left), 0)
        right = max(gain(node.right), 0)
        best = max(best, node.val + left + right)  # 经过本节点的完整路径
        return node.val + max(left, right)          # 只能选一边向上返回

    gain(root)
    return int(best)


def path_sum_iii(root: Optional[TreeNode], target: int) -> int:
    """LeetCode 437: 前缀和 + 哈希, O(n)。

    来源: 综合面经题库-2。
    """
    from collections import defaultdict

    prefix: defaultdict[int, int] = defaultdict(int)
    prefix[0] = 1

    def dfs(node: Optional[TreeNode], acc: int) -> int:
        if node is None:
            return 0
        acc += node.val
        count = prefix[acc - target]
        prefix[acc] += 1
        count += dfs(node.left, acc) + dfs(node.right, acc)
        prefix[acc] -= 1  # 回溯: 路径前缀和只对祖先有效
        return count

    return dfs(root, 0)


# -------------------------------------------------------------------- 拓扑排序
def topo_sort_kahn(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    """Kahn 算法（BFS 入度版）拓扑排序。

    来源: 元戎启行-一面、三星-二面（DAG+拓扑排序）、综合面经题库-1。
    edges 为 (u, v) 有向边。若存在环则返回空列表。
    """
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    indeg = [0] * num_nodes
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    queue = deque(i for i in range(num_nodes) if indeg[i] == 0)
    order: list[int] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return order if len(order) == num_nodes else []


def topo_sort_dfs(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]:
    """DFS 版拓扑排序: 三色标记法, 出栈序逆序即拓扑序, 顺带判环。"""
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * num_nodes
    order: list[int] = []

    def dfs(u: int) -> bool:  # 返回 False 表示有环
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                return False
            if color[v] == WHITE and not dfs(v):
                return False
        color[u] = BLACK
        order.append(u)
        return True

    for u in range(num_nodes):
        if color[u] == WHITE and not dfs(u):
            return []
    return order[::-1]


# -------------------------------------------------------------------- 网格搜索
def num_islands(grid: list[list[str]]) -> int:
    """LeetCode 200: 岛屿数量。DFS 沉岛, 也可 BFS/并查集。

    来源: 综合面经题库-1、蔚来-实习（改条件重做）。
    """
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])

    def sink(r: int, c: int) -> None:
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] == "1":
            grid[r][c] = "0"
            sink(r + 1, c), sink(r - 1, c), sink(r, c + 1), sink(r, c - 1)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                sink(r, c)
    return count


def bfs_board_reachable(board: list[list[str]], start: tuple[int, int]) -> set[tuple[int, int]]:
    """棋盘连通性 BFS（返回从 start 四连通可达的所有格子）。

    来源: 字节跳动-实习-二面（棋盘连通 BFS）。同型题: 蔚来-实习岛屿数量变体。
    """
    from collections import deque as dq

    rows, cols = len(board), len(board[0])
    sr, sc = start
    visited: set[tuple[int, int]] = set()
    queue: dq[tuple[int, int]] = dq([(sr, sc)])
    visited.add((sr, sc))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and board[nr][nc] != "#":
                visited.add((nr, nc))
                queue.append((nr, nc))
    return visited


# -------------------------------------------------------------------- Trie
class Trie:
    """LeetCode 208 实现 Trie。来源: 腾讯-TEG-一面（内存文件系统题）。"""

    def __init__(self) -> None:
        self.children: dict[str, "Trie"] = {}
        self.is_end = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            node = node.children.setdefault(ch, Trie())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        return self._find(prefix) is not None

    def _find(self, s: str) -> Optional["Trie"]:
        node = self
        for ch in s:
            node = node.children.get(ch)
            if node is None:
                return None
        return node
