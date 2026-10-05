"""T5 批次算法题合集（海康/贝壳/好未来/数坤/上海AI实验室/智源）。

对应面试题:
    - 海康威视-一面   "合并两个有序数组"（LeetCode 88）
    - 贝壳            "字符串转换整数 atoi"（LeetCode 8）
    - 好未来          "两个字符串之间的转换"（编辑距离语义，见 dp.py）/ "根到叶路径数字之和"
    - 数坤科技        "在全零方阵中绘制一个内切圆（用 1 填充）"
    - 上海AI实验室    "判断字符串除前两个数字外，后续数字是否均由前两个数字相加组成"（累加序列）
    - 智源研究院-一面 "树形动态规划"（以"打家劫舍 III"树的最大独立集为例）
"""
from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------- 海康 Q6
def merge_sorted_array(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """LC88: 逆向双指针原地合并，O(m+n)，不开新数组。

    从后往前放较大的元素，避免正向合并时 nums1 前缀被覆盖。
    """
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1


# ---------------------------------------------------------------- 贝壳 Q13
def my_atoi(s: str) -> int:
    """LC8: 字符串转 32 位 int。状态: 跳过前导空格 → 可选符号 → 连续数字 → 截断。

    易错点: 符号后必须紧跟数字; 溢出钳制到 [-2^31, 2^31-1]; 中途遇到非数字立即停止。
    """
    INT_MAX, INT_MIN = 2**31 - 1, -2**31
    i, n = 0, len(s)
    while i < n and s[i] == " ":
        i += 1
    sign = 1
    if i < n and s[i] in "+-":
        sign = -1 if s[i] == "-" else 1
        i += 1
    num = 0
    while i < n and s[i].isdigit():
        num = num * 10 + int(s[i])
        if sign == 1 and num > INT_MAX:
            return INT_MAX
        if sign == -1 and -num < INT_MIN:
            return INT_MIN
        i += 1
    return sign * num


# ---------------------------------------------------------------- 好未来 Q5
class TreeNode:
    def __init__(self, val: int = 0,
                 left: "Optional[TreeNode]" = None, right: "Optional[TreeNode]" = None):
        self.val, self.left, self.right = val, left, right


def build_tree(values: list[Optional[int]]) -> Optional[TreeNode]:
    """按层序（None 表示空）建树，方便测试。"""
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = [root]
    i = 1
    while queue and i < len(values):
        node = queue.pop(0)
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def sum_root_to_leaf(root: Optional[TreeNode]) -> int:
    """LC129: 根到叶路径表示的数字之和。DFS 传"高位累积值" cur = cur*10 + node.val。"""
    def dfs(node: Optional[TreeNode], cur: int) -> int:
        if node is None:
            return 0
        cur = cur * 10 + node.val
        if node.left is None and node.right is None:
            return cur
        return dfs(node.left, cur) + dfs(node.right, cur)

    return dfs(root, 0)


# ---------------------------------------------------------------- 数坤 Q8
def draw_inscribed_circle(n: int) -> list[list[int]]:
    """在 n×n 全零方阵中把内切圆内的格子置 1。

    判据用"格心到圆心距离 ≤ r"，格心落在圆周恰好的格子计入;
    返回 n×n 的 0/1 矩阵（题目原意: 端侧画布/掩码生成的模拟）。
    """
    cx = cy = (n - 1) / 2.0
    radius = n / 2.0
    return [[1 if (r - cy) ** 2 + (c - cx) ** 2 <= radius**2 else 0
             for c in range(n)] for r in range(n)]


# ---------------------------------------------------------------- 上海AI实验室 Q7
def is_additive_string(s: str) -> bool:
    """判断 s（数字串）除前两个数字外，后续数字是否均等于前两数之和（斐波那契式累加序列）。

    如 "112358" → 1,1,2,3,5,8 ✓; "199100199" → 1,99,100,199 ✓。
    枚举前两个数的切分点（各 O(n) 种），逐项验证; 注意前导零非法、大数直接 int 处理。
    """
    n = len(s)
    for i in range(1, n):            # 第一个数 s[:i]
        for j in range(i + 1, n):    # 第二个数 s[i:j]
            a, b = s[:i], s[i:j]
            if (len(a) > 1 and a[0] == "0") or (len(b) > 1 and b[0] == "0"):
                continue
            rest = s[j:]
            x, y = int(a), int(b)
            seq_ok = True
            while rest:
                z = x + y
                zs = str(z)
                if not rest.startswith(zs):
                    seq_ok = False
                    break
                rest = rest[len(zs):]
                x, y = y, z
            if seq_ok and j < n:      # 至少存在第三个数（"均由前两个相加组成"）
                return True
    return False


# ---------------------------------------------------------------- 智源-一面 Q4
def house_robber_iii(root: Optional[TreeNode]) -> int:
    """树形 DP（LC337 打家劫舍 III / 树的最大带权独立集）。

    每个节点返回 (不选该节点的最优, 选该节点的最优):
        skip = max(左) + max(右)
        take = node.val + 左.skip + 右.skip
    相邻节点不能同时选 → "没有上司的舞会"同款模型。O(n)。
    """
    def dfs(node: Optional[TreeNode]) -> tuple[int, int]:
        if node is None:
            return (0, 0)
        l_skip, l_take = dfs(node.left)
        r_skip, r_take = dfs(node.right)
        skip = max(l_skip, l_take) + max(r_skip, r_take)
        take = node.val + l_skip + r_skip
        return (skip, take)

    return max(dfs(root))


def tree_max_independent_set_bruteforce(root: Optional[TreeNode]) -> int:
    """指数暴力对拍用: 枚举每个节点选/不选，检查独立性与权重。"""
    nodes: list[TreeNode] = []

    def collect(node: Optional[TreeNode], parent: Optional[TreeNode]):
        if node is None:
            return
        nodes.append(node)
        collect(node.left, node)
        collect(node.right, node)

    collect(root, None)
    parent_of = {id(ch): id(p) for p in nodes for ch in (p.left, p.right) if ch}
    best = 0
    n = len(nodes)
    for mask in range(1 << n):
        ids = {id(nodes[i]) for i in range(n) if mask >> i & 1}
        if any(parent_of.get(v) in ids for v in ids):  # 父子同选 → 非独立集
            continue
        best = max(best, sum(nodes[i].val for i in range(n) if mask >> i & 1))
    return best
