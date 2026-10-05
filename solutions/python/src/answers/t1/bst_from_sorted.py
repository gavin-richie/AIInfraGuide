"""LeetCode 108. 有序数组转换为二叉搜索树（高度平衡）。

来源: 快手-AI-Infra-实习-一面-1 编程题 10。

与 LeetCode 109（有序链表转 BST，见 answers/algorithms/linked_list.py 的
sorted_list_to_bst）同族：链表版用快慢指针找中点 O(n log n)，数组版直接
随机访问中点，O(n)。两种"中点取法"（左中点/右中点）都合法。

is_height_balanced 用于自验"平衡"：任意节点左右子树高度差 <= 1（AVL 定义）。
inorder 校验 BST 性质：中序遍历严格递增。
"""

from answers.algorithms.trees import TreeNode


def sorted_array_to_bst(nums: list[int]) -> TreeNode | None:
    """取中点作根，递归构建左右子树。"""
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid + 1 :])
    return root


def sorted_array_to_bst_iterative(nums: list[int]) -> TreeNode | None:
    """追问版：不用递归（避免深递归栈），显式栈维护 (lo, hi, parent, side)。"""
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    # 栈元素: (区间左端, 区间右端(闭), 挂载父节点, 挂载方向)
    stack: list[tuple[int, int, TreeNode, str]] = []
    if mid - 1 >= 0:
        stack.append((0, mid - 1, root, "left"))
    if mid + 1 < len(nums):
        stack.append((mid + 1, len(nums) - 1, root, "right"))
    while stack:
        lo, hi, parent, side = stack.pop()
        mid2 = (lo + hi) // 2
        node = TreeNode(nums[mid2])
        if side == "left":
            parent.left = node
        else:
            parent.right = node
        if lo <= mid2 - 1:
            stack.append((lo, mid2 - 1, node, "left"))
        if mid2 + 1 <= hi:
            stack.append((mid2 + 1, hi, node, "right"))
    return root


def tree_height(root: TreeNode | None) -> int:
    return 0 if root is None else 1 + max(tree_height(root.left), tree_height(root.right))


def is_height_balanced(root: TreeNode | None) -> bool:
    """AVL 意义的高度平衡检查（后序遍历，-1 表示子树已失衡，提前剪枝）。"""

    def check(node: TreeNode | None) -> int:
        if node is None:
            return 0
        left = check(node.left)
        if left == -1:
            return -1
        right = check(node.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return 1 + max(left, right)

    return check(root) != -1


def inorder_values(root: TreeNode | None) -> list[int]:
    values: list[int] = []
    stack: list[TreeNode] = []
    node = root
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        values.append(node.val)
        node = node.right
    return values
