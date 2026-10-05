"""链表高频题（来源：字节跳动-实习-一二三面、综合面经题库-1/2、美团、小米、快手、阶跃星辰等）。

包含:
    - 反转链表 / 反转区间
    - 合并两个有序链表 / 合并 K 个升序链表
    - 链表判环 + 找环入口（快慢指针, O(1) 空间）
    - 相交链表找交点
    - K 个一组翻转链表
    - 删除排序链表中的重复元素 II
    - 两数相加（链表表示的逆序数）
    - 链表倒数第 K 个节点
    - 复制带随机指针的链表
    - 有序链表转平衡 BST
    - 单链表冒泡排序（蔚来）
"""
from __future__ import annotations

from typing import Optional


class ListNode:
    """LeetCode 风格单链表节点。"""

    __slots__ = ("val", "next")

    def __init__(self, val: int = 0, next: Optional["ListNode"] = None) -> None:
        self.val = val
        self.next = next

    @staticmethod
    def from_list(vals: list[int]) -> Optional["ListNode"]:
        dummy = ListNode(0)
        cur = dummy
        for v in vals:
            cur.next = ListNode(v)
            cur = cur.next
        return dummy.next

    def to_list(self) -> list[int]:
        out, node = [], self
        while node:
            out.append(node.val)
            node = node.next
        return out


class RandomNode:
    """带 random 指针的链表节点（复制随机指针链表）。"""

    __slots__ = ("val", "next", "random")

    def __init__(self, val: int, next: "RandomNode | None" = None, random: "RandomNode | None" = None):
        self.val, self.next, self.random = val, next, random


# ---------------------------------------------------------------- 反转链表
def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """迭代三指针反转，O(n)/O(1)。面试要求能默写。"""
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev


# ---------------------------------------------------------------- 合并有序链表
def merge_two_lists(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    """哨兵节点 + 双指针归并，O(n+m)/O(1)。"""
    dummy = ListNode(0)
    tail = dummy
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def merge_k_lists(lists: list[Optional[ListNode]]) -> Optional[ListNode]:
    """合并 K 个升序链表：小根堆 O(N log K)。

    来源: 字节-豆包实习二面、百度-实习-1、美团-实习。
    也可以分治两两合并，复杂度相同。
    """
    import heapq

    heap: list[tuple[int, int, ListNode]] = []
    for i, node in enumerate(lists):
        if node:
            # (val, 序号防比较 ListNode, node)
            heapq.heappush(heap, (node.val, i, node))
    dummy = ListNode(0)
    tail = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        tail.next = node
        tail = tail.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next


# ---------------------------------------------------------------- 环
def has_cycle(head: Optional[ListNode]) -> bool:
    """快慢指针判环：相遇则有环，O(n)/O(1)。"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    """找环入口。数学结论：相遇后把一个指针放回头部，同速前进再次相遇处即入口。"""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            slow = head
            while slow is not fast:
                slow, fast = slow.next, fast.next
            return slow
    return None


def get_intersection_node(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    """相交链表：双指针走完自己的再走对方的，路径等长后必在交点相遇（或同时为 None）。"""
    pa, pb = a, b
    while pa is not pb:
        pa = pa.next if pa else b
        pb = pb.next if pb else a
    return pa


# ---------------------------------------------------------------- K 个一组翻转
def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """LeetCode 25 Hard。先数够 k 个再局部反转，不足 k 保持原序。

    来源: 腾讯-实习-一面-1、美团-北斗、蔚来。
    """
    dummy = ListNode(0, head)
    prev_group = dummy
    while True:
        # 找本组第 k 个节点
        tail = prev_group
        for _ in range(k):
            tail = tail.next
            if tail is None:
                return dummy.next
        nxt_group = tail.next
        # 反转 [prev_group.next, tail]
        prev, cur = nxt_group, prev_group.next
        while cur is not nxt_group:
            cur.next, prev, cur = prev, cur, cur.next
        new_head, new_tail = prev, prev_group.next
        prev_group.next, new_tail.next = new_head, nxt_group
        prev_group = new_tail


# ---------------------------------------------------------------- 删除重复元素 II
def delete_duplicates_sorted(head: Optional[ListNode]) -> Optional[ListNode]:
    """LeetCode 82：删除有序链表中所有重复节点（一个不留）。

    来源: 美团-校招-一面。
    """
    dummy = ListNode(0, head)
    prev = dummy
    while prev.next and prev.next.next:
        if prev.next.val == prev.next.next.val:
            dup_val = prev.next.val
            while prev.next and prev.next.val == dup_val:
                prev.next = prev.next.next
        else:
            prev = prev.next
    return dummy.next


# ---------------------------------------------------------------- 两数相加
def add_two_numbers(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    """LeetCode 2：逆序链表表示的两数相加，模拟进位。来源: 太初-一二面。"""
    dummy = ListNode(0)
    cur, carry = dummy, 0
    while a or b or carry:
        s = (a.val if a else 0) + (b.val if b else 0) + carry
        carry, digit = divmod(s, 10)
        cur.next = ListNode(digit)
        cur = cur.next
        a = a.next if a else None
        b = b.next if b else None
    return dummy.next


# ---------------------------------------------------------------- 倒数第 K 个
def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """快慢指针间隔 n，快到尾时慢即待删节点前驱。来源: 旷视-一面。"""
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n):
        fast = fast.next
    while fast.next:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return dummy.next


# ---------------------------------------------------------------- 复制随机指针链表
def copy_random_list(head: Optional[RandomNode]) -> Optional[RandomNode]:
    """LeetCode 138：三步法——交错复制 / 复制 random / 拆分，O(n)/O(1)。

    来源: 蚂蚁-实习-一面-1（面试常考哈希法与原地法两种）。
    """
    if head is None:
        return None
    # 1. 每个节点后插入拷贝
    cur = head
    while cur:
        cur.next = RandomNode(cur.val, cur.next, None)
        cur = cur.next.next
    # 2. 拷贝节点的 random 指向原节点 random 的 next
    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
        cur = cur.next.next
    # 3. 拆分两条链
    dummy = RandomNode(0)
    tail, cur = dummy, head
    while cur:
        tail.next = cur.next
        tail = tail.next
        cur.next = cur.next.next
        cur = cur.next
    return dummy.next


# ---------------------------------------------------------------- 有序链表转 BST
def sorted_list_to_bst(head: Optional[ListNode]) -> Optional["TreeNode"]:
    """LeetCode 109：快慢指针找中点作根，递归建左右子树。来源: 快手-实习-一面-1。"""
    # 延迟导入避免模块级循环依赖
    from .trees import TreeNode

    def build(lo: Optional[ListNode], hi: Optional[ListNode]) -> Optional[TreeNode]:
        if lo is hi:
            return None
        slow = fast = lo
        while fast is not hi and fast.next is not hi:
            slow, fast = slow.next, fast.next.next
        root = TreeNode(slow.val)
        root.left = build(lo, slow)
        root.right = build(slow.next, hi)
        return root

    return build(head, None)


# ---------------------------------------------------------------- 单链表冒泡
def bubble_sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """链表版冒泡排序：只交换 val。来源: 蔚来-一面-2。"""
    if head is None or head.next is None:
        return head
    n, node = 0, head
    while node:
        n, node = n + 1, node.next
    for i in range(n - 1):
        swapped, cur = False, head
        for _ in range(n - 1 - i):
            if cur.val > cur.next.val:
                cur.val, cur.next.val = cur.next.val, cur.val
                swapped = True
            cur = cur.next
        if not swapped:
            break
    return head
