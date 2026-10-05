"""快慢指针找链表中点 + 反转链表, 应用: O(1) 空间判断回文链表。

来源: 腾讯-TEG-AI-Infra-一二三面(二面 编程 9)。LeetCode 876 + 206 + 234。

ListNode 复用 answers.algorithms.linked_list 的定义; 反转/找中点在此独立
实现以便给出完整的面试手写版。
"""
from __future__ import annotations

from typing import Optional

from answers.algorithms.linked_list import ListNode


def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """快慢指针: slow 一步 fast 两步。

    偶数长度返回后半个的第一个节点(循环条件 fast.next and fast.next.next),
    奇数长度返回正中; head 为 None 返回 None。
    """
    slow = fast = head
    while fast is not None and fast.next is not None and fast.next.next is not None:
        slow = slow.next  # type: ignore[union-attr]
        fast = fast.next.next
    # 此时 slow 在前半尾部; 回文场景要的是后半起点, 再走一步
    return slow.next if slow is not None else None


def find_middle_or_first(head: Optional[ListNode]) -> Optional[ListNode]:
    """偶数长度返回前半最后一个(奇数返回正中), 供两类题选边站。"""
    slow = fast = head
    while fast is not None and fast.next is not None:
        fast = fast.next.next
        if fast is not None:
            slow = slow.next  # type: ignore[union-attr]
    return slow


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """三指针迭代反转, O(n)/O(1)。"""
    prev: Optional[ListNode] = None
    cur = head
    while cur is not None:
        cur.next, prev, cur = prev, cur, cur.next
    return prev


def is_palindrome(head: Optional[ListNode]) -> bool:
    """LeetCode 234: 找中点 -> 反转后半 -> 同步比较 -> (恢复链表)。

    O(n) 时间 / O(1) 空间。
    """
    if head is None or head.next is None:
        return True
    second = reverse_list(find_middle(head))
    # 保存反转后的头以便恢复
    tail_head = second
    ok = True
    p, q = head, second
    while q is not None:  # 后半 <= 前半长度, 用 q 驱动
        if p.val != q.val:
            ok = False
        p, q = p.next, q.next  # type: ignore[union-attr]
    reverse_list(tail_head)  # 恢复原链表(面试加分: 不破坏输入)
    return ok
