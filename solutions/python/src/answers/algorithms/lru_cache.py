"""LRU Cache（LeetCode 146）。

来源: 综合面经题库-1、字节-AML一面、阿里巴巴-一面-1、阶跃星辰、腾讯 TEG 等,
是全库出现频率最高的手写题之一。

两种实现:
    1. LRUCache        -- 哈希表 + 双向链表, get/put 均 O(1), 面试标准答案
    2. LRUCacheOrdered -- collections.OrderedDict 一行流, 展示 Python 工程写法

C++ 版见 solutions/cpp/include/answers/lru_cache.hpp。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: int = -1, value: int = -1) -> None:
        self.key, self.value = key, value
        self.prev: Optional[_Node] = None
        self.next: Optional[_Node] = None


class LRUCache:
    """哈希表 + 双向链表（带哨兵头尾）实现 O(1) LRU。

    - 哈希表 key -> 链表节点, 实现 O(1) 定位
    - 链表头部 = 最近使用, 尾部 = 最久未用
    - get/put 命中后把节点移到头部; 容量满时淘汰尾部
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.cap = capacity
        self.map: dict[int, _Node] = {}
        self.head, self.tail = _Node(), _Node()  # 哨兵
        self.head.next, self.tail.prev = self.tail, self.head

    # ---- 链表内部操作 ----
    def _remove(self, node: _Node) -> None:
        node.prev.next, node.next.prev = node.next, node.prev

    def _push_front(self, node: _Node) -> None:
        node.next, node.prev = self.head.next, self.head
        self.head.next.prev = node
        self.head.next = node

    # ---- 对外接口 ----
    def get(self, key: int) -> int:
        node = self.map.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._push_front(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        node = self.map.get(key)
        if node is not None:
            node.value = value
            self._remove(node)
            self._push_front(node)
            return
        if len(self.map) == self.cap:
            lru = self.tail.prev
            assert lru is not None and lru.key != -1
            self._remove(lru)
            del self.map[lru.key]
        node = _Node(key, value)
        self.map[key] = node
        self._push_front(node)

    def __len__(self) -> int:
        return len(self.map)

    def keys_mru_to_lru(self) -> list[int]:
        """调试辅助: 按最近使用到最久未用输出 key 序列。"""
        out, node = [], self.head.next
        while node is not self.tail:
            out.append(node.key)
            node = node.next
        return out


class LRUCacheOrdered(OrderedDict):
    """OrderedDict 版: move_to_end(last=False) 即"提升为最新"。"""

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.cap = capacity

    def get(self, key: int) -> int:  # type: ignore[override]
        if key not in self:
            return -1
        self.move_to_end(key, last=False)
        return super().__getitem__(key)

    def put(self, key: int, value: int) -> None:  # type: ignore[override]
        if key in self:
            self.move_to_end(key, last=False)
        elif len(self) >= self.cap:
            self.popitem(last=True)  # 淘汰最久未用
        super().__setitem__(key, value)
