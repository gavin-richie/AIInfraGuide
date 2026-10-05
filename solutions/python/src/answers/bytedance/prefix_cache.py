"""Prefix Cache：radix tree 前缀缓存 + 叶子优先 LRU 淘汰（SGLang 风格）。

对应题目：字节-校招-1 推理优化 5「Prefix Cache 的机制是什么？适用于哪些场景？」

机制要点（与实现的对应关系）：

1. **块粒度匹配**：缓存与 PagedAttention 一样按 block（默认 16 token）组织，
   只有整块命中才算命中——partial block 无法被后续请求复用（KV 尚未算完/写满）。
2. **radix tree 共享**：多条请求的公共前缀（system prompt / few-shot）在树上共享，
   分叉后才各自占块，插拔序列零拷贝。
3. **叶子优先 LRU 淘汰**：只能淘汰叶子（没有孩子的节点）——中间节点被多条
   请求共享，删了会连坐。容量不足时反复摘 last_access 最老的叶子。
4. **命中即可跳过 prefill**：match 返回的 token 数直接对应可复用的 KV block，
   省掉这部分 prefill 计算与访存。

运行验证::

    cd solutions/python && python -m pytest tests/test_bytedance.py -q
"""

from __future__ import annotations


class _Node:
    """radix tree 节点：到该节点为止的前缀是一个 block 边界。

    ``children`` 的 key 是下一块的第一个 token，value 是 ``(块 token 元组, 子节点)``。
    一条边恰好一个 block，因此树高 = 前缀块数，match 天然是 O(块数)。
    """

    __slots__ = ("children", "last_access")

    def __init__(self) -> None:
        self.children: dict[int, tuple[tuple[int, ...], "_Node"]] = {}
        self.last_access = 0

    def is_leaf(self) -> bool:
        return not self.children


class RadixPrefixCache:
    """容量以 KV block 数计的 radix prefix cache。"""

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.root = _Node()
        self._blocks_in_use = 0
        self._clock = 0  # 逻辑时钟，用于 LRU

    # ------------------------------------------------------------------ utils

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _split_seq(self, tokens: list[int] | tuple[int, ...]) -> list[tuple[int, ...]]:
        """把序列切成 block_size 的块，末尾不足一块的部分丢弃（不缓存）。"""
        bs = self.block_size
        return [tuple(tokens[i : i + bs]) for i in range(0, len(tokens) - bs + 1, bs)]

    # ------------------------------------------------------------------- core

    def match(self, tokens: list[int] | tuple[int, ...]) -> int:
        """返回可复用的已缓存前缀 token 数（块粒度，向下取整到 block_size 的倍数）。"""
        node = self.root
        matched = 0
        for block in self._split_seq(tokens):
            child = node.children.get(block[0])
            if child is None or child[0] != block:
                break
            node = child[1]
            node.last_access = self._tick()
            matched += self.block_size
        return matched

    def insert(self, tokens: list[int] | tuple[int, ...]) -> int:
        """缓存一条序列，返回新建的 block 数（共享前缀部分不重复计数）。"""
        node = self.root
        created = 0
        for block in self._split_seq(tokens):
            child = node.children.get(block[0])
            if child is not None and child[0] == block:
                node = child[1]  # 已有共享前缀，仅刷新 LRU
                node.last_access = self._tick()
                continue
            new = _Node()
            node.children[block[0]] = (block, new)
            node = new
            node.last_access = self._tick()
            created += 1
            self._blocks_in_use += 1
        self._evict_to_capacity()
        return created

    @property
    def blocks_in_use(self) -> int:
        return self._blocks_in_use

    # ---------------------------------------------------------------- eviction

    def _evict_to_capacity(self) -> None:
        """叶子优先 LRU：反复摘 last_access 最小的叶子，直到回到容量内。"""
        while self._blocks_in_use > self.num_blocks:
            victim_parent = None
            victim_key = None
            victim_access = None
            stack = [(self.root, None, None)]
            while stack:
                node, parent, key = stack.pop()
                if node is not self.root and node.is_leaf():
                    if victim_access is None or node.last_access < victim_access:
                        victim_parent, victim_key, victim_access = parent, key, node.last_access
                for k, (_, child) in node.children.items():
                    stack.append((child, node, k))
            if victim_parent is None:
                return  # 只剩 root，无可淘汰
            del victim_parent.children[victim_key]
            self._blocks_in_use -= 1


def simulate_hit_rate(
    queries: list[list[int]], num_blocks: int, block_size: int = 16
) -> tuple[float, RadixPrefixCache]:
    """离线回放一组请求，返回 prefix 命中率（命中 token / 总 token）。

    模拟在线流程：先 match（复用 KV，跳过对应 prefill），再 insert 新前缀，
    容量压力下 LRU 淘汰——这是评估"共享 system prompt 能省多少 prefill"的最小模型。
    """
    cache = RadixPrefixCache(num_blocks, block_size)
    total = 0
    hit = 0
    for q in queries:
        hit += cache.match(q)
        total += (len(q) // block_size) * block_size  # 与缓存粒度一致，忽略尾巴
        cache.insert(q)
    return (hit / total if total else 0.0), cache
