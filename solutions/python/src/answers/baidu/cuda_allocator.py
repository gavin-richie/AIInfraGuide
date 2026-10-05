"""Caching Allocator 设计演示 —— 百度实习-一面-1 Q6 "设计比 cudaMalloc 更灵活高效的显存分配方案"。

PyTorch CUDACachingAllocator 的核心思想在 CPU 上的可测试复刻:
    1. size-class 空闲链表(请求按 512B 对齐 + round-up 阶梯归档);
    2. free 只挂回链表不还系统, 同尺寸请求 O(1) 命中;
    3. 分裂与相邻合并(buddy 式)对抗碎片;
    4. 统计接口: 水位 / 碎片率 / 命中率 —— 面试画图讲的那些数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Block:
    offset: int                  # 在虚拟"显存"中的偏移
    size: int
    free: bool = True
    owner: str = ""              # 分配者标签(观测用)

    @property
    def end(self) -> int:
        return self.offset + self.size


class SimpleCachingAllocator:
    """单"设备"显存池: 总量 total_bytes, 按 size-class 组织空闲块。

    用法:
        alloc = SimpleCachingAllocator(1 << 20)
        p1 = alloc.allocate(1024, owner="kv_cache")
        alloc.free(p1)                  # 回到缓存, 下次 1024 直接命中
    """

    ALIGN = 512

    def __init__(self, total_bytes: int):
        self.total = total_bytes
        self.blocks: list[_Block] = [_Block(0, total_bytes)]   # 按偏移有序
        self._live: set[int] = set()                           # 在用的块偏移(free 的合法性校验)
        self.stats = {"alloc": 0, "free": 0, "cache_hit": 0, "split": 0, "merge": 0}

    # ---- 对外接口 -------------------------------------------------------
    def allocate(self, nbytes: int, owner: str = "") -> int:
        if nbytes <= 0:
            raise ValueError("nbytes must be positive")
        need = self._round_up(nbytes)
        idx = self._find_fit(need)
        if idx is None:
            raise MemoryError(f"out of device memory: need {need}, "
                              f"cached_free={self.cached_free_bytes()}")
        blk = self.blocks[idx]
        if blk.size >= need + 2 * self.ALIGN:                  # 分裂: 只切出需要的部分
            tail = _Block(blk.offset + need, blk.size - need)
            self.blocks.insert(idx + 1, tail)
            blk.size = need
            self.stats["split"] += 1
        blk.free, blk.owner = False, owner
        self._live.add(blk.offset)
        self.stats["alloc"] += 1
        return blk.offset

    def free(self, offset: int) -> None:
        if offset not in self._live:
            raise ValueError(f"double free or invalid pointer: {offset}")
        idx = self._index_of(offset)
        blk = self.blocks[idx]
        blk.free = True
        blk.owner = ""
        self._live.discard(offset)
        self.stats["free"] += 1
        self._try_merge(idx)

    def empty_cache(self) -> int:
        """把所有空闲块合并后"还给系统"(演示语义: 计数清零)。返回释放字节数。"""
        freed = sum(b.size for b in self.blocks if b.free)
        self.blocks = [b for b in self.blocks if not b.free]
        # 真实现会 unmap 整个 segment; 这里保留已用块并压缩列表
        return freed

    # ---- 观测接口 -------------------------------------------------------
    def cached_free_bytes(self) -> int:
        return sum(b.size for b in self.blocks if b.free)

    def fragmentation_ratio(self) -> float:
        """外部碎片率 = 空闲但不足以服务"最大历史请求档"的比例(简化: 空闲/总量)。"""
        return self.cached_free_bytes() / self.total

    def cache_hit_rate(self) -> float:
        s = self.stats
        return s["cache_hit"] / s["alloc"] if s["alloc"] else 0.0

    # ---- 内部 -----------------------------------------------------------
    def _round_up(self, n: int) -> int:
        return ((n + self.ALIGN - 1) // self.ALIGN) * self.ALIGN

    def _find_fit(self, need: int) -> int | None:
        """first-fit; 命中"大小刚好(同 size-class)"时记 cache_hit。"""
        for i, b in enumerate(self.blocks):
            if b.free and b.size >= need:
                if b.size < need + 2 * self.ALIGN:
                    self.stats["cache_hit"] += 1
                return i
        return None

    def _index_of(self, offset: int) -> int:
        for i, b in enumerate(self.blocks):
            if b.offset == offset:
                return i
        raise KeyError(f"no live block at offset {offset}")

    def _try_merge(self, idx: int) -> None:
        """与物理相邻的空闲块合并(地址有序, 双向看)。"""
        merged = True
        while merged:
            merged = False
            b = self.blocks[idx]
            if idx + 1 < len(self.blocks) and self.blocks[idx + 1].free:
                b.size += self.blocks[idx + 1].size
                del self.blocks[idx + 1]
                self.stats["merge"] += 1
                merged = True
            if idx > 0 and self.blocks[idx - 1].free:
                self.blocks[idx - 1].size += b.size
                del self.blocks[idx]
                idx -= 1
                self.stats["merge"] += 1
                merged = True


def allocator_demo() -> dict[str, object]:
    """分配-释放-缓存复用-分裂-合并的完整小剧场(文档与测试共用)。

    关键情节: p1 释放后因为右邻 p2 仍在用而无法合并, 留在缓存里;
    同尺寸请求 p3 直接命中该块(O(1) 复用, 免走"系统分配")。全部释放后
    相邻合并还原成整段 —— eager merge 与 caching 的分工。
    """
    a = SimpleCachingAllocator(1 << 20)                     # 1 MiB
    p1 = a.allocate(3 * 1024, "kv_cache")                   # 首次: 从整段分裂
    p2 = a.allocate(2 * 1024, "activations")                # 相邻块, 钉住 p1 的右边界
    a.free(p1)                                              # p1 进缓存(p2 在用, 不合并)
    p3 = a.allocate(3 * 1024, "tmp")                        # 同尺寸: 命中缓存且地址复用
    reused = (p3 == p1)
    a.free(p2)
    a.free(p3)                                              # 相邻合并还原整段
    return {
        "stats": dict(a.stats),
        "cached_free": a.cached_free_bytes(),
        "total": a.total,
        "all_free": a.cached_free_bytes() == a.total,
        "addr_reused": reused,
    }
