"""PagedAttention(vLLM) 的 block-table 语义手写实现。

对应面试题:
    - 字节-AI-Infra-1/2 "PagedAttention 思想/实现与动机"
    - 腾讯-校招-一二面 "Triton 实现 PagedAttention 思路"
    - 快手-一面-2 / 一面-3 "PagedAttention"
    - 京东-校招-一面 "PagedAttention"

核心思想:
    传统: 每个请求按 max_seq_len 连续预留 KV → 内碎片(预留未用) + 外碎片(整理拷贝)
    PagedAttention: KV Cache 切成固定大小 block(如 16 token/block), 逻辑连续
    物理离散, block_table[req][逻辑块号] = 物理 block 槽位 → 近似零浪费,
    显存利用率从 ~20-40% 提到 90%+, 碎片 <4%(vLLM 论文数据)。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class BlockAllocator:
    """物理块分配器: 简化版 vLLM KVCacheManager 的 free-block 栈。"""

    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.float16) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        # 物理块池: (num_blocks, block_size, num_kv_heads, head_dim) * 2(K/V)
        self.k_pool = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free_blocks: list[int] = list(range(num_blocks))

    def allocate(self) -> int:
        if not self.free_blocks:
            raise MemoryError("KV cache 显存耗尽: 触发抢占/换出(对应 vLLM preemption)")
        return self.free_blocks.pop()

    def free(self, block_id: int) -> None:
        self.free_blocks.append(block_id)


class PagedKVCache:
    """每个请求维护 block_table: [逻辑块号] → 物理块号。"""

    def __init__(self, allocator: BlockAllocator) -> None:
        self.alloc = allocator
        self.block_tables: dict[int, list[int]] = {}   # req_id -> 物理块列表
        self.seq_lens: dict[int, int] = {}             # req_id -> 当前长度

    def append_slot(self, req_id: int) -> None:
        """写入第 seq_len 个 token 时: 若正好跨块则先分配新物理块。"""
        bs = self.alloc.block_size
        cur = self.seq_lens.get(req_id, 0)
        if cur % bs == 0:  # 需要新的逻辑块
            self.block_tables.setdefault(req_id, []).append(self.alloc.allocate())
        self.seq_lens[req_id] = cur + 1

    def free_request(self, req_id: int) -> None:
        for b in self.block_tables.pop(req_id, []):
            self.alloc.free(b)
        self.seq_lens.pop(req_id, None)

    def gather_kv(self, req_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按 block_table 把逻辑连续的 KV"组装"出来(gather)。

        真实 vLLM/Triton kernel 不会真的 gather 整段, 而是在 kernel 内
        按 block_table 逐块寻址计算; 这里 gather 是为了展示语义便于对拍。
        """
        blocks = self.block_tables[req_id]
        seq_len = self.seq_lens[req_id]
        k = torch.cat([self.alloc.k_pool[b] for b in blocks], dim=0)[:seq_len]
        v = torch.cat([self.alloc.v_pool[b] for b in blocks], dim=0)[:seq_len]
        return k, v  # (seq_len, num_kv_heads, head_dim)


def paged_attention_decode(
    q: torch.Tensor,            # (num_heads, head_dim) 单 token 的 query
    cache: PagedKVCache,
    req_id: int,
    group_size: int = 1,        # GQA 组大小
) -> torch.Tensor:
    """decode 一步: q 与按 block_table 寻址的 KV 做单 query attention。

    kernel 版本(面经考点)的寻址方式:
        logical_token = i
        block = block_table[i // block_size]
        offset = i % block_size
        k_vec = k_pool[block, offset]     # 一次访存直达物理块
    """
    k, v = cache.gather_kv(req_id)        # (S, n_kv, D)
    q = q.to(k.dtype)                     # 物理池可能为 fp16
    n_kv = k.shape[1]
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
    # (S, H, D) -> (H, S, D)
    k, v = k.permute(1, 0, 2), v.permute(1, 0, 2)
    q = q.unsqueeze(0)                    # (H, 1, D)
    out = F.scaled_dot_product_attention(q, k, v)  # (H, 1, D)
    return out.squeeze(1)


@torch.no_grad()
def paged_attention_demo() -> dict[str, object]:
    """演示: 两个请求交错写入, 物理块离散但逻辑连续, decode 结果与稠密 attention 一致。"""
    torch.manual_seed(0)
    num_kv_heads, head_dim, block_size = 2, 8, 4
    alloc = BlockAllocator(num_blocks=8, block_size=block_size,
                           num_kv_heads=num_kv_heads, head_dim=head_dim)
    cache = PagedKVCache(alloc)

    req_a, req_b = 0, 1
    # 交错追加 6 个 token: A 先 3 个, B 4 个, A 再 3 个 → A 的两个物理块不相邻
    for _ in range(3):
        cache.append_slot(req_a)
    for _ in range(4):
        cache.append_slot(req_b)
    for _ in range(3):
        cache.append_slot(req_a)

    # 把随机 KV 写进物理池, 再校验 gather 后 attention 与稠密版一致
    k_true = torch.randn(6, num_kv_heads, head_dim)
    v_true = torch.randn(6, num_kv_heads, head_dim)
    for i in range(6):
        block = cache.block_tables[req_a][i // block_size]
        alloc.k_pool[block, i % block_size] = k_true[i]
        alloc.v_pool[block, i % block_size] = v_true[i]

    q = torch.randn(num_kv_heads, head_dim)
    got = paged_attention_decode(q, cache, req_a)

    kk, vv = k_true.permute(1, 0, 2), v_true.permute(1, 0, 2)
    # demo 的物理池是 fp16, q 为 fp32 → 参考侧同样转 fp32 对拍
    q32 = q.unsqueeze(0)
    ref = F.scaled_dot_product_attention(q32, kk.float(), vv.float()).squeeze(1)

    return {
        "A_block_table": cache.block_tables[req_a],
        "physical_blocks_discrete": cache.block_tables[req_a][1] != cache.block_tables[req_a][0] + 1,
        "max_abs_err": float((ref - got.float()).abs().max()),
        "allclose": bool(torch.allclose(ref, got.float(), atol=1e-2)),  # fp16 物理池
    }
