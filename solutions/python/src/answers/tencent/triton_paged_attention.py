"""Triton 实现 PagedAttention(decode 阶段)。

来源: 腾讯-AI-Infra-校招-一二面-1(二面 2)。思路讲解见
docs/interview-answers/40-腾讯-推理与cuda.md Q17。

数据布局(对齐 vLLM):
- q:        [num_seqs, num_q_heads, head_dim]              (本轮 token 的 query)
- k_cache:  [num_blocks, block_size, num_kv_heads, head_dim] (物理块池)
- v_cache:  同 k_cache
- block_tables: [num_seqs, max_num_blocks_per_seq] int32   (逻辑块 -> 物理块)
- seq_lens: [num_seqs] int32

kernel: grid = (num_seqs, num_q_heads), 每个程序处理一个 (序列, query head):
1. kv_head = q_head // group_size (GQA 映射);
2. 载入 q 向量到寄存器, 逐 KV 块间接寻址(block table)载入 K/V tile;
3. online softmax(m/l 重缩放)累加, 尾块 mask 掉越界位置;
4. out = acc / l 写回。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr, k_ptr, v_ptr,
    block_tables_ptr, seq_lens_ptr, out_ptr,
    sm_scale,
    group_size,
    stride_qs, stride_qh,
    stride_kz, stride_kn, stride_kh,          # k_cache: [blocks, block_size, kv_heads, dim]
    stride_vz, stride_vn, stride_vh,
    stride_bts,
    stride_os, stride_oh,
    BLOCK_SIZE: tl.constexpr,                  # 必须等于 block_size(2 的幂)
    HEAD_DIM: tl.constexpr,                    # 2 的幂
):
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    kvh = qh // group_size

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + seq * stride_qs + qh * stride_qh + offs_d).to(tl.float32)  # [D]

    seq_len = tl.load(seq_lens_ptr + seq)
    m = tl.full([], float("-inf"), tl.float32)   # running max
    l = tl.full([], 0.0, tl.float32)             # softmax 分母
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    local_n = tl.arange(0, BLOCK_SIZE)           # 块内局部偏移(物理寻址)

    for nb in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        block_id = tl.load(block_tables_ptr + seq * stride_bts + nb)
        offs_n = nb * BLOCK_SIZE + local_n       # 全局 token 位置(仅用于 mask)
        mask_n = offs_n < seq_len

        k = tl.load(
            k_ptr + block_id * stride_kz
            + local_n[:, None] * stride_kn + kvh * stride_kh + offs_d[None, :],
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)                                    # [BLOCK, D]
        qk = tl.sum(q[None, :] * k, 1) * sm_scale           # [BLOCK]
        qk = tl.where(mask_n, qk, -float("inf"))

        m_new = tl.maximum(m, tl.max(qk, 0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(qk - m_new)                              # 越界位置 exp(-inf)=0
        l = l * alpha + tl.sum(p, 0)

        v = tl.load(
            v_ptr + block_id * stride_vz
            + local_n[:, None] * stride_vn + kvh * stride_vh + offs_d[None, :],
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(v * p[:, None], 0)
        m = m_new

    acc = acc / l
    tl.store(out_ptr + seq * stride_os + qh * stride_oh + offs_d,
             acc.to(out_ptr.dtype.element_ty))


def triton_paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """PagedAttention decode. 支持 MHA(group_size=1) 与 GQA。

    要求: block_size 为 2 的幂(16/32/...), head_dim 为 2 的幂(64/128/...),
    所有张量在 cuda 上且缓存连续。
    """
    assert q.is_cuda and k_cache.is_cuda
    num_seqs, num_q_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, dim2 = k_cache.shape
    assert head_dim == dim2 and head_dim in (16, 32, 64, 128, 256), head_dim
    assert num_q_heads % num_kv_heads == 0
    group_size = num_q_heads // num_kv_heads
    assert (block_size & (block_size - 1)) == 0, "block_size 需为 2 的幂"
    assert k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
    assert block_tables.dtype == torch.int32 and block_tables.shape[0] == num_seqs
    assert seq_lens.shape[0] == num_seqs

    sm_scale = 1.0 / (head_dim ** 0.5) if sm_scale is None else sm_scale
    out = torch.empty_like(q)
    grid = (num_seqs, num_q_heads)
    _paged_decode_kernel[grid](
        q, k_cache, v_cache, block_tables, seq_lens.to(torch.int32), out,
        sm_scale, group_size,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
        num_warps=max(1, min(4, head_dim // 32)),
    )
    return out


def paged_attention_torch_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """CPU/GPU 通用参考实现: 按 block table gather 出逻辑连续的 KV 再算标准 attention。"""
    num_seqs, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    group = num_q_heads // num_kv_heads
    sm_scale = 1.0 / (head_dim ** 0.5) if sm_scale is None else sm_scale
    out = torch.empty_like(q)
    for s in range(num_seqs):
        length = int(seq_lens[s])
        n_blocks = (length + block_size - 1) // block_size
        blocks = block_tables[s, :n_blocks]
        # [n_blocks*block_size, kv_heads, dim] 截断到 length
        ks = k_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:length]
        vs = v_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:length]
        for h in range(num_q_heads):
            kvh = h // group
            k = ks[:, kvh].float()                       # [len, dim]
            v = vs[:, kvh].float()
            logits = (q[s, h].float() @ k.T) * sm_scale   # [len]
            p = torch.softmax(logits, dim=-1)
            out[s, h] = (p @ v).to(out.dtype)
    return out


def paged_attention_demo() -> dict[str, object]:
    """构造随机 paged KV(物理块乱序分配), 验证 Triton kernel 与参考实现一致。"""
    torch.manual_seed(0)
    num_seqs, num_q_heads, num_kv_heads, head_dim, block_size = 8, 32, 8, 64, 16
    max_len = 64
    k_cache = torch.randn(256, block_size, num_kv_heads, head_dim,
                          device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(num_seqs, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    seq_lens = torch.randint(1, max_len + 1, (num_seqs,), device="cuda", dtype=torch.int32)
    max_blocks = (max_len + block_size - 1) // block_size
    block_tables = torch.randint(0, 256, (num_seqs, max_blocks),
                                 device="cuda", dtype=torch.int32)
    got = triton_paged_attention_decode(q, k_cache, v_cache, block_tables, seq_lens)
    ref = paged_attention_torch_ref(q, k_cache, v_cache, block_tables, seq_lens)
    max_err = (got - ref).abs().max().item()
    return {"max_err": max_err, "allclose": bool(torch.allclose(got, ref, atol=1e-2)),
            "seq_lens": seq_lens.tolist()}


if __name__ == "__main__":
    print(paged_attention_demo())
