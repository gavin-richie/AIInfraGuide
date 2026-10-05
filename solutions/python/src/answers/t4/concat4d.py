"""四维张量 Concat（元戎启行：描述各维拼接方式 + CUDA kernel 实现）。

对应面试题:
    - 元戎启行-校招-一面-2 编程题 1/2:
      "实现四维张量的 Concat，分别描述在每个维度上进行拼接的处理方式"
      "使用 CUDA 编写四维张量的 Concat Kernel，针对不同维度分别实现"

四个维度拼接的访存特征（面试按此分 case 回答）:
    dim=3 (最内维 W):   每行 (c,h) 对应输出一段 [a_row | b_row | ...]，段内连续。
                        kernel: 一个线程管 out 的一个元素，按 w 判断落在哪个输入
                        （w < w0 选 A，w < w0+w1 选 B，...），源地址 = 行首 + (w - 偏移)。
    dim=2 (H):          每个输出行整行来自同一个输入，输入行在 (n,c) 相同、h 错位 ——
                        退化为"整行 memcpy"，用 2D copy 循环即可，线程按 (row, w) 划分。
    dim=1 (C):          每个 (n,h,w) 列上拼接：以 (n,h) 为单位做"条带 memcpy"，
                        条带长度 = C_i*H*W，块内连续 —— 比 dim=2 粒度更粗、更接近一维拷贝。
    dim=0 (N):          整块连续拷贝，等价于把各输入按顺序 memcpy 到输出 offsets 处，
                        完全合并访存，是最快的情况。

通用索引公式（CUDA kernel 的核心，见 concat4d_manual）:
    out 线性下标 idx 反解 (n,c,h,w)，沿拼接维比较坐标落在哪个输入：
        src_tensor = 第一个满足 coord < 累计偏移 offset_i 的输入
        src_coord  = coord - offset_i（其余维坐标不变）
    再按该输入的 stride 折回线性下标。
"""
from __future__ import annotations

import torch


def concat4d_manual(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    """纯 Python 逐元素版：完整复刻 CUDA kernel 的通用索引推导。

    仅用于小尺寸测试对拍（验证索引公式），大张量请用 concat4d_fast / torch.cat。
    要求所有输入 4D、非拼接维相同、内存连续（contiguous）。
    """
    assert len(tensors) >= 1 and all(t.dim() == 4 for t in tensors)
    dim = dim % 4
    shapes = [t.shape for t in tensors]
    for i in range(4):
        if i != dim:
            assert {s[i] for s in shapes} == {shapes[0][i]}, "非拼接维必须一致"
    out_shape = list(shapes[0])
    out_shape[dim] = sum(s[dim] for s in shapes)
    out = torch.empty(out_shape, dtype=tensors[0].dtype)
    offs = [0]
    for s in shapes:
        offs.append(offs[-1] + s[dim])

    N, C, H, W = out_shape
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    coord = (n, c, h, w)
                    cd = coord[dim]
                    k = 0
                    while k < len(tensors) and not (offs[k] <= cd < offs[k + 1]):
                        k += 1
                    src = tensors[k]
                    scoord = list(coord)
                    scoord[dim] = cd - offs[k]
                    sn, sc, sh, sw = scoord
                    out[n, c, h, w] = src[sn, sc, sh, sw]
    return out


def concat4d_fast(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
    """向量化版：同一个索引公式，用 torch 算术一次性算出所有元素的源下标（gather）。

    等价于 CUDA kernel 的 grid-stride 循环体。
    """
    assert len(tensors) >= 1 and all(t.dim() == 4 for t in tensors)
    dim = dim % 4
    shapes = [t.shape for t in tensors]
    out_shape = list(shapes[0])
    out_shape[dim] = sum(s[dim] for s in shapes)
    device, dtype = tensors[0].device, tensors[0].dtype
    N, C, H, W = out_shape

    # 输出各维坐标网格（flatten 成一维长度 ntot 的向量）
    n_idx = torch.arange(N, device=device)[:, None, None, None]
    c_idx = torch.arange(C, device=device)[None, :, None, None]
    h_idx = torch.arange(H, device=device)[None, None, :, None]
    w_idx = torch.arange(W, device=device)[None, None, None, :]
    grids = torch.broadcast_tensors(n_idx, c_idx, h_idx, w_idx)  # 4 个 (N,C,H,W)

    out = torch.empty(out_shape, dtype=dtype, device=device)
    offs = [0]
    for s in shapes:
        offs.append(offs[-1] + s[dim])
    done = torch.zeros(out_shape, dtype=torch.bool, device=device)
    for k, t in enumerate(tensors):
        cd = grids[dim]
        hit = (cd >= offs[k]) & (cd < offs[k + 1])
        hit &= ~done  # 输入区间互不重叠，保险起见再掩一次
        scoord = list(grids)
        scoord[dim] = cd - offs[k]
        # 按该输入的 stride 折线性下标（用向量化的 4 维下标直接 advanced indexing 更直观）
        out[hit] = t[scoord[0][hit], scoord[1][hit], scoord[2][hit], scoord[3][hit]]
        done |= hit
    return out


# ---------------------------------------------------------------------------
# Triton kernel：dim=3（最内维）场景，one program 处理一个 (n,c,h) 行块
# ---------------------------------------------------------------------------
try:  # triton 为可选依赖，缺失时以下符号不可用（测试会 importorskip）
    import triton
    import triton.language as tl

    @triton.jit
    def _concat_last_kernel(
        a_ptr, b_ptr, out_ptr,
        w_a, w_b,                       # 两输入的 W（last dim 长度）
        BLOCK_W: tl.constexpr,          # 每次处理的元素数
    ):
        # grid = (N*C*H, cdiv(w_a + w_b, BLOCK_W))：行号 = program_id(0)
        row = tl.program_id(0)
        blk = tl.program_id(1)
        w = blk * BLOCK_W + tl.arange(0, BLOCK_W)
        total = w_a + w_b
        mask = w < total
        # w < w_a 的元素来自 A（源列 = w），其余来自 B（源列 = w - w_a）
        from_a = w < w_a
        src_w = tl.where(from_a, w, w - w_a)
        base_out = row.to(tl.int64) * total
        base_in = row.to(tl.int64) * tl.where(from_a, w_a, w_b)
        vals_a = tl.load(a_ptr + base_in + src_w, mask=mask & from_a, other=0)
        vals_b = tl.load(b_ptr + base_in + src_w, mask=mask & ~from_a, other=0)
        tl.store(out_ptr + base_out + w, tl.where(from_a, vals_a, vals_b), mask=mask)

    def triton_concat_last_dim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Triton 版 concat(dim=-1)：行内两段连续拷贝，完全合并访存。"""
        assert a.dim() == 4 and b.shape[:-1] == a.shape[:-1]
        n, c, h, wa = a.shape
        wb = b.shape[-1]
        out = torch.empty(n, c, h, wa + wb, device=a.device, dtype=a.dtype)
        BLOCK_W = 1024
        grid = (n * c * h, triton.cdiv(wa + wb, BLOCK_W))
        _concat_last_kernel[grid](a, b, out, wa, wb, BLOCK_W=BLOCK_W)
        return out

except ImportError:  # pragma: no cover - 无 triton 环境跳过
    pass
