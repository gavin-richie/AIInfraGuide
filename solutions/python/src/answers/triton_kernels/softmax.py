"""Triton fused softmax —— 面试"写一个 Triton kernel"的标准考题。

对应面试题:
    - 快手-实习 "CUDA 写 Norm 算子"(同族)
    - 摩尔线程 "纯文本环境手写 CUDA 算子"(Triton 版本)
    - 太初 "手写 softmax 并行化"

性能要点(与文档 02-CUDA 编程题对照):
    - 一行(或几行)由一个 program(CTA) 处理, 行级并行
    - x_block 先 load 一次到 SRAM, max/exp/sum/写回全部在块内完成, 只读一遍 HBM
    - BLOCK_SIZE 向上取 2 的幂; 尾部用 mask 补 -inf
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr, y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float("-inf"))
    # 数值稳定: 块内减 max(分母永远 <= 1, exp 不溢出)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """行级并行 softmax, 等价 torch.softmax(x, dim=-1)。x: (M, N) 连续。"""
    assert x.is_cuda and x.dim() == 2
    x = x.contiguous()
    y = torch.empty_like(x)
    n_cols = x.shape[1]
    # BLOCK_SIZE 取覆盖一行的最小 2 的幂(上限 64K)
    BLOCK_SIZE = max(triton.next_power_of_2(n_cols), 2)
    _softmax_kernel[(x.shape[0],)](x, y, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y
