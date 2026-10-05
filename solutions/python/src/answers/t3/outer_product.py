"""向量外积的语义参考与 CUDA 实现要点。

来源: 英伟达-AI-Infra-校招-2（编程 11）。C[i,j] = a[i] * b[j]，
计算本身只有一次乘法，考点全在访存:
1. b[j] 对同一 warp 内所有行是共享的（可放 constant memory 或 smem 广播）;
2. 输出按行连续写出（合并访存），a[i] 每线程读一次（warp 内同值——广播不额外耗带宽）;
3. 形状 (M,N) 用二维 grid/block 映射, 边界 mask。
CUDA kernel 内嵌在 docs/interview-answers/80-t3-英伟达与三星.md。
"""
from __future__ import annotations

import torch


def outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch 广播语义参考实现, 等价于 torch.outer / torch.einsum("i,j->ij")。

    >>> outer_product(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
    tensor([[ 3.,  4.,  5.],
            [ 6.,  8., 10.]])
    """
    if a.dim() != 1 or b.dim() != 1:
        raise ValueError("outer_product expects 1-D tensors")
    return a[:, None] * b[None, :]
