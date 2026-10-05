"""Triton fused LayerNorm / RMSNorm —— "CUDA 写 Norm 算子"(快手-实习)的 Triton 版。

对应面试题:
    - 快手-实习 "CUDA 写 Norm 算子"
    - 百度-实习-一面-1 "RMSNorm 为何比 LayerNorm 快"
    - 遂原 "Triton 算子分块"

融合点: mean/var/rsqrt/缩放/平移全部在一个 kernel 内完成, 中间结果不出 SRAM,
省掉 PyTorch 串行版 4-5 次 HBM 往返。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    y = diff * rstd * w + b
    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """等价 F.layer_norm(x, (N,), weight, bias, eps)。x: (M, N)。"""
    assert x.is_cuda and x.dim() == 2
    x = x.contiguous()
    y = torch.empty_like(x)
    n_cols = x.shape[1]
    BLOCK_SIZE = max(triton.next_power_of_2(n_cols), 2)
    _layernorm_kernel[(x.shape[0],)](x, y, weight, bias, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def _rmsnorm_kernel(
    x_ptr, y_ptr, w_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: 无 mean 平移, 只除 RMS = sqrt(mean(x^2)+eps) —— 比 LN 少一次均值计算与减法
    ms = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(ms + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0)
    y = x * rstd * w
    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm(Llama 系): y = x / sqrt(mean(x^2)+eps) * w。"""
    assert x.is_cuda and x.dim() == 2
    x = x.contiguous()
    y = torch.empty_like(x)
    n_cols = x.shape[1]
    BLOCK_SIZE = max(triton.next_power_of_2(n_cols), 2)
    _rmsnorm_kernel[(x.shape[0],)](x, y, weight, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y
