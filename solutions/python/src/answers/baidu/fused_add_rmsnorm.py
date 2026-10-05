"""Triton fused (residual add + RMSNorm) —— 百度实习-一面-3 Q20 "CUDA LayerNorm Kernel" 的进阶变体。

对应面试题:
    - 百度-AI-Infra-实习-一面-3 Q15/Q20 "RMSNorm 的具体实现 / 手写 CUDA LayerNorm"
    - 百度-AI-Infra-实习-一面-1 Q3 "RMSNorm 相比 LayerNorm 性能提升的原因"

融合价值: x + residual 的读取、RMS 归约、缩放写出合成一个 kernel,
中间结果不出 SRAM —— 相比 "add -> rmsnorm" 两个 kernel 省一次 HBM 往返。
大 N(隐藏维超过单 block 装载上限)时用列块循环版(两遍扫: 先累加 sumsq 再归一化)。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr, res_ptr, out_ptr, resout_ptr, w_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """out = rmsnorm(x + res); resout = x + res(供下游残差链继续用, vLLM 同款语义)。"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    base = row * n_cols
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + r                                               # fp32 累加防溢出

    ms = tl.sum(h * h, axis=0) / n_cols                     # mean(h^2), 比 LN 少一次均值
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0)
    y = h * rstd * w

    tl.store(out_ptr + base + cols, y, mask=mask)
    tl.store(resout_ptr + base + cols, h, mask=mask)


def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (norm_out, new_residual), 均与输入同形。x/residual: (M, N)。"""
    assert x.is_cuda and x.dim() == 2 and x.shape == residual.shape
    x, residual = x.contiguous(), residual.contiguous()
    out = torch.empty_like(x)
    new_res = torch.empty_like(x)
    n_cols = x.shape[1]
    BLOCK_SIZE = max(triton.next_power_of_2(n_cols), 2)
    _fused_add_rmsnorm_kernel[(x.shape[0],)](
        x, residual, out, new_res, weight, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out, new_res


@triton.jit
def _fused_add_rmsnorm_looped_kernel(
    x_ptr, res_ptr, out_ptr, resout_ptr, w_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """大 N 版: 一行一个 program, 列块循环两遍扫 —— pass1 累加 sumsq, pass2 归一化写出。

    处理 N >> BLOCK_SIZE(如 d=8192 而 block 只开 1024)的隐藏维,
    对应面试追问"隐藏维大于一个 block 能装的范围怎么办"。
    """
    row = tl.program_id(0)
    base = row * n_cols

    acc = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        h = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32) \
            + tl.load(res_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(resout_ptr + base + cols, h, mask=mask)    # resout 先行物化
        acc += tl.sum(h * h, axis=0)

    rstd = 1.0 / tl.sqrt(acc / n_cols + eps)
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        h = tl.load(resout_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0)
        tl.store(out_ptr + base + cols, h * rstd * w, mask=mask)


def fused_add_rmsnorm_looped(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    eps: float = 1e-6, block: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda and x.dim() == 2 and x.shape == residual.shape
    x, residual = x.contiguous(), residual.contiguous()
    out = torch.empty_like(x)
    new_res = torch.empty_like(x)
    _fused_add_rmsnorm_looped_kernel[(x.shape[0],)](
        x, residual, out, new_res, weight, x.shape[1], eps, BLOCK_SIZE=block,
    )
    return out, new_res


def torch_reference(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                    eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    h = x + residual
    ms = h.float().pow(2).mean(-1, keepdim=True)
    y = (h.float() * torch.rsqrt(ms + eps)).to(x.dtype) * weight
    return y, h
