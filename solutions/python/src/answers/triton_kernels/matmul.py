"""Triton 分块 matmul —— "Triton 如何写一个 GEMM"标准答案。

对应面试题:
    -遂原-一面 "Triton 算子分块、动态分块"
    - 快手/Triton vs CUDA(拼多多)
    - 与 solutions/cuda/src/gemm.cu 的分块思想一一对应

分块策略:
    - 一个 program 计算 C 的一个 (BLOCK_M, BLOCK_N) tile
    - K 维按 BLOCK_K 分步累加: acc += A_tile @ B_tile
    - A_tile/B_tile 经 tl.load 进 SRAM, 访存复杂度从 O(MNK/1) 降到 O(MNK/BLOCK)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak  # (BM, BK)
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn  # (BK, BN)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # FP32 累加防精度损失
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision=INPUT_PRECISION)  # Tensor Core
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor,
                  BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32,
                  allow_tf32: bool = True) -> torch.Tensor:
    """C = A @ B, 等价 torch.matmul(a, b)。a: (M,K) b: (K,N) 2D 连续。

    allow_tf32=True 时 tl.dot 走 TF32 Tensor Core(Ampere+, 快 ~3-8x,
    mantissa 10bit → 相对误差 ~1e-3); 需要严格 FP32 时传 False(ieee)。
    面试考点: 3090/A100 的 FP32 峰值中有一半就是 TF32 贡献。
    """
    assert a.is_cuda and b.is_cuda and a.dim() == 2 and b.dim() == 2
    a, b = a.contiguous(), b.contiguous()
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        INPUT_PRECISION="tf32" if allow_tf32 else "ieee",
    )
    return c
