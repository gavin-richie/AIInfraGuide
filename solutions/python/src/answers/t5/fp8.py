"""FP8 (E4M3 / E5M2) 软件模拟：编码、解码与 W8A8 GEMM 精度对比。

对应面试题:
    - 科大讯飞-飞星 "FP8 的两种格式（E4M3 与 E5M2）有什么区别？各自的计算方式是什么？"
    - 科大讯飞-飞星 "NVIDIA GPU 对 FP8 数据类型有哪些专门的硬件优化？"
    - 小厂-实习-2 "DeepSeek 论文中令你印象深刻的技术要点（FP8 训练方案）"

格式定义（OCP FP8 标准，与 torch.float8_e4m3fn / e5m2 一致）:

    E4M3: 1 符号 + 4 指数(bias=7) + 3 尾数
          - 指数全 1 且尾数全 1 → NaN（无 Inf，溢出饱和）
          - 最大规格化数 2^8 × 1.75 = 448，最小次正规数 2^-9
    E5M2: 1 符号 + 5 指数(bias=15) + 2 尾数
          - 指数全 1 → ±Inf（尾数 0）/ NaN（尾数 ≠ 0）
          - 最大规格化数 2^15 × 1.75 = 57344，最小次正规数 2^-16

计算方式（硬件视角）:
    FP8 本身只做"存储/传输格式"，GEMM 时 Tensor Core 把两个 FP8 矩阵
    相乘、**累加到 FP16/FP32** 的寄存器堆里（accumulate 必须高精度），
    再按 per-tensor/per-block scaling 反缩放回高动态范围域。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FP8Format:
    name: str
    ebits: int
    mbits: int
    bias: int
    max_finite: float
    has_inf: bool


E4M3 = FP8Format("e4m3", ebits=4, mbits=3, bias=7, max_finite=448.0, has_inf=False)
E5M2 = FP8Format("e5m2", ebits=5, mbits=2, bias=15, max_finite=57344.0, has_inf=True)


def _round_half_even(t: float) -> int:
    """四舍五入到偶数（银行家舍入），与硬件 RNE 一致。"""
    f = math.floor(t)
    frac = t - f
    if frac > 0.5:
        return f + 1
    if frac < 0.5:
        return f
    return f + 1 if f % 2 else f  # 恰为 .5 时取偶


def fp8_quantize(x: float, fmt: FP8Format) -> float:
    """把 float 量化到 FP8 网格并返回可表示的值（饱和溢出）。

    规格化数: (-1)^s × 2^(e-bias) × 1.m
    次正规数: (-1)^s × 2^(1-bias) × 0.m   （e=0，填补 0 附近梯度）
    """
    sign = -1.0 if x < 0 else 1.0
    ax = abs(x)
    min_normal = 2.0 ** (1 - fmt.bias)

    if ax >= fmt.max_finite:  # 溢出饱和（E4M3 无 Inf；E5M2 语义上到 Inf，这里按饱和模拟）
        return sign * fmt.max_finite

    if ax < min_normal:
        # 次正规区间：均匀网格，步长 2^(1-bias-mbits)
        step = 2.0 ** (1 - fmt.bias - fmt.mbits)
        q = _round_half_even(ax / step) * step
        return sign * q if q < min_normal else sign * min_normal

    # 规格化区间：保留 mbits 位尾数（RNE）
    m, e = math.frexp(ax)          # ax = m × 2^e, m ∈ [0.5, 1)
    sig = m * 2.0                   # 尾数有效位 1.xxx ∈ [1, 2), 值 = sig × 2^(e-1)
    scale = 2.0**fmt.mbits
    q_sig = _round_half_even(sig * scale) / scale
    if q_sig >= 2.0:                # 尾数进位 1.111→10.000，指数 +1
        q_sig = 1.0
        e += 1
    val = q_sig * 2.0 ** (e - 1)
    return sign * (min(val, fmt.max_finite))


def fp8_bits(x: float, fmt: FP8Format) -> int:
    """返回量化后值的 uint8 位模式: sign<<7 | exp<<mbits | mantissa（用于已知值测试）。"""
    q = fp8_quantize(x, fmt)
    sign = 1 if q < 0 else 0
    ax = abs(q)
    min_normal = 2.0 ** (1 - fmt.bias)
    if ax == 0.0:
        return sign << 7
    if ax < min_normal:  # 次正规
        step = 2.0 ** (1 - fmt.bias - fmt.mbits)
        mant = int(round(ax / step))
        return sign << 7 | mant
    m, e = math.frexp(ax)
    mant = int(round((m * 2.0 - 1.0) * 2.0**fmt.mbits))
    exp_field = (e - 1) + fmt.bias
    return sign << 7 | exp_field << fmt.mbits | mant


def fp8_quantize_tensor(t: torch.Tensor, fmt: FP8Format) -> torch.Tensor:
    """批量量化（逐元素，CPU 上跑的小规模模拟）。"""
    out = torch.empty_like(t, dtype=torch.float32)
    flat_out = out.view(-1)
    for i, v in enumerate(t.view(-1).tolist()):
        flat_out[i] = fp8_quantize(v, fmt)
    return out


def fp8_gemm_sim(a: torch.Tensor, b: torch.Tensor, fmt: FP8Format,
                 per_tensor_scale: bool = True) -> torch.Tensor:
    """模拟 W8A8 FP8 GEMM：A/B 量化到 FP8 → FP32 累加（Tensor Core 语义）。

    per_tensor_scale=True 时按 amax/max_finite 缩放（DeepSeek 训练用更细的
    1×128 block scaling，这里用 per-tensor 演示原理）。
    """
    if per_tensor_scale:
        sa = a.abs().amax() / fmt.max_finite if a.numel() else 1.0
        sb = b.abs().amax() / fmt.max_finite if b.numel() else 1.0
        a_q = fp8_quantize_tensor(a / max(float(sa), 1e-12), fmt) * float(sa)
        b_q = fp8_quantize_tensor(b / max(float(sb), 1e-12), fmt) * float(sb)
    else:
        a_q, b_q = fp8_quantize_tensor(a, fmt), fp8_quantize_tensor(b, fmt)
    return a_q.to(torch.float64) @ b_q.to(torch.float64)


def format_summary() -> str:
    """两种格式的一览（面试口答模板）。"""
    return (
        f"E4M3: bias={E4M3.bias}, 动态范围 ±448, 3 位尾数(相对精度 2^-4≈6.25%), "
        f"无 Inf/饱和溢出 → 前向权重/激活/KV Cache\n"
        f"E5M2: bias={E5M2.bias}, 动态范围 ±57344, 2 位尾数(相对精度 2^-3≈12.5%), "
        f"有 Inf/NaN → 梯度等大动态范围场景"
    )
