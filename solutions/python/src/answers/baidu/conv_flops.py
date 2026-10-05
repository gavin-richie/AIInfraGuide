"""Conv2D 参数量 / FLOPs 计算 —— 百度实习-一面-3 Q17、校招-一面-2 Q5。

对应面试题:
    - "给定 Conv2D 输入 [C=64, W=64, H=128]、卷积核 3x3、输出 [C=128, W=64, H=128],
       计算参数量与 FLOPs"(注意题面 H/W 口径, 按输出 64x128x128 计)
    - "给定一个卷积操作场景, 计算其时间复杂度"(同公式, 再除峰值 FLOPS 估时)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Conv2DSpec:
    c_in: int
    c_out: int
    k: int                      # 方形核 k x k
    h_in: int
    w_in: int
    stride: int = 1
    pad: int = 0
    groups: int = 1             # groups=c_in 即 depthwise
    bias: bool = True


@dataclass(frozen=True)
class Conv2DCost:
    params: int
    flops: int                  # 乘加对计 2
    h_out: int
    w_out: int


def conv2d_params_flops(s: Conv2DSpec) -> Conv2DCost:
    """标准卷积: params = k*k*(c_in/groups)*c_out + bias;
    FLOPs = h_out*w_out*c_out * (k*k*c_in/groups) * 2 (乘加对计 2, MAC 才计 1)。
    """
    h_out = (s.h_in + 2 * s.pad - s.k) // s.stride + 1
    w_out = (s.w_in + 2 * s.pad - s.k) // s.stride + 1
    assert h_out > 0 and w_out > 0, "invalid geometry"
    macs_per_out = s.k * s.k * (s.c_in // s.groups)
    params = macs_per_out * s.c_out + (s.c_out if s.bias else 0)
    flops = h_out * w_out * s.c_out * macs_per_out * 2
    return Conv2DCost(params=params, flops=flops, h_out=h_out, w_out=w_out)


def baidu_interview_case() -> dict[str, int]:
    """实习-一面-3 Q17 的原题数值: 3x3, C_in=64 -> C_out=128, 输出与输入同尺寸(128x64)。

    题面给的是输出尺寸, 故卷积为 same 填充(pad=1, stride=1)。
    """
    s = Conv2DSpec(c_in=64, c_out=128, k=3, h_in=128, w_in=64, pad=1)
    c = conv2d_params_flops(s)
    return {"params": c.params, "flops": c.flops,
            "flops_giga": c.flops // 1_000_000_000, "h_out": c.h_out, "w_out": c.w_out}


def depthwise_vs_standard(c: int = 64, k: int = 3, hw: int = 64) -> dict[str, int]:
    """DW + PW(逐通道滤波 + 1x1 通道混合)对比标准卷积 —— MobileNet 块的计算量账。"""
    std = conv2d_params_flops(Conv2DSpec(c, c, k, hw, hw))
    dw = conv2d_params_flops(Conv2DSpec(c, c, k, hw, hw, groups=c))     # k*k*c
    pw = conv2d_params_flops(Conv2DSpec(c, c, 1, hw, hw))               # c*c
    return {"standard": std.flops, "dw": dw.flops, "pw": pw.flops,
            "dw_pw_total": dw.flops + pw.flops,
            "saving_ratio": round(std.flops / (dw.flops + pw.flops), 2)}
