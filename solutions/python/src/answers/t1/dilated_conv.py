"""手写膨胀（空洞）卷积 Dilated Convolution。

来源: 小米-AI-Infra-实习-一面-1 编程题 5（实现 + 讨论优化方案）。

dilated_conv2d_naive   —— 六重循环直接实现，讲清"膨胀 = 卷积核插空"语义
dilated_conv2d_unfold  —— im2col / F.unfold 展开成矩阵乘，工程实现的主流思路
（与 answers/algorithms 里 LeetCode 题不同，这是算子题，与 solutions/cuda 的
conv 类 kernel 同族，PyTorch 版便于对拍验证。）

面试讨论点（代码可验证）：
1. 语义：kernel 尺寸 k、膨胀率 d 的有效感受野 = k + (k-1)(d-1)；
   膨胀不增加参数量/计算量，只扩大感受野（DeepLab/WaveNet 的做法）；
2. 优化：im2col+GEMM（cuDNN/TRT 的老路线）、Winograd、FFT；
   GPU 上膨胀卷积 gather 地址不连续 → 比 dense conv 的合并访存差，
   常把 (H,W) 展平后按 output tile 分 block，索引换算预计算；
3. 对拍：两种实现都应与 F.conv2d(dilation=d) 一致。
"""

import torch
import torch.nn.functional as F


def dilated_conv2d_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    dilation: int = 1,
    stride: int = 1,
    pad: int = 0,
) -> torch.Tensor:
    """直接循环实现。x: (N, C_in, H, W)，weight: (C_out, C_in, kH, kW)。"""
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    h_out = (h + 2 * pad - dilation * (kh - 1) - 1) // stride + 1
    w_out = (w + 2 * pad - dilation * (kw - 1) - 1) // stride + 1
    if pad:
        x = F.pad(x, (pad, pad, pad, pad))
    out = x.new_zeros(n, c_out, h_out, w_out)
    for b in range(n):
        for o in range(c_out):
            for oy in range(h_out):
                for ox in range(w_out):
                    acc = torch.zeros((), dtype=x.dtype)
                    for ci in range(c_in):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = oy * stride + ky * dilation
                                ix = ox * stride + kx * dilation
                                acc += x[b, ci, iy, ix] * weight[o, ci, ky, kx]
                    out[b, o, oy, ox] = acc + (bias[o] if bias is not None else 0)
    return out


def dilated_conv2d_unfold(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    dilation: int = 1,
    stride: int = 1,
    pad: int = 0,
) -> torch.Tensor:
    """im2col 路线：unfold 把每个输出位置的感受野拉成一行，卷积变 GEMM。

    x: (N, C_in, H, W) -> cols: (N, C_in*kH*kW, L) -> out: (N, C_out, L)
    """
    c_out = weight.shape[0]
    cols = F.unfold(x, weight.shape[-2:], dilation=dilation, padding=pad, stride=stride)
    n, _, length = cols.shape
    weight_mat = weight.reshape(c_out, -1)  # (C_out, C_in*kH*kW)
    out = weight_mat @ cols  # (N, C_out, L) 广播到 batch
    if bias is not None:
        out = out + bias.view(1, c_out, 1)
    h_out = (x.shape[2] + 2 * pad - dilation * (weight.shape[2] - 1) - 1) // stride + 1
    w_out = (x.shape[3] + 2 * pad - dilation * (weight.shape[3] - 1) - 1) // stride + 1
    return out.reshape(n, c_out, h_out, w_out)


def effective_kernel_size(kernel_size: int, dilation: int) -> int:
    """膨胀后的等效核尺寸：k + (k-1)(d-1)。"""
    return kernel_size + (kernel_size - 1) * (dilation - 1)


def receptive_field_two_layers(k: int, d1: int, d2: int) -> int:
    """两层 dilation=d1/d2 的 kxk 堆叠感受野（常见追问：1,2 / 1,1,2,4 ...）。

    逐层公式（stride=1）：r = 1 + Σ_l (k-1)·d_l。单层 d=1 时回到 k。
    """
    r = 1
    for d in (d1, d2):
        r += (k - 1) * d
    return r
