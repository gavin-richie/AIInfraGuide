"""低比特 GEMM: W8A8 / W4A16 fake-quant 与误差分析。

对应面试题:
    - 商汤-一面 11: 是否有低比特位 GEMM 的开发经验
    - 智谱-实习-一面 基础 1/5/6: 量化粒度(W8 per-channel / W4 per-group)、FP8 vs FP4 场景

面试口径(kernel 视角):
    W8A8 : 权重 per-channel INT8 + 激活 per-tensor(或 per-token) INT8,
           INT8 IMMA(mma.m16n8k16.s8)一次算 2048 MAC, 累加用 INT32,
           输出乘 scale_w[c] * scale_a 反量化 —— 计算密度比 FP16 HMMA 翻倍;
    W4A16: 仅权重量化(激活保持 FP16), INT4 权重先在 kernel 内反量化成 FP16 再
           做 HMMA(dequant-in-kernel, Marlin 风格), 收益全在带宽/显存(decode
           memory bound), 而不是算力;
    误差规律(高斯随机权重, 实测量化误差是"信噪比"概念):
        W8A8 per-channel/per-tensor ≈ 1.4e-2 —— 每个 INT8 操作数自身相对噪声
        ~0.9%(absmax≈4σ 时 step=8σ/255, 噪声 std=step/sqrt(12)), 点积按
        sqrt(eps_a^2+eps_w^2) 合成;
        W4A16 group=128(沿 K 维) ≈ 1.2e-1 —— 16 级表示对无结构高斯权重是大
        失真(真实 LLM 权重经 AWQ 等缩放迁移后好得多, 且 decode 场景误差影响
        小于任务感知); group 64 比 128 再降约 1 个点(粒度更细单调有效)。
"""
from __future__ import annotations

import torch

from answers.t2.calibration import compute_scale, fake_quant

__all__ = ["w8a8_matmul", "w4a16_matmul", "quantized_gemm_report"]


def w8a8_matmul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """INT8×INT8 GEMM 的伪量化模拟: y = (a_q · w_q^T) * s_a * s_w[c]。

    a: (M, K) 激活, per-tensor 对称 INT8;
    w: (N, K) 权重, per-channel 对称 INT8(NNLinear 的 weight 布局);
    真实 kernel 中 a_q@w_q.T 在 INT32 累加, 这里用 int32 乘加模拟, 再一次反量化。
    """
    m, k = a.shape
    n = w.shape[0]
    s_a = compute_scale(a, "tensor", num_bits=8)                 # 标量
    s_w = compute_scale(w, "channel", num_bits=8)                # (N, 1)
    a_q = torch.clamp(torch.round(a / s_a), -128, 127).to(torch.int32)
    w_q = torch.clamp(torch.round(w / s_w), -128, 127).to(torch.int32)
    acc = a_q @ w_q.T                                            # (M, N) int32
    return acc.to(torch.float32) * s_a * s_w.view(1, n)


def w4a16_matmul(a: torch.Tensor, w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """W4A16 weight-only GEMM 的伪量化模拟: 权重 per-group INT4, 激活保持 FP16。

    y = a @ (w_q * s_g)^T —— 等价于 kernel 内"边反量化边乘"(dequant-in-kernel)。
    """
    n, k = w.shape
    s_g = compute_scale(w, "group", num_bits=4, group_size=group_size)   # (N, 1)
    w_q = torch.clamp(torch.round(w / s_g), -8, 7)                       # int4 范围
    w_hat = w_q * s_g                                                    # 反量化权重
    return (a.to(torch.float32) @ w_hat.T.to(torch.float32))


def _rel_err(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return ((y_hat - y).norm() / y.norm()).item()


@torch.no_grad()
def quantized_gemm_report(m: int = 256, k: int = 4096, n: int = 256, seed: int = 0) -> dict[str, float]:
    """在 (M,K)x(K,N) 随机 GEMM 上度量 W8A8 / W4A16 的相对误差(FP32 为基准)。

    返回示例(seed=0 / K=4096, CPU 与 GPU 数值一致):
        rel_err_w8a8 ≈ 1.4e-2, rel_err_w4a16g128 ≈ 1.2e-1 —— INT8 点积误差
        ~1.4%(理论 sqrt(2)·0.9%), INT4 无结构权重上 ~12%; W4 收益在带宽
        (bytes_per_weight 0.5 vs 1.0)而非精度, 两者场景不同(见文档 Q11)。
    """
    torch.manual_seed(seed)
    a = torch.randn(m, k) * 0.1          # 激活尺度不影响相对误差结论
    w = torch.randn(n, k) * 0.05

    y_fp = a @ w.T
    return {
        "rel_err_w8a8": _rel_err(w8a8_matmul(a, w), y_fp),
        "rel_err_w4a16_g128": _rel_err(w4a16_matmul(a, w, 128), y_fp),
        "rel_err_w4a16_g64": _rel_err(w4a16_matmul(a, w, 64), y_fp),
        "bytes_per_weight_w8a8": 1.0,
        "bytes_per_weight_w4a16": 0.5,
    }
