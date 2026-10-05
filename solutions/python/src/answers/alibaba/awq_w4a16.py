"""AWQ W4A16 量化完整实现（阿里巴巴-云-AI-Infra-实习-二面-1 Q4）。

面试考点全覆盖：
1. AWQ 核心思想：激活离群通道对应的权重"重要"，用 per-channel 等价缩放
   W'=W·diag(s)、X'=X/s 保护它们——输出不变，但权重量化误差分布更友好；
2. 缩放搜索：s = x_max^α，α∈[0,1) 网格逐层搜（与官方 autoawq 的 auto_scale 一致），
   目标是最小化该层输出 MSE；
3. per-group(group=128) 非对称 INT4 量化 + 位打包（两个 int4 压进一个 uint8），
   模拟 Marlin 类 kernel "一次 load 取多个权重、寄存器内解包"的内存布局；
4. AWQLinear：packed int4 + group scale/zero 存储（显存≈FP16 的 1/4），
   前向解包反量化后走 FP16 GEMM（即 W4A16：只压权重、激活与计算保持 FP16）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def fake_quant_int4(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """per-group 非对称 INT4 伪量化（quant-dequant），用于搜索阶段的误差评估。"""
    q, scale, zero = quantize_int4_asymmetric(w, group_size)
    return dequantize_int4(q, scale, zero, group_size)


def quantize_int4_asymmetric(
    w: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """per-group 非对称 INT4 量化。

    q ∈ [0,15] uint8 形状 [out, in]；scale/zero 形状 [out, in/group]（组内共享）。
    scale = (max-min)/15，zero = round(-min/scale)。
    """
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} 必须能被 group_size={group_size} 整除")
    w_g = w.reshape(out_f, in_f // group_size, group_size)
    w_max = w_g.amax(dim=-1)
    w_min = w_g.amin(dim=-1)
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    zero = torch.round(-w_min / scale).clamp(0, 15)
    q = torch.round(w_g / scale.unsqueeze(-1) + zero.unsqueeze(-1)).clamp(0, 15)
    return q.reshape(out_f, in_f).to(torch.uint8), scale, zero


def dequantize_int4(
    q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group_size: int = 128
) -> torch.Tensor:
    """(q - zero) * scale 还原为 [out, in] 的 float 权重。"""
    out_f, in_f = q.shape
    q_g = q.float().reshape(out_f, in_f // group_size, group_size)
    w = (q_g - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    return w.reshape(out_f, in_f)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """把 uint4 张量沿最后一维两两打包进一个 uint8（低 4 位存偶数位元素）。

    真实 kernel 里对应"一个 32bit load 一次取 8 个 int4 权重"的向量化装载，
    打包顺序必须与 kernel 的解包位操作约定一致（这里：偶数元素在低半字节）。
    """
    if q.shape[-1] % 2 != 0:
        raise ValueError("最后一维长度必须为偶数")
    lo = q[..., 0::2].to(torch.uint8)
    hi = q[..., 1::2].to(torch.uint8)
    return lo | (hi << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """pack_int4 的逆操作：uint8 拆回两个 uint4。"""
    lo = packed & 0x0F
    hi = packed >> 4
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def awq_search_scale(
    w: torch.Tensor, x: torch.Tensor, n_grid: int = 20, group_size: int = 128
) -> torch.Tensor:
    """AWQ 逐层缩放搜索：s = x_max^α，α 网格取值，选输出 MSE 最小的 α。

    等价变换 W·diag(s)、X/s 不改变 FP16 输出，但把"激活大的通道"的权重放大
    再量化，等价于给了这些通道更细的量化分辨率。返回 per-channel scale [in]。
    """
    x_max = x.abs().amax(dim=0).clamp(min=1e-4)  # [in] 每输入通道的激活最大值
    y_ref = x @ w.t()
    best_scale, best_err = None, float("inf")
    for i in range(n_grid):
        alpha = i / n_grid
        s = x_max.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()  # 归一化，避免整体幅度漂移
        w_q = fake_quant_int4(w * s, group_size)
        err = ((x / s) @ w_q.t() - y_ref).pow(2).mean().item()
        if err < best_err:
            best_scale, best_err = s, err
    return best_scale


class AWQLinear(nn.Module):
    """W4A16 Linear：权重以 packed int4 + per-group scale/zero 存储。

    显存占用 ≈ FP16 权重的 1/4（decode 是 memory bound，收益即来自权重读取量）。
    forward 里解包反量化成 FP16 再 GEMM——模拟 Marlin 类 kernel 在寄存器/smem
    内完成"装载即解包、不写回 HBM"的行为（这里用解包后的 FP16 GEMM 表达语义）。
    """

    def __init__(self, linear: nn.Linear, group_size: int = 128, calib_x: torch.Tensor | None = None):
        super().__init__()
        w = linear.weight.data.float()
        if calib_x is not None:
            # AWQ：先搜等价缩放并把 s 折叠进权重（运行时输入需除以 s，
            # 等价地可以在模块里保存 s 用于 forward 前缩放）
            self.register_buffer("act_scale", awq_search_scale(w, calib_x, group_size=group_size))
            w = w * self.act_scale  # 折叠后的权重再量化
        else:
            self.register_buffer("act_scale", None)
        q, scale, zero = quantize_int4_asymmetric(w, group_size)
        self.register_buffer("q_packed", pack_int4(q))
        self.register_buffer("group_scale", scale.half())
        self.register_buffer("group_zero", zero)
        self.group_size = group_size
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

    def dequantized_weight(self) -> torch.Tensor:
        """解包 + 反量化：模拟 kernel 装载路径。"""
        q = unpack_int4(self.q_packed)
        w = dequantize_int4(q.float(), self.group_scale.float(), self.group_zero.float(), self.group_size)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_scale is not None:
            x = x / self.act_scale  # 与权重缩放对消的等价变换
        w = self.dequantized_weight().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w, bias)

    def weight_storage_bytes(self) -> int:
        return self.q_packed.numel() + self.group_scale.numel() * 2 + self.group_zero.numel()


def awq_pipeline_report(seed: int = 0) -> dict[str, float]:
    """端到端演示：构造带激活离群通道的 Linear，对比 naive W4 与 AWQ W4 的输出误差。

    离群通道上激活幅度大 → naive 量化时该通道权重的相对分辨率不足，AWQ 通过
    等价缩放显著降低输出 MSE。这个函数同时是 tests/test_alibaba.py 的核心断言来源。
    """
    torch.manual_seed(seed)
    out_f, in_f, n_tok = 64, 256, 32
    linear = nn.Linear(in_f, out_f, bias=False)
    x = torch.randn(n_tok, in_f)
    # 人为制造 8 个离群通道：激活幅度 ×20（模拟 LLM 中 0.1%-1% 的 outlier 通道）
    outlier_idx = torch.arange(0, in_f, in_f // 8)
    x[:, outlier_idx] *= 20.0
    y_ref = linear(x)

    # naive：直接 INT4（α=0，无激活感知缩放）
    with torch.no_grad():
        w_naive = fake_quant_int4(linear.weight.data.float())
        err_naive = (x @ w_naive.t() - y_ref).pow(2).mean().item()
        awq = AWQLinear(linear, calib_x=x)
        y_awq = awq(x)
        err_awq = (y_awq - y_ref).pow(2).mean().item()

    fp16_bytes = linear.weight.numel() * 2
    return {
        "err_naive_w4": err_naive,
        "err_awq_w4": err_awq,
        "err_ratio": err_awq / max(err_naive, 1e-12),
        "cos_sim": torch.nn.functional.cosine_similarity(
            y_awq.flatten(), y_ref.flatten(), dim=0
        ).item(),
        "storage_bytes": float(awq.weight_storage_bytes()),
        "fp16_bytes": float(fp16_bytes),
        "compression": fp16_bytes / awq.weight_storage_bytes(),
    }
