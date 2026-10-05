"""AWQ W4A16 量化全流程（阿里巴巴-云-AI-Infra-实习-二面-1 Q4）。

对应面试题："量化推理（如 AWQ 路线下的 W4A16）在具体实现上涉及哪些关键环节？
包括 Linear 层改造、数据结构设计、内存布局处理、正确性验证等。"

本模块用纯 PyTorch 复现 AWQ（Activation-aware Weight Quantization）的四个环节：

1. 缩放搜索 ``search_awq_scale``：AWQ 观察到约 1% 的"重要通道"（激活幅度大）
   对量化误差敏感，通过对权重做 per-channel 等价缩放保护它们：
       W' = W * diag(s),   X' = X * diag(s)^-1     （输出不变）
   s 通道维度上取 s = act_scale ** alpha，网格搜 alpha ∈ [0, 1]
   使等价变换后权重的量化误差最小（alpha=0 即退化为 RTN 直接量化）。
2. INT4 打包 ``pack_int4`` / ``unpack_int4``：两个 int4 压进一个 uint8，
   低位在前（与 GPU kernel 的解包顺序一致）；group_size（如 128）个权重
   共享一组 fp16 scale/zero，以 (out_features, in_features // group) 矩阵存放。
3. Linear 层改造 ``W4A16Linear``：权重以 packed uint8 + scale/zero 保存
   （显存 1/4 + 少量 fp16 参数），forward 时反量化回 fp16 再 matmul——
   真实 kernel（Marlin/awq_dequantze/gemv）把反量化内嵌在装载阶段，
   这里以功能等价的方式演示数据结构与布局。
4. 正确性验证 ``layer_output_error``：逐层输出相对误差（真实工程里还做
   余弦相似度 > 0.99、PPL 回归、端到端 benchmark 三级验证）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

GROUP_SIZE = 128  # AWQ 默认 group size，128 个 int4 权重共享一组 scale/zero


# ---------------------------------------------------------------- 1. 缩放搜索
def search_awq_scale(
    layer: torch.nn.Linear,
    x_calib: torch.Tensor,
    alphas: list[float] | None = None,
    n_grid: int = 20,
) -> torch.Tensor:
    """对单个 Linear 搜索 AWQ per-channel 缩放因子。

    Args:
        layer: 待量化的 fp16 Linear（W 形状 [out, in]）。
        x_calib: 校准激活 [N, in]（统计每通道最大幅度，N 条样本拼起来的）。
        alphas: 候选 alpha；None 时在 (0, 1] 均匀取 n_grid 个点。

    Returns:
        最优 per-channel scale s（形状 [in]）。量化时对 W*s 做对称 INT4 RTN，
        推理时激活乘 s^-1。

    搜索准则（与 AWQ 论文一致）：最小化等价变换前后的**输出误差**
        ||X·W - X·Q(W·diag(s))·diag(s)^-1||_F
    激活幅度大的通道误差被自动放大权重，从而选出保护离群通道的 s。
    """
    if alphas is None:
        alphas = [i / n_grid for i in range(1, n_grid + 1)]
    x_max = x_calib.abs().amax(dim=0).clamp(min=1e-6)  # 每个输入通道的激活幅度 [in]
    w = layer.weight.data.float()
    y_ref = x_calib.float() @ w.T
    best_err, best_s = float("inf"), torch.ones(w.shape[1], device=w.device)
    for alpha in alphas:
        s = x_max**alpha
        w_q = fake_quant_int4((w * s).to(torch.float16)).float() / s
        err = ((x_calib.float() @ w_q.T - y_ref) ** 2).mean().item()
        if err < best_err:
            best_err, best_s = err, s
    return best_s.to(torch.float16)


# ---------------------------------------------------------------- 2. INT4 打包
def fake_quant_int4(w: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """对称 per-group INT4 假量化（quant-dequant），用于搜索与参考实现。"""
    q, s = quantize_int4_symmetric(w, group_size)
    return q.float() * s.repeat_interleave(group_size, dim=1).float()  # 反量化回浮点


def quantize_int4_symmetric(
    w: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 per-group INT4：每组一个 fp16 scale，q ∈ [-8, 7]。

    Returns:
        q: int8 张量（数值范围 [-8,7]，供 pack_int4 打包）。
        s: scale，形状 [out, in // group]。
    """
    out_f, in_f = w.shape
    assert in_f % group_size == 0, "in_features 必须能被 group_size 整除"
    wg = w.float().reshape(out_f, in_f // group_size, group_size)
    scale = wg.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    q = torch.round(wg / scale[..., None]).clamp(-8, 7).to(torch.int8)
    return q.reshape(out_f, in_f), scale.to(torch.float16)


def pack_int4(q_int8: torch.Tensor) -> torch.Tensor:
    """两个 int4 打包进一个 uint8（低位在前：q[2i] 在低 4 位）。

    负数按 int4 补码编码（-1 → 0xF）。GPU kernel 按此布局解包：
    ``(packed & 0xF)`` 与 ``(packed >> 4)``，再对 >=8 的值减 16 符号扩展。
    真实 AWQ kernel 还会按线程/tile 访问模式对权重做全局重排（列交错、
    marlin 格式等），这里保持逻辑顺序，语义一致。
    """
    assert q_int8.dtype == torch.int8 and q_int8.shape[-1] % 2 == 0
    flat = q_int8.reshape(-1, 2).to(torch.uint8) & 0xF
    packed = flat[:, 0] | (flat[:, 1] << 4)  # 低位在前
    return packed.reshape(q_int8.shape[0], q_int8.shape[1] // 2)


def _sign_extend4(x: torch.Tensor) -> torch.Tensor:
    """无符号 4 位（0..15）→ int8（-8..7）：>=8 表示补码负数。"""
    x = x.to(torch.int8)
    return torch.where(x >= 8, x - 16, x)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """pack_int4 的逆操作，返回 int8（-8..7）。"""
    low = _sign_extend4(packed & 0xF)
    high = _sign_extend4(packed >> 4)
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


# ---------------------------------------------------------------- 3. Linear 层改造
class W4A16Linear(torch.nn.Module):
    """W4A16 Linear：权重 INT4 + fp16 scale/zero，激活与计算 FP16。

    内存布局（面试要点）：
      - ``qweight``: uint8 [out, in/2]，两个 int4 打包；
      - ``scales``:   fp16 [out, in/group]；
      - 两者都按输出通道为行、输入通道为列的行主序存放，与 fp16 GEMM 的
        B 矩阵布局对齐，反量化后的 tile 可以直接进 mma。
    """

    def __init__(self, layer: torch.nn.Linear, scale_search: torch.Tensor | None = None,
                 group_size: int = GROUP_SIZE):
        super().__init__()
        out_f, in_f = layer.weight.shape
        assert in_f % group_size == 0
        self.out_features, self.in_features, self.group_size = out_f, in_f, group_size
        # AWQ 等价变换：先缩放保护重要通道，再量化（推理时激活除以 scale_search）
        self.register_buffer("inv_awq_scale", (1.0 / scale_search).to(torch.float16)
                             if scale_search is not None else None)
        w = layer.weight.data.to(torch.float16) * (
            scale_search.to(torch.float16) if scale_search is not None else 1.0)
        q, s = quantize_int4_symmetric(w, group_size)
        self.register_buffer("qweight", pack_int4(q))   # uint8 [out, in/2]
        self.register_buffer("scales", s)               # fp16  [out, in/group]
        if layer.bias is not None:
            self.register_buffer("bias", layer.bias.data.to(torch.float16))
        else:
            self.register_buffer("bias", None)

    def dequantize_weight(self) -> torch.Tensor:
        """反量化回 fp16 权重（真实 kernel 中这步融合在装载里，不物化）。"""
        q = unpack_int4(self.qweight).to(torch.float16)  # [out, in]
        s = self.scales.repeat_interleave(self.group_size, dim=1)
        w = q * s
        if self.inv_awq_scale is not None:  # 还原 AWQ 等价变换
            w = w * self.inv_awq_scale
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight()
        return F.linear(x.to(torch.float16), w, self.bias)

    def weight_nbytes(self) -> int:
        """INT4 权重实际占用（packed uint8 + scales）。"""
        return self.qweight.numel() + self.scales.numel() * 2


# ---------------------------------------------------------------- 4. 正确性验证
def layer_output_error(ref_layer: torch.nn.Linear, w4_layer: W4A16Linear,
                       x: torch.Tensor) -> float:
    """逐层输出相对误差：||y_ref - y_w4|| / ||y_ref||，工程阈值通常 < 1%。"""
    with torch.no_grad():
        y_ref = ref_layer(x.to(torch.float16)).float()
        y_w4 = w4_layer(x).float()
    return ((y_ref - y_w4).norm() / y_ref.norm()).item()


def demo(seed: int = 0) -> dict[str, float]:
    """端到端演示：搜索 scale → 打包 → W4A16 前向 → 误差与显存对比。"""
    torch.manual_seed(seed)
    layer = torch.nn.Linear(1024, 1024, dtype=torch.float16)
    # 造出"少数重要通道"的校准激活（模拟 LLM 激活离群值）
    x = torch.randn(256, 1024, dtype=torch.float16)
    x[:, :8] *= 20.0
    s = search_awq_scale(layer, x)
    w4 = W4A16Linear(layer, scale_search=s)
    err_awq = layer_output_error(layer, w4, x)
    w4_rtn = W4A16Linear(layer, scale_search=None)  # alpha=0 退化为 RTN
    err_rtn = layer_output_error(layer, w4_rtn, x)
    fp16_nbytes = layer.weight.numel() * 2
    return {
        "err_rtn": err_rtn,
        "err_awq": err_awq,
        "memory_ratio": w4.weight_nbytes() / fp16_nbytes,
    }
