"""量化校准算法手写实现: minmax / percentile / KL 散度 / MSE, 及量化粒度。

对应面试题:
    - 智谱-实习-一面 基础 1: per-tensor / per-channel / per-group 量化粒度(哪种最细)
    - 智谱-实习-一面 基础 2: minmax 校准与 percentile 校准的核心差异
    - 智谱-实习-一面 基础 3: KL 散度校准 / MSE 校准的基本原理
    - 智谱-实习-一面 编程 8: 手写 minmax 校准与 percentile 校准算法

核心结论(面试口径):
    minmax   : scale = absmax / (2^(b-1)-1), 快但对离群值敏感(一个尖峰就把整个
               量化步长拉大, 绝大多数正常激活被压进少数几个 level);
    percentile: 取 |x| 的 q 分位(如 99.9%)做截断阈值, 牺牲极少数离群点换更细的
               量化步长 —— 差异本质是"阈值取绝对最大还是分布分位";
    KL       : TensorRT entropy 校准。把 |x| 直方图在候选阈值 T 处截断(超阈值质量
               折叠进最后一格模拟 clamp), 再量化到 128 个 level 并反量化回直方图得
               Q, 枚举 T 使 KL(P||Q) 最小 —— 信息论意义下"最保分布"的截断点;
    MSE      : 网格搜索截断阈值(或分位), 直接最小化 fake-quant 前后的输出均方误差。
"""
from __future__ import annotations

import torch

__all__ = [
    "minmax_scale",
    "percentile_scale",
    "kl_calibration_scale",
    "mse_calibration_scale",
    "quantize_to_int",
    "dequantize",
    "fake_quant",
    "compute_scale",
    "calibration_compare",
]


# ------------------------------------------------------------- 基础量化原语
def quantize_to_int(x: torch.Tensor, scale, zero_point=0, num_bits=8) -> torch.Tensor:
    """量化到 int: q = round(x/scale) + zero_point, 并 clamp 到表示范围。

    约定: zero_point == 0 为对称 int8([-2^(b-1), 2^(b-1)-1]);
          zero_point != 0 为非对称 uint8 语义([0, 2^b - 1])。
    """
    if zero_point == 0:
        qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    else:
        qmin, qmax = 0, 2**num_bits - 1
    q = torch.round(x / scale).to(torch.int32) + int(zero_point)
    return q.clamp_(qmin, qmax)


def dequantize(q: torch.Tensor, scale, zero_point=0) -> torch.Tensor:
    """反量化: x_hat = (q - zero_point) * scale。"""
    return (q.to(scale.dtype if torch.is_tensor(scale) else torch.float32) - int(zero_point)) * scale


def fake_quant(x: torch.Tensor, scale, zero_point=0, num_bits=8) -> torch.Tensor:
    """量化+反量化(伪量化), 校准算法用它度量量化误差。scale 可广播(per-channel/group)。"""
    return dequantize(quantize_to_int(x, scale, zero_point, num_bits), scale, zero_point)


# ----------------------------------------------------- 智谱 编程 8: 手写两种校准
def minmax_scale(x: torch.Tensor, num_bits: int = 8, symmetric: bool = True):
    """minmax 校准: 直接取绝对最大(对称)或 [min, max] 全范围(非对称)。

    对称:   scale = absmax / (2^(b-1) - 1),            zero_point = 0
    非对称: scale = (max - min) / (2^b - 1),           zero_point = -round(min/scale)
    """
    if symmetric:
        absmax = x.abs().amax().clamp_(min=1e-12).item()
        return absmax / (2 ** (num_bits - 1) - 1), 0
    xmin, xmax = x.amin().item(), x.amax().item()
    scale = max(xmax - xmin, 1e-12) / (2**num_bits - 1)
    zero_point = -round(xmin / scale)
    zero_point = min(max(zero_point, -2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
    return scale, zero_point


def percentile_scale(x: torch.Tensor, q: float = 99.9, num_bits: int = 8, symmetric: bool = True):
    """percentile 校准: 取 |x| 的 q% 分位做截断阈值, 忽略更极端的离群点。

    torch.quantile 在浮点上做插值, 等价于 numpy.percentile; 生产实现(TensorRT/
    AMS 系)常先做 2048-bin 直方图再取累计频率达到 q% 的 bin 右边界。
    """
    if symmetric:
        threshold = torch.quantile(x.detach().abs().flatten().float(), q / 100.0).item()
        return max(threshold, 1e-12) / (2 ** (num_bits - 1) - 1), 0
    flat = x.detach().flatten().float()
    lo = torch.quantile(flat, 100.0 - q).item()
    hi = torch.quantile(flat, q).item()
    scale = max(hi - lo, 1e-12) / (2**num_bits - 1)
    zero_point = -round(lo / scale)
    return scale, min(max(zero_point, -2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)


# ------------------------------------------------- KL 散度校准(TensorRT 经典)
def kl_calibration_scale(
    x: torch.Tensor,
    num_bins: int = 2048,
    num_levels: int = 128,
    num_bits: int = 8,
) -> float:
    """KL(entropy) 校准: 枚举截断阈值 T, 最小化 KL(P||Q)(TensorRT 经典算法)。

    步骤(与 TensorRT / pytorch-quantization / PPQ 实现一致):
      1. 对 |x| 在 [0, absmax] 上建 num_bins-bin 直方图 H;
      2. 候选阈值 T_i = i * bin_width (i = num_levels..num_bins):
         - 参考分布 P = H[:i], 并把 H[i:] 的离群质量**折叠进 P[i-1]**
           (模拟"截断后离群点全部 clamp 到 T_i"的效果);
         - 候选分布 Q: 用**未折叠**的 H[:i] 按 i/num_levels 个 bin 一组归并到
           num_levels 个 level, 组内质量只摊到原本非零的 bin 上(空 bin 保持 0),
           再归一化 —— Q 刻画"量化+反量化后的真实取值分布";
         - P、Q 分别归一化后计算 KL(P||Q);
      3. 取 KL 最小的 i, scale = T_i / (num_levels - 1)。
    截断太少 → 量化步长过大、Q 与 P 的格点错位大; 截断太多 → 折叠质量造成
    P/Q 在最后一格的失配大 —— KL 在两者间取平衡, 这就是"最保分布"的含义。
    """
    a = x.detach().abs().flatten().float()
    absmax = a.max().item()
    if absmax <= 0:
        return 0.0
    hist, _ = torch.histogram(a, bins=num_bins, range=(0.0, absmax))
    hist = hist.double()
    bin_width = absmax / num_bins
    eps = 1e-12

    best_kl, best_i = float("inf"), num_bins
    for i in range(num_levels, num_bins + 1):
        p = hist[:i].clone()
        p[i - 1] += hist[i:].sum()               # 离群质量折叠进最后一格
        p = p / p.sum().clamp_(min=eps)

        # 每个 bin 归属的 level: 组 [floor(l*i/L), floor((l+1)*i/L)) 精确切分
        bins = torch.arange(i, dtype=torch.double)
        level_of_bin = (bins * num_levels / i).long()          # (i,)
        group_mass = torch.zeros(num_levels, dtype=torch.double)
        group_mass.index_add_(0, level_of_bin, hist[:i])       # 各 level 总质量(未折叠)
        pos = hist[:i] > 0
        pos_per_group = torch.zeros(num_levels, dtype=torch.double)
        pos_per_group.index_add_(0, level_of_bin[pos], torch.ones(int(pos.sum()), dtype=torch.double))
        q_per_bin = group_mass[level_of_bin] / pos_per_group[level_of_bin].clamp_(min=1)
        q_per_bin = q_per_bin * pos                            # 空 bin 保持 0
        q = q_per_bin / q_per_bin.sum().clamp_(min=eps)

        nz = p > 0
        kl = (p[nz] * (torch.log(p[nz] + eps) - torch.log(q[nz] + eps))).sum().item()
        if kl < best_kl:
            best_kl, best_i = kl, i
    return best_i * bin_width / (num_levels - 1)


# ------------------------------------------------------------- MSE 校准
def mse_calibration_scale(
    x: torch.Tensor,
    num_bits: int = 8,
    percentiles: tuple[float, ...] = (100.0, 99.99, 99.95, 99.9, 99.5, 99.0, 98.0, 96.0),
) -> float:
    """MSE 校准: 在候选截断分位上网格搜索, 直接最小化 fake-quant 前后的均方误差。"""
    a = x.detach().abs().flatten().float()
    best_scale, best_mse = None, float("inf")
    qmax = 2 ** (num_bits - 1) - 1
    for pct in percentiles:
        threshold = torch.quantile(a, pct / 100.0).item()
        scale = max(threshold, 1e-12) / qmax
        xq = fake_quant(a, scale, 0, num_bits)
        mse = ((xq - a) ** 2).mean().item()
        if mse < best_mse:
            best_mse, best_scale = mse, scale
    assert best_scale is not None
    return best_scale


# ------------------------------------------------- 智谱 基础 1: 量化粒度
def compute_scale(
    x: torch.Tensor,
    granularity: str = "tensor",
    num_bits: int = 8,
    group_size: int = 128,
) -> torch.Tensor:
    """三种量化粒度的 scale 计算(对称 minmax), 返回可广播的 scale 张量。

    per-tensor : 整个张量一个 scale                -> 标量
    per-channel: 每个输出通道一个 scale(沿 dim0)    -> (C, 1) 权重友好
    per-group  : 沿输入维 K 每 group_size 个元素一组 -> (C, K) 逐元素, 最细
                 (GPTQ/AWQ 的 W4A16 惯例: group_size=128 沿 in_features 切)
    粒度从粗到细: per-tensor < per-channel < per-group, 越细精度越好、scale
    存储与反量化的开销越大。
    """
    if granularity == "tensor":
        return x.abs().amax().clamp_(min=1e-12) / (2 ** (num_bits - 1) - 1)
    if granularity == "channel":
        dims = list(range(x.dim()))
        reduce_dim = dims[1:] if x.dim() > 1 else [0]
        scale = x.abs().amax(dim=reduce_dim, keepdim=True).clamp_(min=1e-12)
        return scale / (2 ** (num_bits - 1) - 1)
    if granularity == "group":
        assert x.size(-1) % group_size == 0, "输入维 K 需能被 group_size 整除"
        lead = x.shape[:-1]                                     # (N,) 或 ()
        g = x.view(*lead, x.size(-1) // group_size, group_size)  # 沿输入维 K 分组
        scale = g.abs().amax(dim=-1).clamp_(min=1e-12)          # (N, K/g)
        scale = scale.repeat_interleave(group_size, dim=-1)      # (N, K) 逐元素可广播
        return scale / (2 ** (num_bits - 1) - 1)
    raise ValueError(f"未知粒度: {granularity}")


# ------------------------------------------------------------- 综合对比
@torch.no_grad()
def calibration_compare(x: torch.Tensor, num_bits: int = 8) -> dict[str, float]:
    """同一份激活上跑四种校准, 返回各自的 scale 与 fake-quant 相对误差。

    用法: x = 正常激活 + 少量离群点 时, percentile/KL/MSE 的误差都应低于 minmax。
    """
    qmax = 2 ** (num_bits - 1) - 1
    s_minmax, _ = minmax_scale(x, num_bits)
    s_pct, _ = percentile_scale(x, 99.9, num_bits)
    s_kl = kl_calibration_scale(x, num_bits=num_bits)
    s_mse = mse_calibration_scale(x, num_bits)

    def rel_err(scale: float) -> float:
        return (((fake_quant(x, scale, 0, num_bits) - x) ** 2).mean().sqrt() / (x**2).mean().sqrt()).item()

    return {
        "scale_minmax": s_minmax,
        "scale_percentile": s_pct,
        "scale_kl": s_kl,
        "scale_mse": s_mse,
        "absmax": x.abs().max().item(),
        "rel_err_minmax": rel_err(s_minmax),
        "rel_err_percentile": rel_err(s_pct),
        "rel_err_kl": rel_err(s_kl),
        "rel_err_mse": rel_err(s_mse),
    }
