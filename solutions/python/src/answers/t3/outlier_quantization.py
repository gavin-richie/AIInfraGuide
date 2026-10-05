"""极度不均匀分布下的量化方案对比（-10000 离群场景）。

来源: 摩尔线程-AI-Infra-实习-二面（量化专项 场景题）。
数据: 绝大多数值集中在 [-1, 1]，极少数离群值 ~ -10000。
per-tensor min-max 的 scale 被离群值撑大到 ~39，正常值全部塌缩到同一个
量化级（信息完全丢失）——本模块给出 4 条路线的量化误差对比:
  1. minmax per-tensor（坏基线）;
  2. percentile 截断（牺牲离群值精度，正常值恢复正常分辨率）;
  3. per-channel（离群只毁掉所在通道，其他通道不受牵连）;
  4. 离群分离/混合精度（LLM.int8 思路: 离群通道走高精度旁路，误差最小）。
理论见 docs/interview-answers/06-量化.md Q6。
"""
from __future__ import annotations

import torch

_EPS = 1e-12


def rmse(x: torch.Tensor, y: torch.Tensor) -> float:
    return ((x.float() - y.float()) ** 2).mean().sqrt().item()


def _fake_quant(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """INT8 假量化: q = clamp(round((x - z)/s), -128, 127), dq = q*s + z。

    zero_point 取 (xmin+xmax)/2（对称化中点），scale 支持逐通道广播。
    """
    scale = torch.clamp(scale, min=_EPS)          # 常数通道 range=0 防除零
    q = torch.clamp(torch.round((x - zero_point) / scale), -128, 127)
    return q * scale + zero_point


def _minmax_params(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xmin, xmax = x.min(), x.max()
    return (xmax - xmin) / 255, (xmin + xmax) / 2


def quantize_minmax(x: torch.Tensor) -> torch.Tensor:
    """整张 tensor 一个 scale 的 min-max 基线——离群值场景下彻底失效。"""
    scale, zero = _minmax_params(x)
    return _fake_quant(x, scale, zero)


def quantize_percentile(x: torch.Tensor, q: float = 0.999) -> torch.Tensor:
    """按 |x| 分位截断 clip 后再 min-max: 离群值被饱和, 正常值恢复分辨率。

    注意返回值是对原始 x 的量化（离群点落在 clamp 边界上），而不是对 clip 后
    数据的恒等变换——评估的是"部署时这样量化会损失多少"。
    """
    abs_max = torch.quantile(x.abs().float(), q)
    clipped = torch.clamp(x, -abs_max, abs_max)
    scale, zero = _minmax_params(clipped)
    return _fake_quant(x, scale, zero)


def quantize_per_channel(x: torch.Tensor) -> torch.Tensor:
    """按通道（dim=0）各取一组 scale/zero——离群只污染所在通道。"""
    xmin = x.min(dim=0).values
    xmax = x.max(dim=0).values
    scale = ((xmax - xmin) / 255).unsqueeze(0)
    zero = ((xmin + xmax) / 2).unsqueeze(0)
    return _fake_quant(x, scale, zero)


def quantize_outlier_split(x: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
    """LLM.int8 式混合精度: 含离群（amax > threshold）的通道走 FP 旁路，其余 INT8。"""
    outlier_channel = x.abs().amax(dim=0) > threshold       # [N]
    safe = x.clone().float()
    if (~outlier_channel).any():
        inlier = x[:, ~outlier_channel]
        safe[:, ~outlier_channel] = quantize_minmax(inlier)
    return safe
