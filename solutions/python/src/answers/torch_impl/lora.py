"""LoRA Adapter 手写实现。

对应面试题:
    - 综合面经题库-1 PyTorch 编程题 "实现 LoRA Adapter"
    - MiniMax-一面 "LoRA 推理是否需 Adapter / A/B 初始化为何不同"
    - 百度-二面-2 "LoRA 低秩为何有效、Rank 不当现象"
    - 京东 "LoRA 秩 r 影响"

公式: y = W0 x + α/r * B(A x), A∈R^{r×d}, B∈R^{d_out×r}
初始化: A ~ N(0, σ²)(随机), B = 0(零) —— 保证训练起点 y = W0 x 不变;
若 A、B 都随机, 一开始就注入随机扰动且梯度方差翻倍, 破坏预训练权重。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """给任意 nn.Linear 套 LoRA: 冻结原权重, 低秩旁路可训。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        assert r > 0
        self.base = base
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r  # 缩放系数 α/r

        in_f, out_f = base.in_features, base.out_features
        # A: 高斯初始化; B: 全零 → 初始时 ΔW = B@A = 0
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

        self.base.weight.requires_grad_(False)  # 冻结主干
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先降维升维再缩放; dropout 只作用于旁路
        return self.base(x) + self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge(self) -> None:
        """推理部署常考: W' = W0 + α/r * B@A 原地合并, 消除额外 kernel。"""
        delta = (self.lora_B @ self.lora_A) * self.scaling  # (out_f, in_f)
        self.base.weight.data += delta
        self.merged = True

    def lora_parameters(self) -> list[nn.Parameter]:
        return [self.lora_A, self.lora_B]


def apply_lora_to_model(model: nn.Module, r: int = 8, alpha: int = 16,
                        target_names: tuple[str, ...] = ("q_proj", "v_proj")) -> nn.Module:
    """工程化用法: 遍历模块树, 把名字匹配的 Linear 替换为 LoRALinear(原地)。"""
    for name, module in model.named_children():
        if name in target_names and isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha))
        else:
            apply_lora_to_model(module, r, alpha, target_names)
    return model


def trainable_param_stats(model: nn.Module) -> dict[str, float]:
    """统计可训练参数占比 —— 面试追问"LoRA 省多少参数"。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "ratio": trainable / max(total, 1)}


@torch.no_grad()
def lora_sanity_check() -> dict[str, object]:
    """验证: 初始(B=0)时输出与原 Linear 完全一致; merge 后输出与 merge 前一致。"""
    torch.manual_seed(0)
    base = nn.Linear(64, 32)
    x = torch.randn(4, 64)
    y0 = base(x)

    lora = LoRALinear(base, r=4)
    y_init = lora(x)  # B=0 → 等于原输出

    # 模拟一步"训练"
    lora.lora_B.normal_(0, 0.02)
    y_trained = lora(x)
    lora.merge()
    y_merged = lora.base(x)  # 合并后走纯 Linear 路径

    return {
        "init_equals_base": bool(torch.allclose(y0, y_init, atol=1e-6)),
        "merge_equals_trained": bool(torch.allclose(y_trained, y_merged, atol=1e-5)),
    }
