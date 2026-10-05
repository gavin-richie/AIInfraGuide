"""SFT 交叉熵损失手写（shift right / loss mask 语义）。

对应面试题:
    - MiniMax-一面 "PyTorch 写 SFT loss(shift right)"
    - 百度-二面-2 "多轮对话 Loss Mask"
    - 抖音电商 "Qwen3 快慢思考"(mask 掉思考段同族)

关键点:
    1. 语言模型 loss = next-token prediction: logits[:, :-1] 对 labels[:, 1:]
       (输入右移一位作为目标, 所谓 "shift right" 其实是对 label 左移)
    2. pad / 无效 token 的 label 置 -100, CrossEntropy 的 ignore_index 自动跳过
    3. 多轮对话: 只对 assistant 段计 loss, prompt 与系统段 mask 成 -100
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sft_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """手动 shift 版(面试手写版)。

    logits: (B, S, V) 模型输出; labels: (B, S) 完整序列的 token id(含 prompt),
    无效位置已置 ignore_index。
    """
    shift_logits = logits[:, :-1, :].contiguous()   # 预测位置 0..S-2
    shift_labels = labels[:, 1:].contiguous()        # 目标位置 1..S-1
    if (shift_labels != ignore_index).sum() == 0:    # 全部被 mask: 返回 0 而非 nan
        return logits.new_zeros((), dtype=torch.float32)
    return F.cross_entropy(
        shift_logits.view(-1, logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def sft_loss_multiturn(
    logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """多轮对话版: loss_mask (B,S) 标出 assistant token 位置。

    实现: 非 mask 位置 label 置 -100 后复用 ignore_index 机制。
    """
    masked_labels = labels.masked_fill(loss_mask == 0, -100)
    return sft_loss(logits, masked_labels)


def sft_loss_manual(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """完全手动版(展示原理): log_softmax + NLL + mask 均值, 不用 F.cross_entropy。"""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100                       # (B, S-1)

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    # gather 出目标 token 的 logprob: (B, S-1)
    token_lp = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * valid                        # 无效位置清零
    return -token_lp.sum() / valid.sum().clamp(min=1)  # 只对有效 token 求均值


@torch.no_grad()
def sft_loss_equivalence() -> dict[str, object]:
    """验证手动版与 F.cross_entropy 版数值一致; 演示 mask 位置不计梯度。"""
    torch.manual_seed(0)
    B, S, V = 2, 8, 100
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    labels[:, :3] = -100  # 前 3 个是 prompt, 不计 loss

    l1 = sft_loss(logits, labels)
    l2 = sft_loss_manual(logits, labels)
    l3 = sft_loss_multiturn(logits, labels, (labels != -100).float())
    return {
        "max_err_fce_vs_manual": float((l1 - l2).abs()),
        "max_err_mask_vs_ignore": float((l1 - l3).abs()),
        "allclose": bool(torch.allclose(l1, l2, atol=1e-5) and torch.allclose(l1, l3, atol=1e-5)),
    }
