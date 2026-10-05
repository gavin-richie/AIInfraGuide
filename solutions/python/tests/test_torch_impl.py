"""torch_impl 模块测试。"""
import torch
import torch.nn as nn

from answers.torch_impl.lora import LoRALinear, apply_lora_to_model, trainable_param_stats, lora_sanity_check
from answers.torch_impl.sft_loss import sft_loss, sft_loss_manual, sft_loss_multiturn, sft_loss_equivalence
from answers.torch_impl.broadcasting import (
    repeat_vs_expand, view_vs_contiguous, broadcast_4d_elementwise,
    broadcast_chain, broadcast_4d_semantics,
)
from answers.torch_impl.flow_matching import train_and_sample_demo
from answers.torch_impl.nms import iou_xyxy, nms_torch, nms_vectorized


def test_lora_init_and_merge():
    res = lora_sanity_check()
    assert res["init_equals_base"], "B=0 初始时必须等于原输出"
    assert res["merge_equals_trained"], "merge 后必须与 merge 前等价"


def test_lora_freezes_base():
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, r=4)
    assert not lora.base.weight.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad
    stats = trainable_param_stats(lora)
    # 可训练 = A(4*32) + B(16*4) = 192
    assert stats["trainable"] == 4 * 32 + 16 * 4


def test_lora_apply_to_model():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)

    model = apply_lora_to_model(Block(), r=4)
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert isinstance(model.out_proj, nn.Linear) and not isinstance(model.out_proj, LoRALinear)


def test_sft_loss_shift_semantics():
    # 单元素 sanity: 若 logits 在位置 t 强烈偏好 labels[t+1], loss 应接近 0
    logits = torch.full((1, 3, 4), -20.0)
    labels = torch.tensor([[0, 1, 2]])
    logits[0, 0, 1] = 20.0  # 位置 0 预测 token 1
    logits[0, 1, 2] = 20.0  # 位置 1 预测 token 2
    loss = sft_loss(logits, labels)
    assert loss < 0.01


def test_sft_loss_equivalence():
    res = sft_loss_equivalence()
    assert res["allclose"], res


def test_sft_loss_mask_skips_prompt():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 50)
    labels = torch.randint(0, 50, (2, 8))
    full = sft_loss(logits, labels)
    masked = sft_loss_multiturn(logits, labels, torch.ones(2, 8))
    assert torch.allclose(full, masked, atol=1e-6)
    # 全 mask → 防除零, loss 有限
    all_masked = sft_loss_multiturn(logits, labels, torch.zeros(2, 8))
    assert torch.isfinite(all_masked)


def test_repeat_vs_expand():
    res = repeat_vs_expand()
    assert res["expand_stride"] == (0, 1)
    assert res["repeat_storage_bytes"] > res["expand_storage_bytes"]
    assert not res["expand_is_contiguous"]
    assert res["equal_values"]


def test_view_vs_contiguous():
    res = view_vs_contiguous()
    assert not res["transpose_contiguous"]
    assert res["view_after_transpose_error"] is not None
    assert res["reshape_ok"] == (12,)
    assert res["contiguous_then_view_ok"] == (12,)


def test_broadcast_4d():
    a = torch.randn(2, 1, 6, 1)
    b = torch.randn(1, 3, 1, 5)
    got = broadcast_4d_elementwise(a, b)
    ref = a * b
    assert got.shape == (2, 3, 6, 5)
    assert torch.equal(got, ref)
    sem = broadcast_4d_semantics((2, 1, 6, 1), (1, 3, 1, 5))
    assert sem["out_shape"] == (2, 3, 6, 5)
    # 广播维 stride 为 0: a 在 dim1/3 广播, b 在 dim0/2 广播
    assert sem["strides"][0][1] == 0 and sem["strides"][0][3] == 0
    assert sem["strides"][1][0] == 0 and sem["strides"][1][2] == 0


def test_broadcast_chain():
    a, b, c = torch.randn(1, 256), torch.randn(256, 128), torch.randn(128, 256)
    out = broadcast_chain(a, b, c)
    assert out.shape == (128, 256)
    ref = (a * b.t()) * c
    assert torch.equal(out, ref)


def test_flow_matching_end_to_end():
    res = train_and_sample_demo(dim=2, steps=300)
    assert res["clustered_near_targets"], res
    # 双峰目标: |x| 均值应接近 2, y 方差接近 0.1
    assert 1.2 < res["sample_abs_x_mean"] < 2.8
    assert res["sample_y_std"] < 0.5
    # loss 有不可约条件方差下限, 只需有限且不再爆增
    assert res["final_loss"] < 2.5


def test_nms():
    boxes = torch.tensor([
        [0, 0, 10, 10],
        [1, 1, 11, 11],     # 与第 0 框 IoU 高 → 应被抑制
        [20, 20, 30, 30],   # 独立框 → 保留
        [50, 50, 60, 60],   # 独立框但分低
    ], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    keep = nms_torch(boxes, scores, threshold=0.5)
    assert keep == [0, 2, 3]
    assert nms_vectorized(boxes, scores, 0.5) == keep
    # IoU 边界: 完全重叠 IoU=1, 完全相离 IoU=0
    assert iou_xyxy(boxes[0], boxes[0]) == 1.0
    assert iou_xyxy(boxes[0], boxes[2]) == 0.0
