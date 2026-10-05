"""阿里巴巴批次题解测试：AWQ W4A16 / GRPO-PPO / BST 迭代器 / 线性 SVM。

对应文档 docs/interview-answers/30 ~ 33-阿里-*.md，
运行：cd solutions/python && python -m pytest tests/test_alibaba.py -v
"""

import math

import torch

from answers.alibaba.awq_w4a16 import (
    AWQLinear,
    awq_pipeline_report,
    awq_search_scale,
    dequantize_int4,
    fake_quant_int4,
    pack_int4,
    quantize_int4_asymmetric,
    unpack_int4,
)
from answers.alibaba.bst_iterator import (
    BSTIteratorFlatten,
    BSTIteratorGenerator,
    BSTIteratorStack,
    iterate_all,
)
from answers.alibaba.grpo import (
    grpo_group_advantage,
    grpo_loss,
    importance_ratio,
    k3_kl_estimator,
    masked_mean,
    ppo_clipped_surrogate,
    ppo_loss,
    ppo_vs_grpo_demo,
)
from answers.alibaba.svm import LinearSVM, hinge_loss, make_separable_blobs, rbf_kernel
from answers.algorithms.trees import TreeNode

import torch.nn as nn


# ---------- AWQ W4A16（云-实习-二面-1 Q4） ----------


def test_int4_pack_unpack_roundtrip():
    torch.manual_seed(0)
    q = torch.randint(0, 16, (8, 128), dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)


def test_int4_quant_dequant_error():
    torch.manual_seed(0)
    w = torch.randn(32, 128) * 0.05  # LLM 权重典型幅度
    q, scale, zero = quantize_int4_asymmetric(w, group_size=128)
    assert q.dtype == torch.uint8 and int(q.max()) <= 15 and int(q.min()) >= 0
    w_rec = dequantize_int4(q.float(), scale, zero)
    rel_err = (w_rec - w).norm() / w.norm()
    assert rel_err < 0.15, f"per-group INT4 相对误差过大: {rel_err:.3f}"


def test_awq_linear_matches_fp16_and_saves_memory():
    torch.manual_seed(0)
    linear = nn.Linear(256, 64, bias=True)
    awq = AWQLinear(linear, group_size=128)  # 无校准数据：naive per-group INT4
    x = torch.randn(16, 256)
    y_fp16, y_w4 = linear(x), awq(x)
    cos = torch.nn.functional.cosine_similarity(y_fp16.flatten(), y_w4.flatten(), dim=0)
    assert cos > 0.98, f"W4A16 输出与 FP16 余弦相似度 {cos:.4f} 过低"
    # 显存压缩：fp16 权重 32768B → packed+scale ≈ 8.5KB，约 1/3.8
    fp16_bytes = linear.weight.numel() * 2
    assert awq.weight_storage_bytes() * 3 < fp16_bytes
    assert awq.weight_storage_bytes() * 5 > fp16_bytes


def test_awq_search_scale_beats_naive_on_outliers():
    rep = awq_pipeline_report(seed=0)
    assert rep["err_ratio"] < 1.0, (
        f"AWQ 输出 MSE 应低于 naive W4: naive={rep['err_naive_w4']:.3e}, "
        f"awq={rep['err_awq_w4']:.3e}"
    )
    assert rep["cos_sim"] > 0.95
    assert 3.0 < rep["compression"] < 5.0


def test_awq_scale_preserves_fp16_output_before_quant():
    """等价缩放 W·diag(s)、X/s 在不量化时必须不改变输出（AWQ 的前提）。"""
    torch.manual_seed(1)
    w, x = torch.randn(16, 128), torch.randn(8, 128)
    s = awq_search_scale(w, x, n_grid=4)
    y1 = x @ w.t()
    y2 = (x / s) @ (w * s).t()
    assert torch.allclose(y1, y2, atol=1e-4)


# ---------- GRPO / PPO（淘天-一面-2 Q1-Q5） ----------


def test_importance_ratio_matches_definition():
    logp_new = torch.tensor([-1.0, -2.0, -0.5])
    logp_old = torch.tensor([-1.1, -1.8, -0.5])
    assert torch.allclose(importance_ratio(logp_new, logp_old), torch.exp(logp_new - logp_old))


def test_ppo_clip_blocks_gradient_outside_trust_region():
    # 正优势 + ratio 过大(>1+eps)：clip 分支生效，梯度为 0
    logp_new = torch.tensor([2.0, 0.5, 0.9, 0.5]).log().requires_grad_(True)
    logp_old = torch.zeros(4)
    adv = torch.tensor([1.0, 1.0, 1.0, -1.0])
    # ratio = [2.0, 0.5, 0.9, 0.5]; eps=0.2 → 区间 [0.8, 1.2]
    surr = ppo_clipped_surrogate(torch.exp(logp_new.detach() - logp_old), adv, 0.2)
    assert surr[0] == 1.2 and surr[1] == 0.5  # A>0: 取 min(2.0, 1.2) / min(0.5, 0.8)
    assert surr[3] == -0.8  # A<0: min(-0.5, -0.8) → clip 托底
    (-ppo_clipped_surrogate(torch.exp(logp_new - logp_old), adv, 0.2)).sum().backward()
    grad = logp_new.grad
    assert grad[0] == 0.0, "ratio>1+eps 且 A>0 时应无梯度（封顶）"
    assert grad[3] == 0.0, "ratio<1-eps 且 A<0 时应无梯度（托底）"
    assert grad[1] != 0.0 and grad[2] != 0.0, "信任域内应有梯度"


def test_grpo_group_advantage_properties():
    r = torch.tensor([0.0, 0.0, 1.0, 1.0])
    adv = grpo_group_advantage(r)
    assert abs(adv.mean().item()) < 1e-6, "组内优势零均值（减去组基线）"
    assert abs(adv.std().item() - 1.0) < 1e-4, "组内标准化"
    assert torch.all(grpo_group_advantage(torch.ones(8)).abs() < 1e-6), "全同奖励 → 零优势"


def test_k3_kl_estimator_unbiased_and_nonnegative():
    torch.manual_seed(0)
    logits_p, logits_q = torch.randn(6), torch.randn(6) * 1.3
    p = torch.softmax(logits_p, -1)
    logp, logq = torch.log(p), torch.log(torch.softmax(logits_q, -1))
    kl_exact = (p * (logp - logq)).sum()
    kl_k3_expectation = (p * k3_kl_estimator(logp, logq)).sum()  # 对 P 取期望
    assert (k3_kl_estimator(logp, logq) >= 0).all(), "k3 逐点非负"
    assert torch.allclose(kl_k3_expectation, kl_exact, atol=1e-5), "k3 对 P 的期望 == 精确 KL"


def test_grpo_loss_matches_manual():
    torch.manual_seed(0)
    n = 10
    logp_new = torch.randn(n)
    logp_old = logp_new - 0.1 * torch.randn(n)
    logp_ref = torch.randn(n)
    mask = torch.ones(n)
    mask[:3] = 0  # 前 3 个是 prompt token，不参与损失
    adv = torch.randn(n)

    got = grpo_loss(logp_new, logp_old, adv, logp_ref, beta=0.04, token_mask=mask)
    ratio = torch.exp(logp_new - logp_old)
    per_token = torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
    pg = -(per_token * mask).sum() / mask.sum()
    kl = torch.exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1
    manual = pg + 0.04 * (kl * mask).sum() / mask.sum()
    assert torch.allclose(got, manual, atol=1e-6)


def test_masked_mean_skips_prompt_tokens():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(masked_mean(x, torch.tensor([0.0, 1.0, 1.0, 1.0])).item(), 3.0)
    assert math.isclose(masked_mean(x, None).item(), 2.5)


def test_ppo_vs_grpo_demo_consistency():
    demo = ppo_vs_grpo_demo()
    assert demo["adv_corr_ppo_grpo"] > 0.9, "组基线与 critic 基线的优势方向应高度一致"
    assert demo["clipped_gain"] < demo["unclipped_gain"], "clip 应压低正优势下的大 ratio 收益"


# ---------- BST 迭代器（淘天-一面-2 Q8） ----------


def _demo_tree() -> TreeNode:
    return TreeNode.from_level_order([7, 3, 15, None, None, 9, 20])


def test_bst_three_iterators_agree():
    root = _demo_tree()
    expect = [3, 7, 9, 15, 20]
    for it_cls in (BSTIteratorFlatten, BSTIteratorStack, BSTIteratorGenerator):
        it = it_cls(root)
        out = []
        while it.hasNext():
            out.append(it.next())
        assert out == expect, f"{it_cls.__name__} 中序序列错误"
    # hasNext 在 next 之后语义正确
    it = BSTIteratorStack(root)
    assert it.hasNext() and it.next() == 3 and it.hasNext() and it.next() == 7


def test_bst_iterator_empty_and_left_skewed():
    assert iterate_all(None) == []
    # 左斜树（树高 = n）：受控栈内存 O(h) 的极端情形
    left_skew = TreeNode.from_level_order([5, 4, None, 3, None, 2, None, 1])
    assert iterate_all(left_skew) == [1, 2, 3, 4, 5]


# ---------- 线性 SVM（实习-一二面 二面 Q6） ----------


def test_hinge_loss_hand_example():
    w = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0])
    X = torch.tensor([[1.0, 0.0], [-2.0, 0.0]])
    y = torch.tensor([1.0, 1.0])  # 第二个样本 margin=-2，hinge=3
    val = hinge_loss(w, b, X, y, C=1.0)
    assert torch.allclose(val, torch.tensor((3.0 + 0.0) / 2 + 0.5), atol=1e-6)


def test_linear_svm_separates_blobs():
    X, y = make_separable_blobs(seed=0)
    svm = LinearSVM(n_features=2, C=1.0, lr=0.1, epochs=300).fit(X, y)
    assert svm.accuracy(X, y) >= 0.99, "线性可分数据上训练精度应≈100%"
    # 收敛后权重应接近法向 [1,1]/√2 方向（两簇中心连线方向）
    cos = torch.nn.functional.cosine_similarity(svm.w, torch.tensor([1.0, 1.0]), dim=0)
    assert cos > 0.9


def test_rbf_kernel_properties():
    X = torch.randn(5, 3)
    K = rbf_kernel(X, X, gamma=0.5)
    assert K.shape == (5, 5) and torch.allclose(K, K.T, atol=1e-6)
    assert torch.allclose(torch.diag(K), torch.ones(5)) and (K > 0).all()
