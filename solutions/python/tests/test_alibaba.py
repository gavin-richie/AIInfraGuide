"""阿里巴巴批次题解测试：AWQ W4A16 / GRPO / BST 迭代器 / SVM。"""
import math

import numpy as np
import pytest
import torch

from answers.alibaba.awq_w4a16 import (
    W4A16Linear, demo, fake_quant_int4, layer_output_error, pack_int4,
    quantize_int4_symmetric, search_awq_scale, unpack_int4,
)
from answers.alibaba.bst_iterator import (
    BSTIteratorFlatten, BSTIteratorMorris, BSTIteratorStack, TreeNode, iter_values,
)
from answers.alibaba.grpo import (
    POLICY_GRAD_EVOLUTION, grpo_loss, group_relative_advantage,
    importance_ratio, k3_kl_estimator, markov_property_example, ppo_clip_loss,
)
from answers.alibaba.svm import LinearSVM


# ---------------------------------------------------------------- AWQ W4A16
def test_int4_pack_roundtrip():
    vals = torch.arange(-8, 8, dtype=torch.int8).reshape(1, -1)
    packed = pack_int4(vals)
    assert packed.dtype == torch.uint8 and packed.shape == (1, 8)
    unpacked = unpack_int4(packed)
    assert unpacked.dtype == torch.int8
    assert torch.equal(unpacked, vals)
    # 低位在前：q[0]=-8 在低 4 位 → 第一个字节低半字节 = 0x8
    assert (packed[0, 0] & 0xF).item() == 8


def test_int4_quantize_dequantize_error():
    torch.manual_seed(0)
    w = torch.randn(256, 512, dtype=torch.float16)
    q, s = quantize_int4_symmetric(w, group_size=128)
    assert q.min() >= -8 and q.max() <= 7
    w_dq = q.float() * s.float().repeat_interleave(128, dim=1)
    # per-group 对称 INT4 相对误差应在百分之几量级
    rel = (w.float() - w_dq).norm() / w.float().norm()
    assert rel < 0.15


def test_awq_equivalent_transform():
    """等价变换恒等式：(X·diag(s)^-1)·(diag(s)·W)^T == X·W^T。"""
    torch.manual_seed(0)
    x = torch.randn(64, 128, dtype=torch.float16)
    w = torch.randn(64, 128, dtype=torch.float16)
    s = torch.rand(128, dtype=torch.float16) + 0.5
    y_left = (x / s) @ (w * s).T
    y_right = x @ w.T
    # fp16 计算的舍入噪声内应一致
    assert torch.allclose(y_left.float(), y_right.float(), rtol=2e-2, atol=1e-2)


def test_w4a16_linear_accuracy_and_memory():
    torch.manual_seed(0)
    layer = torch.nn.Linear(512, 512, dtype=torch.float16)
    x = torch.randn(128, 512, dtype=torch.float16)
    w4 = W4A16Linear(layer)
    err = layer_output_error(layer, w4, x)
    # randn 权重上 per-group 对称 INT4 的理论相对误差约 12%（步长 Δ=2max/15）
    assert err < 0.15
    # 显存：权重 1/4（packed uint8）+ scales 开销
    ratio = w4.weight_nbytes() / (layer.weight.numel() * 2)
    assert 0.2 < ratio < 0.3


def test_awq_beats_rtn_with_outliers():
    """有激活离群值时，AWQ 缩放搜索应不差于 RTN（alpha=0）。"""
    stats = demo(seed=0)
    assert stats["err_awq"] <= stats["err_rtn"] * 1.05
    assert stats["memory_ratio"] < 0.3


def test_awq_scale_search_uses_output_error():
    torch.manual_seed(3)
    layer = torch.nn.Linear(256, 256, dtype=torch.float16)
    x = torch.randn(64, 256, dtype=torch.float16)
    x[:, :4] *= 32.0  # 注入离群通道
    s = search_awq_scale(layer, x)
    assert s.shape == (256,) and (s > 0).all()
    # 离群通道的保护幅度应高于普通通道
    assert s[:4].min() >= s[4:].max() - 1e-3


# ---------------------------------------------------------------- GRPO / PPO
def test_importance_ratio():
    logp = torch.tensor([0.0, 1.0], requires_grad=True)
    logp_old = torch.tensor([0.0, 0.5])
    r = importance_ratio(logp, logp_old)
    assert torch.allclose(r, torch.exp(logp - logp_old).detach())


def test_ppo_clip_zero_ratio_is_vanilla_pg():
    """r=1（θ=θ_old）时 clip 不起作用，loss 退化为 -E[A]。"""
    logp = torch.tensor([0.1, -0.2, 0.3])
    adv = torch.tensor([1.0, -2.0, 0.5])
    loss = ppo_clip_loss(logp, logp.clone(), adv, eps=0.2)
    assert torch.isclose(loss, -adv.mean())


def test_ppo_clip_blocks_overshoot():
    """A>0 且 r 超上界后，目标不再随 r 增大（梯度截断）；A<0 同理压下界。"""
    adv = torch.tensor([1.0])
    loss_hi_1 = ppo_clip_loss(torch.tensor([math.log(1.4)]), torch.tensor([0.0]), adv, eps=0.2)
    loss_hi_2 = ppo_clip_loss(torch.tensor([math.log(5.0)]), torch.tensor([0.0]), adv, eps=0.2)
    assert torch.isclose(loss_hi_1, loss_hi_2)  # 都被 clip 到 1.2

    logp = torch.tensor([math.log(5.0)], requires_grad=True)
    loss = ppo_clip_loss(logp, torch.tensor([0.0]), adv, eps=0.2)
    loss.backward()
    assert logp.grad is not None and logp.grad.abs() < 1e-6  # 梯度归零

    adv_neg = torch.tensor([-1.0])
    loss_lo_1 = ppo_clip_loss(torch.tensor([math.log(0.6)]), torch.tensor([0.0]), adv_neg, eps=0.2)
    loss_lo_2 = ppo_clip_loss(torch.tensor([math.log(0.1)]), torch.tensor([0.0]), adv_neg, eps=0.2)
    assert torch.isclose(loss_lo_1, loss_lo_2)


def test_group_advantage_zero_mean():
    rewards = torch.tensor([1.0, 2.0, 3.0, 6.0])
    adv = group_relative_advantage(rewards)
    assert adv.mean().abs() < 1e-6
    assert adv.argmax() == 3 and adv.argmin() == 0
    adv_no_std = group_relative_advantage(rewards, normalize_std=False)
    assert torch.allclose(adv_no_std, rewards - rewards.mean())


def test_k3_kl_unbiased_and_nonnegative():
    """k3 = r - log r - 1（r=π_ref/π_θ）逐点非负，均值逼近解析 KL。"""
    torch.manual_seed(0)
    n = 200_000
    # x ~ p = N(0,1)，q = N(1,2)：
    # 解析 KL[p||q] = ln(σq/σp) + (σp² + (μp-μq)²)/(2σq²) - 1/2
    kl_analytic = math.log(2.0) + (1.0 + 1.0) / (2 * 4.0) - 0.5
    x = torch.randn(n)
    logp = -0.5 * x**2 - 0.5 * math.log(2 * math.pi)
    mu, sigma = 1.0, 2.0
    logq = -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
    k3 = k3_kl_estimator(logp, logq)
    assert (k3 >= 0).all()
    assert abs(k3.mean().item() - kl_analytic) < 0.05


def test_grpo_loss_finite_and_has_grad():
    torch.manual_seed(0)
    logp = torch.randn(8, requires_grad=True)
    logp_old = logp.detach() - 0.1
    logp_ref = logp.detach() + 0.1
    rewards = torch.randn(8)
    loss, adv = grpo_loss(logp, logp_old, logp_ref, rewards)
    assert torch.isfinite(loss)
    loss.backward()
    assert logp.grad is not None and torch.isfinite(logp.grad).all()


def test_markov_property_example():
    flags = markov_property_example()
    assert flags["markov_process"] and flags["non_markov_process"]
    # 演进脉络表结构完整（供文档引用）
    assert [row[0] for row in POLICY_GRAD_EVOLUTION][-1] == "GRPO"


# ---------------------------------------------------------------- BST 迭代器
def _build_bst(vals):
    def insert(node, v):
        if node is None:
            return TreeNode(v)
        if v < node.val:
            node.left = insert(node.left, v)
        else:
            node.right = insert(node.right, v)
        return node
    root = None
    for v in vals:
        root = insert(root, v)
    return root


def _serialize(node):
    if node is None:
        return None
    return (node.val, _serialize(node.left), _serialize(node.right))


@pytest.mark.parametrize("vals", [
    [5, 3, 8, 1, 4, 7, 9, 2, 6],
    [42],
    list(range(20, 0, -1)),
])
def test_bst_three_iterators_agree(vals):
    root = _build_bst(vals)
    expected = sorted(vals)
    assert list(iter_values(BSTIteratorFlatten(root))) == expected
    assert list(iter_values(BSTIteratorStack(root))) == expected
    assert list(iter_values(BSTIteratorMorris(root))) == expected


def test_bst_morris_restores_tree_structure():
    """Morris 遍历建立的线索必须全部拆除，树结构恢复原样。"""
    root = _build_bst([5, 3, 8, 1, 4])
    before = _serialize(root)
    assert list(iter_values(BSTIteratorMorris(root))) == [1, 3, 4, 5, 8]
    assert _serialize(root) == before


def test_bst_iterator_partial_consume():
    root = _build_bst([10, 5, 15, 3, 7])
    it = BSTIteratorStack(root)
    assert it.hasNext() and it.next() == 3 and it.next() == 5
    assert it.hasNext()


# ---------------------------------------------------------------- SVM
def _make_separable(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 2))
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    X += y[:, None] * 0.3  # 拉开边界
    return X, y.astype(float)


def test_svm_minibatch_separable():
    X, y = _make_separable()
    svm = LinearSVM(n_features=2, lam=1e-3, seed=0)
    losses = svm.fit(X, y, epochs=30, batch_size=16)
    assert (svm.predict(X) == y).mean() == 1.0
    assert losses[-1] < losses[0]
    assert svm.hinge_loss(X, y) < 0.15


def test_svm_fullbatch_gradient_descent():
    X, y = _make_separable(n=100, seed=1)
    svm = LinearSVM(n_features=2)
    losses = svm.fit_fullbatch(X, y, lr=0.1, epochs=500, C=1.0)
    assert (svm.predict(X) == y).mean() == 1.0
    assert losses[-1] < losses[0]


def test_fake_quant_int4_matches_group_quantize():
    torch.manual_seed(1)
    w = torch.randn(64, 256, dtype=torch.float16)
    fake = fake_quant_int4(w, group_size=128)
    q, s = quantize_int4_symmetric(w, group_size=128)
    ref = q.float() * s.repeat_interleave(128, dim=1).float()
    assert torch.allclose(fake, ref)
