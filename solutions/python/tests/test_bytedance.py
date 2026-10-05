"""字节跳动面经配套代码测试（CPU 可跑）。

覆盖：prefix_cache / sparse_attention / moe / rlhf，全部与解析公式或朴素参考对拍。
"""
import math

import pytest
import torch

from answers.bytedance.moe import (
    diagnose_routing,
    expert_utilization,
    load_balance_loss,
    routing_entropy,
    routing_stats,
    router_z_loss,
    topk_route,
)
from answers.bytedance.prefix_cache import RadixPrefixCache, simulate_hit_rate
from answers.bytedance.rlhf import dpo_loss, grpo_advantages, k3_kl_estimator, ppo_clip_loss
from answers.bytedance.sparse_attention import (
    block_scores,
    block_sparse_attention,
    dense_attention_with_mask,
    select_blocks,
    sparse_flops,
)


# ------------------------------------------------------------------ prefix cache


class TestRadixPrefixCache:
    def test_insert_and_match_full(self):
        cache = RadixPrefixCache(num_blocks=8, block_size=4)
        seq = list(range(12))  # 3 blocks
        assert cache.insert(seq) == 3
        assert cache.match(seq) == 12  # 完整命中
        assert cache.blocks_in_use == 3

    def test_partial_block_not_matched(self):
        """尾巴不足一块不缓存、不命中——块粒度语义。"""
        cache = RadixPrefixCache(num_blocks=8, block_size=4)
        cache.insert(list(range(10)))  # 2 整块 + 2 个尾巴
        assert cache.blocks_in_use == 2
        assert cache.match(list(range(10))) == 8

    def test_shared_prefix_radix(self):
        """分叉后共享前缀只存一份。"""
        cache = RadixPrefixCache(num_blocks=8, block_size=4)
        base = [1, 2, 3, 4, 5, 6, 7, 8]
        cache.insert(base + [9, 10, 11, 12])  # 3 blocks
        created = cache.insert(base + [99, 98, 97, 96])  # 共享 2 块，只新增 1 块
        assert created == 1
        assert cache.blocks_in_use == 4  # 2 共享 + 2 分叉 = 树里共 4 块
        assert cache.match(base + [9, 10, 11, 12]) == 12
        assert cache.match(base + [99]) == 8  # 第二条只共享前 2 块

    def test_lru_leaf_eviction(self):
        """容量不足时淘汰最久未访问的叶子，被逐出的序列 match 归零。"""
        cache = RadixPrefixCache(num_blocks=2, block_size=4)
        cache.insert([0, 0, 0, 0])  # t1
        cache.insert([1, 1, 1, 1])  # t2
        cache.insert([2, 2, 2, 2])  # t3 容量 2 → 淘汰 t1（最老叶子）
        assert cache.blocks_in_use == 2
        assert cache.match([0, 0, 0, 0]) == 0
        assert cache.match([1, 1, 1, 1]) == 4
        assert cache.match([2, 2, 2, 2]) == 4

    def test_shared_node_not_evicted(self):
        """中间共享节点不能单独淘汰：淘汰叶子时保留公共前缀。"""
        cache = RadixPrefixCache(num_blocks=3, block_size=4)
        common = [7, 7, 7, 7]
        cache.insert(common + [1, 1, 1, 1])  # t1: A + B1
        cache.insert(common + [2, 2, 2, 2])  # t2: A + B2（共享 A）
        cache.insert(common + [3, 3, 3, 3])  # t3: A + B3 → 容量 3 淘汰 B1
        assert cache.match(common) == 4  # 公共前缀仍在
        assert cache.match(common + [2, 2, 2, 2]) == 8

    def test_simulate_hit_rate(self):
        """10 条请求 = 4 块共享 system prompt + 1 块私有；首条冷启动 →
        命中率 = 9×16 / (10×20) = 0.72（稳态 0.8）。"""
        queries = [list(range(100, 116)) + [i] * 4 for i in range(10)]
        rate, _ = simulate_hit_rate(queries, num_blocks=64, block_size=4)
        assert rate == pytest.approx(0.72)


# ---------------------------------------------------------------- sparse attention


class TestBlockSparseAttention:
    def test_block_scores_shape_and_masked_mean(self):
        """块分数 = 块内真实 token pairwise 均值（= 均值池化内积），尾巴块只算真实 token。"""
        q = torch.randn(1, 2, 20, 8)
        k = torch.randn(1, 2, 18, 8)
        s = block_scores(q, k, block_size=8)
        assert s.shape == (1, 2, 3, 3)  # ceil(20/8)=3, ceil(18/8)=3
        assert torch.isfinite(s).all()  # 尾巴块也有有效代表分（无 -inf 屏蔽）
        # 尾巴块 (q 2 个真实 token, k 2 个真实 token) 手算校验
        manual = (q[0, 0, 16:20].mean(0) * k[0, 0, 16:18].mean(0)).sum()
        assert s[0, 0, -1, -1].item() == pytest.approx(manual.item(), rel=1e-4)

    def test_select_blocks_union(self):
        torch.manual_seed(0)
        scores = torch.randn(1, 1, 6, 6)
        mask = select_blocks(scores, top_k=2, n_local=1)
        # 每行至少 top-2 ∪ 窗口(≤3)，top-k 的两列必为 True
        for i in range(6):
            top2 = torch.topk(scores[0, 0, i], 2).indices
            assert mask[0, 0, i, top2].all()
            for j in range(6):
                if abs(i - j) <= 1:
                    assert mask[0, 0, i, j]

    def test_matches_dense_with_same_mask(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 3, 37, 16) for _ in range(3))  # 非整块尾巴
        out, mask = block_sparse_attention(q, k, v, block_size=8, top_k=3, n_local=1)
        ref = dense_attention_with_mask(q, k, v, mask)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_topk_equals_dense_when_all_kept(self):
        torch.manual_seed(1)
        q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
        out, mask = block_sparse_attention(q, k, v, block_size=4, top_k=4, n_local=4)
        assert mask.all()
        dense_ref = dense_attention_with_mask(q, k, v, torch.ones_like(mask))
        assert torch.allclose(out, dense_ref, atol=1e-5)

    def test_flops_saving(self):
        sparse, dense = sparse_flops(4096, 64, block_size=16, top_k=8, n_local=1)
        assert sparse < dense
        assert sparse / dense == pytest.approx((8 + 3) * 16 / 4096)


# --------------------------------------------------------------------------- MoE


class TestMoE:
    def test_routing_stats_manual(self):
        probs = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
        topk = torch.tensor([[0], [1], [0]])
        f, p = routing_stats(probs, topk)
        assert torch.allclose(f, torch.tensor([2 / 3, 1 / 3]))
        assert torch.allclose(p, probs.mean(0))

    def test_load_balance_loss_matches_formula(self):
        """aux loss 与定义式 E·Σf·P 手算对拍；完全均匀时取最小值 1。"""
        probs = torch.softmax(torch.randn(6, 4), dim=-1)
        topk = torch.tensor([[0], [1], [2], [3], [0], [1]])
        f, p = routing_stats(probs, topk)
        expected = 4 * (f * p).sum().item()
        assert load_balance_loss(probs, topk).item() == pytest.approx(expected)

        probs_u = torch.full((8, 4), 0.25)
        topk_u = torch.tensor([[i % 4] for i in range(8)])  # 每专家 2 token，均匀
        assert load_balance_loss(probs_u, topk_u).item() == pytest.approx(1.0)

    def test_collapse_detection(self):
        """80% token 挤到专家 0 + 低熵 → collapsed=True。"""
        logits = torch.full((50, 4), 0.0)
        logits[:, 0] = 10.0  # 几乎 one-hot 到专家 0
        probs = torch.softmax(logits, dim=-1)
        topk = torch.argmax(probs, dim=-1, keepdim=True)
        diag = diagnose_routing(probs, topk, collapse_share=0.5)
        assert diag["collapsed"] is True
        assert diag["routing_entropy"] < 0.1

    def test_balanced_routing_not_flagged(self):
        torch.manual_seed(0)
        logits = torch.randn(64, 4)
        gates, topi = topk_route(logits, top_k=2)
        diag = diagnose_routing(torch.softmax(logits, -1), topi)
        assert diag["collapsed"] is False

    def test_entropy_bounds(self):
        n = 5
        uniform = torch.full((3, n), 1.0 / n)
        assert routing_entropy(uniform) == pytest.approx(math.log(n))
        onehot = torch.eye(n)[:2]
        assert routing_entropy(onehot) == pytest.approx(0.0, abs=1e-6)

    def test_z_loss_value(self):
        logits = torch.zeros(4, 8)  # logsumexp = ln 8
        assert router_z_loss(logits).item() == pytest.approx(math.log(8) ** 2)

    def test_topk_route_renormalized(self):
        torch.manual_seed(0)
        gates, ids = topk_route(torch.randn(10, 6), top_k=2)
        assert gates.shape == ids.shape == (10, 2)
        assert torch.allclose(gates.sum(-1), torch.ones(10))
        assert ids.shape[1] == 2


# -------------------------------------------------------------------------- RLHF


class TestRLHF:
    def test_ppo_clip_manual(self):
        """单样本手算：r=0.5, A=1, ε=0.2 → clip 到 0.8，loss = -min(0.5, 0.8) = -0.5。"""
        logp = torch.tensor([math.log(0.5)])
        old_logp = torch.tensor([0.0])  # π_old = 1
        adv = torch.tensor([1.0])
        assert ppo_clip_loss(logp, old_logp, adv, clip_eps=0.2).item() == pytest.approx(-0.5)

    def test_ppo_clip_direction(self):
        """A<0 时上界 1+ε 生效：r=1.5 → clip 1.2；loss = -min(-1.5, -1.2) = 1.5。"""
        logp = torch.tensor([math.log(1.5)])
        old_logp = torch.tensor([0.0])
        adv = torch.tensor([-1.0])
        assert ppo_clip_loss(logp, old_logp, adv, clip_eps=0.2).item() == pytest.approx(1.5)

    def test_k3_kl(self):
        """q=p 时逐项为 0；按 q 加权求和与解析 KL(q‖p) 对拍（k3 无偏）。"""
        q_logp = torch.log(torch.tensor([0.5, 0.5]))
        p_logp = torch.log(torch.tensor([0.25, 0.75]))
        k3 = k3_kl_estimator(q_logp, p_logp)
        analytic = (torch.exp(q_logp) * (q_logp - p_logp)).sum()
        weighted = (torch.exp(q_logp) * k3).sum()
        assert weighted.item() == pytest.approx(analytic.item(), rel=1e-4)
        same = k3_kl_estimator(q_logp, q_logp)
        assert torch.allclose(same, torch.zeros_like(same), atol=1e-7)

    def test_dpo_loss_manual(self):
        """β=0.1：两条样本 logits=0.2/0.1，loss=-mean(logσ)；隐式奖励与准确率手算。"""
        out = dpo_loss(
            policy_chosen_logp=torch.tensor([2.0, 1.0]),
            policy_rejected_logp=torch.tensor([0.0, 0.0]),
            ref_chosen_logp=torch.tensor([0.0, 0.0]),
            ref_rejected_logp=torch.tensor([0.0, 0.0]),
            beta=0.1,
        )
        expected = -(
            torch.log(torch.sigmoid(torch.tensor(0.2)))
            + torch.log(torch.sigmoid(torch.tensor(0.1)))
        ) / 2
        assert out["loss"].item() == pytest.approx(expected.item(), rel=1e-5)
        assert out["implicit_reward"].tolist() == pytest.approx([0.2, 0.1])
        assert out["accuracy"].item() == pytest.approx(1.0)

    def test_dpo_ref_offsets(self):
        """ref 同步移动时 logits 不变（DPO 学的是相对 ref 的提升）。"""
        torch.manual_seed(0)
        pc = torch.randn(4)
        pr = torch.randn(4)
        base = dpo_loss(pc, pr, torch.zeros(4), torch.zeros(4), beta=0.2)
        shifted = dpo_loss(pc + 3, pr + 3, torch.full((4,), 3.0), torch.full((4,), 3.0), beta=0.2)
        assert torch.allclose(base["loss"], shifted["loss"])

    def test_grpo_advantages_manual(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = grpo_advantages(rewards, group_size=2)
        assert adv.tolist() == pytest.approx([-1.0, 1.0, -1.0, 1.0])

    def test_grpo_zero_std_group(self):
        rewards = torch.tensor([5.0, 5.0, 1.0, 3.0])
        adv = grpo_advantages(rewards, group_size=2)
        assert adv[:2].tolist() == [0.0, 0.0]

    def test_grpo_dr_variant(self):
        rewards = torch.tensor([1.0, 2.0])
        adv = grpo_advantages(rewards, group_size=2, use_std=False)
        assert adv.tolist() == pytest.approx([-0.5, 0.5])

    def test_grpo_invalid_group(self):
        with pytest.raises(ValueError):
            grpo_advantages(torch.tensor([1.0, 2.0, 3.0]), group_size=2)
