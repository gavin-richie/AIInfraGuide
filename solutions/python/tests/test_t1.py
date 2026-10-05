"""T1 大厂批次题解测试：快手/美团/蚂蚁/华为/京东/小米/OPPO/vivo/网易/拼多多/B站。

对应文档 docs/interview-answers/60 ~ 63-t1-*.md，
运行：cd solutions/python && python -m pytest tests/test_t1.py -v
"""

import math

import torch

from answers.t1.bst_from_sorted import (
    inorder_values,
    is_height_balanced,
    sorted_array_to_bst,
    sorted_array_to_bst_iterative,
    tree_height,
)
from answers.t1.dilated_conv import (
    dilated_conv2d_naive,
    dilated_conv2d_unfold,
    effective_kernel_size,
    receptive_field_two_layers,
)
from answers.t1.memory_model import (
    DEEPSEEK_MLA,
    QWEN3_8B,
    fit_max_concurrent_requests,
    inference_memory_report,
    kv_cache_bytes,
    mla_equivalent_gqa_heads,
    mla_kv_per_token_per_layer,
    mla_vs_mha_compression,
    weights_bytes,
)
from answers.t1.rl_basics import (
    compute_gae,
    cross_entropy,
    dpo_loss,
    grpo_group_advantage,
    grpo_loss,
    kl_divergence,
    k3_kl_estimator,
    mse_vs_ce_gradient_demo,
    ppo_clipped_surrogate,
    shannon_entropy,
)
from answers.t1.word_ladder import (
    bfs_word_ladder,
    bidirectional_bfs,
    ladder_all_shortest,
)
from answers.t1.zero_bubble import (
    compare_schedules,
    simulate,
    summary_table,
    verify_dependencies,
)


# ---------- 快手-一面：单词接龙 LC127 ----------


def test_word_ladder_basic():
    assert bfs_word_ladder("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
    assert bfs_word_ladder("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0
    assert bfs_word_ladder("a", "a", ["a"]) == 1


def test_word_ladder_bidirectional_matches_bfs():
    import random

    rng = random.Random(0)
    alphabet = "abc"
    words = ["".join(rng.choice(alphabet) for _ in range(4)) for _ in range(200)]
    words = sorted(set(words))
    for end in ("aabb", "bbaa", "cafe"[:4]):
        assert bidirectional_bfs("abab", end, words) == bfs_word_ladder("abab", end, words)


def test_word_ladder_all_shortest():
    paths = ladder_all_shortest("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"])
    assert sorted(paths) == sorted(
        [["hit", "hot", "dot", "dog", "cog"], ["hit", "hot", "lot", "log", "cog"]]
    )


# ---------- 快手-实习-一面-1：有序数组转平衡 BST LC108 ----------


def test_sorted_array_to_bst():
    nums = list(range(1, 16))  # 15 个元素
    root = sorted_array_to_bst(nums)
    assert inorder_values(root) == nums  # BST 性质
    assert is_height_balanced(root)  # AVL 平衡
    assert abs(tree_height(root.left) - tree_height(root.right)) <= 1


def test_sorted_array_to_bst_iterative_equivalent():
    nums = [-10, -3, 0, 5, 9]
    a, b = sorted_array_to_bst(nums), sorted_array_to_bst_iterative(nums)
    assert inorder_values(a) == inorder_values(b) == nums
    assert is_height_balanced(a) and is_height_balanced(b)


def test_sorted_array_to_bst_empty():
    assert sorted_array_to_bst([]) is None
    assert inorder_values(sorted_array_to_bst([7])) == [7]


# ---------- 小米-实习-一面-1：膨胀卷积 ----------


def test_dilated_conv_matches_pytorch():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 12, 12)
    w = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    for dilation in (1, 2, 3):
        ref = torch.nn.functional.conv2d(x, w, bias, dilation=dilation, padding=dilation)
        naive = dilated_conv2d_naive(x, w, bias, dilation=dilation, pad=dilation)
        unfold = dilated_conv2d_unfold(x, w, bias, dilation=dilation, pad=dilation)
        assert torch.allclose(naive, ref, atol=1e-5)
        assert torch.allclose(unfold, ref, atol=1e-5)


def test_dilated_conv_receptive_field():
    assert effective_kernel_size(3, 1) == 3
    assert effective_kernel_size(3, 2) == 5
    assert effective_kernel_size(3, 4) == 9
    # DeepLab 经典堆叠：两层 3x3, dilation 1,2 -> 感受野 7
    assert receptive_field_two_layers(3, 1, 2) == 7


# ---------- 京东/快手/华为/蚂蚁：显存估算与 MLA 换算 ----------


def test_kv_cache_bytes_formula():
    # Llama-2-7B: 32 层 32 头 dim128 fp16, 单请求 4096 token = 恰好 2 GiB
    got = kv_cache_bytes(32, 32, 128, 4096, dtype_bytes=2)
    assert got == 2 * 32 * 32 * 128 * 4096 * 2 == 2 * 1024**3


def test_qwen3_8b_memory_report():
    report = inference_memory_report(QWEN3_8B, batch=1, seq=8192)
    assert abs(report["weights_gb"] - 8.2e9 * 2 / 1024**3) < 0.05  # bf16 ≈ 15.26GB
    # KV = 2*36*8*128*8192*2 = 603MB（GQA 8 头）
    assert abs(report["kv_cache_gb"] - 2 * 36 * 8 * 128 * 8192 * 2 / 1024**3) < 0.05
    assert report["total_gb"] > report["weights_gb"]
    # W8 量化：权重减半
    report_w8 = inference_memory_report(QWEN3_8B, seq=8192, weight_bits=8)
    assert abs(report_w8["weights_gb"] * 2 - report["weights_gb"]) < 0.05


def test_weights_bytes_quantization():
    assert weights_bytes(1e9, 16) == 2 * 10**9  # 2 GB（十进制口径）
    assert weights_bytes(1e9, 4) == 5 * 10**8


def test_mla_gqa_equivalence():
    # DeepSeek-V3: (512+64) 元素 / (2*128 每 GQA 头) = 2.25 个等价 KV head
    assert abs(mla_equivalent_gqa_heads() - 2.25) < 1e-9
    # 每 token 每层 576 元素，fp8 = 576 B
    assert mla_kv_per_token_per_layer() == 576
    assert mla_kv_per_token_per_layer(dtype_bytes=2) == 1152
    # 对比 128 头 MHA：32768/(576) ≈ 56.9 倍压缩
    comp = mla_vs_mha_compression(num_q_heads=128)
    assert abs(comp - 32768 / 576) < 1e-9 and 56 < comp < 57


def test_fit_max_concurrent_requests():
    # 24GB 卡部署 Qwen3-8B bf16：权重 15.26GB + 预留 2.5GB，剩 ~6.2GB，
    # 每请求(4096 token) KV = 2*36*8*128*4096*2B = 0.5625GB -> 11 个并发
    n = fit_max_concurrent_requests(24.0, QWEN3_8B, seq=4096, reserved_gb=2.5)
    assert n == 11
    # KV 量化 fp8 后翻倍
    n_fp8 = fit_max_concurrent_requests(
        24.0, QWEN3_8B, seq=4096, kv_dtype_bytes=1, reserved_gb=2.5
    )
    assert n_fp8 == 2 * n


# ---------- 京东：RL 基础与 PPO/DPO/GRPO ----------


def test_entropy_ce_kl_identity():
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    p = torch.softmax(logits, dim=-1)
    q = torch.softmax(torch.randn(4, 10), dim=-1)
    # H(p,q) = H(p) + KL(p||q)
    lhs = cross_entropy(p, q)
    rhs = shannon_entropy(p) + kl_divergence(p, q)
    assert torch.allclose(lhs, rhs, atol=1e-5)
    # KL >= 0 且 p==q 时为 0；KL 不对称
    assert (kl_divergence(p, p) < 1e-8).all()
    assert (kl_divergence(p, q) >= -1e-8).all()
    assert not torch.allclose(kl_divergence(p, q), kl_divergence(q, p))


def test_k3_estimator_unbiased():
    torch.manual_seed(0)
    n = 200_000
    p_logits = torch.randn(n)
    ref_logits = torch.zeros(n)
    # E_{x~π}[k3] 的均值逼近 KL(π||ref)（单边高斯 KL = 0.5*log(1+σ²)... 这里只验非负与量级）
    est = k3_kl_estimator(p_logits - torch.logsumexp(p_logits, 0), torch.zeros(n))
    assert (est > -1e-6).all()


def test_mse_vs_ce_saturated_gradient():
    z = torch.tensor([-10.0])
    grads = mse_vs_ce_gradient_demo(z, target=1.0)
    # 错误且自信（y=1, σ(-10)≈0）时：MSE 梯度消失，CE 梯度 ≈ -1
    assert abs(grads["grad_mse"].item()) < 1e-3
    assert abs(grads["grad_ce"].item() + 1.0) < 1e-3
    # 接近正确时两者都小
    grads_ok = mse_vs_ce_gradient_demo(torch.tensor([5.0]), target=1.0)
    assert abs(grads_ok["grad_ce"].item()) < 0.01


def test_gae_manual_case():
    # T=3, gamma=λ=1, 无 done: δ_t = r_t + V_{t+1} - V_t, A_t = Σ δ
    rewards = torch.tensor([1.0, 0.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    next_values = torch.tensor([0.5, 0.5, 0.5])
    adv, ret = compute_gae(rewards, values, next_values, dones=torch.zeros(3))
    assert torch.allclose(adv, torch.tensor([2.0, 1.0, 1.0]))
    assert torch.allclose(ret, torch.tensor([2.5, 1.5, 1.5]))
    # done 截断：最后一步终止则不 bootstrap（δ_2 = 1 + 0 - 0.5）
    adv2, _ = compute_gae(rewards, values, next_values, dones=torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(adv2[-1], torch.tensor(0.5))


def test_ppo_clip_behavior():
    adv = torch.tensor([1.0])
    old = torch.tensor([0.0])
    # ratio = exp(Δlogπ) 在 [1-ε, 1+ε] 内：surrogate = ratio·A 正常变化
    small = ppo_clipped_surrogate(torch.log(torch.tensor([1.1])), old, adv, clip_eps=0.2)
    assert abs(small.item() - 1.1) < 1e-6
    # ratio 远超上界且 adv>0：被 clip 在 1+ε，不再随 ratio 增大（信任域）
    big1 = ppo_clipped_surrogate(torch.log(torch.tensor([3.0])), old, adv, clip_eps=0.2)
    big2 = ppo_clipped_surrogate(torch.log(torch.tensor([10.0])), old, adv, clip_eps=0.2)
    assert abs(big1.item() - 1.2) < 1e-6
    assert abs(big1.item() - big2.item()) < 1e-6


def test_dpo_loss_direction():
    # 隐式奖励差越大 loss 越小；反例越大 loss 越大
    loss_good, margin_good = dpo_loss(
        torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-1.0]), torch.tensor([-1.0]), beta=0.1
    )
    loss_bad, margin_bad = dpo_loss(
        torch.tensor([-2.0]), torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]), beta=0.1
    )
    assert margin_good > 0 > margin_bad
    assert loss_good < loss_bad
    # 完全等于参考策略：margin=0, loss=log2
    loss0, margin0 = dpo_loss(
        torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0])
    )
    assert abs(margin0.item()) < 1e-6
    assert abs(loss0.item() - math.log(2)) < 1e-6


def test_grpo_group_advantage():
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 4.0]])
    adv = grpo_group_advantage(rewards)
    # 全相同奖励的组：优势为 0（GRPO 对"组内无差异"无梯度信号）
    assert torch.allclose(adv[0], torch.zeros(3), atol=1e-4)
    # 第二组均值 2、std 2：adv = [-1, 0, 1]
    assert torch.allclose(adv[1], torch.tensor([-1.0, 0.0, 1.0]), atol=1e-4)


def test_grpo_loss_with_kl():
    log_new = torch.tensor([0.0, 0.0])
    log_old = torch.tensor([0.0, 0.0])
    adv = torch.tensor([1.0, -1.0])
    # on-policy 且优势对称：surrogate 部分为 0
    loss0 = grpo_loss(log_new, log_old, adv)
    assert abs(loss0.item()) < 1e-6
    # 策略 == 参考策略时 k3 项为 0
    loss_self = grpo_loss(log_new, log_old, adv, log_probs_ref=log_new, kl_coef=0.1)
    assert abs(loss_self.item()) < 1e-6
    # k3 是逐点可为负的无偏估计（E[k3]=KL>=0），偏离参考策略时 loss 发生变化
    loss_kl = grpo_loss(log_new, log_old, adv, log_probs_ref=torch.tensor([-0.1, -0.1]), kl_coef=0.1)
    assert abs(loss_kl.item() - loss0.item()) > 1e-3


# ---------- 美团-实习：Zero-Bubble 流水线模拟 ----------


def test_zero_bubble_validity():
    for p, m in ((2, 4), (4, 8), (4, 16), (8, 32), (6, 12)):
        for res in compare_schedules(p, m):
            verify_dependencies(res)
            assert res.busy_time == m * 3  # f=b=w=1


def test_1f1b_bubble_formula():
    # 1F1B 首卡气泡 = (p-1)(f+b)（b 为输入梯度段；B/W 合并实现即经典公式）
    for p, m in ((4, 8), (8, 32), (3, 10), (6, 24)):
        res = simulate(p, m, f=1, b=1, w=1, policy="1f1b")
        assert res.max_stage_idle == (p - 1) * (1 + 1)
        assert res.total_time == res.busy_time + (p - 1) * (1 + 1)


def test_gpipe_same_makespan_worse_memory():
    # 本模型里 GPipe 与 1F1B makespan 相同（同为保功调度），但激活驻留 = m
    for p, m in ((4, 8), (8, 32)):
        g, one = simulate(p, m, policy="gpipe"), simulate(p, m, policy="1f1b")
        assert g.total_time == one.total_time
        assert max(g.peak_activations) == m
        assert max(one.peak_activations) <= p


def test_zb_h1_bubble_formula():
    # 论文：ZB-H1 气泡 = (p-1)(f+b-w)；f=b=w 时为 1F1B 的 1/2，
    # 论文"减到 1/3"对应其实验中的 F/B/W 配比
    for p, m in ((4, 8), (8, 32), (6, 12)):
        one = simulate(p, m, policy="1f1b")
        h1 = simulate(p, m, policy="zb_h1")
        assert h1.max_stage_idle == (p - 1) * (1 + 1 - 1)
        assert h1.max_stage_idle < one.max_stage_idle


def test_zb_h2_zero_bubble():
    # m 足够大（>= 2p）时 ZB-H2 所有 stage 零 idle；总时长 = (m+p-1)f + m(b+w)
    for p, m in ((4, 8), (4, 16), (8, 32), (8, 64), (6, 24)):
        h2 = simulate(p, m, policy="zb_h2")
        assert h2.max_stage_idle == 0
        assert h2.total_time == (m + p - 1) * 1.0 + m * (1 + 1)


def test_zero_bubble_ordering():
    # 时间：gpipe == 1f1b > zb_h1 >= zb_h2；ZB 用激活显存换时间
    for p, m in ((4, 8), (8, 32), (6, 24)):
        g, one, h1, h2 = compare_schedules(p, m)
        assert g.total_time == one.total_time
        assert one.total_time > h1.total_time >= h2.total_time
        assert h2.bubble_ratio < one.bubble_ratio
        assert max(h2.peak_activations) >= max(one.peak_activations)


def test_zb_h2_memory_tradeoff():
    # ZB-H2 激活等到延后的 W 才释放：首卡峰值驻留 ~2p-1 量级，大于 1F1B 的 p
    p, m = 8, 32
    one, h2 = simulate(p, m, policy="1f1b"), simulate(p, m, policy="zb_h2")
    assert max(one.peak_activations) <= p
    assert max(h2.peak_activations) > p


def test_summary_table_runs(capsys):
    print(summary_table(p=4, m=16))
    out = capsys.readouterr().out
    for name in ("gpipe", "1f1b", "zb_h1", "zb_h2"):
        assert name in out
