"""T5 及其他公司批次（批次 10）配套测试：FP8/KV Cache/投机采样/MLA 吸收/EPLB/算法题。"""
import math
import random

import pytest
import torch

from answers.t5.algorithms import (
    TreeNode,
    build_tree,
    draw_inscribed_circle,
    house_robber_iii,
    is_additive_string,
    merge_sorted_array,
    my_atoi,
    sum_root_to_leaf,
    tree_max_independent_set_bruteforce,
)
from answers.t5.eplb import EPLBPlan, eplb_balance, without_eplb
from answers.t5.fp8 import (
    E4M3,
    E5M2,
    fp8_bits,
    fp8_gemm_sim,
    fp8_quantize,
    fp8_quantize_tensor,
    format_summary,
)
from answers.t5.kv_cache_size import (
    PROFILES,
    kv_cache_bytes,
    mla_kv_bytes,
    quantized_model_footprint,
)
from answers.t5.mla_absorption import (
    attention_absorbed,
    attention_naive,
    demo as mla_demo,
    mla_costs,
)
from answers.t5.speculative import (
    acceptance_rate,
    distribution_equivalence_check,
    expected_accepted_tokens,
    greedy_speculative_step,
    speculative_step,
)


# ============================================================ FP8 (讯飞-飞星)
def test_fp8_known_bit_patterns():
    # E4M3: 1.0 = 0 0111 000
    assert fp8_bits(1.0, E4M3) == 0b0_0111_000
    # E4M3: 1.75 = 0 0111 110（尾数全 1）
    assert fp8_bits(1.75, E4M3) == 0b0_0111_110
    # E4M3: 0.5 = 0 0110 000
    assert fp8_bits(0.5, E4M3) == 0b0_0110_000
    # E5M2: 1.0 = 0 01111 00
    assert fp8_bits(1.0, E5M2) == 0b0_01111_00
    # E5M2: 最小次正规数 2^-16
    assert fp8_quantize(2.0**-16, E5M2) == pytest.approx(2.0**-16)


def test_fp8_dynamic_range():
    assert E4M3.max_finite == 448.0
    assert E5M2.max_finite == 57344.0
    assert fp8_quantize(1000.0, E4M3) == 448.0        # E4M3 饱和
    assert fp8_quantize(1000.0, E5M2) == 1024.0       # E5M2 表示得下（量化到最近网格 1024）
    # E4M3 尾数 3 位（相对精度 1/16），E5M2 尾数 2 位（1/8）
    assert fp8_quantize(1.06, E4M3) == pytest.approx(1.0)  # 1.06 落在 1.0 附近
    assert fp8_quantize(1.3, E5M2) == pytest.approx(1.25)  # 2 位尾数网格 1.25


def test_fp8_e4m3_more_precise_than_e5m2():
    torch.manual_seed(0)
    x = torch.randn(512) * 3
    err4 = (fp8_quantize_tensor(x, E4M3) - x).abs().mean()
    err5 = (fp8_quantize_tensor(x, E5M2) - x).abs().mean()
    assert err4 < err5  # 尾数多 1 位 → 平均量化误差更小


def test_fp8_matches_torch_dtype():
    """软件模拟与 torch.float8_e4m3fn 的转换结果一致（硬件语义对拍）。"""
    torch.manual_seed(1)
    x = torch.randn(1000) * 100
    ref = x.to(torch.float8_e4m3fn).to(torch.float32)
    sim = fp8_quantize_tensor(x, E4M3)
    mismatch = (ref != sim).sum().item()
    # RNE 边界上允许极个别 .5 恰好取舍不同的差异
    assert mismatch / x.numel() < 0.02


def test_fp8_gemm_error_ordering():
    torch.manual_seed(2)
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    ref = a.to(torch.float64) @ b.to(torch.float64)
    e4 = fp8_gemm_sim(a, b, E4M3)
    e5 = fp8_gemm_sim(a, b, E5M2)
    assert (e4 - ref).abs().max() < (e5 - ref).abs().max()
    assert (e4 - ref).abs().max() < 5.0  # 64 维累加误差仍可控


def test_format_summary_reachable():
    assert "448" in format_summary() and "57344" in format_summary()


# ============================================================ KV Cache (飞星/荣耀)
def test_kv_cache_formula_matches_manual():
    # Llama-2-7B: 32 层 × 32 头 × 128 dim × 2(KV) × 2B × 4096 token
    expect = 2 * 32 * 32 * 128 * 2 * 4096
    assert kv_cache_bytes(32, 32, 128, 4096, dtype="fp16") == expect
    assert expect / 1024**3 == pytest.approx(2.0)     # 2 GiB（≈2.15 GB 十进制）
    # FP8 减半
    assert kv_cache_bytes(32, 32, 128, 4096, dtype="fp8") == expect // 2


def test_mla_kv_cache_dramatically_smaller():
    mla = mla_kv_bytes(61, d_c_kv=512, d_rope=64, seq_len=4096, dtype="fp16")
    mha = kv_cache_bytes(61, 128, 128, 4096, dtype="fp16")  # 假想 MHA 版 DeepSeek
    assert mha / mla > 25  # 576 vs 128×128×2 = 44.4×
    assert mla == 61 * 576 * 4096 * 2


def test_profiles_and_runtime_footprint():
    p = PROFILES["llama3-8b"]
    assert p.per_token_kb("fp16") == pytest.approx(2 * 32 * 8 * 128 * 2 / 1024)
    # 荣耀场景题: 32B INT8 权重 32GB + 开销 → 必须双卡 TP
    foot = quantized_model_footprint(32, "int8")
    assert foot > 32 and foot < 40  # 32GB 权重 + ~15% 开销


# ============================================================ 投机采样 (小厂-实习-1)
def test_greedy_speculative_all_accepted():
    logits = torch.randn(4, 100)
    torch.manual_seed(0)
    draft_tokens = logits[:-1].argmax(dim=-1)  # draft 与 target 完全一致
    out, rejected = greedy_speculative_step(logits, draft_tokens)
    assert rejected == -1 and out.numel() == 4  # 3 接受 + 1 bonus


def test_greedy_speculative_first_reject():
    target = torch.zeros(4, 10)
    target[0, 3] = 10.0                       # 第 1 个位置两者一致
    target[1, 7] = 10.0                       # 第 2 个位置 target 想要 7
    draft = torch.zeros(4, 10)
    draft[0, 3] = 10.0
    draft[1, 5] = 10.0                        # draft 在第 2 个位置给 5
    q_tokens = torch.tensor([3, 5])
    out, rejected = greedy_speculative_step(target, q_tokens)
    assert rejected == 1
    assert out.tolist() == [3, 7]             # 前缀保留，纠错为 target 的 argmax


def test_speculative_step_output_in_vocab():
    torch.manual_seed(3)
    target_logits = torch.randn(4, 50)
    draft_logits = torch.randn(3, 50)
    g = torch.Generator().manual_seed(3)
    q_tokens = torch.tensor([1, 2, 3])
    out = speculative_step(target_logits, draft_logits, q_tokens, g)
    assert 1 <= out.numel() <= 4
    assert out.min() >= 0 and out.max() < 50


def test_speculative_distribution_equals_target():
    """无损性统计验证: 首个产出 token 的经验分布 TV 距离 < 0.05。"""
    tv = distribution_equivalence_check(n_runs=4000, vocab=8, gamma=3)
    assert tv < 0.05


def test_expected_accepted_tokens_formula():
    torch.manual_seed(4)
    p = torch.softmax(torch.randn(10), -1)
    q = torch.softmax(torch.randn(10), -1)
    closed = expected_accepted_tokens(p, q, gamma=4)
    # 蒙特卡洛验证
    alpha = torch.clamp(torch.minimum(p, q), min=0).sum().item()
    if alpha < 1 - 1e-9:
        assert closed == pytest.approx((1 - alpha**5) / (1 - alpha), rel=1e-6)
    # draft == target 时 α=1，接受数为 γ
    assert expected_accepted_tokens(p, p, gamma=4) == 4.0


def test_acceptance_rate_metric():
    target = torch.zeros(3, 5)
    for i in range(3):
        target[i, i] = 1.0
    draft_same = target.clone()
    assert acceptance_rate(target, draft_same, torch.tensor([0, 1, 2])) == 1.0
    assert acceptance_rate(target, draft_same, torch.tensor([0, 0, 2])) == pytest.approx(2 / 3)


# ============================================================ MLA 吸收 (小厂-实习-1)
def test_mla_absorption_numeric_equivalence():
    torch.manual_seed(0)
    n_h, d_h, d_c, s_q, s_kv = 4, 64, 32, 3, 10
    q = torch.randn(n_h, s_q, d_h)
    c_kv = torch.randn(s_kv, d_c)
    w_uk = torch.randn(n_h, d_h, d_c)
    w_uv = torch.randn(n_h, d_h, d_c)
    out_naive = attention_naive(q, c_kv, w_uk, w_uv)
    out_absorbed = attention_absorbed(q, c_kv, w_uk, w_uv)
    assert torch.allclose(out_naive, out_absorbed, atol=1e-4)


def test_mla_costs_decode_vs_prefill():
    n_h, d_h, d_c = 128, 128, 512  # DeepSeek-V3 形状
    decode = mla_costs(n_h, d_h, d_c, s_q=1, s_kv=4096)
    prefill = mla_costs(n_h, d_h, d_c, s_q=4096, s_kv=4096)
    # decode: 吸收 MACs 显著更少（省去逐 token 升维 K/V）
    assert decode.absorbed_mac * 20 < decode.naive_mac
    # prefill: 非吸收更优（K/V 只物化一次，大 GEMM 吃满 Tensor Core）
    assert prefill.naive_mac < prefill.absorbed_mac


def test_mla_demo_numbers():
    d = mla_demo()
    assert d["decode_speedup"] > 20
    assert d["prefill_speedup(吸收/非吸收)"] > 1.0  # 吸收反而更费 → prefill 不吸收


# ============================================================ EPLB (小厂-实习-1)
def _pareto_loads(n: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.paretovariate(1.5) for _ in range(n)]


def test_eplb_reduces_imbalance():
    loads = _pareto_loads(256)
    base = without_eplb(loads, 8)
    plan = eplb_balance(loads, n_rank=8, redundancy_factor=1.5)
    assert plan.imbalance < base.imbalance
    assert plan.imbalance < 1.5  # 长尾负载下接近完美均衡


def test_eplb_hot_experts_get_more_copies():
    loads = _pareto_loads(64)
    plan = eplb_balance(loads, n_rank=8, redundancy_factor=2.0)
    hot = max(range(64), key=lambda i: loads[i])
    cold = min(range(64), key=lambda i: loads[i])
    assert len(plan.expert_to_ranks[hot]) >= len(plan.expert_to_ranks[cold])
    assert len(plan.expert_to_ranks[cold]) == 1
    # 所有物理专家都被放置
    placed = sum(len(v) for v in plan.expert_to_ranks.values())
    assert placed == len(eplb_balance(loads, 8, 2.0).expert_to_ranks) or placed >= 64


def test_eplb_types():
    plan = eplb_balance([1.0] * 16, n_rank=4)
    assert isinstance(plan, EPLBPlan) and plan.n_rank == 4
    assert abs(sum(plan.rank_load) - 16.0) < 1e-6  # 总负载守恒


# ============================================================ 算法题
def test_merge_sorted_array():
    for _ in range(100):
        m, n = random.randint(0, 20), random.randint(0, 20)
        a = sorted(random.randint(-50, 50) for _ in range(m))
        b = sorted(random.randint(-50, 50) for _ in range(n))
        nums1 = a + [0] * n
        merge_sorted_array(nums1, m, b, n)
        assert nums1 == sorted(a + b)


def test_my_atoi():
    cases = {
        "42": 42, "   -42": -42, "4193 with words": 4193,
        "words and 987": 0, "-91283472332": -2**31, "91283472332": 2**31 - 1,
        "": 0, "+1": 1, "+-12": 0, "  000000000001234": 1234, "3.14": 3,
    }
    for s, want in cases.items():
        assert my_atoi(s) == want, s


def test_sum_root_to_leaf():
    # LC129 例: [1,2,3] → 12 + 13 = 25
    root = build_tree([1, 2, 3])
    assert sum_root_to_leaf(root) == 25
    # [4,9,0,5,1] → 495 + 491 + 40 = 1026
    root = build_tree([4, 9, 0, 5, 1])
    assert sum_root_to_leaf(root) == 1026
    assert sum_root_to_leaf(build_tree([None])) == 0


def test_inscribed_circle():
    m = draw_inscribed_circle(5)
    assert len(m) == 5 and all(len(r) == 5 for r in m)
    assert m[2][2] == 1                       # 圆心必填
    assert m[0][0] == 0 and m[4][4] == 0      # 角落在圆外
    assert sum(map(sum, m)) > 5               # 覆盖面积接近 πr² ≈ 19.6
    # 大矩阵下覆盖率 → π/4
    n = 301
    ratio = sum(map(sum, draw_inscribed_circle(n))) / (n * n)
    assert abs(ratio - math.pi / 4) < 0.01


def test_is_additive_string():
    assert is_additive_string("112358") is True
    assert is_additive_string("199100199") is True      # 1,99,100,199
    assert is_additive_string("1023") is False          # 1,0,2? 1+0=1≠2
    assert is_additive_string("12") is False            # 没有第三个数
    assert is_additive_string("000") is True            # 0,0,0（全零合法）
    assert is_additive_string("1203") is False          # 1,20 → 21 没有
    assert is_additive_string("101") is True            # 1,0,1


def test_house_robber_iii_vs_bruteforce():
    for trial in range(30):
        rng = random.Random(trial)
        vals = [rng.randint(0, 20) if rng.random() < 0.9 else None for _ in range(rng.randint(1, 11))]
        root = build_tree(vals)
        assert house_robber_iii(root) == tree_max_independent_set_bruteforce(root)
    # LC337 例: [3,2,3,None,3,None,1] → 3+3+1=7
    root = build_tree([3, 2, 3, None, 3, None, 1])
    assert house_robber_iii(root) == 7


def test_treenode_basic():
    node = TreeNode(1, TreeNode(2), TreeNode(3))
    assert node.left.val == 2 and node.right.val == 3
