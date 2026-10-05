"""百度批次配套测试: RL(PPO/GAE/V-trace/GRPO 变体)、FA 数学轨迹、KV Cache 量化、
caching allocator、fused add+RMSNorm(Triton, GPU)、Conv 账、算法题。"""
import math

import pytest
import torch

from answers.baidu.ppo_gae import (
    compute_gae, gae_lambda_extremes, clip_gradient_mask, ppo_clip_loss, vtrace_correction,
)
from answers.baidu.grpo_variants import (
    grpo_advantage, grpo_loss, sequence_vs_token_mean_demo, gspo_sequence_ratio,
)
from answers.baidu.fa_math import (
    fa2_forward_with_trace, flash_attention_correctness, causal_block_count, sram_feasibility,
)
from answers.baidu.kv_cache_quant import quantize_kv_per_token_head, attention_with_quant_kv, kv_quant_metrics
from answers.baidu.cuda_allocator import SimpleCachingAllocator, allocator_demo
from answers.baidu.conv_flops import conv2d_params_flops, Conv2DSpec, baidu_interview_case, depthwise_vs_standard
from answers.baidu.algorithms import (
    TreeNode, level_order_with_levels, MyQueue, invert_tree, invert_tree_iterative,
    multiply, remove_duplicates, remove_duplicates_at_most_twice,
)


# ============================== PPO / GAE ==============================

def test_gae_matches_direct_sum():
    """小例子对拍: A_t = sum_i (gamma*lam)^i * delta_{t+i} 的暴力展开。"""
    torch.manual_seed(0)
    T = 8
    rewards = torch.randn(T)
    values = torch.randn(T)
    next_values = torch.cat([values[1:], torch.zeros(1)])
    dones = torch.zeros(T)
    gamma, lam = 0.9, 0.8
    adv, ret = compute_gae(rewards, values, next_values, dones, gamma, lam)

    delta = rewards + gamma * next_values - values
    for t in range(T):                                   # 直接按定义求和
        want = sum((gamma * lam) ** i * delta[t + i].item() for i in range(T - t))
        assert adv[t].item() == pytest.approx(want, abs=1e-5)
    assert torch.allclose(ret, adv + values)


def test_gae_lambda_extremes():
    """lambda=0 -> 一步 TD; lambda=1 -> MC 回报减基线(单 episode)。"""
    torch.manual_seed(1)
    T = 6
    rewards = torch.randn(T)
    values = torch.randn(T)
    next_values = torch.cat([values[1:], torch.zeros(1)])
    dones = torch.zeros(T)
    gamma = 0.95
    td, mc = gae_lambda_extremes(rewards, values, next_values, dones, gamma)

    delta = rewards + gamma * next_values - values
    assert torch.allclose(td, delta, atol=1e-6)          # lambda=0
    for t in range(T):                                   # lambda=1: sum gamma^i r - V(s_t)
        mc_ref = sum(gamma ** i * rewards[t + i].item() for i in range(T - t)) - values[t].item()
        assert mc[t].item() == pytest.approx(mc_ref, abs=1e-5)


def test_gae_episode_boundary():
    """dones=1 截断跨 episode 的信用分配。"""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    next_values = torch.tensor([0.0, 0.0, 0.0])
    dones = torch.tensor([0.0, 1.0, 0.0])                # t=1 是 episode 末尾
    adv, _ = compute_gae(rewards, values, next_values, dones, gamma=1.0, lam=1.0)
    # t=2 新 episode: A_2 = r_2 = 1; t=1: A_1 = r_1 = 1(不再借用 t=2); t=0: A_0 = r_0+r_1 = 2
    assert adv.tolist() == pytest.approx([2.0, 1.0, 1.0])


def test_ppo_clip_one_sided_truncation():
    """A>0 且 r 越上界 / A<0 且 r 越下界时梯度为 0; 反方向越界仍有梯度。"""
    eps = 0.2

    def grad_of(log_ratio, a):
        logp = torch.zeros(4, requires_grad=True)
        logp_old = -log_ratio * torch.ones(4)
        loss = ppo_clip_loss(logp, logp_old, torch.full((4,), float(a)), eps=eps)
        loss.backward()
        return logp.grad

    g = grad_of(0.5, +1.0)      # A>0, r=e^0.5>1.2: 上界截断 -> 梯度 0
    assert torch.allclose(g, torch.zeros(4))
    g = grad_of(-0.5, +1.0)     # A>0, r<0.8: min 选未截断分支 -> 有梯度(正: 增大 r)
    assert (g < 0).all()
    g = grad_of(-0.5, -1.0)     # A<0, r<0.8: 下界截断 -> 梯度 0
    assert torch.allclose(g, torch.zeros(4))
    g = grad_of(0.5, -1.0)      # A<0, r>1.2: 未截断分支有梯度(继续压 r)
    assert (g > 0).all()

    # 掩码口径与 autograd 一致
    logp = torch.log(torch.tensor([1.5, 1.5, 0.7, 0.7]))
    logp_old = torch.zeros(4)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    mask = clip_gradient_mask(logp, logp_old, adv, eps=eps)
    assert mask.tolist() == [True, True, True, True]


def test_vtrace_properties():
    torch.manual_seed(2)
    N = 6
    behavior_logp = torch.randn(N)
    target_logp = behavior_logp + torch.randn(N) * 2      # 明显 stale
    values = torch.randn(N)
    next_values = torch.cat([values[1:], torch.zeros(1)])
    rewards = torch.randn(N)
    vtrace, rho = vtrace_correction(behavior_logp, target_logp, values, next_values, rewards,
                                    gamma=0.99, rho_bar=1.0, c_bar=1.0)
    ratio = torch.exp((target_logp - behavior_logp))
    assert torch.all(rho <= torch.clamp(ratio, max=1.0) + 1e-6)   # rho 被截断
    assert torch.isfinite(vtrace).all()                            # 无爆炸

    # 同策略(rho=c=1)且 V=0: vtrace 退化为折扣回报
    z = torch.zeros(N)
    v2, rho2 = vtrace_correction(z, z, z, torch.zeros(N), rewards, gamma=0.9)
    want = torch.tensor([sum(0.9 ** i * rewards[t + i].item() for i in range(N - t))
                         for t in range(N)])
    assert torch.allclose(v2, want, atol=1e-5)
    assert torch.allclose(rho2, torch.ones(N))


# ============================== GRPO 变体 ==============================

def test_grpo_advantage_standardized():
    rewards = torch.tensor([3.0, 1.0, 2.0, 6.0])
    adv = grpo_advantage(rewards)
    assert adv.mean().abs() < 1e-6                          # 组内零均值
    assert adv.std().item() == pytest.approx(1.0, abs=1e-4)  # 归一
    # 全对/全错组: std=0 -> 优势近 0(DAPO dynamic sampling 过滤对象)
    assert torch.allclose(grpo_advantage(torch.tensor([1.0, 1.0, 1.0])),
                          torch.zeros(3), atol=1e-5)


def test_grpo_loss_sequence_vs_token_mean():
    G, T = 4, 8
    torch.manual_seed(0)
    logp = torch.randn(G, T)
    logp_old = logp - 0.05 * torch.randn(G, T)              # 轻微 off-policy
    adv_seq = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.zeros(G, T)
    mask[0, :2] = 1                                         # 短序列 2 token
    mask[1, :T] = 1                                         # 长序列 8 token
    mask[2, :4] = mask[3, :4] = 1

    seq_loss = grpo_loss(logp, logp_old, adv_seq, mask, reduction="sequence")
    tok_loss = grpo_loss(logp, logp_old, adv_seq, mask, reduction="token")
    assert torch.isfinite(seq_loss) and torch.isfinite(tok_loss)

    # 等长时两种归一化应完全一致
    mask_uniform = torch.ones(G, T)
    assert grpo_loss(logp, logp_old, adv_seq, mask_uniform, "sequence") == \
        pytest.approx(grpo_loss(logp, logp_old, adv_seq, mask_uniform, "token"), abs=1e-6)

    demo = sequence_vs_token_mean_demo()
    assert demo["sequence_mean"] == pytest.approx(1.0)
    assert demo["token_mean"] == pytest.approx(1.0)
    assert demo["long_sequence_token_share"] == pytest.approx(0.9)   # 36/40


def test_grpo_dapo_clip_higher():
    """eps_hi 放宽后, 上界方向的截断阈值右移 —— A>0 的低概率 token 能继续升。"""
    logp = torch.tensor([[0.0]])
    logp_old = torch.tensor([[0.0]])
    adv = torch.tensor([1.0])
    mask = torch.tensor([[1.0]])
    # r 固定 1.0 时两版损失相同(未触界); 触界样本(r=1.25)时 DAPO 版仍有梯度
    logp_hi = torch.log(torch.tensor([[1.25]]))
    base = ppo_clip_loss(logp_hi, torch.zeros(1), adv, eps=0.2)
    dapo = grpo_loss(logp_hi, torch.zeros((1, 1)), adv, mask, reduction="token",
                     eps=0.2, eps_hi=0.28)
    assert base.item() == pytest.approx(-1.2)               # clip 在 1.2
    assert dapo.item() < base.item()                        # 1.25 未被 1.28 截断, 损失更低且梯度在


def test_gspo_sequence_ratio_is_geometric_mean():
    logp = torch.tensor([[math.log(2.0), math.log(8.0)]])   # token 比率 2 与 8
    logp_old = torch.zeros(1, 2)
    mask = torch.ones(1, 2)
    r = gspo_sequence_ratio(logp, logp_old, mask)
    assert r.item() == pytest.approx(math.sqrt(16.0))       # 几何平均 4, 而算术连乘是 16


# ============================== FA 数学推导 ==============================

def test_fa2_trace_matches_sdpa():
    res = flash_attention_correctness()
    assert res["passed"], res


def test_fa2_trace_online_rescaling():
    """轨迹: 每行的 m/l 随块推进单调不减, alpha<=1, 最终 l == softmax 分母。"""
    torch.manual_seed(0)
    S, D = 96, 32
    q, k, v = torch.randn(S, D), torch.randn(S, D), torch.randn(S, D)
    out, trace = fa2_forward_with_trace(q, k, v, block_size=32, causal=False)

    scale = 1.0 / math.sqrt(D)
    logits = (q @ k.T) * scale
    m_full = logits.max(dim=-1).values
    for t in trace:
        assert all(a <= 1.0 + 1e-6 for a in t["alpha"])
    for t0, t1 in zip(trace, trace[1:]):                    # m 逐行单调不减;
        assert all(x <= y + 1e-6 for x, y in zip(t0["m"], t1["m"]))
        # 而 l 会被 alpha 重缩放(可能变小) —— 真不变量是未归一化分母 l*exp(m) 单调增
        raw0 = [li * math.exp(mi) for li, mi in zip(t0["l"], t0["m"])]
        raw1 = [li * math.exp(mi) for li, mi in zip(t1["l"], t1["m"])]
        assert all(x <= y + 1e-3 for x, y in zip(raw0, raw1))
    l_final = torch.tensor(trace[-1]["l"])
    want_l = torch.exp(logits - m_full.unsqueeze(-1)).sum(-1)
    assert torch.allclose(l_final, want_l[:32], atol=1e-4)
    want_out = torch.softmax(logits, dim=-1) @ v
    assert torch.allclose(out[:32], want_out[:32], atol=1e-4)


def test_causal_block_count():
    # 暴力对照: Q 块 i 需要 KV 块 j 当且仅当 j*bc < (i+1)*br(块内存在 k<=q 的位置)
    def brute(seq, br, bc):
        nq, nk = -(-seq // br), -(-seq // bc)
        cnt = 0
        for i in range(nq):
            for j in range(nk):
                if j * bc < (i + 1) * br:
                    cnt += 1
        return cnt

    for seq, br, bc in ((64, 16, 16), (128, 32, 16), (100, 32, 32)):
        full, causal, saved = causal_block_count(seq, br, bc)
        assert causal == brute(seq, br, bc)
        assert saved == full - causal
    assert causal_block_count(64, 16, 16) == (16, 10, 6)    # 1+2+3+4


def test_sram_feasibility():
    assert sram_feasibility(128, 128, 64)                   # 49KB < 192KB
    assert not sram_feasibility(1024, 1024, 128)            # 超出 SRAM


# ============================== KV Cache 量化 ==============================

def test_quantize_kv_roundtrip():
    torch.manual_seed(0)
    kv = torch.randn(2, 4, 16, 32)
    q, s = quantize_kv_per_token_head(kv, bits=8)
    assert q.abs().max() <= 127.5
    assert s.shape == (2, 4, 16, 1)                         # per-token per-head
    err = (q * s - kv).abs().max()
    assert err <= s.max()                                   # 量化误差不超过半个 scale 格 * 2


def test_kv_quant_int8_vs_int4():
    m = kv_quant_metrics()
    assert m["int8_rel_err"] < 0.05, m
    assert m["int8_rel_err"] < m["int4_rel_err"]            # int8 优于 int4
    assert m["int8_last_token_top1_agree"] >= 0.9


# ============================== Caching Allocator ==============================

def test_allocator_demo_roundtrip():
    d = allocator_demo()
    assert d["all_free"]                                    # 全部释放并合并回整块
    assert d["addr_reused"]                                 # 缓存命中且地址复用
    assert d["stats"]["cache_hit"] >= 1
    assert d["stats"]["merge"] >= 1
    assert d["stats"]["split"] >= 1


def test_allocator_reuse_hits_cache():
    """释放后的块被存活邻居钉住, 同尺寸请求直接复用同一地址。"""
    a = SimpleCachingAllocator(1 << 20)
    p1 = a.allocate(2048, "a")
    p2 = a.allocate(2048, "b")                              # 钉住 p1 右邻
    a.free(p1)
    p3 = a.allocate(2048, "c")
    assert p3 == p1                                         # 地址复用
    assert a.stats["cache_hit"] == 1
    a.free(p2)
    a.free(p3)
    assert a.cache_hit_rate() == pytest.approx(1 / 3)        # 3 次分配 1 次命中


def test_allocator_fragmentation_and_merge():
    a = SimpleCachingAllocator(64 * 1024)
    ptrs = [a.allocate(4 * 1024) for _ in range(8)]
    for p in ptrs:
        a.free(p)
    assert a.cached_free_bytes() == 64 * 1024               # 相邻合并回整段
    with pytest.raises(MemoryError):
        a.allocate(1 << 30)
    with pytest.raises(ValueError):
        a.free(ptrs[0])                                     # double free


# ============================== Conv 账 ==============================

def test_conv_baidu_case():
    """原题: 3x3, 64->128, 输出 64x128x128。params=73856, FLOPs~=1.21G(乘加对计 2)。"""
    c = baidu_interview_case()
    assert c["params"] == 3 * 3 * 64 * 128 + 128
    assert c["flops"] == 64 * 128 * 128 * (3 * 3 * 64) * 2
    assert c["flops_giga"] == 1


def test_conv_stride_pad_groups():
    spec = Conv2DSpec(c_in=3, c_out=16, k=3, h_in=32, w_in=32, stride=2, pad=1)
    c = conv2d_params_flops(spec)
    assert (c.h_out, c.w_out) == (16, 16)
    assert c.params == 3 * 3 * 3 * 16 + 16
    dw = conv2d_params_flops(Conv2DSpec(64, 64, 3, 64, 64, groups=64))
    assert dw.params == 3 * 3 * 64 + 64                     # k*k*c + bias


def test_depthwise_saving():
    d = depthwise_vs_standard(c=64, k=3, hw=64)
    assert d["dw_pw_total"] < d["standard"]
    assert 5 <= d["saving_ratio"] <= 9                      # 约 1/k^2 量级(6~8x)


# ============================== 算法题 ==============================

def _tree() -> TreeNode:
    return TreeNode.from_list([3, 9, 20, None, None, 15, 7])


def test_level_order_with_levels():
    out = level_order_with_levels(_tree())
    assert out == [(3, 0), (9, 1), (20, 1), (15, 2), (7, 2)]
    assert level_order_with_levels(None) == []
    # 层序数组本身按层排列: 层号应非严格递增
    lv = [l for _, l in out]
    assert lv == sorted(lv)


def test_my_queue():
    q = MyQueue()
    seq = []
    for x in (1, 2):
        q.push(x)
    seq.append(q.peek())
    seq.append(q.pop())
    q.push(3)
    q.push(4)
    while not q.empty():
        seq.append(q.pop())
    assert seq == [1, 1, 2, 3, 4]                           # FIFO 语义
    # 与 list 对拍交错操作
    q2, ref = MyQueue(), []
    for op in (1, 0, 1, 1, 0, 1, 0, 0):                     # 1=push, 0=pop
        if op:
            q2.push(42)
            ref.append(42)
        else:
            assert q2.pop() == ref.pop(0)


def test_invert_tree():
    root = _tree()
    inv = invert_tree(root)
    assert (inv.val, inv.left.val, inv.right.val) == (3, 20, 9)
    assert (inv.left.left.val, inv.left.right.val) == (7, 15)
    root2 = TreeNode.from_list([4, 2, 7, 1, 3, 6, 9])
    inv2 = invert_tree_iterative(TreeNode.from_list([4, 2, 7, 1, 3, 6, 9]))
    assert level_order_with_levels(inv2) == level_order_with_levels(
        TreeNode.from_list([4, 7, 2, 9, 6, 3, 1]))
    assert root2 and inv2


def test_multiply():
    cases = [("2", "3", "6"), ("123", "456", "56088"), ("0", "99999", "0"),
             ("99999", "99999", str(99999 * 99999)),
             ("12345678901234567890", "98765432109876543210",
              str(12345678901234567890 * 98765432109876543210))]
    for a, b, want in cases:
        assert multiply(a, b) == want
    import random
    random.seed(0)
    for _ in range(50):                                     # 随机对拍
        a = random.randint(0, 10 ** 12)
        b = random.randint(0, 10 ** 12)
        assert multiply(str(a), str(b)) == str(a * b)


def test_remove_duplicates():
    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    k = remove_duplicates(nums)
    assert k == 5 and nums[:k] == [0, 1, 2, 3, 4]
    assert remove_duplicates([]) == 0
    assert remove_duplicates([1]) == 1
    nums2 = [1, 1, 1, 2, 2, 3]
    k2 = remove_duplicates_at_most_twice(nums2)
    assert k2 == 5 and nums2[:k2] == [1, 1, 2, 2, 3]


# ============================== Triton fused add+RMSNorm (GPU) ==============================

triton = pytest.importorskip("triton")

from answers.baidu.fused_add_rmsnorm import (  # noqa: E402
    fused_add_rmsnorm, fused_add_rmsnorm_looped, torch_reference,
)

pytestmark_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_fused_add_rmsnorm_matches_torch():
    torch.manual_seed(0)
    for m, n in ((8, 128), (33, 4096), (4, 4097)):
        x = torch.randn(m, n, device="cuda")
        res = torch.randn(m, n, device="cuda")
        w = torch.randn(n, device="cuda")
        y, new_res = fused_add_rmsnorm(x, res, w)
        y_ref, res_ref = torch_reference(x, res, w)
        assert torch.allclose(y, y_ref, atol=1e-4), (m, n)
        assert torch.allclose(new_res, res_ref, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_fused_add_rmsnorm_looped_large_n():
    """d=8192 超过单块装载, 列块循环两遍扫版本对拍。"""
    torch.manual_seed(0)
    x = torch.randn(16, 8192, device="cuda")
    res = torch.randn(16, 8192, device="cuda")
    w = torch.randn(8192, device="cuda")
    y, new_res = fused_add_rmsnorm_looped(x, res, w, block=1024)
    y_ref, res_ref = torch_reference(x, res, w)
    assert torch.allclose(y, y_ref, atol=1e-4)
    assert torch.allclose(new_res, res_ref, atol=1e-4)
