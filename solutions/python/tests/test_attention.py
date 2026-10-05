"""Attention 系列正确性测试(GPU 或 CPU 均可跑)。"""
import pytest
import torch
import torch.nn.functional as F

from answers.attention.mha import MultiHeadAttention, sdpa_reference, naive_softmax_overflow_demo
from answers.attention.gqa import GroupedQueryAttention, kv_cache_size, gqa_matches_mha_when_full_sharing
from answers.attention.mla import MultiHeadLatentAttention, absorb_equivalence_demo
from answers.attention.online_softmax import (
    softmax_stable, online_softmax, online_attention_row, softmax_naive,
)
from answers.attention.flash_attention import flash_attention_v1, flash_attention_v2, flash_matches_reference
from answers.attention.paged_attention import paged_attention_demo


def test_mha_matches_sdpa():
    torch.manual_seed(0)
    mha = MultiHeadAttention(64, 8)
    mha.eval()
    x = torch.randn(2, 16, 64)
    out = mha(x, x, x, causal=True)
    # 参考实现: 手动投影 + F.sdpa
    b, s, e = x.shape
    q = mha.q_proj(x).view(b, s, 8, 8).transpose(1, 2)
    k = mha.k_proj(x).view(b, s, 8, 8).transpose(1, 2)
    v = mha.v_proj(x).view(b, s, 8, 8).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = mha.out_proj(ref.transpose(1, 2).contiguous().view(b, s, e))
    assert torch.allclose(out, ref, atol=1e-5)


def test_mha_cross_attention():
    torch.manual_seed(1)
    mha = MultiHeadAttention(32, 4)
    mha.eval()
    q_in, kv_in = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
    out = mha(q_in, kv_in, kv_in, causal=False)
    assert out.shape == (2, 5, 32)


def test_softmax_overflow_demo():
    res = naive_softmax_overflow_demo()
    assert res["naive_max_exp"] == float("inf")
    assert res["naive_has_nan"]
    assert abs(res["stable_sum"] - 1.0) < 1e-6


def test_online_softmax_matches_stable():
    for n in (1, 7, 8, 9, 33, 100):
        x = torch.randn(n) * 10  # 放大数值考验稳定性
        assert torch.allclose(online_softmax(x), softmax_stable(x), atol=1e-5), n


def test_naive_softmax_overflows():
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    assert torch.isnan(softmax_naive(x)).any()


def test_online_attention_row():
    torch.manual_seed(0)
    q = torch.randn(16)
    k, v = torch.randn(64, 16), torch.randn(64, 16)
    got = online_attention_row(q, k, v)
    scores = (k @ q) / 16**0.5
    ref = torch.softmax(scores, dim=0) @ v
    assert torch.allclose(got, ref, atol=1e-5)


def test_gqa_impl_and_kv_formula():
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(64, 8, n_kv_heads=2)
    gqa.eval()
    x = torch.randn(2, 12, 64)
    assert gqa(x).shape == (2, 12, 64)
    # KV 公式: GQA(2 kv 头) 相对 MHA(8 头) 缓存降为 1/4
    mha_bytes = kv_cache_size(32, 32, 128, 1, 2048, dtype_bytes=2)
    gqa_bytes = kv_cache_size(32, 8, 128, 1, 2048, dtype_bytes=2)
    assert gqa_bytes == mha_bytes // 4
    assert mha_bytes == 2 * 32 * 32 * 128 * 1 * 2048 * 2
    res = gqa_matches_mha_when_full_sharing()
    assert res["allclose"]


def test_flash_v1_v2_match_reference():
    res = flash_matches_reference()
    assert res["both_close"], res


def test_mla_absorb_equivalence():
    res = absorb_equivalence_demo()
    assert res["allclose"], res


def test_mla_naive_forward_shape():
    mla = MultiHeadLatentAttention(d_model=64, n_heads=4, d_head=16, d_latent=32)
    mla.eval()
    x = torch.randn(2, 10, 64)
    assert mla.forward_naive(x).shape == (2, 10, 64)
    # KV cache 只存 latent: 32 维 vs 朴素 MHA 的 4*16=64 维 → 一半
    assert mla.d_latent == mla.n_heads * mla.d_head // 2


def test_paged_attention():
    res = paged_attention_demo()
    assert res["allclose"], res
    assert res["physical_blocks_discrete"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_gpu_attention():
    dev = "cuda"
    mha = MultiHeadAttention(128, 8).to(dev).eval()
    x = torch.randn(4, 256, 128, device=dev)
    assert mha(x, x, x, causal=True).shape == x.shape
    fa = flash_attention_v2(
        torch.randn(2, 4, 512, 64, device=dev),
        torch.randn(2, 4, 512, 64, device=dev),
        torch.randn(2, 4, 512, 64, device=dev),
        causal=True,
    )
    assert fa.shape == (2, 4, 512, 64)
