"""Triton kernel 测试(仅在有 GPU + triton 的环境跑)。"""
import pytest
import torch

triton = pytest.importorskip("triton")

from answers.triton_kernels.softmax import triton_softmax
from answers.triton_kernels.matmul import triton_matmul
from answers.triton_kernels.layernorm import triton_layernorm, triton_rmsnorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")


def test_softmax():
    for n_cols in (37, 128, 1024, 4097):
        x = torch.randn(64, n_cols, device="cuda") * 5
        got = triton_softmax(x)
        ref = torch.softmax(x, dim=-1)
        assert torch.allclose(got, ref, atol=1e-5), n_cols


def test_matmul():
    torch.manual_seed(0)
    for m, n, k in ((64, 64, 64), (128, 256, 32), (33, 17, 129)):
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda")
        ref = a @ b
        # ieee 精度: 与 cuBLAS fp32 严格一致
        got = triton_matmul(a, b, allow_tf32=False)
        assert torch.allclose(got, ref, atol=1e-3), (m, n, k)
        # tf32: Tensor Core 加速, 相对误差 ~1e-3 量级
        got_tf32 = triton_matmul(a, b, allow_tf32=True)
        rel = (got_tf32 - ref).abs().max() / ref.abs().max()
        assert rel < 5e-3, (m, n, k, float(rel))


def test_layernorm_and_rmsnorm():
    x = torch.randn(128, 512, device="cuda")
    w = torch.randn(512, device="cuda")
    b = torch.randn(512, device="cuda")
    got = triton_layernorm(x, w, b, eps=1e-5)
    ref = torch.nn.functional.layer_norm(x, (512,), w, b, eps=1e-5)
    assert torch.allclose(got, ref, atol=1e-4)

    got = triton_rmsnorm(x.float(), w, eps=1e-6)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(got, ref, atol=1e-4)


def test_benchmark_report(capsys):
    """输出 kernel 实测带宽(GPU 上), 供解答文档引用。"""
    x = torch.randn(4096, 4096, device="cuda")
    for _ in range(3):
        triton_softmax(x)
    torch.cuda.synchronize()

    import time

    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    for _ in range(50):
        triton_softmax(x)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 50
    gbps = x.numel() * x.element_size() * 2 / (ms / 1e3) / 1e9  # 读+写
    with capsys.disabled():
        print(f"\n[triton softmax 4096x4096 fp32] {ms:.3f} ms, 有效带宽 {gbps:.0f} GB/s")
