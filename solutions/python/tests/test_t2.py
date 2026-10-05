"""T2 独角兽批次(MiniMax/商汤/旷视/智谱/阶跃)题解测试。

对应文档 docs/interview-answers/70-t2-算子与量化.md 与 71-t2-训练与算法.md,
运行: cd solutions/python && python -m pytest tests/test_t2.py -v
"""

import itertools
import math
import random

import torch

from answers.t2.calibration import (
    calibration_compare,
    compute_scale,
    fake_quant,
    kl_calibration_scale,
    minmax_scale,
    mse_calibration_scale,
    percentile_scale,
)
from answers.t2.gemm_tile import GemmTileConfig, tile_resource_report
from answers.t2.quantized_gemm import quantized_gemm_report, w8a8_matmul, w4a16_matmul
from answers.t2.scheduling import max_complete_activities


# ================================ 智谱: 校准算法(基础 2/3 + 编程 8) ================================
class TestCalibration:
    def test_minmax_scale_formula(self):
        x = torch.tensor([-3.0, -1.0, 0.5, 7.0])
        s, zp = minmax_scale(x)
        assert math.isclose(s, 7.0 / 127)
        assert zp == 0
        s_asym, zp_asym = minmax_scale(x, symmetric=False)
        assert math.isclose(s_asym, 10.0 / 255)
        assert 76 <= zp_asym <= 77  # -round(min/s) = round(76.5)
        xq = fake_quant(x, s_asym, zp_asym)
        assert (xq - x).abs().max() <= s_asym / 2 + 1e-7

    def test_fake_quant_error_bound(self):
        torch.manual_seed(0)
        x = torch.randn(2048)
        s, _ = minmax_scale(x)
        xq = fake_quant(x, s)
        # 范围内元素误差 <= step/2, 且不超过量化格点边界
        assert (xq - x).abs().max() <= s / 2 + 1e-7
        assert xq.abs().max() <= s * 127 + 1e-6

    def test_percentile_ignores_outliers(self):
        torch.manual_seed(0)
        x = torch.randn(8192)
        x[:8] *= 100  # 千分之一离群点
        s_mm, _ = minmax_scale(x)
        s_pct, _ = percentile_scale(x, q=99.9)
        assert s_pct < s_mm / 10  # 截断离群点后步长小一个量级

    def test_percentile_scale_value(self):
        x = torch.linspace(0.0, 1.0, 1001)
        s, _ = percentile_scale(x, q=99.9)
        assert math.isclose(s, 0.999 / 127, abs_tol=1e-3)

    def test_kl_no_truncation_on_uniform(self):
        torch.manual_seed(0)
        u = torch.rand(200_000)
        s = kl_calibration_scale(u)
        assert s * 127 >= 0.98 * u.max().item()  # 均匀分布无尾部 -> 不截断

    def test_kl_truncates_rare_outliers(self):
        torch.manual_seed(0)
        x = torch.randn(8192)
        x[:8] *= 50
        s = kl_calibration_scale(x)
        assert s * 127 < 0.5 * x.abs().max().item()  # 明显截断尖峰尾部

    def test_kl_moderate_truncation_on_normal(self):
        torch.manual_seed(0)
        x = torch.randn(200_000)
        s = kl_calibration_scale(x)
        t = s * 127
        assert 3.0 < t < x.abs().max().item()  # 温和截断(实测 ~4.5 / absmax ~5)

    def test_mse_calibration_is_argmin_over_grid(self):
        torch.manual_seed(0)
        x = torch.randn(8192)
        s = mse_calibration_scale(x)
        err_s = (fake_quant(x, s) - x).pow(2).mean().item()
        for pct in (100.0, 99.9, 99.0, 98.0):  # 逐个候选都不优于 argmin
            s_i = torch.quantile(x.abs(), pct / 100.0).item() / 127
            err_i = (fake_quant(x, s_i) - x).pow(2).mean().item()
            assert err_s <= err_i + 1e-12

    def test_calibration_compare_outlier_summary(self):
        torch.manual_seed(0)
        x = torch.randn(4096)
        x[torch.rand(4096) < 0.01] *= 20
        r = calibration_compare(x)
        # 离群数据上 percentile 与 KL 都选择截断(阈值低于 absmax)
        assert 0 < r["scale_percentile"] < r["scale_minmax"]
        assert 0 < r["scale_kl"] < r["scale_minmax"]
        assert all(r[f"rel_err_{k}"] > 0 for k in ("minmax", "percentile", "kl", "mse"))


# ============================== 智谱: 量化粒度(基础 1)与商汤: 低比特 GEMM(11) ==============================
class TestGranularity:
    def test_scale_shapes_and_broadcast(self):
        w = torch.randn(64, 256)
        assert compute_scale(w, "tensor").ndim == 0
        assert tuple(compute_scale(w, "channel").shape) == (64, 1)
        sg = compute_scale(w, "group", group_size=32)          # 沿输入维 K 分组
        assert tuple(sg.shape) == (64, 256)
        assert torch.allclose(sg.view(64, 8, 32)[:, :, 0], sg.view(64, 8, 32)[:, :, 1])  # 组内共享

    def test_finer_granularity_less_error(self):
        torch.manual_seed(0)
        w = torch.randn(64, 512)

        def err(s):
            return (fake_quant(w, s, num_bits=8) - w).pow(2).mean().item()

        e_tensor = err(compute_scale(w, "tensor"))
        e_channel = err(compute_scale(w, "channel"))
        e_group = err(compute_scale(w, "group", group_size=16))  # 64 通道分 4 组
        assert e_tensor > e_channel > e_group  # 粒度越细误差越小(最细=per-group)


class TestQuantizedGemm:
    def test_w8a8_small_case(self):
        a = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        w = torch.tensor([[2.0, -1.0], [-4.0, 0.25]])
        y = w8a8_matmul(a, w)
        rel = ((y - a @ w.T).norm() / (a @ w.T).norm()).item()
        assert rel < 0.05

    def test_w8a8_and_w4a16_error_report(self):
        r = quantized_gemm_report(seed=0)
        assert r["rel_err_w8a8"] < 0.02          # INT8 点积 ~1.4%(理论 sqrt(2)*0.9%)
        assert r["rel_err_w8a8"] < r["rel_err_w4a16_g128"]
        assert r["rel_err_w4a16_g128"] < 0.25    # INT4 无结构权重 ~20%
        assert r["bytes_per_weight_w4a16"] == 0.5  # W4 的收益在带宽

    def test_w4a16_group_scale_exact_on_grid(self):
        # 每组(group_size=8 即每行) amax=7、s=1: 权重本身是整数格点, 量化无损
        w = (torch.arange(64, dtype=torch.float32).view(8, 8) % 15 - 7)
        w[:, 0] = 7.0  # 保证每行 amax=7
        a = torch.randn(4, 8)
        y = w4a16_matmul(a, w, group_size=8)
        assert torch.allclose(y, a @ w.T, atol=1e-5)


# ================================ 商汤: GEMM tile 与寄存器预算(2/3/6) ================================
class TestGemmTile:
    def test_classic_128x128_is_8x8(self):
        rep = tile_resource_report(GemmTileConfig(BM=128, BN=128, BK=64, threads=256))
        assert rep["accum_layout"] == "8x8"
        assert rep["accum_regs_per_thread"] == 64
        assert rep["fma_per_thread_per_k"] == 64        # 每 k 步 64 次 FMA
        assert rep["lds128_per_thread_per_k"] == 4      # A/B 各 2 次 LDS.128
        assert rep["instr_ratio_fma_vs_lds"] == 16.0    # 计算访存指令比 16:1
        assert not rep["spill_risk"]
        assert rep["macs_per_tile_kstep"] == 128 * 128 * 64

    def test_oversized_tile_spills(self):
        rep = tile_resource_report(GemmTileConfig(BM=256, BN=256, BK=64, threads=256))
        assert rep["accum_regs_per_thread"] == 256  # 16x16 累加器独占满寄存器上限
        assert rep["spill_risk"]

    def test_small_tile_underutilizes(self):
        rep = tile_resource_report(GemmTileConfig(BM=64, BN=64, BK=64, threads=256))
        assert rep["accum_layout"] == "4x4"
        assert rep["instr_ratio_fma_vs_lds"] == 4.0  # 计算密度不足

    def test_smem_multibuffer_accounting(self):
        cfg = GemmTileConfig(BM=128, BN=128, BK=64, threads=256, stages=3, dtype_bytes=2)
        rep = tile_resource_report(cfg)
        assert rep["smem_bytes_per_cta"] == 3 * (128 * 64 + 64 * 128) * 2


# ================================ 阶跃星辰: 区间调度(编程 7) ================================
class TestScheduling:
    def test_basic_selection(self):
        shows = [(1, 3), (2, 5), (4, 7), (1, 8), (5, 9), (8, 10)]
        assert max_complete_activities(shows) == [(1, 3), (4, 7), (8, 10)]

    def test_empty_and_single(self):
        assert max_complete_activities([]) == []
        assert max_complete_activities([(5, 9)]) == [(5, 9)]

    def test_back_to_back_not_conflict(self):
        assert len(max_complete_activities([(0, 1), (1, 2), (2, 3)])) == 3

    def test_greedy_matches_bruteforce(self):
        random.seed(0)
        for _ in range(200):
            n = random.randint(1, 8)
            shows = []
            for _ in range(n):
                s, e = sorted(random.randint(0, 10) for _ in range(2))
                shows.append((s, e))
            greedy = len(max_complete_activities(shows))
            best = 0
            for r in range(n, 0, -1):
                for comb in itertools.combinations(shows, r):
                    last_end = -1
                    ok = all(s >= last_end and (last_end := e) >= s for s, e in sorted(comb, key=lambda t: t[1]))
                    if ok:
                        best = max(best, r)
            assert greedy == best
