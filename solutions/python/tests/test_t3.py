"""t3 包测试: T3 芯片/硬件厂面试题的编程实现（英伟达/三星/摩尔线程/沐曦/中兴）。"""
import math

import pytest
import torch

from answers.t3.clock_angle import clock_angle, parse_clock
from answers.t3.first_one import first_one, first_one_general
from answers.t3.gpu_memory_estimator import kv_cache_bytes, training_memory
from answers.t3.minimal_transformer import MinimalTransformer
from answers.t3.outer_product import outer_product
from answers.t3.outlier_quantization import (
    quantize_minmax, quantize_outlier_split, quantize_percentile,
    quantize_per_channel, rmse,
)
from answers.t3.print_in_turn import run_print_in_turn
from answers.t3.two_sum import two_sum


# ---------- 英伟达: 时针分针夹角 ----------

@pytest.mark.parametrize("h,m,expect", [
    (12, 0, 0.0),      # 重合
    (3, 0, 90.0),
    (6, 0, 180.0),     # 正对 → 180
    (9, 0, 90.0),
    (3, 45, 157.5),    # 时针 90+22.5, 分针 270
    (2, 20, 50.0),     # 经典易错: 时针 60+10=70, 分针 120
    (0, 0, 0.0),
])
def test_clock_angle(h, m, expect):
    assert math.isclose(clock_angle(h, m), expect, abs_tol=1e-9)


def test_clock_angle_ms_precision():
    # 3:45:30 → 分钟数 225.5 → 时针 112.75, 分针 1353%360=273 → 160.25
    assert math.isclose(clock_angle(3, 45, 30), 160.25, abs_tol=1e-9)
    # 毫秒让结果连续变化: ±500ms 分针动 3°/小时方向, 差值远小于 0.2°
    a1, a2 = clock_angle(3, 45, 30, 0), clock_angle(3, 45, 30, 500)
    assert 0 < abs(a2 - a1) < 0.2
    assert 0 <= a2 <= 180


def test_parse_clock():
    assert parse_clock("3:45") == {"hour": 3, "minute": 45, "second": 0, "ms": 0}
    assert parse_clock("3:45:30")["second"] == 30
    assert parse_clock("3:45:30.5")["ms"] == 500
    with pytest.raises(ValueError):
        parse_clock("3")


# ---------- 英伟达: 三线程顺序打印(不死锁) ----------

def test_print_in_turn_order():
    out = run_print_in_turn(9, n_threads=3)
    assert out == list(range(1, 10))


def test_print_in_turn_large_and_two_threads():
    assert run_print_in_turn(20, n_threads=3) == list(range(1, 21))
    assert run_print_in_turn(6, n_threads=2) == list(range(1, 7))


def test_print_in_turn_survives_worker_crash():
    # 线程在打印 4 时崩溃: 已打印 1..3, 其余线程应有序退出而不是死锁挂起
    out = run_print_in_turn(9, n_threads=3, fail_at=4)
    assert out == [1, 2, 3]
    # 崩在第一个数: 1 未打印, 不死锁
    assert run_print_in_turn(9, n_threads=3, fail_at=1) == []


# ---------- 三星: 有序 01 串二分 ----------

@pytest.mark.parametrize("s,expect", [
    ("000111", 3), ("0111", 1), ("111", 0), ("1", 0),
    ("000", -1), ("", -1), ("0" * 100 + "1" * 100, 100),
])
def test_first_one(s, expect):
    assert first_one(s) == expect == first_one_general(s)


def test_first_one_log_complexity():
    import random
    random.seed(0)
    for _ in range(50):
        k = random.randint(0, 64)
        s = "0" * k + "1" * (64 - k)
        assert first_one(s) == first_one_general(s)


# ---------- 沐曦: 两数之和 ----------

def test_two_sum():
    assert two_sum([2, 7, 11, 15], 9) == (0, 1)
    assert two_sum([3, 2, 4], 6) == (1, 2)
    assert two_sum([3, 3], 6) == (0, 1)
    with pytest.raises(ValueError):
        two_sum([1, 2], 10)


# ---------- 英伟达: 向量外积语义 ----------

def test_outer_product():
    a, b = torch.tensor([1.0, 2.0, -1.0]), torch.tensor([3.0, 4.0])
    out = outer_product(a, b)
    assert out.shape == (3, 2)
    assert torch.equal(out, torch.outer(a, b))
    assert torch.equal(out, torch.einsum("i,j->ij", a, b))
    assert out[2, 1].item() == -4.0
    with pytest.raises(ValueError):
        outer_product(torch.ones(2, 2), b)


# ---------- 摩尔线程: 极度不均匀分布量化 ----------

def _make_skewed_data(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(256, 128, generator=g) * 2 - 1     # 主体在 [-1, 1]
    x[0, 0] = -10000.0                                 # 离群集中在通道 0 的少量元素
    return x


def test_minmax_quantization_collapses_with_outlier():
    x = _make_skewed_data()
    dq = quantize_minmax(x)
    # scale 被 -10000 撑大 → 正常值全部塌缩到同一个量化级, 信息完全丢失
    normal_dq = dq[:, 2:]
    assert torch.equal(normal_dq, normal_dq[0, 0].expand_as(normal_dq))
    assert rmse(x[:, 2:], normal_dq) > 10


def test_outlier_split_beats_minmax():
    x = _make_skewed_data()
    assert rmse(x, quantize_outlier_split(x)) < 0.05 * rmse(x, quantize_minmax(x))


def test_percentile_trades_outlier_for_inliers():
    x = _make_skewed_data()
    dq = quantize_percentile(x, q=0.999)
    # 正常主体恢复分辨率
    assert rmse(x[:, 2:], dq[:, 2:]) < 0.05
    # 离群值被饱和到 clip 边界, 误差巨大（该方案的代价）
    assert rmse(x[:, :1], dq[:, :1]) > 600


def test_per_channel_localizes_damage():
    x = _make_skewed_data()
    dq = quantize_per_channel(x)
    # 无离群的通道不受 -10000 牵连
    assert rmse(x[:, 2:], dq[:, 2:]) < 0.05
    # 含离群的通道 0 仍被撑爆
    assert rmse(x[:, :1], dq[:, :1]) > 10


# ---------- 沐曦: 手写 Transformer ----------

def test_transformer_shapes_and_causality():
    torch.manual_seed(0)
    model = MinimalTransformer(vocab=16, d_model=32, n_heads=4, n_layers=2)
    idx = torch.randint(0, 16, (2, 10))
    logits = model(idx)
    assert logits.shape == (2, 10, 16)
    # 因果性: 截断后缀不改变前面位置的 logits
    logits_prefix = model(idx[:, :6])
    assert torch.allclose(logits[:, :6], logits_prefix, atol=1e-5)


def test_transformer_learns_next_token_rule():
    torch.manual_seed(0)
    model = MinimalTransformer(vocab=8, d_model=32, n_heads=4, n_layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # 可学习的确定性规则: 每个位置 = 起始 token + 步数 (mod 8), 即"预测当前 token+1"
    start = torch.randint(0, 8, (16, 1))
    idx = (start + torch.arange(12).unsqueeze(0)) % 8
    tgt = idx.roll(-1, dims=1)
    losses = []
    for _ in range(200):
        loss = torch.nn.functional.cross_entropy(
            model(idx).reshape(-1, 8), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5               # loss 至少降一半


# ---------- 中兴: 显存估算 ----------

def test_training_memory_16n_rule():
    m = training_memory(7e9)                          # 7B, fp16+Adam, DDP
    assert math.isclose(m["params_gb"], 14.0, abs_tol=0.01)
    assert math.isclose(m["grads_gb"], 14.0, abs_tol=0.01)
    assert math.isclose(m["optimizer_gb"], 84.0, abs_tol=0.01)
    assert math.isclose(m["total_gb"], 112.0, abs_tol=0.05)  # 16N 法则


def test_training_memory_zero_stages():
    n = 7e9
    assert math.isclose(training_memory(n, zero_stage=1, world_size=8)["total_gb"],
                        28.0 + 84.0 / 8, abs_tol=0.05)
    assert math.isclose(training_memory(n, zero_stage=2, world_size=8)["total_gb"],
                        14.0 + (14.0 + 84.0) / 8, abs_tol=0.05)
    assert math.isclose(training_memory(n, zero_stage=3, world_size=8)["total_gb"],
                        112.0 / 8, abs_tol=0.05)


def test_training_memory_200b_16gpu_is_infeasible_without_offload():
    total = training_memory(200e9, zero_stage=3, world_size=16)["total_gb"]
    per_gpu_budget = 80.0                             # H100 80GB
    assert total > per_gpu_budget                     # → 必须 offload/量化, 见 83 篇 Q5


def test_kv_cache_bytes():
    # 04 篇 Q7 例: Qwen-7B 全注意力头, B=1 S=2048 fp16 → ~1.07 GB
    b = kv_cache_bytes(layers=32, kv_heads=32, head_dim=128, batch=1,
                       seq_len=2048, dtype_bytes=2)
    assert math.isclose(b / 1e9, 1.07, abs_tol=0.01)
    # GQA 8 头 → 1/4
    g = kv_cache_bytes(layers=32, kv_heads=8, head_dim=128, batch=1,
                       seq_len=2048, dtype_bytes=2)
    assert g * 4 == b
