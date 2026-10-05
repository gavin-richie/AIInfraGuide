"""T4 车企面经专属代码测试（元戎 concat4d/计算图内存、小鹏 旋转矩形/uint8 第 K 大、
蔚来 矩阵旋转、理想 两数之和/最短路径）。"""
import itertools
import random

import pytest
import torch

from answers.t4.basics import (
    min_path_sum, rotate_matrix_general, rotate_matrix_inplace, rotate_matrix_pythonic,
    two_sum,
)
from answers.t4.graph_memory import (
    Op, assign_arena_offsets, build_graph, min_peak_bruteforce, min_peak_greedy,
    peak_memory, peak_no_free,
)
from answers.t4.kth_largest_uint8 import kth_largest_uint8_counting, kth_largest_uint8_torch
from answers.t4.rect_overlap import (
    aabb_overlaps, rect_corners, rotated_rects_overlap_area,
)

# ---------------------------------------------------------------------------
# 元戎：四维张量 Concat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_concat4d_manual_matches_cat(dim):
    from answers.t4.concat4d import concat4d_manual

    torch.manual_seed(0)
    common = [2, 3, 4]  # 非拼接三维
    sizes = [common[:dim] + [s] + common[dim:] for s in (2, 3, 5)]
    ts = [torch.randn(*s) for s in sizes]
    assert torch.equal(concat4d_manual(ts, dim), torch.cat(ts, dim=dim))


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_concat4d_fast_matches_cat(dim):
    from answers.t4.concat4d import concat4d_fast

    torch.manual_seed(1)
    common = [4, 6, 5]
    ts = [torch.randn(*(common[:dim] + [s] + common[dim:])) for s in (3, 7, 1)]
    assert torch.equal(concat4d_fast(ts, dim), torch.cat(ts, dim=dim))


def test_concat4d_single_tensor():
    from answers.t4.concat4d import concat4d_fast, concat4d_manual

    x = torch.randn(2, 2, 2, 2)
    assert torch.equal(concat4d_fast([x], 2), x)
    assert torch.equal(concat4d_manual([x], 0), x)


def test_triton_concat_last_dim():
    triton = pytest.importorskip("triton")
    from answers.t4.concat4d import triton_concat_last_dim

    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")
    torch.manual_seed(2)
    for n, c, h, wa, wb in ((2, 3, 4, 16, 32), (1, 1, 1, 7, 13), (4, 8, 5, 1025, 2000)):
        a = torch.randn(n, c, h, wa, device="cuda")
        b = torch.randn(n, c, h, wb, device="cuda")
        got = triton_concat_last_dim(a, b)
        assert torch.equal(got, torch.cat([a, b], dim=-1))


# ---------------------------------------------------------------------------
# 元戎：计算图最小内存
# ---------------------------------------------------------------------------


def _diamond_graph():
    # a(10) -> c(30), d(40); b(20) -> c
    return build_graph(
        [Op("a", [], 10), Op("b", [], 20), Op("c", ["a", "b"], 30), Op("d", ["a"], 40)]
    )


def test_graph_peak_memory_hand_example():
    g = _diamond_graph()
    # 顺序 a,b,c,d: b 在 c 处释放 → 峰值 80（a=10+c=30+d=40 同时活）
    assert peak_memory(["a", "b", "c", "d"], g) == 80
    # 顺序 a,d,b,c: a 全程活 → 峰值 100
    assert peak_memory(["a", "d", "b", "c"], g) == 100
    # 不释放：全部累加
    assert peak_no_free(g) == 100


def test_graph_greedy_matches_bruteforce_on_diamond():
    g = _diamond_graph()
    peak, order = min_peak_greedy(g)
    best, _ = min_peak_bruteforce(g)
    assert peak == best == 80
    assert sorted(order) == ["a", "b", "c", "d"]


def test_graph_greedy_never_worse_than_nofree_on_random_dags():
    rng = random.Random(42)
    for _ in range(30):
        n = rng.randint(3, 6)
        ops = []
        for i in range(n):
            k = rng.randint(0, min(i, 2))
            inputs = rng.sample([f"t{j}" for j in range(i)], k) if k else []
            ops.append(Op(f"t{i}", inputs, rng.randint(1, 50)))
        g = build_graph(ops)
        peak, _ = min_peak_greedy(g)
        best, _ = min_peak_bruteforce(g)
        assert best <= peak <= peak_no_free(g), (best, peak, peak_no_free(g))


def test_graph_arena_offsets_valid():
    from answers.t4.graph_memory import _annotate_last_use

    g = _diamond_graph()
    _, order = min_peak_greedy(g)
    arena, offsets = assign_arena_offsets(order, g)
    assert arena <= peak_no_free(g)
    # 不变量：时间上活跃区间重叠的 tensor，地址区间不得重叠
    _annotate_last_use(order, g)
    for t1, t2 in itertools.combinations(order, 2):
        s1 = order.index(g.tensors[t1].producer)
        e1 = g.tensors[t1].last_use
        s2 = order.index(g.tensors[t2].producer)
        e2 = g.tensors[t2].last_use
        time_overlap = s1 < e2 and s2 < e1
        if time_overlap:
            addr_overlap = (
                offsets[t1] < offsets[t2] + g.ops[t2].out_bytes
                and offsets[t2] < offsets[t1] + g.ops[t1].out_bytes
            )
            assert not addr_overlap, (t1, t2)


# ---------------------------------------------------------------------------
# 小鹏：任意朝向矩形重叠面积
# ---------------------------------------------------------------------------


def test_rect_overlap_axis_aligned():
    r1 = (0, 0, 4, 4, 0)   # [-2,2]^2
    r2 = (1, 1, 4, 4, 0)   # [-1,3]^2
    assert rotated_rects_overlap_area(r1, r2) == pytest.approx(9.0)


def test_rect_overlap_identical_and_disjoint():
    r = (0, 0, 2, 3, 30)
    assert rotated_rects_overlap_area(r, r) == pytest.approx(6.0)
    assert rotated_rects_overlap_area((0, 0, 2, 2, 45), (10, 10, 2, 2, 0)) == 0.0
    assert aabb_overlaps((0, 0, 4, 4, 0), (1, 1, 4, 4, 0))
    assert not aabb_overlaps((0, 0, 2, 2, 45), (10, 10, 2, 2, 0))


def test_rect_overlap_rotated_45():
    # 单位正方形与其绕中心旋转 45° 的副本：交集为正八边形, 面积 2(√2-1)
    r1 = (0, 0, 1, 1, 0)
    r2 = (0, 0, 1, 1, 45)
    assert rotated_rects_overlap_area(r1, r2) == pytest.approx(2 * (2**0.5 - 1), rel=1e-9)
    # 尺寸放大 2 倍面积放大 4 倍
    assert rotated_rects_overlap_area((0, 0, 2, 2, 0), (0, 0, 2, 2, 45)) == pytest.approx(
        4 * 2 * (2**0.5 - 1), rel=1e-9
    )


def test_rect_corners_counterclockwise():
    pts = rect_corners(0, 0, 2, 4, 0)
    assert [tuple(round(p, 9) for p in q) for q in pts] == [(-1.0, -2.0), (1.0, -2.0), (1.0, 2.0), (-1.0, 2.0)]
    # 逆时针序（叉积同号）
    for i in range(4):
        (x1, y1), (x2, y2), (x3, y3) = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        assert cross > 0


# ---------------------------------------------------------------------------
# 小鹏：uint8 第 K 大（256 桶计数）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 5, 17, 100])
def test_kth_largest_uint8(k):
    torch.manual_seed(3)
    arr = torch.randint(0, 256, (100,), dtype=torch.uint8)
    ref = sorted(arr.tolist(), reverse=True)[k - 1]
    assert kth_largest_uint8_counting(arr, k) == ref
    assert kth_largest_uint8_torch(arr, k) == ref


def test_kth_largest_uint8_duplicates_and_extremes():
    arr = torch.tensor([5, 5, 5, 2, 0, 255], dtype=torch.uint8)
    assert kth_largest_uint8_counting(arr, 1) == 255
    assert kth_largest_uint8_counting(arr, 2) == 5
    assert kth_largest_uint8_counting(arr, 4) == 5
    assert kth_largest_uint8_counting(arr, 6) == 0
    assert kth_largest_uint8_torch(arr, 6) == 0


# ---------------------------------------------------------------------------
# 蔚来：矩阵旋转 / 理想：两数之和、最短路径
# ---------------------------------------------------------------------------


def test_rotate_matrix_variants():
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ref = rotate_matrix_pythonic(m)
    assert ref == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    assert rotate_matrix_general(m) == ref
    m2 = [row[:] for row in m]
    rotate_matrix_inplace(m2)
    assert m2 == ref
    # 非方阵 2x3
    rect = [[1, 2, 3], [4, 5, 6]]
    assert rotate_matrix_general(rect) == [[4, 1], [5, 2], [6, 3]]
    assert rotate_matrix_pythonic(rect) == [[4, 1], [5, 2], [6, 3]]


def test_two_sum():
    assert two_sum([2, 7, 11, 15], 9) == (0, 1)
    assert two_sum([3, 2, 4], 6) == (1, 2)
    assert two_sum([3, 3], 6) == (0, 1)
    with pytest.raises(ValueError):
        two_sum([1, 2], 10)


def test_min_path_sum():
    grid = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
    assert min_path_sum(grid) == 7
    assert min_path_sum([[1]]) == 1
    assert min_path_sum([[1, 2, 3]]) == 6
    assert min_path_sum([[1], [2], [3]]) == 6
