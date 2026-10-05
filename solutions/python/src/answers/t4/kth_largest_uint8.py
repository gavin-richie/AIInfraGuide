"""uint8 数组第 K 大（小鹏：CUDA 实现，256 桶计数排序思想）。

对应面试题:
    - 小鹏汽车-AI-Infra-实习 编程题 3: "使用 CUDA 实现在 uint8 数组中查找第 K 大的值"

关键观察: 值域只有 0~255 → 不需要比较排序，**256 桶直方图 + 从高到低前缀扫描**
即可 O(n + 256) 得到第 K 大。比快速选择更适合 GPU（无分支、可并行归约）。

CUDA 实现要点（对应 solutions/cuda/src/histogram_scan.cu 的直方图思路）:
    kernel1: 每 block 在 shared memory 建私有 256 桶直方图，末尾一次性
             atomicAdd 合并到全局 hist（冲突域从全 grid 缩到单 block，实测 ~20x 加速）;
    kernel2: 单 block 对 256 个桶做从 255 往 0 的前缀和（exclusive scan，255 侧累计），
             第一个累计和 >= K 的桶的下标即第 K 大的值。
    复杂度 O(n/并行度 + 256)，与 K 无关；数值即下标，天然稳定。

本文件给出 host 侧的等价实现（对拍/教学用），并含 torch 向量化版。
"""
from __future__ import annotations

import torch


def kth_largest_uint8_counting(arr: torch.Tensor, k: int) -> int:
    """256 桶计数版：与 CUDA kernel 逻辑一致的 host 实现。

    arr: uint8 张量（任意形状，内部 flatten）；k 从 1 开始（1 = 最大值）。
    """
    assert arr.dtype == torch.uint8, "值域题，输入必须是 uint8"
    flat = arr.flatten()
    n = flat.numel()
    assert 1 <= k <= n
    hist = torch.bincount(flat.long(), minlength=256)  # kernel1 的全局直方图
    cum = 0
    for v in range(255, -1, -1):  # kernel2 的从高到低前缀扫描
        cum += int(hist[v])
        if cum >= k:
            return v
    raise AssertionError("unreachable")


def kth_largest_uint8_torch(arr: torch.Tensor, k: int) -> int:
    """向量化版（同逻辑）：rev_cum[v] = #{x >= v} 单调不增，第 K 大 = 最大满足
    rev_cum[v] >= k 的 v（即最后一个满足条件的下标）。"""
    hist = torch.bincount(arr.flatten().long(), minlength=256)
    rev_cum = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])  # rev_cum[v] = #{x >= v}
    idx = torch.nonzero(rev_cum >= k).flatten()
    return int(idx[-1].item())
