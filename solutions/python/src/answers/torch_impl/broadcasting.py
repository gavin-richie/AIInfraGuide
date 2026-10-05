"""torch 广播 / 视图语义演示 —— 综合面经题库-1 基础题 9 + CUDA 编程题 1/2。

对应面试题:
    - 综合-1 Q9 "torch.repeat 与 torch.expand 的差异"
    - 综合-5 "view vs contiguous"
    - 综合-1 CUDA 编程题 1 "4D tensor broadcast elementwise 乘"
    - 综合-1 CUDA 编程题 2 "(A * B) * C 广播乘"
"""
from __future__ import annotations

import torch


def repeat_vs_expand() -> dict[str, object]:
    """expand: 返回视图(0 stride), 不拷贝数据, 只能扩展大小为 1 的维度;
    repeat: 真实 tile 拷贝, 任意维度可重复, 占用新内存。
    面试要点: expand 后的视图不连续, 做 .view() 会报错, 需 .reshape/.contiguous。"""
    x = torch.arange(3).unsqueeze(0)      # (1, 3)
    e = x.expand(4, 3)                    # 视图, stride=(0,1)
    r = x.repeat(4, 1)                    # 拷贝
    return {
        "expand_stride": tuple(e.stride()),   # (0, 1) —— 广播维度 stride 为 0
        "expand_storage_bytes": e.untyped_storage().nbytes(),
        "repeat_storage_bytes": r.untyped_storage().nbytes(),
        "expand_is_contiguous": bool(e.is_contiguous()),
        "equal_values": bool(torch.equal(e, r)),
    }


def view_vs_contiguous() -> dict[str, object]:
    """view 要求内存连续且形状兼容(总元素一致, 维度切分不跨 stride);
    transpose/expand 后需 contiguous() 拷贝成连续再 view, 或直接用 reshape。"""
    x = torch.arange(12).view(3, 4)
    t = x.transpose(0, 1)                  # (4,3) 非连续
    err = None
    try:
        t.view(-1)
    except RuntimeError as ex:
        err = "RuntimeError(view 需要连续张量)"
    return {
        "transpose_contiguous": bool(t.is_contiguous()),
        "view_after_transpose_error": err,
        "reshape_ok": tuple(t.reshape(-1).shape),
        "contiguous_then_view_ok": tuple(t.contiguous().view(-1).shape),
    }


def broadcast_4d_elementwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """纯 Python 版 4D 广播乘(对应 CUDA 编程题 1 的语义定义):

    从右对齐补 1 → 各维取 max → stride 为 0 的维度即广播维。
    CUDA kernel 里每个线程按输出 index 反解各输入 index(广播维 stride=0)。
    """
    a, b = torch.broadcast_tensors(a, b)
    return a * b


def broadcast_chain(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """(A * B) * C, A:(1,256), B:(256,128), C:(128,256) —— 综合题库-1 CUDA 编程题 2。

    陷阱点: A(1,256) 与 B(256,128) 从右对齐后 256 vs 128 不兼容, 直接 a*b 会
    RuntimeError。可广播路径: A 先与 B^T(128,256) 相乘 → (128,256), 恰好与 C
    同形, 再逐元素乘 C。CUDA kernel 版见文档 02: 广播维 stride=0 的 index 反解。
    """
    if b.shape[1] != a.shape[-1]:
        b = b.t()          # (128, 256)
    ab = a * b             # (1,256)*(128,256) → (128,256)
    return ab * c          # (128,256)


def broadcast_4d_semantics(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> dict[str, object]:
    """返回两形状广播后的结果形状与各输入的"广播 stride"(CUDA kernel 实现所需)。

    做法: 造一个 dummy 张量 expand 到目标形状, 直接读它的 stride——
    广播维 stride 为 0, kernel 中用 ((i * stride) ) 反解输入下标即可。
    """
    shape = torch.broadcast_shapes(a_shape, b_shape)
    strides = []
    for s in (a_shape, b_shape):
        dummy = torch.empty(s)
        strides.append(tuple(dummy.expand(shape).stride()))
    return {"out_shape": tuple(shape), "strides": strides}
