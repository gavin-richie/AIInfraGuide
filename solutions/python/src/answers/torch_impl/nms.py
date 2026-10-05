"""PyTorch 版 NMS（C++ 手写版见 solutions/cpp/src/nms.cpp）。

对应面试题:
    - 蔚来 "C++ 写 NMS+IoU"
    - 综合-6 C++ 编程题 1 "NMS"、CUDA 编程题 5-2 "IoU"
    - 小厂 "ByteTrack"(同族)

NMS 流程: 按 score 降序 → 取最大者保留 → 删除与其 IoU > threshold 的框 → 重复。
"""
from __future__ import annotations

import torch


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> float:
    """两个 (x1,y1,x2,y2) 框的 IoU。

    inter_w = max(0, min(ax2,bx2) - max(ax1,bx1)); inter_h 同理;
    iou = inter / (area_a + area_b - inter)
    """
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms_torch(boxes: torch.Tensor, scores: torch.Tensor, threshold: float = 0.5) -> list[int]:
    """手写 NMS(纯 Python 循环版, 面试要求能写)。boxes: (N,4) xyxy, scores: (N,)。"""
    order = scores.argsort(descending=True).tolist()
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [
            j for j in order
            if iou_xyxy(boxes[i], boxes[j]) <= threshold
        ]
    return keep


def nms_vectorized(boxes: torch.Tensor, scores: torch.Tensor, threshold: float = 0.5) -> list[int]:
    """向量化 NMS(O(N^2) 一次性算 IoU 矩阵), 工程版思路。"""
    if boxes.numel() == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ix1 = torch.maximum(x1[i], x1[rest])
        iy1 = torch.maximum(y1[i], y1[rest])
        ix2 = torch.minimum(x2[i], x2[rest])
        iy2 = torch.minimum(y2[i], y2[rest])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        ious = inter / (areas[i] + areas[rest] - inter)
        order = rest[ious <= threshold]
    return keep
