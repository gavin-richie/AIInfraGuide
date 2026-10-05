"""任意朝向两矩形的重叠面积（小鹏：不平行于坐标轴的矩形求交）。

对应面试题:
    - 小鹏汽车-AI-Infra-一面 编程题 6:
      "计算平面上两个任意朝向矩形（不平行于坐标轴）的重叠面积"

思路: 任意朝向矩形是凸四边形 → 两凸多边形的交集仍是凸多边形 →
      Sutherland-Hodgman 逐边裁剪：用矩形 B 的 4 条半平面依次裁剪 A 的顶点序列，
      交集顶点求出后用鞋带公式算面积（不相交时面积为 0）。

    快速否判（面试加分）: 先做轴对齐包围盒（AABB）粗筛 + SAT 分离轴：
    两凸多边形不相交当且仅当存在一条分离轴（两矩形共 4+4=8 条候选法向）。
"""
from __future__ import annotations

import math


Point = tuple[float, float]


def rect_corners(cx: float, cy: float, w: float, h: float,
                 angle_deg: float = 0.0) -> list[Point]:
    """中心 (cx,cy)、宽高 (w,h)、逆时针旋转 angle_deg 的矩形四个顶点（逆时针序）。"""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [
        (cx + ca * dx - sa * dy, cy + sa * dx + ca * dy)
        for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
    ]


def _clip_halfplane(poly: list[Point], p: Point, q: Point) -> list[Point]:
    """用直线 pq 的左侧半平面（多边形顶点为逆时针序）裁剪 poly。"""
    px, py = p
    nx, ny = q[1] - p[1], -(q[0] - p[0])  # 边方向逆时针 → 外法向为 (dy, -dx) 的右侧判定
    out: list[Point] = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        d_cur = (cur[0] - px) * nx + (cur[1] - py) * ny
        d_nxt = (nxt[0] - px) * nx + (nxt[1] - py) * ny
        if d_cur <= 0:  # 在保留侧
            out.append(cur)
        if (d_cur < 0 < d_nxt) or (d_nxt < 0 < d_cur):  # 跨越边界 → 交点
            t = d_cur / (d_cur - d_nxt)
            out.append((cur[0] + t * (nxt[0] - cur[0]),
                        cur[1] + t * (nxt[1] - cur[1])))
    return out


def convex_intersection(a: list[Point], b: list[Point]) -> list[Point]:
    """两凸多边形（逆时针顶点序）的交集多边形，可能为空。"""
    poly = a
    n = len(b)
    for i in range(n):
        if not poly:
            return []
        poly = _clip_halfplane(poly, b[i], b[(i + 1) % n])
    return poly


def polygon_area(poly: list[Point]) -> float:
    """鞋带公式（逆时针为正）。"""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def rotated_rects_overlap_area(
    r1: tuple[float, float, float, float, float],
    r2: tuple[float, float, float, float, float],
) -> float:
    """r = (cx, cy, w, h, angle_deg)。返回两任意朝向矩形的重叠面积（不相交为 0）。"""
    a = rect_corners(*r1)
    b = rect_corners(*r2)
    inter = convex_intersection(a, b)
    return polygon_area(inter) if len(inter) >= 3 else 0.0


def aabb_overlaps(r1: tuple, r2: tuple) -> bool:
    """AABB 粗筛：包围盒不相交则矩形必不相交（面试口头补充的加速）。"""
    def aabb(r):
        pts = rect_corners(*r)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)
    ax0, ay0, ax1, ay1 = aabb(r1)
    bx0, by0, bx1, by1 = aabb(r2)
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
