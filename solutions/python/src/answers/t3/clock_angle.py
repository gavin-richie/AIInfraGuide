"""时针与分针的夹角（含毫秒级精度）。

来源: 英伟达-AI-Infra（编程 4，追问毫秒级精度）。

核心公式: 分别算出时针/分针从 12 点整转过的角度再作差,
  hour_deg   = 30*(h%12) + 0.5*m + m/60 的小数部分（时针每分钟走 0.5°）
  minute_deg = 6*m                       （分针每分钟走 6°）
夹角取 min(d, 360-d)。毫秒精度只是把时间统一换算成连续量（分钟数），
公式完全不变——面试考点在于"不要用 5.5*m 这类整分钟离散公式"。
"""
from __future__ import annotations


def clock_angle(hour: int, minute: int, second: int = 0, ms: int = 0) -> float:
    """返回 [0, 180] 的时针-分针夹角（度），支持秒与毫秒。

    >>> clock_angle(3, 0)
    90.0
    >>> clock_angle(3, 45)   # |90 + 22.5 - 270| = 157.5
    157.5
    """
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59 and 0 <= ms <= 999):
        raise ValueError("invalid time component")
    # 统一换算成连续分钟数，公式对任意精度成立
    minutes = (hour % 12) * 60 + minute + second / 60 + ms / 60_000
    hour_deg = 0.5 * minutes          # 360°/12h = 0.5°/min
    minute_deg = 6.0 * minutes        # 360°/60min = 6°/min
    diff = abs(hour_deg - minute_deg) % 360
    return min(diff, 360 - diff)


def parse_clock(s: str) -> dict:
    """解析 "3:45" / "3:45:30" / "3:45:30.500" 三种格式。"""
    parts = s.split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"unsupported format: {s}")
    hour, minute = int(parts[0]), int(parts[1])
    second, ms = 0, 0
    if len(parts) == 3:
        sec = parts[2].split(".")
        second = int(sec[0])
        ms = int(sec[1].ljust(3, "0")[:3]) if len(sec) == 2 else 0
    return {"hour": hour, "minute": minute, "second": second, "ms": ms}
