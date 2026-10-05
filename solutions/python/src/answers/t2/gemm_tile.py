"""GEMM tile 资源预算推演: 为什么每线程算 8x8, 寄存器会不会溢出。

对应面试题:
    - 商汤-一面 2: 单个线程计算 C 矩阵 8x8 个元素的原因
    - 商汤-一面 3: 寄存器是否会发生溢出, 如何应对
    - 商汤-一面 6: 单个 tile 内执行了多少次计算

推演模型(经典 smem tiling GEMM, 对应 solutions/cuda/src/gemm.cu 的 tiled 版):
    一个 CTA 算 (BM, BN) 的 C tile, BK 为 k 步长:
    - 每线程累加器个数 = BM*BN/threads; 256 线程、128x128 tile → 64 个 fp32
      累加寄存器, 排成 8x8 —— 选 8x8 是让"每 k 步 64 次 FMA"只摊 2+2 次
      LDS.128 向量装载, 计算访存指令比 16:1, 足够摊薄 smem 带宽;
    - 寄存器上限 255/thread(Ampere), 超过则 spill 到 local memory(实为
      global), 性能断崖 —— tile 变大前先算这笔账;
    - 共享内存 = stages * (BM*BK + BK*BN) * dtype_bytes, 多级流水(多缓冲)
      用 smem 换全局访存延迟隐藏。
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GemmTileConfig", "tile_resource_report"]


@dataclass(frozen=True)
class GemmTileConfig:
    """CTA 级 GEMM tiling 配置。"""

    BM: int                 # C tile 行数
    BN: int                 # C tile 列数
    BK: int                 # k 方向步长
    threads: int = 256
    stages: int = 4         # smem 多缓冲级数(Ampere 典型 3-4)
    dtype_bytes: int = 2    # fp16/bf16=2, fp32=4, int8=1
    regs_addr: int = 16     # 寻址/循环变量等杂项寄存器(经验值)
    regs_misc: int = 32     # 展开临时量/调度余量(经验值)


def tile_resource_report(cfg: GemmTileConfig) -> dict[str, object]:
    """返回该 tile 配置的每线程寄存器/共享内存/计算量预算与溢出判断。

    对应面试答案的三笔账:
      1. accum_per_thread = BM*BN/threads(8x8 → 64);
      2. est_regs = 累加器 + A/B 片段(双缓冲 x2) + 杂项; spill 判定 >255;
      3. 每 k 步每线程 64 次 FMA、4 次 LDS.128 → 指令比 16:1。
    """
    assert cfg.BM * cfg.BN % cfg.threads == 0, "BM*BN 需被线程数整除"
    accum = cfg.BM * cfg.BN // cfg.threads
    # 线程内累加器排成尽量方的 r x c(如 64 -> 8x8, 128 -> 8x16)
    rows = int(accum**0.5)
    while rows > 1 and accum % rows != 0:
        rows -= 1
    cols = accum // rows
    layout = f"{rows}x{cols}" if rows > 1 else f"1x{accum}"

    # 每 k 步线程需要 A 的 r 个 / B 的 c 个寄存器片段(双缓冲 x2)
    frags_per_thread = 2 * (rows + cols)
    est_regs = accum + frags_per_thread + cfg.regs_addr + cfg.regs_misc

    smem_bytes = cfg.stages * (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * cfg.dtype_bytes

    # 每 k 步: 8x8 -> 64 FMA; LDS.128 一次 16B(8 个 fp16/4 个 fp32)
    elems_per_ld128 = 16 // cfg.dtype_bytes
    lds_a = 2 * max(1, -(-rows // elems_per_ld128))   # 双缓冲 x ceil
    lds_b = 2 * max(1, -(-cols // elems_per_ld128))
    lds128_per_k = lds_a + lds_b
    fma_per_k = accum

    # Ampere(sm_86): 64K regs/SM, 256 线程时 1 CTA 满 256 regs/thread, 2 CTA 需 <=128
    return {
        "accum_regs_per_thread": accum,
        "accum_layout": layout,
        "est_regs_per_thread": est_regs,
        "spill_risk": est_regs > 255,
        "occupancy_hint": (
            "预算内可 2 CTA/SM" if est_regs <= 65536 // cfg.threads // 2 else "预算下仅 1 CTA/SM(以 ptxas -v 实测为准)"
        ),
        "smem_bytes_per_cta": smem_bytes,
        "smem_exceeds_sm86": smem_bytes > 101376,  # sm_86 每 CTA 上限 99KB(101376B)
        "fma_per_thread_per_k": fma_per_k,
        "lds128_per_thread_per_k": lds128_per_k,
        "instr_ratio_fma_vs_lds": round(fma_per_k / lds128_per_k, 1),
        "macs_per_tile_kstep": cfg.BM * cfg.BN * cfg.BK,
    }
