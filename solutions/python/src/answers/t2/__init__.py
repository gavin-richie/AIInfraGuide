"""T2 独角兽批次(MiniMax/商汤/旷视/智谱/阶跃)配套代码。

子模块:
    calibration    -- 量化校准算法: minmax / percentile / KL 散度 / MSE + 量化粒度(智谱真题)
    quantized_gemm -- 低比特 GEMM 误差分析: W8A8 / W4A16 fake-quant(商汤真题)
    gemm_tile      -- GEMM tile 寄存器/共享内存预算推演: 8x8 per-thread 分块(商汤真题)
    scheduling     -- 区间调度/活动选择(阶跃真题)
"""
