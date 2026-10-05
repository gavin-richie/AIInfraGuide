"""百度批次面试题配套实现。

对应 docs/interview-answers/50~53 百度四篇:

    ppo_gae           -- PPO clip 损失 / GAE(lambda) / V-trace 异步修正（一面-3 RL 全卷）
    grpo_variants     -- GRPO 组基线优势 + 序列级 vs token 级平均（一面-3 Q5）
    fa_math           -- FlashAttention 数学推导逐步轨迹（实习-一面-1 Q1）
    kv_cache_quant    -- KV Cache per-token-per-head INT8 量化对拍（实习-一面-3 Q8）
    cuda_allocator    -- caching allocator 设计演示（实习-一面-1 Q6）
    fused_add_rmsnorm -- Triton fused (residual add + RMSNorm)（实习-一面-3 Q20）
    conv_flops        -- Conv2D 参数量/FLOPs 计算（实习-一面-3 Q17）
    algorithms        -- 本批新写算法题（队列/镜像/字符串相乘/去重/层序带层级）
"""
