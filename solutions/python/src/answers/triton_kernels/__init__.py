"""Triton kernel 手写集(需 GPU + triton)。

    softmax    -- fused row softmax
    matmul     -- 分块 GEMM(tl.dot 走 Tensor Core)
    layernorm  -- fused LayerNorm / RMSNorm
"""
