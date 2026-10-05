"""Attention 系列手写实现(面试"手写 X"高频题)。

    mha              -- Multi-Head Attention + causal mask + softmax 溢出演示
    gqa              -- GQA/MQA 统一实现 + KV Cache 显存公式
    mla              -- DeepSeek MLA(latent 压缩 + 权重吸收两条前向路径)
    online_softmax   -- Online Softmax 三步推导(单遍流式)
    flash_attention  -- FA v1/v2 分块实现(内外循环交换的对比)
    paged_attention  -- PagedAttention block-table 语义 + decode 寻址
"""
