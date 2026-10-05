"""字节跳动面经配套代码。

对应 docs/interview-answers/20~24 篇：

- ``prefix_cache.py``   Prefix Cache（radix tree + LRU，SGLang 风格）—— 校招-1 推理优化 5
- ``sparse_attention.py`` 动态 block-sparse attention（top-k 块 + 局部窗）—— 实习-一二三面 二面 11
- ``moe.py``            MoE 路由统计 / 负载均衡损失 / 路由坍缩检测 —— 实习-一面-1 推理优化 6-8
- ``rlhf.py``           PPO clip / KL(k3) / DPO / GRPO advantage —— 抖音电商训练优化 2/3/6

全部可在 CPU 上运行，测试见 tests/test_bytedance.py。
"""
