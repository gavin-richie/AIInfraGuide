---
title: "面试题解答库"
description: "AI Infra 面试题解答总索引：首批覆盖「综合」题库 10 个文件全部题目与全库高频手写编程题，配可运行代码工程"
pubDate: 2026-10-05
topic: "总览"
order: 0
tags: ["面试解答"]
---

# 面试题解答库

本目录是 [docs/interview](../interview) 面试题的**独立解答库**（不改动原题目文件）。首批覆盖：

- **「综合」题库 10 个文件的全部 ~185 道题**（AI-Infra-一面、AI-Infra-校招-1、综合面经题库 1-7、AI-Infra-面经-1）——与其他公司面经重复度最高的核心题；
- **全库高频手写编程题**（CUDA kernel / Attention 系列 / C++ / 算法题），全部工程化实现并**在 RTX 3090 上实测验证**。

## 目录

| 解答篇 | 内容 | 主要题源 |
|---|---|---|
| [01 CUDA 基础与体系结构](./01-cuda-基础与体系结构.md) | 编程模型/内存层级/warp/bank conflict/Stream/CUDA Graph/TMA/ITS | 校招-1、综合-6、综合-3、面经-1 |
| [02 CUDA 编程题](./02-cuda-编程题.md) | 手写 kernel 15 题：归约/Softmax/GEMM/转置/直方图/scan/广播乘/池化/IoU | 校招-1、综合-6、综合-1、面经-1 |
| [03 GEMM 与算子优化](./03-gemm与算子优化.md) | Roofline/融合方法论/Warp Specialization/NCCL timeout 排查 | AI-Infra-一面、综合-4、面经-1 |
| [04 Attention 与 KV Cache](./04-attention与kvcache.md) | FA v1/v2/v3、GQA/MQA/MLA、PagedAttention、Flash Decoding、KV 优化 | 综合-1/2、AI-Infra-一面 |
| [05 推理引擎与优化](./05-推理引擎与优化.md) | vLLM/Chunked Prefill/PD 分离/Mooncake/DiT/dLLM/训练耗时估算 | 综合-1/3/7 |
| [06 量化](./06-量化.md) | GPTQ/AWQ/SmoothQuant、FP8/FP4、Marlin、校准 | 综合-2/7、面经-1 |
| [07 分布式训练](./07-分布式训练.md) | All-Reduce 推导/ZeRO/TP-SP-PP/EP/NCCL/MoE/长序列 | 综合-3/5/7 |
| [08 编译器与性能分析](./08-编译器与性能分析.md) | TVM/MLIR/多面体/Nsight/Graph Fusion 算法 | 面经-1、综合-5 |
| [09 C++ 与系统基础](./09-cpp与系统基础.md) | 虚函数/智能指针/移动语义/并发/GIL/IPC/coredump | 面-1、综合-6、综合-5 |
| [10 C++ 编程题](./10-cpp编程题.md) | LRU/内存池/单例/智能指针/NMS/Conv2D/双线性/LayerNorm | 综合-6、综合-1、面经-1 |
| [11 深度学习与训练](./11-深度学习与训练.md) | BN/LN/RMSNorm/轻量化网络/蒸馏/优化器/LoRA/训练排障 | 面经-1、综合-7 |
| [12 算法题](./12-算法题.md) | LeetCode 高频 ~45 题：链表/树/DP/滑窗/堆/位运算 | 综合-1/2/5 |

## 配套代码工程

所有编程题的代码在仓库根 [solutions/](../../solutions) 下三个子工程，**不是单文件示例**，可构建、可测试、可复现实测数据：

```
solutions/
├── cpp/     C++17+CMake，10 道手写题（LRU/内存池/智能指针/NMS/Conv2D...），ctest 9/9
├── cuda/    CUDA+CMake(sm_86)，14 类 kernel + CPU 对拍 + --bench 实测，ctest 6/6
└── python/  PyTorch+Triton+算法题，MHA/GQA/MLA/FA v1v2/PagedAttention/LoRA/SFT loss，
             pytest 76/76（GPU 上运行）
```

一键验证（需 RTX 3090 / sm_86 GPU、CUDA 12、PyTorch 2.x）：

```bash
bash solutions/verify_all.sh
```

或分工程运行：

```bash
cd solutions/cpp   && cmake -B build && cmake --build build && ctest --test-dir build
cd solutions/cuda  && cmake -B build && cmake --build build && ctest --test-dir build
cd solutions/python && pip install -e . && pytest tests -v
```

### 实测数据摘录（RTX 3090）

| kernel | naive | 优化 | 加速比 |
|---|---|---|---|
| Reduce 1M floats | atomic 1.962 ms | 树归约+warp shuffle 0.016 ms | **124×** |
| Histogram 1M bytes | 全局原子 0.330 ms | smem 私有化 0.017 ms | **19.9×** |
| GEMM 1024³ FP32 | naive 2166 GFLOPS | smem tiling 2804 GFLOPS | 1.3× |
| Transpose 2048² | naive 548 GB/s | smem+padding 547 GB/s（L2 命中压缩差距，4096² 下 601 vs 537） | 1.0-1.12× |
| Triton fused softmax 4096² | — | 0.163 ms / 825 GB/s | — |

## 说明

- 解答聚焦"要点 + 公式 + 代码引用"的精炼风格，题干标注来源文件；
- 后续批次将逐步补充 T0-T5 各公司面经的专属题目（各篇内已预留同族题的交叉引用）；
- 发现错误欢迎指正：代码以测试为准（工程里每个 kernel/算法都有对拍），文档以代码为准。
