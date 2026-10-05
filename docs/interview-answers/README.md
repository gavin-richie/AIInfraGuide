---
title: "面试题解答库"
description: "AI Infra 面试题解答总索引：综合题库 185 题全解 + 字节/阿里/腾讯/百度与 T2-T5 公司面经 1000+ 题解，配可运行三语言代码工程（cpp 10 / cuda 9 / python 277 项测试）"
pubDate: 2026-10-05
topic: "总览"
order: 0
tags: ["面试解答"]
---

# 面试题解答库

本目录是 [docs/interview](../interview) 面试题的**独立解答库**（不改动原题目文件）。当前覆盖：

- **「综合」题库 10 个文件的全部 ~185 道题**（AI-Infra-一面、AI-Infra-校招-1、综合面经题库 1-7、AI-Infra-面经-1）——与其他公司面经重复度最高的核心题；
- **T0-T5 公司面经批次**（批次 2-10）：字节跳动、阿里巴巴、腾讯（含 TEG）、百度、T2 AI 独角兽（商汤/旷视/智谱/MiniMax/阶跃）、T3 芯片/硬件厂（英伟达/三星/寒武纪/沐曦/燧原/飞腾/中兴等）、T4 车企/自动驾驶（元戎/小马/蔚来/小鹏/理想等）、T5 长尾公司（讯飞/海康/传音/米哈游等）——合计 **1000+ 道题目解答**，同族题交叉引用 01-12 篇、公司特有追问完整展开；
- **全库高频手写编程题**（CUDA kernel / Attention 系列 / C++ / 算法题），全部工程化实现并**在 RTX 3090 上实测验证**。

## 目录

### 综合题库（01-12）

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

### 公司题解（20-104，批次 2-10）

| 解答篇 | 内容 | 主要题源 |
|---|---|---|
| [20 字节 · 推理与量化](./20-字节跳动-推理与量化.md) | KV Cache/PagedAttention/vLLM/Orca/Prefix Cache/MoE 推理与专家分工诊断/Agent 模块 | 字节 AML/豆包/抖音/实习面经 |
| [21 字节 · CUDA 与算子](./21-字节跳动-cuda与算子.md) | CUDA 优化策略/卷积/共享内存/kernel fusion/手写 GEMM 与 MLA/Sparse Attention kernel | 字节校招/实习一二三面 |
| [22 字节 · C++ 与系统](./22-字节跳动-cpp与系统.md) | vtable/内存对齐/动静态库/cache line/页表 TLB/原子操作/调试工具链 | 字节实习一二三面/AML |
| [23 字节 · 训练与模型基础](./23-字节跳动-训练与模型基础.md) | DeepSpeed/ZeRO/分布式对比/BLIP/CLIP/ViT/Swin/优化器/Qwen 演进/Decoder-only | 抖音电商/豆包 |
| [24 字节 · RLHF 与场景题](./24-字节跳动-rlhf与场景题.md) | SFT 数据流程/PPO/DPO 对比/GRPO/微调评估/刷赞识别/项目方法论/编程题总索引 | 抖音电商/豆包 |
| [30 阿里 · 推理与量化](./30-阿里-推理与量化.md) | 推理瓶颈/vLLM/MoE 架构与路由/TP-EP 并行/AWQ W4A16 完整实现/互联拓扑与硬件趋势 | 阿里云/淘天/控股集团 |
| [31 阿里 · 训练与算法](./31-阿里-训练与算法.md) | GRPO/PPO 强化学习/预训练长上下文/蒸馏/Scaling Law/经典 ML（SVM 等）/推荐系统 | 淘天/AI-Infra 校招 |
| [32 阿里 · 多模态与 Agent](./32-阿里-多模态与agent.md) | Qwen-VL 训练流程/多模态对齐/RAG/Agent 记忆编排与可观测/MCP 评测/Claude Code 工程 | 阿里云/淘天 |
| [33 阿里 · 算子与系统](./33-阿里-算子与系统.md) | Warp Specialization/kernel fusion/MLIR-TVM/GridSample/显存排查/锁/TCP-UDP/Linux 基础 | 阿里各 BU 面经 |
| [40 腾讯 · 推理与 CUDA](./40-腾讯-推理与cuda.md) | FP4/FlashAttention/MoE/GQA/RoPE、fastllm 与 vLLM 源码、Triton PagedAttention、MPI/HPC、C++ 系统 | 腾讯 6 份 AI Infra 面经（~66 题） |
| [41 腾讯 TEG · 数据库与 OS](./41-腾讯-teg-数据库与os.md) | MySQL/分布式数据库 19 题（MVCC/B+树/执行计划）、Linux OS 6 题、MiniOb/xv6/OceanBase | 腾讯 TEG 一二三面 39 题 |
| [42 腾讯 · 编程题](./42-腾讯-编程题.md) | 双堆滑动窗口中位数/内存文件系统（字典树）/有序数组插入/快慢指针/K 组翻转/最大子数组/popcount | 腾讯 7 份面经编程题 |
| [50 百度 · 推理与算子](./50-百度-推理与算子.md) | Attention/QKV/FA 数学推导/Llama 推理算子/RMSNorm/量化粒度与 KV 量化/allocator 设计/Conv FLOPs/RT-DETR | 百度实习/校招 6 份面经 |
| [51 百度 · RL 与后训练](./51-百度-rl与后训练.md) | Trust Region→PPO/GAE 数学/GRPO/DAPO-GSPO 变体/Agentic RL/异步 RL 修正/DualPipe/ZeRO 显存推导/FSDP/SFT-DPO-LoRA 排障/RAG 搜索 | 百度实习/校招 |
| [52 百度 · 系统与 C++](./52-百度-系统与cpp.md) | OpenMP 竞态/Cache 映射/C++ 全家桶/Go GMP/TCP-UDP/sendfile/K8s/etcd vs Redis/Kafka/cgroup | 百度实习/校招 |
| [53 百度 · 算法题](./53-百度-算法题.md) | LCS/层序带层级/合并 K 链表/右视图/栈队列/滑窗/字符串相乘/验证栈序列 + CUDA LayerNorm/向量加 | 百度实习/校招 |
| [70 T2 · 算子与量化](./70-t2-算子与量化.md) | 商汤 GEMM 8×8 分块/Tensor Core/低比特、旷视 CUDA 算子与 LLaMA 部署、智谱校准算法手写、阶跃 DMA/RDMA/pin 内存 | 商汤/旷视/智谱/阶跃 |
| [71 T2 · 训练与算法](./71-t2-训练与算法.md) | MiniMax RAG/Agent/SFT/PPO-DPO-GRPO/MoE/ZeRO/LoRA、阶跃序列并行与容错、T2 五家编程题汇总 | MiniMax/阶跃等 T2 五家 |
| [80 T3 · 英伟达与三星](./80-t3-英伟达与三星.md) | SIMT/occupancy/bank conflict 粒度/ILP 深度题、时针分针毫秒精度/三线程不死锁、三星英文面/C++/网络/编译器 | 英伟达/三星 |
| [81 T3 · 国产 GPU 厂商](./81-t3-国产gpu厂商.md) | 寒武纪 GEMM 8×8/Little's Law、壁仞 K 维寄存器压力、摩尔线程离群量化、沐曦 Triton、遂原 Triton vs CUDA、燧原 MLIR/XLA/TVM | 寒武纪/壁仞/摩尔线程/沐曦/遂原/燧原 |
| [82 T3 · C++ 与系统](./82-t3-cpp与系统.md) | 太初内存管理/智能指针/虚函数、北极雄芯线程同步/DCL/malloc 底层、后摩梯度与量化（AdaQuant/I-ViT）、原粒 reduce 精度/pytest | 太初/北极雄芯/后摩/原粒 |
| [83 T3 · 飞腾与中兴](./83-t3-飞腾与中兴.md) | 飞腾 Decoder-only/RMSNorm/matmul+反量化/SpMV/Nsight；中兴三大并行/DeepSpeed/显存 16N 法则/200B 16 卡方案设计 | 飞腾/中兴 |
| [90 T4 · 编译器与 MLIR](./90-t4-编译器与mlir.md) | 元戎自定义 Dialect/torch-mlir/Bufferization、小马 MLIR 设计目标、蔚来 TVM 架构与图优化调度、动静态图 | 元戎启行/小马智行/蔚来/卓驭 |
| [91 T4 · 推理部署与 TensorRT](./91-t4-推理部署与tensorrt.md) | TensorRT/OpenVINO/Plugin、vLLM 显存优化与 PD 分离、CUDA Graph/PTX/编译流程、昇腾 NPU 与车载部署、BEV/跟踪/卡尔曼 | 蔚来/小鹏/理想/辉羲等车企 |
| [92 T4 · C++ 与系统](./92-t4-cpp与系统.md) | C++11-20 特性/面向对象/多继承/异步/map、Python 切片装饰器、虚拟内存/OS 调度/TCP-IP/K8s、项目软素质框架 | 车企/自动驾驶面经 |
| [93 T4 · 编程与算法](./93-t4-编程与算法.md) | 4D 张量 Concat CUDA kernel/计算图最小内存、uint8 第 K 大、旋转矩形交叠、NMS+IoU/reduce_sum/conv2d/岛屿/背包、GEMM/Softmax | 元戎/小鹏/蔚来/辉羲/理想/文远 |
| [100 T5 · 推理引擎](./100-t5-推理引擎.md) | vLLM scheduler 调度与抢占、投机采样 vLLM vs SGLang、MLA 矩阵吸收复杂度、EPLB、DeepSeek V3.2 DSA、NPU 推理瓶颈、分布式推理框架设计 | 小厂/Teleai/虾皮/飞星/传音/南湖 |
| [101 T5 · 量化与算子](./101-t5-量化与算子.md) | FP8 E4M3/E5M2 与计算/FP8 KV Cache、GPTQ/SmoothQuant、量化精度排查与校准、bank conflict、CUDA softmax/reduce、卷积 GEMM、NEON、达芬奇架构 | 讯飞/海康/传音/智源/好未来等 |
| [102 T5 · 系统与部署](./102-t5-系统与部署.md) | 推理框架通信协议、IO 多路复用、HTTP/2、Docker 原理、Linux 内存管理与零拷贝、锁与并发、C++/Python 基础、PyTorch 工具链、ncnn/RK3588 端侧 | 中科类脑/曙光/小光子/南湖/小厂 |
| [103 T5 · 算法题](./103-t5-算法题.md) | 合并有序数组、手写 MHA 与 Mask 位置、编辑距离、atoi、根到叶路径求和、内切圆、树形 DP、手写智能指针、TopK、CUDA 百万浮点数求和 | 海康/联想/虾皮/贝壳/好未来/数坤/南湖 |
| [104 T5 · 训练与 RL 杂题](./104-t5-其余公司杂题.md) | LoRA/P-Tuning、预训练后训练、GRPO/GSPO、RM/DPO、Megatron TP、DualPipe、FP8 训练、Rollout 优化与权重同步、TF-IDF/RAG、YOLOv5/ByteTrack、CLIP、软性问题 | 米哈游/贝壳/数坤/传音/识渊等 |

> 批次 6（对应编号段未完）完成后将继续并入。

## 配套代码工程

所有编程题的代码在仓库根 [solutions/](../../solutions) 下三个子工程，**不是单文件示例**，可构建、可测试、可复现实测数据：

```
solutions/
├── cpp/     C++17+CMake，11 个测试：LRU/内存池/智能指针/NMS/Conv2D + 语言基础
│            （对齐/constexpr/完美转发/迭代器删除/unique_ptr 数组），ctest 10/10
├── cuda/    CUDA+CMake(sm_86)，CPU 对拍 + --bench 实测，ctest 9/9：
│            归约/Softmax/GEMM(naive/smem/8×8 寄存器 tile)/转置/Norm/Layout/
│            直方图 scan/池化 IoU/行归约 + 向量加 grid-stride/SiLU/外积/
│            4D Concat(通用+末维)/uint8 第 K 大
└── python/  PyTorch+Triton+算法题，MHA/GQA/MLA/FA v1v2/PagedAttention/LoRA/SFT loss
             + 字节/阿里/腾讯/百度/T2-T5 各批次模块（AWQ/GRPO/校准/调度/量化 GEMM...），
             pytest 277/277（GPU 上运行）
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
- 公司批次与 01-12 篇的同族题（FA/KV Cache/ZeRO/量化等）交叉引用原篇，公司特有追问在批次篇内展开；各批次勘误节已汇总：**暂无 01-12 篇勘误**；
- 后续批次完成后按编号段继续并入本目录；
- 发现错误欢迎指正：代码以测试为准（工程里每个 kernel/算法都有对拍），文档以代码为准。
