---
title: "GEMM 与算子优化"
description: "Roofline、GEMM 优化方法论、算子融合、kernel 级优化与调试的面试题解答"
pubDate: 2026-10-05
topic: "算子优化"
order: 3
tags: ["算子优化", "高性能计算", "CUDA"]
---

覆盖题库：`AI-Infra-一面`（1-4、6）、`AI-Infra-综合面经题库-4`（2-9）、`AI-Infra-综合面经题库-3`（Q4 部分见分布式篇）、`AI-Infra-面经-1`（高性能计算 6/11/12）。

## Q1. Roofline 模型与"是否达到计算瓶颈"（AI-Infra-一面 1）

**原理**：把 kernel 性能画在"算术强度 AI（FLOP/Byte）→可达性能"的二维图上：

$$P = \min\left(P_{peak},\ AI \times BW\right)$$

- AI 小于机器平衡点 `P_peak/BW` 时在**访存屋顶**下（memory bound），性能 = AI × BW；
- AI 大于平衡点后在**计算屋顶**下（compute bound），性能 = P_peak。

3090：FP32 35.6 TFLOPS / HBM 936 GB/s → 平衡点 ≈ 38 FLOP/Byte；FP16 Tensor Core 71 TFLOPS → 平衡点 ≈ 76。

**判断是否到计算瓶颈**：① 算出 kernel 的理论 AI；② 实测性能与两个屋顶比较。若实测点贴近计算屋顶 → 已是 compute bound，优化要减少 FLOPs（算法层）或用更低精度/Tensor Core；若贴近访存屋顶 → 优化访存（融合、tile、向量化）。工具上 Nsight Compute 的 SOL 段直接给出 SM% 与 Memory%，取其高者即瓶颈类型。

## Q2. GPU 卡数成倍扩展时，瓶颈在哪（AI-Infra-一面 2）

扩展时**不能线性加速的环节**依次暴露：

1. **通信**：DP 的 AllReduce 通信量 ∝ 参数量（与卡数无关），卡越多单步通信占比越高；TP 的 AllReduce 限制在 NVLink 域内，跨机 TP 基本不可行。
2. **负载均衡**：数据不均（MoE 路由、长序列 batch）使部分卡空转，MFU 随规模下降。
3. **straggler 与同步**：同步 SGD 的最慢卡决定步长时间；尾延迟（慢盘机、网络抖动）被放大。
4. **全局资源**：参数服务器/存储带宽、数据预处理 pipeline、CPU dataloader、PCIe/NVLink 拓扑不均衡。
5. **小 batch 摊薄**：固定 token 量下卡数翻倍 → 每卡的 per-step 工作减半，kernel 并行度下降、launch 开销占比上升。

优化：通信-计算重叠（梯度分桶 + bucket AllReduce）、ZeRO/PP 切分通信、EP+负载均衡 loss、分层通信（机内 NVLink + 机间 IB）、容错与弹性调度。

## Q3. GEMM 一定是计算瓶颈型吗？优化思路（AI-Infra-一面 3）

**不一定**。是否 compute bound 取决于 AI 与并行度：

- 大 M/N/K、数据复用充分（tile 命中）→ compute bound（典型训练 prefill）。
- 瘦矩阵（decode：M=batch 很小）→ tile 效率低，A/B 装载占主导 → memory bound。
- 小尺寸时 launch/调度开销主导，两个屋顶都贴不上。

**优化思路**（由粗到细）：
1. 算法层：Strassen（工程少用）、Winograd（卷积）、低精度（TF32/BF16/FP8/INT8 + Tensor Core）。
2. 并行结构：二维/三维 tile 划分，block 计算一个 C tile；K 维切分 + split-k（解决 M、N 小并行度不足）。
3. 访存：shared memory staging（A/B tile 复用 16-32 次）、寄存器 tile（每线程 4-8 累加器）、double buffer（装载/计算重叠）、float4 向量化。
4. 计算指令：`mma`（warp 级矩阵片段，Tensor Core）、`#pragma unroll` 展开 K 内循环、FMA。
5. 布局：A 行主 B 列主让装载合并；预打包（padding 到 16 对齐）。
6. 自动调优：cute/cutlass 模板扫 tile 尺寸。

实测对照：`solutions/cuda` naive 2166 GFLOPS → 16×16 smem tile 2804 GFLOPS（1024³，3090 FP32 峰值 35.6 TFLOPS 的 8%，教学版与 cutlass 的差距正是上面 3-5 步的价值）。

## Q4. 性能调优如何定位瓶颈、用什么工具（AI-Infra-一面 4、综合-4 经验）

标准流程（**先看整体再看局部**）：

1. **系统级 Nsight Systems**（时间线）：kernel 占比、gap（空泡：同步/launch/数据传输）、多流重叠情况。回答"该优化哪个 kernel"。
2. **kernel 级 Nsight Compute**：SOL（SM%/Mem%）、占用率、roofline 落点、warp stall 原因（长 scoreboard=访存等、short scoreboard=smem 等、分支、 barrier）、指令吞吐、L1/L2 命中率。回答"这个 kernel 慢在哪"。
3. **source 级**：`-lineinfo` + SASS 视图看寄存器溢出（local memory 访问）、bank conflict 计数。
4. 简易法：事件计时（cudaEvent）夹住可疑段、逐步注释法。
5. Python 层：`torch.profiler`（= Nsight 的封装）、`ncu -k regex` 只测目标 kernel。

## Q5. kernel 级优化经验 / CuTe DSL / 手写融合（综合-4 Q2）

回答框架（把项目经验套进去）：

- **选型**：简单 elementwise/reduction 用 Triton（迭代快、autotune）；性能极限场景（Tensor Core 调度、TMA、warp specialization）用 CUTLASS/CuTe（layout 代数把 tile 划分、smem staging、mma 分配写成类型安全的组合子）或直接 CUDA + PTX。
- **融合动机**：省 HBM 往返（每融合一个中间张量省 2 次全量读写）+ 省 launch；典型如 RMSNorm+QKV 投影、matmul+反量化（W4A16 解包内嵌在装载里）、FA 整个 attention 一个 kernel。
- **方法**：先 fusion 空间分析（生产者-消费者图，相邻且数据可驻留 smem/寄存器的算子对），再写 kernel；用 ncu 对比融合前后 SOL。

## Q6. kernel fusion 倾向哪种方式实现（综合-4 Q3）

两条路线：

1. **模板/元编程生成**（Triton、TVM、CuTe）：调度原语组合，可移植性好、autotune 扫参数——适合量大面广的算子库。
2. **手写 CUDA**：控制力最强，能做 warp specialization、TMA pipeline、cp.async 多级流水——适合 FA/GEMM 这种被反复调用、值得榨极限的热点。

工程判断：**融合的收益主要来自访存节省**，先估算（中间张量字节×2 / 总耗时），值得做再用框架生成验证语义，最后考虑手写压榨。本仓库给出 Triton 融合样例：`solutions/python/src/answers/triton_kernels/`（softmax/layernorm/rmsnorm 融合版）。

## Q7. 融合后反而变慢的原因（综合-4 Q4）

1. **寄存器压力**：融合后单 kernel 活跃变量多 → 占用率下降甚至寄存器溢出到 local memory（最常见）。
2. **并行度不匹配**：两个可大并行度的算子被串在一个 kernel 里，后者必须等前者 block 内完成，无法各自打满 SM（grid 变小）。
3. **损失向量化/特化**：单独 kernel 可以按各自的最优 tile/vectorize 编译，融合后互相牵制。
4. **smem 超限**：两段 smem 需求叠加 → 驻留 block 减少。
5. **warp divergence 增多**、指令缓存压力。
6. **本来就不是访存瓶颈**：融合只省访存，如果 kernel 是 compute bound，收益趋零而开销增加。
排查：ncu 对比 occupancy、寄存器数、L1TEX/local 指标。

## Q8. Hopper Warp Specialization 原理（综合-4 Q5）

把一个 block 内的 warp 分成**角色化的生产者/消费者**：producer warp 专门做 TMA（`cp.async.bulk.tensor`）全局→shared 的异步装载，consumer warp 专门做 mma 计算，两组 warp 通过 **mbarrier**（arrival/wait barrier）同步——producer 到达"数据就绪"屏障，consumer 等待后开始计算，同一时刻 smem 里的 tile 双缓冲轮转。

相比 Ampere 的"所有 warp 一体化（装载+计算+同步都做）"：
- 装载与计算天然重叠（不用 `__syncthreads()` 全体 barrier）；
- 异步 TMA 引擎释放 ALU，搬运大 tile 不占线程指令；
- 寄存器按角色分配（producer 无需大量累加器）。

FA3、CUTLASS Hopper GEMM 都基于此。去掉 WS 的性能损失维度（字节-实习同款追问）：异步重叠消失（装载阻塞计算）、屏障粒度变粗、寄存器分配变差 → 典型吞吐掉 20-40%。

## Q9. 千卡 NCCL Timeout 定位（综合-4 Q6）

思路：**先定位"谁没到集合通信点"**。

1. **看日志时间线**：NCCL debug 日志（`NCCL_DEBUG=INFO`）中每个 rank 的 collective 序列，找出最后一条；卡住的 rank = 没执行到该 collective。
2. **常见根因**：
   - 某卡**进程挂了/OOM**（torch OOM 先死，其他 rank 在 AllReduce 等待）→ 对应 rank 日志最后有 traceback；
   - **慢卡**（ECC 错误降频、MIG 误配、被其他任务抢占）→ nvidia-smi 看利用率/时钟；
   - **网络**：IB 链路 flap、PFC 风暴、nccl 拓扑 fallback 到 TCP；
   - **逻辑 bug**：条件分支导致部分 rank 多调/少调一次通信（rank 数不等式）；
   - **挂起的 host 线程**：dataloader 卡死、CPU 侧死锁。
3. **工具**：py-spy dump 各进程栈（最快的手段，直接看到卡在哪个 python 栈）、gdb attach C++ 栈、`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`、nv-smi + ibstat。
4. **止损**：`NCCL_ASYNC_ERROR_HANDLING=1`/`TORCH_NCCL_ASYNC_ERROR_HANDLING`（新版本）让集合通信超时抛异常而不是永久挂起，配合 elastic restart。

## Q10. RL 场景 MoE 优化（综合-4 Q7）

RL（rollout + 训练混合负载）下 MoE 的痛点与手段：

- **路由动态性**：rollout 时 batch 小且 token 分布随策略漂移 → expert 负载不均更剧烈。手段：EPLB（按统计冗余复制热 expert）、动态容量因子、lost-token 控制。
- **训练通信**：EP 的 dispatch/combine 是两跳 all-to-all，与反向重叠加（DeepEP 的低精度 RDMA 内核、hook 机制把通信藏进计算）；Two-Batch Overlap 让 micro-batch 的通信与另一批的计算重叠。
- **rollout 吞吐**：vLLM/SGLang 的 EP 推理 + expert 并行预取；小 batch 用 TP×EP 混合避免单 expert 权重加载瓶颈。
- **奖励偏差**（原题"路由奖励偏差"）：RL 的 reward 依赖完整序列 forward，路由 dropout/容量截断会造成 train/infer 不一致 → 用 aux loss 校正 + 评测时无容量限制。

## Q11. 上线前的性能优化工作（综合-4 Q8）

推理上线 checklist（按收益排序）：

1. 图优化：算子融合（LN+GEMM、Silu+mul）、constant folding、消除 view/transpose。
2. Kernel：GEMM 走 cutlass/cublasLt、attention 走 FA、量化 kernel（W4A16/W8A8）。
3. 执行引擎：CUDA Graph 消除 launch、多 batch 分档；pinned memory + 异步 H2D。
4. 系统层：continuous batching、chunked prefill、prefix cache 复用、PD 分离。
5. 验证：精度回归（ppl/下游任务）+ 延迟/吞吐 SLO 压测（benchmark_serving）+ 长稳（显存碎片、 fragmentation、多轮会话泄漏）。

## Q12. 用 Agent 自动生成 CUDA kernel（综合-4 Q9）

可回答的工程框架（也是当前热点）：

- **候选生成**：LLM（含 reasoning 模型）从 spec（算子语义 + 输入 shape/dtype + 目标硬件）生成 Triton 优先（可编译验证快）、CUDA 兜底；few-shot 提供 cutlass/cute 模板。
- **验证闭环**：编译 → 与 PyTorch 参考对拍（随机 shape + 边界 shape + 数值容差）→ 正确才进入下一步。
- **性能闭环**：autotune 扫参数（Triton `@autotune` 或生成器自己枚举 tile 配置）→ ncu 提取 SOL/bandwidth → 把 profile 文本反馈给 LLM 迭代（类似 AlphaCode 的采样-过滤）。
- **护栏**：数值容差分级、禁止 UB（compute-sanitizer）、超时预算、与人类基线对比决定采纳。
- 现实结论：简单 elementwise/reduction 类 Agent 已能超越新手；复杂融合（FA 级）仍需人类模板与领域知识。

## Q13. conv+BN 融合公式推导（面经-1 HPC 11）

推理时 BN 是固定仿射，可整段折进卷积权重：

$$y = \frac{W*x + b - \mu}{\sqrt{\sigma^2+\epsilon}} \cdot \gamma + \beta = W' * x + b'$$

$$W' = W \cdot \frac{\gamma}{\sqrt{\sigma^2+\epsilon}},\qquad b' = (b - \mu)\cdot\frac{\gamma}{\sqrt{\sigma^2+\epsilon}} + \beta$$

可融合的原因：卷积与仿射都是**线性算子**，线性复合仍是线性 → 合并成一次卷积。训练时 μ/σ 是 batch 统计量且反向需要分开的梯度，故不融合。同族：conv+BN+ReLU（ReLU 非线性只能"跟随"在融合卷积后，无需展开）。

## Q14. 推理框架中卷积的实现方式（面经-1 HPC 12）

1. **im2col + GEMM**：把输入按感受野展开成矩阵，与权重矩阵相乘——复用极致优化的 GEMM，是 cudnn/主流框架默认路径之一；代价是 im2col 的额外显存与访存。
2. **直接卷积（direct）**：在 kernel 里就地滑窗，省展开开销，需精心 tile；小通道/特定 shape 有优势。
3. **Winograd**：3×3 s=1 时 F(2×2,3×3) 把乘法从 36 降到 16（×2.25 理论加速），代价是加法与变换 kernel。
4. **FFT**：大 kernel 有效，频域乘法；内存与精度开销大，深度学习少用。
5. **Tensor Core 路径**：implicit GEMM（把 im2col 的地址计算内嵌在装载里，不物化展开矩阵）+ NCHW→NHWC（c 连续）。
