---
title: "字节跳动 · CUDA 与算子优化"
description: "字节跳动 AI Infra 面经算子方向题解：CUDA 优化策略 / 卷积优化 / 共享内存 / kernel fusion / warp specialization / 手写 GEMM 与 MLA / Sparse Attention kernel"
pubDate: 2026-10-05
topic: "公司题解-字节跳动"
order: 21
tags: ["CUDA", "算子优化", "公司题解"]
draft: false
---

覆盖题库：`AI-Infra-一面-2`（算子优化 1-3、编程 13）、`AI-Infra-实习-一面-1`（算子优化 1-5）、`AI-Infra-校招-一面`（2/3）、`AI-Infra-实习-一二三面`（二面 11、14）。

**新增配套代码**：`solutions/python/src/answers/bytedance/sparse_attention.py`（动态块稀疏 attention）：

```bash
cd solutions/python && pip install -e . && python -m pytest tests/test_bytedance.py -q   # 27 passed
```

## Q1. CUDA 编程的常见优化策略（AI-Infra-一面-2 算子优化 1）

同《01 CUDA 基础与体系结构》Q8 的优化维度总表。字节版答题骨架（配合本仓库 3090 实测数据最有说服力）：

| 层次 | 策略 | 仓库实测示例（RTX 3090） |
|---|---|---|
| 访存 | 合并访问、smem tiling、向量化（float4）、padding 消 bank conflict | GEMM 16×16 tiling 2804 GFLOPS（《02 CUDA 编程题》Q3）；转置 smem+padding（《02》Q4） |
| 并行度 | grid/block 调满 SM、occupancy、warp 级原语（shuffle） | 归约树 + warp shuffle 0.016ms，124× 于原子版（《02》Q1） |
| 私有化 | smem/寄存器私有化减少全局原子 | 直方图 smem 私有化 19.9×（《02》Q5） |
| 算法 | 分块、scan 换 scan-free、register tiling、循环展开 | Online Softmax（《02》Q2） |
| 系统级 | fusion、双缓冲、异步（cp.async/TMA）、CUDA Graph、warp specialization | fusion 变慢的边界见 Q5 |

口诀：**先 profile 再优化**（ncu 看 SOL SM/SMEM/带宽占比定 compute/memory bound，见《08 编译器与性能分析》Q5）。

## Q2. 卷积操作有哪些优化方法（AI-Infra-一面-2 算子优化 2）

推理框架卷积落地见《03 GEMM 与算子优化》Q14。本题按技术谱系展开：

1. **im2col + GEMM**（最通用）：把每个输出位置的感受野展开成行，卷积变矩阵乘，复用高度优化的 GEMM（cuBLAS/cutlass）。代价是显存膨胀 $(C\cdot k\cdot k)$ 倍与变换开销。
2. **implicit GEMM**（cuDNN/cutlass 路线）：不物化 im2col，在 GEMM kernel 内部按索引规则即时取输入（借助 smem tiling），省显存省搬运。
3. **Winograd**：数学变换把乘加次数降低（F(2×2,3×3) 理论 2.25×），适合小卷积核、stride=1；变换本身有开销且数值误差大，FP16/INT8 下要小心。
4. **FFT**：大核（11×11+）时把卷积变频域乘法；小核不划算。
5. **数据布局与结构**：NHWC + channels-last 让通道维连续（向量化读、Tensor Core 友好，《02》Q6 有 NCHW→NHWC kernel）；depthwise 用专门 kernel（ MobileNet 族）。
6. **系统层**：conv+BN+ReLU 融合（《03》Q13 公式推导）、INT8/FP8 Tensor Core、半精度累加。
7. ** stride>1 / dilated / 分组卷积**：分组卷积按组切 kernel 提升并行；空洞卷积等效大核时可与 im2col 统一处理。

## Q3. GPU 共享内存的工作原理及使用方式（AI-Infra-一面-2 算子优化 3）

同《01》Q2（内存层级）/Q7（smem 注意事项）。要点收拢：

- **物理**：SM 内的片上 SRAM（3090 每 SM 128KB，与 L1 共享物理容量可配比），延迟 ~20-30 cycle，比全局内存低一个量级；按 block 私有，生命周期与 block 相同。
- **组织**：按 32 个 bank × 4B 划分；warp 内线程按地址交错映射 bank，同一 warp 访问不同 bank 并行无冲突；同地址要 broadcast；冲突时硬件串行化（bank conflict），解法是 padding（+1 列）或 swizzle——《02》Q4 转置 kernel 实测。
- **使用方式**：`__shared__` 声明；典型模式是 *staged tiling*——全局内存协同装入 tile → `__syncthreads()` → tile 内复用计算（GEMM 每元素复用 BLOCK 次）；生命周期结束前复用（ping-pong 双缓冲）；动态大 smem 需 `cudaFuncSetAttribute`。
- **为什么快**：把"多次不规则的全局访问"换成"一次合并的全局访问 + 多次片上访问"，本质是用程序员知道的复用结构换带宽。

## Q4. kernel fusion 通常倾向哪种实现方式（实习-一面-1 算子优化 2）

同《03 GEMM 与算子优化》Q6：优先**代码生成/模板库路线**（Triton/CUTLASS CuTe、torch.compile），产出的 kernel 可维护、可移植、自动调优；手写 CUDA 留给热点与极致场景。取舍维度：elementwise/规约型 fusion 平台编译器已经很成熟，GEMM 型 epilogue fusion（乘加、量化反量化、mask+softmax）是 CuTe/epilogue 机制的强项。

## Q5. fusion 后性能反而下降的原因（实习-一面-1 算子优化 3）

同《03 GEMM 与算子优化》Q7。核心口径：**fusion 省的是 kernel 间中间结果的 HBM 往返与 launch 开销，付的代价是并行度/寄存器/occupancy**。变慢的典型情形：

1. 融合的算子**并行结构冲突**（reduce + broadcast：规约维度变化导致部分线程空转）；
2. 融合后寄存器/smem 压力上升 → occupancy 掉 → 延迟隐藏失败；
3. 原来 A、B 可在不同 stream 重叠或各自打满不同瓶颈，融合后变成单一瓶颈（如把 memory-bound 算子塞进 compute-bound GEMM 的 prologue，拖慢整体）；
4. 编译器生成的融合 kernel 不及手工分写的极优 kernel（GEMM 这种深优化算子最常见）；
5. launch 开销本来就不显著（kernel 都很大）时，fusion 的收益上限低，任何副作用都净亏。

## Q6. Hopper 架构中 warp specialization 的机制与底层实现（实习-一面-1 算子优化 4）

同《03 GEMM 与算子优化》Q8。机制三层：**异步拷贝引擎**（TMA 提供 bulk 全局→smem 的异步传输，独立于 SM 流水线）、**warp 角色分工**（producer warp 只负责 TMA 装载，consumer warp 专注 mma 计算，mbarrier 做阶段完成同步）、**寄存器/smem 双缓冲流水**（装载与计算重叠，TMA 传输期间 SM 不空转）。Hopper 前 cp.async 是较弱的版本（按 16B 粒度、无 bulk 维度搬运）。

## Q7. 去掉 warp specialization，只保留 tiling + shared memory 的性能损失体现在哪些方面（实习-一面-1 算子优化 5）

这是 Q6 的逆向追问，按损失来源拆解：

1. **装载与计算串行化**：没有专职 producer + 双缓冲流水，smem tile 装载期间 consumer warp 无数据可算，SM 出现"装载气泡"；GEMM 的 smem 带宽与 Tensor Core 峰值差距越大气泡占比越大。
2. **同步开销显性化**：warp specialization 用 mbarrier 只在"阶段边界"同步 producer/consumer；普通版本每块装载完要 `__syncthreads()` 全 block 屏障，warp 间负载不均时互相等待。
3. **寄存器压力转移到计算 warp**：统一 warp 既要发装载又要算，地址计算/循环控制占用的寄存器挤占累加器空间，可能被迫减小 tile → 复用率下降。
4. **TMA 的收益拿不满**：TMA 的 bulk 传输设计上就是给 producer warp 全权驱动的（单线程发 descriptor、硬件搬运多维 tile）；混在计算 warp 里发 TMA 指令，descriptor 构造与发射穿插在 mma 流水中。
5. **量化口径**：典型 Hopper GEMM（cutlass 3.x）从 warp-specialized 退回朴素 tiling 版本，实测损失约 20-40%；但**小 kernel 或非计算密集场景损失小甚至为零**（本来就不是靠流水打满的）。

## Q8. kernel 级别优化经验：CUTE DSL / 手写 CUDA 算子融合（实习-一面-1 算子优化 1）

同《03》Q5（CuTe 分层抽象：Layout 代数解耦逻辑布局与硬件线程映射；手写 CUDA 融合的组织方式）。答字节这题的素材链：`solutions/cuda/` 14 类 kernel（归约/直方图/GEMM/转置等，含 --bench 实测）+ Triton 融合 softmax（`solutions/python/src/answers/triton_kernels/`）。

## Q9. 用 CUDA 编写 GEMM 并说明优化方法与性能评估指标（AI-Infra-校招-一面 2）

完整实现同《02 CUDA 编程题》Q3（naive → smem tiling 两版，对拍 + GFLOPS 实测），瓶颈分析同《03》Q1-Q3。字节要求的"评估指标"口径：

- **吞吐**：GFLOPS（$2MNK/t$）、有效带宽 GB/s（$bytes/t$，bytes = 读写矩阵总量）；
- **利用率**：与峰值的比值（3090 FP32 35.6 TFLOPS、HBM 936 GB/s），配合 Roofline 判断离瓶颈还有多远（《03》Q1）；
- **正确性对拍**：与 CPU/BLAS 比 allclose，误差随 K 累积要说明（FP32 累加顺序）；
- **稳定性**：多尺度（256/1024/4096）性能曲线，只报单点数字不可信；
- 优化叙事顺序：合并访存 → smem tiling（复用）→ register tiling（线程算分块）→ 双缓冲/CUTLASS 风格流水 → Tensor Core（mma/WGMMA）。

仓库实测（RTX 3090，FP32）：naive 2166 GFLOPS → 16×16 smem tiling 2804 GFLOPS（`./build/test_gemm --bench`）。

## Q10. 手写实现 MLA（AI-Infra-校招-一面 3）

同《04 Attention 与 KV Cache》Q3：核心是** latent 压缩 + 权重吸收**——$c^K_t = W^{DKV} h_t$（KV 共享 latent，$d_c \ll 2 n_h d_h$），attention 可在 latent 空间直接算（$W^{UK}$ 吸收进 $W^UQ$、$W^{UV}$ 吸收进输出投影）；RoPE 部分解耦单独缓存（decoupled RoPE，见《04》Q8）。实现：`solutions/python/src/answers/attention/mla.py` 的 `forward_naive`（展开路径）与 `forward_absorb`（吸收路径），两路径数值等价有测试（`test_mla_absorb_equivalence`）。

## Q11. Sparse Attention 的 Python kernel 如何实现（AI-Infra-实习-一二三面 二面 11）

**先讲路线再写代码**（本题是"研究项目+工程能力"双重考察）：

- **静态模式**：local/strided/global/random mask 固定，离线构建 mask，kernel 按 mask 稀疏遍历（Sparse Transformer、BigBird、Longformer）；
- **动态选择**：对每个 KV 块打代表分（Quest 用页内 K 逐维 min/max 与 q 乘出上界；简化版用块内均值池化内积），每个 query 块保留 top-k 块 + 局部窗兜底。

**实现**（`solutions/python/src/answers/bytedance/sparse_attention.py`）：

1. `block_scores`：块代表分 = 块内真实 token 的 pairwise 均值 = 均值池化向量的内积（用按块求和向量做 masked mean，padding token 零贡献，天然支持非整块尾巴）；
2. `select_blocks`：`topk(scores) ∪ 局部窗 [i-w, i+w]` 的布尔块 mask（滑窗兜底防近邻丢失）；
3. `block_sparse_attention`：token 级 mask 由块 mask `repeat_interleave` 展开，dense attention 套 mask——**与 dense+同 mask 数值严格相等**（测试对拍），稀疏化的收益在"省掉的那部分分数计算与 KV 读取"，不在数值近似。

复杂度：每行只算 $k_{top}+w$ 个块，`sparse_flops` 给出对比（S=4096、block=16、top-8、窗 1 时 QK^T 计算量降到 ~4.3%）。工业级注意点：块选择本身要先算块分数（一次 O(S·d) 的池化内积，远小于 attention）；decode 场景省的是 KV 读取，与 PagedAttention 结合时块选择直接在页粒度做（Quest 的做法）。

## Q12. 如何编写 GPU 并行计算相关代码（AI-Infra-实习-一二三面 二面 14）

方法论题，给可复用的流程（本仓库 solutions/ 工程即范例）：

1. **问题抽象**：找数据并行维度（每个输出元素独立吗？有归约依赖吗？），确定每线程做什么（map / reduce / scan / stencil）；
2. **资源画像**：输入输出字节数与 FLOPs，算 arithmetic intensity，预判 compute/memory bound，据此定优化力度；
3. **骨架实现**：naive 版本先行（正确的参照系），grid/block 形状按"输出形状 + 每线程工作量"定，边界用 guard 处理；
4. **对拍验证**：CPU 参考实现 + 多组随机规模 allclose（数值题注明误差累积），这一步不能省——GPU bug 多为静默错误；
5. **迭代优化**：合并访存 → smem tiling/私有化 → warp 原语 → occupancy 调整 → 异步/双缓冲 → fusion；每步一个改动，用 ncu（SOL 指标）验证瓶颈转移；
6. **基准输出**：固定时钟、预热、多次取中位数、报告 GFLOPS/GB/s 与峰值比，附规模扫描。

## Q13. 编程题：求图的最短连通路径长度（AI-Infra-一面-2 编程 13）

同《12 算法题》"树与图"节的 MST：n 个节点间的最短连通方案 = 最小生成树总边权，Kruskal（排序边 + 并查集）O(E log E) 与 Prim（小根堆）O(E log V) 双实现，代码 `solutions/python/src/answers/algorithms/graphs.py`。若题目是"单源最短路"则 Dijkstra（同文件惰性删除堆版）；面试先与面试官确认是无权 BFS（连通性）/加权单源/全图连通哪种。
