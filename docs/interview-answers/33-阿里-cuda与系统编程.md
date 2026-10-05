---
title: "阿里 · CUDA 与系统编程"
description: "阿里巴巴 AI Infra 面试题解·CUDA/编译/系统/编程篇：Warp Specialization、MLIR vs TVM、算子优化工程方法论、互联拓扑建模、GPU 生态、显存长稳排查、CUDA 向量加、LRU、MHA、BST 迭代器等手写题"
pubDate: 2026-10-05
topic: "公司题解-阿里巴巴"
order: 33
tags: ["面试解答", "阿里巴巴", "CUDA", "编译器", "编程题"]
draft: false
---

覆盖题库（来源缩写）：`阿1/阿2`=阿里巴巴-AI-Infra-1/2、`一1/一2`=AI-Infra-一面-1/2、`实1`=AI-Infra-实习-1、`实一二`=AI-Infra-实习-一二面、`校`=AI-Infra-校招-一面、`云实`=云-AI-Infra-实习、`云二1`=云-AI-Infra-实习-二面-1、`控实`=控股集团-AI-Infra-实习-一面、`淘2`=淘天-AI-Infra-一面-2。

CUDA/融合/编译器同族题引用 [03 GEMM 与算子优化](./03-gemm与算子优化.md)、[08 编译器与性能分析](./08-编译器与性能分析.md)、[01 CUDA 基础](./01-cuda-基础与体系结构.md)、[09 C++ 与系统基础](./09-cpp与系统基础.md)；编程题引用 `solutions/` 三个工程，新增代码在 `solutions/python/src/answers/alibaba/`。

## Q1. 去掉 Warp Specialization，只保留 tile 分块与 Shared Memory 优化，性能下降反映在哪些维度？（阿1 Q4）

与 [03](./03-gemm与算子优化.md) Q8 同题，按"可测量的维度"组织答案（阿里考官期待 ncu 指标级回答）：

1. **吞吐（SM Throughput / SOL SM%）**：一体化模式下装载与计算串行（`__syncthreads()` 全体屏障隔开），计算 warp 在等数据 → SM 利用率掉；
2. **访存（DRAM/L2 带宽利用率）**：没有 TMA 异步引擎与多级流水，装载不能与计算重叠，HBM 出现空闲窗口，有效带宽下降；
3. **Warp Stall 结构**：stalled on barrier / short scoreboard（smem 等待）占比显著上升（ncu 的 Warp State Statistics 段）；
4. **占用与寄存器**：一体式 warp 既要装载寄存器又要累加器 → 单 warp 寄存器多、驻留 block 少；WS 可用 `setmaxnreg` 按角色动态分配（producer 少寄存器、consumer 多）；
5. **指令开销**：搬运由线程指令（LDG/STS）完成，占掉 ALU 发射槽，TMA 则是引擎代劳。
   典型合计：FA3/CUTLASS Hopper GEMM 去掉 WS 掉 **20-40%**（见 [03](./03-gemm与算子优化.md) Q8）。

## Q2. Hopper 架构下 Warp Specialization 的机制原理与底层实现（阿1 Q8）

与 [03](./03-gemm与算子优化.md) Q8 同题，底层四件套（能说出指令/原语名是加分项）：

1. **角色分工**：block 内 warp 分 producer（装载）/consumer（计算），编译期/启动期固定角色——不再全体 warp 做"装载-屏障-计算-屏障"循环；
2. **TMA（`cp.async.bulk.tensor`）**：一次引擎级异步搬运整个多维 tile 到 smem，硬件按 **tensor map**（tiled 布局描述符，含 shape/stride/box）直接展开地址——线程只发一条指令；支持 **cluster 跨 smem 直写与 multicast**（同一数据广播到多卡内多 SM 的 smem，TP 场景省带宽）；
3. **mbarrier**：异步屏障对象，两相位轮转（phase parity）；producer `cp.async.bulk` 时 `mbarrier.arrive.expect_tx` 声明字节数，数据落 smem 后硬件自动"到达"；consumer `mbarrier.try_wait` 等相位翻转——比 `__syncthreads` 粒度细且非阻塞；
4. **多级流水**：smem 划分 K 个 stage（如 4），producer 领先消费 N-1 stage，装载延迟被计算覆盖；配 `setmaxnreg` 做寄存器再分配。
   与 Ampere 对比：Ampere 的 `cp.async`（L2STSA 路径）仍需每线程搬一段 + 全体 barrier；Hopper 把"地址展开 + 搬运 + 到达同步"全下放硬件。

## Q3. kernel 层面的优化经验：CuTe DSL 或手写 CUDA 实现 fusion 算子（阿1 Q5）

与 [03](./03-gemm与算子优化.md) Q5 同题。框架：**选型**（elementwise/reduction → Triton 迭代快；热点极限 → CUTLASS/CuTe 或裸 CUDA）；**CuTe 的价值**——layout 代数（`Layout`/`Tile`/`Hierarchy` 组合子）把 tile 划分、smem staging、mma 片段分配写成类型安全的声明，编译期展开，改 tile 形状不动逻辑；**融合动机**——每融合一个中间张量省 2 次全量 HBM 读写 + launch 开销（RMSNorm+QKV、matmul+反量化见 [30](./30-阿里-推理与量化.md) Q14）；**方法**——先生产者-消费者图分析 fusion 空间（相邻且中间张量可驻留 smem/寄存器），ncu 对比融合前后 SOL。

## Q4. kernel fusion 倾向哪种技术方案？理由？（阿1 Q6）

与 [03](./03-gemm与算子优化.md) Q6 同题：模板/元编程生成（Triton/TVM/CuTe：可移植、autotune、适合量大面广）vs 手写 CUDA（控制力最强：WS/TMA/多级流水，适合 FA/GEMM 级热点）。工程判断：**先估收益**（中间张量字节×2/总耗时），值得做再用框架验证语义，最后手写压榨。本仓库 Triton 融合样例：`solutions/python/src/answers/triton_kernels/`。

## Q5. 有没有 kernel fusion 后性能反而变差的情况？原因？（阿1 Q10）

与 [03](./03-gemm与算子优化.md) Q7 同题，六大原因速记：寄存器压力升/溢出、并行度不匹配（grid 变小）、向量化与特化受损、smem 超限、warp divergence 增多、**本来就不是访存瓶颈**（compute bound 融合无收益）。排查：ncu 对比 occupancy/寄存器数/local memory 指标。

## Q6. MLIR 与 TVM 各自的设计理念及主要差异？（阿2 Q1）

与 [08](./08-编译器与性能分析.md) Q1/Q2 同族：

- **MLIR**：**编译器基础设施**（meta-compiler）——不绑定语言/目标，提供**多级 IR + 方言**（dialect：linalg/tosa/memref/arith/vector...）与**渐进下降**（pass 逐级降抽象），下游自选方言组合构建自己的编译器（IREE、torch-mlir、XLA 部分路线）。理念：**复用编译器工程**，适配深度学习/编译器研究的快速分化；
- **TVM**：**端到端深度学习部署编译器**——前端（Relay/Relax 图 IR）→ 调度优化（AutoTVM/Ansor/Meta-Schedule 自动搜 schedule）→ 后端代码生成（CUDA/ROCm/VLIW/NPU）。理念：**自动化性能优化 + 广 target 覆盖**，开箱即用。
- **差异总结**：定位（基础设施 vs 产品）、IR 组织（多级方言栈 vs 固定两级图/张量 IR）、优化来源（pass 框架自行组合 vs 内置自动调度器）、社区（LLVM 系编译器生态 vs 部署生态）。一句话：**用 MLIR 可以造一个 TVM，TVM 不必关心 MLIR**。

## Q7. TVM 中 Relay 调度原语与 TIR 调度原语各承担什么角色？（阿2 Q2）

- **Relay（图级 IR）**：表达"算什么"——函数/算子调用图、ADT、自动微分；它的"调度"其实是**图变换 pass**（算子融合 fuse_ops、布局改写、常量折叠、算子降级到 TIR 的调度任务）——决定哪些算子融合、走什么后端（BYOC）；
- **TIR（张量级 IR）**：表达"怎么算"——循环嵌套、buffer 访问；**schedule 原语**在这里：`split/reorder/fuse`（循环变换）、`bind`（绑定 threadIdx/blockIdx）、`cache_read/cache_write`（插入 smem/寄存器缓存）、`compute_at/compute_inline`（计算位置与内联）、`tensorize`（映射 tensor core 指令）、`blockize`（循环升块）；
- **角色关系**：Relay 的图优化决定"调用哪些 TIR kernel、融合边界在哪"；TIR 调度原语决定"每个 kernel 内部的并行与访存结构"。新一代 Relax 把两者拉近（统一张量 IR + 交叉算子调度），可以补这句显示跟进。

## Q8. 单算子性能优化的常见方法（GEMM/Conv 策略）？（阿2 Q3）

与 [03](./03-gemm与算子优化.md) Q3（GEMM：tile/smem staging/寄存器 tile/double buffer/向量化/mma/split-k）与 Q14（Conv：im2col+GEMM、direct、Winograd、FFT、implicit GEMM + NHWC）同题。回答顺序：算法层（低精度/Winograd）→ 并行结构（二维 tile、K 切分）→ 访存（smem/寄存器复用、double buffer、128bit 装载）→ 指令（mma/展开）→ 自动调优。

## Q9. 工程角度如何对一个算子做性能优化？优化思路、收益评估与极限分析（云二1 Q3）

方法论（比 Q8 多"收益评估与极限分析"两环）：

1. **基线与画像**：naive 实现 + ncu/nsys 定位（SOL、stall 原因）——先证明"值得优化"（时间占比 × 理论提升空间）；
2. **极限分析（上界）**：算出理论 Roofline——所需最小访存字节（输入+输出一次读写）与 FLOPs，得带宽/算力上界时间；实测/上界比值就是剩余优化空间；例：elementwise 4096² fp32 = 64MB，3090 936GB/s → 极限 0.137ms，做到 0.15ms 即 91% 达成，可停手；
3. **优化手段**按 Q8 逐层施加，每步 A/B 实测（锁频 `nvidia-smi -lgc`、warmup + 多轮取中位/均值、同 shape 矩阵），记录每项收益归因；
4. **收益评估**：单 kernel 加速比 + **端到端传导**（Q10）；记录负优化并回滚；
5. **验收**：正确性对拍（多 shape/边界 + compute-sanitizer）+ 精度容差回归。
   本仓库范例：`solutions/cuda` 的 naive vs smem 优化对照（README 实测表）与 `solutions/python/src/answers/triton_kernels/`（fused softmax 825 GB/s，达 3090 理论带宽 88%）。

## Q10. 算子优化的效果如何传导到整网层面？如何与多卡互联、Profiling、软硬件协同衔接？（云二1 Q7）

- **传导路径**：单 kernel 时间 → 层/图时间（kernel 占比加权）→ 引擎级（是否被 launch 开销/同步/拷贝掩盖——**小 kernel 加速 50% 但只占 3% 时间，端到端无感**，Amdahl 定律）；图调度层面：融合减少中间物化与 launch，改写执行序提高 overlap；
- **多卡互联**：kernel 变快 → 计算与通信的比例变化（TP 的 AllReduce 等待暴露，见 [30](./30-阿里-推理与量化.md) Q17）；反向——通信优化（overlap、低精度传输）等效给 kernel 让出带宽；EP 场景算子优化改变 dispatch/combine 的 token 分布与负载均衡（[31](./31-阿里-训练与算法.md) Q12）；
- **Profiling 链路**：ncu（kernel 级）→ nsys（时间线：gap/overlap/通信）→ 拓扑与通信工具（nccl-tests、ibstat）三层贯通，端到端归因；
- **软硬件协同**：cuBLAS/cuDNN 的 heuristic 选 kernel、架构特化编译（arch flag/PTX 版本）、驱动/固件（ECC、时钟）、以及把算子特性反馈到硬件选型（带宽 vs 算力配比，见 Q13）。

## Q11. 如何对互联拓扑进行建模？（云实 Q2）

- **层次图模型**：节点=设备（GPU/NIC/交换机/CPU），边按介质分类型（NVLink/NVSwitch/PCIe/IB/以太网），每条边标 **带宽（单向/双向聚合）与时延**；关键聚合量：**bisection bandwidth（对分带宽）**、任意两卡 hop 数、组内/组间带宽比（典型 8 卡 NVLink 域 vs 跨机 400G）；
- **通信代价模型**：集合通信建模到拓扑——ring allreduce 时间 ≈ $2V(N-1)/(\text{链路有效带宽})$ + 启动时延，跨机 ring 域分层（机内 NVLink + 机间 IB 两段环）；据此回答"TP 放哪 8 卡、EP 怎么排专家、DP 哪些卡成组"；
- **实测标定**：nccl-tests（allreduce/alltoall 带宽-消息量曲线）、`nvidia-smi topo -m`、ibstat/ibping；把实测矩阵填回模型；
- **用途**：并行策略自动选择（策略代价 = 计算 + 通信模型预估）、任务调度（拓扑感知放置）、故障域划分。
   结合工作讲：把自己项目里"策略选择/调度"如何用到模型（或没用到、踩了什么坑）讲实即可。

## Q12. 不同 GPU 生态之间存在哪些主要差异？（云二1 Q6）

- **NVIDIA CUDA**：事实标准——编程模型（.cu/PTX）、库（cuBLAS/cuDNN/CUTLASS）、工具（nvcc/ncu/nsight）、集合通信 NCCL，成熟度与文档最全；
- **AMD ROCm/HIP**：HIP 提供 CUDA 迁移层（hipify 半自动），MI 系列 FP8 支持；库对应物（hipBLAS/RCCL）齐全度与稳定性差距仍在，个别算子/版本组合坑多；
- **国产**：昇腾 CANN（AscendC/算子开发套件，mindspore 生态亲和）、海光 DCU（ROCm 系）、摩尔线程 MUSA、天数/壁仞等——共性差异点在**算子库覆盖、编译器成熟度、文档与社区、稀疏/低精度指令支持、框架适配层（torch_npu 等）**；
- **迁移成本维度**：源码（API 兼容层）、性能（等价 kernel 需重调优 tile/向量宽度）、数值（TF32/FP8 语义差异）、工具链（profiler 对应物）；
- **工程策略**：用中间表示屏蔽——Triton/PyTorch 2 compile 的多后端、ONNX 系、自研算子库抽象层；热点算子按后端各写一版（模板化生成，呼应 [03](./03-gemm与算子优化.md) Q6）。

## Q13. 未来硬件形态的发展趋势（云实 Q3 开放题）

给 4 条主线 + 各自依据（观点题答结构）：

1. **Scale-up 域扩大 + 互联成主角**：NVLink/NVSwitch 域从 8 卡到 NVL72 级（机架级 GPU 一台"计算机"），光互联/OCS 进入机内；UALink/UEC 开放生态追赶——原因是 TP/EP 与 MoE 的通信需求增长快于单卡算力；
2. **内存墙应对**：HBM 容量/带宽代际跳升（3D 堆叠、HBM4）、CXL 内存池化、存算近邻（PIM）试点——decode 是 memory bound（[30](./30-阿里-推理与量化.md) Q1），带宽即吞吐；
3. **低精度与专用化**：FP8/FP4 原生 Tensor Core、推理专用加速器（KV cache 管理硬件化）、DPU 卸载通信与存储——单位 token 成本驱动；
4. **功耗与交付形态**：液冷、整机柜交付（NVL72 式）、供电架构升级；集群级"解耦"（PD 分离、KV 为中心的架构，见 [05](./05-推理引擎与优化.md) Q4）反过来定义硬件（内存型节点 vs 算力型节点）。
   收尾观点：软件定义硬件协同（算子特性 → 硬件特性反馈闭环，Q10）会越来越强，AI Infra 工程师的生态位正在"编译器+互联+系统"的交叉处。

## Q14. 显存异常与长期稳定性问题如何定位排查？（云二1 Q2）

**先分类现象再选工具**：

- **OOM（突发）**：看分配点——`torch.cuda.memory_snapshot()`/`memory_summary()` 定位大块；区分"真用满"（activation/batch 过大，计算显存预算表核对）vs "碎片化"（reserved 高 allocated 低，`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解）；
- **缓慢增长（泄漏）**：重复同一请求跑 N 轮画显存曲线；常见根因：计算图被引用（loss 带 `requires_grad` 的中间量被全局持有）、缓存无上限（prefix cache/KV 池、CUDA Graph 捕获列表）、异常路径未释放；分段二分（注释模块）+ snapshot diff；
- **硬件/驱动**：`dmesg` 里 Xid 错误码（79 ECC 双位错、48/63 页错误等）、`nvidia-smi -q` 看 ECC/remapped pages——硬件退化要换卡而不是修代码；
- **长稳（soak）体系**：压测脚本（固定 QPS + 混合长短序列/多轮会话）跑 24-72h，监控显存/带宽/时延/错误率四条曲线；回归门禁（显存增量断言：同负载 N 轮后增量 < 阈值）；崩溃自动保留 core dump 与现场（torch save + 环境指纹）；节点健康巡检（ECC、温度、降频）+ 自动驱逐。
   案例化回答：挑一个"碎片化→expandable_segments 解决""泄漏→snapshot 定位到缓存"的真实故事讲。

## Q15. GPU 和 CPU 的体系结构差异，各自适合什么任务？（实一二·一面 Q4）

| 维度 | CPU | GPU |
|---|---|---|
| 核 | 少（十几~百）强核，乱序执行/分支预测/大缓存，主频 3GHz+ | 数千 ALU，SIMT，靠海量线程隐藏延迟（吞吐导向） |
| 内存 | DDR 带宽低、cache 大（几十 MB）层级深 | HBM 带宽 ~1TB/s、smem/L2 程序可控 |
| 延迟 vs 吞吐 | 低延迟（等待数据的同时切换少，靠预取/cache） | 高吞吐（切换 warp 隐藏访存延迟） |
| 并行模型 | 任务级/MIMD | 数据级（warp 锁步，分支代价 divergence） |

适合：CPU——逻辑复杂、分支密集、串行依赖、低延迟响应（OS/数据库/控制流）；GPU——数据并行、计算密度高、规则访存（矩阵乘/卷积/物理仿真/渲染）。Amdahl 定律给加速上界；实际系统异构协作（CPU 调度预处理 + GPU 计算）。GPU 内部细节（SM/warp/smem）见 [01 CUDA 基础](./01-cuda-基础与体系结构.md)。

## Q16. CUDA 编程中并行性与并发性的区别？（实一二·二面 Q3）

- **并行（parallelism）**：**同一时刻**多个任务真正同时执行——GPU 上 = 不同 SM 上的 block/warp 同时跑（硬件多执行单元）；空间维度概念；
- **并发（concurrency）**：**同一时间窗内**多个任务都处于推进中（时间片轮转/交替）——单核也能并发；CUDA 里典型 = **多 stream 的 kernel 并发执行**（资源足够时不同 kernel 占不同 SM 同时跑）、kernel 与 H2D 拷贝重叠、单 stream 内 kernel 依旧**串行**；
- 关系：并发是任务组织属性，并行是硬件执行属性；并发调度到多硬件单元就成为并行。GPU 语境的经典应用：多 stream + event 依赖做传输/计算 overlap（[01](./01-cuda-基础与体系结构.md) 的 Stream 章）、concurrent kernel 提高 SM 占用（小 kernel 互补）。

## Q17. 【编程】用 CUDA 实现向量加法（实一二·一面 Q10）

考点：grid/block 一维划分、越界 guard、`<<<blocks, threads>>>` 启动参数。

```cpp
__global__ void vec_add_kernel(const float *a, const float *b, float *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;   // 全局一维索引
    if (i < n) c[i] = a[i] + b[i];                      // 越界 guard（n 不被整除时）
}
void vec_add(const float *a, const float *b, float *c, size_t n) {
    int threads = 256;
    size_t blocks = (n + threads - 1) / threads;        // 向上取整
    vec_add_kernel<<<blocks, threads>>>(a, b, c, n);
}
```

- 代码位置：`solutions/cuda/src/reduce.cu`（`vec_add_kernel`/`vec_add`）
- 验证命令：`cd solutions/cuda && cmake -B build && cmake --build build && ctest --test-dir build -R reduce`
- 追问链：为什么 256 线程（wave 数量/占用）、float4 向量化怎么改、n 很大时 grid 维度上限（用 2D 或 grid-stride loop）——grid-stride 版：`for (i = tid; i < n; i += gridDim.x * blockDim.x)`。

## Q18. 锁的概念及常见类型（实一二·二面 Q1）

与 [09 C++ 与系统基础](./09-cpp与系统基础.md) Q14 同族。锁=保证互斥访问共享资源的同步原语，常见类型：**互斥锁**（mutex，阻塞让出）、**自旋锁**（忙等，临界区极短/多核友好）、**读写锁**（读共享写独占，读多写少）、**条件变量**（等待-通知，配合 mutex）、**信号量**（计数，控制并发度/生产者消费者）、**递归锁**（同线程可重入）、**乐观并发**（CAS/atomic，无锁）。扩展：死锁四条件（互斥/持有等待/不可剥夺/循环等待）与破坏手段（锁排序、try_lock、粗化）；分布式锁一句话（Redis setnx/RedLock、etcd lease）。Python 有 GIL 时锁退化为"保护非原子操作"（[09](./09-cpp与系统基础.md) Q19）。

## Q19. UDP 与 TCP 分别适用什么场景？优缺点？（实一二·一面 Q5）

| | TCP | UDP |
|---|---|---|
| 连接 | 面向连接（三次握手） | 无连接（发即走） |
| 可靠性 | 确认重传、排序、流控、拥塞控制 | 尽力而为，可能丢/乱序/重复 |
| 传输单元 | 字节流 | 数据报（有边界） |
| 头开销 | 20+ 字节 | 8 字节 |
| 时延 | 重传/拥塞回避引入抖动 | 恒定低延迟 |

场景：TCP——网页/API/文件传输/数据库（正确性优先）；UDP——实时音视频（WebRTC/直播，迟到不如丢）、DNS 查询、游戏、组播/广播、QUIC（UDP 上重建可靠传输 + 0-RTT，HTTP/3 的选择正是"两者结合"的例证）。训练集群关联点：NCCL 底层 IB 可靠传输 vs 自实现 UDP 协议（低延迟集合通信研究）。

## Q20. Linux 下查看磁盘使用情况的常用命令（实一二·一面 Q6）

- `df -h`：**文件系统**级别已用/可用/挂载点（第一反应）；
- `du -sh <dir>` / `du -h --max-depth=1`：目录/文件实际占用（排查大文件）；
- `lsblk`：块设备与挂载树；`fdisk -l`/`parted`：分区表；
- `iostat -x 1`：**io 利用率/吞吐/IOPS/await**（性能视角，区分"满"与"忙"）；`iotop`：按进程 io；
- `ncdu`：交互式空间分析；`stat <f>`：文件 inode/大小；`mount | column -t`：挂载选项。
追问常带：`df` 与 `du` 不一致的两种原因（已删除但进程持有的文件句柄——`lsof | grep deleted`；稀疏文件）。

## Q21. 【编程】实现一个 LRU Cache（一1 Q7）

哈希表 + 双向链表，O(1) get/put。详解见 [10 C++ 编程题](./10-cpp编程题.md) Q1（C++ 版，字节/阿里/阶跃同题）。

- 代码位置：`solutions/cpp/include/answers/lru_cache.hpp`（C++17）；`solutions/python/src/answers/algorithms/lru_cache.py`（Python 双链表版 + `LRUCacheOrdered` OrderedDict 版）
- 验证命令：`cd solutions/cpp && cmake -B build && cmake --build build && ctest --test-dir build`；`cd solutions/python && python -m pytest tests/test_algorithms.py -q -k lru`

## Q22. 【编程】二叉树的层序遍历（实1 Q5）

BFS + 队列，`for _ in range(len(queue))` 逐层收割带层级；变体右视图/锯齿遍历同框架。见 [12 算法题](./12-算法题.md) 树小节。

- 代码位置：`solutions/python/src/answers/algorithms/trees.py`（`level_order` / `right_side_view`）
- 验证命令：`cd solutions/python && python -m pytest tests/test_algorithms.py -q -k tree`

## Q23. 【编程】手写 Multi-Head Attention（实1 Q6）

要点：QKV 线性投影 → reshape `(B,S,H,d)` → transpose → scaled dot-product（`scale=1/√d`）+ causal mask → softmax → 加权 V → merge heads → 输出投影；工程版用 `F.scaled_dot_product_attention`。详解见 [04 Attention 与 KV Cache](./04-attention与kvcache.md) Q5（三变体 + 常见错误）。

- 代码位置：`solutions/python/src/answers/attention/mha.py`（`MultiHeadAttention`）；GQA/MQA/MLA 同目录
- 验证命令：`cd solutions/python && python -m pytest tests/test_attention.py -q`

## Q24. 【编程】判断链表是否存在环（实一二·一面 Q9）

快慢指针（Floyd 判圈）：fast 两步 slow 一步，相遇则有环；找入环点=相遇后一指针回头，同速再相遇。O(1) 空间。

- 代码位置：`solutions/python/src/answers/algorithms/linked_list.py`（`has_cycle` / `detect_cycle`）
- 验证命令：`cd solutions/python && python -m pytest tests/test_algorithms.py -q -k cycle`
- 追问：哈希集合版 O(n) 空间；为什么快慢一定相遇（相对速度 1，环形内逐步逼近）。

## Q25. 【编程】数组中的第 K 个最大元素（一2 Q10，LeetCode 215）

小根堆 size=k（O(n log k)）或 quickselect（平均 O(n)，最坏 O(n²) 随机化 pivot 规避）；面试先写堆版再口述 quickselect。

- 代码位置：`solutions/python/src/answers/algorithms/topk_heap.py`（`kth_largest` 堆版 / `quickselect_kth_largest` 快选版）
- 验证命令：`cd solutions/python && python -m pytest tests/test_algorithms.py -q -k topk`
- 内嵌核心（堆版 6 行）：维护 size=k 小根堆，遍历完堆顶即第 k 大：`heapq.heappushpop(h, x)`（len==k 时）。

## Q26. 【编程】接雨水（校 Q11，LeetCode 42）

双指针 O(n)/O(1)：每根柱子能接的水 = `min(左边最高, 右边最高) - 自身高度`；双指针维护两侧运行最大，矮的一侧结算（矮侧的答案已由其侧 max 决定）。DP 版见同文件。

- 代码位置：`solutions/python/src/answers/algorithms/dp.py`（`trap_rain_water`，DP 与双指针两版）
- 验证命令：`cd solutions/python && python -m pytest tests/test_algorithms.py -q -k rain`

## Q27. 【编程】最长有效括号（控实 Q6，LeetCode 32）

栈底存"最后一个未匹配位置"作哨兵：遇 `(` 压下标，遇 `)` 弹栈——弹出后栈空则压当前下标（新哨兵），否则以 `i - stack[-1]` 更新答案。O(n)/O(n)；DP/两遍计数法见文件注释。

- 代码位置：`solutions/python/src/answers/algorithms/sliding_window.py`（`longest_valid_parentheses`）
- 验证命令：`cd solutions/python && python -m pytest tests/test_algorithms.py -q -k paren`

## Q28. 【编程】二叉搜索树迭代器（淘2 Q8，LeetCode 173，要求从暴力到最优多解法）

三解法演进（详见 `bst_iterator.py` 模块注释）：

1. **暴力**：构造时全量中序展开成数组，next/hasNext O(1)，空间 O(n)；
2. **最优（受控栈）**：栈维护"当前最小节点及其祖先左链"，空间 O(h)，next 均摊 O(1)（每条边至多进出栈一次）；
3. **生成器版**：`yield` 惰性中序，语义同 2。

```python
class BSTIterator:                      # 解法 2：均摊 O(1)，空间 O(h)
    def __init__(self, root):
        self.stack = []
        self._push_left(root)
    def _push_left(self, node):
        while node:
            self.stack.append(node); node = node.left
    def next(self) -> int:
        node = self.stack.pop()
        self._push_left(node.right)     # 右子树的左链接上
        return node.val
    def hasNext(self) -> bool:
        return bool(self.stack)
```

- 代码位置：`solutions/python/src/answers/alibaba/bst_iterator.py`（三解法 + `iterate_all` 对拍）
- 验证命令：`cd solutions/python && python -m pytest tests/test_alibaba.py -q -k bst`
- 追问：均摊分析（n 次 next 总代价 O(n)）；如果允许修改树节点可做后继指针（O(1) 空间 Morris 遍历思想）。

## Q29. 【编程】手写 Transformer 的核心结构（实一二·二面 Q7）

分层写：词嵌入 + 位置编码（可学习/sin-cos 或 RoPE）→ N ×（**多头自注意力** + **FFN（两层 GEMM + 激活）**，各配残差与 LayerNorm——Pre-LN 现代主流）→ 输出投影。核心 MHA 见 Q23；白板简化版：

```python
class Block(nn.Module):                              # Pre-LN Transformer block
    def __init__(self, d, h, ffn=4 * d):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), MultiHeadAttention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
    def forward(self, x, mask=None):                 # (B, S, d)
        x = x + self.attn(self.ln1(x), mask)         # 残差流（[31] Q16）
        return x + self.ffn(self.ln2(x))

class MiniTransformer(nn.Module):
    def __init__(self, vocab, d=256, n_layers=4, h=8, max_len=1024):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)          # 面试可用可学习 PE，口述 RoPE 更优
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(n_layers))
        self.head = nn.Linear(d, vocab, bias=False)  # 常与 tok embedding 权重共享
    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.size(1), device=ids.device))
        for blk in self.blocks:
            x = blk(x, causal_mask(ids.size(1)))     # 下三角 mask
        return self.head(x)                          # logits -> CE loss（[31] Q27）
```

- 代码位置：`solutions/python/src/answers/attention/mha.py`（MHA 主体）；FFN/LN 手写 CUDA 版 `solutions/cuda/src/norm.cu`；Triton 版 `solutions/python/src/answers/triton_kernels/`
- 验证命令：`cd solutions/python && python -m pytest tests/test_attention.py -q`
- 追问链：Pre-LN vs Post-LN（训练稳定性）、causal mask 实现、权重共享（tie embedding）省参数、RoPE 的相对位置性质（[04](./04-attention与kvcache.md) Q8）。

## Q30. 项目经历类题目（阿2 Q5、云二1 Q1、实一二·二面 Q5、云实 Q4）

- **MLIR/TVM 工作怎么讲**（阿2 Q5）：按"问题（哪些算子/后端）→ 方言与 pass 设计（在哪个 IR 层做变换、为什么）→ 与现有 pass 的组合 → 评测（精度对拍 + 性能对比 baseline）"展开；准备一个具体 pass 的下降/变换例子；
- **算子优化/融合/图优化/稳定性工作经历**（云二1 Q1）：四个方向各备一个量化结果的故事，串联用 Q9-Q14 的方法论；阿里二面必下钻到实现细节与数字；
- **模型结构深入提问**（实一二·一面 Q8）/ **项目再追问**（二面 Q5）：对简历上每个模型能画出结构图、说清维度流（张量 shape 变化）与训练细节（数据量/超参/损失）；
- **围绕项目深入讨论**（云实 Q4）：通用框架见 [31](./31-阿里-训练与算法.md) Q32。

## 勘误与备注

- 本篇为批次 3 新增。与 [03](./03-gemm与算子优化.md)/[08](./08-编译器与性能分析.md)/[09](./09-cpp与系统基础.md)/[10](./10-cpp编程题.md)/[12](./12-算法题.md) 重复的题目：以专题篇与 `solutions/` 工程为准，本篇提供阿里问法下的要点复述与验证命令；新增可运行代码仅 `solutions/python/src/answers/alibaba/`（BST 迭代器等，见 Q28）与 `tests/test_alibaba.py`。
- Q12 GPU 生态、Q13 硬件趋势为时效性内容，以机制对比与推理链条为主作答。
