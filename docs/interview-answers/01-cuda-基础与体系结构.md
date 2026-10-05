---
title: "CUDA 基础与体系结构"
description: "CUDA 编程模型、内存层级、warp/bank conflict/Stream/CUDA Graph 等高频八股题解答"
pubDate: 2026-10-05
topic: "CUDA 基础"
order: 1
tags: ["CUDA", "高性能计算"]
---

覆盖题库：`AI-Infra-校招-1`（基础知识 1-8）、`AI-Infra-综合面经题库-6`（CUDA 基础 1-13）、`AI-Infra-综合面经题库-3`（Q1）、`AI-Infra-面经-1`（高性能计算 1-6、13）。

## 1. CUDA 软件层级模型（Grid / Block / Thread）

**（校招-1 基础 1）**

三层组织：一个 kernel 启动一个 **Grid**，Grid 划分为多个 **Block**，每个 Block 包含若干 **Thread**。要点：

- Block 内线程可通过 `__syncthreads()` 同步、通过 shared memory 通信；**Block 之间没有全局同步原语**（只能靠 kernel 边界或 cooperative groups / 原子轮询）。
- 硬件上以 **warp（32 线程）** 为调度单位：Block 按 warp 拆分发到 SM，warp 是真正"同时"执行 SIMT 的粒度。
- `threadIdx / blockIdx / blockDim / gridDim` 四个内建变量做坐标映射：`gid = blockIdx.x * blockDim.x + threadIdx.x`。
- Grid/Block 维度选择本质是在控制**并行度（占用 SM）** 与**同步/通信粒度（block 内共享资源）** 的折中。

## 2. CUDA 内存层级

**（校招-1 基础 2、面经-1 HPC 2）**

| 层级 | 作用域 | 容量级(sm_86) | 延迟 | 生命周期 |
|---|---|---|---|---|
| 寄存器 | 线程私有 | 255 个/线程 | ~1 cycle | kernel 内 |
| Local Memory（溢出） | 线程私有 | 走 L1/L2 | 慢 | kernel 内 |
| Shared Memory | block 内共享 | 最多 100KB/block（默认上限 48KB） | ~30 cycle | block 生存期 |
| L1/L2 Cache | 硬件管理 | L1 128KB/SM 与 smem 共享、L2 6MB | — | 硬件管理 |
| Global Memory | 全部线程 | 显存 24GB | 400-800 cycle（命中 L2 更低） | 可跨 kernel |

编程要点：`__global__`（核函数）、`__device__`、`__shared__`、`__constant__`（常量缓存广播读）、`__managed__`（统一内存）。

## 3. CPU 与 GPU 架构的本质差异

**（校招-1 基础 3）**

- **核心设计目标**：CPU 面向低延迟（大缓存、分支预测、乱序执行、深流水线，少量强核 + SIMD 宽度小）；GPU 面向高吞吐（海量简单 ALU、以线程级并行 TLP 隐藏内存延迟，而非靠缓存/预测）。
- **控制逻辑占比**：CPU 约 20-30% 晶体管在控制/缓存；GPU 80%+ 在计算单元。
- **上下文切换**：CPU 切换重量级（us 级）；GPU warp 切换零开销（寄存器硬件换组，SIMT 调度器每个 cycle 选就绪 warp）。
- **内存带宽**：3090 HBM ~936GB/s vs 服务器 CPU ~100-200GB/s，GPU 靠带宽养活大量 ALU。
- 一句话：**CPU 用复杂换延迟，GPU 用数量换吞吐**——所以 GPU 适合数据并行、分支少、计算密度高的任务。

## 4. SM 与 SP 的关系

**（校招-1 基础 4）**

- **SP（Streaming Processor，即 CUDA Core）**：最基础的单精度浮点/整数运算单元，执行一条标量指令。
- **SM（Streaming Multiprocessor）**：完整的小型处理器，包含：调度器（warp scheduler）、寄存器堆、Shared Memory/L1、若干 SP、若干 LD/ST 单元、SFU（特殊函数单元）；Hopper 前每 SM 含 4 个处理分区（partition）。
- 一个 SM 同时驻留多个 block（上限 16/32 个、2048 线程），调度其中的 warp 执行。3090 有 82 个 SM、每个 SM 128 个 FP32 SP。
- 类比：SM = 一个车间（调度 + 缓存 + 工位），SP = 车间里的一台机床。

## 5. CUDA Stream：同步流与异步流

**（校招-1 基础 5、综合-6 Q3）**

- Stream 是 GPU 上的**操作队列**：同一流内按序执行，不同流之间可以乱序/并发（需要硬件资源空闲）。
- 典型用法：`cudaMemcpyAsync` + kernel 分散到多个 stream，让**拷贝引擎（DMA）与计算引擎重叠**。
- **default stream（legacy NULL stream）**：隐式全局同步——它会对其他所有阻塞流做 synchronize（除非编译加 `--default-stream per-thread`），因此多流优化时若混入默认流，所有流的并发性被破坏，这是"default stream 潜在问题"的标准答案。
- 同步层次：`cudaStreamSynchronize`（流级）、`cudaDeviceSynchronize`（设备级）、`cudaEvent`（流间依赖）、`cudaStreamWaitEvent`（跨流依赖的推荐方式）。

## 6. L1 / L2 缓存的角色

**（校招-1 基础 6）**

- **L1**：每 SM 私有，物理上与 Shared Memory 共享同一块 SRAM（可配置划分）。负责：局部变量/寄存器溢出的 local memory 缓存、global load 的缓存（`__ldg` 只读路径走 texture/一级缓存）。
- **L2**：全芯片共享（3090 6MB），所有 Global Memory 访问的必经之路，也是**跨 block/SM 数据交换的"会合点"**（如 atomic 操作在 L2 完成）。GPU L2 是 write-back 的。
- 优化直觉：L1 命中看空间局部性（同 warp 连续地址），L2 命中看时间局部性与数据复用（tile 复用、persistent block）。

## 7. 使用 Shared Memory 的注意事项

**（校招-1 基础 7、综合-3 Q1、面经-1 HPC 13）**

1. **必须同步**：写后读之间 `__syncthreads()`；避免在条件分支里调用（部分线程不到达会死锁/未定义）。
2. **bank conflict**：shared memory 划分为 32 个 bank（每 bank 宽 4 字节、每 cycle 服务一个地址）。同一 warp 内多个线程访问**同一 bank 的不同地址**时串行化（n 路 conflict 延迟 ×n）。解决办法：
   - padding：`tile[32][33]`（+1 错位）；
   - 交换读写方向（转置 kernel 教科书做法）；
   - 广播访问（同地址）不冲突。
   - 实测见 `solutions/cuda`：矩阵转置 naive 547GB/s → smem+padding 版本写出合并后明显改善（4096² 下 601GB/s vs 537GB/s，L2 命中会压缩差距）。
3. **容量与占用率**：smem 越大每 SM 可驻留 block 越少，需用 `cudaFuncSetAttribute` 申请超过 48KB 的动态 smem（sm_86 上限 100KB）。
4. Global Memory 关注点则是**合并访存（coalescing）**：warp 内 32 个地址落在尽量少的 128B 事务内。

## 8. Kernel 性能优化的维度

**（校招-1 基础 8、面经-1 HPC 6）**

一个实用的 checklist：

1. **先测量**：Nsight Compute 看 SOL（Speed Of Light）、占用率、每指令周期数。
2. **访存**：合并访存、向量化（float4）、shared memory tiling、常量/纹理内存、减少冗余读写。
3. **计算**：减少非必要计算、用 FMA/SFU（`__expf` 等 intrinsics）、`#pragma unroll` 展开、ILP（每线程多独立变量打破依赖链）。
4. **并行度**：调整 block/grid 尺寸让 occupancy 与每个线程的工作量平衡；grid-stride loop。
5. **执行效率**：避免 warp divergence（分支按 warp 对齐）、避免 shared memory conflict。
6. **调度与启动**：小 kernel 合并（CUDA Graph）、减少 host-device 同步、多流重叠。
7. **算法层**：换复杂度更低的算法（如 scan 代替 sort）、分块复用（GEMM tiling）。

## 9. CUDA Graph 与 kernel launch 流程

**（综合-6 Q1）**

**launch 流程**：host 调用 `<<<>>>` → CUDA runtime 驱动接口把参数打包、命令写入推送缓冲区 → GPU 前端队列取出、分配 block 到 SM → SM 调度 warp 执行。一次 launch 的 CPU 侧开销 ~5-10μs，GPU 侧 gap ~2-3μs。

**CUDA Graph**：把一串 kernel/拷贝的 DAG 一次性捕获（stream capture）或手动构建，之后 `cudaGraphLaunch` 一次提交。收益：①消除每 kernel 的 CPU launch 开销（小 kernel 密集场景数倍提升）；②驱动可见全局依赖，可做更好的调度。代价：静态形状友好、动态控制流要条件节点；显存需按峰值预留（vLLM 中 CUDA Graph 与 batch size 分档配合的原因）。

## 10. blockSize / gridSize 如何确定

**（综合-6 Q2）**

- **blockSize**：通常是 128/256（warp 整数倍）；要考虑每线程寄存器数（`__launch_bounds__` 约束）、smem 用量对占用率的影响、尾块浪费（数据量不是 block 整数倍时最后一块的闲置）。归约类常用 256，GEMM 用 128-256 并配合每线程多元素。
- **gridSize**：让 `(grid × block)` 至少覆盖数据；大 数据用 grid-stride loop 把 grid 固定在 `SM数 × 每SM驻留block数`（persistent kernel），减少尾部效应。
- 终极答案：**用 Nsight Compute 扫参数**，理论值（占用率计算器）只是起点。

## 11. bank conflict 的产生原因与解决

**（综合-6 Q4，与第 7 题合参）**

见第 7 题。补充：32 个 bank 按地址交错分布（bank = (addr/4) % 32），因此**按列访问二维数组**（stride 恰为 32 的倍数）是 32 路 conflict 的典型场景；解决方案 padding / 改布局 / `cudaSharedMemConfig`（8 字节 bank 模式，适用 double）。

## 12. threadfence 的作用

**（综合-6 Q5）**

`__threadfence()` 保证**本线程在 fence 之前的所有 global/shared 写，对其他 block 的线程可见**（内存屏障 + 顺序性），但**不同步线程**（区别于 `__syncthreads()` 只保证 block 内有序）。

经典用途——跨 block 生产者-消费者：block A 写完数据后 `__threadfence()`，再置 flag（atomic），block B 读到 flag 后一定能看到数据。没有 fence 时写可能乱序到达，B 看到 flag 却读到旧数据。flags 版本用 `__threadfence_block()`；`atomicThreadFence` 语义类似。

## 13. 如何调试 CUDA kernel

**（综合-6 Q6）**

1. **printf 调试**：device `printf`（缓冲有限，适合少量线程）。
2. **cuda-gdb**：断点、单步、查寄存器（Linux 下主用）。
3. **错误检查**：每次 launch 后 `cudaGetLastError()` + 同步后 `cudaDeviceSynchronize()` 检查；封装宏。
4. **边界法**：先在极小数据（如 4×6 矩阵）上跑，结果与 CPU 对拍——本仓库 `solutions/cuda` 的所有 kernel 都带 CPU 对拍测试，这是最实用的习惯。
5. **工具**：compute-sanitizer（memcheck/racecheck 查越界与竞态）、Nsight Compute（逐指令/计数器）、Nsight Systems（时间线）。
6. 数值问题：`nan`/`inf` 检查、`-lineinfo` 保留行号、`__fdividef` 与 IEEE 差异。

## 14. Unified Memory 与 Zero-Copy Memory

**（综合-6 Q7）**

- **Unified Memory（`cudaMallocManaged`）**：单一虚拟地址空间，页在 CPU/GPU 间按缺页迁移（driver 按 hint/预取 `cudaMemPrefetchAsync` 调度）。简化编程；性能取决于迁移命中率。
- **Zero-Copy（`cudaHostAlloc` mapped + host 指针直接在 kernel 用）**：数据始终在 host 内存，GPU 通过 PCIe 远程读，**每次访问都过总线**，只适合一次访问、极小数据或 stream 流水。
- 区别一句话：UM 是"数据会搬家"，Zero-Copy 是"数据不搬家但每次都远程取"。

## 15. CUDA 中的排序算法

**（综合-6 Q8）**

- **bitonic sort（双调排序）**：数据无关的比较交换网络，O(n log²n)，最适合 GPU（无分支、规整并行），块内 shared memory 排序的经典。
- **radix sort（基数排序）**：工程首选（Thrust/CUB 的实现），对整型/浮点（可按位序重解释）分 pass 按 key 的高低位分桶，每 pass 是一次 scan + scatter，GPU 上 O(k·n) 且全并行。
- 归并路线：块内排序（bitonic/odd-even）→ 块间归并（双调合并网络或 multiway merge）。
- 底层依赖都是 **scan（前缀和）**——见 `solutions/cuda/src/histogram_scan.cu` 的 scan kernel。

## 16. sin 在哪个硬件单元执行？

**（综合-6 Q9）**

**SFU（Special Function Unit）**。每 SM 分区含 4 个 SFU，通过硬件近似（查表 + 插值）实现 `__sinf`（快速版）；标准 `sinf` 是软件多项式（FMA 在 FP 单元上算，精度更高更慢）。SFU 还支持：`__expf/__log2f`、`rsqrtf`、`rcp`、`sqrt` 近似。吞吐：SFU 每 SM 每 cycle 只有 4（FP32 是 128），所以 transcendental 函数是 1/32 的 FP32 速率——softmax 里 `__expf` 的选择是真实优化点。

## 17. Volta ITS（Independent Thread Scheduling）

**（综合-6 Q10）**

Volta 前后 warp 内分支的处理差异：

- **Pre-Volta**：warp 收敛点追踪，分歧路径**串行执行**，且有时双方都执行到 reconvergence point 才合并（有 bug 式的惰性调度）。
- **Volta ITS**：每线程独立 PC 与活跃掩码，调度器可以按任意线程子集调度，保证**进展保证（forward progress）**。divergence 代价仍在（SIMT 串行化），但不会互相饿死，且 `__syncwarp` 需要显式同步同一 warp 的线程（此前 warp 同步假设不再免费）。

## 18. 3090 单 block 最大 Shared Memory

**（综合-6 Q11）**

Ampere GA102（sm_86）：每 SM 128KB SRAM（L1+smem 共享），**单 block 上限 99KB（100KB 减去 1KB 保留）**，静态/常规动态申请默认上限 48KB——超过要用 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` 显式申请。同代 A100（sm_80）为单 block 163KB 上限（SRAM 192KB）。

## 19. PTX 与 SASS 的区别

**（综合-6 Q12）**

- **PTX**（Parallel Thread Execution）：NVIDIA 的**虚拟 ISA**（类 RISC、寄存器数量无限、稳定跨代），nvcc 前端输出。可被驱动 JIT 编译到具体架构，保证二进制向后兼容。
- **SASS**：真实硬件指令集（每代不同），nvcc 后端（ptxas）从 PTX 编译生成，最终执行的就是 SASS。
- 链路：CUDA C++ → NVVM IR（LLVM）→ PTX → ptxas → SASS → fatbin（可同时含多架构 SASS + PTX）。`cuobjdump -sass` 反汇编、`nvdisasm` 看 PTX。
- 工程含义：只发 PTX 的包需要目标机 JIT（首次启动慢、驱动必须够新）；产品化要 `-gencode` 预编译各架构 SASS。
- （本仓库 CUDA 工程实测中就踩过这个坑：CMake 未指定架构时只编出 sm_52 的 SASS+PTX，在 3090 上依赖驱动 JIT，`compute-sanitizer` 直接报 unsupported toolchain。）

## 20. 标称 TFLOPS 怎么计算

**（综合-6 Q13）**

$$\text{TFLOPS} = \text{SM 数} \times \frac{\text{每 SM 每 cycle 的 FMA 数} \times 2}{(\text{FLOP/FMA})} \times \text{频率}$

以 3090 为例：82 SM × 128 FP32 core × 2 FLOP(FMA) × 1.70GHz ≈ **35.6 TFLOPS**（FP32）。FP16 Tensor Core：82 × 512（每 SM 第三代 TC 每 cycle 512 FP16 FLOP）× 1.70GHz ≈ 71 TFLOPS（dense）。注意"标称"都是 FMA 计 2 FLOP、boost 频率、dense 精度；稀疏模式标称再 ×2。

## 21. Global Memory 与 Shared Memory 访存关注点（总结）

**（综合-3 Q1）**

- Global：合并访存（128B 事务）、对齐、L2 局部性、避免 host-device 频繁传输、分页锁定内存（pinned）提高 DMA 吞吐。
- Shared：同步正确性、bank conflict、容量与占用率折中、生命周期短（block 级）、动态 smem 的重叠（复用 buffer 减容量）。

## 22. Cache 工作原理 / 局部性 / 计算与访存密集

**（面经-1 HPC 3/4/5）**

- Cache 按 line（CPU 64B，GPU L2 128B）组织，命中层级决定延迟；映射方式（直接映射/组相联/全相联）影响冲突失效。
- 提升命中率：**时间局部性**（刚访问的尽快再访问，如 tile 内多轮复用）+ **空间局部性**（顺序访问、按行遍历行优先矩阵、结构体按访问模式排布 SoA/AoS）。
- 判定计算/访存密集：算术强度 $AI = \frac{\text{FLOPs}}{\text{Bytes}}$，与机器平衡点（如 3090：35.6TFLOPS / 936GB/s ≈ 38 FLOP/Byte）比较：AI 高于平衡点为计算瓶颈，反之为访存瓶颈——这就是 **Roofline 模型**（详见《GEMM 与算子优化》篇 Q1）。

## 23. Hopper TMA：优势、调用方式、是否过 L1（综合-1 基础 1）

**TMA（Tensor Memory Accelerator）**是 Hopper 起的专用异步数据搬运引擎：

**优势**：
1. **一个指令搬一个多维 tile**（最多 5D），替代每线程一条 LDG——指令数、地址计算大幅减少；
2. **异步**：`cp.async.bulk.tensor` 提交后线程立即返回，配合 mbarrier 做生产者-消费者流水（warp specialization 的基础）；
3. **硬件越界处理**：Box 超出 tensor 边界自动补零/填充，kernel 里不再需要边界分支；
4. **swizzle 内建**：装载时硬件完成 128B swizzle，直接匹配 smem bank 布局，免 padding；
5. 释放 SM 的 LSU 压力，深流水下带宽利用率更高。

**调用方式**：host 侧用 `cuTensorMapEncodeTiled` 创建 **TensorMap 描述符**（全局地址 + 形状 + stride + box 尺寸 + swizzle 模式），以 `const __grid_constant__ CUtensorMap` 参数传入 kernel；device 侧 `cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes` 发起搬运，`mbarrier.arrive.expect_tx` 登记字节数，consumer `mbarrier.try_wait` 等完成。

**是否过 L1**：TMA 全局→共享的搬运**默认绕过 L1**（直接 GMEM→SMEM，减少 L1 污染，"bulk" 语义）；L2 仍会经过。这也是它与 `cp.async`（Ampere，逐 16B，走 L1 路径）的关键差异之一。

## 24. 按列遍历行优先矩阵为何劣化、劣化的是哪个指标（综合-2 基础 2）

行优先存储中同一列的元素相距 `cols×4` 字节。按列访问时：
- **CPU**：每次取新列元素都要**换 cache line**（64B），空间局部性归零——劣化的是 **cache 命中率（命中率暴跌、每元素 miss）**，等效带宽 = line 利用率 4/64 = 6%；
- **GPU**：warp 内相邻线程地址跨行跳跃 → 一个 128B 事务只能服务 4B 数据 → 劣化的是**合并访存（coalescing）/ 全局内存事务效率**（sector 利用率），带宽损失可达 8-32 倍。

解决：按行遍历、缓存分块（tile 内先行后列）、转置后处理、或 SoA 数据布局重排。
