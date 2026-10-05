---
title: "T3 芯片厂题解（二）：国产 GPU 厂商"
description: "寒武纪 GEMM 8×8/Little's Law、壁仞 K 维寄存器压力、摩尔线程离群量化、沐曦 Triton 与 CUDA、遂原 Triton vs CUDA 全套、燧原 MLIR/XLA/TVM 编译器题"
pubDate: 2026-10-05
topic: "公司题解-T3"
order: 81
tags: ["面试解答", "CUDA", "量化", "编译器", "国产GPU"]
draft: false
---

覆盖题库：`寒武纪-AI-Infra-实习`、`壁仞科技-AI-Infra-实习`（一面）、`壁仞科技-AI-Infra-实习-一面`、`摩尔线程-AI-Infra-实习-二面`、`沐曦-AI-Infra-实习`、`沐曦-AI-Infra-实习-一面`、`遂原科技-AI-Infra-实习-一面`、`燧原科技-AI-Infra-社招-一面` 全部题目。CUDA 理论重叠部分引用《[01](./01-cuda-基础与体系结构.md)》《[02](./02-cuda-编程题.md)》《[03](./03-gemm与算子优化.md)》各篇并补充追问深度。

# 寒武纪（实习 一轮：算子优化）

## Q1. GEMM 中为什么通常让单个线程处理 8×8 的计算块？

**（寒武纪 实习 1-1）**

本质是**用寄存器复用换全局访存 + 构造 ILP**（原理与英伟达 ILP 题同源，见《[80 篇](./80-t3-英伟达与三星.md)》Q9）：

1. **访存密度提升 32 倍**：一个线程算 C 的 8×8=64 个输出，只需读 A 的 8 个行片段 + B 的 8 个列片段（配 smem staging 后是从 shared memory 读，但复用逻辑相同）。对比每线程 1 个输出：每输出要读 2 份数据；8×8 块：每 64 个输出读 16 份 → **每输出访存 1/32**；
2. **64 个独立累加器 = 64 条无依赖 FMA 链**：等待 global/smem 装载时计算不停（ILP 隐藏延迟），这正是低 occupancy 的 GEMM kernel 仍能打满吞吐的原因；
3. **指令开销摊薄**：地址计算、循环变量、`__syncthreads()` 每 block 的同步成本被 64 个输出分摊；
4. **warp 级 tile 的标准拼法**：64×64 的 block tile = 64 线程（2 warp）× 每线程 8×8；mma/Tensor Core 时代变为 warp 级片段（m16n8k16），但"每线程多累加器"的原则不变；
5. **代价（追问点）**：64 累加器 + 操作数 ≈ 128+ 寄存器 → occupancy 掉到 ~25%；smem 装载还需 double buffer 覆盖。8×8 是**寄存器预算（255 上限）与复用率**的平衡点，常见 4×4 / 8×8 / 8×4。

微内核形态（寄存器 tile GEMM 的核心循环）：

```cuda
// BM=BN=64, 64 threads, 每线程 8x8; A/B tile 已在 shared memory
float acc[8][8] = {0};                    // 64 个独立累加器
for (int k = 0; k < BK; ++k) {
    float a[8], b[8];
    #pragma unroll                       // 展开构造 ILP
    for (int i = 0; i < 8; ++i) a[i] = As[ty*8 + i][k];   // A 行片段
    #pragma unroll
    for (int j = 0; j < 8; ++j) b[j] = Bs[k][tx*8 + j];   // B 列片段
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] += a[i] * b[j];     // 64 条 FMA 无依赖背靠背发射
}
```

工程化：`solutions/cuda/src/gemm.cu::gemm_regtile_8x8`（BM=BN=64、BK=16，64 线程每线程 8×8 累加器），ctest `test_gemm` 对拍 + --bench 实测。

## Q2. 使用 CUDA 实现算子的主要难点

**（寒武纪 实习 1-2）**

分四类讲：

1. **正确性**：索引/边界（一维化下标算错是第一大 bug 源）、同步（`__syncthreads()` 放在分支内死锁；跨 block 依赖缺 `__threadfence()`）、原子操作的顺序性、**数值精度**（浮点不满足结合律，归约顺序不同结果不同——《[02 篇](./02-cuda-编程题.md)》Q1 的原粒追问）；
2. **性能与体系结构细节**：合并访存、bank conflict（padding/swizzle）、occupancy 与寄存器/smem 的三方权衡、warp divergence、launch 开销与小 kernel 合并；
3. **并行化设计**：选择并行维度（哪一维做 block、哪一维做 warp）、负载均衡（变长行、MoE 路由）、多级归约的边界处理（partial sum 的第二 kernel）；
4. **工程与调试**：异步执行模型（host-device 同步点设计、错误检查宏）、调试手段有限（printf 有缓冲上限、cuda-gdb 慢——完整方法论见《[01 篇](./01-cuda-基础与体系结构.md)》Q13）、性能回归测试（每个 kernel 配 CPU 对拍，本仓库 `solutions/cuda` 即此实践）。

## Q3. 是否用 CUDA 实现过前缀和（prefix sum）？

**（寒武纪 实习 1-3）**

实现过/知道三件事：块内 **Hillis-Steele** 包含式扫描（log₂n 轮、跨度翻倍、每轮一次 `__syncthreads()`）；多 block 全局前缀和三步走——①各块内扫描 → ②对每块末元素（或块和）再扫描得块偏移 → ③加回偏移（Blelloch 两遍 scan 是省带宽的精细化版本）；底层依赖 shuffle：warp 内 `__shfl_up_sync` 免 smem 完成扫描。完整代码与对拍：`solutions/cuda/src/histogram_scan.cu`（《02 篇》Q11）。

## Q4. 什么是 bank conflict，减少方法

**（寒武纪 实习 1-4、沐曦 实习 2）**

32 bank × 4B、warp 级粒度判定、广播不算冲突——完整机制见《[01 篇](./01-cuda-基础与体系结构.md)》Q7/Q11（英伟达面试的"粒度"深度版见《[80 篇](./80-t3-英伟达与三星.md)》Q3）。减少方法：**padding**（`tile[32][33]` 错一格）；**swizzle**（XOR 交织地址，GEMM 里不浪费带宽，CUTLASS 必用）；改数据布局（SoA / 转置访问方向）；广播/组播利用；64-bit 数据注意两 phase 各自判。**GEMM 场景**（沐曦追问）：A tile 按 k 连续读、B tile 按列读必 32 路 conflict → swizzle 或 padding 二选一，sm_80+ 用 `cp.async`/TMA + swizzle mode 让硬件在装载时完成重排。实测：`solutions/cuda/src/transpose.cu` 的 naive vs smem+padding 版（《02》Q4）。

## Q5. Little's Law 中访存延迟与计算延迟的关系

**（寒武纪 实习 1-5）**

Little's Law：**并发量 L = 吞吐率 λ × 延迟 W**。GPU 上的两种应用：

1. **访存**：要达到峰值带宽 BW，需要 in-flight 字节数 ≥ BW × 内存延迟。3090：936GB/s × ~500ns ≈ **468KB 的并发访存请求**。单 warp 一次 load 只有几百字节 → 必须靠**更多 warp（TLP/occupancy）或每线程更多 outstanding load（MLP，float4 × 多元素）**凑够 L——这就是"occupancy 用来填带宽"的定量表述（《80 篇》Q2/Q9）；
2. **计算**：要隐藏 FMA 依赖链的延迟，需要独立指令数 ≥ 延迟周期 × 发射吞吐 → 多累加器（ILP）或 warp 切换（TLP）。

**关系**：访存延迟（400-800 cycle）远大于计算延迟（4 cycle 依赖链），所以访存隐藏靠"量"（并发请求多），计算隐藏靠"独立链"（ILP）；两者共享寄存器预算，kernel 优化的本质是**用最小资源满足 Little's Law 的 L**。

## Q6. 介绍参与的开源项目

**（寒武纪 实习 1-6、壁仞 实习 1-1 同族）**

框架：项目定位一句话 → 我贡献了什么（PR 主题）→ 难点与收益（如"修了 FFN 融合 kernel 的访存次数计算，用 ncu 复测验证"）→ 对上游代码组织的理解。要点：说得出 upstream 的贡献流程与代码结构（例如本仓库即是开源题解/代码工程的组织实践）。

# 壁仞科技

## Q7. GEMM：K 维度远大于 M/N 时对循环体的影响（寄存器压力等）

**（壁仞 实习 3）**

设 M=64、N=64、K=1e6 这种"深而窄"的形态：

1. **并行度坍缩**：grid = (M/BM)×(N/BN) 个 block，M/N 小 → block 数远小于 SM 数（64×64/128×128 → 1 个 block！），大量 SM 空转。**解法：split-K**——把 K 切成 p 段分给 p 组 block 各算部分和（常驻寄存器的 acc 不变），最后用一个 reduction kernel 或 atomicAdd 合并；代价：额外显存写读 + 合并开销，p 一般取 2-8；
2. **循环体影响**：K 大 → 内循环迭代次数多 → 每次迭代从 smem 装载 A/B 切片 + FMA，**访存指令与循环开销占比上升**；`#pragma unroll` 展开 K（4-8 步）摊薄循环控制并构造 MLP；double buffer（装载与计算重叠）收益变大（流水线填充成本被长 K 摊薄）；
3. **寄存器压力**：注意 K 大**不增加累加器**（acc 数只由 BM×BN/线程数决定，8×8=64 不变）；真正的压力来自展开后持有更多的 A/B 片段寄存器和地址寄存器——展开过度会 spill 到 local memory 反而变慢；
4. **数值精度**：K 很长时顺序累加的浮点误差线性累积（float 相对误差 ~K·ε），split-K 顺带把累加变成树形结构，误差降为 ~log 级（与原粒"reduce 顺序影响精度"同源，见《[82 篇](./82-t3-cpp与系统.md)》原粒 Q2）；
5. **访存特性**：AI = 2MNK/(2(MK+NK+MN)) ≈ 2MN/(M+N)，与 K 无关且很小 → 天然 memory bound；A/B 的 tile 在 L2 的复用率决定成败。

## Q8. NEON 汇编指令的基本用法

**（壁仞 实习 4）**

ARM 的 128-bit SIMD，31 个 `v0~v31` 向量寄存器，每条指令处理 4×fp32 / 8×fp16 / 16×int8：

```c
// intrinsics 版（面试写这个，汇编口头提）
float32x4_t va = vld1q_f32(a + i);        // 装载 4 个 float（对应汇编 vld1.32 {d0,d1},[r0])
float32x4_t vc = vmlaq_f32(vc, va, vb);   // vc += va * vb （VFMA，单条指令）
vst1q_f32(c + i, vc);                     // 写回
```

要点：**每元素独立 lane**（不是 dot product，累加要用 `vpaddq`/reduce）；CPU 端 GEMM 微内核用 NEON 4×fp32 × 寄存器块（如 12×4 微内核）叠加 cache blocking，思路与 GPU 8×8 线程块同源（Q1）——装载进寄存器、复用、展开 ILP；汇编层面关注：`vld1/vld2/vld4`（交错装载）、`vfma` 融合乘加、`vtrn/vzip`（数据重排，消冲突）、尾块用 mask（`bic`）处理非 4 倍数长度。

## Q9. CUDA 程序的调试工具与方法

**（壁仞 实习 5）**

完整方法论引用《[01 篇](./01-cuda-基础与体系结构.md)》Q13：错误检查宏（每次 launch 后 `cudaGetLastError` + 同步检查）→ 小数据 CPU 对拍（最实用）→ compute-sanitizer（memcheck/racecheck 查越界与竞态）→ cuda-gdb 断点 → device printf（限量）→ `-lineinfo` + Nsight Compute 定位行级 stall。工具分工：nsys 看时间线、ncu 看单 kernel 指标（《[08 篇](./08-编译器与性能分析.md)》Q5）。

## Q10. 判断两个链表是否相交（O(n+m)）/ TopK 用大根堆还是小根堆

**（壁仞 实习一面 3/4）**

**链表相交**（LeetCode 160）：双指针 —— `pA` 走完 A 接着走 B，`pB` 走完 B 接着走 A，两者走过的长度都是 a+b，必在交点同步相遇（无交点则同时到 nullptr）；O(n+m) 时间 O(1) 空间。注意先各自遍历求长度差再对齐起点的做法等价。已有实现：`algorithms/linked_list.py::get_intersection_node`。

**TopK（最大的 K 个数）**：**小根堆**，容量 K——堆顶是"当前第 K 大"，是 K 个候选里最弱者；新元素 > 堆顶才入堆。O(n log k)。反问"为什么不用大根堆"：大根堆要全量建堆 O(n) + 取 K 次 O(k log n)，或无法在线维护 K 个候选。更快的选择：quickselect 平均 O(n)（但无序、不支持流式）。**口诀：求最大 K 用小根堆（小顶堆守门），求最小 K 用大根堆**。已有实现与测试：`algorithms/topk_heap.py::kth_largest / quickselect_kth_largest / top_k_frequent*`。

```bash
cd solutions/python && pytest tests/test_algorithms.py -q
```

# 摩尔线程（实习 二面）

## Q11. INT8 量化与 W4A16 量化的流程及适用场景

**（摩尔线程 二面 2）**

**INT8（W8A8，权重+激活都量化）流程**：①校准（几百条代表性数据跑前向，统计各层激活分布）→ ②权重 per-channel 对称量化、激活 per-tensor 非对称/静态或动态量化；激活离群值明显时先做 **SmoothQuant** 等价缩放迁移 → ③可选 QAT/部分层回退（首尾层、norm 保 fp16）→ ④部署：整数矩阵乘（INT8 Tensor Core，峰值是 FP16 的 2×）+ scale 反量化融合在 epilogue。**场景**：compute bound 的 prefill/大 batch、训练侧量化——收益来自算力翻倍。

**W4A16（Weight-Only）流程**：①校准集上逐层重建（**GPTQ**：Hessian 贪心逐列量化 + 误差补偿；**AWQ**：按激活幅度做 per-channel 缩放保护少量显著通道）→ ②权重压到 4bit + per-group scale（128 一组）→ ③推理：装载时寄存器内解包（位操作）→ FP16/BF16 计算的融合 kernel（Marlin 是标杆）。**场景**：decode 的 memory bound 阶段——收益来自**权重读取量 ÷4**（显存 1/4、瘦 GEMM 接近 4× 带宽收益）。详见《[06 量化](./06-量化.md)》Q1/Q2/Q3。

一句话对比：**W8A8 提升算力利用率（计算受限），W4A16 提升带宽利用率（访存受限）**；W4A16 校准更讲究（4bit 容错小），W8A8 更怕激活离群值。

## Q12. 数据分布极度不均匀（-10000 离群）选什么量化方案

**（摩尔线程 二面 3，场景题）**

先算为什么坏：min-max 的 scale = (max−min)/255 ≈ 10001/255 ≈ **39.2**，而主体数据在 [-1,1] → 正常值全部塌缩到同一个量化级，信息全丢。方案按优先级（《[06 篇](./06-量化.md)》Q6 同源）：

1. **离群分离/混合精度（首选，LLM.int8 思路）**：检测含离群（amax > 阈值）的通道/元素抽到高精度旁路，其余走 INT8——GEMM 拆成 `x_inlier @ W_int8 + x_outlier @ W_fp16` 两块；误差最小；
2. **per-group / per-channel 细粒度**：离群只污染所在组，其他组 scale 正常；若离群是**通道型**（LLM 典型），per-channel 权重 + SmoothQuant 迁移即可救回；
3. **percentile/KL 截断**：按 99.9% 分位 clip——若离群值本身语义不重要（如传感器毛刺），牺牲它换主体分辨率；
4. **变换**：log1p 压缩动态范围后再量化（对称反变换），适合长尾分布统计量的场景。

可运行对比（4 种方案的 RMSE）：`solutions/python/src/answers/t3/outlier_quantization.py`：

```bash
cd solutions/python && pytest tests/test_t3.py -k "outlier or minmax or percentile or per_channel" -q
# 结论: 离群分离 RMSE < minmax 的 5%; percentile 正常主体误差 <0.05 但离群点误差 ~1e4
```

## Q13. 手写一个 CUDA 算子（纯文本环境）

**（摩尔线程 二面 4）**

纯文本/白板环境的策略：**选自己最熟的 elementwise 起手**（向量加/SiLU/缩放），把"三件套"写稳——**索引、边界、步长**，再按追问深化。模板（SiLU：`x·sigmoid(x)`）：

```cuda
__global__ void silu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)                             // 边界: 纯文本环境最易漏
        y[i] = x[i] / (1.0f + __expf(-x[i]));   // SiLU = x * sigmoid(x)
}
// 启动: silu_kernel<<<(n + 255) / 256, 256>>>(x, y, n); CUDA_CHECK(cudaGetLastError());
```

工程化：`solutions/cuda/src/elementwise.cu::silu`（精确 expf 版），ctest `test_elementwise` 与 CPU 对拍（误差 < 1e-5 相对量级）。

追问演进准备好：①grid-stride 版（grid 固定大小，支持任意 n、persistent 形态）；②float4 向量化（对齐 + 尾块 fallback）；③与 bias/scale 融合成一个 kernel（省一次读写）；④归约类怎么写（引到《02》Q1）。验证习惯：写完立刻说"我会用 CPU 参考实现对拍 + ncu 看带宽"。

# 沐曦

## Q14. CUDA 编程中并行性与并发性的区别

**（沐曦 实习一面 2）**

**并行（parallelism）**：多执行单元**同一时刻**真同时执行——GPU 多 SM 同时跑多个 warp、CPU 多核。**并发（concurrency）**：任务在**一段时间内交替推进**，逻辑上"同时"，单核时间片轮转也是并发。CUDA 里的对应：多 SM 的 warp 并行执行是并行；同一 SM 上多个 warp 分时共享执行部件是并发调度（SIMT 调度器每 cycle 挑就绪 warp）；多个 stream 各自的任务在计算/拷贝引擎间**并发执行**（overlap），引擎内部各自并行。面试收口：**并行看硬件执行单元，并发看任务推进重叠**；GPU 的吞吐来自两者叠加。

## Q15. C++ 面向对象三大特性

**（沐曦 实习一面 3）**

**封装**（访问控制隐藏实现，不变量保护，`class` 默认 private）、**继承**（代码复用 + is-a 子类型，构造顺序基类先、析构反序）、**多态**（同一接口不同行为：**静态多态** = 重载/模板/CRTP，编译期绑定；**动态多态** = 虚函数 + vtable，运行期绑定——底层机制见《[09 篇](./09-cpp与系统基础.md)》Q8，构造函数中调用虚函数的行为见《[82 篇](./82-t3-cpp与系统.md)》太初 Q7）。

## Q16. std::map 与 std::unordered_map 底层对比

**（沐曦 实习一面 4）**

| | `map` | `unordered_map` |
|---|---|---|
| 底层 | 红黑树（有序平衡 BST） | 开链哈希表（bucket 数组 + 链表/红黑树节点） |
| 查找/插入 | O(log n) 稳定 | 均摊 O(1)，最坏 O(n)（全碰撞），rehash 时抖动 |
| 迭代 | **按 key 有序** | 无序，rehash 后迭代器失效 |
| 内存 | 每节点 3 指针 + 颜色 | bucket 数组 + 每节点 1-2 指针 + 负载因子预留 |
| key 要求 | `operator<`（严格弱序） | hash + `==`；自定义类型要写 `hash` 特化 |
| 选择 | 区间查询/有序遍历/最坏稳定（实时系统） | 纯点查、追求平均性能 |

追问点：负载因子默认 1，超过 rehash（bucket 翻倍）；`reserve` 预分配避免运行中 rehash；红黑树保证"最长路径 ≤ 2×最短"使最坏 O(log n)。与 CUDA 结合：kernel 内不可用 STL，但 host 侧算子元数据管理常用两者。

## Q17. new、malloc、智能指针的使用场景与区别；new 与 malloc 的底层实现

**（沐曦 实习一面 5/6、北极雄芯 malloc 同族）**

**区别表**（引用《[09 篇](./09-cpp与系统基础.md)》Q6）：malloc 是库函数返回 `void*` 需手动算大小、失败返回 NULL；new 是运算符可重载、调用构造函数、失败抛 `bad_alloc`、`new[]` 额外记录元素个数供 `delete[]` 析构；delete/free 错配是 UB。**智能指针场景**：所有权明确唯一用 `unique_ptr`（零开销）、共享所有权用 `shared_ptr`（控制块引用计数，线程安全的计数 + 非线程安全的对象）、观察不拥有用 `weak_ptr`/裸指针；RAII 是异常安全的根基（手写实现见《[10 篇](./10-cpp编程题.md)》Q9）。

**malloc 底层（glibc ptmalloc）**：小块（<128KB tcache/fastbin/small bin 尺度）从**堆**（`brk` 扩展）分配，维护分级空闲链表——thread cache（无锁）→ fastbin（单链 LIFO）→ small/large bin（size 分桶）→ unsorted bin（回收站，先尝试合并相邻空闲块防碎片）；大块（≥128KB 默认 `mmap_threshold`）直接 **mmap** 单独映射，free 即 `munmap` 归还 OS。每块前有元数据头（prev_size/size + 标志位），这就是 malloc(1) 实际占 32B 的原因、也是越界写踩坏下一个块元数据导致崩溃的机制（《09》Q20 的数组越界题）。**new 底层**：`operator new`（默认转调 malloc）→ placement new 在裸内存上构造。完整展开见《[82 篇](./82-t3-cpp与系统.md)》北极雄芯 Q6。

## Q18. lambda 表达式的语法与使用方式

**（沐曦 实习一面 7）**

语法 `[捕获](参数) mutable -> 返回类型 { 函数体 }`。**捕获**：`[x]` 值捕获（拷贝进闭包对象，默认 const，`mutable` 才可改）、`[&x]` 引用捕获（悬垂风险）、`[=]/[&]` 全捕、`[this]` 捕获 this（成员访问，本质是捕指针）。**实现原理**：编译器生成一个匿名仿函数类，捕获变量即其成员，lambda 体是 `operator()`——所以**有捕获的 lambda 有状态、不可转函数指针；无捕获的可以**。使用：STL 算法谓词（`sort(..., [](a,b){...})`）、`std::thread` 入口、回调与 `std::function`（类型擦除带来堆分配与间接调用开销，模板参数传 lambda 可避免）。深度内容见《[09 篇](./09-cpp与系统基础.md)》Q13。

## Q19. 手写 Transformer 结构 / 两数之和 / 哈希算法题

**（沐曦 实习一面 8/9/10）**

**手写 Transformer**：必答组件链——`embedding + 位置编码 → N × [RMSNorm → 因果 MHA（QKV 投影→分头→scaled dot-product→mask→合并）→ 残差, RMSNorm → FFN(SwiGLU/GELU) → 残差] → 最终 norm → LM head`。可运行精简实现（含因果掩码、可学习位置编码、RMSNorm）：

```python
# solutions/python/src/answers/t3/minimal_transformer.py（节选, 完整可训练）
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
mask = torch.triu(torch.ones(s, s, dtype=torch.bool), 1)
att = att.masked_fill(mask, float("-inf"))     # decoder-only 的因果掩码
y = F.softmax(att, dim=-1) @ v
```

测试验证两件事：因果性（截断后缀不改变前缀 logits）与可学习性（200 步拟合一一映射规则，loss 降一半以上）。Attention 家族完整版（MHA/GQA/MLA/FA）在《[04 篇](./04-attention与kvcache.md)》Q4/Q5 与 `answers/attention/`。

```bash
cd solutions/python && pytest tests/test_t3.py -k transformer -q
```

**两数之和**（LC 1）：哈希一遍法——遍历时查 `target - x` 是否见过，O(n)/O(n)。实现：`answers/t3/two_sum.py`（`pytest tests/test_t3.py -k two_sum`）。对比：暴力 O(n²)、排序+双指针 O(n log n) 但需恢复下标。

**哈希类题型套路**：需要"此前见过什么"就上哈希——两数之和（值→下标）、字母异位词分组（排序串做 key / 计数元组做 key）、最长无重复子串（滑窗 + 最后出现位置，`algorithms/sliding_window.py::length_of_longest_substring`）、前 K 高频（哈希计数 + 桶/堆，`algorithms/topk_heap.py::top_k_frequent*`）。

## Q20. GEMM 项目的 tiling 策略 / 行 reduce sum CUDA 算子

**（沐曦 实习 1/2/3）**

**tiling 策略**：回答框架 = 制约因素（《[80 篇](./80-t3-英伟达与三星.md)》Q4：寄存器/smem/wave 量化/L2 复用）+ 起点（BM×BN 64-128、BK 32-64、每线程 8×8 累加器，见本篇 Q1 微内核）+ 实测调整（ncu 看 SOL 与 L2 hit，扫 tile 四元组；本仓库教学版 16×16 smem tiling 已 1.3×，`solutions/cuda/src/gemm.cu`，《02》Q3）。

**行 reduce sum**：一行一个 **warp**（不是 block）：warp 内 `__shfl_down_sync` 归约免去 `__syncthreads()`，一个 block 挂 8 个 warp 同时算 8 行——比"一行一 block"调度开销小、比"一线程扫一行"并行度高。完整 kernel（已对拍）在《[02 篇](./02-cuda-编程题.md)》Q13：`solutions/cuda/src/pool_iou.cu::row_sum_warp_kernel`。

# 遂原科技（实习 一面：量化 + Triton 全套）

## Q21. 量化策略选择依据：为何 INT8，A100/H100 对量化的支持

**（遂原 3）**

**选 INT8 的依据**：①部署硬件有 INT8 Tensor Core（A100 起）；②权重+激活带宽减半、显存减半；③校准后精度损失可接受（<0.5% ppl 波动）。更深一层按瓶颈选：decode 是 memory bound → weight-only（W4A16）收益直给；prefill/大 batch compute bound → W8A8 吃 INT8 算力翻倍；显存装不下模型是硬约束 → 优先 weight-only。

**硬件支持**（《[06 篇](./06-量化.md)》Q5）：**A100**：INT8 Tensor Core（624 TOPS，FP16 的 2×）+ BF16，**无 FP8**；**H100**：FP8（E4M3/E5M2，1979 TFLOPS dense）+ INT8；**B200**：原生 FP4（NV FP4 块 scale）。所以 A100 上做 FP8 无硬件加速（只能模拟），H100 起才是真收益；W4A16 的 Marlin kernel 需 sm_75+。

## Q22. 量化对象是权重还是 KV Cache，scale 如何确定

**（遂原 4）**

**对象选择**：权重量化——省显存 + decode 带宽，静态离线做（GPTQ/AWQ）；KV Cache 量化——省的是**随序列长度增长**的显存与带宽（长上下文/大 batch 收益大），权重小模型场景 KV 反而是大头。两者正交可叠加。激活一般**不落盘量化**（每步动态算）。

**scale 确定**（《06 篇》Q4）：**权重**——静态，per-channel 对称（max|x|/127）或 W4 的 per-group(128)；GPTQ/AWQ 在校准集上逐层"重建"微调 scale/权重补偿量化误差。**激活**——动态 per-token/per-tensor（运行时 amax）或静态校准（minmax/percentile 99.9%/KL 散度）。**KV Cache**——K 和 V 分开处理：K 按通道（channel）非对称（离群集中在特定 channel）、V 按 token；用 per-block/per-head scale（FP8 KV 几乎无损；INT4 KV 需要 KIVI 式通道级处理）。校准数据必须来自真实业务分布（几百条即可）。

## Q23. 量化后精度损失如何评测

**（遂原 5、原粒 Q3 同族）**

三层：①**逐层快速筛查**——量化前后激活的余弦相似度/MSE，定位"坏层"（首尾层、down_proj 常见）回退 FP16；②**任务级**——PPL（快速但敏感度低，PPL 正常不代表无损）→ 下游 benchmark（MMLU/GSM8K/代码）对比 fp16 baseline；③**业务回归**——长文本/多轮/重复类敏感 case（量化对极端值分布的任务掉分更狠）。完整口径见《[06 篇](./06-量化.md)》Q7。

## Q24. Triton 算子的实现逻辑（分块策略）；baseline 选择与数据类型

**（遂原 6/7）**

**Triton 编程模型**：一个 program instance ≈ 一个 block；`tl.program_id` 拿块坐标，`tl.arange(0, BLOCK)` 构造块内向量（编译期常量），`tl.load(ptr + offs, mask=..., other=0)` 带掩码装载（越界自动补零——这就是"边界处理被语言原语吸收"），`tl.dot` 映射 Tensor Core，块内归约 `tl.max/tl.sum`。**分块策略**：最后一维（连续）切块保证合并访存；reduce 类一行一块（`BLOCK = next_pow2(row_len)`，mask 补齐）；matmul 按 (BM, BN, BK) 三级 tile + `num_stages`（软件流水级数）autotune。本仓库示例：`answers/triton_kernels/`（softmax/layernorm/matmul，`pytest tests/test_triton.py`，4096² softmax 实测 0.16ms / 825GB/s）。

**baseline 选择**：对照组要"官方且同语义"——torch eager 算子、`torch.compile`、`F.sdpa`（attention）、cuBLAS（matmul）。**数据类型必须对齐**：fp16 与 fp16 比（跨精度比较无意义）；注意 PyTorch matmul 默认开 TF32（`torch.backends.cuda.matmul.allow_tf32`）——baseline 比 Triton fp32 快一倍常是 TF32 偷跑，对拍时显式关掉或同开。

## Q25. 性能提升数据来源；动态分块策略与算子配置

**（遂原 8）**

**数据来源**：`triton.testing.do_bench`（预热 + 多轮回调取中位数，锁 CUDA 图/flush L2 选项）或手写 cudaEvent 计时 + `torch.cuda.synchronize`；配套正确性对拍（`torch.testing.assert_close` 设 atol/rtol）。可信度三要素：预热（剔除 JIT 编译）、多轮取中位（抗调度抖动）、同一 baseline 同一 dtype。进阶：ncu 直接量 kernel 的 DRAM 吞吐与 SOL（《[08 篇](./08-编译器与性能分析.md)》Q5）。

**动态分块/配置**：不同 shape 的最优 BLOCK/stages/num_warps 不同——①`@triton.autotune(key=["M","N","K"])` 按 key 缓存每次实测最优配置；②shape 分档（bucketing）：运行时按档位查表选 kernel 变体；③极端 shape 走特化分支（变长序列把 padding 补到 2 的幂再算）。工程上要防止 autotune 在线上第一次触发（预热阶段扫完）。

## Q26. 为什么选 Triton 而不是 CUDA；profiling 与后续优化

**（遂原 9/10）**

**选 Triton 的理由**：①开发效率——Python 写块级伪代码，索引/掩码/共享内存同步全自动，迭代周期比 CUDA 短一个量级；②自动调优——autotune 扫 (BLOCK, stages, warps) 免手工；③**可移植性（国产芯片公司的关键）**——Triton 前端可换后端（多家国产 GPU/AMD 都在适配），一次实现多平台跑；④实测中 elementwise/reduction/常规 matmul 的性能已达手写 90%+。

**CUDA 仍不可替代的场景**：需要精细控制 TMA、warp specialization、`cp.async` 手动流水、内联 PTX（`ldmatrix`/`stmatrix`）、cluster/DSMEM、非标准 tile 形状的极限 kernel（FA3 级）；调试工具链（SASS 级、sanitizer）也更成熟。**结论**：业务算子库用 Triton 快速铺面，热点 kernel 用 CUDA/CUTLASS 压榨极限。

**profiling 与指标**：ncu 看——DRAM Throughput（对照峰值：A100 2039GB/s、3090 936GB/s，见 Q28）、L2 Hit Rate（tile 复用）、SM/Memory SOL 谁高谁低、Occupancy、Warp Stall 原因。**后续优化套路**：带宽型 kernel → 向量化（最大合并宽度）、消除冗余读写（融合）、L2 局部性（block 调度顺序）；计算型 → tl.dot/TC 利用率、展开。方法论引用《[03 篇](./03-gemm与算子优化.md)》Q4。

## Q27. Attention 在端到端延迟中的占比；decode 是 compute 还是 memory bound；KV 量化提升什么；A100 带宽

**（遂原 11/12/13）**

**占比**：分阶段——**prefill**（长序列 O(S²) 计算）：attention 通常占 30-60%（FA 之前可到 70%+），FA 之后剩下的是 GEMM 为主；**decode**：每步 1 token，attention 是读 KV 的 GEMV（memory bound），大权重 GEMM 占主导，attention 占比随上下文变长/模型变小而上升。测量方法：nsys/torch profiler 时间线按 kernel 聚类，不要拍脑袋。

**decode 的瓶颈**：**memory bound**——每 token 的 FLOPs 固定（2N），而要从显存搬 权重 + KV（字节 ∝ N + S），batch 小时算术强度 AI ≈ 2N/(2N+KV bytes) < 1，远低于机器平衡点（A100：19.5TFLOPS/2TB/s ≈ 10）。（《[05 篇](./05-推理引擎与优化.md)》Q1）。

**KV 量化提升的是**：①**带宽**——KV 读取字节数 ÷2（FP8）/÷4（INT4）→ TPOT 直接下降（上下文越长越明显）；②**显存容量**→同卡能塞更大 batch/更长上下文/更多并发序列，**吞吐**随之上升；③KV 换出（CPU offload）时 PCIe 流量也减半。它**不提升**权重读取部分——那要靠 weight-only 量化。

**A100 理论带宽**：SXM 80GB（HBM2e）**2039 GB/s**（约 2.0TB/s）；PCIe 版 1935 GB/s。对照：V100 900、H100 SXM 3350、3090 936、4090 1008 GB/s。数字顺手推出算力利用：实测 kernel 带宽 / 峰值 = 带宽利用率（memory bound kernel 85%+ 算合格）。

# 燧原科技（社招 一面：编译器）

## Q28. MLIR 的表示方式与优化流程

**（燧原 1）**

**表示**：一切皆可嵌套的**方言（Dialect）**结构——每个 dialect 定义一组 op/类型/属性（TableGen .td 声明），同一程序可以同时含有多个 dialect 的 op（`tensor/linalg` 算子级 → `affine` 循环多面体 → `vector` 向量 → `gpu/nvvm/llvm` 硬件级），块（Region/Block）+ SSA 值构成可嵌套控制流。**相比单层 IR 的价值**：每层抽象只做该层最优的变换，lowering 路径可按目标硬件定制——芯片厂商（燧原）在 `linalg` 之上接自研 dialect（描述自家加速单元），之下替换 codegen 后端。

**优化流程（渐进 lowering pipeline）**：`导入(torch-mlir/stablehlo) → 图级 pass（常量折叠、融合、布局变换/shape 推导） → linalg → tiling/融合（转成循环） → affine → 多面体调度（交换/分块/倾斜） → vector → 向量化（映射 SIMD/矩阵指令） → bufferization（张量→memref，in-place 分析决定 buffer 复用——内存规划的核心） → llvm/自研后端`。每个 pass 是 pattern rewrite（DAG-to-DAG 重写规则），pass 管道按 `-pass_pipeline` 组合。多面体模型与 MLIR 的关系见《[08 篇](./08-编译器与性能分析.md)》Q2/Q3。

## Q29. XLA 在动态性方面的不足及应对

**（燧原 2）**

XLA 的设计假设是**静态 shape**（编译期已知全部形状，才能做融合/布局/内存规划的全局优化），动态性的不足：

1. **动态 shape 触发重编译**：每个新 shape 一 JIT 编译（百 ms 到秒级），变长输入（LLM decode 的 seq 增长、变长 batch）造成编译风暴；缓解靠 shape bucketing（pad 到分档），但 padding 浪费算力；
2. **动态算子支持弱**：`dynamic_slice/scatter/gather/unique` 等数据依赖形状/索引的算子难以静态优化，落到慢路径；控制流（while/条件）的 XLA 图化也弱；
3. **自定义算子困难**：新算子要写 HLO 或者走 custom call 桥接回手写 kernel——这恰是 NPU 厂商"手写算子 + custom call"的生存空间；
4. **编译延迟本身**：大图编译分钟级，服务冷启动慢。

应对方案：shape 分档 + padding；`--xla_try_split_tensor`/缓存（二级缓存复用编译产物）；热点子图手写算子 + custom call 注入；或改用 torch.compile（Inductor 动态 shape 支持 `dynamic=True`、graph break 回退 eager，动态性友好得多）。

## Q30. TVM 自动代码生成的现状与局限，NPU 手写算子与自动生成的性能对比

**（燧原 3）**

**现状**：TVM（relay/relax 前端 + TE/TIR 调度）+ Ansor/Meta-Schedule 自动搜索调度，在**标准 GPU/CPU** 上能逼近手工库（conv/matmul 常规 shape 达 cuBLAS/cudnn 的 80-95%）；Triton 式"块级编程 + autotune"进一步降低门槛。

**局限**：①搜索空间按"smem + 线程层级"的经典 GPU 模型构造，**NPU 的非标准内存层级**（多级片上存储/专用 DMA/奇特的 cluster 形状/固定功能的卷积-矩阵单元）难以映射，需要重写整个 schedule 空间（bring-your-own-codegen 成本高）；②搜索是**每 shape 每硬件**重复进行的，编译时间小时级，产品化要预编译海量 kernel；③指令集/性能模型不公开的 NPU 上，cost model 无法建立，搜索退化为盲试；④融合机会受限于调度原语表达力（跨算子的复杂流水/WS 类还是人写得快）。

**性能对比（经验值）**：通用算子自动生成 vs NPU 手写库——常规 shape 手写通常仍领先 10-30%（手写能利用文档不公开的指令特性、手工流水与双缓冲、算子间融合的特化），极端 shape 差距更大；自动生成赢在覆盖长尾 shape 与迭代速度。**业界折中**：热点算子（GEMM/卷积/attention）手写 + 图编译器自动融合长尾（燧原"MLIR + 手写算子库"正是此路线；vLLM/PyTorch 生态同样 cuBLAS/FA 手写 + Inductor 调度）。TVM 基础概念见《[08 篇](./08-编译器与性能分析.md)》Q1。

---

**下一篇**：[82 T3 芯片厂题解（三）C++ 与系统](./82-t3-cpp与系统.md)（太初/北极雄芯/后摩/原粒）。
