---
title: "CUDA 编程题"
description: "手写 CUDA kernel 面试题全集解答：归约/Softmax/GEMM/转置/直方图/前缀和/广播乘等，附可运行工程与 3090 实测数据"
pubDate: 2026-10-05
topic: "CUDA 基础"
order: 2
tags: ["CUDA", "高性能计算"]
---

覆盖题库：`AI-Infra-校招-1`（编程题 1-5）、`AI-Infra-综合面经题库-6`（CUDA 编程题 1-5）、`AI-Infra-综合面经题库-1`（CUDA 编程题 1-3）、`AI-Infra-面经-1`（编程题 1-5，OpenCL 版思路相同）。

**所有代码在 [solutions/cuda](/solutions/cuda) 工程**（CMake + sm_86），每题给出：思路 → 代码位置 → 验证命令 → RTX 3090 实测。

### 工程构建与验证

```bash
cd solutions/cuda
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build            # 6 个测试全部对拍通过
./build/test_gemm --bench         # 正确性 + 性能输出
```

## Q1. Reduction 归约求和（综合-6 编程 1、B站、腾讯、原粒、小厂）

**优化路径**（面试按此顺序讲）：

1. **原子加版**：每线程 `atomicAdd(out, x)`——串行化，1M 元素 3090 实测 **1.96 ms**。
2. **树归约**：block 内 shared memory 对折（每轮活跃线程减半，`__syncthreads()` 同步），每 block 产出 partial，再由第二个小 kernel 收尾。
3. **warp shuffle 收尾**：折到剩 64 个元素后，最后一个 warp 用 `__shfl_down_sync` 免同步归约。
4. 进一步：每线程先加载多个元素（网格跨步/ILP）减少同步轮数；本实现每线程取 2 元素。

**代码**：`solutions/cuda/src/reduce.cu`
**实测**：`./build/test_reduce --bench`

```
[bench reduce n=1048576] atomic 1.962 ms | two-stage(tree+warp shuffle) 0.016 ms | 有效带宽 265 GB/s | 加速比 124.0x
```

**数值精度追问**（原粒）：归约顺序不同结果不同（浮点不满足结合律），float 大规模求和误差可达 1e-3 相对量级；对拍时用 double CPU 参考值 + 相对容差（测试里就是这么做的）。

## Q2. 数值稳定 Softmax / Online Softmax（综合-6 编程 2、太初、海康、三星、飞腾）

三遍式（max → sum → 写回）合并为"一行一个 block"的两遍归约版：

```
pass1: 块内归约求 m = max(row)
pass2: 块内归约求 s = Σ exp(x - m)     ← 减 max 防 exp 溢出
pass3: out = exp(x - m) / s
```

数据驻留 shared memory，全局内存只读一遍。**Online softmax**（FA 的数学基础）：单遍流式维护 `(m, l)`，新块到来时 `l = l·exp(m_old - m_new) + Σ exp(x - m_new)`，历史分子重归一化。

**代码**：`solutions/cuda/src/softmax.cu`（两版 kernel）；PyTorch 逐步推导版 `solutions/python/src/answers/attention/online_softmax.py`
**实测**：`./build/test_softmax --bench`

```
[bench softmax 64x4096] rowblock 0.008 ms(有效带宽 259 GB/s) | online(1 warp/行演示) 0.945 ms
```

（数据量小时 launch 开销占比大；Triton 版 4096×4096 实测 0.163ms / 825GB/s，见 `solutions/python/tests/test_triton.py`。）

## Q3. GEMM 矩阵乘（校招-1 编程 1、综合-6、百度、快手、美团-北斗）

两级实现：

1. **naive**：每线程算 C 的一个元素，A 行 × B 列各扫 K。
2. **smem tiling**：16×16 tile，A/B tile 装入 shared memory 复用 16 次，全局访存降为 1/16。

**代码**：`solutions/cuda/src/gemm.cu`（Triton 对照版 `solutions/python/src/answers/triton_kernels/matmul.py`）
**实测**：

```
[bench gemm 1024x1024x1024] naive 0.992 ms (2165.6 GFLOPS) | smem-tiled 0.766 ms (2804.1 GFLOPS) | 加速比 1.3x
```

继续优化路线（面试加分）：寄存器 tile（每线程 4-8 个输出）、float4 向量装载、double buffering（装载与计算重叠）、warp 级 tile（mma/Tensor Core）、`__launch_bounds__` 控寄存器。评估指标：TFLOPS/SOL、每 SM 的 tile 数、L2 hit rate。详见《GEMM 与算子优化》篇。

## Q4. 矩阵转置（综合-6 编程 3、英伟达、小鹏）

- naive：读合并、**写不合并**（相邻线程写地址跨 `cols` 步长）。
- smem 版：`tile[32][33]` padding 消 bank conflict；装载/写出双向合并——注意二维 block 要 `dim3(32,32)` 启动（一维启动会得到全 0 结果，本工程实测踩过）。

**代码**：`solutions/cuda/src/transpose.cu`
**实测**：

```
[bench transpose 2048x2048] naive 0.061 ms (548 GB/s) | smem+padding 0.061 ms (547 GB/s)   ← 16MB 数据 L2 大量命中, 差距被压缩
4096x4096 实测: smem 601 GB/s vs naive 537 GB/s
```

**为什么有时加速不明显**：数据量小（16MB）时 3090 的 6MB L2 吸收了大量 uncoalesced 写；数据放大到 64MB（L2 放不下）后 smem 版优势显现。面试时给出"L2 命中影响测量"的分析本身就是加分项。

## Q5. Histogram 频数统计（校招-1 编程 4、蚂蚁）

两版（元素范围 0~255 正好 256 bin）：

1. **global atomic**：所有线程原子抢全局 256 个计数器。
2. **smem 私有化**：每 block 先在 shared memory 里建局部直方图，最后一次性原子合并到全局——冲突域从全 grid 缩到单 block。

**代码**：`solutions/cuda/src/histogram_scan.cu`
**实测**：

```
[bench histogram n=1048576] global-atomic 0.330 ms | smem-privatized 0.017 ms | 加速比 19.9x
```

## Q6. NCHW → NHWC 布局转换（校招-1 编程 5）

纯访存 kernel，关键在索引公式：`out[n][h][w][c] = in[n][c][h][w]`，即

```
src = ((n*C + c)*H + h)*W + w          // c 的 stride = H*W
dst = (((n*H + h)*W + w)*C + c)        // c 的 stride = 1
```

NHWC 让 c 维连续 → 卷积 im2col/向量化装载更友好。**代码**：`solutions/cuda/src/layout.cu`（在 `test_norm_layout` 中与 CPU 对拍通过）。

## Q7. 4D 广播 elementwise 乘 / (A*B)*C 链式广播（综合-1 CUDA 编程 1/2）

torch broadcast 的 CUDA 实现标准做法：**广播维 stride 置 0**，由输出线性下标逐维反解输入下标：

```cuda
// 输出 (on,oc,oh,ow), 输入 stride 为 sa_*/sb_* (广播维=0)
out[i] = a[n*sa_n + c*sa_c + h*sa_h + w*sa_w] * b[n*sb_n + ...];
```

语义与代码：`solutions/cuda/src/layout.cu`（kernel）、`solutions/python/src/answers/torch_impl/broadcasting.py`（`broadcast_shapes` + dummy tensor 的 `expand().stride()` 直接拿到广播 stride，对拍通过）。

**(A(1,256) · B(256,128) · C(128,256)) 的陷阱**：A 与 B 从右对齐后 256 vs 128 不兼容，直接乘报错；正确路径是 A·Bᵀ → (128,256) 再与 C 逐元素乘——考的就是"广播规则 + 形状判断"。

## Q8. Embedding Sparse Feature Pooling（综合-1 CUDA 编程 3）

A：100 万个 ID（0~999），B：100 万个 float，求 `C[i] = Σ B[j] where A[j]==i`。本质是 **segment sum / scatter-add**：

```cuda
__global__ void pool(const int*__restrict__ A, const float*__restrict__ B,
                     float* C, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) atomicAdd(&C[A[j]], B[j]);   // 版本1: 原子
}
```

优化版（同 histogram 思路）：先按 A 排序/分桶 → 每 block 处理一个 ID 段做无原子归约；或 warp 级聚合（lane 间先 shuffle 归约同 ID 再一次原子）。数值精度问题同 Q1。

## Q9. 平均池化 Avg Pooling（综合-6 编程 4）

每线程一个输出窗口，窗口内累加取平均：

```cuda
out[oy][ox] = (Σ_{ky,kx} in[oy*s+ky][ox*s+kx]) / (k*k)
```

**代码**：`solutions/cuda/src/pool_iou.cu`（`test_misc` 中对拍通过）。进阶：shared memory 缓存输入 tile 支持重叠窗口复用（stride < ksize 时）；global pooling 用归约。

## Q10. 两组 BBox 的 IoU（综合-6 编程 5）

```cuda
__device__ float iou(const float* a, const float* b) {
    float iw = max(0, min(a[2],b[2]) - max(a[0],b[0]));
    float ih = max(0, min(a[3],b[3]) - max(a[1],b[1]));
    float inter = iw * ih;
    float uni = area(a) + area(b) - inter;
    return uni > 0 ? inter / uni : 0;
}
```

M 个框两两 IoU 矩阵用 `(M+15)/16 × (M+15)/16` grid。**代码**：`solutions/cuda/src/pool_iou.cu`；NMS 的完整实现见《C++ 编程题》篇。**对拍**：IoU(box0, box0)=1、相离=0、(0,0,10,10)×(1,1,11,11)=81/119≈0.6807。

## Q11. 前缀和 Prefix Sum（美团-北斗）

块内 **Hillis-Steele** 扫描：log₂(n) 轮、每轮跨度翻倍、全并行：

```cuda
for (d = 1; d < blockDim.x; d <<= 1) {
    float add = (tid >= d) ? temp[tid-d] : 0;
    __syncthreads();
    temp[tid] += add;
    __syncthreads();
}
```

多 block 全局前缀和需要三步：①各块内扫描；②对每块末元素再做一次扫描得块偏移；③加回偏移（或 Blelloch 两遍 scan + chained kernel）。**代码**：`solutions/cuda/src/histogram_scan.cu`（块内版，`test_misc` 对拍）。

## Q12. 奇偶下标原地分区（校招-1 编程 5）

题：把数组中**奇数下标**的元素移到左侧、**偶数下标**的元素移到右侧。先讲清三种解法的取舍（面试加分点）：

**① 稳定版（O(n) 额外空间）**——gather 一次写出，最简单可靠：

```cpp
std::vector<int> out;
for (int i = 1; i < n; i += 2) out.push_back(a[i]);  // 奇下标在前
for (int i = 0; i < n; i += 2) out.push_back(a[i]);  // 偶下标在后
a = std::move(out);
```

**② 不稳定原地版（O(1) 空间）**——双指针交换：左指针找仍在左半区的偶来源元素、右指针找仍在右半区的奇来源元素，交换。实现上用"当前位置"判断归属：位置 `p < n_odd` 属于左半区。由于交换后来源标记丢失，工程上通常**按值域外的辅助位**或直接接受不稳定：

```cpp
int n_odd = n / 2;                    // 奇下标元素个数
// 若元素可从下标恢复(如 a[i]==i 的测试), 可直接判 p%2; 一般情况用辅助 bool 数组
```

**③ 严格稳定 + O(1) 空间**：交错序列的稳定原地分区无简单线性解，经典做法是分治块交换 O(n log n)（先递归处理两半，再交换内层错位块）。面试能指出"稳定原地分区本质是块交换/循环移位族问题，线性稳定需 O(n) 空间"即是亮点。

**GPU 版**：这就是纯 gather——每个输出线程反解源下标 `src = (i < n_odd) ? 2*i+1 : 2*(i-n_odd)`，一次写出（CUDA 版思路与 Q6 NCHW→NHWC 同构）。

## Q13. 行 Reduce Sum（沐曦）

一行一个 **warp**（而不是一个 block）：warp 内 shuffle 归约，免去 `__syncthreads()`，一个 block 挂 8 个 warp 同时处理 8 行，调度开销远小于"一行一 block"版本。**代码**：`solutions/cuda/src/pool_iou.cu` → `row_sum_warp_kernel`（`test_misc` 对拍通过）。

## Q14. 卷积 / Pooling / 计算图（面经-1 编程 1/3）

- **卷积**：直接卷积（C++ 版见 `solutions/cpp/src/vision.cpp` 的 conv2d + 手算对拍测试）；GPU 工程版路线 im2col + GEMM（reshape 成矩阵乘复用高度优化 GEMM）/ Winograd（3×3, stride=1 FLOPs 降 2.25x）/ FFT（大 kernel）。卷积时间复杂度 $O(H \cdot W \cdot C_{in} \cdot C_{out} \cdot K^2)$（百度-校招常考）。
- **Pooling**：见 Q9，max pool 把累加换成 `fmaxf`。
- **计算图**：节点=算子、边=张量依赖；面试手写一般是拓扑排序 + 前向遍历执行（见《算法题》篇拓扑排序），反向自动微分链式法则按拓扑逆序。

## Q15. OpenCL 矩阵乘与向量求和（面经-1 编程 5）

OpenCL 与 CUDA 概念映射：host/context ↔ runtime、command queue ↔ stream、work-item ↔ thread、work-group ↔ block、local memory ↔ shared memory、NDRange ↔ grid/block 维度。kernel 语言差异：`get_global_id(0)` ↔ `blockIdx.x*blockDim.x+threadIdx.x`。矩阵乘/向量求和的实现思路与本文 Q3/Q1 完全一致，只是索引 API 换掉。OpenCL 注意：减少分支与掩码（warp 式 SIMD 单元对分支按掩码串行）、`__local` barrier 用 `barrier(CLK_LOCAL_MEM_FENCE)`。
