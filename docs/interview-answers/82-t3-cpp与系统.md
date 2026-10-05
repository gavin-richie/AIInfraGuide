---
title: "T3 芯片厂题解（三）：C++ 与系统"
description: "太初 C++ 内存管理/智能指针/虚函数与 ML 基础、北极雄芯线程同步/DCL/malloc 底层、后摩梯度与量化（AdaQuant/I-ViT）、原粒 reduce 精度与 pytest"
pubDate: 2026-10-05
topic: "公司题解-T3"
order: 82
tags: ["面试解答", "C++", "编译器", "量化", "系统"]
draft: false
---

覆盖题库：`太初-AI-Infra-一面`、`太初-AI-Infra-实习-一面-1`、`太初-AI-Infra-实习-一面-2`、`北极雄芯-AI-Infra-一面`、`后摩智能-AI-Infra-实习`、`原粒半导体-AI-Infra-一面` 全部题目。C++ 基础与《[09 C++ 与系统基础](./09-cpp与系统基础.md)》《[10 C++ 编程题](./10-cpp编程题.md)》重叠的部分以引用为主，本篇补充追问深度。

# 太初（一面：C++ 与 GPU）

## Q1. C++ 中的内存管理机制有哪些

**（太初 一面 1）**

**静态分区**（《[09 篇](./09-cpp与系统基础.md)》Q7）：代码段 .text、只读 .rodata、.data（已初始化全局/静态）、.bss（未初始化）、堆（malloc/new 向上增长）、栈（局部自动变量向下增长）、mmap 映射区（动态库/大块分配）。

**管理机制分四层答**：

1. **栈**：编译器自动管理（函数帧移动 SP），快但小（默认 1-8MB）、生命周期严格绑定作用域；
2. **堆**：手动 `malloc/free`、`new/delete`（底层 ptmalloc 的 brk/mmap + bins，详见本篇北极雄芯 Q6）；
3. **RAII / 智能指针**（C++ 核心）：资源获取即初始化，析构函数自动释放——`unique_ptr`（独占零开销）、`shared_ptr`（引用计数控制块）、`weak_ptr`（打破循环引用）；锁的 `lock_guard`、文件的 fstream 同理；
4. **池化与自定义分配器**：内存池（预分配大块自切分，避免频繁系统调用与碎片，`solutions/cpp/include/answers/memory_pool.hpp`）、`std::allocator`/pmr（容器级策略）、GPU 侧 `cudaMalloc` 走自己的分配器（频繁小分配同样要池化，PyTorch 的 CachingAllocator 即此思想）。

**检测工具**：valgrind memcheck、AddressSanitizer（编译期 `-fsanitize=address`，更快）、`_CrtDumpMemoryLeaks`（Windows）。泄漏/越界/双重释放三大问题。

## Q2. 共享内存的概念及使用场景

**（太初 一面 2）**

按两个语境分别答（面试先确认或都答）：

**① CUDA shared memory**（《[01 篇](./01-cuda-基础与体系结构.md)》Q2/Q7）：SM 内片上 SRAM（与 L1 共享物理块），block 内线程共享、`__syncthreads()` 同步，~30 cycle 延迟。**场景**：数据分块复用（GEMM/conv tile 装载一次复用 16-32 次）、线程间交换（转置/scan/dot）、归约的中间站、私有化原子（histogram per-block 计数）。注意事项：容量-占用率折中、bank conflict（padding/swizzle）、>48KB 要 `cudaFuncSetAttribute` 显式申请。

**② 进程间共享内存（System V/POSIX IPC）**：`shmget`/`mmap(MAP_SHARED)` 让多个进程映射**同一物理内存**——**最快的 IPC**（无内核拷贝与上下文切换，对比管道/socket 的两次拷贝）。**场景**：高频进程间数据交换（推理服务与采集进程共享帧缓冲、数据库共享 buffer pool）。**坑**：自身不带同步，需配信号量/文件锁/原子变量/环形缓冲区设计；进程异常退出时内存不自动释放（`ipcs -m` 查看、`ipcrm` 清理）。

## Q3. 智能指针的实现原理；能否管理一段连续内存

**（太初 一面 3/4）**

**原理**：RAII + **引用计数控制块**。`shared_ptr` 构造时（`make_shared` 时对象与控制块合并一块分配）创建控制块：强计数（shared 数）+ 弱计数（weak 数）。拷贝 → 原子递增强计数；析构 → 原子递减，减到 0 析构对象；弱计数归 0 才释放控制块。计数操作是**原子的**（线程安全），但**指向的对象本身不保证线程安全**；`enable_shared_from_this` 解决"对象内部拿到自身 shared_ptr"。`unique_ptr` 无计数、零开销、编译期禁止拷贝（只 move）。手写实现（考点级代码）见《[10 篇](./10-cpp编程题.md)》Q9。

**管理连续内存**：

```cpp
std::unique_ptr<float[]> p(new float[1024]);           // ① 数组特化, delete[] 正确
auto q = std::make_unique<float[]>(1024);              // ② 推荐, 值初始化
std::shared_ptr<float> r(new float[1024],               // ③ shared 需自定义 deleter
                         std::default_delete<float[]>());
// CUDA 显存同理: 自定义 deleter 调 cudaFree
struct CudaDeleter { void operator()(float* p) const { cudaFree(p); } };
std::unique_ptr<float, CudaDeleter> dev(nullptr);
```

工程化对拍（数组特化 / `shared_ptr<float[]>` + 自定义 deleter 调用计数）：`solutions/cpp/tests/test_language_basics.cpp`。

要点：`shared_ptr` 无数组特化（`shared_ptr<T[]>` 到 C++17 才进标准且无 `operator[]`），必须给 deleter，否则 `delete p` 对数组是 UB；工程上连续内存更推荐 `std::vector`/`std::array`（还有 size），智能指针管裸内存多用于对接 C/驱动接口。

## Q4. C++ 多态的实现方式；虚函数底层机制；构造函数中调用虚函数的行为

**（太初 一面 5/6/7）**

**多态分类**：静态（编译期）——函数重载、模板、CRTP（奇递归模板，见《[09 篇](./09-cpp与系统基础.md)》Q9）；动态（运行期）——虚函数。

**虚函数底层**：每个**含虚函数的类**有一张 vtable（函数指针数组 + RTTI/type_info 指针），每个对象前部藏一个 vptr 指向所属类的 vtable。`base->f()` 编译成两步间接：`(*(base->vptr)[idx_of_f])(base)`。代价：一次额外内存读取 + 阻止内联；多重继承/虚继承下有多个 vptr 与 thunk 调整 this（飞腾二面同题，见《[83 篇](./83-t3-飞腾与中兴.md)》飞腾 Q4）。

**构造函数中调用虚函数**：**静态绑定到当前正在构造的类**——基类构造期间 vptr 指向**基类** vtable，派生类的 override 还不存在（派生类部分尚未初始化），所以调用的是基类版本。不是 UB，但是"违反直觉的行为"，设计上应禁止（规范里要求构造函数内不得调用（纯）虚函数——若是**纯**虚函数且无定义，直接链接错/纯虚调用崩溃）。同理析构函数中调虚函数绑定到当前正在析构的类。**想要"构造期多态"**：两阶段初始化（构造后显式 init）或工厂后处理。对比记忆：三星考"构造函数**能否声明为**虚函数"（不能，机制+语义双理由，见《[80 篇](./80-t3-英伟达与三星.md)》Q22），太初考"构造中**调用**虚函数"（行为层面）。

## Q5. GPU 硬件架构 / Reduce 优化策略 / GEMM 优化方法

**（太初 一面 8/9/10）**

**GPU 架构**：答"CPU vs GPU 设计哲学 + SM 结构"——《[01 篇](./01-cuda-基础与体系结构.md)》Q3/Q4：CPU 用复杂换延迟（大缓存/分支预测/乱序），GPU 用数量换吞吐（海量 ALU、TLP 隐藏延迟）；SM = 车间（warp 调度器 + 寄存器堆 + smem/L1 + SP/SFU），SP = 机床；3090：82 SM × 128 FP32 core × 1.7GHz ≈ 35.6 TFLOPS。

**Reduce 优化**（完整版《[02 篇](./02-cuda-编程题.md)》Q1 + 3090 实测 124×）：原子加（1.96ms，串行化）→ block 内树归约（smem 对折 + `__syncthreads()`）→ 折到 32 线程内换 `__shfl_down_sync` 免同步 → 每线程多元素（ILP/grid-stride）→ 两级 kernel（block partial + 收尾）。加上数值稳定性（树形归约误差 O(log n)，见本篇原粒 Q2）。

**GEMM 优化**（《[03 篇](./03-gemm与算子优化.md)》Q3）：tiling（smem 复用）→ 寄存器 tile（每线程 8×8 累加器，本篇 [81 篇](./81-t3-国产gpu厂商.md) Q1 微内核）→ double buffer → float4 → Tensor Core mma → split-K（小 M/N）→ autotune。本仓库 naive 2166 → smem-tiled 2804 GFLOPS 实测在 `solutions/cuda/src/gemm.cu`。

# 太初（实习 一面-1：训练与推理）

## Q6. Transformer 的整体结构 / BERT 与 GPT 的架构差异

**（太初 实习-1 2/3）**

**结构**（decoder-only 主流形态，可跑实现见 `answers/t3/minimal_transformer.py` 与《[80 篇](./80-t3-英伟达与三星.md)》沐曦 Q19）：`token embedding + 位置编码 → L × [RMSNorm → 因果 MHA → 残差, RMSNorm → FFN(SwiGLU) → 残差] → RMSNorm → LM head`。关键细节：Pre-Norm（训练稳定）、GQA（KV 头减少）、RoPE（相对位置，加在 Q/K 上）。

**BERT vs GPT**：

| | BERT | GPT |
|---|---|---|
| 骨架 | Transformer **Encoder** 堆叠 | **Decoder** 堆叠（无 cross-attention） |
| 注意力 | **双向**（互相可见） | **因果掩码**（只看左侧） |
| 预训练 | **MLM**（mask 15% 完形填空）+ NSP | 自回归 **next-token prediction** |
| 位置编码 | 可学习绝对 | 可学习绝对（GPT-1/2）→ RoPE 等相对（现代） |
| 归一化 | Post-LN | Pre-LN → RMSNorm |
| 擅长 | 理解类（分类/NER/检索句向量） | 生成类（对话/代码/few-shot） |

一句话：**编码器双向理解，解码器因果生成**；BERT 训练时 [MASK] 造成预训练-微调不一致（GPT 无此问题）。

## Q7. 数据并行的原理与实现方式

**（太初 实习-1 4）**

每卡持**完整模型副本**，各卡吃不同 micro-batch 独立前向反向 → 梯度 **AllReduce**（ring：2(N-1)/N 倍参数量通信）同步 → 各卡以相同梯度更新。实现：PyTorch DDP（bucket 化梯度 + 反向计算重叠通信、`no_sync` 支持梯度累积）；**数据并行不切显存**（单卡装不下就没法纯 DP）——切显存用 ZeRO-1/2/3（把优化器状态/梯度/参数也分片，是"数据并行的显存优化变体"，完整对照表见《[07 篇](./07-分布式训练.md)》Q3，中兴"三大并行"题见《[83 篇](./83-t3-飞腾与中兴.md)》中兴 Q2）。

## Q8. FasterTransformer 框架的核心特性

**（太初 实习-1 5）**

NVIDIA 出品的**纯 C++/CUDA 推理引擎**（后被 TensorRT-LLM 取代）：①**高度优化的 kernel**——fused MHA（一个 kernel 完成 QKV 变换+attention）、fused FFN、half/bf16 + Tensor Core；②**自研 KV Cache 管理**（CacheManager 按层按头组织，支持 beam search）；③**算子融合 + 算法特化**（自回归解码走专用路径、INT8/fp8 量化 kernel）；④C++ 前端消 Python 开销，提供 **Triton Inference Server** 后端（动态 batching/多实例）；⑤MPI 多卡张量并行。局限：图固定、动态 shape 支持弱、无 continuous batching（对比 vLLM，见《[05 篇](./05-推理引擎与优化.md)》Q2）。

## Q9. 常见优化器及特点

**（太初 实习-1 6）**

完整版见《[11 深度学习与训练](./11-深度学习与训练.md)》Q14（该题目标注太初）：**SGD+momentum**（v=μv+g，泛化好、需调 lr）；**Adam**（一阶 m/二阶 v 自适应 + bias correction，lr 均摊好调）；**AdamW**（解耦 weight decay——Adam 的 L2 正则与自适应更新冲突）；**Adafactor**（省显存：v 的低秩分解，大模型预训练）；**Lion**（符号更新 + 大 lr）；**Muon**（动量矩阵 Newton-Schulz 正交化，大 batch 预训练吞吐/稳定兼得）。加学习率调度：warmup + cosine 衰减。

## Q10. 随机森林与 GBDT

**（太初 实习-1 7/8）**

**随机森林**：bagging + 决策树——bootstrap 采样 + 每次分裂**随机特征子集**（去相关），多树投票/平均。降低**方差**、可并行、OOB（袋外样本）自带验证、给特征重要性。**GBDT**：boosting——串行地每轮拟合上一轮的**负梯度**（回归树为基学习器），shrinkage（小学习率）+ 子采样抗过拟合，降低**偏差**。对比：RF 并行/稳/不易过拟合，GBDT 精度上限更高/需精调。**XGBoost**：GBDT + 二阶泰勒 + 显式正则 + 列采样/缺失值方向学习；**LightGBM**：直方图分裂（连续值分桶）+ leaf-wise 生长（更深、更快）+ GOSS/EFB。

## Q11. 线上服务推理如何提升吞吐量

**（太初 实习-1 9）**

分场景——**LLM 服务**（主答案，《[05 篇](./05-推理引擎与优化.md)》Q12 综述）：系统层 continuous batching（2-20×）、chunked prefill/PD 分离、prefix cache 复用；执行层 CUDA Graph 消 launch、多 batch 分档；kernel 层 FA、量化 kernel（W4A16 显存 ÷4）；算法层投机解码（2-3×）、蒸馏小模型。**传统模型服务**：批处理合并请求（凑 GPU 饱和 batch）、线程池 + 异步推理队列、模型量化/蒸馏、缓存热点结果、多实例水平扩展 + 负载均衡、模型编译（TensorRT/torch.compile）。

## Q12. 链表表示的两数相加

**（太初 实习-1 10）**

LeetCode 2：两个**逆序**存数的链表，逐位相加进位（`carry = s // 10`），哑头简化头插。O(max(m,n))。已有实现与测试：`algorithms/linked_list.py::add_two_numbers`（`tests/test_algorithms.py`）。变体追问：正序存储（先反转或用栈）、两数相乘（`dp.py::string_multiply`）。

# 太初（实习 一面-2）

## Q13. 手写 softmax 并并行化优化

**（太初 实习-2 2）**

串行三步（max → Σexp(x−m) → 归一化），并行化路线（面试按层次讲，全部可跑）：

1. **行级并行（CUDA）**：一行一个 block，块内两遍归约（max、sum），数据驻留 smem——《[02 篇](./02-cuda-编程题.md)》Q2，3090 实测 64×4096 只需 0.008ms；
2. **Online softmax**：单遍流式维护 (m, l)，历史部分和重缩放——FA 的数学基础，PyTorch 推导版 `answers/attention/online_softmax.py`；
3. **Triton 版**：一行一个 program，`tl.load` 掩码装载 + `tl.max/tl.sum`，4096² 实测 **0.16ms / 825GB/s**（`answers/triton_kernels/softmax.py`，`pytest tests/test_triton.py`）。

数值稳定（减 max）是必答第一步；追问融合：softmax + dropout + scale 一个 kernel、或 log-softmax 用 logsumexp。

```bash
cd solutions/python && pytest tests/test_triton.py tests/test_attention.py -q
```

# 北极雄芯（一面）

## Q14. 对 AI 编译器的了解

**（北极雄芯 1）**

两层答：**图编译器**（框架级）——图捕获（torch.export/ONNX）、算子融合、常量折叠、布局/重排、调度（见《[08 篇](./08-编译器与性能分析.md)》Q6 的图优化开放题）；**kernel 编译器**（张量编译器）——TVM（计算-调度分离 + 自动搜索，Q1）、XLA（静态 shape HLO 全优化但动态性弱——燧原同族题见《[81 篇](./81-t3-国产gpu厂商.md)》Q29）、MLIR（多 dialect 渐进 lowering，芯片厂商定制后端的基建，Q28）、Triton（块级编程 + autotune）。趋势：手写热点算子库 + 编译器自动融合长尾的混合路线。

## Q15. 线程同步的常见方法

**（北极雄芯 2）**

**C++ 层**：`std::mutex`（+ `lock_guard/unique_lock` RAII）、`std::condition_variable`（等待-通知，配谓词循环防虚假唤醒——三线程打印题的基座，见《[80 篇](./80-t3-英伟达与三星.md)》Q16）、`std::semaphore`/`std::latch`/`std::barrier`（C++20）、读写锁 `shared_mutex`、`std::once_flag + call_once`、`std::atomic`（无锁计数/标志，memory_order 控制）、`std::future/promise`（一次性的结果同步）。**跨进程**：POSIX 信号量、共享内存 + 原子/自旋（本篇太初 Q2）、文件锁。**CUDA 侧**（岗位相关加分）：`__syncthreads()`（block 内屏障，不能放分支）、`__syncwarp()`（Volta 后显式）、原子操作、`atomicThreadFence`/`__threadfence()`（内存序）、cooperative groups（grid 级同步）。死锁四条件与避免（锁排序、`std::scoped_lock` 多锁）见《[09 篇](./09-cpp与系统基础.md)》Q14。

## Q16. 单例模式的实现；DCL 的隐患；其他设计模式

**（北极雄芯 3/4/5）**

**单例**：首选 Meyers（C++11 局部静态线程安全，三行）；DCL 用于"控制实例化时机"的场景。完整代码见《[10 篇](./10-cpp编程题.md)》Q6 与 `solutions/cpp` 的两种实现。

**DCL 的隐患（经典深度题）**：

```cpp
static T* inst;
if (inst == nullptr) {                    // ① 第一次检查（无锁快路径）
    lock(mtx);
    if (inst == nullptr) {                // ② 第二次检查
        inst = new T();                   // ③ 危险！
    }
}
```

`new T()` 实际是三步：**分配内存 → 构造对象 → 把地址写给 inst**。编译器/CPU 可能重排为 **分配 → 写 inst → 构造**：线程 B 在 ① 处看到 inst 非空直接使用，但对象**尚未构造**——未定义行为。C++11 前 DCL 是公认不可移植的（Java 靠 volatile 语义修复、C 靠原子）。**修复**：`std::atomic<T*>` + `memory_order_acquire`（读）/`release`（写）建立 happens-before；或直接 `std::call_once`/Meyers 单例（编译器保证静态初始化的线程安全与顺序）。这题的底层是 C++ 内存模型（happens-before、acquire/release，《09》Q7）。

**其他设计模式**（各一句场景）：**工厂/抽象工厂**（按类型造对象，配合注册宏解耦——三星算子注册题正是它，见《[80 篇](./80-t3-英伟达与三星.md)》Q28）；**观察者**（事件订阅，回调通知）；**策略**（算法族运行期可换，kernel 选择策略）；**适配器**（接口转换，C 驱动包装成 RAII）；**装饰器**（动态叠加职责，Python 里的 @decorator 同名不同物）；**享元**（权重共享）。

## Q17. malloc 的底层实现

**（北极雄芯 6）**

glibc **ptmalloc**（详细版见《[81 篇](./81-t3-国产gpu厂商.md)》沐曦 Q17，此处提纲）：**两个来源**——小块走堆（`brk`/`sbrk` 扩展 top chunk），大块（≥128KB 可调）直接 `mmap`（free 即 `munmap` 归还 OS，避免堆顶永久膨胀）。**空闲管理 bins**：tcache（每线程无锁缓存，64 个尺寸类）→ fastbin（≤160B 单向链 LIFO，不合并）→ small bin/large bin（按尺寸分桶的 双向链）→ unsorted bin（新释放块先入，分配时遍历尝试**合并相邻空闲块**防碎片）。**块结构**：每块头部 prev_size/size 元数据 + 标志位（PREV_INUSE），所以 malloc(n) 实际占 n+16 对齐到 16——**越界写踩坏下一块的 size 字段 = 堆损坏延迟崩溃**（最难查的 bug 类，用 ASan 提前暴露）。**多线程**：arena 机制（主 arena + 每线程最多 8N 个次 arena）减少锁竞争。`malloc_trim()`/`M_TRIM_THRESHOLD` 可调归还策略——长稳服务显存/内存上涨排查的常考点。

## Q18. NVIDIA Profiler 工具的使用；PyTorch 底层；CUDA 算子优化手段

**（北极雄芯 7/8/9）**

**Profiler**：分工记住三件套——**Nsight Systems**（系统时间线：kernel 时序/空隙 gap、memcpy 占比、多流重叠、CPU launch 延迟 → 回答"该优化谁"）；**Nsight Compute**（单 kernel：SOL SM%/Mem%、Occupancy、Warp Stall 原因、L1/L2 命中、指令/寄存器统计 → 回答"它慢在哪"）；旧 `nvprof` 已废弃，Python 侧 `torch.profiler` 是 nsys 的封装。判读经验：SM% 与 Mem% 谁贴近 100 谁是瓶颈；都低 = 并行度不足。完整版《[08 篇](./08-编译器与性能分析.md)》Q5、《[03 篇](./03-gemm与算子优化.md)》Q4。

**PyTorch 底层**：分层——**ATen**（算子库，C++ 实现，`add/mm/conv` 调 cuBLAS/cudnn）→ **dispatcher**（按 dispatch key 路由 CPU/CUDA/Autograd/...，自定义算子注册点，见《[80 篇](./80-t3-英伟达与三星.md)》Q28）→ **autograd**（前向记录计算图，backward 沿拓扑逆序链式求导）→ **Tensor**（Storage + stride 的视图分离，`view` 免拷贝、`reshape` 可能拷贝）→ c10 基础库。eager 逐算子执行（python 开销大）→ `torch.compile`（Inductor：图捕获 + 融合 + Triton codegen）。**CUDA 算子优化手段**：引用《[01 篇](./01-cuda-基础与体系结构.md)》Q8 七步 checklist。

## Q19. Linux 常用命令

**（北极雄芯 10、原粒/中兴同族）**

分组清单见《[80 篇](./80-t3-英伟达与三星.md)》Q27：文件/查看/进程/资源/网络/权限 + GPU 岗位专属（`nvidia-smi`、`nsys/ncu`）。能讲 2-3 个"组合技"案例（如 `find . -name "*.cu" | xargs grep -l "atomicAdd"`、`tail -f` + `grep --line-buffered`、管道 + `awk` 提取列）比报清单更显熟练。

# 后摩智能（实习）

## Q20. 梯度下降的基本原理；链式法则求导

**（后摩 1）**

**梯度下降**：损失 L(θ) 沿负梯度方向下降最快（泰勒一阶展开的推论），迭代 `θ ← θ − η·∇L`。批量/随机/小批量（SGD，深度学习标配：噪声有助泛化 + 可并行）。**链式法则**：复合函数 `dy/dx = dy/du · du/dx`——**反向传播就是链式法则在计算图上的动态规划**：前向记录每个算子的局部梯度，反向沿拓扑逆序逐层相乘；乘法路径共享（一个中间量被多条路径用 → 梯度累加），这就是"复用中间结果让总代价与节点数线性"的来源。例：`y=f(g(x))`，`∂L/∂x = ∂L/∂y · f'(g) · g'(x)`。

## Q21. ReLU 在零点处的导数

**（后摩 2）**

数学上**不存在**（左右导数 0 与 1 不等）。实践的约定：取 **0**（主流框架如此）——称为**次梯度（subgradient）**，凸分析意义下 0 与 1 之间任取都合法。为什么不影响训练：输入恰好等于浮点 0 的概率是零测度事件；工程反而要防的是**dead ReLU**（大梯度把大片神经元推到恒 ≤0 → 换 LeakyReLU/GELU 或降 lr）。

## Q22. Softmax 的实现与数值溢出

**（后摩 3）**

见《[02 篇](./02-cuda-编程题.md)》Q2 与本篇太初 Q13（同一考点三星/飞腾/后摩都考）：**减 max** 后再做 exp（fp16 下 `exp(x>11)` 直接 inf，不减必炸）；分母防 0（至少一个 exp(0)=1 保证非零）；underflow 的负端属正常。GPU 并行版 3090 实测、Triton 825GB/s 版全在仓库。

## Q23. AdaQuant 的核心思想

**（后摩 4）**

**AdaQuant**（Nagel et al., 2021，后训练量化）：逐层把**量化 scale 与一个小的权重扰动 ΔW 联合优化**——冻结 round 行为（STE），最小化**逐层输出重构误差** `‖W x − Q(W+ΔW, s) x‖`，允许量化后的权重略微"漂移"来补偿 round 误差；采用**逐层（后扩展 block 级）交替优化**：先固定 scale 优化 ΔW，再更新 scale。与近亲的区别：**GPTQ** 是基于 Hessian 的贪心逐列量化 + 误差补偿（同样逐层重构，但无显式 ΔW 自由度）；**bias correction**（AdaQuant 论文同作者前作）只校正输出均值偏移。效果：W4 掉点显著小于纯 min-max PTQ，成本远低于 QAT。定位在《[06 篇](./06-量化.md)》Q3 的 GPTQ/AWQ 对比表中补一行即可。

## Q24. I-ViT 的技术要点

**（后摩 5）**

**I-ViT**（Integer-only ViT, 2023）：全整数推理的 ViT——连 softmax/GELU/LayerNorm 里的**除法、指数、开方**都不落浮点，端侧芯片无 FPU/低精度浮点弱时也能跑：

1. **Shift-max / Shift-GELU**：用**2 的幂次近似除法**（除以 2^k 退化为移位），softmax 的分母归一化改为定点标度 + 移位；
2. **ISE（Integer Square-root Emulation）**：LayerNorm 的 `1/√var` 用整数迭代（牛顿法/逐位逼近）近似平方根倒数；
3. 全程 INT8 矩乘（逐层 scale 传递：tensor 的 scale 链在量化图里静态推导），非线性算子用**定点查表/分段多项式**；
4. 训练时 STE 对齐整数语义，精度接近 W8A8 基线（ImageNet 掉点 <1%）。

答题落点：**"非线性算子是全整数量化的拦路虎，I-ViT 用移位+迭代近似+查表把它们整数化"**；同类考点：非线性算子量化的 LUT 方案（原粒 Q4）、全整数的 softmax（商汤/端侧厂高频）。

# 原粒半导体（一面）

## Q25. CUDA kernel 常见优化方法

**（原粒 1）**

引用《[01 篇](./01-cuda-基础与体系结构.md)》Q8：测量（ncu SOL）→ 访存（合并/float4/smem tiling）→ 计算（FMA/intrinsics/unroll/ILP）→ 并行度（block/grid、grid-stride）→ 执行效率（divergence、bank conflict）→ 调度（Graph/多流）→ 算法层。配实测弹药：本仓库 reduce 124×、histogram 19.9× 的优化记录（《02 篇》Q1/Q5）。

## Q26. Reduce 计算顺序不同是否影响数值精度

**（原粒 2）**

**影响**。浮点加法**不满足结合律**：`(a+b)+c ≠ a+(b+c)`（每步都舍入）。典型灾难：大数吃小数——`(1e8 + 1) + (-1e8) = 0` 但 `1e8 + (1 + (-1e8)) = 1`；1M 个 float 顺序求和误差可达 1e-3 相对量级。**误差规律**：顺序累加误差 O(n·ε)，**树形/成对归约 O(log n·ε)**——GPU 的树归约天然比 CPU 顺序循环准；Kahan 补偿求和（维护误差项）可达接近 double 精度。**对拍实践**（本仓库做法）：CPU double 算参考值，断言相对误差 `|x−ref|/|ref| < tol` 而非绝对相等——`solutions/cuda` 每个 reduce 测试都如此；`answers/t3/` 的量化模块同样用 RMSE 而不是逐位相等。

## Q27. 如何验证和确定计算精度

**（原粒 3）**

分四步：①**确定参考**——更高精度（double/long double）或标准库/官方 kernel 做基准；②**定义误差度量与容差**——相对误差 + 绝对误差联合（`assert_close(rtol=1e-5, atol=1e-8)`，绝对容差保护接近 0 的值）、逐元素统计最大/分位误差而不只看均值；③**控制变量**——关闭编译器激进优化对数值的影响（`-ffast-math`、TF32 开关、`--use_fast_math` 的 `__expf` 与 `expf` 差异，见《[01 篇](./01-cuda-基础与体系结构.md)》Q16 SFU）；④**分级验收**——单元级（kernel 对拍）→ 逐层级（激活 cos-sim/MSE）→ 任务级（ppl/下游指标，见本篇遂原 Q23 同款、《[06 篇](./06-量化.md)》Q7）。进阶：ulp（unit in the last place）衡量舍入是否在 1-2 ulp 内。

## Q28. 常见的非线性算子

**（原粒 4）**

**激活类**：ReLU（分段线性）、GELU（x·Φ(x)，高斯误差或 tanh 近似）、SiLU/Swish（x·sigmoid(x)）、Sigmoid/Tanh（指数族）；**归一化后的非线性**：softmax（exp+归一化）、log-softmax；**损失**：cross-entropy（log）；**其它**：mish、PReLU。**GPU 实现**：ReLU/SiLU 直接算（`__expf` 走 SFU，吞吐 1/32 于 FP32——软化函数是真实成本，《01》Q16）；**量化后的非线性算子**：定点域不便直接算 exp/tanh → **查表 LUT**（int8 输入 256 桶查表）或分段多项式，与 I-ViT 的移位近似同族（本篇后摩 Q24）。

## Q29. pytest 的常用命令与功能

**（原粒 5）**

常用命令：`pytest -q`（简洁）、`-v`（逐用例）、`-k "pattern"`（按名过滤）、`-x`（首错即停）、`--tb=short/long`（回溯长度）、`-s`（显示 print）、`--maxfail=3`、`-m mark`（标记过滤）、`--durations=10`（最慢用例）、`-n auto`（xdist 并行）、`--lf`（只跑上次失败）。**功能**：`@pytest.fixture`（依赖注入 + `yield` 做 teardown）、`@pytest.mark.parametrize`（数据驱动，本仓库 `test_t3.py` 的时钟夹角就用它）、`assert` 重写给出富比较信息、`pytest.approx`（浮点容差）、`conftest.py` 共享 fixture、mark 的 `skip/xfail`。本仓库即范例：`solutions/python/tests/`，`pytest tests -q` 全绿。

## Q30. 项目经历介绍

**（原粒 6、遂原 1/2、摩尔线程 1、飞腾 9、中兴 1 等所有"项目深入提问"同框架）**

统一框架（STAR 变体）：一句话定位 → 规模指标 → 我的角色 → 技术难点 1-2 个（讲**决策与权衡**：为什么这样选、对比过什么、量化收益多少）→ 结果与反思。追问防线：每个数字能给出处（怎么测的）、每个优化能讲三层（现象 → 根因 → 方案 → 验证）。可参考《[80 篇](./80-t3-英伟达与三星.md)》Q12 的英文版与中文版框架。

---

**下一篇**：[83 T3 芯片厂题解（四）飞腾与中兴](./83-t3-飞腾与中兴.md)（算子实战/推理优化/分布式训练）。
