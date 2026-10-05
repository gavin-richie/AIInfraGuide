---
title: "百度 · 系统与 C++"
description: "百度 AI Infra 面试题解·系统与 C++ 篇：OpenMP 竞态、Cache 映射、C++ 全家桶（智能指针/多态/map/deque）、Go GMP 与反射、TCP/UDP/sendfile、K8s Informer/Pod 创建、etcd vs Redis、Kafka 顺序消费、cgroup/namespace"
pubDate: 2026-10-05
topic: "公司题解-百度"
order: 52
tags: ["面试解答", "百度", "C++", "操作系统", "Go", "云原生"]
draft: false
---

覆盖题库（来源缩写）：`一4`=百度-AI-Infra-一面-4、`二1`=二面-1、`三`=三面——**三个文件是同一场面试的拆分稿，题目大量重复**（一4 的一/二/三面 = 二1 的一面 = 三的全卷），本篇按去重后的一次性题单解答，来源标注（一4·一面）等；`二3`=二面-3（Go/云原生）、`实2`=实习-2、`实一1`=实习-一面-1、`校`=校招、`校一2`=校招-一面-2、`校二2`=校招-二面-2。与已有篇目重复的题（智能指针、new/malloc、内存分区等）引用 [09 C++ 与系统基础](./09-cpp与系统基础.md) 并补百度追问。

## Q1. OpenMP 并行区如何开启；并行区 `sum++` 与串行结果是否一致（一4·一面 Q2/Q3、二1 Q1/Q2、三 Q1/Q2/Q3）

```c
#pragma omp parallel [num_threads(n) ...]   // 每个线程都执行区块
{
    // 并行区
}
#pragma omp parallel for                     // 循环迭代在线程间划分
for (int i = 0; i < N; i++) ...
```

`for(int i=0;i<100;i++){sum++;}` 并行执行**结果不一致**：`sum++` 是读-改-写三步，多线程交错导致更新丢失（race condition），结果 ≤100 且不确定。保证正确性的手段：

1. **归约子句**：`#pragma omp parallel for reduction(+:sum)`——每线程私有累加，结束合并（推荐，无锁）；
2. **临界区**：`#pragma omp critical { sum++; }`——串行化，正确但慢；
3. **原子操作**：`#pragma omp atomic`（硬件原子指令，比 critical 轻）；
4. **手写锁**：`omp_set_lock/unset_lock` 或 std::mutex；
5. 每线程私有副本（`private(firstprivate)` 子句）+ 手动合并。

追问点：reduction 为什么快（无竞争无锁，合并只在末尾一次）；atomic 与 critical 的区别（原子=单条指令级、critical=代码块级互斥）。与 GPU 归约的"原子加 vs 树归约"同构（[02 CUDA 编程题](./02-cuda-编程题.md) Q1 的 124× 加速即同一思想）。

## Q2. GPU 内存层次结构：各级作用域（一4·一面 Q4、二1 Q3、三 Q1）

见 [01 CUDA 基础与体系结构](./01-cuda-基础与体系结构.md) Q2 全解。速答表：

| 层级 | 作用域 | 容量级(3090) | 延迟 |
|---|---|---|---|
| Register | 单线程 | 256KB/SM | ~1 cycle |
| Local Memory | 单线程（寄存器溢出/大数组） | 在显存里 | 慢（走 L1/L2） |
| Shared Memory / L1 | **同 block** | 128KB/SM（可配比） | ~30 cycle |
| L2 Cache | **全 GPU（所有 SM）** | 6MB | ~200 cycle |
| Global Memory | 全 GPU + host 可见 | 24GB | ~400+ cycle |

关键性质：shared memory 由程序员显式管理（`__shared__`，block 结束失效）；L2 跨 SM 一致；local memory 是"背锅侠"（寄存器不够时的溢出区，性能反直觉地差）。追问常考：shared 与 L1 同一物理 SRAM、按 `cudaFuncSetAttribute` 配比；constant/texture memory 的广播缓存特性。

## Q3. Cache 映射方式：直接映射 / 全相联 / 组相联（一4·一面 Q5、三 Q4）

- **直接映射**：主存块只能进唯一 cache 行（`行号 = 块地址 mod 行数`）——硬件最简、查找最快，但**冲突失效**严重（两个热块映射同一行互相踢）；
- **全相联**：任意块进任意行——无冲突失效、利用率最高，但**比较器成本指数级**（每行都要 tag 比对），只适合小容量（TLB）；
- **组相联**：折中——先按索引定组（直接映射），组内任意放（全相联）。$n$ 路组相联 = 每组 n 行；路数越大冲突越少但面积/功耗越高。主流 L1 4-8 路、L2/L3 8-16 路；
- 追加考点：**替换策略**（LRU/PLRU/随机）、**写策略**（write-back vs write-through、写分配）、伪共享与 cache line（64B）——与 GPU 的 bank conflict、warp 调度局部性类比着答加分。

## Q4. C++ 智能指针分类与场景（一4·一面 Q6、三 Q5、实2 Q4、实一1 Q8、校二2 Q3，百度全批次重复）

见 [09](./09-cpp与系统基础.md) Q3 全解（四种指针、线程安全、手写实现）。速答：`unique_ptr` 独占（move-only，零开销，默认首选）；`shared_ptr` 引用计数共享（控制块，原子计数）；`weak_ptr` 打破循环引用（不增计数，lock() 升级）；（auto_ptr 已废）。**百度特有追问**：

- **unique_ptr 如何保证唯一**（实一1 Q9）：拷贝构造与拷贝赋值 `=delete`，只留移动构造/赋值——所有权只能转移不能复制（`std::move` 后源指针置空）；c++14 的 `make_unique` 免裸 new；
- **shared_ptr 计数归零后何时析构**（实一1 Q10）：最后一个强引用释放时（`use_count` 0→析构管理对象）；若还有 weak_ptr，**对象析构、控制块保留**（weak 查计数/lock 需要），最后一个 weak_ptr 析构时才释放控制块——两次析构的细节是高频考点；
- **为什么 make_shared 好**：对象与控制块一次分配（缓存友好、少一次失败路径），且免裸 new 间隙泄漏。

## Q5. C++ 多态：概念、实现、重写 vs 重载（一4·一面 Q7、三 Q6、实2 Q8）

见 [09](./09-cpp与系统基础.md) Q8。多态 = 同一接口不同行为。**编译期**：重载（同名不同参，名字修饰决议）、模板；**运行期**：虚函数 + 继承（基类指针/引用调到派生类实现，vtable 间接跳转）。实2 Q8 追问 vtable 细节：每个多态类一张虚函数表（编译期生成），对象头部藏 `vptr` 指向它；调用 `p->f()` 编译为 `(*p->vptr)[f_slot](p)`——一次间接寻址 + 无法内联的成本；构造时 vptr 按构造顺序逐层设置（所以构造函数里调虚函数不多态）。**重写(override)**：派生类重定义基类 virtual（签名相同 + 协变返回，运行期决议）；**重载(overload)**：同作用域同名异参（编译期决议）——两者机制、时机、位置都不同。

## Q6. 动态链接 vs 动态绑定（一4·一面 Q8、三 Q7）

- **动态链接**：**装载/链接期**概念——程序不在编译期把库代码拷进来，运行时由动态链接器（ld.so）把 `.so` 映射进地址空间、完成符号重定位（GOT/PLT 惰性绑定）。好处：共享一份库内存、可独立升级、插件机制；
- **动态绑定**：**语言/运行期语义**概念——调用目标在运行时依对象实际类型决定（C++ 虚函数经 vtable；动态类型语言按对象方法表）；
- 区别一句话：前者是"代码/符号怎么组合进进程"，后者是"这条调用指令跳到哪个函数"。有交集但正交：静态链接的程序照样有虚函数动态绑定；动态链接的 C 函数调用是静态绑定（PLT 只是跳板，目标确定）。

## Q7. STL map 底层；AVL vs 红黑树（一4·二面 Q10/Q11、三 Q2）

- **map 底层**（一4 Q10）：红黑树（平衡二叉搜索树）——节点存 key/value/三色标记/三指针；有序遍历（中序）、增删查 $O(\log n)$ 最坏保证；迭代器失效规则：删除只失效被删节点，插入不失效（稳定引用）；
- **AVL vs 红黑树**（一4 Q11）：AVL 严格平衡（左右子树高差 ≤1）——查询更快（树高低），但插入/删除触发更多旋转（最多 $O(\log n)$ 次回溯旋转）；红黑树近似平衡（最长路径 ≤2×最短）——旋转次数摊还 O(1)（插入至多 2 旋转、删除至多 3），**写多读多用红黑树**。这就是 STL/Linux 内核选红黑树、数据库内存索引选 AVL/B+树的原因。追问可延伸：B/B+树为磁盘生（高扇区、矮树、节点=页），与红黑树的内存假设对比。

## Q8. unordered_map 底层与哈希冲突解决（一4·二面 Q12/Q13、三 Q3/Q4）

**底层**：开链法哈希表——`bucket_count` 个桶，每桶一个单链表（节点含 key/value/next）；负载因子 `size/bucket_count` 超过 `max_load_factor`（默认 1.0）触发 rehash（桶数组扩容到质数/2 倍附近并重新分布，**迭代器全部失效、指针引用仍有效**）。查找：hash(key) 定桶 → 桶内 `==` 精确比较。平均 $O(1)$、最坏 $O(n)$（同桶）。

**冲突解决法**（Q13）：①**开链/拉链法**（unordered_map 用；无装填上限、删除方便，但指针跳转 cache 不友好）；②**开放寻址**（线性/平方/双重探测：冲突就探下一个空位；cache 友好，但删除要墓碑标记、装填因子上去性能崩，需预留空间）；③**再哈希**（换哈希函数，少用）；④**公共溢出区**。追问点：定制 hash（字符串 FNV/murmur）、`reserve` 预分配避免 rehash、为什么桶数取质数（开链无要求，开放寻址法里模质数分散更均匀）。

## Q9. 计算机存储层次与 CPU 可见寄存器、PC（一4·二面 Q14/Q15、三 Q5/Q6）

- **存储层次**（金字塔）：寄存器（几十~几百，<1ns）→ L1/L2/L3 SRAM Cache（1-10ns，KB~MB）→ 主存 DRAM（~100ns，GB）→ SSD/磁盘（µs~ms）→ 网存/磁带。设计依据：**局部性**（时间+空间）+ 容量-速度-价格权衡，越上层越小越快越贵；
- **程序员可见寄存器分类**（Q15）：通用寄存器（rax/rbx…或 r0-r30）、指针类（SP 栈指针、BP 帧基址、PC/IP）、变址/基址（SI/DI）、标志寄存器（FLAGS/PSW）、浮点/SIMD（xmm/ymm/zmm）。严格说 PC/SP 对**系统程序员**（汇编/编译器）可见、对高级语言透明；
- **PC 的作用**：存下一条（x86 是当前指令的下一地址）待取指令的地址——顺序执行自增、分支/调用/中断时被改写；它是"取指-译码-执行"循环的驱动器。追问：分支预测就是猜 PC 的下一个值（见 Q12 CPU 优化题）。

## Q10. 常用操作系统命令（一4·二面 Q16、三 Q7）

按场景分组答（比背清单加分）：**进程/CPU**：`ps aux/top/htop/mpstat`、`nice/renice`、`lscpu`；**内存**：`free -h`、`vmstat`、`smem`；**IO/磁盘**：`iostat`、`df -h`、`du -sh`、`iotop`、`fio`（压测）、`dd`；**网络**：`ss -tlnp`/`netstat`、`ip`、`ping/traceroute`、`tcpdump`、`curl/wget`、`ethtool`；**文件**：`find/grep/awk/sed`、`lsof`（谁打开着文件/端口）、`stat`；**排障组合**：`dmesg`（内核日志）、`ltrace/strace`（系统调用跟踪）、`perf`（采样剖析）、`ncdu`；**GPU（Infra 岗必带）**：`nvidia-smi`（`dmon`/`pmon` 变体）、`nvtop`。答题技巧：报"定位什么问题用什么命令"的配对，如"进程吃满 CPU → top 找 pid → perf top 看热点函数"。

## Q11. main 之前运行函数的三种方法（一4·三面 Q17、三 Q1）

1. **全局/静态对象的构造函数**（最常见）：C++ 全局对象在 `main` 前由运行时初始化（`__libc_csu_init`/`.init_array` 驱动）；
2. **`__attribute__((constructor))`**（GCC/Clang）：`void before_main(void) __attribute__((constructor));`——链接器把它放进 `.init_array`，共享库加载时也执行（插件钩子常用）；
3. **平台段**：写 `.init_array`/链接器脚本、或 `#pragma startup`（MSVC 历史用法）。

追问：初始化顺序陷阱（跨编译单元全局对象初始化顺序未定 → 用函数内 static（Meyers 单例）惰性化，见 [10 C++ 编程题](./10-cpp编程题.md) Q6）；析构在 `main` 后（`.fini_array`）。链接/装载细节见 [09](./09-cpp与系统基础.md) Q17。

## Q12. 调试：条件断点与堆栈监视（一4·三面 Q18/Q19、三 Q2/Q3）

- **条件断点**（Q18）：gdb `break file:line if i == 100`（条件表达式真才停）；`ignore 1 99` 跳过前 99 次；硬件断点 vs 软件断点（int3 改写 vs 调试寄存器 DR0-3，后者可监听**数据写**=watchpoint：`watch var` / `watch -l *addr`）。IDE（VS/CLion/VSCode）在断点属性里填 Condition/Hit Count；
- **堆栈监视**（Q19）：`bt`/`bt full`（全帧带局部变量）、`frame n`+`info locals`、`info registers`、`x/16xw $sp`（查栈内存）；多线程 `thread apply all bt`（死锁排查标配）、`info threads`；运行时观察可用 `display expr`（每步自动打印）。延伸：coredump 事后分析（`gdb ./prog core`，见 [09](./09-cpp与系统基础.md) Q20 的崩溃排查全流程）、AddressSanitizer/Valgrind 抓内存错误。

## Q13. 程序内存分段布局（一4·三面 Q20、三 Q4、实2 Q3、校一2 Q2）

见 [09](./09-cpp与系统基础.md) Q7 全解。从低到高：**text 代码区**（只读可执行，常量字面量附近）→ **data 已初始化全局/静态** → **bss 未初始化全局/静态**（不占文件体积，加载时清零）→ **heap 堆**（brk/mmap 向上长，malloc/new 管理）→ 共享库映射区 → **stack 栈**（向下长，函数帧/局部变量，默认 8MB 可 `ulimit -s` 调）→ 命令行参数与环境变量。追问：栈为什么快（移动 SP 即分配、天然 cache 局部）；堆为什么慢（分配器查找/锁/碎片）；全局 static 与函数内 static 分别落在 data/bss（初始化否）但符号作用域不同；`.bss` 与 `.data` 的文件体积差异（`size a.out` 验证）。

## Q14. STL deque 底层实现（一4·三面 Q21、三 Q5）

**分段连续**：一个指针数组（map，中控区）指向多个固定大小的缓冲区（chunk，如 512B 或 8 个元素）；两端插入在头尾 chunk 内 O(1)，chunk 满则中控区扩一个指针（map 满则整体迁移）。迭代器四成员：cur（chunk 内位置）、first/last（chunk 边界）、node（中控区槽位）——跨 chunk 时迭代器自己跳，对用户呈现"伪连续"。vs vector：头插 O(1)（vector O(n)）、扩容不需整块连续内存（大对象更友好）、随机访问两级寻址略慢（`map[i/chunk][i%chunk]`）；vs list：支持随机访问、cache 更友好。使用场景：滑动窗口/双端队列/工作窃取任务队列。

## Q15. 寄存器 vs Cache 谁快；Cache 与主存的技术异同（一4·三面 Q22、三 Q6）

- 寄存器更快（<1ns、与 CPU 同频、编译器直接寻址命名访问；Cache 是 SRAM 额外一跳、靠地址映射查）；
- **Cache（SRAM）**：触发器存位（6T），不需刷新、快、贵、密度低——故只做小容量层级；
- **主存（DRAM）**：电容存位（1T1C），漏电需 ms 级刷新、慢一个量级、便宜、密度高——做 GB 级主存；
- 异同总结：都是易失存储；工艺（触发器 vs 电容+刷新）、延迟、成本、容量互补，靠层次结构拼出"又大又快"的幻觉；再往下 SSD 是 Flash（非易失、块擦写），又隔两三个量级——接 Q9 的金字塔。

## Q16. CUDA 内存分类；kernel 与 `__global__`；最大线程数（一4·三面 Q23-Q25、三 Q7-Q9）

- **内存分类**（Q23）：按作用域/生命周期——寄存器（线程私有、kernel 生命周期）、local memory（线程私有但落在显存，寄存器溢出）、shared memory（block 共享、kernel 生命周期）、global memory（全局持久，跨 kernel，host 可拷）、constant（只读广播，grid 共享）、texture/surface（空间局部缓存）、pinned host memory（锁页，DMA 高速拷贝）。详见 [01](./01-cuda-基础与体系结构.md) Q2/Q21；
- **kernel 与 `__global__`**（Q24）：kernel = 在 GPU 上执行、从 host（或 dynamic parallelism 从 device）启动的函数；`__global__` 声明它是 kernel（void 返回、只能被 `<<<>>>` 调用、不可递归（传统约束）、参数经 constant memory 传）；配套 `__device__`（仅 GPU 调）、`__host__`（CPU 编译）；
- **最大线程数**（Q25）：三层口径——`cudaOccupancyMaxPotentialBlockSize` 探测最优 block 尺寸；`cudaDeviceProp.maxThreadsPerBlock`（每 block 上限，通常 1024）；`maxThreadsPerMultiProcessor`（每 SM 并发上限，如 3090=1536）；grid 维度上限（x 维 $2^{31}-1$）。写法：先 `cudaGetDeviceProperties` 打印，或宏 `MAX_THREADS` 保守 1024——答出"三层级 + 怎么查"即满分。

## Q17. 贪心 vs 动态规划（一4·三面 Q26、三 Q10）

- **贪心**：每步取局部最优且**不回头**——要求问题有贪心选择性质（局部最优可推出全局最优）+ 最优子结构；效率高（常 $O(n\log n)$ 级），正确性需要证明（交换论证），反例：找零无合适币制时贪心错；
- **DP**：枚举所有子问题 + 记忆化/递推，**全局视角**——只需最优子结构 + 重叠子问题；正确性稳但复杂度高（状态×转移）；
- 区别一句话：贪心是"每步只算一条路"，DP 是"每步保留所有路再取最优"；贪心是 DP 的特例（能证明无后效性时剪掉其余分支）。判别法：举反例测试贪心（活动选择贪心对、0-1 背包贪心错——分数背包才对）；做题信号："最多/最少/最长"+约束强 → 试贪心并证明，约束互相牵制 → DP。

## Q18. 常见图搜索算法 BFS/DFS（一4·三面 Q27、三 Q11）

- **BFS**：队列逐层扩展，边权相等时最短路（层数=距离）；空间 $O(b^d)$；适合层序/最浅目标/社交圈；
- **DFS**：栈（递归）一路到底回溯，空间 $O(depth)$；适合路径存在性/拓扑排序/割点桥/回溯类枚举；
- **扩展**：带权最短路 Dijkstra（非负权，优先队列 $O((V+E)\log V)$）、Bellman-Ford（可负权，$O(VE)$）、Floyd（多源 $O(V^3)$）、A*（启发式加速 BFS/Dijkstra）、双向 BFS/IDS；连通性并查集；
- 工程题绑定：图遍历的迭代器写法、避免递归爆栈、网格题的坐标压缩。代码：`solutions/python/src/answers/algorithms/graphs.py` 与 `trees.py`（topo_sort_kahn/dfs、num_islands、level_order 等对拍实现，测试 `test_algorithms.py`）。

## Q19. Go 篇之一：Tag、反射、Slice/Array、Map（二3 Q1-Q4）

- **Tag 映射机制**（Q1）：struct 字段后 `` `json:"name,omitempty"` `` 的元字符串——`reflect.TypeOf(obj).Field(i).Tag` 运行期解析（`Tag.Get("json")`），编码库（encoding/json、orm）据此改字段名/跳过/内联；是"编译期存字符串、运行期反射消费"的元数据通道，零运行时开销（不用就没成本）；
- **反射原理与场景**（Q2）：`interface{}` 值 = (动态类型, 动态值) 二元组；reflect 包拿到 `Type/Value` 后可读字段、调方法、构造值（`reflect.New`）。实现依赖类型元数据（rtype/itab）。场景：json/gob 序列化、orm 映射、依赖注入、mock 框架、通用 deep-equal。代价：慢（绕过内联、大量分配）、编译期类型检查丢失——"能不用就不用，库代码才用"；
- **Slice vs Array 扩容**（Q3）：Array 是值类型（长度是类型的一部分，赋值整拷贝、不能扩容）；slice = (ptr, len, cap) 头的引用视图，`append` 超 cap 时重分配：旧版本翻倍，**1.18 后新规则**：<256 翻倍，之后约 1.25 倍渐进（且按 size class 取整）——追问 cap 变化要按版本答。扩容后旧数组不变（共享段可见旧数据），copy-on-append 陷阱由此来；
- **Map 底层与无序遍历**（Q4）：hmap 结构 + 桶数组（bmap，每桶 8 槽，溢出桶链）；hash(key) 定桶，桶内 tophash 快筛；扩容翻倍（或等量整理溢出桶），渐进式搬迁。**遍历无序是故意的**：起始桶随机化（防程序员依赖顺序），且扩容搬迁中键分布变化——Go 官方明确不保证顺序。**有序 map 实现**：单独维护 key 切片/有序结构（如 `keys` 收集后 sort，或用第三方 orderedmap：双向链表+索引 map），遍历走链表。补充高频：map 并发读写 panic（`fatal error: concurrent map writes`，不可 recover）→ `sync.Map` 或分片锁。

## Q20. GMP 调度模型详解（二3 Q5）

三角色：**G**（goroutine，含栈与状态）、**M**（OS 线程，真正跑代码）、**P**（逻辑处理器，持可运行 G 的本地队列，数量 = `GOMAXPROCS`，默认=核数）。

- **调度循环**：M 绑定 P 才能执行 G；本地队列（256）+ 全局队列；**work stealing**——本地空则先从全局取，再从其他 P 偷一半；
- **阻塞处理**：G 发起系统调用 → M 陷入内核，M 与 P 解绑（P 转给其他 M 继续跑队里 G）；网络 IO 走 netpoller（epoll）G 被挂起不占线程；
- **抢占**：1.13 前基本靠协作（函数序言检查栈/抢占点），1.14+ 基于信号的**异步抢占**（sysmon 检测超时 G 发 SIGURG）——纯计算死循环也能被调度走；
- **设计动机**：用户态调度开销 µs 级 vs 线程切换 µs+页表开销；M:N 复用少量线程跑海量 G；
- 追问常备：goroutine 栈初始 8KB（旧 2K）按需增长（拷贝式、连续栈）；`GOMAXPROCS` 与 cgroup CPU limit 的配合问题（容器里 P 过多导致 throttling）；对比其他语言绿色线程（Java 虚拟线程/协程）。

## Q21. TCP / UDP / HTTP / HTTPS 对比（二3 Q6）

- **TCP**：面向连接（三次握手）、可靠有序（ACK+重传+序号）、流量控制（滑动窗口）/拥塞控制（慢启动/拥塞避免/快恢复）、字节流、头部 20B+，开销大延迟高——文件传输/DB 连接/HTTP 底座；
- **UDP**：无连接不可靠、数据报文边界保留、头部 8B、一发即走——实时音视频、DNS、游戏、QUIC 在 UDP 上自建可靠性（HTTP/3）；
- **HTTP**：应用层请求-响应协议（无状态，靠 cookie/session 补状态；1.1 长连接/管线化弱、2 多路复用+头部压缩、3 走 QUIC 解决队头阻塞）；
- **HTTPS**：HTTP + TLS：非对称握手协商对称密钥（RSA/ECDHE）→ 对称加密传输 + 证书链验身份（CA 签名）+ 完整性（AEAD）——防窃听/篡改/冒充。四者关系分层：HTTPS ⊃ HTTP over TLS over TCP；对比维度记四轴：连接/可靠性/开销/场景。

## Q22. 浏览器输入 URL 后的完整流程（二3 Q7）

1. **URL 解析**与 HSTS 检查；
2. **DNS 解析**：浏览器缓存 → 系统 hosts/缓存 → 本地 DNS（递归）→ 根→顶级→权威（迭代），可能 LDNS 命中；
3. **建立连接**：TCP 三次握手（同源复用连接/QUIC 免握手）；TLS 握手（ClientHello→证书验证→密钥协商→对称通道）；
4. **发送 HTTP 请求**（方法/头/cookie），可能命中 CDN 边缘（302/直接回源）；
5. **服务端处理**：LB→应用/网关→渲染 HTML；
6. **浏览器渲染**：解析 HTML→DOM、CSS→CSSOM、合成渲染树→布局（reflow）→绘制（repaint）→合成层上屏；遇 JS 阻塞解析（async/defer）、执行期间可能再发请求（XHR/图片）；
7. **连接管理**：keep-alive 复用或关闭。答题加分：每步给一个可观测点（devtools 的 waterfall：DNS/Connect/TLS/TTFB/Download 各一列）与一个优化点（预解析 dns-prefetch、TLS1.3 0-RTT、CDN、HTTP/2 复用）。

## Q23. send / write / mmap / sendfile 的数据传输路径（二3 Q8）

以"把文件内容发到 socket"为例，四种拷贝路径：

- **read+write（write）**：4 次拷贝 4 次切换——DMA 盘→页缓存，CPU 页缓存→用户缓冲，CPU 用户缓冲→socket 缓冲，DMA socket→网卡；
- **mmap+write**：文件映射进用户空间（页缓存直接当用户缓冲），省"页缓存→用户缓冲"一次 CPU 拷贝（3 次拷贝）；适合还要**加工数据**的场景（用户态可读文件内容）；
- **sendfile（2.1+）**：`sendfile(out_fd, in_fd, ...)` 全程用户态不接触数据——DMA 盘→页缓存，CPU 页缓存→socket 缓冲（或网卡支持 SG-DMA 时**零 CPU 拷贝**：只传描述符），2 次拷贝 2 次切换；
- **send（普通 socket 写）**= write 同路径（用户缓冲→内核 socket 缓冲→网卡），对比 mmap/sendfile 的意义在于它是"用户已持有数据"的基线；
- 一句话总结：**零拷贝的本质是减少 CPU 参与的拷贝与上下文切换**；nginx `sendfile on`、kafka 日志追加+sendfile 顺序消费都靠它。追问 splice（管道中转）、`MSG_ZEROCOPY`。

## Q24. 进程 / 线程 / 协程（二3 Q9）

见 [09](./09-cpp与系统基础.md) Q18 与 Q19（进程线程、IPC 全解）。三句话定位：**进程**=资源分配单位（独立地址空间/文件表，隔离最强、切换最贵：页表/TLB/缓存失效）；**线程**=调度执行单位（共享进程地址空间，切线程不切页表但仍陷内核、栈切换）；**协程**=用户态调度的轻量执行流（切换=保存几个寄存器+PC，ns 级、可百万级并发，阻塞 IO 要靠 epoll 异步化才不拖垮线程）。联系：一个进程多个线程；一个线程上多协程分时；Go 的 GMP（Q20）是"协程+少量线程"的工业实现。场景：隔离选进程（Chrome 每标签）、并行计算选线程（GIL 外可真并行，见 [09](./09-cpp与系统基础.md) Q19）、高并发 IO 选协程。

## Q25. K8s Informer 工作原理（二3 Q10）

Informer = client-go 里"本地缓存 + watch 驱动"的控制器骨架：

1. **ListAndWatch**：启动先 **List** 全量对象进本地 store（缓存），再建立 **Watch** 长连接收增量事件（ADDED/MODIFIED/DELETED）；
2. **DeltaFIFO**：watch 事件先进队列（去重合并 delta），Reflector 持续泵入；
3. **本地缓存（Indexer）**：线程安全 store + 索引（namespace/label），读走本地——**API Server 的读压力被缓存吸收**（这是存在意义）；
4. **事件分发**：Informer 按 resource version 顺序回调用户注册的 `ResourceEventHandler`（OnAdd/OnUpdate/OnDelete）；
5. **Resync**：周期性把全量缓存重放一遍 OnUpdate（兜底 handler 里的状态漂移，非重新 List）；watch 断线用 resourceVersion 续传，失效（410）则重 List；
- **SharedInformer**：同资源多控制器共享一份缓存与连接（省内存连接）。自定义控制器套路：Informer + WorkQueue（handler 里只入队，worker 消费做 reconcile——失败重试）——答"为什么 handler 里不直接干活"（重入/限速/解耦）是加分项。

## Q26. Docker 底层原理；PID 隔离与关闭（二3 Q11/Q12、实2 Q20）

**原理三大件**：

1. **Namespace**：隔离视图——pid/net/mnt/uts/ipc/user/time（容器只见自己的进程树/网络栈/挂载点）；
2. **Cgroup**：限制资源——CPU/内存/IO/设备配额（见 Q28 与校招题 Q4）；
3. **联合文件系统（OverlayFS）**：镜像分层只读层 + 容器可写层（写时复制 copy-up），分层构建与共享节省存储；
   加上 veth pair + bridge/NAT 实现网络，runc 创建隔离进程跑入口程序。镜像 ≠ 虚拟机：共享宿主内核、毫秒启动、无 Guest OS。

**PID 隔离**（Q12）：pid namespace 层级化——容器内 PID 1 是入口进程，宿主侧它另有全局 PID；`setns()` 加入、`unshare()`/`clone(CLONE_NEWPID)` 创建。**关闭该隔离**：`docker run --pid=host`（容器与宿主共享 PID namespace，能看见宿主进程——调试工具容器常用）；对应地 `--ipc=host`/`--network=host`/`--userns=host` 关其他隔离。追问：为什么容器里 PID 1 要处理信号与僵尸回收（`docker stop` 发 SIGTERM）。

## Q27. etcd vs Redis 的索引实现差异与设计动机（校 Q1）

- **etcd**：MVCC + **B+树（boltdb，附属 lsbolt）**持久化 key 的多版本，内存里 **key → 最新 revision 的 B-tree 索引**（treeIndex），value 落盘按 revision 顺序追加（bolt 单文件 B+树）。watch 依赖 revision 单调（客户端带 rev 续传不丢事件）；**选 B+树**：为持久化事务（serializable snapshot）、范围查询（prefix/compact）与磁盘友好；
- **Redis**：内存数据库，主索引是**全局哈希表**（dict，渐进式 rehash），O(1) 点查；辅助结构按类型（skiplist 的 zset、双端链表 list、紧凑列表 listpack 等）；**选哈希**：纯内存、点查延迟优先、无磁盘布局约束；
- **动机差异**：etcd 服务于"分布式一致性元数据"（Raft 强一致、watch、lease），读多写少、持久可靠压倒延迟；Redis 服务"缓存/计数器"，吞吐延迟压倒持久语义（异步落盘）。同题引申：etcd 的 Raft quorum 写放大 vs Redis 单机写；为什么 K8s 选 etcd 不选 Redis（一致性等级）。

## Q28. cgroup 与 namespace 的原理和作用（校 Q4）

- **namespace**：隔离**视图**——让进程看见"自己的"系统（PID 树/网络栈/挂载/user 映射），六/七种类型；解决"看不见别人"；
- **cgroup**：限制**用量**——控制进程组可用的 CPU（权重/配额 `cpu.cfs_quota_us`）、内存上限（超限 OOM kill）、IO 带宽、设备访问；层级化目录（cgroup v1 多层级挂载 / v2 统一层级）；解决"用不多拿"；
- 协作构成容器（见 Q26）：namespace 管"看到什么"，cgroup 管"能用多少"。K8s 的 request/limit（校 Q6）底层就是 cgroup：request 影响 CPU shares（v1）/weight（v2）与调度决策，CPU limit 映射 CFS quota（`cfs_quota/cfs_period`，throttling 机制：超配额的线程被节流）；memory limit 映射 `memory.max`，超限 cgroup OOM（容器被杀=OOMKilled 状态）。追问点：CPU throttling 长尾延迟排查（`/sys/fs/cgroup/cpu.stat` 的 nr_throttled）、Guaranteed QoS（request=limit）避免节流。

## Q29. K8s Pod 创建流程；CSI 与 CNI 的调用阶段（校 Q5）

**创建全流程**：kubectl 提交 → **API Server** 校验写入 etcd → scheduler watch 到未调度 Pod，**过滤（资源/亲和/污点）+ 打分**选出 Node，回写 `nodeName` → kubelet（目标节点）watch 到分给自己的 Pod → **CRI 创建 sandbox（pause 容器）→ CNI 配网络 → CSI 挂卷 → 启动 init/业务容器（postStart 钩子、探针）** → 上报状态。

**CNI 调用阶段**：sandbox 创建后、业务容器启动**前**——kubelet 调 CNI 插件为 pod 的 netns 配 veth/IP/路由（bridge/calico/flannel 各自实现）。**CSI 阶段**：业务容器启动前挂卷——ControllerPublish（attach 到节点）→ NodeStage（格式化挂全局目录）→ NodePublish（bind mount 到容器路径）；pod 删除时逆序 unmount。记忆锚：**网络在 sandbox 后、容器前；存储在容器前，两段式（节点级+容器级）**。追问：为什么 pause 容器先起（持有 netns/ipc ns，业务容器加入它）。

## Q30. Kafka 顺序消费、消息不丢失、顺序写文件机制（校 Q2）

- **顺序性保证**：分区内有序——**同一 key（如订单 id）hash 到同一 partition**，生产端 `max.in.flight.requests≤5` + 幂等（`enable.idempotence`）或事务防重排；消费端**单线程消费单分区**（或手动按 key 再分发到本地内存队列）。跨分区全局有序不提供（代价太大）；
- **不丢失三段**：**生产**acks=all（ISR 全确认）+ retries + 幂等；**broker** replication.factor≥3、min.insync.replicas≥2、unclean.leader.election=false；**消费**先处理完再提交 offset（手动提交/事务），at-least-once + 消费幂等去重；
- **顺序写文件机制**：分区=只追加（append-only）日志段文件，**磁盘顺序写接近内存随机写**（页缓存+顺序 IO 免寻道）；消费靠 page cache 热数据（zero-copy sendfile 直接发，见 Q23）；段文件按大小/时间滚动，索引稀疏（offset→物理位置的稀疏索引）。这套"追加日志 + 顺序 IO + 零拷贝 + 页缓存"是 Kafka 吞吐的四大支柱，答的时候串起来。

## Q31. 操作系统的文件管理机制（校 Q3）

- **分层**：VFS（虚拟文件系统统一接口）→ 具体文件系统（ext4/xfs/tmpfs/procfs）→ 页缓存 → 块设备驱动 → 磁盘；
- **文件系统数据结构**（ext4 口径）：超级块（全局元信息）、inode（文件元数据+数据块指针/ extents，**文件名在目录项里不在 inode**）、目录（特殊文件，内含文件名→inode 映射）、空闲空间位图； inode 定位路径：路径逐级查目录项；
- **读写路径**：write 先进**页缓存**（dirty page）→ pdflush/回写线程按阈值刷盘（fsync 强制）——延迟写提升吞吐但断电风险；mmap 绕过用户缓冲（见 Q23）；
- **权限与管理**：rwx 与 ACL、打开文件表（fd→file→inode 三级结构，dup/fork 共享偏移）、磁盘配额、journal 日志防崩溃不一致（ext4 journal）；
- **Linux 目录组织**（若被追问"文件管理"的广义）：FHS（/etc /var /proc...）、inode 用尽 vs 磁盘满两种"满"。

## Q32. C++ 基础连发：static / const / 四种 cast / .so 符号 / 析构（实2 Q1/Q2/Q5/Q6/Q7）

- **static**（Q1）：见 [09](./09-cpp与系统基础.md) Q2 全解（函数内/文件内/类成员三作用域 + 单例应用）；
- **const**（Q2）：修饰变量（常量、必须初始化）/指针（`const int*`（指物常）vs `int* const`（指针常））/函数参数（只读语义）/成员函数（不改成员，const 对象只能调它）/返回值；`constexpr` 编译期常量与函数；
- **四种 cast**（Q5）：见 [09](./09-cpp与系统基础.md) Q10（static_cast/dynamic_cast/const_cast/reinterpret_cast 语义与安全等级）；
- **查看 .so 符号**（Q6）：`nm -D libfoo.so`（T=导出 U=未定义）、`objdump -T/-t`、`readelf -s --dyn-syms`、`ldd`（依赖）、`c++filt`（demangle）；配合 `dlopen/dlsym` 运行时查询；排查符号冲突/版本（ABI）问题的基本功；
- **析构与参数/返回值**（Q7）：析构**不能带参数**（无重载可能——调用时机由运行时决定，无参可传）、**无返回值**（语法禁止，对象销毁无值可返回）；构造函数同样无返回值。隐式调用（作用域结束/delete）决定了签名必须是"无参无返回"。

## Q33. 虚构造/虚析构；RAII；move 与左右值（实2 Q9/Q10/Q11、实一1 Q12）

- **构造/析构能否为虚**（Q9）：**虚构造不存在**——虚函数机制靠对象里的 vptr，而构造时对象（和 vptr）还没建好，鸡生蛋问题；需要"虚构造"语义用 clone() 惯用法（virtual 成员函数返回 `new Derived(*this)`）。**虚析构必须有条件地有**：经基类指针 delete 派生类对象时，非虚析构只调基类析构（派生部分泄漏）；基类设计给继承就声明 `virtual ~Base() = default;`（[09](./09-cpp与系统基础.md) Q8 有 vtable 全解）；
- **RAII**（Q10）：资源获取即初始化——资源（内存/锁/文件/socket）的生命周期绑定栈对象：构造获取、析构释放（确定性、异常安全——栈展开自动调析构）。应用：智能指针、`lock_guard`、`unique_lock`、`fstream`；是"C++ 忘记 free/死锁"问题的系统性答案；
- **std::move 与左右值**（Q11、实一1 Q12）：**左值**=有名字可取地址（变量/解引用），**右值**=临时的将亡值（字面量、函数返回临时、`x+1`）；右值引用 `T&&` 绑定右值，使其可被"掏空"（ steal 内部资源，留空壳）。`std::move` 本身不移动——只是 `static_cast<T&&>` 转成右值表达式，触发移动构造/移动赋值的重载决议（[09](./09-cpp与系统基础.md) Q4 的移动/转发全解）。

## Q34. C++11 重要新特性（实一1 Q7、实一1 Q11 成员模板函数）

分组报：**语言**：auto/decltype、右值引用与移动语义、lambda、range-for、nullptr、constexpr、初始化列表 `{}`、可变参数模板、`override/final`、委托构造、enum class、static_assert；**库**：智能指针、`std::thread/mutex/future`、atomic、unordered 容器、array/forward_list/tuple、regex、chrono、function/bind；**并发内存模型**正式定义（happens-before）。追问**成员函数可以是模板吗**（实一1 Q11）：可以（普通成员/构造函数都行，如 CRTP 与 std::enable_if 魔法），**但虚函数不能是模板**——vtable 需要编译期固定函数指针集合，模板实例数量不定（这是 [09](./09-cpp与系统基础.md) Q9 CRTP 存在的理由：编译期多态替代虚函数）。

## Q35. new vs malloc；vector 的存储；栈/堆 vector（校一2 Q3/Q4）

- **new vs malloc**（Q3）：见 [09](./09-cpp与系统基础.md) Q6：new = 分配+调构造+类型安全+失败抛 bad_alloc（可 `new (nothrow)`）+可重载/定位 new；malloc 只给裸内存返回 void*。delete/delete[] 与之配对（错配 UB）；
- **vector 存储**（Q4）：vector 对象本身很小（三个指针/迭代器：begin/end/capacity），**元素永远在堆上连续分配**——不管 vector 对象在栈上还是 `new` 出来的堆上。两种写法的区别只是"vector 对象这一小块元数据"放哪：栈上对象随作用域自动析构；`new vector<int>` 则对象在堆上，需要手动 delete（且内层又是堆——"堆上套堆"）。追问：容量增长（翻倍/1.5 倍、搬运拷贝、迭代器失效）、`reserve` 与 `resize` 区别（[09](./09-cpp与系统基础.md) Q12）。

## Q36. 迭代 vs 递归；CPU 分支预测优化（校二2 Q4/Q5）

- **迭代 vs 递归**（Q4）：递归代码简洁（树/回溯天然递归）但有函数调用开销、栈空间 O(depth)（深了爆栈）、重复子问题（无记忆化时指数级，如朴素 fib）；迭代省栈高效，但树形问题改写难（手动显式栈模拟）。优化路线：尾递归（编译器可转迭代，但 C++ 标准不保证）、记忆化/DP、显式栈化。选型：线性结构用迭代，树/图/回溯用递归+记忆化，深递归上生产改显式栈；
- **CPU 分支预测**（Q5）：流水线遇条件分支需提前取指 → 硬件**分支预测器**猜方向与目标：静态（编译器 hint/向前向后启发）到动态（1/2 位饱和计数器、两级自适应、TAGE 等感知历史模式，现代准确率 >95%）；**投机执行**沿预测路径执行，猜错则冲刷流水线（15-20 cycle 惩罚，Spectre 类侧信道的根源）。编译器侧优化：`__builtin_expect`/`[[likely]]`、分支消除（cmov、位运算、查表）、循环展开与分离（把不可预测分支挪出热循环）；数据侧：分支密集循环按分支值预排序（经典案例：有序数组快于无序求和）。

## Q37. Python 篇：内存管理 / 装饰器 / import 机制（实2 Q12/Q13/Q14）

- **内存管理**（Q12）：**引用计数**为主（即时回收，循环引用无解）+ **分代标记-清除 GC** 补循环引用（0/1/2 三代，阈值触发，`gc` 模块可控）；对象池与小整数/字符串驻留（-5~256、interning）；内存池：pymalloc 的 arena→pool→block（<512B 走池）；GIL 与内存的关系见 [09](./09-cpp与系统基础.md) Q19；
- **装饰器**（Q13）：本质是**高阶函数**——接收函数返回新函数（`@deco` ≡ `f = deco(f)`），在不动原函数代码的情况下加逻辑（日志/计时/缓存/重试/鉴权）。带参装饰器再多一层工厂；`functools.wraps` 保元信息；类装饰器（`__call__`）。手写一个带缓存/计时的装饰器是常见追问；
- **import 过程**（Q14）：①`sys.modules` 缓存查（**模块只执行一次**——单例副产品的根源）；②未命中则按 `sys.path` **查找**（finder：内置→frozen→路径 finder，包优先目录/`__init__.py`）；③**编译**：.py → pyc 字节码缓存（`__pycache__`，mtime 校验）；④**执行**模块顶层代码（定义函数/类，产生命名空间）；⑤绑定到当前命名空间并把模块对象给 importer。循环导入死锁与 `if __name__ == "__main__"` 的机制都从这条流程讲。

## Q38. Git：merge vs rebase；撤回已提交 commit（实2 Q15/Q16）

- **merge vs rebase**（Q15）：merge 把两分支头做一个三方合并的新提交（保留真实分叉历史，"歪巴树"）；rebase 把当前分支的提交**摘下来重放到**目标分支之上（历史线性干净，但**改写提交**——已推送的公共分支禁止 rebase）。团队规范：私有分支跟进主干用 rebase（清爽），合入主干用 merge（--no-ff 留痕）；
- **撤回 commit**（Q16）：**未推送**：`git reset --soft HEAD~1`（改动留暂存区，推荐）/`--mixed`（默认，留工作区）/`--hard`（丢弃）；**已推送共享**：`git revert <commit>`（生成反向提交，安全不改历史）再 push；改最后一条未推：`git commit --amend`；多步交互整理：`git rebase -i`（squash/fixup）。追问点：reset 与 revert 的本质（移动引用 vs 新提交）、reflog 兜底找回（`git reflog` + reset 到误删前的 SHA）。
