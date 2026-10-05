---
title: "T4 车企题解：C++ 与系统"
description: "T4 车企/自动驾驶面经 C++ 与系统题全解：C++11/14/17/20 特性、面向对象/继承/多继承/异步编程、map 与 unordered_map、const 与动态库、Python 切片与装饰器、虚拟内存/OS 调度/TCP-IP/K8s、项目与软素质题答题框架"
pubDate: 2026-10-05
topic: "公司题解-T4"
order: 92
tags: ["公司题解", "C++ 基础", "操作系统", "计算机网络"]
draft: false
---

覆盖题库：`蔚来-AI-Infra-实习-HR面`（1）、`蔚来-AI-Infra-实习-一二三面`（一面 6-11、二面 5-10、三面 2/4）、`蔚来-AI-Infra-实习-一面-1`（3）、`蔚来-AI-Infra-实习`（5/6）、`小鹏汽车-AI-Infra-一面`（1-3）、`小鹏汽车-AI-Infra`（1）、`卓驭-AI-Infra-校招-一面`（5）、`文远知行-AI-Infra-校招-一面`（5-9）、`大疆-AI-Infra-校招-一面/二面`、`卓驭-AI-Infra-校招-二面`、`易控智驾-AI-Infra-二面`（5/6）及其余公司项目题。

与[09 篇](./09-cpp与系统基础.md)同族的题（cast/智能指针/虚函数/进程线程/IPC 等）给增量并交叉引用；推理/CUDA 题见[90](./90-t4-编译器与mlir.md)/[91](./91-t4-推理部署与tensorrt.md)篇，编程题见[93 篇](./93-t4-编程与算法.md)。

## 一、C++ 语言

### Q1. C++11/14/17/20 各版本新特性（蔚来-HR面 1、三面 2）

按"特性 + 一句原理/用途"准备，每版挑 5-8 个能展开的：

- **C++11（现代 C++ 起点）**：`auto`/`decltype`、范围 for、`nullptr`（强类型空指针，解决 `0` 与指针重载歧义）、**右值引用与移动语义**（`std::move`/`forward`，[09 篇 Q4](./09-cpp与系统基础.md)）、lambda、**智能指针** `unique_ptr/shared_ptr/weak_ptr`、`constexpr`、可变参数模板、`std::thread/mutex/atomic/future` 并发库、`enum class`、`override/final/default/delete`、初始化列表、`static_assert`、模板别名、`unordered_map/set`、`tuple`、`chrono`；
- **C++14（小步完善）**：泛型 lambda（参数 `auto`）、lambda **初始化捕获** `[p=std::move(ptr)]`（唯一能移动捕获的方式）、函数返回值推导、`constexpr` 放宽（可写循环）、`std::make_unique`、变量模板、二进制字面量 `0b1010`、`shared_timed_mutex`；
- **C++17（工程利器）**：**结构化绑定** `auto [k, v] : map`、`if/switch` 带初始化、`constexpr if`（编译期分支，模板代码瘦身）、**CTAD**（`std::pair p(1, 2.0)` 类模板参数推导）、fold 表达式、`std::optional/variant/any`（`solutions/cpp/include/answers/` 内存池返回值就用了 optional）、`string_view`（零拷贝切片）、`filesystem`、**并行算法** `std::for_each(par)`、inline 变量；
- **C++20（四大件）**：**concepts**（模板约束，替代 SFINAE）、**coroutines**（co_await/co_yield）、**modules**（替代头文件，编译提速）、**ranges**（管道式算法 `views::filter|transform`）、三路比较 `<=>`、`constexpr` 大扩展、`consteval`、`std::span`（数组视图，kernel 参数传递友好）、`std::format`、`jthread`/`stop_token`、atomic 等待通知、指定初始化器。

车载岗常追问：**项目里用过哪个版本的什么特性**——准备 2-3 个具体例子（如用 `make_unique` + 移动语义优化 kernel launcher 的 buffer 传递、`string_view` 避免 plugin 参数拷贝）。

### Q2. C++ 面向对象的三大特性及具体体现（蔚来-一二三面 一面 6）

**封装**：访问控制（public/protected/private）把数据与操作绑定，隐藏实现（getter/ invariant 维护、Pimpl）；**继承**：is-a 关系复用与扩展（虚函数多态的基础，见 Q3）；**多态**：同一接口不同行为——运行期（虚函数，vptr/vtable 动态分派，[09 篇 Q8](./09-cpp与系统基础.md)）与编译期（重载、模板、CRTP，[09 篇 Q9](./09-cpp与系统基础.md)）。加分：讲"组合优于继承"原则与车载代码里的例子（推理引擎的 backend 抽象：`class InferEngine { virtual void run() }` 派生 TRT/NPU 后端）。

### Q3. C++ 继承机制（蔚来-一二三面 一面 7）

语法之外要能讲底层：派生类对象内含基类子对象（单继承共享 vptr、成员追加；多继承按声明序排列多个基类子对象、跨基类调用 this 指针 thunk 偏移；虚继承经 vbptr 共享虚基类避免菱形重复——完整布局见[09 篇 Q8](./09-cpp与系统基础.md)）。构造/析构顺序：基类构造→成员→派生类构造，析构逆序；继承下的访问：public/protected/private 继承改变可见性；**继承 + 虚析构**是资源安全底线（`delete base_ptr` 漏调派生析构）；名字遮蔽与 using 声明。原则补充：继承表达"是"，组合表达"有"；接口继承（纯虚）与实现继承分开。

### Q4. 虚函数的实现原理与使用场景（蔚来-一二三面 一面 8）

完整答案见[09 篇 Q8](./09-cpp与系统基础.md)：vtable（虚函数指针数组+RTTI）+ 对象头 vptr，调用 `(*(p->vptr)[slot])(p)` 两级间接实现动态分派；vtable 在只读数据段、每对象 8B vptr 成本；构造不可虚（对象未成型）、析构必须虚。场景：运行期才确定实现的接口（推理 backend、算子注册的 dispatch、设备抽象）。补充车载追问：**为什么不所有函数都虚**——禁止内联 + 间接跳转 + 分支预测压力，热点路径（kernel 启动循环）用 CRTP/模板静态分派（[09 篇 Q9](./09-cpp与系统基础.md)）。

### Q5. 多继承的特点与注意事项（蔚来-一二三面 二面 5）

特点：一个类多个基类子对象拼接；**多 vptr**、this 指针跨基类偏移（thunk）；**菱形继承**下公共基类子对象重复（两份 B 数据不一致）。注意事项：①菱形用 `virtual` 继承让公共基类共享（代价 vbptr 间接寻址，构造由最派生类负责）；②二义性：同名成员需 `A::f()` 限定；③接口多继承 + 实现单继承是工程惯例（Java/C# interface 的由来）；④虚析构在首个 polymorphic 基类上必须有。测试代码：`solutions/cpp/include/answers/`（布局验证见 09 篇 Q8 的分析）。

### Q6. 智能指针的种类与使用经验（蔚来-一二三面 二面 6）

`unique_ptr`（独占 move-only 零开销，默认首选）/`shared_ptr`（控制块原子计数）/`weak_ptr`（破循环、观察）三件套 + 线程安全粒度（控制块原子、对象本身不保证）完整见[09 篇 Q3](./09-cpp与系统基础.md)。**使用经验**（结合项目讲）：所有权语义设计 API（sink 参数 `unique_ptr<T>` 传值 = 转移）；`make_shared` 一次分配（控制块与对象同块，缓存友好；缺点是无法提前释放）；工厂返回 `unique_ptr` 隐式升级 shared；CUDA buffer 用 `unique_ptr<void, Deleter>` 自定义删除器（`cudaFree`）；`enable_shared_from_this` 用于异步回调中保活自身。手写实现（atomic 计数 + 移动语义 + 数组特化）：`solutions/cpp/include/answers/smart_ptr.hpp`。

### Q7. static 与 const 的区别（蔚来-一二三面 二面 7）

**static 四用法**（局部静态/内部链接/静态成员变量/静态成员函数）见[09 篇 Q2](./09-cpp与系统基础.md)；**const**：变量只读、指针的顶层/底层 const、const 成员函数（不改成员、const 对象只能调它）、修饰引用传参（`const T&` 免拷贝）。组合考点：`static const int` 类内直接初始化（整型）；类内 `static` 成员需类外定义（C++17 `inline static` 解）；const 成员函数里想改某成员 → `mutable`（缓存/计数器）；**const 与编译期**：`constexpr` 才是编译期常量，const 只承诺运行期只读（初始化可以是运行期值——高频陷阱，见 Q10 找错题）。

### Q8. C++ 异步编程的实现方式与使用经验（蔚来-一二三面 二面 8）

层级递进：①**std::thread + 条件变量/回调**（原始，要自己管生命周期与同步）；②**std::async/future/promise/packaged_task**（`std::async(std::launch::async, f)` 返回 future，`get()` 阻塞取值/异常；promise 手动 set_value 打通线程；注意 future 析构会阻塞 async 线程、`std::launch::deferred` 语义）；③**线程池 + 任务队列**（生产者-消费者，工程标配——async 每次起线程开销大；车载推理引擎的多流任务派发即此）；④**epoll/io_uring 事件循环**（IO 密集，Proactor/Reactor）；⑤**C++20 协程**（co_await 挂起恢复，无栈协程，把异步回调写成同步样式）。经验示例：推理服务的请求处理 = 线程池收请求 → GPU 流异步 enqueue → future/promise 回填结果；坑：future 只能 get 一次（shared_future）、捕获 this 的 lambda 生命周期、异常跨线程（future 会转发）。

### Q9. 四种 cast 的区别与底层机制（蔚来-实习 5、小鹏-一面 1）

速查见[09 篇 Q10](./09-cpp与系统基础.md)：`static_cast`（编译期可验证的相关类型转换，不做运行期检查）、`dynamic_cast`（多态下行/侧转，**运行期查 RTTI**：沿 vtable 里的 type_info 继承链比对，失败返回 nullptr/抛 bad_cast，有开销）、`const_cast`（唯一能去 const，本质改对象的类型视图不改对象）、`reinterpret_cast`（位级重解释，指针↔整数/不相关指针）。**底层机制**对比是 T4 增量：前三种编译成"带检查的类型视图调整"（dynamic_cast 才有运行期代码），reinterpret_cast 是零指令（仅编译期类型改变）。父转子必须 dynamic_cast（static_cast 不检查，错误的下行是 UB）。

### Q10. 父类指针转子类的安全性问题与内存布局约束（蔚来-实习 6）

安全性：**只有对象动态类型确实是该子类时**下行转换才合法。`static_cast` 强转一个实际是 Base 的指针 → 后续访问 Derived 成员读到越界/错位内存（UB，静默破坏，[09 篇 Q20](./09-cpp与系统基础.md) 的越界排查同族）；`dynamic_cast` 失败安全返回 nullptr（先判空）。内存布局约束：Derived 在 Base 子对象后追加成员，`p->derived_member` 按 Derived 布局取偏移——若实际对象是 Base，偏移越界；**多继承下**指针值本身会变（this 调整），跨基类强转不是"同一个地址换类型"。工程规范：下行前 `dynamic_cast` + 判空，或用 visitor/虚函数把行为放回派生类，避免下行。

### Q11. 动态链接库的依赖顺序问题及解决（小鹏-一面 2）

现象：链接可执行文件时 `-la -lb`，若 a 依赖 b，**被依赖的库必须放后面**（gcc/ld 从左到右扫描，符号未定义时向后找——写反了报 undefined reference）。解决：①调整顺序（依赖者在前）；②`--start-group ... --end-group` 循环依赖兜底（慢）；③重复列库；④CMake 的 `target_link_libraries(target PRIVATE a b)` 自动按依赖拓扑排（现代做法）；⑤运行期 `ldd` 查依赖缺失、`rpath/$ORIGIN`/`LD_LIBRARY_PATH` 解决加载路径（部署时 so 找不到的问题，与编译顺序是两件事）。原理见[09 篇 Q17](./09-cpp与系统基础.md)。

### Q12. 阅读代码找出 4 处编译错误（考察 const 的作用域与约束）（小鹏-一面 3）

题面不固定，按 const 高频错误清单现场对照：

1. **const 对象调用非 const 成员函数**：`const Foo f; f.bar();`（bar 非 const）→ error：passing 'const Foo' as 'this'；
2. **const 成员函数内修改成员**：`void bar() const { m_x = 1; }` → error：assignment of member in read-only object（要改加 `mutable`）；
3. **非 const 引用绑定临时/const 对象**：`void f(int& x); f(42)` 或 `const int a=1; f(a)` → error：cannot bind non-const lvalue reference；
4. **const 声明与使用不匹配**：`const int* p; *p = 1`（pointee 只读）；`int* const q = &a; q = &b`（指针本身只读）——顶层/底层 const 弄反；
5. 类内 `static const std::string x = "s";`（非整型不能类内初始化，C++17 前）；
6. 返回局部变量的 `const&`（生命周期，编译器可警告/UB）；
7. const 成员函数调用同对象非 const 成员函数（传播）。

答题技巧：先区分**顶层/底层 const**，再按"对象→成员函数→引用绑定→初始化规则"四处扫。

### Q13. map 与 unordered_map 的底层实现与复杂度；vector 底层（小鹏-AI-Infra 1、蔚来-一二三面 一面 9）

`map`：红黑树（平衡 BST），有序遍历、最坏 O(log n) 查插删、迭代器稳定、节点式分配（缓存不友好）；`unordered_map`：开链哈希（bucket + 链表，负载因子超阈值 rehash 翻倍），平均 O(1) 最坏 O(n)（冲突/攻击）、无序、rehash 使迭代器失效、需 hash + equality。选择：要有序/range 查询/最坏界 → map；纯点查 → unordered_map。`vector`：连续数组、扩容 2 倍搬移（[09 篇 Q12](./09-cpp与系统基础.md)）、缓存最优。追问 hash 表 rehash 的均摊、`reserve` 预防、自定义类型作 key（写 hash/equal）。

## 二、Python 基础

### Q14. Python 切片操作反转列表（蔚来-一二三面 二面 9）

```python
rev = lst[::-1]          # step=-1 的切片，生成新列表（浅拷贝）
lst.reverse()            # 原地翻转，返回 None（高频坑）
rev_iter = reversed(lst) # 迭代器，零拷贝遍历用
```

原理：切片协议 `s[start:stop:step]` 走 `slice` 对象（`__getitem__(slice(None, None, -1))`），step 为负时从尾部往头部取——对序列类型是 O(n) 的新列表。延伸：numpy/torch 张量同样 `x[::-1]`（返回视图而非拷贝，与 list 不同）；字符串反转 `s[::-1]` 判回文。

### Q15. Python 装饰器的原理与用法（蔚来-一二三面 二面 10）

装饰器 = **接收函数返回函数的高阶函数**，`@deco` 是 `f = deco(f)` 的语法糖；实现机制是闭包（内层函数捕获外层变量，`__closure__` 存 cell）。标准模板与坑：

```python
import functools

def deco(func):
    @functools.wraps(func)              # 保住 __name__/docstring（高频追问）
    def wrapper(*args, **kwargs):       # *args/**kwargs 透传（转发任意签名）
        ...before...
        return func(*args, **kwargs)    # 调原函数
    return wrapper

@deco                                    # 等价 my_func = deco(my_func)
def my_func(): ...
```

带参装饰器再包一层（三层函数）；类装饰器用 `__call__`；多个装饰器自下而上应用、自上而下执行。用途：计时/profiler 埋点（CUDA kernel 的 launch 统计）、缓存（`functools.cache`）、重试、注册表（算子注册表模式：`@REGISTER_OP("conv")`——推理框架常见）。

## 三、操作系统、网络与云

### Q16. 多进程与多线程的区别及相关知识（蔚来-一面-1 3）

进程=资源单位（独立地址空间/fd），线程=调度单位（共享地址空间、私有栈/TLS）；切换成本（线程换寄存器栈 ~us，进程换页表 TLB 失效）、隔离性、通信方式（IPC vs 共享内存+同步）完整见[09 篇 Q18](./09-cpp与系统基础.md)。相关扩展：协程（用户态调度，切换 ns 级）；多线程资源共享细节见 Q18。

### Q17. Linux 多进程间通信的方式（蔚来-一二三面 一面 10）

管道/FIFO、消息队列、**共享内存**（mmap/shm，最快，配信号量/互斥锁同步）、信号量、信号、socket（本机/跨机，通用）、内存映射文件、eventfd；选型（大数据量共享内存、低频消息队列、跨机 socket）完整见[09 篇 Q19](./09-cpp与系统基础.md)。

### Q18. 多线程之间如何进行资源共享（蔚来-一二三面 一面 11）

先分类共享什么：**堆/全局变量/静态变量/代码段**天然共享；栈私有（逃逸出函数的指针被别的线程用 = 悬垂风险）。同步原语按场景选：互斥锁（临界区）、读写锁/`shared_mutex`（读多写少）、原子变量（计数器/flag，无锁）、条件变量（事件等待，配谓词防虚假唤醒）、信号量（资源池配额）、屏障/`latch`（阶段对齐）。正确性要点：最小临界区；锁粒度与死锁四条件/预防（全局加锁序，[09 篇 Q14](./09-cpp与系统基础.md)）；**false sharing**（两线程各写相邻变量落同一缓存行，`alignas(64)` 隔离——车载多线程性能题高频）；无锁结构（CAS 循环、ABA）适合指针/计数。GPU 侧对应：线程块内 smem + `__syncthreads`，跨块 global+atomic（[91 篇 Q43](./91-t4-推理部署与tensorrt.md)）。

### Q19. 多进程与多线程在性能方面的区别（卓驭-校招一面 5）

**创建/切换开销**：进程创建（fork 复制页表 COW）与切换（页表/TLB 刷新）都显著贵于线程（仅寄存器/栈）；**通信开销**：线程共享内存零成本（但要同步），进程要 IPC（管道拷贝/共享内存 setup）；**并行能力**：多核利用两者都行（进程绑定核避迁移）；**GIL 特例**（Python）：CPU 密集多线程无法多核 → multiprocessing；**隔离与稳定性**：进程崩不牵连（车载功能隔离/安全岛的设计依据）、线程崩全进程崩；**IO 密集**：线程/协程切换更便宜。经验法则：计算隔离/容错 → 多进程；高并发任务协作 → 多线程/协程 + 线程池。

### Q20. 虚拟内存的概念与作用（文远-校招一面 5）

每个进程看到连续私有的"虚拟地址空间"，由 **MMU 经页表（多级，x86-64 四级/五级）+ TLB** 翻译到物理页；缺页异常按需调入/换出。作用：①**隔离与保护**（进程间不可见，权限位 R/W/X）；②**地址空间抽象**（链接期固定地址、堆栈远隔）；③**按需分配与超额承诺**（malloc 只映射虚拟，触页才给物理）、swap 扩容；④**共享**（同一物理页映射多进程：动态库/共享内存）、**COW**（fork 后写时复制）；⑤大页（2MB/1GB TLB 命中提升，NUMA/大内存服务调优点）。与 GPU 的呼应：Unified Memory 本质是把虚拟内存抽象扩展到 host+device（[01 篇 §14](./01-cuda-基础与体系结构.md)）。

### Q21. 进程与线程的区别；操作系统如何调度（文远-校招一面 6）

区别部分见 Q16。**OS 调度**：调度器管理就绪队列，按时间片轮转 + 优先级（CFS：按 vruntime 红黑树选最"亏"的进程，权重由 nice 值定；实时类 SCHED_FIFO/RR 抢占普通类）；调度时机（时钟中断/阻塞/唤醒/显式 yield）；上下文切换（保存寄存器/切换页表/换内核栈）；多核负载均衡与亲和性（`taskset`/cgroup，推理服务绑核避免 cache 抖动）；线程是内核调度实体（1:1 模型），同进程线程切换不换页表。车规延伸：RTOS 的优先级抢占（硬实时截止期）vs Linux CFS（公平吞吐）。

### Q22. TCP/IP 协议栈各层的作用（文远-校招一面 7）

- **应用层**（HTTP/gRPC/自定义协议）：语义与数据格式；
- **传输层**（TCP/UDP）：端到端**进程**间通信——TCP 面向连接、可靠（序号/ACK/超时重传/滑动窗口/拥塞控制 slow-start-cubic-BBR）、UDP 无连接低开销（实时视频/训练同步的 RDMA 之外路径）；
- **网络层**（IP/ICMP）：主机到主机**寻址与路由**（分片、TTL、最长前缀匹配）；
- **数据链路层**（以太网/ARP）：同一链路内帧传输、MAC 寻址；
- **物理层**：比特传输。

答此题配一个数据包的完整旅程（应用 send → 分段 → 加 IP 头路由 → ARP 找 MAC → 帧传输 → 对端逆向上交）最清楚。分布式训练/推理相关延伸：NCCL 优先 RDMA（旁路内核）的原因即各层开销（[07 篇](./07-分布式训练.md)）。

### Q23. 网络通信中如何通过 IP 地址找到目标主机（文远-校招一面 8）

发送方视角分步：①**路由决策**：目的 IP 与本机子网比较（掩码），同网段直发，否则查路由表（`route -n`）选默认网关/对应出口；②**下一跳 MAC 解析**：**ARP**（广播 "谁是 192.168.1.1"，目标回 MAC，本网段直接问目标本身），ARP 表缓存；③**封装成帧**经交换机（MAC 表转发）/路由器逐跳中继，每个路由器重复"查表→ARP→转发"（IP 头 TTL 递减）；④目的主机收帧解包。补充：跨网段时帧的 MAC 是网关的、IP 始终是目标的（两层地址分工）；DNS 解决的是"名字→IP"（在找主机之前）；NAT 让私网共用公网 IP（出向改写）。注意题面若问"通过 IP 找到主机"是路由+ARP，别答成 DNS。

### Q24. Kubernetes 的基本架构（文远-校招一面 9）

**控制面**：API Server（唯一入口，REST + etcd 读写）、etcd（集群状态存储）、Scheduler（按资源/亲和性选 node 绑 Pod）、Controller Manager（调和循环：Deployment/ReplicaSet 控制器把实际状态拉向期望）。**节点**：kubelet（节点代理：按 PodSpec 拉起/健康检查容器）、kube-proxy（Service 的网络规则：iptables/IPVS）、容器运行时（containerd）。**核心概念**：**Pod** = 最小调度单元（一或多容器，共享网络命名空间同 IP、可共享 volume——sidecar 模式的基础）；Deployment（副本集 + 滚动更新）；Service（稳定虚拟 IP 负载到 Pod 集）；ConfigMap/Secret（配置注入）；Namespace（隔离）。推理服务相关：GPU 设备插件（`nvidia.com/gpu` 资源）、HPA 自动扩缩、亲和性让副本散开（[91 篇 Q51](./91-t4-推理部署与tensorrt.md) 调度的集群视角）。

## 四、项目经历与软素质题

T4 公司（大疆/卓驭二面/蔚来 HR 等）有大量项目与行为题，此类题没有标准答案，给出答题框架与要点索引。

### Q25. 项目题的通用答题框架（蔚来/小鹏/元戎/辉羲/理想/文远 项目题通用）

用 **STAR 变体（背景-难点-方案-量化结果-反思）**：①一句话说清项目目标与你的角色；②**难点要具体到技术**（"第 3 个 conv 占 40% 耗时且 smem 超 100KB"而不是"性能不好"）；③方案讲**选型对比**（为什么 CUDA 手写而不是 Triton/为什么 bufferization 选 one-shot）；④**量化结果**（时延 xx ms → yy ms，accelerate N×，实测命令能复现——本仓库 solutions/ 的写法就是范本）；⑤反思与下一步（泛化性/自动化）。高频项目追问对照：TensorRT 项目（[91 篇 Q3](./91-t4-推理部署与tensorrt.md) 转换坑清单）、量化工作（[06 篇 Q4](./06-量化.md) 校准方法）、CUDA 优化工作（[91 篇 Q26](./91-t4-推理部署与tensorrt.md) 五步流程）、MLIR 编译器项目（[90 篇 Q2-Q9](./90-t4-编译器与mlir.md)）、训练优化（[07 篇 Q13](./07-分布式训练.md) 排查）。"回顾项目哪些环节有改进空间"（大疆-一面 3、"改进方向"蔚来-二面 2）：诚实给 1-2 个真实局限 + 已知解法，比"没有"好。

### Q26. 大疆车载/卓驭业务了解与择业动机（大疆-一面 4、二面 1-4；卓驭-二面 1-5）

技术侧准备：大疆车载（卓驭）的成行平台——**惯导双目视觉**为主的路线（不依赖激光雷达/高精图）、自研 SoC 与算法栈；对应 AI Infra 岗的落点：双目立体匹配/BEV 感知模型的 NPU 部署与量化、端侧时延优化。行为题按个人经历准备即可，通用要点：为何车载（技术密度：软硬一体 + 实时系统 + 视觉算法闭环；业务确定性：量产落地）要能说出**与互联网推理岗的差异认知**（时延确定性/功耗/安全 vs 吞吐）；职业规划给 1 年（上手业务成为某模块 owner）与 3 年（纵深：成为端侧推理/编译器方向专家）两层；"最有成就感/影响最大的人"选能**折射技术价值观**的例子（量化结果、工程审美、严谨验证）。

### Q27. 自动驾驶 HPC 的软素质问答（卓驭-二面 1-5、易控-二面 5/6、辉羲-一面 1、蔚来 HR/部门介绍、理想-一面 4/5、云-三面 1、文远-一面 1）

- **岗位需要的性格特质**（卓驭-二面 1）：严谨（数值正确性优先于速度——对拍文化）、数据驱动（一切优化有 benchmark）、耐心排查（性能问题定位链路长）、沟通（与算法/硬件团队三方接口）；
- **团队协作与技术难题**（易控-二面 5/6）：难题分析给方法论（复现→隔离→二分定位→最小可复现→修复→回归），协作给跨职能例子（算法同学要新算子，如何一起 trade-off 精度/速度）；
- **蔚来 HR/辉羲一面**：实习动机、时间保证、base 意向等常规题；蔚来-二面 4 的"部门业务介绍（训练框架方向）"是反问环节——准备好 2 个有技术含量的问题（团队的工具链形态、性能基线如何管理）；
- **理想-一面 4/5、云-三面 1**：实验室方向与项目综述用 Q25 框架压缩版；现场编程题保持基础题的手感（[93 篇](./93-t4-编程与算法.md)全量练习）。
