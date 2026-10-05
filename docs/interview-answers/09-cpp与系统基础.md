---
title: "C++ 与系统基础"
description: "C++ 八股全集：虚函数/智能指针/移动语义/内存模型/并发，以及 GIL/IPC/coredump 排查等系统题"
pubDate: 2026-10-05
topic: "C++ 基础"
order: 9
tags: ["C++ 基础"]
---

覆盖题库：`AI-Infra-面经-1`（C++ 基础 1-21）、`AI-Infra-综合面经题库-6`（C++ 基础 14-20）、`AI-Infra-综合面经题库-5`（Q2/3）、`AI-Infra-一面`（6）。

**配套手写实现**：`solutions/cpp/include/answers/` 下有 smart_ptr.hpp（UniquePtr/SharedPtr）、singleton.hpp（Meyers/DCL）、lru_cache.hpp、memory_pool.hpp、ring_buffer.hpp，测试 9/9 通过（`ctest --test-dir solutions/cpp/build`）。

## Q1. 指针与引用（面-1 C1）

- 相同：都是间接访问另一对象。
- 不同：引用必须初始化且**不可重绑定**（无空引用、无"引用数组/引用的引用"）；指针可空、可重赋值、可算术；sizeof(指针)=8 而引用取目标大小；有多级指针无多级引用；引用底层通常仍是指针但语言层面无地址可取。
- 使用原则：可以为空/需要重指向用指针，其余默认引用（现代 C++ 传参：只读 `const T&`，转移 `T&&`）。

## Q2. static 的所有用法（面-1 C2）

1. **局部静态变量**：存 .data/.bss，首次执行初始化（C++11 线程安全），函数退出仍存活；
2. **全局/函数级 static 变量**：内部链接，仅本翻译单元可见（与普通全局的外部链接区别）；
3. **static 成员变量**：类级共享，类外定义（C++17 inline static 可类内初始化）；
4. **static 成员函数**：无 this，只能访问静态成员，可用类名调用。

## Q3. 智能指针与线程安全（面-1 C3、综合-6 C16）

- `unique_ptr`：独占，零开销，move-only；数组特化 `unique_ptr<T[]>`。
- `shared_ptr`：控制块（强/弱计数）+ 对象指针；拷贝 +1、析构 -1、归零释放；别名构造、`enable_shared_from_this`。
- `weak_ptr`：观察者，不增引用计数，`lock()` 提升解决循环引用（A↔B 缓存/观察者模式）。
- **线程安全粒度（高频陷阱）**：引用计数的增减是原子的（多个线程各持副本拷贝/析构安全）；但**同一个 shared_ptr 对象**被多线程读写（如一个线程 reset，另一个 operator=）仍需加锁；指向的对象本身并发访问也要外部同步。一句话："shared_ptr 的控制块线程安全，shared_ptr 对象与 T 对象不保证"。
- 手写版：`solutions/cpp/include/answers/smart_ptr.hpp`（atomic 引用计数 + 移动语义 + 数组特化），`tests/test_smart_ptr.cpp` 含 8 线程 ×1000 次拷贝的计数正确性测试。

## Q4. 右值引用 / move / forward（面-1 C4/5）

- 右值引用 `T&&` 绑定将亡值/纯右值，是**移动语义**与**完美转发**的语言基础。
- `std::move`：无条件 cast 成右值引用（不移动任何东西），让重载决议选择移动构造/赋值（窃取资源：指针接管 + 源置空）。
- `std::forward<T>`：**有条件** cast——模板参数 T 为左值引用时转回左值，否则保持右值；配合转发引用 `T&&`（模板推导语境）实现参数原样透传（`emplace_back` 的实现原理）。
- 区别一句话：move 面向"我要转移"，forward 面向"我替别人传"。

## Q5. 拷贝构造为何必须传引用（面-1 C6）

若传值：调用拷贝构造需要先拷贝实参 → 又调用拷贝构造 → 无限递归。语言直接禁止（传值声明不合法）。加 const：允许临时对象（右值）绑定 + 不改实参。

## Q6. new/malloc 区别与 new 底层（面-1 C7、综合-6 同族）

| | malloc | new |
|---|---|---|
| 类型 | 库函数 | 运算符（可重载） |
| 大小 | 手动算字节 | 编译器用 sizeof + 构造 |
| 构造 | 无 | 调构造函数；new[] 记录元素个数供 delete[] 析构 |
| 失败 | 返回 NULL | 抛 bad_alloc（nothrow 版返回空） |
| 释放 | free | delete（错配 delete/delete[] 是 UB） |

new 底层：`operator new`（默认 → malloc）→ placement new 构造。**new[] 会在头部多存 count**（实现定义），delete[] 读它逐个逆序析构——这就是"内存池支持 new[]/delete[]"考点，见 `solutions/cpp/include/answers/memory_pool.hpp`（placement new 逐构造 + 逆序析构，测试验证构造/析构次数）。

## Q7. C++ 内存模型 / 内存分区（面-1 C8、百度同族）

分区：**代码段(.text)、只读数据(.rodata)、已初始化数据(.data)、未初始化(.bss)、堆(malloc/new 向上增长)、栈(局部变量向下增长)、内存映射区(mmap: so/大 malloc)**。C++11 起"内存模型"还指**抽象机 + happens-before + memory_order**（原子/屏障语义），与 DCL 单例的 acquire/release 直接相关（见 Q18）。

## Q8. 虚构造/虚析构与多态底层（面-1 C9/10、综合-6 C14）

- **构造函数不能虚**：虚表由构造过程逐层建立，对象未成型没有 vptr 可用；语义上"构造时需知确切类型"。
- **析构要虚**：`Base* p = new Derived; delete p;` 若 ~Base 非虚 → 只调用基类析构 → 派生类资源泄漏。基类基本都应 `virtual ~Base()`。
- **实现机制**：每个多态类一张 **vtable**（虚函数指针数组 + RTTI/type_info），对象头部藏 **vptr** 指向所属类的 vtable；调用 `p->f()` → `(*(p->vptr)[slot])(p)`——两级间接实现动态分派。vtable 存在二进制的只读数据段，每个对象一份 vptr（8B 空间成本）。
- **内存布局**：单继承：Base 子对象在前，派生成员追加，vptr 共享；多继承：按声明序多个基类子对象（多 vptr），this 指针在跨基类调用时偏移调整（thunk）；虚继承：共享虚基类通过 **vbptr/vtable 偏移**间接定位，避免菱形重复。
- **模板多态**：编译期（CRTP/概念），零开销但类型膨胀；与虚函数运行期多态是"静态 vs 动态"两条路。
- **vtable 顺序**：按声明顺序排列，同层级覆盖同 slot（base 与 derived 的关系），多重继承多张表。

## Q9. CRTP 静态多态（综合-6 C18）

```cpp
template <typename D> struct Base {
    void interface() { static_cast<D*>(this)->impl(); }   // 编译期分派
};
struct Derived : Base<Derived> { void impl(); };
```

原理：基类模板注入派生类型，`static_cast<D*>` 在编译期解析到具体实现——无 vtable/vptr、可内联，把"运行期虚调用"变成"编译期实例化"。用途：静态接口（expression template、计数 mixin、仿真虚函数）；腾讯 TEG 考过"CRTP 减少虚函数开销"即此。

## Q10. 四种 cast（综合-6 C15）

- `static_cast`：相关类型间编译期可验证转换（数值、上行指针、显式构造），不检查运行期类型；
- `dynamic_cast`：多态类型安全下行/侧转，运行期查 RTTI，失败指针返回 nullptr（引用抛 bad_cast），有开销；
- `const_cast`：加减 const/volatile（唯一能去 const 的）；
- `reinterpret_cast`：位级重解释（指针↔整数、不相关指针），最危险。
- "父类指针转子类"用什么：必须 dynamic_cast（static_cast 不做运行期检查，父转子安全性无法静态保证）。

## Q11. 函数模板声明与定义分离（综合-6 C17）

**不能**放两个 .cpp 而不特殊处理：实例化发生在使用点，别的 TU 看不到定义 → 链接错误（除非编译器做隐式实例化外链）。解法：①头文件内含定义（主流）；②显式实例化声明 `extern template` + 一个 TU 里显式实例化所有用到的类型；③C++20 modules 缓解。类模板成员函数同理。

## Q12. vector 扩容 / resize vs reserve（综合-6 C19、面-1 C11）

- **扩容**：满了按 **2 倍**（MSVC）/1.5 倍（libstdc++ 新版）申请新内存 → 元素移动（move 若 noexcept，否则拷贝）→ 释放旧内存。**为什么 2 倍**：均摊 O(1) push_back（几何增长）；倍率 <φ≈1.618 时旧内存块可被复用（1.5 的理由），2 倍利用率/复用折中且实现简单。
- **reserve(n)**：只改 capacity（预分配，不构造元素，size 不变）；**resize(n)**：改 size（构造/析构到 n 个元素），可能引起扩容。迭代器/指针在扩容后全部失效（内存搬家）——高频追问。

## Q13. lambda（面-1 C12）

语法 `[捕获](参数) -> 返回类型 { 体 }`。捕获：`[]` 无、`[=]` 值捕获全部、`[&]` 引用、`[x, &y]` 混合、`[=, &x]`、`[this]`、C++14 初始化捕获 `[p=std::move(ptr)]`（唯一能移动捕获的方式）。实现：编译器生成匿名闭包类，捕获即成员，`operator()` 默认 const（`mutable` 去除）；无捕获 lambda 可转函数指针。悬垂引用：值捕获引用变量的生命周期陷阱。

## Q14. 锁 / 条件变量 / 死锁（面-1 C13/14/15）

- 锁家族：`std::mutex`（互斥）、`recursive_mutex`（同线程可重入）、`shared_mutex`（读写锁）、`spinlock`（自旋，临界区极短、多核）；`atomic` 无锁变量。
- **lock_guard vs unique_lock**：lock_guard 构造即锁、析构解锁、不可中途解锁（零开销、默认首选）；unique_lock 支持延迟锁/try_lock/unlock/移动/配合条件变量（wait 需要中途解锁能力）。
- **条件变量 vs 互斥锁**：mutex 保护临界区互斥；condition_variable 解决"等待某条件成立"——`wait(lock, pred)` 内部"解锁+睡眠+唤醒后重新加锁再查谓词"，避免忙等。必须配合 mutex 与谓词（防虚假唤醒）。
- **死锁四条件**：互斥、持有并等待、不可剥夺、循环等待。预防：按全局顺序加锁（破坏循环等待，最实用）、`std::scoped_lock(m1,m2)` 一次性原子获取、try_lock+退避、缩短临界区。

## Q15. 线程安全单例（面-1 C16、综合-6 C20）

三种正确写法（手写见 `solutions/cpp/include/answers/singleton.hpp`）：

1. **Meyers**：`static Singleton& inst(){ static Singleton s; return s; }` —— C++11 magic static 线程安全（首选）；
2. **DCL + atomic**：`load(acquire)`/`store(release)`——必须用原子与内存序，裸指针 DCL 因"new 与赋值重排"会被其他线程看到未构造完的对象（经典坑，北极雄芯原题）；
3. **call_once**：`std::call_once(flag, []{ ptr = new S; })`。

DCL 现代正确写法（工程版）：

```cpp
static T* inst() {
    T* p = ptr_.load(std::memory_order_acquire);
    if (!p) { std::lock_guard lk(m_); p = ptr_.load(std::memory_order_relaxed);
              if (!p) { p = new T(); ptr_.store(p, std::memory_order_release); } }
    return p;
}
```

## Q16. 限制对象只能在堆上创建（面-1 C17）

把析构函数（或构造）声明为 private/protected → 栈对象无法构造/析构 → 只能 `new`；再提供 `destroy()` 成员 `delete this`（或友元工厂函数）。反之"只能在栈上"= private operator new。

## Q17. 编译流程 / 动静态库 / C++ 调 C（面-1 C18/19）

- 流程：**预处理(#include/宏展开) → 编译(.s 汇编) → 汇编(.o 目标文件) → 链接(符号解析+重定位)**。对应 gcc 选项 -E/-S/-c。
- 静态库(.a)：链接期拷贝进可执行文件，体积大、部署独立、更新需重链；动态库(.so)：运行期加载映射，多进程共享一份代码、可热更新，但有符号查找与加载开销（ldd/依赖顺序、rpath 问题——小鹏考过依赖顺序：链接时被依赖的放后面）。
- C++ 调 C：`extern "C"` 包裹声明（抑制 name mangling）；C 头文件惯用 `#ifdef __cplusplus extern "C" {`；反向（C 调 C++）只能通过 extern "C" 包装的接口函数。

## Q18. 进程与线程 / 大小端（面-1 C20/21）

- 进程：资源分配单位（独立虚拟地址空间、fd 表）；线程：调度单位（共享地址空间/堆/全局变量，私有栈/寄存器/TLS）。切换成本：线程只换寄存器与栈（~us 内），进程要换页表/CR3（TLB 失效）。隔离性：进程崩不互相影响，线程崩进程崩。
- 大小端判断（至少两种）：①union 重解释（`union{int i; char c;}; i=1; c==1 → 小端`）；②指针重解释（`*(char*)&x == 1`）；③ `std::endian::native`（C++20）。网络字节序为大端（htons/ntohs）。

## Q19. GIL 与 IPC（综合-5 Q2/3）

**GIL**：CPython 解释器全局锁——同一时刻仅一个线程执行 Python 字节码。影响：CPU 密集多线程无法多核加速（应 multiprocessing / C 扩展释放 GIL / 换解释器）；IO 线程在阻塞调用时释放 GIL 所以多线程 IO 并发有效。历史原因：引用计数非线程安全的低成本保护。
**IPC 方式**：管道（pipe 带血缘、FIFO 命名）、消息队列、共享内存（shm/mmap，最快、需自配同步）、信号量、socket（本机/跨机）、信号、文件锁。选型：大数据量→共享内存；低频消息→队列/socket；跨机必 socket。

## Q20. 数组越界破坏其他结构 + coredump 排查（AI-Infra-一面 6）

场景：堆上越界写踩坏相邻分配的元数据/对象，崩溃点远离案发现场。方法：

1. **开启 core dump**（`ulimit -c unlimited`），gdb 加载：`gdb ./app core`；
2. `bt` 看崩溃栈 → `frame N` 到现场 → `info registers`/`x/32gx` 查看受害内存附近；越界写通常在**崩溃位置的内存附近不可见**——注意看"是谁的邻接内存"；
3. **addr2line** 定位行号（编译带 -g）；对堆结构破坏：glibc 的 `MALLOC_CHECK_=3` / `libefence` / **AddressSanitizer（首选，能直接报越界写位置）**；
4. 静态排查：崩溃对象附近的成员变量异常（指针值像浮点数等"被写花"特征）→ 推断写入内容的来源；
5. 防复现：container `.at()`/`gsl::span`、ASAN CI、watchpoint（`watch *addr` 硬件断点抓写入者）。
