---
title: "C++ 编程题"
description: "LRU/内存池/单例/智能指针手写、NMS/Conv2D/双线性插值/LayerNorm 等 C++ 手写题，附 CMake 工程与测试"
pubDate: 2026-10-05
topic: "C++ 编程"
order: 10
tags: ["C++ 基础", "编程题"]
---

覆盖题库：`AI-Infra-综合面经题库-6`（C++ 编程题 1-5）、`AI-Infra-综合面经题库-1`（算法 1/2）、`AI-Infra-面经-1`（编程题 4）。

**配套工程** [solutions/cpp](/solutions/cpp)（C++17 + CMake + 自研轻量断言），9 个测试全绿：

```bash
cd solutions/cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build      # 9/9 passed
```

## Q1. LRU 缓存（综合-1 算法 3，字节/阿里/阶跃同题）

**结构**：`unordered_map<key, list::iterator>` + `list<pair<K,V>>`，get/put 均 O(1)。

```cpp
int get(int key) {
    auto it = map_.find(key);
    if (it == map_.end()) return -1;
    items_.splice(items_.begin(), items_, it->second);  // O(1) 移到表头, 迭代器不失效
    return it->second->second;
}
void put(int key, int value) {
    if (map_.count(key)) { /* 更新 + splice 头部 */ return; }
    if (map_.size() == cap_) { map_.erase(items_.back().first); items_.pop_back(); }
    items_.emplace_front(key, value);
    map_[key] = items_.begin();
}
```

要点：`splice` 不拷贝节点、迭代器不失效；淘汰取 `back()`。**代码**：`include/answers/lru_cache.hpp`，测试含 LeetCode 146 全用例 + 容量 1 + 0 容量异常。Python 双向链表手写版与 OrderedDict 版对照：`solutions/python/src/answers/algorithms/lru_cache.py`。

## Q2. NMS 非极大值抑制（综合-6 C++ 编程 1、面-1 编程 4、蔚来）

流程：按 score 降序 → 取最大保留 → 抑制与其 IoU>thr 的框 → 循环。

```cpp
std::vector<int> nms(boxes, scores, thr) {
    order = argsort(scores, desc);
    keep = [];
    while (!order.empty()) {
        top = order.front(); keep.push_back(top);
        删除 order 中 iou(boxes[top], boxes[j]) > thr 的 j;
    }
}
```

**代码**：`solutions/cpp/src/vision.cpp`（iou + nms），测试：4 框标准用例、0.68 IoU 抑制、低分不抑制高分（顺序独立性）。GPU/Python 版见 `solutions/cuda/src/pool_iou.cu`、`solutions/python/src/answers/torch_impl/nms.py`（手写 + 向量化两版）。**改进方案**（面-1 DL12 追问）：Soft-NMS（高 IoU 框分数衰减而非删除，适应密集遮挡）、IoU 阈值随类别/分数自适应、cluster-NMS、DIoU/CIoU-NMS（把中心距/长宽比纳入抑制度量）。

## Q3. Conv2D（综合-6 C++ 编程 2、面-1 编程 1）

直接卷积实现 + 手算对拍（差分核/全 1 核/恒等核三种验证），**代码**：`solutions/cpp/src/vision.cpp::conv2d`，输出尺寸 `OH=(H-K)/S+1`；复杂度 $O(OH \cdot OW \cdot C_{in} \cdot C_{out} \cdot K^2)$。工程优化路线（面试口述）：im2col+GEMM、Winograd、FFT、implicit GEMM on Tensor Core（详见《CUDA 编程题》Q14 与《GEMM 与算子优化》Q14）。

## Q4. 双线性插值（综合-6 C++ 编程 3）

$$f(x,y) \approx (1-dy)\big[(1-dx)f_{00} + dx f_{01}\big] + dy\big[(1-dx)f_{10} + dx f_{11}\big]$$

先水平后垂直两次线性；越界 clamp/补零策略要对齐目标语义。**代码**：`solutions/cpp/src/vision.cpp::bilinear_sample`，测试：整点还原、中心=四点均值、越界 clamp。

## Q5. LayerNorm（综合-6 C++ 编程 4）

$$y = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} \cdot w + b,\quad \mu=\frac{1}{n}\sum x,\ \sigma^2=\frac{1}{n}\sum (x-\mu)^2$$

实现用 **Welford 单遍算法**（增量 mean/M2，比两遍少一次扫描且数值更稳）。**代码**：`solutions/cpp/src/layernorm.cpp`，测试验证归一后均值 0/方差 1 与手算 affine。CUDA 版在 `solutions/cuda/src/norm.cu`（与 PyTorch F.layer_norm 对拍）。**RMSNorm 追问**：去掉 μ 与减法，$y = x/\sqrt{\text{mean}(x^2)+\epsilon} \cdot w$——少一次归约与减法，快在访存与计算双省（Llama 系标配）。

## Q6. 单例模式实现（综合-6 C++ 编程 5）

Meyers + DCL（acquire/release）+ call_once 三版与并发验证：**代码** `solutions/cpp/include/answers/singleton.hpp`，`tests/test_singleton.cpp` 用 16 线程并发抢 DCL，断言构造次数 == 1、地址一致。原理详解见《C++ 与系统基础》Q15。

## Q7. 内存池：new Foo[] / delete[]（综合-1 算法 1、快手 Memory Pool）

**固定块 + 自由链表**：

- `grow()`：按 slab 一次 `::operator new` 大块，切成 block_size 的块串成侵入式链表（空闲块头即 next 指针）；
- `allocate`：弹链头 O(1)；`deallocate`：压回 O(1)；
- **new[]/delete[] 语义**：`PoolArray` 逐块取内存 → **placement new 逐个构造** → 析构时**逆序显式析构**再归还——正是编译器 new[]/delete[] 的底层行为（new[] 头部记录元素个数，delete[] 据此知道析构多少次）。

**代码**：`include/answers/memory_pool.hpp`，测试：块复用不加 slab、20 块跨 3 个 slab、10 次构造 10 次析构。malloc 底层追问（北极雄芯）：glibc malloc = bins（fast/small/large）+ brk/mmap 阈值 + 合并；内存池的意义是消除"每次系统调用 + 元数据"开销并提高局部性。

## Q8. float 相等比较（综合-1 算法 2）

```cpp
bool nearly_equal(float a, float b,
                  float rel = 1e-6f, float abs_ = 1e-9f) {
    float diff = std::fabs(a - b);
    return diff <= abs_                      // 绝对误差: 接近 0 时
        || diff <= rel * std::fmax(std::fabs(a), std::fabs(b));  // 相对误差
}
```

原理：浮点误差随量级缩放，纯 `==`/固定 epsilon 都错（大数下溢出精度、小数下绝对误差过松）。对照 Python 实现：`solutions/python/src/answers/algorithms/math_bits.py::float_equal`。工程进阶：ULP 距离（`std::nextafter` 数格数）、NaN 特判（`x!=x`）。

## Q9. 智能指针手写（字节/百度/小米）

UniquePtr（移动语义/数组特化 release/reset）+ SharedPtr（atomic 控制块），**代码**：`include/answers/smart_ptr.hpp`，测试含 8 线程计数原子性验证。设计细节见《C++ 与系统基础》Q3。

## Q10. 循环缓冲区（蔚来-一面-2）

环形队列（+1 格区分空满、`(size_t)(tail-head)` 回绕安全求 size、mutex 版），**代码**：`include/answers/ring_buffer.hpp`，测试含 SPSC 100k 条不丢不乱。无锁进阶：SPSC 用 head/tail 各自原子 + acquire/release，单生产单消费免锁（乐观读序）。

## Q11. 实现计算图（面-1 编程 2）

节点=算子（value/grad/inputs），边=张量依赖；核心是**拓扑排序驱动执行**：

```cpp
class Node { vector<Node*> inputs; std::function<Tensor(vector<Tensor>)> fn;
             Tensor value, grad; };
vector<Node*> topo_sort(nodes);            // Kahn/DFS, 见《算法题》篇
for (auto* n : order) n->value = n->fn(gather(n->inputs));
// 反向: 逆拓扑序, 链式法则累加 grad
```

要点：缓存前向值（反向要用）、按需反传（从 loss 起）、显式 `backward` 由输出梯度乘局部导数累加到输入 grad。
