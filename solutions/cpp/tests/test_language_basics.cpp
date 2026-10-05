// 语言基础片段测试 —— 各批次文档内嵌 C++ 片段的工程化对拍
// 覆盖题库: 22-字节 Q3(内存对齐)/Q19(vector 释放内存)、40-腾讯 Q38(编译期计算)、
// 80-t3 #30(迭代器安全删除)、82-t3 Q3(unique_ptr 管理连续内存)、
// 102-t5 Q28(完美转发)。
#include "answers/testing.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// ---------------- 22-字节 Q3: 内存对齐 ----------------
// struct A { char; int; short; } = 1 + 3pad + 4 + 2 + 2pad = 12, 不是 7;
// 成员按对齐数降序排(struct B)更省 = 8。
struct A {
    char c;
    int i;
    short s;
};
struct B {
    int i;
    short s;
    char c;
};
static_assert(sizeof(A) == 12, "A 应为 12(尾部 pad 到 4 的倍数)");
static_assert(sizeof(B) == 8, "B 应为 8(降序排列免尾部 pad)");
static_assert(alignof(A) == 4 && alignof(B) == 4, "最大成员对齐数 = struct 对齐数");

// ---------------- 40-腾讯 Q38: 编译期计算 ----------------
// constexpr 递归建 CRC32 查表 + constexpr 平方, 全部编译期完成。
// (consteval "只能编译期"是 C++20 关键字, C++17 用 constexpr + static_assert 等效约束)
constexpr uint32_t crc_row(uint32_t c) {
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    return c;
}
constexpr uint32_t crc_table_at(int i) { return crc_row((uint32_t)i); }
static_assert(crc_table_at(1) == 0x77073096u, "CRC32 表首项(标准多项式)");
static_assert(crc_table_at(255) == 0x2D02EF8Du, "CRC32 表末项");
constexpr int sq(int x) { return x * x; }
static_assert(sq(4) == 16);

// ---------------- 102-t5 Q28: 完美转发 ----------------
// 万能引用 + std::forward: 按推导结果还原左右值, 工厂不改变值类别。
struct Tracer {
    static int copies, moves;
    Tracer() = default;
    Tracer(const Tracer &) { ++copies; }
    Tracer(Tracer &&) noexcept { ++moves; }
};
int Tracer::copies = 0;
int Tracer::moves = 0;

template <typename T, typename... Args>
std::unique_ptr<T> make(Args &&...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// ---------------- 82-t3 Q3: unique_ptr 管理连续内存 ----------------
// 数组特化走 delete[]; shared_ptr 需自定义 deleter; 自定义 deleter 计数调用。
struct DeleterCounter {
    int *count;
    void operator()(float *p) const {
        ++*count;
        delete[] p;
    }
};

int main() {
    {  // 内存对齐(编译期已 static_assert, 运行期再验地址对齐)
        A a{};
        B b{};
        EXPECT(reinterpret_cast<uintptr_t>(&a) % alignof(A) == 0);
        EXPECT(reinterpret_cast<uintptr_t>(&b) % alignof(B) == 0);
        EXPECT(sizeof(a) == 12 && sizeof(b) == 8);
    }
    {  // vector 释放内存: swap 技巧 / shrink_to_fit / move
        std::vector<int> v;
        v.reserve(1000);
        v.insert(v.end(), 10, 1);
        EXPECT(v.capacity() >= 1000);

        std::vector<int>().swap(v);       // 空 vector 与 v 交换, 内存随临时对象析构
        EXPECT(v.capacity() == 0);
        EXPECT(v.empty());

        std::vector<int> w;
        w.reserve(512);
        w.insert(w.end(), 8, 1);
        w.shrink_to_fit();                // 请求收缩到 size(libstdc++/libc++ 均生效)
        EXPECT(w.capacity() <= (size_t)8);

        std::vector<int> u;
        u.reserve(256);
        u.insert(u.end(), 5, 1);
        std::vector<int> moved = std::move(u);  // u 进入有效但未指定状态(通常为空)
        EXPECT(moved.size() == 5 && moved.capacity() >= 256);
        EXPECT(u.empty());                // libstdc++/libc++ move 后源为空
    }
    {  // 迭代器遍历中删除: erase 返回下一个有效迭代器
        std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        for (auto it = v.begin(); it != v.end();) {
            if (*it % 2 == 0) it = v.erase(it);
            else ++it;
        }
        std::vector<int> want{1, 3, 5, 7, 9};
        EXPECT(v == want);

        // 等价的erase-remove 惯用法(遍历删除的算法化写法)
        std::vector<int> w{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        w.erase(std::remove_if(w.begin(), w.end(),
                               [](int x) { return x % 2 == 0; }),
                w.end());
        EXPECT(w == want);
    }
    {  // 完美转发: 左值 -> 拷贝, 右值 -> 移动, 工厂零拷贝惩罚
        Tracer lvalue;
        Tracer::copies = Tracer::moves = 0;
        auto p1 = make<Tracer>(lvalue);
        EXPECT(Tracer::copies == 1 && Tracer::moves == 0);   // 左值被拷贝
        Tracer::copies = Tracer::moves = 0;
        auto p2 = make<Tracer>(Tracer{});
        EXPECT(Tracer::copies == 0 && Tracer::moves == 1);   // 右值被移动
        (void)p1;
        (void)p2;
    }
    {  // unique_ptr 数组 + 自定义 deleter
        auto p = std::make_unique<float[]>(1024);  // 值初始化为 0
        EXPECT(p[0] == 0.f && p[1023] == 0.f);
        p[7] = 3.14f;
        EXPECT(p[7] == 3.14f);

        int calls = 0;
        {
            std::shared_ptr<float[]> q(new float[64], DeleterCounter{&calls});
            q[0] = 1.f;  // C++17 起数组形式支持 operator[]
            EXPECT(q.use_count() == 1);
        }  // 离开作用域, deleter 被调
        EXPECT(calls == 1);
    }
    RETURN_TEST_MAIN();
}
