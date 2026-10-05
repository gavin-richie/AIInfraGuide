// 智能指针测试: 所有权转移/引用计数/数组特化/跨线程计数
#include "answers/smart_ptr.hpp"
#include "answers/testing.h"

#include <atomic>
#include <thread>
#include <vector>

struct Tracked {
    static std::atomic<int> alive;
    Tracked() { ++alive; }
    ~Tracked() { --alive; }
};
std::atomic<int> Tracked::alive{0};

int main() {
    using answers::UniquePtr;
    using answers::SharedPtr;

    {  // UniquePtr 基本语义
        auto p = UniquePtr<Tracked>(new Tracked);
        EXPECT(Tracked::alive.load() == 1);
        EXPECT(p.get() != nullptr);
        EXPECT(static_cast<bool>(p));
        p.reset(new Tracked);
        EXPECT(Tracked::alive.load() == 1);  // 旧的被析构
        UniquePtr<Tracked> q = std::move(p); // 移动
        EXPECT(p.get() == nullptr && !static_cast<bool>(p));
        EXPECT(q.get() != nullptr);
        Tracked *raw = q.release();          // 放弃所有权
        EXPECT(q.get() == nullptr);
        delete raw;
        EXPECT(Tracked::alive.load() == 0);
    }
    {  // 数组特化
        auto arr = UniquePtr<int[]>(new int[8]);
        for (int i = 0; i < 8; ++i) arr[i] = i * i;
        EXPECT(arr[7] == 49);
    }
    {  // SharedPtr 引用计数
        {
            auto sp1 = SharedPtr<Tracked>(new Tracked);
            EXPECT(sp1.use_count() == 1);
            {
                auto sp2 = sp1;  // 拷贝 +1
                EXPECT(sp1.use_count() == 2);
                auto sp3 = sp2;
                EXPECT(sp1.use_count() == 3);
            }                     // sp2/sp3 析构 -2
            EXPECT(sp1.use_count() == 1);
            EXPECT(Tracked::alive.load() == 1);
        }
        EXPECT(Tracked::alive.load() == 0);  // 最后一个释放时对象析构
    }
    {  // SharedPtr 跨线程拷贝安全(计数原子性)
        auto sp = SharedPtr<Tracked>(new Tracked);
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&sp] {
                for (int i = 0; i < 1000; ++i) {
                    auto local = sp;  // 拷贝 +1, 析构 -1
                }
            });
        }
        for (auto &th : threads) th.join();
        EXPECT(sp.use_count() == 1);
        EXPECT(Tracked::alive.load() == 1);
    }
    RETURN_TEST_MAIN();
}
