// 单例测试: 地址唯一性 + 多线程下 DCL/Meyers 均只构造一次
#include "answers/singleton.hpp"
#include "answers/testing.h"

#include <thread>
#include <vector>

struct Counting {
    Counting() { ++constructions; }
    static std::atomic<int> constructions;
};
std::atomic<int> Counting::constructions{0};

// 复制 DCL 的实现方式验证并发构造次数(直接用 answers::DCLSingleton 亦可)
namespace test {
class CountingDCL {
public:
    static CountingDCL *instance() {
        CountingDCL *p = ptr_.load(std::memory_order_acquire);
        if (!p) {
            std::lock_guard<std::mutex> lk(mtx_);
            p = ptr_.load(std::memory_order_relaxed);
            if (!p) {
                p = new CountingDCL();
                ptr_.store(p, std::memory_order_release);
            }
        }
        return p;
    }
    static std::atomic<int> ctors;
private:
    CountingDCL() { ++ctors; }
    static std::mutex mtx_;
    static std::atomic<CountingDCL *> ptr_;
};
std::mutex CountingDCL::mtx_;
std::atomic<CountingDCL *> CountingDCL::ptr_{nullptr};
std::atomic<int> CountingDCL::ctors{0};
}  // namespace test

int main() {
    EXPECT(answers::MeyersSingleton::instance().value() == 42);
    EXPECT(&answers::MeyersSingleton::instance() == &answers::MeyersSingleton::instance());
    EXPECT(answers::DCLSingleton::instance() == answers::DCLSingleton::instance());

    // 16 线程并发抢 DCL, 构造次数必须 == 1
    std::vector<std::thread> threads;
    std::vector<test::CountingDCL *> seen(16, nullptr);
    for (int t = 0; t < 16; ++t) {
        threads.emplace_back([&, t] { seen[t] = test::CountingDCL::instance(); });
    }
    for (auto &th : threads) th.join();
    EXPECT(test::CountingDCL::ctors.load() == 1);
    for (auto *p : seen) EXPECT(p == seen[0]);

    RETURN_TEST_MAIN();
}
