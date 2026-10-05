// 单例模式 —— 综合-6 C++ 编程题、北极雄芯(DCL 隐患)、三星
//
// 三个版本按推荐顺序:
//   1. Meyers 单例: C++11 起函数局部 static 初始化由编译器保证线程安全
//      (魔法静态 magic statics), 首选答案。
//   2. DCL(双检锁): 面试必问的坑 —— 指针必须用 std::atomic +
//      acquire/release 内存序, 裸 new 无内存屏障会因指令重排导致
//      "对象地址已写入、构造未完成" 被其他线程读到。
//   3. call_once: 等价 std::once_flag 方案。
#pragma once

#include <atomic>
#include <mutex>

namespace answers {

// ---------------- Meyers Singleton(推荐) ----------------
class MeyersSingleton {
public:
    MeyersSingleton(const MeyersSingleton &) = delete;
    MeyersSingleton &operator=(const MeyersSingleton &) = delete;

    static MeyersSingleton &instance() {
        static MeyersSingleton inst;  // C++11 线程安全首次初始化
        return inst;
    }
    int value() const { return v_; }

private:
    MeyersSingleton() = default;
    int v_ = 42;
};

// ---------------- DCL Singleton(讲清内存序) ----------------
class DCLSingleton {
public:
    DCLSingleton(const DCLSingleton &) = delete;
    DCLSingleton &operator=(const DCLSingleton &) = delete;

    static DCLSingleton *instance() {
        DCLSingleton *p = ptr_.load(std::memory_order_acquire);  // 与 release 配对
        if (p == nullptr) {
            std::lock_guard<std::mutex> lock(mtx_);
            p = ptr_.load(std::memory_order_relaxed);            // 双检第二查
            if (p == nullptr) {
                p = new DCLSingleton();
                // release: 保证上面 new 的写(构造)先于指针发布对其他线程可见
                ptr_.store(p, std::memory_order_release);
            }
        }
        return p;
    }

private:
    DCLSingleton() = default;
    static std::mutex mtx_;
    static std::atomic<DCLSingleton *> ptr_;
};

// C++11 起 inline 变量(或类外定义)避免链接重复定义
inline std::mutex DCLSingleton::mtx_;
inline std::atomic<DCLSingleton *> DCLSingleton::ptr_{nullptr};

}  // namespace answers
