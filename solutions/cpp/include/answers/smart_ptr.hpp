// 智能指针手写 —— 字节-实习-一面-4、百度-实习-一面-1、小米、小厂
//
// UniquePtr: 独占所有权, 移动语义, 零开销抽象(无引用计数)
// SharedPtr: 控制块引用计数(atomic 线程安全), 拷贝 +1 / 析构 -1 / 归零释放
//
// 面试要点:
// - unique_ptr 禁用拷贝(=delete), 只允许移动; 移动后源指针置空
// - shared_ptr 引用计数增减必须原子操作(多线程拷贝安全);
//   但所指对象本身的并发读写仍需外部同步 —— "引用计数线程安全 ≠ 对象线程安全"
// - 控制块与对象分离, 才能支持别名构造/weak_ptr(此处简化实现)
#pragma once

#include <atomic>
#include <cstddef>
#include <utility>

namespace answers {

template <typename T>
class UniquePtr {
public:
    UniquePtr() = default;
    explicit UniquePtr(T *p) : ptr_(p) {}
    ~UniquePtr() { delete ptr_; }

    // 禁拷贝
    UniquePtr(const UniquePtr &) = delete;
    UniquePtr &operator=(const UniquePtr &) = delete;

    // 允许移动: 转移所有权, 源置空
    UniquePtr(UniquePtr &&o) noexcept : ptr_(o.release()) {}
    UniquePtr &operator=(UniquePtr &&o) noexcept {
        if (this != &o) reset(o.release());
        return *this;
    }

    T *get() const { return ptr_; }
    T &operator*() const { return *ptr_; }
    T *operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

    T *release() {  // 放弃所有权(不 delete)
        T *p = ptr_;
        ptr_ = nullptr;
        return p;
    }
    void reset(T *p = nullptr) {
        T *old = ptr_;
        ptr_ = p;
        delete old;
    }

private:
    T *ptr_ = nullptr;
};

// 数组特化: delete[] 而不是 delete(误配是 UB)
template <typename T>
class UniquePtr<T[]> {
public:
    UniquePtr() = default;
    explicit UniquePtr(T *p) : ptr_(p) {}
    ~UniquePtr() { delete[] ptr_; }
    UniquePtr(const UniquePtr &) = delete;
    UniquePtr &operator=(const UniquePtr &) = delete;
    UniquePtr(UniquePtr &&o) noexcept : ptr_(o.release()) {}
    UniquePtr &operator=(UniquePtr &&o) noexcept {
        if (this != &o) reset(o.release());
        return *this;
    }
    T &operator[](size_t i) const { return ptr_[i]; }
    T *get() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    T *release() {
        T *p = ptr_;
        ptr_ = nullptr;
        return p;
    }
    void reset(T *p = nullptr) {
        T *old = ptr_;
        ptr_ = p;
        delete[] old;
    }

private:
    T *ptr_ = nullptr;
};

// ------------------------------------------------------------ SharedPtr
struct ControlBlock {
    std::atomic<long> refs{1};
    virtual ~ControlBlock() = default;  // 多态析构(支持定制删除器思路)
};

template <typename T>
class SharedPtr {
public:
    SharedPtr() = default;
    explicit SharedPtr(T *p) : ptr_(p), ctrl_(p ? new DefaultCB(p) : nullptr) {}

    SharedPtr(const SharedPtr &o) : ptr_(o.ptr_), ctrl_(o.ctrl_) {
        if (ctrl_) ctrl_->refs.fetch_add(1, std::memory_order_relaxed);
    }
    SharedPtr(SharedPtr &&o) noexcept : ptr_(o.ptr_), ctrl_(o.ctrl_) {
        o.ptr_ = nullptr;
        o.ctrl_ = nullptr;
    }
    SharedPtr &operator=(const SharedPtr &o) {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            ctrl_ = o.ctrl_;
            if (ctrl_) ctrl_->refs.fetch_add(1, std::memory_order_relaxed);
        }
        return *this;
    }
    SharedPtr &operator=(SharedPtr &&o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            ctrl_ = o.ctrl_;
            o.ptr_ = nullptr;
            o.ctrl_ = nullptr;
        }
        return *this;
    }
    ~SharedPtr() { release(); }

    T *get() const { return ptr_; }
    T &operator*() const { return *ptr_; }
    T *operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    long use_count() const { return ctrl_ ? ctrl_->refs.load(std::memory_order_relaxed) : 0; }

private:
    struct DefaultCB : ControlBlock {
        explicit DefaultCB(T *p) : obj(p) {}
        ~DefaultCB() override { delete obj; }
        T *obj;
    };

    void release() {
        if (ctrl_ && ctrl_->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete ctrl_;  // 控制块析构时顺带 delete 对象(引用归零)
        }
        ptr_ = nullptr;
        ctrl_ = nullptr;
    }

    T *ptr_ = nullptr;
    ControlBlock *ctrl_ = nullptr;
};

}  // namespace answers
