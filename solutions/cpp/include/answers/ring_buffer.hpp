// 环形缓冲区(循环队列) —— 蔚来-一面-2 "C++ 循环缓冲区"
//
// 生产者/消费者语义:
//   - 环形索引 head(读) / tail(写), 空满判断用 (write_idx - read_idx)
//     的无符号回绕性质: size = (size_t)(tail_ - head_)
//   - 单生产者单消费者(SPSC)时 head/tail 各自只被一个线程写,
//     配合 acquire/release 可以做到无锁(这里给出 mutex 简版,
//     无锁版思路见 docs/interview-answers/10-cpp-编程题.md)
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace answers {

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity) : buf_(capacity + 1) {}  // 留一格区分空满

    bool push(const T &v) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (full()) return false;
        buf_[mask(tail_)] = v;
        ++tail_;
        return true;
    }

    bool pop(T &out) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (empty()) return false;
        out = buf_[mask(head_)];
        ++head_;
        return true;
    }

    bool empty() const { return head_ == tail_; }
    bool full() const { return size() == capacity(); }
    size_t size() const { return static_cast<size_t>(tail_ - head_); }  // 回绕安全
    size_t capacity() const { return buf_.size() - 1; }

private:
    size_t mask(uint64_t idx) const { return idx % buf_.size(); }

    std::vector<T> buf_;
    uint64_t head_ = 0;  // 读位置(单调递增, 取模用)
    uint64_t tail_ = 0;  // 写位置
    mutable std::mutex mtx_;
};

}  // namespace answers
