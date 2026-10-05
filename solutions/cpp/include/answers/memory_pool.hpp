// 内存池 —— 综合面经题库-1 算法题 1(支持 new Foo[] / delete[] 语义)、
//          快手-一面-2(Memory Pool 设计)、北极雄芯(malloc 底层)
//
// 固定块大小 + 自由链表(free list)内存池:
//   - 预申请大块 Slab(避免频繁 malloc 系统调用)
//   - 空闲块串成侵入式链表: free_head -> free block -> ...
//   - allocate: 弹链表头, O(1); deallocate: 压回链表头, O(1)
//   - "new Foo[]/delete[]" 语义: PoolArray 按 n*sizeof(T) 从池里取连续内存,
//     用 placement new 逐个构造 / 显式逐个析构 —— 正是 new[]/delete[] 的底层行为,
//     也是该面试题想考的点: new[] 会多存元素个数以便 delete[] 知道析构几次
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace answers {

class MemoryPool {
public:
    MemoryPool(size_t block_size, size_t blocks_per_slab = 1024)
        : block_size_(align_up(std::max(block_size, sizeof(void *)))), slab_blocks_(blocks_per_slab) {}

    ~MemoryPool() {
        for (void *slab : slabs_) ::operator delete(slab);
    }

    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;

    void *allocate() {
        if (!free_head_ && !grow()) throw std::bad_alloc();
        void *p = free_head_;
        free_head_ = *static_cast<void **>(p);  // 块头本身就是 next 指针
        return p;
    }

    void deallocate(void *p) {
        if (!p) return;
        *static_cast<void **>(p) = free_head_;  // 压回链表头
        free_head_ = p;
    }

    size_t block_size() const { return block_size_; }
    size_t slab_count() const { return slabs_.size(); }

private:
    struct FreeNode {
        FreeNode *next;
    };

    static size_t align_up(size_t n, size_t align = sizeof(void *)) {
        return (n + align - 1) & ~(align - 1);
    }

    bool grow() {  // 新 Slab 切成块挂上自由链表; 返回 false 表示分配失败
        size_t bytes = block_size_ * slab_blocks_;
        void *slab = ::operator new(bytes, std::nothrow);
        if (!slab) return false;
        slabs_.push_back(slab);
        char *base = static_cast<char *>(slab);
        free_head_ = base;  // 倒序串联也行, 正序串更直观
        for (size_t i = 0; i + 1 < slab_blocks_; ++i) {
            char *cur = base + i * block_size_;
            *reinterpret_cast<char **>(cur) = cur + block_size_;
        }
        *reinterpret_cast<char **>(base + (slab_blocks_ - 1) * block_size_) = nullptr;
        return true;
    }

    size_t block_size_;
    size_t slab_blocks_;
    void *free_head_ = nullptr;
    std::vector<void *> slabs_;
};

// ---------------- new T[] / delete[] 语义封装 ----------------
template <typename T>
class PoolArray {
public:
    static_assert(std::is_trivially_copyable_v<T> || true, "");

    PoolArray(MemoryPool &pool, size_t n) : pool_(pool), n_(n) {
        // 从池里拿 n 个"连续"块: 固定块池不保证跨块连续, 这里要求
        // block_size >= sizeof(T) 且逐块 allocate, 演示用; 生产实现会用
        // 多尺寸池或按 n*sizeof(T) 找合适尺寸类(伙伴系统思路)。
        chunks_.reserve(n);
        for (size_t i = 0; i < n; ++i) chunks_.push_back(static_cast<T *>(pool_.allocate()));
        for (size_t i = 0; i < n; ++i) new (chunks_[i]) T{};  // placement new 逐个构造
    }

    ~PoolArray() { destroy(); }

    PoolArray(const PoolArray &) = delete;
    PoolArray &operator=(const PoolArray &) = delete;

    T &operator[](size_t i) { return *chunks_[i]; }
    const T &operator[](size_t i) const { return *chunks_[i]; }
    size_t size() const { return n_; }

    // 等价 delete[]: 逆序显式析构(与构造相反) + 归还内存
    void destroy() {
        if (destroyed_) return;
        destroyed_ = true;
        for (size_t i = n_; i > 0; --i) chunks_[i - 1]->~T();
        for (T *p : chunks_) pool_.deallocate(p);
    }

private:
    MemoryPool &pool_;
    size_t n_;
    std::vector<T *> chunks_;
    bool destroyed_ = false;
};

}  // namespace answers
