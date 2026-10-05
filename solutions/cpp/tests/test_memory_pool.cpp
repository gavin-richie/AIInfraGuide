// 内存池测试: 分配/回收/复用/new[]-delete[] 语义
#include "answers/memory_pool.hpp"
#include "answers/testing.h"

#include <cstring>

struct POD {
    int a;
    float b;
};

struct CountCtor {
    static int ctors, dtors;
    CountCtor() { ++ctors; }
    ~CountCtor() { ++dtors; }
};
int CountCtor::ctors = 0;
int CountCtor::dtors = 0;

int main() {
    using answers::MemoryPool;
    using answers::PoolArray;

    {  // 基本分配/回收: 块复用(不增加 slab)
        MemoryPool pool(sizeof(POD), 16);
        std::vector<void *> ps;
        for (int i = 0; i < 16; ++i) ps.push_back(pool.allocate());
        size_t slabs = pool.slab_count();
        EXPECT(slabs == 1);
        for (void *p : ps) pool.deallocate(p);
        void *q1 = pool.allocate();
        void *q2 = pool.allocate();
        EXPECT(pool.slab_count() == 1);          // 全部复用, 没有新 slab
        EXPECT(q1 != q2);
        pool.deallocate(q1);
        pool.deallocate(q2);
    }
    {  // 超出 slab 自动扩容
        MemoryPool pool(sizeof(POD), 8);
        std::vector<void *> ps;
        for (int i = 0; i < 20; ++i) ps.push_back(pool.allocate());
        EXPECT(pool.slab_count() == 3);  // 8 + 8 + 8 ≥ 20
        for (void *p : ps) pool.deallocate(p);
    }
    {  // new[]/delete[] 语义: n 次构造 n 次析构
        CountCtor::ctors = CountCtor::dtors = 0;
        MemoryPool pool(sizeof(CountCtor), 64);
        {
            PoolArray<CountCtor> arr(pool, 10);
            EXPECT(CountCtor::ctors == 10);
            EXPECT(CountCtor::dtors == 0);
        }  // 作用域结束 → destroy()
        EXPECT(CountCtor::dtors == 10);
    }
    {  // 写入数据可读回(内存确实可用)
        MemoryPool pool(sizeof(POD), 4);
        auto *p = static_cast<POD *>(pool.allocate());
        p->a = 42;
        p->b = 3.14f;
        EXPECT(p->a == 42);
        EXPECT_NEAR(p->b, 3.14f, 1e-6);
        pool.deallocate(p);
    }
    RETURN_TEST_MAIN();
}
