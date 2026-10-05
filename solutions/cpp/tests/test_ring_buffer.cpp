// 环形缓冲区测试: FIFO 语义/空满/并发生产消费
#include "answers/ring_buffer.hpp"
#include "answers/testing.h"

#include <thread>
#include <vector>

int main() {
    using answers::RingBuffer;
    {  // FIFO 与空满边界
        RingBuffer<int> rb(4);
        EXPECT(rb.empty() && !rb.full());
        int out = -1;
        EXPECT(!rb.pop(out));  // 空取失败
        EXPECT(rb.push(1) && rb.push(2) && rb.push(3) && rb.push(4));
        EXPECT(rb.full());
        EXPECT(!rb.push(5));   // 满写失败
        EXPECT(rb.pop(out) && out == 1);
        EXPECT(rb.pop(out) && out == 2);
        EXPECT(!rb.full());
        EXPECT(rb.push(5) && rb.push(6));  // 回绕写入
        EXPECT(rb.full());
        EXPECT(rb.pop(out) && out == 3);
        EXPECT(rb.pop(out) && out == 4);
        EXPECT(rb.pop(out) && out == 5);
        EXPECT(rb.pop(out) && out == 6);
        EXPECT(rb.empty());
    }
    {  // SPSC 单生产单消费 100k 条不丢不乱
        RingBuffer<int> rb(1024);
        constexpr int kN = 100000;
        std::thread producer([&] {
            for (int i = 0; i < kN;) {
                if (rb.push(i)) ++i;
            }
        });
        std::vector<int> got;
        got.reserve(kN);
        std::thread consumer([&] {
            int v;
            while ((int)got.size() < kN) {
                if (rb.pop(v)) got.push_back(v);
            }
        });
        producer.join();
        consumer.join();
        EXPECT((int)got.size() == kN);
        bool ordered = true;
        for (int i = 0; i < kN; ++i) {
            if (got[i] != i) { ordered = false; break; }
        }
        EXPECT(ordered);
    }
    RETURN_TEST_MAIN();
}
