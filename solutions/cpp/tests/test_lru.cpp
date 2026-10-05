// LRU Cache 测试(LeetCode 146 标准用例 + 边界)
#include "answers/lru_cache.hpp"
#include "answers/testing.h"

int main() {
    using answers::LRUCache;
    {
        LRUCache cache(2);
        cache.put(1, 1);
        cache.put(2, 2);
        EXPECT(cache.get(1) == 1);   // 访问 1 → 2 最旧
        cache.put(3, 3);             // 淘汰 2
        EXPECT(cache.get(2) == -1);
        cache.put(4, 4);             // 淘汰 1
        EXPECT(cache.get(1) == -1);
        EXPECT(cache.get(3) == 3);
        EXPECT(cache.get(4) == 4);
        EXPECT(cache.size() == 2);
    }
    {  // 更新已存在 key 不触发淘汰
        LRUCache cache(2);
        cache.put(1, 1);
        cache.put(2, 2);
        cache.put(1, 10);            // 更新 1 且提升为最新
        EXPECT(cache.get(1) == 10);
        cache.put(3, 3);             // 淘汰 2
        EXPECT(cache.get(2) == -1);
        EXPECT(cache.get(1) == 10);
    }
    {  // 容量 1 反复淘汰
        LRUCache cache(1);
        cache.put(1, 1);
        cache.put(2, 2);
        EXPECT(cache.get(1) == -1);
        EXPECT(cache.get(2) == 2);
        auto keys = cache.keys_mru_to_lru();
        EXPECT(keys.size() == 1 && keys[0] == 2);
    }
    {  // 0 容量抛异常
        bool threw = false;
        try { LRUCache bad(0); } catch (const std::invalid_argument &) { threw = true; }
        EXPECT(threw);
    }
    RETURN_TEST_MAIN();
}
