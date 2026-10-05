// LRU Cache —— 全库出现频率最高的 C++ 手写题
// (综合面经题库-1、字节-AML、阿里-一面、阶跃星辰、腾讯 TEG ...)
//
// O(1) 的标准结构: std::unordered_map<key, list::iterator> + std::list 双向链表
// - 链表头部 = 最近使用, 尾部 = 最久未用
// - 命中: splice 移到头部(O(1), 迭代器不失效)
// - 淘汰: back() + map.erase
//
// 与 Python 版(solutions/python/src/answers/algorithms/lru_cache.py)对照阅读。
#pragma once

#include <cstddef>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace answers {

class LRUCache {
public:
    explicit LRUCache(size_t capacity) : cap_(capacity) {
        if (capacity == 0) throw std::invalid_argument("capacity must be > 0");
    }

    int get(int key) {
        auto it = map_.find(key);
        if (it == map_.end()) return -1;
        // splice 把节点移到表头, 不拷贝不失效迭代器
        items_.splice(items_.begin(), items_, it->second);
        return it->second->second;
    }

    void put(int key, int value) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            it->second->second = value;
            items_.splice(items_.begin(), items_, it->second);
            return;
        }
        if (map_.size() == cap_) {  // 淘汰链表尾(最久未用)
            map_.erase(items_.back().first);
            items_.pop_back();
        }
        items_.emplace_front(key, value);
        map_[key] = items_.begin();
    }

    size_t size() const { return map_.size(); }

    // 调试辅助: 从最近到最久输出 key
    std::vector<int> keys_mru_to_lru() const {
        std::vector<int> out;
        out.reserve(items_.size());
        for (auto &kv : items_) out.push_back(kv.first);
        return out;
    }

private:
    size_t cap_;
    std::list<std::pair<int, int>> items_;                       // (key, value)
    std::unordered_map<int, std::list<std::pair<int, int>>::iterator> map_;
};

}  // namespace answers
