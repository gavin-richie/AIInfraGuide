"""LeetCode 127. 单词接龙（单词梯）。

来源: 快手-AI-Infra-一面 编程题 7。

bfs_word_ladder   —— 单向 BFS，通配符建邻接表，O(N*L) 建图 + O(N*L) 搜索
bidirectional_bfs —— 双向 BFS，从 beginWord/endWord 两端交替扩展小侧，
                     搜索空间从 b^d 降到 2*b^(d/2)，长词典下快一个数量级
"""

from collections import deque


def _build_pattern_map(word_list: list[str]) -> dict[str, list[str]]:
    """把每个词按位替换成通配符（hot -> *ot / h*t / ho*），同模式的词相邻。"""
    pattern_map: dict[str, list[str]] = {}
    for word in word_list:
        for i in range(len(word)):
            key = word[:i] + "*" + word[i + 1 :]
            pattern_map.setdefault(key, []).append(word)
    return pattern_map


def bfs_word_ladder(begin_word: str, end_word: str, word_list: list[str]) -> int:
    """单向 BFS。返回最短变换序列长度（含首尾），不可达返回 0。"""
    if end_word not in word_list:
        return 0
    pattern_map = _build_pattern_map(word_list)
    visited = {begin_word}
    queue: deque[tuple[str, int]] = deque([(begin_word, 1)])
    while queue:
        word, length = queue.popleft()
        if word == end_word:
            return length
        for i in range(len(word)):
            key = word[:i] + "*" + word[i + 1 :]
            for neighbor in pattern_map.get(key, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, length + 1))
            # 该模式已展开过，清掉可避免重复扫描（可选的小剪枝）
            pattern_map[key] = []
    return 0


def bidirectional_bfs(begin_word: str, end_word: str, word_list: list[str]) -> int:
    """双向 BFS：每轮只扩展当前较小的那一侧 frontier，两端相遇即得最短路。"""
    if end_word not in word_list:
        return 0
    pattern_map = _build_pattern_map(word_list)
    front, back = {begin_word}, {end_word}
    visited = {begin_word, end_word}
    length = 1
    while front and back:
        # 始终扩展较小的一侧，让两侧 frontier 大小尽量平衡
        if len(front) > len(back):
            front, back = back, front
        next_front: set[str] = set()
        for word in front:
            for i in range(len(word)):
                key = word[:i] + "*" + word[i + 1 :]
                for neighbor in pattern_map.get(key, ()):
                    if neighbor in back:  # 与另一侧相遇
                        return length + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_front.add(neighbor)
        front = next_front
        length += 1
    return 0


def ladder_all_shortest(begin_word: str, end_word: str, word_list: list[str]) -> list[list[str]]:
    """附加：BFS 分层记录前驱，回溯出所有最短路径（面试常见追问 LC126 雏形）。"""
    if end_word not in word_list:
        return []
    pattern_map = _build_pattern_map(word_list)
    parents: dict[str, list[str]] = {begin_word: []}
    layer = {begin_word}
    found = False
    while layer and not found:
        next_layer: set[str] = set()
        layer_parents: dict[str, list[str]] = {}
        for word in layer:
            for i in range(len(word)):
                for neighbor in pattern_map.get(word[:i] + "*" + word[i + 1 :], ()):
                    if neighbor == end_word:
                        found = True
                    if neighbor not in parents:
                        layer_parents.setdefault(neighbor, []).append(word)
                        next_layer.add(neighbor)
        for child, dads in layer_parents.items():
            parents[child] = dads
        layer = next_layer
    if not found:
        return []

    def backtrack(node: str) -> list[list[str]]:
        if not parents[node]:
            return [[node]]
        return [[*path, node] for dad in parents[node] for path in backtrack(dad)]

    return backtrack(end_word)
