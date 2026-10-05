"""图论题：最短连通路径（MST）、最短路、并查集。

来源: 字节-AI-Infra-一面-2（最短连通路径编程）、文远知行-一面（DFS+二分）等。
网格 BFS/岛屿数量/拓扑排序见 trees.py。
"""
from __future__ import annotations

import heapq
from collections import defaultdict


class DisjointSet:
    """并查集(路径压缩 + 按秩合并), 近 O(1) 均摊。"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径减半
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal_mst(n: int, edges: list[tuple[int, int, int]]) -> tuple[int, list[tuple[int, int, int]]]:
    """最小生成树(Kruskal): 边排序 + 并查集, O(E log E)。

    "最短连通路径/最低成本连通所有点"通常即 MST。
    返回 (总权值, 选中的边列表)。若图不连通则抛异常。
    来源: 字节-AI-Infra-一面-2。
    """
    total = 0
    chosen: list[tuple[int, int, int]] = []
    dsu = DisjointSet(n)
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if dsu.union(u, v):
            total += w
            chosen.append((u, v, w))
            if len(chosen) == n - 1:
                return total, chosen
    if len(chosen) != n - 1:
        raise ValueError("graph is not connected")
    return total, chosen


def prim_mst(n: int, adj: dict[int, list[tuple[int, int]]]) -> int:
    """Prim: 小根堆扩边, 适合稠密图, O(E log V)。"""
    visited = [False] * n
    heap = [(0, 0)]  # (权值, 顶点) 从 0 号开始
    total = 0
    count = 0
    while heap and count < n:
        w, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total += w
        count += 1
        for v, wv in adj.get(u, []):
            if not visited[v]:
                heapq.heappush(heap, (wv, v))
    if count != n:
        raise ValueError("graph is not connected")
    return total


def dijkstra(n: int, adj: dict[int, list[tuple[int, int]]], src: int) -> list[int]:
    """Dijkstra 单源最短路(非负权): O(E log V)。

    与 BFS 的区别: BFS 按层数扩展(无权图), Dijkstra 按当前最短距离扩展(带权图)。
    """
    dist = [float("inf")] * n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:  # 过期条目
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return [int(d) if d != float("inf") else -1 for d in dist]


def minimize_max_dfs(n_nodes: int, edges: list[tuple[int, int, int]]) -> int:
    """"最大化最小值"型 DFS+二分模板: 求从 0 到 n-1 路径上最大边权的最小可能值。

    来源: 文远知行-一面（DFS+二分优化）。二分答案 limit, DFS 只走 <= limit 的边,
    判断是否连通。也可用 Kruskal 重构树/并查集逐步加边。
    """
    adj: defaultdict[int, list[int]] = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    def reachable(limit: int) -> bool:
        stack, seen = [0], {0}
        while stack:
            u = stack.pop()
            if u == n_nodes - 1:
                return True
            for v, w in adj[u]:
                if w <= limit and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    weights = sorted({w for _, _, w in edges})
    lo, hi = 0, len(weights) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if reachable(weights[mid]):
            hi = mid
        else:
            lo = mid + 1
    return weights[lo] if weights else 0
