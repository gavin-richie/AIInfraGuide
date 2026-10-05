"""algorithms 包全量测试。"""
import random

import pytest

from answers.algorithms.linked_list import (
    ListNode, reverse_list, merge_two_lists, merge_k_lists, has_cycle, detect_cycle,
    get_intersection_node, reverse_k_group, delete_duplicates_sorted, add_two_numbers,
    remove_nth_from_end, copy_random_list, sorted_list_to_bst, bubble_sort_list,
    RandomNode,
)
from answers.algorithms.lru_cache import LRUCache, LRUCacheOrdered
from answers.algorithms.min_stack import (
    MinStack, bst_iterator, TwoQueuesStack, validate_stack_sequences,
)
from answers.algorithms.trees import (
    TreeNode, level_order, right_side_view, inorder_nonrecursive, max_path_sum,
    path_sum_iii, topo_sort_kahn, topo_sort_dfs, num_islands, bfs_board_reachable, Trie,
)
from answers.algorithms.topk_heap import (
    kth_largest, quickselect_kth_largest, kth_smallest_sorted_matrix,
    median_of_k_sorted_arrays, top_k_frequent, top_k_frequent_buckets,
    MedianFinder, max_sliding_window, sliding_window_median,
)
from answers.algorithms.sliding_window import (
    longest_consecutive, length_of_longest_substring_k_distinct,
    length_of_longest_substring, three_sum, longest_valid_parentheses, search_matrix,
)
from answers.algorithms.dp import (
    max_sub_array, longest_common_subsequence, edit_distance, longest_palindromic_substring,
    next_permutation, find_median_sorted_arrays, trailing_zeroes_of_factorial,
    circle_return_ways, min_distance_two_keys, string_multiply, max_profit_stock,
    knapsack_01, knapsack_complete, trap_rain_water, min_segment_length,
    k_coverage_intervals, min_rooms_meetings,
)
from answers.algorithms.math_bits import (
    hamming_weight, hamming_weight_shift, first_one_index, is_power_of_two,
    my_pow, my_sqrt, newton_sqrt, float_equal,
    nth_char_rotated_concat, nth_char_rotated_concat_bruteforce,
)
from answers.algorithms.graphs import DisjointSet, kruskal_mst, prim_mst, dijkstra, minimize_max_dfs
from answers.algorithms.sorting import (
    quick_sort, quick_sort_inplace, heap_sort, merge_sort, insertion_sort,
    uppercase_to_lowercase_sorted, is_palindrome, check_inclusion,
)


# ---------------------------------------------------------------- 链表
def test_reverse_and_merge():
    a = ListNode.from_list([1, 3, 5])
    b = ListNode.from_list([2, 4, 6])
    merged = merge_two_lists(a, b)
    assert merged.to_list() == [1, 2, 3, 4, 5, 6]
    assert reverse_list(merged).to_list() == [6, 5, 4, 3, 2, 1]


def test_merge_k_lists():
    lists = [ListNode.from_list(x) for x in ([1, 4, 5], [1, 3, 4], [2, 6])]
    assert merge_k_lists(lists).to_list() == [1, 1, 2, 3, 4, 4, 5, 6]
    assert merge_k_lists([]) is None
    assert merge_k_lists([None, ListNode.from_list([7])]).to_list() == [7]


def test_cycle():
    head = ListNode.from_list([3, 2, 0, -4])
    tail = head
    while tail.next:
        tail = tail.next
    tail.next = head.next  # 环入口在节点 2
    assert has_cycle(head)
    assert detect_cycle(head).val == 2
    no_cycle = ListNode.from_list([1, 2])
    assert not has_cycle(no_cycle) and detect_cycle(no_cycle) is None


def test_intersection():
    common = ListNode.from_list([8, 9])
    a = ListNode.from_list([4, 1]); a_tail = a
    while a_tail.next: a_tail = a_tail.next
    a_tail.next = common
    b = ListNode.from_list([5, 6]); b_tail = b
    while b_tail.next: b_tail = b_tail.next
    b_tail.next = common
    assert get_intersection_node(a, b) is common


def test_reverse_k_group():
    head = ListNode.from_list([1, 2, 3, 4, 5])
    assert reverse_k_group(head, 2).to_list() == [2, 1, 4, 3, 5]
    head = ListNode.from_list([1, 2, 3, 4, 5])
    assert reverse_k_group(head, 3).to_list() == [3, 2, 1, 4, 5]  # 不足 k 保持原序


def test_duplicates_and_add():
    head = ListNode.from_list([1, 2, 3, 3, 4, 4, 5])
    assert delete_duplicates_sorted(head).to_list() == [1, 2, 5]
    a = ListNode.from_list([2, 4, 3])  # 342
    b = ListNode.from_list([5, 6, 4])  # 465
    assert add_two_numbers(a, b).to_list() == [7, 0, 8]  # 807
    head = ListNode.from_list([1])
    assert add_two_numbers(head, None).to_list() == [1]


def test_remove_nth_and_copy_random():
    head = ListNode.from_list([1, 2, 3, 4, 5])
    assert remove_nth_from_end(head, 2).to_list() == [1, 2, 3, 5]

    n1, n2, n3, n4 = RandomNode(7), RandomNode(13), RandomNode(11), RandomNode(10)
    n5 = RandomNode(1)
    n1.next, n2.next, n3.next, n4.next = n2, n3, n4, n5
    n2.random, n3.random, n4.random, n5.random = n1, n5, n1, n1
    c1 = copy_random_list(n1)
    vals, rand_vals = [], []
    node = c1
    while node:
        vals.append(node.val)
        rand_vals.append(node.random.val if node.random else 0)
        node = node.next
    assert vals == [7, 13, 11, 10, 1]
    assert rand_vals == [0, 7, 1, 7, 7]


def test_sorted_list_to_bst_and_bubble():
    head = ListNode.from_list([-10, -3, 0, 5, 9])
    root = sorted_list_to_bst(head)
    assert root.val == 0
    assert inorder_nonrecursive(root) == [-10, -3, 0, 5, 9]  # BST 中序 = 有序

    head = ListNode.from_list([4, 2, 5, 1, 3])
    assert bubble_sort_list(head).to_list() == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------- LRU / 栈
def test_lru_cache():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1       # 访问 1 → 2 变最旧
    cache.put(3, 3)                # 淘汰 2
    assert cache.get(2) == -1
    cache.put(4, 4)                # 淘汰 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4
    assert cache.keys_mru_to_lru() == [4, 3]
    with pytest.raises(ValueError):
        LRUCache(0)


def test_lru_ordered_dict():
    cache = LRUCacheOrdered(2)
    cache.put(1, 1); cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)
    assert cache.get(2) == -1
    assert cache.get(1) == 1 and cache.get(3) == 3


def test_min_stack():
    s = MinStack()
    for v in (5, 3, 7, 3, 8):
        s.push(v)
    assert s.get_min() == 3
    s.pop(); s.pop()            # 弹出 8、3 → 栈 [5,3,7]
    assert s.get_min() == 3
    s.pop(); s.pop()            # 弹出 7、3 → 栈 [5]
    assert s.get_min() == 5


def test_bst_iterator():
    root = TreeNode.from_level_order([7, 3, 15, None, None, 9, 20])
    assert list(bst_iterator(root)) == [3, 7, 9, 15, 20]


def test_two_queues_stack():
    s = TwoQueuesStack()
    for v in (1, 2, 3):
        s.push(v)
    assert s.top() == 3
    assert s.pop() == 3
    assert s.pop() == 2


def test_validate_stack_sequences():
    assert validate_stack_sequences([1, 2, 3, 4, 5], [4, 5, 3, 2, 1])
    assert not validate_stack_sequences([1, 2, 3, 4, 5], [4, 3, 5, 1, 2])


# ---------------------------------------------------------------- 树
def test_level_order_right_view():
    root = TreeNode.from_level_order([3, 9, 20, None, None, 15, 7])
    assert level_order(root) == [[3], [9, 20], [15, 7]]
    assert right_side_view(root) == [3, 20, 7]


def test_path_problems():
    root = TreeNode.from_level_order([-10, 9, 20, None, None, 15, 7])
    assert max_path_sum(root) == 42
    root = TreeNode.from_level_order([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1])
    assert path_sum_iii(root, 8) == 3


def test_topo_sort():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert topo_sort_kahn(4, edges) == [0, 1, 2, 3]
    assert topo_sort_dfs(4, edges) == [0, 2, 1, 3] or topo_sort_dfs(4, edges) == [0, 1, 2, 3]
    assert topo_sort_kahn(3, [(0, 1), (1, 2), (2, 0)]) == []  # 环
    assert topo_sort_dfs(3, [(0, 1), (1, 2), (2, 0)]) == []


def test_num_islands_and_bfs():
    grid = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"],
    ]
    assert num_islands([row[:] for row in grid]) == 3
    board = [[".", ".", "#"], [".", "#", "."], [".", ".", "."]]
    reach = bfs_board_reachable(board, (0, 0))
    assert (0, 0) in reach and (0, 2) not in reach and (2, 2) in reach


def test_trie():
    t = Trie()
    t.insert("apple")
    assert t.search("apple") and not t.search("app")
    assert t.starts_with("app")
    t.insert("app")
    assert t.search("app")


# ---------------------------------------------------------------- TopK/堆
def test_kth_largest():
    nums = [3, 2, 1, 5, 6, 4]
    assert kth_largest(nums, 2) == 5
    assert quickselect_kth_largest(nums[:], 2) == 5
    assert kth_largest([1], 1) == 1
    with pytest.raises(ValueError):
        kth_largest([1, 2], 5)


def test_kth_smallest_matrix():
    matrix = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
    assert kth_smallest_sorted_matrix(matrix, 8) == 13


def test_median_k_sorted():
    assert median_of_k_sorted_arrays([[1, 3], [2], [4, 5]]) == 3.0
    assert median_of_k_sorted_arrays([[1, 2], [3, 4]]) == 2.5
    assert median_of_k_sorted_arrays([[7]]) == 7.0


def test_topk_frequent():
    assert top_k_frequent(["i", "love", "leetcode", "i", "love", "coding"], 2) == ["i", "love"]
    assert top_k_frequent(["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], 4) == \
        ["the", "is", "sunny", "day"]
    assert sorted(top_k_frequent_buckets([1, 1, 1, 2, 2, 3], 2)) == [1, 2]


def test_median_finder():
    mf = MedianFinder()
    for x in (1, 2, 3):
        mf.add_num(x)
    assert mf.find_median() == 2.0
    mf.add_num(4)
    assert mf.find_median() == 2.5


def test_sliding_window_max_median():
    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert sliding_window_median([1, 3, -1, -3, 5, 3, 6, 7], 3) == [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]


# ---------------------------------------------------------------- 滑窗/双指针
def test_longest_consecutive():
    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longest_consecutive([]) == 0


def test_k_distinct():
    assert length_of_longest_substring_k_distinct("eceba", 2) == 3
    assert length_of_longest_substring("abcabcbb") == 3


def test_three_sum():
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
    assert three_sum([0, 0, 0]) == [[0, 0, 0]]


def test_longest_valid_parentheses():
    assert longest_valid_parentheses("(()") == 2
    assert longest_valid_parentheses(")()())") == 4
    assert longest_valid_parentheses("") == 0


def test_search_matrix():
    m = [[1, 4, 7, 11], [2, 5, 8, 12], [3, 6, 9, 16], [10, 13, 14, 17]]
    assert search_matrix(m, 5) and not search_matrix(m, 15)


# ---------------------------------------------------------------- DP
def test_max_sub_array():
    assert max_sub_array([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6


def test_lcs_and_edit():
    assert longest_common_subsequence("abcde", "ace") == 3
    assert edit_distance("horse", "ros") == 3


def test_longest_palindrome():
    assert longest_palindromic_substring("babad") in ("bab", "aba")
    assert longest_palindromic_substring("cbbd") == "bb"


def test_next_permutation():
    nums = [1, 2, 3]
    next_permutation(nums)
    assert nums == [1, 3, 2]
    nums = [3, 2, 1]
    next_permutation(nums)
    assert nums == [1, 2, 3]
    nums = [1, 1, 5]
    next_permutation(nums)
    assert nums == [1, 5, 1]


def test_median_sorted_arrays():
    assert find_median_sorted_arrays([1, 3], [2]) == 2.0
    assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5
    assert find_median_sorted_arrays([], [1]) == 1.0


def test_factorial_zeroes_and_others():
    assert trailing_zeroes_of_factorial(5) == 1
    assert trailing_zeroes_of_factorial(25) == 6   # 25/5 + 25/25
    assert circle_return_ways(10, 4) > 0
    # 1 步不可能回原点; 2 步 = 一顺一逆两种顺序 → 2 种走法
    assert circle_return_ways(10, 1) == 0
    assert circle_return_ways(10, 2) == 2
    assert min_distance_two_keys(3) == 3
    assert min_distance_two_keys(6) == 5
    assert string_multiply("123", "456") == "56088"
    assert string_multiply("0", "12345") == "0"


def test_stock_and_knapsack():
    assert max_profit_stock([7, 1, 5, 3, 6, 4]) == 5
    assert knapsack_01([1, 3, 4], [15, 20, 30], 4) == 35
    assert knapsack_complete([1, 3, 4], [15, 20, 30], 4) == 60  # 1+3: 15+20=35? 完全背包可重复: 4×1=60


def test_trap_water():
    assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6


def test_intervals():
    # 4 个点 [1,2,5,6], 2 条线段: 覆盖 {1,2} 和 {5,6} → 最短 1
    assert min_segment_length([1, 2, 5, 6], 2) == 1
    assert min_segment_length([1, 10], 1) == 9
    assert k_coverage_intervals([(1, 3), (2, 4), (5, 7)], 2) == 2  # [2,3] 被 1,2 覆盖 + [3,4] → 2
    assert min_rooms_meetings([(0, 30), (5, 10), (15, 20)]) == 2
    assert min_rooms_meetings([(1, 2)]) == 1


# ---------------------------------------------------------------- 位运算
def test_hamming():
    assert hamming_weight(11) == 3       # 0b1011
    assert hamming_weight_shift(11) == 3
    assert hamming_weight(0) == 0
    assert first_one_index(0b100) == 2
    assert is_power_of_two(64) and not is_power_of_two(65) and not is_power_of_two(0)


def test_pow_sqrt_float():
    assert my_pow(2.0, 10) == pytest.approx(1024.0)
    assert my_pow(2.0, -2) == pytest.approx(0.25)
    assert my_sqrt(8) == 2 and my_sqrt(16) == 4
    assert newton_sqrt(9.0) == pytest.approx(3.0)
    assert float_equal(0.1 + 0.2, 0.3, rel_tol=1e-9)
    assert not float_equal(1.0, 1.1)


def test_rotated_concat_string():
    s = "abc"
    # 暴力展开对拍
    for n in range(1, 100):
        assert nth_char_rotated_concat(s, n) == nth_char_rotated_concat_bruteforce(s, n), n


# ---------------------------------------------------------------- 图
def test_mst_and_dijkstra():
    #   0 --1-- 1
    #   |      /|
    #   3     2 4
    #   |   /
    #   2
    edges = [(0, 1, 1), (0, 2, 3), (1, 2, 2), (1, 3, 4)]
    total, chosen = kruskal_mst(4, edges)
    assert total == 7
    # 排序后选 1+2+4, (0,2,3) 因成环被跳过
    assert sorted(e[:2] for e in chosen) == [(0, 1), (1, 2), (1, 3)]

    adj = {0: [(1, 1), (2, 3)], 1: [(2, 2), (3, 4)], 2: [(3, 10)], 3: []}
    assert prim_mst(4, adj) == 7
    assert dijkstra(4, adj, 0) == [0, 1, 3, 5]


def test_disjoint_set():
    dsu = DisjointSet(5)
    assert dsu.union(0, 1) and dsu.union(1, 2)
    assert dsu.find(0) == dsu.find(2)
    assert not dsu.union(0, 2)


def test_minimize_max_dfs():
    edges = [(0, 1, 3), (0, 2, 1), (1, 3, 2), (2, 3, 5)]
    # 0→2→1→3 路径最大边 = max(1,2,2)... 该图 0-1(3),0-2(1),1-3(2),2-3(5)
    # 最小化最大边: 0→2(1)→... 2-3 是 5; 0→1(3)→3(2) 最大 3 → 答案 3
    assert minimize_max_dfs(4, edges) == 3


# ---------------------------------------------------------------- 排序
def test_sorts():
    data = [random.Random(42).randint(-100, 100) for _ in range(200)]
    expect = sorted(data)
    assert quick_sort(data) == expect
    quick_sort_inplace(data)
    assert data == expect
    assert merge_sort(data) == expect
    assert insertion_sort(data[:]) == expect
    assert heap_sort(data[:]) == expect


def test_uppercase_lowercase_sorted():
    assert uppercase_to_lowercase_sorted("BaNaNa") == "aaabnn"
    assert uppercase_to_lowercase_sorted("CBA") == "abc"


def test_palindrome_and_inclusion():
    assert is_palindrome("abcba") and not is_palindrome("abcdba")
    assert check_inclusion("ab", "eidbaooo")
    assert not check_inclusion("ab", "eidboaoo")
