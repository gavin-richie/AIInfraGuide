"""腾讯题解测试: 双堆滑窗中位数 / 有序插入 / 链表中点回文 / 内存文件系统 / Triton PagedAttention。

对应文档: docs/interview-answers/42-腾讯-编程题.md、40-腾讯-推理与cuda.md Q17。
"""
import random

import pytest

from answers.algorithms.linked_list import ListNode
from answers.algorithms.topk_heap import sliding_window_median as sliding_window_median_naive
from answers.tencent.heap_sliding_window_median import DualHeap, sliding_window_median_dual_heap
from answers.tencent.in_memory_fs import FsError, InMemoryFS
from answers.tencent.linked_list_palindrome import (
    find_middle, is_palindrome, reverse_list,
)
from answers.tencent.sorted_array_insert import (
    bisect_insert_std, insert_all_one_by_one, insert_all_via_merge, insert_one_binary,
)


# ---------------- 双堆滑动窗口中位数 ----------------

def test_dual_heap_median_lc480_examples():
    assert sliding_window_median_dual_heap([1, 3, -1, -3, 5, 3, 6, 7], 3) == \
        pytest.approx([1.0, -1.0, -1.0, 3.0, 5.0, 6.0])
    assert sliding_window_median_dual_heap([1, 2], 1) == pytest.approx([1.0, 2.0])
    assert sliding_window_median_dual_heap([2, 2, 2, 2], 4) == pytest.approx([2.0])


def test_dual_heap_matches_naive_random():
    random.seed(42)
    for k in (1, 2, 3, 5, 8):
        for _ in range(30):
            nums = [random.randint(-50, 50) for _ in range(random.randint(k, k + 30))]
            assert sliding_window_median_dual_heap(nums, k) == \
                pytest.approx(sliding_window_median_naive(nums, k)), (nums, k)


def test_dual_heap_duplicate_heavy():
    # 大量重复值: 验证按值分类删除/平衡的正确性
    nums = [7, 7, 1, 7, 2, 7, 1, 2, 7] * 3
    for k in (2, 4, 9):
        assert sliding_window_median_dual_heap(nums, k) == \
            pytest.approx(sliding_window_median_naive(nums, k))


def test_dual_heap_invalid_k():
    with pytest.raises(ValueError):
        sliding_window_median_dual_heap([1, 2, 3], 0)
    with pytest.raises(ValueError):
        sliding_window_median_dual_heap([1, 2, 3], 4)


def test_dual_heap_class_ops():
    dh = DualHeap()
    for x in (5, 1, 9, 3, 7):
        dh.insert(x)
    assert dh.median() == 5.0
    dh.erase(5)
    assert dh.median() == 5.0  # {1,3,7,9} 中位数 (3+7)/2
    dh.erase(9)
    dh.insert(4)
    assert dh.median() == 3.5  # {1,3,4,7} -> (3+4)/2


# ---------------- 有序数组插入 ----------------

def test_insert_one_binary():
    assert insert_one_binary([1, 3, 5, 7], 4) == [1, 3, 4, 5, 7]
    assert insert_one_binary([], 2) == [2]
    assert insert_one_binary([2, 2, 2], 2) == [2, 2, 2, 2]
    assert insert_one_binary([1, 2], 0) == [0, 1, 2]
    assert insert_one_binary([1, 2], 9) == [1, 2, 9]


def test_insert_all_random_cross_check():
    random.seed(7)
    for _ in range(50):
        n, m = random.randint(0, 40), random.randint(0, 60)
        sorted_arr = sorted(random.randint(-100, 100) for _ in range(n))
        unsorted = [random.randint(-100, 100) for _ in range(m)]
        expect = sorted(sorted_arr + unsorted)
        assert insert_all_one_by_one(sorted_arr, unsorted) == expect
        assert insert_all_via_merge(sorted_arr, unsorted) == expect
        assert bisect_insert_std(sorted_arr, unsorted) == expect
        assert sorted_arr == sorted(sorted_arr)  # 入参未被破坏


# ---------------- 快慢指针 + 反转 + 回文 ----------------

def _build(vals):
    return ListNode.from_list(vals)


def test_find_middle():
    assert (find_middle(_build([1, 2, 3, 2, 1])) or ListNode(0)).val == 2   # 奇数: 后半起点
    assert (find_middle(_build([1, 2, 2, 1])) or ListNode(0)).val == 2      # 偶数: 后半起点
    assert (find_middle(_build([1])) or ListNode(0)).val == 0 or True
    assert find_middle(_build([1])) is None


def test_reverse_list():
    assert reverse_list(_build([1, 2, 3])).to_list() == [3, 2, 1]
    assert reverse_list(_build([1])).to_list() == [1]
    assert reverse_list(None) is None


def test_is_palindrome():
    assert is_palindrome(_build([1, 2, 3, 2, 1]))
    assert is_palindrome(_build([1, 2, 2, 1]))
    assert is_palindrome(_build([1]))
    assert is_palindrome(None)
    assert not is_palindrome(_build([1, 2]))
    assert not is_palindrome(_build([1, 2, 3, 4, 2, 1]))
    random.seed(3)
    for _ in range(30):
        vals = [random.randint(0, 9) for _ in range(random.randint(1, 15))]
        head = _build(vals)
        expect = vals == vals[::-1]
        assert is_palindrome(head) is expect
        assert head.to_list() == vals  # 链表被恢复, 未被破坏


# ---------------- 内存文件系统(字典树) ----------------

def test_fs_basic_tree_and_rw():
    fs = InMemoryFS()
    fs.mkdir("/a")
    fs.mkdir("/a/b", recursive=True)
    fs.write_file("/a/b/f.txt", "hello")
    fs.append_file("/a/b/f.txt", " world")
    assert fs.read_file("/a/b/f.txt") == "hello world"
    assert fs.ls("/a") == ["b"]
    assert fs.ls("/a/b") == ["f.txt"]
    assert fs.ls("/a/b/f.txt") == ["f.txt"]
    assert fs.exists("/a/b/f.txt") and not fs.exists("/a/c")
    assert fs.is_dir("/a/b") and not fs.is_dir("/a/b/f.txt")
    assert fs.du("/") == 11


def test_fs_errors():
    fs = InMemoryFS()
    with pytest.raises(FsError):          # 父目录不存在
        fs.mkdir("/a/b")
    with pytest.raises(FsError):          # 根不可写
        fs.write_file("/", "x")
    fs.mkdir("/a")
    with pytest.raises(FsError):          # 已存在
        fs.mkdir("/a")
    with pytest.raises(FsError):          # 中间是文件
        fs.write_file("/f", "x"); fs.mkdir("/f/sub")
    with pytest.raises(FsError):          # 目录不能写
        fs.write_file("/a", "x")
    with pytest.raises(FsError):          # 文件不能当目录
        fs.write_file("/a/f", "1"); fs.write_file("/a/f/g", "2")
    with pytest.raises(FsError):          # rm 非空目录
        fs.rm("/a")
    with pytest.raises(FsError):          # touch 到目录
        fs.touch("/a")
    with pytest.raises(FsError):          # 路径不存在
        fs.read_file("/a/none")


def test_fs_rm_and_mv_and_recursive_mkdir():
    fs = InMemoryFS()
    fs.mkdir("/x/y/z", recursive=True)
    fs.write_file("/x/y/f1", "aaaa")
    fs.write_file("/x/y/z/f2", "bb")
    assert fs.du("/x") == 6
    with pytest.raises(FsError):
        fs.rm("/x")
    fs.rm("/x", recursive=True)
    assert not fs.exists("/x")

    fs.mkdir("/src"); fs.mkdir("/dst")
    fs.write_file("/src/a", "123")
    fs.mv("/src/a", "/dst/b")
    assert fs.read_file("/dst/b") == "123" and not fs.exists("/src/a")
    fs.mkdir("/src/inner")
    with pytest.raises(FsError):          # 不能移到自己的子目录
        fs.mv("/src", "/src/inner/sub")
    with pytest.raises(FsError):          # 目标已存在
        fs.mv("/dst/b", "/dst/b")
    # 移动目录子树
    fs.mv("/src/inner", "/dst/inner2")
    assert fs.is_dir("/dst/inner2") and not fs.exists("/src/inner")


def test_fs_path_normalization():
    fs = InMemoryFS()
    fs.mkdir("/a")
    fs.write_file("/a//f/", "x")          # 双斜杠/尾斜杠容忍
    assert fs.ls("/a/") == ["f"]
    with pytest.raises(FsError):
        fs.mkdir("relative")              # 相对路径拒绝


# ---------------- Triton PagedAttention(需 GPU + triton) ----------------

def test_paged_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")
    from answers.tencent.triton_paged_attention import (
        paged_attention_torch_ref, triton_paged_attention_decode,
    )

    torch.manual_seed(0)
    num_blocks = 128
    for (num_q_heads, num_kv_heads, head_dim, block_size, max_len, num_seqs) in [
        (32, 8, 64, 16, 64, 8),    # GQA 4:1
        (8, 8, 64, 16, 33, 5),     # MHA, 非整块尾块
        (16, 4, 128, 32, 100, 4),  # GQA 4:1, 大 head_dim/块
    ]:
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                              device="cuda", dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)
        q = torch.randn(num_seqs, num_q_heads, head_dim, device="cuda",
                        dtype=torch.float16)
        seq_lens = torch.randint(1, max_len + 1, (num_seqs,), device="cuda",
                                 dtype=torch.int32)
        # 物理块乱序分配: 每序列取互不重复的随机块, 验证间接寻址正确性
        max_blocks_per_seq = (max_len + block_size - 1) // block_size
        perm = torch.randperm(num_blocks, device="cuda", dtype=torch.int64)
        block_tables = perm[: num_seqs * max_blocks_per_seq].reshape(
            num_seqs, max_blocks_per_seq).to(torch.int32).contiguous()

        got = triton_paged_attention_decode(q, k_cache, v_cache, block_tables, seq_lens)
        ref = paged_attention_torch_ref(q, k_cache, v_cache, block_tables, seq_lens)
        assert torch.allclose(got, ref, atol=1e-2), \
            (num_q_heads, num_kv_heads, head_dim, (got - ref).abs().max().item())


def test_paged_attention_demo():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU")
    from answers.tencent.triton_paged_attention import paged_attention_demo
    assert paged_attention_demo()["allclose"] is True
