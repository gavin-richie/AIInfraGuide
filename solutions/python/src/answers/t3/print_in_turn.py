"""三个线程按顺序循环打印递增数字（不死锁版）。

来源: 英伟达-AI-Infra（编程 5，追问"某线程异常退出如何不死锁"）。

要点:
1. 用一个 Condition + turn 变量即可: 线程 i 等待 turn == i 时打印下一个数并唤醒所有人;
2. 不死锁的三个防线:
   - wait(timeout) 带超时, 即使通知丢失也能周期性醒来检查"轮到我了吗/该退出了吗";
   - worker 全程 try/finally, 异常时在 finally 里把 turn 推给下一个线程并 notify_all,
     保证其他线程不会永远等一个已死的线程;
   - 停止标志 stop 由主线程设置, 各线程醒来先看 stop 再看 turn。
"""
from __future__ import annotations

import threading


def run_print_in_turn(
    n: int,
    n_threads: int = 3,
    fail_at: int | None = None,
) -> list[int]:
    """线程 0 打印 1, 线程 1 打印 2, ... 线程 0 打印 n_threads+1, 循环到 n。

    fail_at: 让打印该数字的线程抛异常（模拟线程崩溃），用于验证不死锁。
    返回实际打印成功的数字序列（供测试断言）。
    """
    if n_threads < 1 or n < 1:
        raise ValueError("n_threads and n must be positive")

    cond = threading.Condition()
    turn = 0          # 该哪个线程打印
    next_num = 1      # 下一个要打印的数字
    stop = False      # 任一线程异常后置位, 让其他线程有序退出
    printed: list[int] = []
    printed_lock = threading.Lock()

    def worker(tid: int) -> None:
        nonlocal turn, next_num, stop
        while True:
            with cond:
                # 防线 1: 带超时的 wait, 通知丢失也能周期性醒来
                while not stop and turn != tid:
                    cond.wait(timeout=0.05)
                if stop:
                    return
                num = next_num
            try:
                if fail_at is not None and num == fail_at:
                    raise RuntimeError(f"worker {tid} crashed at {num}")
                with printed_lock:
                    printed.append(num)
                with cond:
                    next_num += 1
                    if next_num > n:
                        stop = True
                    turn = (tid + 1) % n_threads
                    cond.notify_all()
            except Exception:
                # 防线 2: 异常时推进 turn 并广播, 其他线程继续工作而非死锁
                with cond:
                    stop = True
                    turn = (tid + 1) % n_threads
                    cond.notify_all()
                return

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        # 防线 3: 主线程 join 也带超时, 兜底避免测试本身挂死
        t.join(timeout=10)
        if t.is_alive():
            raise TimeoutError("deadlock detected: worker did not finish")
    return printed
