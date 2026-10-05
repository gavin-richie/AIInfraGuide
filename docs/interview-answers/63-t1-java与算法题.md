---
title: "T1 大厂 · Java 系与编程题总索引"
description: "美团校招 Java/MySQL/网络题全解：HashMap 红黑树、线程池、synchronized 与 AQS、GC 排查、最左前缀、EXPLAIN、开机到浏览器全链路、UDP 视频会议；附本批次全部编程题索引"
pubDate: 2026-10-05
topic: "公司题解-T1"
order: 63
tags: ["面试解答", "美团", "Java", "MySQL", "计算机网络", "算法"]
draft: false
---

覆盖题库：`美校`=美团-AI-Infra-校招-一面（一面：Java 集合/并发/JVM/MySQL；二面：Java 语言细节/并发/网络/MySQL）。AI Infra 岗的 Java 系考察集中于美团校招，本篇全解；同批次的 CUDA/算法编程题在文末给**全批次总索引**（代码在 `solutions/`，已有多数在 [02](./02-cuda-编程题.md)/[10](./10-cpp编程题.md)/[12](./12-算法题.md) 覆盖）。

## 一、集合与语言基础（美校 一面）

### Q1. HashMap 为什么引入红黑树？

JDK8 前 bucket 是链表，哈希冲突或恶意碰撞（hash DoS）下查询退化 O(n)。树化后冲突链变平衡二叉搜索树，查询 O(log n)——把**最坏情况从线性降到对数**，防御性设计。树化条件：单桶节点数 ≥8 **且** 数组容量 ≥64（否则先扩容）；退化阈值 6（滞后带防在 7/8 间抖动反复树化/链化）。

### Q2. 既然红黑树查询快，为什么不从一开始就用？

①**理想情况下链表足够**：负载因子 0.5、hash 均匀时，单桶长度符合泊松分布，长度到 8 的概率 ≈ 千万分之六（0.00000006）——树化是极小概率的兜底，不是常态路径；②**成本**：TreeNode 占用约普通 Node 的 2 倍内存（含红黑指针/父子链），全用树浪费；③**维护代价**：树化/旋转/链化在写路径上更贵，绝大多数桶短链表插入删除更快。设计哲学：**为常见情形优化（数组+短链），为最坏情形兜底（树化）**。

### Q3. 用二叉排序树或 AVL 树替代红黑树可行吗？

- **普通 BST**：不可——插入有序数据退化成链表 O(n)，比红黑树更差；
- **AVL**：查询更快（更严格平衡，高度差 ≤1，log n 常数更小），但**插入/删除要更多旋转**（AVL 恢复平衡可能 O(log n) 次旋转，红黑树 O(1) 次旋转+变色）——HashMap 是写多读杂的场景，红黑树在插入删除与查询之间取平衡，工程上更划算；
- 结论：查询极密集且很少改（内存数据库索引）可用 AVL；通用容器选红黑树。加分：B/B+ 树是磁盘场景（节点=页）的对应选择（MySQL 索引）。

### Q4. Java 代码运行结果（美校 二面 1）

```java
int a = 3;
Integer b = null;
int c = a + b;        // (1)
long d = 1L + 0x7fffffff;   // (2)
int e = a + (int)d;   // (3)
```

1. `a + b` 对 b **自动拆箱** `b.intValue()` → **NullPointerException**（程序在此中断）。修法：判空（`b == null ? 0 : b`）或保持 Integer 用 `Objects` 工具；
2. `0x7fffffff` = 2147483647（int 最大值）；`1L + x` 把运算提升为 **long**：2147483648，**不溢出** ✓；
3. `(int)d` 把 2147483648 截断回 int：低 32 位是 0x80000000 → **-2147483648**（补码回绕）；e = 3 + (-2147483648) = **-2147483645**。
考点：拆箱 NPE、整数提升、强转的位截断（不是饱和取值）。

### Q5. `Integer a = 567; Integer b = 567; a == b` 输出？（美校 二面 2）

输出 **false**。`IntegerCache` 只缓存 **-128 ~ 127**（可调上限 `-XX:AutoBoxCacheMax`），567 超出缓存 → 两次 `valueOf` 各自 new 对象 → 引用比较 false。改成 `a.equals(b)` 得 true（或先拆箱成 int 比较值）。引申：`Integer == int` 会拆箱比值（true，但有 NPE 风险）；任何包装类型比较一律用 equals。

## 二、并发与线程池（美校 一面/二面）

### Q6. 线程池核心参数与工作原理

`ThreadPoolExecutor` 七参数：`corePoolSize`（常驻核心数）、`maximumPoolSize`（最大线程）、`keepAliveTime + unit`（非核心线程空闲存活）、`workQueue`（任务队列：ArrayBlockingQueue 有界 / LinkedBlockingQueue 默认无界 / SynchronousQueue 直递 / DelayQueue）、`threadFactory`（命名/优先级）、`handler`（拒绝策略）。

**工作流程**（必背顺序）：提交任务 → ①线程数 < core：建核心线程执行；②否则入 `workQueue`；③队列满且线程数 < max：建非核心线程；④队列满且线程数 = max：**触发拒绝策略**（Abort 抛异常默认 / CallerRuns 提交者自己跑 / Discard 静默丢 / DiscardOldest 丢队首）。注意：**先入队后扩非核心**（不是先建满 max）。追问：无界队列的坑（max 失效、OOM）；CPU 密集 N+1 / IO 密集 2N 或 N×(1+等待/计算)；`execute` vs `submit`（返回 Future）；为什么不用 Executors 静态工厂（无界队列/无限线程的 OOM 风险，阿里规约）。

### Q7. 大量 synchronized 效率低，如何优化？

①**减小临界区**：只锁必要行（粗粒度方法锁 → 细粒度块锁）、缩短持锁时间（耗时 IO/计算移出锁外）；②**减小锁粒度**：拆锁（ConcurrentHashMap 的分段/JDK8 CAS+sync 桶级锁）、锁分离（读写锁 `ReadWriteLock`/`StampedLock` 读多写少）；③**无锁化**：CAS/`Atomic*`/`LongAdder`（高竞争计数）、`ThreadLocal` 去共享、COW（`CopyOnWriteArrayList` 读多写极少）；④**并发容器**替换同步包装（`ConcurrentLinkedQueue`、`BlockingQueue` 生产者消费者解耦）；⑤**异步化**：请求排队（actor/disruptor 模型）；⑥注意锁对象选择（不用 String 常量/Integer 缓存实例这类共享对象，`this` vs class 对象）。先 profile 确认锁竞争（jstack 多线程 BLOCKED 在同一 monitor）再对症。

### Q8. synchronized 与 AQS 锁的区别、性能差异在哪？

| | synchronized | AQS（ReentrantLock 等） |
|---|---|---|
| 实现 | JVM 内置 monitor（monitorenter/exit，对象头 Mark Word） | JUC 底层框架：volatile `state` + CLH 变体双向队列 + `LockSupport.park/unpark` |
| 释放 | 异常/退出自动释放 | 必须手动 `unlock()`（finally） |
| 中断/超时 | 不可中断、不可超时 | `lockInterruptibly`、`tryLock(timeout)` |
| 公平性 | 非公平 | 可选公平/非公平 |
| 条件变量 | 一个（wait/notify） | 多个 `Condition`（精准唤醒） |
| 性能 | JDK6 后锁升级（偏向→轻量级 CAS 自旋→重量级 monitor）后，低中竞争下与 AQS 相当 | 高竞争下队列排队+挂起更稳定（避免自旋烧 CPU），可限流（Semaphore）/共享（读写锁） |

性能差异本质：synchronized 轻量级靠自旋 CAS，竞争加剧升级重量级（内核 mutex，用户/内核态切换）；AQS 竞争者直接入队 park（主动让出 CPU）+ 支持公平排队——**高竞争下 AQS 的行为更可预测**；低竞争下 synchronized 更简单且够快。选择：能用 sync 就用（简单安全），需要可中断/超时/公平/多条件/高级同步器才上 AQS 系。

### Q9. Java 并发知识体系（美校 二面 3）

按层次答：①**可见性/有序性/原子性**三性问题与 JMM（happens-before、volatile 内存屏障、指令重排）；②**互斥**：synchronized/Lock、锁升级；③**协作**：wait/notify、Condition、CountDownLatch/CyclicBarrier/Semaphore、CompletableFuture；④**无锁**：CAS（ABA 用 AtomicStampedReference）、Atomic 家族、LongAdder；⑤**容器**：ConcurrentHashMap/CopyOnWrite/BlockingQueue；⑥**线程管理**：ThreadPoolExecutor、ThreadLocal（泄漏点：线程池复用 + 不 remove）；⑦**异步模型**：Future/CompletableFuture、虚拟线程（JDK21）。每个层次给一句原理 + 一个场景。

### Q10. Java 常用锁的分类（美校 二面 4）

按维度：**乐观（CAS）vs 悲观（阻塞）**；**公平 vs 非公平**（排队 vs 抢占，吞吐 vs 饥饿）；**可重入 vs 不可**（synchronized/ReentrantLock 可重入，同线程重复获取不死锁）；**读写**（ReadWriteLock/StampedLock——读共享写独占，后者乐观读免锁）；**自旋 vs 挂起**（轻量级锁自旋 burns CPU 换切换开销）；**偏向**（单线程无竞争免 CAS）；**分段/桶级**（ConcurrentHashMap）；**分布式锁**（Redis SETNX/RedLock、ZooKeeper 临时节点）。

## 三、JVM 与 GC（美校 一面）

### Q11. 线上频繁 Full GC 如何排查解决？

**排查链**：①`jstat -gcutil <pid> 1000` 确认频率与哪个区触发（老年代 O？元空间？promotion failed?）；②`jmap -histo` 看对象分布 → `jmap -dump`（或 `-XX:+HeapDumpOnOutOfMemoryError`）导堆 → **MAT/JProfiler** 看支配树找大对象/泄漏源；③对照代码定位常见根因：**无界缓存/集合**（Map 只进不出）、**大结果集查询**（一次拉百万行）、**ThreadLocal 未 remove**（线程池长寿命线程）、**一次性大分配**（大数组/大报文）、System.gc() 被调（RMI/NIO DirectByteBuffer 回收触发）、新生代太小导致对象过早晋升。

**解决**：扩堆/调新生代比例（`-Xmx -Xmn -XX:SurvivorRatio`）、换 G1/ZGC（大堆低停顿）、`-XX:+DisableExplicitGC`、代码修泄漏、限流大查询（分页/流式）、缓存加上限与过期。

### Q12. CMS / G1 的适用场景；了解 ZGC 吗？

- **CMS**（老年代并发标记-清除）：目标最短停顿（标记与用户线程并发）；代价：**内存碎片**（标记清除无压缩，触发 Serial Old 压缩时超长停顿/concurrent mode failure）、浮动垃圾、对 CPU 敏感。适合**延迟敏感、堆中等（≤8G）、老年代对象存活率高**的场景；JDK14 已移除；
- **G1**：Region 化（分代只是角色），全局标记后按**停顿预测模型**优先回收垃圾最多的 Region（Garbage First），`-XX:MaxGCPauseMillis` 可设目标；标记整理/复制无碎片、大堆（6G-几十 G）下停顿可控（默认 200ms 目标）。适合**大堆 + 要求可预测停顿**；
- **ZGC**：着色指针（colored pointer）+ **读屏障**实现并发整理——标记/转移/重定位几乎全并发，停顿 **<1ms 且与堆大小无关**（TB 级堆可），JDK17+ 生产可用；代价：吞吐略低（读屏障开销 ~几个百分点）、需要更多染色元数据。选型一句话：小堆吞吐优先 Parallel；延迟敏感中堆 CMS 已被 G1 取代；超大堆+极致延迟 ZGC。

## 四、MySQL（美校 一面/二面）

### Q13. 最左前缀匹配原则

联合索引 `(a,b,c)` 的 B+ 树按 a 排序、a 相同按 b、再按 c——**索引只能从最左列开始连续使用**：`a=?` ✓、`a=? and b=?` ✓、`a=? and b=? and c=?` ✓；`b=?`（缺 a）✗ 走不了（但可 index skip scan，弱化版）；`a=? and c=?` 用到 a，c 断开。**范围查询后面的列失效**（`a>1` 后 b、c 无法继续定位索引区间）。设计原则：等值列在前、范围列放最后、区分度高的在前；`order by` 也能吃索引（避免 filesort）。

### Q14. 联合索引 (a,b,c)，查询 `a > 1 AND c = 1`，索引用况？

走 `(a,b,c)` 索引的 **range 扫描**：`a > 1` 用索引定位区间（type=range，key=a,b,c）；**b 缺失导致 c 无法用于索引定位**（中间断了）；但 c 在索引列里，MySQL 5.6+ 的 **ICP（Index Condition Pushdown）** 会在存储引擎层用 `c=1` 过滤索引项，**减少回表次数**（EXPLAIN Extra 显示 `Using index condition`）。要点：不是"只有 a 生效其他全废"，而是"a 定位区间 + c 索引层过滤"；若 select 的列都在索引内还能覆盖索引免回表。优化：改写为 `a > 1 and b is not null and c = 1` 之类通常无益——更好的办法是让等值条件在前（调整索引为 (c,a,b) 若查询模式固定）。

### Q15. 慢查询定位与排查

①开启慢日志（`slow_query_log`，`long_query_time`）；②`EXPLAIN` 分析：**type**（ALL 全表扫要警惕；range/ref 较好；const/eq_ref 优）、**key/possible_keys**（是否用了预期索引）、**rows**（预估扫描行数）、**Extra**（`Using filesort` 排序未走索引、`Using temporary` 临时表、`Using index` 覆盖索引好信号、`Using index condition` ICP）；③进一步 `show profile` / performance_schema / trace 看优化器成本。**常见治理**：加/改索引（最左前缀+区分度）、避免函数包裹索引列（`date(t)` ✗）、大分页（延迟关联：先查 id 再 join）、只 select 需要列（覆盖索引）、改写（exists/子查询拆分）、数据量大走分库分表/ES。

### Q16. EXPLAIN 关注哪些字段？

`type`（访问类型，好坏序 system > const > eq_ref > ref > range > index > ALL）、`key`/`possible_keys`、`rows`（扫描行数估计）、`Extra`（filesort/temporary/using index/ICP）、`key_len`（判断联合索引用了几列）、`id` 与 select_type（子查询/派生表顺序）。

### Q17. MySQL 应用层核心知识点（美校 二面 8）

①**索引**：B+ 树（矮胖 3-4 层、叶子链表利于范围）、聚簇 vs 二级（回表、覆盖索引、最左前缀）；②**事务**：ACID 实现（undo/redo/锁+MVCC），隔离级别（RR 默认，MVCC 快照读避免幻读、当前读加间隙锁）；③**日志**：redo（崩溃恢复, WAL）、undo（回滚+MVCC）、binlog（复制/归档，格式 row）、**两阶段提交**（redo prepare + binlog + commit）；④**锁**：行锁/间隙锁/临键锁、死锁排查（`SHOW ENGINE INNODB STATUS`）；⑤**高可用与扩展**：主从复制（异步/半同步、延迟治理）、读写分离、分库分表（垂直/水平、ShardingSphere）；⑥**性能**：连接池、慢查询治理、参数（buffer pool）。

### Q18. SQL 关键字执行顺序

```
FROM / JOIN → ON → WHERE → GROUP BY → HAVING → SELECT（含窗口函数）→ DISTINCT → ORDER BY → LIMIT
```

推论：WHERE 里不能用 SELECT 的别名（SELECT 还没执行）也不能用聚合函数（分组未发生）——过滤聚合要用 HAVING；GROUP BY 后 SELECT 非聚合列依赖 sql_mode（only_full_group_by 报错）；LIMIT 在 ORDER BY 后。甲骨文式的八步口诀 + 两个推论就是满分答案。

### Q19. 表设计与索引怎么答（美校 二面 10）

给真实例子模板：业务表（如订单）——字段：主键 `id BIGINT AUTO_INCREMENT`（或分布式 ID）、业务唯一键 `order_no` 加唯一索引、金额 `DECIMAL`（不用 float）、状态 `TINYINT`、时间 `DATETIME(3)` 默认 CURRENT_TIMESTAMP、逻辑删除 `is_deleted`；索引：**唯一索引**（order_no）、**联合索引**按查询模式（如 `(user_id, status, created_at)`——等值在前、范围在后）、避免冗余索引（(a) 与 (a,b) 重复）；字符集 utf8mb4；大字段拆表（明细/扩展分离）；适度反范式（冗余计数列）。再讲演进：慢查询驱动加索引 → 分区/归档 → 分库分表（shard key 选择避免热点）。

## 五、网络与系统（美校 二面）

### Q20. 从按下开机按钮到打开浏览器进入面试链接，底层经历了什么？

分七段讲：①**加电自检**：电源 PG 信号 → CPU 复位从 0xFFFFFFF0 取第一条指令 → **BIOS/UEFI** 自检（内存/设备枚举）；②**引导**：UEFI 读 ESP 分区的 boot loader（GRUB/Windows bootmgr）→ 加载内核镜像与 initramfs → 跳转内核入口；③**内核初始化**：mm/调度器/中断/驱动 → 挂载真实 rootfs → 启动 1 号进程 **systemd**；④**用户空间**：systemd 按 target 拉起服务（网络 NetworkManager/DHCP 拿 IP、DNS 配置、图形栈 X/Wayland）→ 登录界面 → 桌面环境；⑤**启动浏览器**：双击 → shell fork+exec（ELF 加载、ld.so 解析动态库、ASLR/保护）→ GUI 进程渲染窗口；⑥**网络**：输入 URL → DNS 解析（缓存→系统→递归查询）→ TCP 三次握手（SYN/SYN-ACK/ACK，拥塞控制起步）→ TLS 握手（证书验证、密钥交换）→ HTTP(S) 请求 → 服务端 LB/网关/服务处理 → 响应；⑦**渲染**：HTML 解析建 DOM、CSSOM → 布局 → 绘制 → 合成上屏；会议链接再走 **WebRTC**：信令（WebSocket）→ ICE/STUN 打洞 → SRTP over UDP 媒体流。每段能展开一层（如 TLS 的证书链验证、TCP 的 cwnd 慢启动）就是加分。

### Q21. 视频会议为什么用 UDP 而不是 TCP？

实时媒体的第一诉求是**低延迟与平稳抖动，不是完整可靠**：①TCP 的重传对实时流是负优化——迟到的帧等同丢失（晚 500ms 的画面不如丢掉），重传还加剧拥塞；②TCP **队头阻塞**（一个包丢，后续已到的数据也得等）在应用层（音视频交织）放大延迟抖动；③TCP 拥塞控制激进降速会瞬间砍画质；④实时流本就允许有损（丢帧/P 帧损坏可跳到下个 I 帧）。UDP 之上自建：**应用层可靠性按需补**——NACK/RTX 选择性重传关键帧、FEC 前向纠错、jitter buffer 平滑、带宽估计（GCC）自适应码率。WebRTC 即此架构（SRTP+DTLS over UDP，ICE 打洞，TCP 仅作信令与穿透失败回退）。对照：直播/点播可缓冲用 TCP/HLS 也行，双向会议必须 UDP。

## 六、本批次编程题总索引（T1 公司）

> 代码分三个工程（[solutions/](../../solutions/README.md)）：python（`pip install -e . && pytest tests -v`，本批次新增 `tests/test_t1.py` 29 项）、cuda（ctest）、cpp（ctest）。已有实现标注文档位置，本批次新增标注 `t1`。

| 题目 | 来源 | 实现 | 文档 |
|---|---|---|---|
| LeetCode 127 单词接龙（BFS/双向 BFS） | 快手-一面 | `answers/t1/word_ladder.py` 🆕 | [60](./60-t1-快手与美团.md) 编程题 |
| LeetCode 108 有序数组转平衡 BST | 快手-实习-一面-1 | `answers/t1/bst_from_sorted.py` 🆕 | [60](./60-t1-快手与美团.md) 编程题 |
| LeetCode 378 有序矩阵第 K 小 | 快手-实习-一面-2 | `answers/algorithms/sliding_window.py` | [12](./12-算法题.md) |
| CUDA Norm 算子 | 快手-实习 | `solutions/cuda/src/norm.cu` | [02](./02-cuda-编程题.md) |
| LRU 缓存 | 快手-校招-2 | `answers/algorithms/lru_cache.py` + cpp | [10](./10-cpp编程题.md) Q1 |
| Online Softmax / FA 伪代码 | 快手-校招-2 | `answers/attention/online_softmax.py`、`flash_attention.py` | [02](./02-cuda-编程题.md) Q2 |
| n! 尾随零 / 股票一次买卖 | 快手-校招-一面 | `answers/algorithms/math_bits.py、dp.py` | [12](./12-算法题.md) |
| 合并 K 个升序链表 | 美团-实习 | `answers/algorithms/linked_list.py` | [12](./12-算法题.md) |
| LeetCode 82 删除重复节点 II | 美团-校招-一面 | `answers/algorithms/linked_list.py` | [12](./12-算法题.md) |
| 行列有序矩阵搜目标值 | 美团-校招-二面 | `answers/algorithms/sliding_window.py::search_matrix` | [12](./12-算法题.md) |
| LeetCode 25 K 个一组翻转链表 | 美团-北斗 | `answers/algorithms/linked_list.py::reverse_k_group` | [12](./12-算法题.md) |
| CUDA 前缀和（基础+优化） | 美团-北斗 | `solutions/cuda/src/histogram_scan.cu` | [02](./02-cuda-编程题.md) Q11 |
| CUDA GEMM（基础+优化） | 美团-北斗 | `solutions/cuda/src/gemm.cu` | [02](./02-cuda-编程题.md) Q3 |
| Zero-Bubble 流水线模拟 | 美团-实习 | `answers/t1/zero_bubble.py` 🆕 | [60](./60-t1-快手与美团.md) Q39 |
| 复制随机指针链表 | 蚂蚁-实习-一面-1 | `answers/algorithms/linked_list.py` | [12](./12-算法题.md) |
| CUDA Histogram（原子 vs smem） | 蚂蚁-实习-一面-2 | `solutions/cuda/src/histogram_scan.cu` | [02](./02-cuda-编程题.md) Q5 |
| 手写 GQA | 京东-实习 | `answers/attention/gqa.py` | [04](./04-attention与kvcache.md) Q4 |
| 快速排序 | 京东-校招-一面 | `answers/algorithms/sorting.py` | [12](./12-算法题.md) |
| 最长回文子串 | 小米-一面 | `answers/algorithms/dp.py` | [12](./12-算法题.md) |
| 矩阵转置（CUDA） | 小米-实习-一二面 | `solutions/cuda/src/transpose.cu` | [02](./02-cuda-编程题.md) Q4 |
| 手写膨胀卷积 + 优化 | 小米-实习-一面-1 | `answers/t1/dilated_conv.py` 🆕 | [62](./62-t1-小米oppo-vivo-网易等.md) Q22 |
| 手写 MHA / MGA(GQA) | 小米-校招、拼多多 | `answers/attention/mha.py、gqa.py` | [62](./62-t1-小米oppo-vivo-网易等.md) Q1 |
| 合并两个有序链表 | 小米-实习-一二面 | `answers/algorithms/linked_list.py` | [12](./12-算法题.md) |
| 数据流 Top-K 高频（哈希+小顶堆） | 网易-校招 | `answers/algorithms/topk_heap.py::top_k_frequent` | [12](./12-算法题.md) |
| CUDA 大数组多级归约 | B站-实习-一面 | `solutions/cuda/src/reduce.cu` | [02](./02-cuda-编程题.md) Q1 |
| C++ 编程题（原题无题干） | 拼多多 | C++ 手写全家桶 | [10](./10-cpp编程题.md) |

**Top-K 高频要点**（网易）：哈希表计数 + 大小为 k 的小顶堆（堆顶是当前第 k 大的"守门员"），新元素大于堆顶则替换并下沉——总复杂度 O(n log k)、空间 O(n)；数据流场景堆常驻即可实时更新；对比桶排序版（按频次分桶，O(n) 但要离线知道频次范围）`top_k_frequent_buckets`。
