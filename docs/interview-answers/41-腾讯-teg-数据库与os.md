---
title: "腾讯 TEG 题解：数据库与操作系统"
description: "腾讯 TEG AI Infra 一二三面 39 题全解：MySQL/分布式数据库 19 题（JOIN/Buffer Pool/MVCC/B+树/分布式执行计划）、Linux OS 6 题（启动/中断/内核态切换）、MiniOb/xv6/OceanBase 工程实践"
pubDate: 2026-10-05
topic: "公司题解-腾讯"
order: 41
tags: ["公司题解", "数据库", "操作系统"]
draft: false
---

覆盖题库：`腾讯-TEG-AI-Infra-一二三面` 全部 39 题。TEG 这套题是本库唯一成体系的**数据库/OS 专项**（01-12 篇未覆盖），因此独立成篇认真展开；数据库题以 InnoDB/MySQL 与 OceanBase 为参照系，OS 题以 Linux x86-64 为准。编程题见[42 篇](./42-腾讯-编程题.md) Q7/Q8/Q9。

## 一、数据库（一面 Q2-Q20）

### Q1. 聚合函数与 GROUP BY 的底层实现（TEG 一面 2）

两条技术路线，执行器按代价选择：

- **Hash 聚合（主流）**：扫表/下游算子输出 → 以 group key 哈希到内存哈希表 → 命中则**增量更新聚合状态**（SUM 累加、COUNT 计数、MAX/MIN 比较、AVG 拆成 SUM+COUNT），未命中插入新状态 → 输出阶段遍历哈希表。等值分组 O(n)，不要求输入有序。内存放不下时**分两阶段**（并行场景天然如此）：局部聚合 → 按 key shuffle/repartition → 全局聚合合并（partial count + partial count = final count）；单机溢出则 **grace 落盘**：按 hash(key) 分区写盘，逐分区重入内存聚合。
- **Sort 聚合**：先按 group key 排序（利用索引序免排序），相同 key 连续出现，流式更新状态、换组即输出。适合 key 已有序（索引扫描）或本身要求 ORDER BY。
- MySQL 行为：`EXPLAIN` 出现 `Using temporary; Using filesort` 多为 sort/tmp-table 路线（内存 TEMP_TABLE 引擎，超 `tmp_table_size` 转磁盘）；8.0 没有 hash agg（除 8.0.31 前的递归 CTE 等特殊路径），PG/Oracle/OB 均有 hash agg。

### Q2. JOIN 的实现方法及优化策略（索引优化、哈希表优化）（TEG 一面 3）

四种物理 join：

| 算法 | 复杂度 | 要求/适用 |
|---|---|---|
| Nested Loop（简单嵌套） | O(n·m) | 无索引小表，纯暴力 |
| Index Nested Loop | O(n·log m) | 内表 join 列有索引——**OLTP 首选** |
| Block Nested Loop（BNL） | O(n·m/块) | join buffer 分块读外表，减少内表扫描次数 |
| Hash Join | O(n+m) | 等值 join，小表 build 哈希表，大表 probe |
| Sort-Merge Join | O(n log n + m log m) | 等值/范围 join，输入已按 key 有序时 O(n+m) |

优化策略：**索引优化**——内表 join 列建 B+ 二级索引（覆盖索引免回表）、驱动表选**过滤后行数小**的表（小表驱动大表）、连接列类型一致避免隐式转换弃用索引；**哈希表优化**——build 侧选小表（放得进内存）、哈希函数均匀、开放/分离链表实现 cache 友好、分区并发 build；优化器层面 join reorder（Q4）。MySQL：8.0.18 起正式支持 hash join（替代 BNL），等值无索引时自动启用。

### Q3. 哈希 JOIN 中一侧表远大于另一侧，如何优化（TEG 一面 4）

前提：build 侧 = 小表。若小表也放不进内存、或 probe 大表想省扫描：

1. **Grace Hash Join（分区法）**：用 hash(key) 把**两张表都划分**成 k 个分区（相同 key 必落同分区），逐分区把小表载入内存做经典 hash join——把"随机内存需求"转成"顺序磁盘 IO"，代价是两表各读写一遍；分区仍太大则**递归划分**（再哈希换盐）；
2. **Hybrid Hash Join**：划分时第一个（热）分区的 build 表**驻留内存**直接参与 probe，省掉这对分区的落盘回读；
3. **Bloom Filter 剪枝**：build 完对 build 侧 key 建 bf，probe 侧先查 bf，不存在的行**根本不落盘/不进 join**——对大表选择性高的场景（外键 join、星型模型）减少写盘与后续分区工作量；
4. **统计/并行**：分区数取 2 的幂便于位截断；各分区独立并行 build/probe。
数据库实例：PG `work_mem` 不足自动 grace；OceanBase/Spark 的 shuffle hash join 即分布式 grace（先按 key shuffle 再各节点内存 hash join）。

### Q4. 数据库优化器的作用及实现原理（TEG 一面 5）

作用：把**逻辑执行计划**（关系代数树）变换并选择**物理执行计划**（算子算法+访问路径+join 顺序），使估计代价最小。两层实现：

- **RBO（规则优化）**：模式匹配重写——谓词下推、常量折叠、列裁剪、子查询去关联、外连接消除、视图合并、分区裁剪；
- **CBO（代价优化）**：① 逻辑变换规则生成等价计划空间（join 交换/结合）；② **代价模型**估算每个物理算子：`cost = IO + CPU + MEM(+NET)`，输入是统计信息（行数、NDV、直方图、top-n、索引信息）；③ 搜索算法在计划空间选最优：join reorder 是 NP 难——表数 ≤10 用**动态规划**（System R 起源，left-deep/tree），更多用**贪心/随机化**（Simulated Annealing，ORCA）；配合**分支限界/剪枝**（interesting order、当前部分代价超界即弃）与 memo（Cascade/Volcano 框架，group-expression 去重）。
- 产物即 EXPLAIN 看到的计划；估计失准（相关列、过期统计）是"选错计划"的根源 → hint/plan baseline 人工干预。

### Q5. EXPLAIN 语句的执行机制（TEG 一面 6）

`EXPLAIN SELECT ...` 并不真正跑查询，而是把语句走一遍 parse → resolve → **optimize**，把优化器选中的物理计划格式化输出（MySQL 每行一个算子，id 表算子树层次）。关键字段判读：

- **type**（访问路径质量）：`system > const > eq_ref（主键/唯一索引 join）> ref（普通索引等值）> range > index（扫全索引）> ALL（全表扫描，最差）`；
- **key/key_len**：实际用的索引与前缀长度；**rows/filtered**：估计行数与过滤比例（代价模型核心输出）；
- **Extra**：`Using index`（覆盖索引免回表）、`Using where`（server 层过滤）、`Using temporary`（临时表，group by/distinct 常见）、`Using filesort`（排序未走索引）、`Using join buffer (hash join)`。
进阶：`EXPLAIN FORMAT=JSON`（含完整代价）；`EXPLAIN ANALYZE`（8.0.18+，**真正执行**并输出每算子实际行数/耗时/循环数，对比 rows 估计即可定位统计失准）；optimizer_trace 看决策过程。

### Q6. Buffer Pool 的设计与实现（TEG 一面 7）

InnoDB 缓冲池 = 磁盘页（默认 16KB）的内存缓存，所有数据页读写必经：

- **组织结构**：`buf_pool` 由多个 **chunk**（连续内存）组成，实例化可拆多 pool 减少 latch 竞争；页通过**页哈希表**（space+page_no → frame）快速定位，哈希表无锁读 + 读写锁写；
- **三条链表**：**free list**（空闲页）→ 读取磁盘页时摘取；**LRU list**（已用页，见 Q8 的改进版 midpoint LRU）管理淘汰；**flush list**（脏页，按**最老修改 LSN** 排序）供 checkpoint 刷脏；
- **读写路径**：读页先查页哈希，未命中→ free list 取页 → 异步/同步读盘（预读：线性/随机预读检测）；修改页 → 变脏入 flush list，由**后台刷脏线程**（page cleaner，自适应刷脏速率防抖动）+ **自适应哈希索引**（AHI 热点页索引）；
- **并发**：页级读写 latch + 意向锁；change buffer 把二级索引修改先合并进页减少随机读；doublewrite 保证页写入原子性（Q7）。
设计要点一句话：**用页哈希做定位、用 LRU 做替换、用 flush list + LSN 做 WAL 的刷脏与恢复基准**。

### Q7. 脏页的管理策略及数据丢失问题的解决方案（redo log）（TEG 一面 8）

矛盾：事务提交要求持久化，但直接每次刷脏页随机 IO 太慢。**WAL（Write-Ahead Log）**：修改先**顺序写 redo log**（物理逻辑页修改记录），事务即算提交；脏页延后批量刷。

- **崩溃恢复**：重启后从 checkpoint 起重放 redo（幂等物理修改）到最新 LSN，未提交事务用 **undo log** 回滚——已提交不丢（redo 已落盘）、未提交不见（undo）；
- **LSN 与 checkpoint**：每个脏页记修改 LSN，flush list 按 LSN 有序；checkpoint 推进 = 最老脏页 LSN 之前 redo 可复用；redo 空间不足时强制刷脏推进 checkpoint（抖动来源）；
- **页部分写坏**：16KB 页可能只写一半（OS 页 4KB）导致恢复无法解析 → **doublewrite buffer**：先顺序写两页到共享表空间再写数据文件，坏页可从 doublewrite 副本恢复；
- 脏页管理策略：FIFO/优先刷 LSN 老/避免驱逐热点（LRU 淘汰时脏页需先刷或换出）；组合参数：`innodb_max_dirty_pages_pct`（脏页上限水位）、`innodb_io_capacity`（刷脏速率匹配磁盘）。

### Q8. LRU 的缺陷（预读失效、缓存污染）及改进方案（TEG 一面 9）

经典 LRU 两个攻击面：① **预读失效**：预读到但未命中的页排在队头，把真正热点挤到队尾（预读页若不再访问，白白污染）；② **缓存污染**：全表扫描/一次性大查询把整个表灌进池，驱逐掉业务热点（扫描一次后再也不访问）。

改进：**InnoDB midpoint LRU（5/8 规则）**——LRU 链表分 **young（新子表）/ old（老子表）**，新读入页插在 old 区头部（5/8 处）而非链表头；只有**在 old 区停留超过 1s（innodb_old_blocks_time）后再次被访问**才晋升 young——一次性扫描的页在 old 区自生自灭，不进 young。其他学术/工业变体：**LRU-K**（按第 K 次历史访问时间淘汰，抗扫描）、**2Q/ARC**（FIFO+LRU 双队列自适应比例）、**CLOCK/近似 LRU**（用引用位近似，省链表锁）、缓存污染的访问频率方案 LFU（计数衰减）+ **TinyLFU/W-TinyLFU**（现代缓存如 Caffeine）。

### Q9. 并发 B+ 树的实现机制（TEG 一面 10）

问题：多线程同时查找/插入/删除时，树的结构变更（SMO，节点分裂/合并）会让父节点指针与兄弟指针失效。

- **Latch Crabbing（锁步/蟹行）**：自根向下，先锁子节点再**放父节点**（保持"锁住的孩子+父亲"两代，children 锁住后父可释放），到叶后操作。读用共享 latch、写用排他 latch；叶层沿 next 指针遍历还要防死锁——**统一从左到右**加锁（或对右邻先 try-lock 不可得则回根部重走）；
- **乐观锁读**：假设 SMO 罕见——读路径全程共享 latch，只在最终修改时校验版本号（页面 LSN）未变，变了从根重试；
- **SMO 顺序**：分裂时**先完成新右兄弟并挂好父指针，最后改原节点**，保证任意瞬间树可遍历（与 blink 思路一致）；叶子 next 指针更新顺序保证范围扫描不丢页；
- **Blink 树**：每个内部节点加"右指针 + 高键（high key）"，查找时若 key 超过高键则顺右指针走——**写者只锁单个节点**即可完成分裂（兄弟未满时），读者几乎无锁，是 B-link/LOCK-free 方向（如 masstree/ART）的基础；
- 叶层并发：行锁（事务层）与页 latch（存储层）分离；热点页可**批量/延迟合并**（space reservation）。

### Q10. 火山模型与其他执行模型（物化模型）的比较（TEG 一面 11）

- **火山（Iterator）模型**：每个算子实现 `open/next/close`，`next` 被**拉取**一行，向上逐层迭代。优点：通用、流式、内存占用小（逐元组）；缺点：**逐元组虚函数调用**（解释器开销）、cache 不友好、无法向量化、并行度低；
- **物化模型**：算子**一次算完整个结果**再交给上层（推 push 风格）。优点：中间结果可复用（子查询/算子共享）、下游多次扫描便宜；缺点：内存暴涨、首行延迟高、流水线被打断；
- 折中与演进：**向量化（X100/MonetDB，DuckDB/OB 均采用）**——next 一次返回一批（如 1024 行）列式数据，虚调用摊薄 + 列内 SIMD；**编译执行（HyPer）**——把整条查询 LLVM codegen 成扁平循环，消除解释开销；pipeline 执行（TEG 二面追问，见 Q28）。
面试答法：火山=拉、物化=推+攒，工程主流是"**火山骨架 + 批量化（向量化）+ 热点路径编译**"。

### Q11. 行存储与列存储的适用场景（TEG 一面 12）

- **行存**：一行的所有列连续存放（+ 聚簇索引组织）。点查/范围查整行取出 O(1) IO、更新/插入就地完成 → **OLTP**（交易、状态更新）；
- **列存**：同列连续（同类型高压缩：RLE/字典/位图），查询只读需要的列（**列裁剪**），聚合扫描量成倍下降；延迟物化（late materialization）先在编码列上过滤再拼行；但整行重建、单点更新代价高 → **OLAP**（宽表聚合、扫描密集）；
- 混合与趋势：HTAP 引擎行列混合（OceanBase 行为主的 macro block + 列式投影；TiFlash 列存副本；SQL Server/Oracle double storage），增量 delta 行存 + 主存列存合并（ LSM 风格）调和更新与扫描；本质是**访问模式（tuple 宽窄读 vs 列窄读）决定布局**，与 GPU AoS/SoA 同构（[40 篇 Q29](./40-腾讯-推理与cuda.md)）。

### Q12. 事务与 MVCC 的基本概念（TEG 一面 13）

- **事务 ACID 及实现**：原子性=undo log 回滚；持久性=redo log WAL；隔离性=锁 + **MVCC**；一致性=前三者 + 约束；
- **隔离级别与并发异常**：脏读 / 不可重复读 / 幻读；RU、RC（解决脏读）、RR（InnoDB 默认，快照读解决不可重复读+大部分幻读）、Serializable（锁区间）；
- **InnoDB MVCC**：每行隐藏列 `trx_id`（最后修改事务）与 `roll_ptr`（指向 **undo log 版本链**）；读时生成 **ReadView**（活跃事务集合 m_ids、最小 min_trx_id、下一个 max_trx_id、自身 trx_id）。版本可见性判定（沿链回溯）：`trx_id < min` 可见；`trx_id >= max` 不可见；在 m_ids 中（还活跃）不可见；否则可见；
- **RC vs RR 的差别只在 ReadView 生成时机**：RC 每条语句生成（能看见别人已提交的新行→不可重复读），RR 事务首个快照读生成并复用；
- 快照读（普通 select）走 MVCC 无锁；当前读（`for update`/`update`/`delete`）读最新版 + 加行锁/间隙锁解决幻读。长事务危害：undo 链无法清理、ReadView 陈旧拖慢查询——运维必杀长事务。

### Q13. 从单机数据库扩展为分布式数据库的设计思路（TEG 一面 14）

四个正交子问题逐层设计：

1. **数据分片（sharding）**：哈希 or 范围切分（Q14），分片粒度 = 若干连续 key 区间（slot/range tablet），可分裂合并；
2. **元数据管理**：分片位置/副本/lease 的全局目录——中心式 Placement Driver（三副本共识，带租约防脑裂）或 P2P（一致性哈希 + gossip）；元数据变更走共识 + 两阶段；
3. **算子分布式执行**：生成全局计划 → 下推各分片（局部 scan/聚合/join）→ **Exchange 算子**做数据重分布（shuffle/broadcast/gather）→ 全局归并（Q18/Q19）；MPP 并行（每分片独立 pipeline，参 TiDB/OB 的 exchange）；
4. **分布式事务与一致性**：2PC（协调者+参与者，阻塞问题用 Paxos 协调者高可用解决）、Percolator 模型（中心 TSO 授时 + primary lock 提交）、OCC（乐观 + 冲突检测）；时钟：TSO 集中授时 / HLC 混合逻辑时钟 / TrueTime（Spanner 原子钟）；
5. 底座：多副本共识（Raft/Paxos，见 Q16）+ 本地存储引擎（LSM/B+，岩石 RocksDB）+ 跨机房容灾。

### Q14. 哈希分片与范围分片的区别（TEG 一面 15）

- **哈希分片**：`shard = hash(key) % N`。分布均匀、热点天然打散；但**丧失范围语义**（相邻 key 随机落盘）、range scan 要全分片扫描+归并；扩容需 rehash 大规模搬移（一致性哈希只搬 1/N，或虚拟桶/槽位表间接层解耦）；
- **范围分片**：按 key 区间切分，分片元数据记 `[start, end)`。支持 range scan/前缀查询单分片命中；写入热点在单调递增 key（自增 id/时间戳）时全打到一个分片（需预分裂+散列盐）；扩容只需**分裂**最忙分片迁移一半数据，代价小；
- 实践折中：二级组合——先哈希到固定虚拟桶（如 2^10 slot）再按 slot 区间范围管理（Redis Cluster 的 16384 slot、OceanBase table group），兼得均匀与可迁移；负载均衡器按分片大小/QPS 自动分裂搬移。

### Q15. 元数据节点的高可用方案（TEG 一面 16）

元数据（路由表/分片状态）是全局单点，必须高可用：

- **主从复制**：主节点写从节点异步复制——主宕丢未复制数据（异步丢失窗口）；半同步缓解但仍有脑裂风险；
- **共识协议（主流）**：元数据日志走 **Raft/Paxos** 多数派复制，选主由多数派裁决——单节点故障不中断、不丢已提交数据；成员变更（joint consensus/单步变更）支持运维摘除；
- **租约（lease）**：leader 持有时限租约才能服务/决策，时钟偏差上限内不会双主；follower 在租约到期后可发起选举——防网络分区时旧主复活（双写脑裂）；
- 工程：元数据三/五副本跨机架部署；**follower 读** + 缓存（客户端缓存路由，watch 失效更新）；元数据模块示例：TiKV PD（Raft + etcd）、OceanBase RootService、HDFS ZKFC+standby NN、GFS Master。

### Q16. MySQL 主从复制中如何保证强一致性（TEG 一面 17）

复制链路：主库事务写 **binlog** → dump 线程推送 → 从库 IO 线程写 relay log → SQL 线程重放。默认**异步复制**：主提交即返回，从库还没收到——主宕则丢失；且读从库是最终一致。

- **半同步复制（semi-sync）**：事务提交需至少一个从库**收到 binlog**（AFTER_SYNC，即 lossless semi-sync：等 relay 落盘后才向客户端返回）才返回——不丢已确认事务；AFTER_COMMIT 模式旧版有幻读风险（binlog 收到但未重放，客户端切从读到旧值）；
- **等位点（强一致读）**：写后立刻读从库的场景，`WAIT_FOR_EXECUTED_GTID_SET` / `sync_with_pending_gtid`——读请求带 binlog 位点，从库追上该位点才执行；或会话级 `session_sync`（5.7 之后 8.0 的 write-set/read consistency 特性）；
- **组复制 MGR / Paxos**：多数派确认 binlog（基于 Paxos变种 consensus），主宕自动选主，RPO=0；等价思路即把"复制"升级为"共识"；
- **代价与权衡**：每提交多一程网络 RTT（同机房亚毫秒、跨城几十 ms），吞吐下降——真"强一致"要支付延迟预算；全局读强一致还需 GTID 追赶读。
结论句式：异步→半同步（不丢）→位点等待（读己之写）→共识组复制（自动failover 的强一致）。

### Q17. `SELECT * FROM table` 的执行计划（TEG 一面 18）

全流程 + 计划树两个层面答：

- **流程**：连接器（认证/权限）→ 解析器词法语法（select_lex/AST）→ 预处理器（语义：表/列存在性、权限、`*` 展开为全列）→ 优化器（无 WHERE 可过滤：全表扫描代价最低；无 join 无排序，不需要选索引）→ 执行器循环调用存储引擎接口取行 → server 层按 `*` 输出全列；
- **计划树（题面给的答案）**：`Project(*) ← TableScan(table)`——投影算子在扫描算子之上；InnoDB 的 TableScan = **沿聚簇索引叶子层的双向链表顺序整页读取**（不是"没有索引"，聚簇索引就是表本身），页经 Buffer Pool；
- 细节加分：`SELECT *` 无法利用覆盖索引且列裁剪失效（网络/内存放大），生产禁用；Server 层与引擎层分工（handler 接口 `rnd_next`）；EXPLAIN 显示 type=ALL。

### Q18. 分布式场景下需要增加哪些算子来实现跨节点查询（TEG 一面 19）

在单机算子之上增加**数据移动（Exchange）算子族**：

- **ExchangeSender/ExchangeReceiver（收发对）**：计划树的"切分线"，按下游需要的分布重排数据。分布方式：**Gather**（各分片→汇总节点）、**RePartition/Hash Shuffle**（按 hash(key) 发往对应分片）、**Broadcast**（小表全量复制到各分片）、**RoundRobin**（负载均衡摊开）；
- 依赖 Exchange 的改造：**分布式聚合**=局部聚合 → shuffle by group key → 全局聚合；**分布式 join**=两表 shuffle by join key（shuffle/partisan join）或小表 broadcast（broadcast join）；**分布式排序**=局部 sort → 归并 gather（Merge Gather）；
- 其他配套：限制算子的分片传播（谓词下推到 scan）、DFO（Data Flow Operation）调度、每算子支持"部分/最终"两态（partial/final aggregation）。
对应实现：TiDB ExchangeExecutor、OB 的 EXCHANGE IN/OUT、PG-XC/Citus 的 tuple routing。

### Q19. 带 WHERE 条件和 GROUP BY COUNT 的分布式执行计划设计（TEG 一面 20）

以 `SELECT g, COUNT(*) FROM t WHERE x > 10 GROUP BY g`、表按非 g 列分片为例，计划自底向上：

```
各分片（DOP=n）:  Scan(t) → Filter(x>10)          -- 谓词下推，先过滤
                → PartialAgg(group by g, count_局部)   -- 局部聚合压缩数据量
Exchange:        HashShuffle by g                  -- 同组数据路由到同分片
全局层:           FinalAgg(COUNT = Σ count_局部)      -- 合并局部状态
                → Gather(合并到协调节点) → Sort/Project/客户端
```

设计要点：① 两阶段聚合的**正确性条件**：COUNT/SUM/MAX/MIN 可加，AVG 必须拆 SUM+COUNT 再除；② shuffle 选 hash(g) 保证同组同分片，才能全局聚合；③ 若 GROUP BY 列恰是分片键，可省 shuffle（one-phase local agg）；④ 数据倾斜：热 group 再加盐二次聚合；⑤ 谓词下推 + 列裁剪让网络只传 (g, 1)；⑥ Filter 选择率低时可大幅缩小 exchange 数据量——这就是"算子下推"的价值叙事。

## 二、操作系统（一面 Q21-Q26）

### Q20. Linux 系统的启动过程（TEG 一面 21）

四级接力：

1. **固件（BIOS/UEFI）**：上电 → CPU 从复位向量执行固件 → POST 自检 → 枚举启动设备；UEFI 比 BIOS 多 GPT/安全启动/直接读 FAT32 的 EFI 分区；
2. **Bootloader（GRUB2）**：载入内核镜像与 initramfs 到内存，传命令行参数（root=/ rw quiet...），跳转内核入口；
3. **内核初始化（start_kernel）**：解压 vmlinuz → 架构初始化（页表/mm/sched/中断/时钟/RCU）→ 驱动 probe（或 initramfs 中的驱动）→ **挂载真实 rootfs**（initramfs 的使命：含 root 所在磁盘/网络/加密驱动，pivot_root 切换）→ 启动 1 号进程；
4. **用户空间**：PID 1 = systemd（传统 sysvinit）→ 按 target/unit 拉起服务（网络/ssh/日志）→ getty/login → shell。
面试深挖点：为什么需要 initramfs（根设备驱动鸡生蛋）、内核命令行参数、`dmesg` 看启动日志、systemd 并行化与 cgroups。

### Q21. 中断处理模块的实现机制（TEG 一面 22）

- **入口**：设备发中断 → 中断控制器（x86: APIC/IOAPIC，ARM: GIC）按路由投递到某 CPU → CPU 查 **IDT**（中断描述符表）拿到入口 → 保存被打断上下文（pt_regs）进内核栈 → 切到内核态执行 handler；
- **上半部（hardirq）**：关本地中断、快速处理（应答硬件、拷贝数据、标记），不可睡眠；
- **下半部（延迟处理）**：**softirq**（静态类型，网络/块层，同类型可多核并行）、**tasklet**（同类型串行，渐淘汰）、**workqueue**（内核线程执行，**可睡眠**）、threaded IRQ（handler 只唤醒内核线程）；
- **注册**：`request_irq`/`devm_request_threaded_irq` 绑定中断号与处理函数；中断亲和性（smp_affinity）定向到某核；`/proc/interrupts` 观测；
- 与 AI Infra 关联：GPU 驱动的 doorbell/完成中断、NVMe 完成队列中断、网卡 RSS 多队列中断分流都是同一机制；polling（忙轮询/混合中断+轮询）是高吞吐路径的低延迟优化。

### Q22. CPU 如何处理多个并发中断（TEG 一面 23）

- **排队与嵌套**：每 CPU 有 pending 位图/队列；执行 hardirq 时本地中断**默认关**（IF=0），同类不可重入；**更高级别中断（如 NMI）可嵌套**打断当前 handler；softirq 在开中断环境执行可被新中断打断；
- **屏蔽层次**：spin_lock_irqsave 关本地中断保护共享数据；中断控制器可 mask 单个 IRQ 线；多核下同 handler 可并行（按亲和性分发），所以中断数据结构要 per-CPU（软中断统计、RCU 就是为此设计）；
- **分发策略**：APIC 把设备中断路由到指定核；RSS 网卡多队列 → 多核并行；中断风暴时切轮询（NAPI：先中断后 poll 预算）；
- **不能丢的场景**：中断到来时在跑另一个中断 → 记 pending，开中断后补执行（retrigger）；等待期间多次同类中断合并成一次处理（网络收包合并）。

### Q23. 进程与线程（包括用户态线程）的区别（TEG 一面 24）

- **进程**=资源分配单位（地址空间/文件表/信号处理独立）；**线程**=调度执行单位（共享进程资源，各有栈/寄存器/TLS）；内核线程（1:1，Linux `task_struct` + CLONE_VM）切换走内核调度，可真并行；
- **用户态线程（协程）**：在用户空间库调度的执行流——切换不进内核（保存少量寄存器即可，~10ns 级 vs 内核线程 ~1-10μs）；分**有栈**（goroutine/libco/boost.coroutine）与**无栈**（C++20 协程/JS async，状态存闭包）；
- **映射模型**：N:1（全部用户线程挂一个内核线程，一个阻塞全体阻塞，已被弃）、1:1（pthread）、**M:N**（goroutine 的 GMP：M 内核线程 + P 调度上下文 + G 协程，P 本地队列 + work stealing）；
- 核心权衡：用户态线程便宜（百万级并发）但**不能独立利用多核**（须挂多个内核线程）且**阻塞系统调用会把宿主内核线程一起挂住**（须 hook 系统调用/加 nonblocking+epoll 池）。

### Q24. 用户态线程的调度方式及可能存在的问题（TEG 一面 25）

- **调度方式**：① **协作式**——代码显式 `yield`/IO 点让出（libco/python 旧生成器）：实现简单但一个线程死循环整核卡死；② **抢占式**——Go runtime 的 sysmon 后台监测 G 运行超 10ms 或处于系统调用过久，发起抢占（改写其执行标志/发送信号，在**函数序言/循环回边 safepoint** 生效）；
- **问题清单**：① 阻塞系统调用/缺页陷阱导致宿主内核线程挂起 → 整 M 上的 G 队列全停（解法：handoff M，把 P 移交别的 M；hook 阻塞调用改 epoll）；② 无时钟抢占的协作实现会饿死其他协程；③ 栈管理：有栈协程要可增长栈（分段栈/连续栈拷贝）；④ 与宿主线程库交互（TLS、信号、mutex 未让出）；⑤ 调度公平性与局部性（NUMA/work-stealing 的迁移成本）；⑥ 与 IO 多路复用的集成（把 yield 点接到 epoll，如 goroutine netpoller）。

### Q25. 用户态到内核态的切换过程（TEG 一面 26）

以系统调用为例（x86-64 Linux）：

1. **触发方式**：`syscall` 指令（软件陷入）/ 中断 / 异常（缺页、除零）；
2. **入口准备**（CPU 硬件 + MSR）：`syscall` 从 MSR（`IA32_STAR/LSTAR`）取内核入口 RIP 与段选择子，**硬件保存 RIP→RCX、RFLAGS→R11**，特权级 3→0，栈切到该 CPU 的内核栈（TSS rsp0）；
3. **软件保存现场**：入口 stub 把参数寄存器外的 caller-saved 寄存器压入内核栈形成 **pt_regs**；SYSCALL_WRAPPER 框架按系统调用号（rax）查 `sys_call_table` 跳转实现；
4. **内核执行**：此时可睡眠/调度/阻塞——调度发生时 pt_regs 连同内核栈一起被切走；
5. **返回**：`sysret`（或 iretq）恢复 RCX/R11 的 RIP/RFLAGS，回到用户态原地址继续；
- **成本**：直接开销几十到百 ns（无地址空间切换，页表不变——内核空间共享映射）；KPTI（熔断缓解）给每次切换加 CR3 切换 + PCID，成本翻数倍；上下文切换（进程级）另加页表切换 TLB 冲刷。深挖：vDSO 把 gettimeofday 等只读调用**免陷入**完成；io_uring 减少系统调用次数是同一经济学。

## 三、TEG 二面：数据库工程实践（MiniOb/OceanBase/xv6）

### Q26. 基于 MiniOb 阐述一条 SQL 查询的完整执行流程（TEG 二面 2）

MiniOb（OceanBase 开源的教学数据库，全国数据库大赛题目）一条 `SELECT ... WHERE ...` 的流水线：

1. **词法/语法分析**：lex/flex 产 token，yacc/bison 按文法归约出 **AST/Stmt**（SelectStmt）；
2. **语义解析（resolve）**：查 sys 表/Schema 校验表、列存在性，解析 `*`、别名，绑定列到表与类型；
3. **逻辑计划生成**：Stmt → LogicalPlan 算子树（TableScan → Filter → Project），表达式规范化；
4. **优化**：MiniOb 基础版近乎 pass-through，可自行加 RBO（谓词下推、常量折叠、投影列裁剪），进阶加 CBO；
5. **物理执行**：PhysicalPlan → 火山式算子 `open/next/close` 逐元组拉取；
6. **存储层**：TableScan 调 RecordHandler（buffer pool 管理页 → 文件 I/O），Filter 用 B+ 索引（IndexScanner 范围扫描）或全扫，Tuple 组装行值；
7. **返回**：ResultSet 序列化写客户端协议（miniob 自定义通信协议）。
对照 MySQL：parser（bison 同款血统）→ resolver → optimizer → executor → InnoDB handler 接口，层次一一对应——这是"读过一个真实内核"的展示题。

### Q27. 查询过程中最影响性能的关键环节（TEG 二面 3）

按代价从大到小排：① **优化器选错计划**（join 顺序组合爆炸、统计失准）——差计划可慢几个数量级，影响最大；② **随机 I/O 与 Buffer Pool 命中率**（回表、索引缺失、缺页风暴）——机械盘两个数量级差，SSD 也在；③ **执行器逐元组开销**（虚函数、表达式解释、无向量化）——CPU bound 场景 5-10 倍空间；④ 锁/latch 等待与日志刷盘（WAL fsync 延迟，组提交缓解）；⑤ 网络往返与应用层。定位手段：`EXPLAIN ANALYZE` 看每算子实际耗时、performance_schema、慢查询日志——**先量化再归因**。

### Q28. 火山模型与 Pipeline 模型的关系（TEG 二面 5）

- 火山是**算子接口协议**（拉取、逐元组），pipeline 是**执行调度组织方式**：把计划树按"**pipeline breaker**"（阻塞算子：sort/hash-build/materialize，必须收齐才能吐）切成若干条 **pipeline**（一段无需阻塞的算子链），每条 pipeline 内可按 morsel（数据分片）**推模式并行**，pipeline 之间在断点汇合再驱动下一条（HyPer 的 morsel-driven 并行）；
- 关系总结：pipeline 把火山树的"逐层拉"改造成"**分段推 + 段内并行**"，算子仍可保留 open/next 骨架（batch 化的 next），但调度单位从"一次 next"升为"一条 pipeline 上的 morsel 流"；
- 收益：核间并行不靠算子内部开线程（避免算子内并行度叠加）、cache/NUMA 亲和（morsel 本地处理）、阻塞点显式化便于 spill/调度；代价：调度器复杂、断点处仍要物化。
对应实现：HyPer 论文（morsel-driven parallelism）、DuckDB 的 push-based execution、OceanBase 的 DFO/pipeline 引擎（Px 并行）。

### Q29. 突破现有执行模型瓶颈的优化方向（TEG 二面 6）

题面已给两个方向（CRTP 减少虚函数、物化批量返回），展开成体系：

1. **批处理/物化**：next 返回一批（百-千行）而非一行，虚调用与解释开销摊薄，行存转段内列式对 SIMD 友好（向量化执行，DuckDB/OB 2.0+）；
2. **消除虚函数**：**CRTP** 静态多态（模板在编译期绑定，见[09 篇 Q9](./09-cpp与系统基础.md)）、LLVM **codegen**（HyPer：把算子链编译成扁平循环，解释完全消失）；
3. **并行**：morsel/pipeline 级并行（Q28）、NUMA 感知分片；
4. **编译期特化**：按查询 plan 即时生成机器码（含常量折叠的类型特化），JIT 表达式求值；
5. **异步 I/O/预取**：scan 与计算重叠（uo_pread/io_uring 批量预读）；
6. **近似/下推**：聚合下推到存储层（在页上直接算）、采样估计。
答题结构：先指出瓶颈本质（**解释开销 + 逐元组 cache 不友好 + 串行**），再给对应技术，标注工业案例（X100/HyPer/DuckDB/OB pipeline）。

### Q30. OceanBase 项目的优缺点分析（TEG 二面 7）

- **优点**：① 原生分布式架构（share nothing，Paxos 多副本，**RTO < 8s** 的金融级高可用）；② **HTAP 一体**：一套引擎行存为主的 micro block（列式投影），TP 与 AP 混合负载；③ 线性水平扩展（增删节点自动负载均衡、分片自动分裂）；④ 高压缩比存储（encoding + 列压缩，成本优势宣传 70%↓）；⑤ MySQL/Oracle 兼容模式 + 多租户资源隔离（单集群多业务）；⑥ 经过蚂蚁/银行大促交易峰值验证（TPC-C/TPC-H 榜）；
- **缺点/权衡**：① 部署运维复杂（obproxy/OBServer/oms/ocp 组件链长，需要 OCP/OMS 配套，学习曲线陡）；② 小规模场景（单机/两节点）资源开销与复杂度不占优（最小分布式单元 3 副本）；③ 生态（第三方工具/人才池/社区案例）弱于 MySQL/PG；④ 准确说"兼容 MySQL"而非"就是 MySQL"，边角语法/行为差异带来迁移成本；⑤ 社区版与企业版功能差异（监控/备份/安全）。
答题技巧：先分类（架构/功能/工程/生态）再说优缺点，落在"分布式收益 vs 复杂度代价"的权衡框架。

### Q31. 介绍 xv6 项目及其带来的收益（TEG 二面 8）

xv6 = MIT 6.1810（原 6.S081）课程教学用 Unix（约万行 C/RISC-V，2006 年重写的 V6），覆盖：系统调用陷入、进程与调度、页表与地址空间、trap（中断/异常/系统调用统一路径）、文件系统（inode/log 崩溃一致性）、锁与多核、（新版本）网络栈与 virtio 驱动。

- **价值点**：① 一万行读"完整"内核——Linux 两千万行没法整体理解，xv6 能建立**全链路心智模型**（一次 write() 从 syscall 到磁盘中断）；② lab 实战：COW fork、lazy allocation、多线程/mutex、日志文件系统、网络——每个 lab 都是真实内核难点；③ 面试素材密度高：能对照讲"Linux 的实现与 xv6 差在哪"（如 xv6 无 per-CPU 变量/无抢占内核/xv6 调度为简单轮转、Linux CFS/EEVDF）；
- **答法模板**：项目做了哪些 lab → 收获（trap 上下文、页表自映射、崩溃一致性 log）→ 迁移到工作中的能力（读内核代码、排查系统级 bug 的自信）。

### Q32. 项目经历与协作类问题（TEG 一面 1 / 二面 1 / 三面 1-2）

软性问题，结构化作答：

- **介绍项目/分工**：STAR（Situation 任务背景 - Task 我负责的模块 - Action 技术方案与关键决策 - Result 量化结果）。分工题说清**模块边界**（我做什么、队友做什么、接口怎么定）+"个人贡献可量化"（性能数字/通过率/代码量）；
- **最影响性能的环节/是否有优化器设计**：见 Q27/Q4，把通用答案落到自己项目的具体模块；
- **团队分歧**：① 先对齐目标与约束（不是对齐方案）；② 用**数据**裁决（benchmark/实验/论文），小成本 spike 验证；③ 都可行时按可回滚性/维护成本选；④ 仍僵持则引入更高层决策并公开记录 trade-off；⑤ 事后复盘沉淀为规范。忌答"我一般都听别人的"或"我说服所有人"。

## 四、TEG 编程题

TEG 三道编程题（一面 27 内存文件系统/字典树、二面 9 快慢指针+反转链表、三面 3 大写转小写排序）与另 6 份腾讯面经的编程题统一在[42 篇](./42-腾讯-编程题.md)解答，代码在 `solutions/python/src/answers/tencent/`，`pytest tests/test_tencent.py -v` 一键验证。
