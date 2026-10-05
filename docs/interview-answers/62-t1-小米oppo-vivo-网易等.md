---
title: "T1 大厂 · 小米 / OPPO / vivo / 网易 / 拼多多 / B站"
description: "小米手写 MHA 与 AllReduce 通信量、SGLang/MTP、膨胀卷积；OPPO Stream 与 NHWC、vivo Halide/TVM、网易 C++/CUDA 全家桶、拼多多 Triton vs CUDA、B站协程与归约的面试题解"
pubDate: 2026-10-05
topic: "公司题解-T1"
order: 62
tags: ["面试解答", "小米", "OPPO", "vivo", "网易", "拼多多", "B站", "算子优化", "编译器"]
draft: false
---

覆盖题库（来源缩写）：`米一`=小米-AI-Infra-一面、`米实一二`=小米-实习-一二面、`米实1`=小米-实习-一面-1、`米校`=小米-校招、`米校面`=小米-校招-一面；`O二`=OPPO-实习-二面、`O云一`=OPPO-云-实习-一面；vivo=vivo-校招；网易=网易-校招；`拼`=拼多多-AI-Infra；`B一`=B站-实习-一面。

基础同族题全部引用既有篇目：CUDA 基础 [01](./01-cuda-基础与体系结构.md)、编程题 [02](./02-cuda-编程题.md)、算子优化 [03](./03-gemm与算子优化.md)、Attention [04](./04-attention与kvcache.md)、推理引擎 [05](./05-推理引擎与优化.md)、量化 [06](./06-量化.md)、分布式 [07](./07-分布式训练.md)、编译器 [08](./08-编译器与性能分析.md)、C++ [09](./09-cpp与系统基础.md)、训练 [11](./11-深度学习与训练.md)、算法 [12](./12-算法题.md)。新代码：`answers/t1/dilated_conv.py`。

## 一、小米

### Q1. 手写 MGA（Multi-head Group Attention）与手写 MHA（米校 编程 1/2、拼 编程 9）

MGA 即 GQA 的别名（Grouped/GMHA），仓库实现：`answers/attention/mha.py::MultiHeadAttention`、`answers/attention/gqa.py::GroupedQueryAttention`（含与 MHA 在全共享时等价的对拍 `test_attention.py`）。手写要点（按面试评分点）：

1. `__init__`：`Wq: (n_q*d, d_model)`、`Wk/Wv: (n_kv*d, d_model)`（GQA：n_kv < n_q）、`Wo`；GQA 的 `group_size = n_q // n_kv`；
2. forward：投影 → reshape `(B, S, H, d)` → transpose；KV 广播用 `repeat_interleave(group_size, dim=1)`（或 kernel 内索引 `kv_head = q_head // group_size`）；
3. 注意力：`scores = qk^T/sqrt(d)` → causal mask（上三角 -inf）→ softmax → 加权 V → 合并 head → Wo；
4. 追问加 KV Cache/增量 decode、RoPE 位置编码、数值稳定（softmax 前减 max / 用 fp32 累加）。
5. 只写流程不给代码的兜底答法：embedding → 每 head 线性投影 → 缩放点积（causal）→ concat → 输出投影；GQA 差异只在 KV 头共享与广播。

### Q2. vLLM 与 SGLang 的底层源码架构（米校 推理 3）

- **vLLM**（[05](./05-推理引擎与优化.md) Q2）：分层——API server / scheduler（iteration 级、prefix-aware、抢占）/ block manager（PagedAttention 显存管理）/ worker（模型执行、自定义 CUDA/Triton kernel）/ connector（多卡 TP-PP）。V1 重写：一步调度 + async scheduling（CPU 调度与 GPU 计算重叠）+ piecewise CUDA Graph；
- **SGLang** 特性：①**RadixAttention**——以 radix tree 管理**所有请求的 KV 前缀复用**（不止 system prompt，任意共享前缀/分支采样/多轮），节点即 KV 块、LRU 逐出；②**零开销 CPU 调度器**（runtime 与 python 解耦的 futures/override 机制，消除 GIL/解释器开销）；③**overlap scheduler**——CPU 准备下一 step 与 GPU 执行当前 step 流水（vLLM V1 的 async scheduling 同思路）；④前端 DSL（`gen/select/fork`）表达约束生成，编译到 compressed FSM；⑤投机解码（EAGLE）与 PD 分离支持；
- 对比一句话：vLLM 强在 PagedAttention 生态与调度通用性，SGLang 强在 KV 复用数据结构（radix tree）与运行时去 python 化；两者都在向 piecewise graph + 异步调度收敛。

### Q3. KV Cache 的加载方式（米校 推理 4）

①**随请求在线生成**（默认 prefill 算出）；②**prefix cache 命中复用**（块级引用计数，vLLM/SGLang radix tree）；③**CPU swap**（显存不够换出、需要时换回，vLLM swapping）；④**磁盘/对象存储分层 + 异步预取**（Mooncake KV 池，见 [61](./61-t1-蚂蚁华为京东.md) Q22）；⑤**跨节点传输加载**（PD 分离的 prefill→decode KV 迁移，RDMA 直写目标卡的 paged buffer）；⑥**离线预计算**（文档/系统提示词预跑落盘）。要点：加载要与布局（block size/dtype/量化）一致 + 与计算重叠。

### Q4. PD 分离的作用与实现（米校 推理 5）

见 [60](./60-t1-快手与美团.md) Q1（作用与 AF 分离）与 [61](./61-t1-蚂蚁华为京东.md) Q48（调度队列实现）、[05](./05-推理引擎与优化.md) Q5。小米追问实现细节：入口路由 + KV 传输引擎（分块 RDMA、与 prefill 重叠）+ staging 队列 + 故障回退。

### Q5. vLLM 部署实测吞吐（米校 推理 6）

给口径与量级（面试要的是"你真的跑过"）：Qwen2.5-7B-Instruct，A100-80G×1，vLLM 0.6+，`--gpu-memory-utilization 0.9 --max-model-len 8192`：并发 32、输入 1K/输出 256 → 总吞吐 ~2000-3000 tok/s（batch 越大越接近 1/batch 的线性放大，decode 每 token 每 batch 读一遍权重）；TTFT p99 < 500ms。3090/24G 上 7B-AWQ：~1000-1500 tok/s。影响因素排序：batch 上限（KV 水位）> 量化（W4 带宽 ¼）> TP（受通信）> 调度策略。测量方法：`vllm bench serve`（RPS 扫描出 goodput 曲线）或自压（记录 TTFT/TPOT 分位数）。

### Q6. MTP（Multi-Token Prediction）机制（米校 推理 7）

DeepSeek-V3 的 MTP：主干预测 token t+1 之外，接 **1-2 个串联的预测头**（每头一层独立 transformer + 共享主干的倒数第二层隐状态 + 残差），训练时同时预测 t+2、t+3（深度监督，token 级密度↑）。**训练收益**：更强的表示（未来目标逼着隐状态前瞻）、数据效率提升。**推理收益（投机解码）**：MTP 头当 draft 模型——一次前向出 k 个候选 token，主干并行验证（一个 batch 里同时算 k 步的 logits，接受前缀最长的候选），接受率通常 70-85%，decode 从每步 1 token 到 1.5-2+ token/step。对比 EAGLE/Medusa：Medusa 是并行多头（加多个头）、EAGLE 用隐状态自回归（更准）、MTP 串联头训练推理一体且可当正式能力。加速上限 = 每 step 验证多 token 的并行度 − 无效计算比。

### Q7. YOLO 剪枝与部署（米一 项目 3）

- **剪枝**：结构化通道剪枝为主——BN γ 系数 L1 排序（Network Slimming：γ 加正则后接近 0 的通道删）、按敏感度逐层回退（每层剪 10/30/50% 各测 mAP，敏感层少剪）；配合蒸馏恢复（剪后 teacher-student 微调）；YOLOv5/8 的 torch.prune 或自研 mask 训练；
- **部署链**：PyTorch → ONNX（opset 17，静态/动态 shape）→ TensorRT（FP16/INT8 PTQ 校准：entropy/minmax，几百张校准图）→ engine 序列化；预处理（letterbox/NHWC/归一化）与 NMS 的 plugin 化（或移到后处理）；
- **加速技术**：INT8（~2×）、层融合（conv+BN+SiLU）、NMS plugin（batched NMS kernel）、动态 batch（profile）、CUDA Graph 封推理段；常见坑：ONNX 算子不支持（改写/自定义 plugin）、INT8 精度回退层（混合精度 per-layer）。

### Q8. 模型压缩技术综述：蒸馏与量化（米一 推理 5）

蒸馏（[11](./11-深度学习与训练.md) Q6）：logits 蒸馏（KL(T)，T=2-4 软标签暗知识）/ 中间层对齐（MSE/attention map）/ 生成式蒸馏（MiniLLM/KD for LLM，序列级）；小模型+大数据+教师集成收益最大。量化（[06](./06-量化.md) 全景）：PTQ（RTN/GPTQ/AWQ/SmoothQuant）vs QAT；W4A16/W8A8/KV FP8。组合技术：剪枝+蒸馏+量化三件套（典型 CV 流水线：剪 30% → 蒸馏恢复 → INT8 部署，总加速 3-5×）。

### Q9. Python 深浅拷贝（米实一二 一面 4）

`copy.copy`（浅：新容器、共享元素引用——嵌套 list 内层共享）vs `copy.deepcopy`（深：递归复制，处理循环引用 via memo）。特例：不可变对象（tuple/int）拷贝常返回自身；`list[:]`/`dict.copy()` 是浅拷贝。必须深拷贝的场景：嵌套结构要独立修改内层（如配置树、batch 数据的增广副本）。C++ 同族（网易 Q4）见 Q28。

### Q10. C++ 智能指针与设计动机（米实一二 一面 5、二面 23、网易 C++ 6）

见 [09](./09-cpp与系统基础.md) Q3：`unique_ptr` 独占（move-only、零开销）；`shared_ptr` 引用计数（控制块、原子计数线程安全但对象不安全）；`weak_ptr` 破循环引用/观察者。**shared_ptr 的设计动机**（小米原话问）：所有权不明确时自动生命周期管理——多个所有者共享、最后一个释放（RAII 推广），替代手写 new/delete 的泄漏/悬挂；代价：控制块开销 + 原子计数 + 无法直接表达"独占"（故先选 unique_ptr）。循环引用场景：A、B 互持 shared → 计数永不归零 → 一方改 weak_ptr，用前 `lock()` 提升。

### Q11. 写时拷贝 COW 与零拷贝（米实一二 一面 6/7）

- **COW**：复制时只复制指针/页表并标记只读，首次写入触发缺页/异常再真复制（Linux fork 的页级 COW；`std::string` SSO 时代的 COW 已被 C++11 禁止——多线程下引用计数开销与 iterators 失效）；实现三件套：引用计数 + 只读标记 + 写前检查复制；
- **零拷贝**：数据不经用户态中转——`sendfile`（页缓存→网卡，FTP/静态文件）、`mmap`（文件映射进用户态省一次 copy）、`splice`（管道缓冲挪移）、RDMA/GPU 的 GPUDirect（网卡↔显存直写）；DPDK 旁路内核。收益：省 CPU copy 带宽与上下文切换；深度学习里的零拷贝：pinned memory + 异步 H2D、NVLink P2P 访问他卡显存、CUDA Unified Memory 按页迁移。

### Q12. 矩阵转置编程题（米实一二 一面 8）

`answers/algorithms`…CUDA 版：`solutions/cuda/src/transpose.cu`（naive vs smem+padding 消 bank conflict，2048²/4096² 实测带宽对比，[02](./02-cuda-编程题.md) Q4）。要点：合并访存（按 tile 转 smem 再写出）、padding（+1 列）打散 bank 冲突、L2 命中对测量的影响（小矩阵掩盖差距的现象本身就是考 点）。

### Q13. 大模型分布式训练完整流程与并行策略选择（米实一二 二面 9/10）

流程：资源/目标评估（参数量→显存下界、目标 tokens→算力）→ 并行设计（见 [61](./61-t1-蚂蚁华为京东.md) Q29 的 3D 组合：TP 机内、PP 跨机、DP 外层、MoE 加 EP）→ 框架落地（Megatron/DeepSpeed 配置：micro-batch、重算策略、通信桶）→ 数据与 loss（序列打包、梯度累积等效 batch）→ 稳定性（loss scale、梯度范数监控）→ 起训前 dry-run（单步显存/MFU 预估）→ 弹性与断点（异步 checkpoint）。选择依据一句话：**单卡放不下→TP；单机放不下→PP；还要扩→DP/ZeRO；MoE→EP；长序列→CP**。

### Q14. TP 为何有按行/按列两种切分（米实一二 二面 11）

见 [07](./07-分布式训练.md) Q4：$Y=XA^T$ 列切（输出分片免通信）vs $Y=AX$ 行切（部分和需 AllReduce）；Megatron 组合（attention QKV 列切+O 行切、FFN up 列切+down 行切）让每层只 2 次 AllReduce；推导式回答见 [61](./61-t1-蚂蚁华为京东.md) Q36 的 4 卡 A×B。含义：切分方向决定通信原语（AG vs AR）与中间结果是否需要物化。

### Q15. Megatron 序列并行 SP（米实一二 二面 12）

见 [07](./07-分布式训练.md) Q5：TP 基础上把 LN/Dropout 等非权重算子的**序列维**切到各卡，AllReduce 拆成 RS+AG（总通信量不变），激活显存 /tp；与 context parallel 的区别（SP 管 LN 激活、CP 管 attention 长序列）。

### Q16. Transformer 层算子与 Encoder/Decoder 差异（米实一二 二面 13/14）

算子清单：embedding 查表、RoPE（三角函数/复数乘）、RMSNorm、QKV GEMM、attention（bmm+softmax，kernel 化为 FA）、O 投影、residual add、FFN（gate/up/down GEMM + SwiGLU）；解码端另有采样（top-k/p、温度）。Encoder：双向注意力、无 KV cache、判别任务；Decoder(-only)：causal mask、自回归、KV cache、生成；经典 encoder-decoder（T5）另有 cross-attention（Q 来自 decoder、KV 来自 encoder）。差异表格见 [61](./61-t1-蚂蚁华为京东.md) Q32（T5 vs GPT-2）。

### Q17. FA 核心思想与 Online Softmax（米实一二 二面 15/16）

见 [04](./04-attention与kvcache.md) Q1 与 [02](./02-cuda-编程题.md) Q2：online softmax 分块维护 $(m, l)$ 并重缩放累积输出，公式 $\text{out}_j = \frac{\sum e^{x-m}}{l}$ 的流式更新（$m_{new} = \max$、缩放因子 $\alpha = e^{m_{old}-m_{new}}$）；FA = online softmax + 分块 QK/PV + 寄存器驻留，访存 $O(S)$。代码 `answers/attention/online_softmax.py、flash_attention.py`。

### Q18. CUDA 优化方法与访存优化、计算访存重叠（米实一二 二面 18/19/20、B站 理论 5）

见 [01](./01-cuda-基础与体系结构.md)/[03](./03-gemm与算子优化.md) 汇总：访存优化——合并访存（128b 向量化 float4）、smem tiling、常量/纹理/`__restrict__`（L1/L2）、bank conflict 消除（padding/swizzle）、occupancy 与延迟隐藏（足够 warp 数）、异步拷贝（cp.async/TMA）双缓冲；计算优化——寄存器 tiling、指令吞吐（避免超越函数、用 `__expf`）、ILP；**重叠**——双缓冲（load 与 compute 流水）、多流（H2D 与计算）、kernel 内 warp specialization（生产/消费者）、CUDA Graph 降 launch 间隙。B站追问"确认瓶颈后针对性调优"见 Q35。

### Q19. L1/L2 区别与 smem/L1 关系（米实一二 二面 21/22）

- **L1**：每 SM 私有 (~256KB Ampere 中与 smem 共享物理)，延迟 ~30 cycle，容量小、自动管理（透明缓存，命中率不可控）；**L2**：全卡共享（A100 40MB），延迟 ~200 cycle，所有 SM 一致视图（也是卡内通信的中转——NVLink P2P 经 L2）；
- **smem 与 L1 同一块物理 SRAM**（按需划分 `cudaFuncSetAttribute`）：smem 是**编程显式管理**的部分（可控延迟、可做线程间交换与同步），L1 部分自动缓存 global；bank 结构只作用于 smem 部分；用哪个：需要跨线程共享/确定性延迟/复用模式明确 → smem；只读复用随缘 → L1 足矣。

### Q20. Block 是软件还是硬件概念（米实一二 二面 17）

**软件概念**：编程模型的组织单位（≤1024 线程 + 共享 smem + `__syncthreads`）；硬件侧只是被 SM **调度驻留**（SM 可放多个 block、block 不跨 SM）——没有叫 "block" 的硬件部件，对应关系是 block→SM 的驻留与资源占用（寄存器/smem 决定每 SM 放几个）。thread 同理是软件概念，**warp 才是硬件调度实体**（对应 warp scheduler）。

### Q21. AllReduce 实现方式、Ring 通信量、Tree 优势（米实一二 二面 24/25/26）

见 [07](./07-分布式训练.md) Q1 完整推导：Ring（RS+AG 两阶段，每卡通信 $2\Psi\frac{N-1}{N}$ 与卡数无关、带宽最优）；Tree（$O(\Psi\log N)$、延迟低，小消息快）；NCCL 按消息大小自动选 Ring/Tree/双环，另有层次（NVLink 域+域间）与 SHARP（交换机内归约）。追问应对：写 Ring 单卡发送量推导（N-1 轮 × Ψ/N 两阶段）；Tree 的优势答"深度 log N、小消息延迟主导时更快，但带宽利用不如环流水"。

### Q22. 膨胀卷积手写与优化（米实1 编程 5）

新代码 `answers/t1/dilated_conv.py`：`dilated_conv2d_naive`（六重循环讲语义：感受野 $k+(k-1)(d-1)$，参数/计算量不变）+ `dilated_conv2d_unfold`（im2col→GEMM，`F.unfold` 自带 dilation/padding/stride），与 `F.conv2d(dilation=d)` 三方对拍（pytest）。**优化讨论**：①im2col+GEMM 吃 cuDNN/ cutlass；②膨胀卷积 gather 地址跨步 d，合并访存差——GPU 上按 output tile 分 block、行主序预计算索引，或等价转为 dense conv（对小 d 填零膨胀 kernel，trade 计算换访存）；③Winograd/FFT 不适配非连续采样（推导前提破坏）；④DeepLab 场景的 multi-grid（1,2,4）级联感受野公式（`receptive_field_two_layers`：$1+\sum (k-1)d_l$，(1,2)→7）。对拍与感受野断言在 `tests/test_t1.py`。

### Q23. Fastllm 核心技术（米校面 推理 1）

纯 C++ 无 python 依赖的 LLM 推理框架（面向 CPU/ARM 与低端 GPU）：①**量化**：INT4/INT8 weight-only（运行时反量化与计算融合、per-layer/per-channel 混合）；②**ARM NEON/AVX 向量化**（int4 解包用 SIMD 位操作，dot product 用 VSDOT/SDOT）；③算子手写（matmul 分块 + cache 友好布局、attention 的 KV 增量计算）；④内存池 + 线程池（无锁任务队列）；⑤支持 GPU（CUDA/OpenCL）统一抽象。定位对比：vLLM（GPU 服务化吞吐）vs llama.cpp/Fastllm（端侧/CPU 生态、int4 生态成熟）。答"读过源码"要有细节锚点：量化 pack 布局、NEON 解包循环、thread pool 的 shard 策略任选其一展开。

## 二、OPPO

### Q24. 算子融合何时有收益（O二 理论 2）

见 [03](./03-gemm与算子优化.md) Q6 与 [61](./61-t1-蚂蚁华为京东.md) Q13：判据（访存密集链、单产单消、高频、tile 兼容）；典型场景（norm+act、gemm+bias+act、decode 的 add+norm+rope 链）；反例（中间结果多分支引用、融合后 occupancy 掉）。

### Q25. CUDA Stream 异步原理与前提（O二 理论 3）

Stream = GPU 命令队列，launch 是**异步入队**（CPU 立即返回）；同一 stream 内保序，不同 stream 无序并发（由硬件连接/资源仲裁实际并发度）。**前提条件**：①数据无冲突——读写同一 buffer 的 kernel 必须同 stream 或 event 同步（否则 race）；②资源够——smem/寄存器/grid 使并发 kernel 能同时驻留，否则排队；③**正确同步**——event/callback 衔接跨 stream 依赖；④H2D 异步要 pinned memory 才真异步（可分页内存会隐式 stage 同步）；⑤注意默认 stream 的同步语义（legacy 阻塞其它 stream，用 per-thread/非默认 stream 规避）。典型应用：计算流 + 拷贝流重叠（double buffering）、多 batch 并发。

### Q26. 最优线程数配置如何确定（O二 理论 4）

推导链：①先定**每线程工作量**（向量化 128bit 访存 → 每线程 4×dtype 元素）；②block 尺寸 = warp 数 × 32：常见 128/256（4-8 warp，足够隐藏延迟又不过度拆分），必须是 32 倍数；③occupancy 检查：寄存器（255/线程上限，每 SM 64K 寄存器 → block 数 = 65536/(threads×regs/thread)）、smem 同理；④**memory-bound** kernel 高 occupancy 优先（多 warp 换延迟）；**compute-bound** 中等 occupancy + 更多 ILP/寄存器反而更快（少切换、循环展开）；⑤grid = 数据量/块工作量，余数处理（guard 或 tail block）；⑥最终 NCU 实测调（理论给候选集，实验定胜者）。

### Q27. Shared Memory 计算结果出错的排查（O二 理论 5）

按"同步 → 竞争 → 越界 → bank"排查：①**缺同步**：写后读之间没有 `__syncthreads()`（或条件分支里同步导致部分线程死等的 UB——分支内 syncthreads 要求全 block 到达）；②**race**：多线程写同一 smem 地址没有归属划分或原子；③**越界/生命周期**：tile 边界 guard 漏写（halo 区域）、动态 smem 越界踩别的 block 数据；④**bank conflict 不会错只会慢**——但 padding 改了索引若没同步对齐会错；⑤工具：compute-sanitizer（racecheck 查 smem 竞争、initcheck）、CUDA_DEBUG=1、printf 分块验证、对拍（同输入 CPU 参考实现二分定位出错 tile）。"特殊 shape 才错"→ 多半是尾块 guard/索引取整问题。

### Q28. NHWC vs NCHW（O二 理论 7）

- **NCHW**：PyTorch 原生（框架习惯）、CPU/BN 统计友好（C 连续）；卷积在 GPU 上要对 channel 求和，跨步大不合并；
- **NHWC**：TensorRT/cuDNN 快路径（C 在最后一维连续，卷积的 reduce 维连续读，向量化 + tensor core 的 layout 要求（TC 输入 K 维需连续）、多 channel 的 cache line 利用满）；代价：transpose 开销（NCHW 数据需 layout 转换，[02](./02-cuda-编程题.md) Q6 的 kernel）；
- 选择：**训练与推理都建议在 GPU 上全程 NHWC**（框架层自动，PyTorch 的 channels_last）；输入输出端做一次转换；若网络很浅或 CPU 部署则 NCHW 免转换。C=1/3 的 CV 原始图像常 4 通道对齐（padded NHWC）。

### Q29. 何时放弃 Shared Memory（O二 理论 8）

①**访问一次、无复用**：elementwise/map kernel——smem 中转纯增开销（写+读两次 SRAM），直接寄存器/global；②**数据量大于 tile 收益**：占满 smem 反而 occupancy 掉（驻留 block 变 1），总吞吐下降；③**bank conflict 无法消除且工作集小**：L1/L2 命中率高（<L2 的热点数据）时 global 走 cache 反而快（L1 延迟与 smem 同量级，且免管理）；④**异步流水已够**：cp.async/TMA 直接 global→寄存器的 deep pipeline（Hopper GEMM 风格）省掉显式 smem；⑤线程间无需交换。判断口径：复用次数 ×（global 延迟-smem 延迟）> smem 管理+同步开销才值得。

### Q30. 快速上手陌生推理框架的方法论（O云一 理论 2）

①**先跑通官方 example**（bench 脚本看默认配置）；②读架构三件事：**入口 API → 调度循环 → 显存/KV 管理**（每个推理框架的骨架都这三块）；③对照已知框架做映射（vLLM 的 scheduler ↔ SGLang 的 runtime）；④改一个 plugin/后端算子摸清扩展点；⑤profile 一遍（Nsight/torch profiler）看时间线建立性能直觉；⑥看 issue/CHANGELOG 里的高频关键词。体现"方法论 + 上限速度"即可。

### Q31. 精度转换的数值稳定性（O云一 理论 3）

①**先 scale 后转换**：FP32→FP8/INT8 前除以 amax（delayed/per-tensor scale），回到高精度再乘回；②**累加保持高精度**：低精度输入、FP32 累加器（Tensor Core 的 FP32 accumulate、CPU 的 int32 累加）；③**避免上溢/下溢链**：bf16 范围大精度低、fp16 精度高范围小——softmax/logits 通道用 fp32 算（`attn_softmax in fp32`）；④**舍入模式**：round-to-nearest-even 为默认，量化用对称截断注意负数边界；⑤**逐步对拍**：转换前后逐层误差监控（cosine/MSE），先小算子后整网；⑥混合精度训练的 loss scale（动态）是同类问题的训练侧方案。参考 [06](./06-量化.md) Q5/Q6。

### Q32. Warp/Block 划分与 SIMT 的联系（O云一 理论 4）

SIMT：单指令多线程——一个 warp 32 线程共享 PC 与指令流，硬件按 32-lane 执行（active mask 处理分支）。**划分与 SIMT 的内在联系**：①block 尺寸必须 32 倍数，否则尾 warp 浪费 lane（31 线程也占一个完整 warp 的调度槽）；②分支粒度若 <32 线程必然 divergence（串行执行两路径）——数据布局应让相邻线程走同分支（按块连续处理）；③归约/交换类操作天然对齐 warp（shuffle 32 轮内完成）→ block 内组织成 warp 层级（warp shuffle + smem 跨 warp）；④lane id 与地址映射决定合并访存（相邻 lane 相邻地址）——所以"thread(idx) → 数据索引"的映射公式是 SIMT 友好性的核心（`(blockIdx*blockDim+threadIdx)` 线性映射）。

### Q33. CV 推理服务优化开放题（O云一 开放 5）

按层展开：①**模型**：量化（INT8 PTQ/QAT）、剪枝蒸馏、输入分辨率动态档位；②**kernel/引擎**：TensorRT 编译（fusion+profile）、batching 合并小图、NMS plugin、预处理 GPU 化（decode+resize 在 GPU，NVJPEG/DALI）；③**服务**：动态 batch（攒批窗口 <延迟预算）、多实例负载均衡、流水线并行（预处理/推理/后处理三级）、CPU 线程池与 zero-copy（pinned + 异步）；④**指标闭环**：单卡 QPS、p99 延迟、GPU 利用率三件套，profiler 找空泡（H2D 等待→异步化，CPU 预处理慢→GPU 化）；⑤量化收益给数字（INT8 ~2×、TRT ~1.5-3× vs eager、动态 batch 小图场景 2-5× QPS）。

## 三、vivo（编译器方向）

### Q34. HalideIR 与 Schedule 原语（vivo 理论 3）

Halide 把算法（算什么）与调度（怎么算：循环序、并行、分块、向量化）分离；**HalideIR** 是其多层循环+计算的中间表示（后来成为 TVM 的 Reley→TE 底座）。常见 **Schedule 原语**：

- `split`（循环拆内外层，tile 的基础）、`fuse`（循环合并）、`reorder`（交换循环序——改变访存局部性/并行维度）；
- `tile`（split 两个维度+reorder = 分块）、`parallel`（外层多线程）、`vectorize`（内层 SIMD）、`unroll`（展开提 ILP）；
- `compute_at`/`compute_inline`/`compute_root`（计算位置的嵌套选择——决定中间结果放寄存器/smem/显存与复用窗口，融合的本质）；
- `cache_read/cache_write`（引入中间缓存层级）、`bind`（绑定 thread/threadblock，GPU）、`storage_align/padding`（bank 冲突）。
运作机制：调度在循环 AST 上做变换生成等价代码，编译器按调度做依赖分析（同步插入）与存储分配——同一算法换调度性能可差百倍（经典 Halide blur 示例）。

### Q35. TVM 图级/算子级优化与硬件支持（vivo 理论 5、Q34 同族）

- **图级**（Relay/Relax）：常量折叠、算子融合（injective/elementwise 横向融合 + reduction 复合）、layout 变换（NCHW→NCHWc）、target 注解切分子图（host/GPU/NPU）；
- **算子级**（TE/TIR）：Halide 式调度搜索（见 Q34 原语），模板+枚举的 auto-scheduler（Ansor：草图→随机完备搜索→进化调优）；
- **硬件支持**：target 描述（ISA/位宽/smem 大小/并行度）驱动约束（如 NEON 128bit → vectorize 长度 4×fp32）、硬件内联汇编（tensorize 把小矩阵乘映射到 MMA 指令）、自动调优时 on-device 实测（RPC）反馈真实频率/缓存。
详见 [08](./08-编译器与性能分析.md) Q1（vivo 同族已在首批覆盖）。

### Q36. 主流编译框架对比与自动调优 vs 手动（vivo 理论 4/6）

对比见 [08](./08-编译器与性能分析.md) Q1（TVM/XLA/TRT/MLIR）：XLA（JIT、图级为主+LLVM）、TensorRT（闭源引擎、autotune 选 tactic）、MLIR（编译器基础设施，多层 Dialect）、TVM（开源全栈+调度搜索）。**自动调优 vs 手动**：自动（Ansor/MetaSchedule）——覆盖广、无专家依赖、可迁移，但搜索时间小时级、峰值性能仍差手工 10-30%（非常规融合/全局数据流重构想不到）；手动（CUTLASS/Triton 半自动）——极致性能与确定性，但人力成本高、shape 特化难泛化。趋势：自动为主 + 手写模板/规则注入（MetaSchedule 的 schedule 空间由专家写 sketch）。

## 四、网易

### Q37. 高吞吐低延迟推理服务的架构与工程要点（网易 推理 2）

架构分片：**接入层**（路由、限流、鉴权）→ **调度层**（请求队列、动态 batching、优先级/抢占、PD 分离可选）→ **执行层**（GPU worker 池，模型实例 + KV 管理）→ **支撑**（prefix/KV 缓存、指标与追踪、灰度发布）。工程要点：①延迟预算拆解（TTFT/TPOT SLO → 每环节配额）；②batching 策略（延迟窗口攒批，goodput 最大化）；③显存水位与抢占（KV 满时 swap/重算）；④多模型混部（按 QPS 弹性、MIG/时间片）；⑤观测（p50/p99 分位、GPU 利用率、KV 命中率、队列深度告警）；⑥容错（实例健康检查、请求幂等重试）。参考 [05](./05-推理引擎与优化.md) Q12 综述。

### Q38. C++ 内存分区 / new vs malloc / 内存对齐 sizeof（网易 C++ 3/4/8）

- 分区与 new/malloc 见 [09](./09-cpp与系统基础.md) Q6/Q7（栈/堆/静态-全局/常量区/代码段；new 调构造+类型安全+可重载+抛异常，malloc 裸内存）；
- **对齐规则**：每个成员放在其大小（或 alignof）整数倍偏移；结构体总大小为最大对齐的整数倍。算题模板：

```cpp
struct S { char a; int b; char c; };   // a:0, pad 1-3, b:4-7, c:8, pad 9-11 → sizeof=12
struct T { int b; char a; char c; };   // b:0-3, a:4, c:5, pad 6-7 → sizeof=8（成员排序影响大小）
struct U { char a; double b; char c; };// 1+7pad+8+1+7pad = 24
```

对齐原因：硬件按字长取数（未对齐总线两次访问/异常）、SIMD 要求 16/32/32B 对齐（`alignas(32)` 恰好覆盖 AVX-512）、原子操作要求自然对齐否则总线锁。`alignof/alignas` 与 `posix_memaligned/malloc` 手工对齐分配。

### Q39. 虚函数表与多态（网易 C++ 7）

见 [09](./09-cpp与系统基础.md) Q8 与 [61](./61-t1-蚂蚁华为京东.md) Q50（vtable 结构、槽位按基类声明序、vptr 在构造时设置、纯虚函数 = 槽位为 `_pure_virtual` 或抽象类无 vtable 实例化禁令）。

### Q40. 右值引用与移动语义（网易 C++ 9）

见 [09](./09-cpp与系统基础.md) Q4：`T&&` 绑定将亡值，`std::move` 是 static_cast 不移动，移动构造"偷"资源（指针交换 + 置空源）、`std::forward` 完美转发保持值类别；价值：深拷贝变 O(1)（vector 扩容、临时对象传递）、`unique_ptr` 只可移动的所有权语义；`noexcept` 移动构造是 vector 扩容选 move 的前提。

### Q41. SIMT 与线程局部变量的存储位置（网易 CUDA 10/11）

见 Q32（SIMT）与 [01](./01-cuda-基础与体系结构.md)：局部变量优先**寄存器**（每线程私有、索引静态可定）；不能静态索引/超出 255 上限/递归 → **local memory**（物理在 global，带 L1/L2 缓存但慢）→ 排查 `ptxas -v` 的 "spill"；`__shared__` 显式 smem、`__device__` 全局。寄存器分配由编译器（存活分析），动态索引数组即使小也会 spill（改 smem 或展开）。

### Q42. 共享内存矩阵乘、warp shuffle、向量化、bank conflict（网易 CUDA 12-16）

集中见 [02](./02-cuda-编程题.md) Q3（smem tiling GEMM：复用推导与代码）、[01](./01-cuda-基础与体系结构.md)（warp shuffle 归约的 `__shfl_down_sync` 蝶形、float4 向量化合并访存 128bit、bank conflict 32 bank×4B 与 padding/swizzle 消除，示例图解）。

### Q43. CPU 缓存工作原理与替换策略（网易 基础 17）

层级 L1d/L1i（~4-8 cycles, 32-64KB/核）→ L2（~12 cycles, 512KB-2MB/核）→ L3（~40 cycles, 数十 MB 共享）→ 内存（~200+ cycles）；组织：set-associative（地址 = tag + index + offset，组内 N 路映射）；**局部性**：时间（重复访问同址留 cache）+ 空间（相邻地址同 line 预取，line 64B）；替换策略：LRU（真 LRU / pseudo-LRU 树）、随机（高压下抗 scan）、LFU 少用；写策略：write-back + write-allocate 为主；cache 一致性 MESI（核间）。程序侧：行对齐数据结构、按访问序遍历、伪共享规避（`alignas(64)` 每线程计数器）。

## 五、拼多多

### Q44. Transformer 相比 MLP 的核心优势（拼 基础 1）

MLP：固定权重融合、无位置间交互（token 间无信息路由）、参数与输入维度二次增长、对排列无自适应。Transformer 的三点：①**动态路由**：attention 权重由内容决定（数据相关的连接），同一组参数可对任意 token 组合做不同混合——MLP 的融合系数是静态的；②**全局感受野 + 路径长度 1**：任意两 token 直连，长程依赖建模无衰减链；③**参数效率与可扩展**：参数量与序列无关（可变长输入）、Scaling 律好（结构归纳偏置少但容量随数据增长）。代价：O(S²) 与外推问题（MLP 反而 O(S·d²) 且线性）——所以有线性/稀疏注意力研究试图兼得。

### Q45. MHA/GQA/MLA 与 Triton vs CUDA（拼 基础 2/5）

- MHA/GQA/MLA 见 [04](./04-attention与kvcache.md) Q4/Q5 与 [60](./60-t1-快手与美团.md) Q17（含 MLA↔GQA head 换算代码）；
- **Triton vs CUDA 算子开发**：①抽象层级——Triton block 级编程（`tl.load/tl.dot`，自动处理向量化/合并访存/bank，编程模型接近"每 CTA 一个 tile"），CUDA/C++ thread 级（warp/block/同步全手工）；②性能——CUDA 上限更高（TMA/warp specialization/内联 PTX、精细到指令），Triton 通常到手写峰值的 80-95%（GEMM/归一化类已够）；③开发效率——Triton 数十行、Python 生态、autotune 内建（`@triton.autotune`），CUDA 需 arch 适配与长时间调优；④生态——Triton 与 PyTorch/编译器亲和（`torch.compile` 后端）、可移植 NVIDIA/AMD；CUDA 库生态（cutlass/cub）与工具链（NCU 全特性）更全；⑤调度边界——复杂 producer/consumer 流水（FA3 的 warp spec）Triton 表达弱。选择：业务算子/快速迭代 → Triton；极端性能/奇异特性 → CUDA。仓库两版对照：`answers/triton_kernels/`（softmax/matmul/layernorm）vs `solutions/cuda/`。

### Q46. 训练中的访存优化与 KL 算子优化（拼 训练 6/7）

- **训练访存优化**：激活重算（选择性重算 attention 中间量）、激活量化（bf16 存储的 activation offload）、ZeRO 分片减少冗余、layout 优化（NHWC/channels_last、算子输出融合避免物化）、dataloader 的 pinned+prefetch（H2D 与计算重叠）、梯度桶化与累积（减通信次数）、pin memory + `non_blocking`；MoE 训练的 dispatch buffer 复用；
- **KL 散度算子优化**（拼多多的业务算子题）：逐元素 $p\log(p/q)$ + reduce——优化点：①log 只算一次（`log_p`/`log_q` 复用，减超越函数一半）；②`logsumexp` 稳定化合并（减 max + exp 内联）；③reduce 用 warp shuffle 两级归约不落 global；④向量化 128bit 访存 + `__expf` 快数学；⑤低精度（bf16 输入 fp32 累加）；⑥GRPO 场景的 k3 估计器只需 log 概率差（免全分布），参考 `answers/t1/rl_basics.py::k3_kl_estimator`。

### Q47. 算法研究与 Infra 的工作重点差异（拼 基础 3）

研究：问题定义（benchmark/指标）、假设-实验闭环（消融）、创新点与新 SOTA、paper 写作；评价 = 指标提升与想法新颖度。Infra：工程问题定义（SLO/成本）、性能分析方法论（profiling→瓶颈→手段→验证）、代码质量与可维护性（测试/监控）、跨层知识（硬件-系统-算法栈）；评价 = 端到端加速/降本与稳定性。交叉点：性能敏感的研究（训练效率）与有算法品味的 infra（调度策略、量化方法选择）。答"两边都懂一点 + 一边深"最稳。

## 六、B站

### Q48. 进程/线程/协程（B一 理论 1）

见 [09](./09-cpp与系统基础.md) Q18 基础上补协程：**协程 = 用户态调度的轻量执行流**——挂起/恢复由程序自己 yield（无内核介入），切换只保存少量寄存器（~ns 级，线程 ~μs）；单线程内万级协程可行（内存 KB 级 vs 线程栈 MB 级）。资源：进程（独立地址空间/文件表，隔离最强创建最贵）→ 线程（共享地址空间，私有栈/寄存器/PC）→ 协程（共享线程的一切，仅执行状态）。调度：进程/线程内核抢占式；协程协作式（cooperative，一个不让出就饿死别人——CPU 密集任务不适合）+ 运行时事件循环（asyncio 依赖 IO 多路复用）。IO 密集高并发 → 协程；计算并行 → 多进程（GIL 规避）/多线程（C++）。

### Q49. 以进程 vs 线程为调度单位的公平性（B一 理论 2）

- **进程调度**：每进程近似独立配额 → 多进程场景单"用户/任务"的份额由其进程数决定（fork 炸弹可多占）；隔离好（优先级/配额按进程核算清晰）；
- **线程调度**：所有线程平权竞争 → 同进程多线程在 CPU 紧张时"偷"走他人时间；但若调度器按**进程组/会话**聚合配额再组内分线程（cgroup、FSS/公平共享调度器、Windows 的 per-process quantum），可消除这种不公平；
- GPU 侧对照：kernel 粒度调度无抢占 → 长 kernel 让并发流饿等（时间片 vs 完成驱动的差异），MPS/MIG 是 GPU 的"公平配额"手段。答题给"不公平的来源（共享实体粒度）+ 治理（分层配额）"框架。

### Q50. 集合通信的语义与并行策略对应（B一 理论 3）

AllReduce（DP 梯度）、AllGather（ZeRO-3 参数、TP 列切输出）、All-to-All（MoE dispatch/combine）、ReduceScatter（ZeRO-2/SP）、Broadcast（初始化）——完整语义、实现与通信量见 [07](./07-分布式训练.md) Q1/Q2 与 [60](./60-t1-快手与美团.md) Q22。

### Q51. 用 profiling 判断访存 vs 计算瓶颈（B一 理论 4/5）

见 [03](./03-gemm与算子优化.md) Q1（Roofline 判据：实测 FLOPS/带宽 与机器平衡点）+ [08](./08-编译器与性能分析.md) Q5（NCU 指标：DRAM Throughput vs SM Throughput，先看哪个接近上限； stall 原因：long scoreboard=等内存、selected/数学管道满=计算）；**针对性调优**：访存型 → 合并访存/向量化/smem/cache 复用/数据布局/降低精度（bytes 减半）/重算换存储；计算型 → 张量核心化/精简指令（快速数学、去超越函数）/ILP 与寄存器复用/共享计算（融合消除重复算）；调度型（gap 多）→ CUDA Graph/批量化/多流。加 B站 特色：block 尺寸实验矩阵（128-512 扫描）与尾块处理（grid 对齐）。

### Q52. CUDA 大数组归约：smem 块内 + 多级 kernel（B一 编程 7）

`solutions/cuda/src/reduce.cu` + [02](./02-cuda-编程题.md) Q1：两级方案——kernel1 每 block 归约出部分和（smem 树归约 + warp shuffle 收尾）写 partial[n_blocks]；kernel2 再归约 partial（或 atomicAdd 单值）；3090 实测 1M float 从 atomic 1.96ms 到 0.016ms（124×）。要点：向量化装载（float4）、`__syncthreads` 层级、warp shuffle 替代最后几层 smem、grid-stride loop 处理任意 N、尾数 guard。追问"为何不用 atomic 加到底"——冲突串行 + 每次原子过 L2 的延迟。

## 编程题索引（本篇）

| 题目 | 来源 | 实现 |
|---|---|---|
| 手写膨胀卷积 + 优化讨论 | 米实1 | `answers/t1/dilated_conv.py`（naive+unfold 双实现与 conv2d 对拍） |
| 手写 MHA / MGA(GQA) | 米校、拼多多 | `answers/attention/mha.py、gqa.py`（[04](./04-attention与kvcache.md) Q4/Q5） |
| 矩阵转置 | 米实一二 | `solutions/cuda/src/transpose.cu`（[02](./02-cuda-编程题.md) Q4） |
| 合并两个有序链表 | 米实一二 | `answers/algorithms/linked_list.py::merge_two_lists`（[12](./12-算法题.md)） |
| 最长回文子串 | 米一 | `answers/algorithms/dp.py::longest_palindromic_substring`（[12](./12-算法题.md)） |
| CUDA 多级归约 | B站 | `solutions/cuda/src/reduce.cu`（[02](./02-cuda-编程题.md) Q1） |
| C++ 编程题 | 拼多多 | 原题无题干；C++ 全家桶见 [10 C++ 编程题](./10-cpp编程题.md) |

验证：`cd solutions/python && pip install -e . && pytest tests/test_t1.py -v`。
