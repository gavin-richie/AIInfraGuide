---
title: "腾讯题解：推理优化与 CUDA/HPC"
description: "腾讯 6 份 AI Infra 面经（约 66 题）题解：FP4/FlashAttention/MoE/GQA/RoPE 推理题、fastllm 与 vLLM 源码、Triton 实现 PagedAttention、MPI/TCP/Hyper-Q 等 HPC 题、C++ 与系统基础题"
pubDate: 2026-10-05
topic: "公司题解-腾讯"
order: 40
tags: ["公司题解", "推理优化", "高性能计算", "CUDA"]
draft: false
---

覆盖题库：`腾讯-AI-Infra`（推理优化 10）、`腾讯-AI-Infra-实习`（基础 9）、`腾讯-AI-Infra-实习-一二面`（基础/HPC 22）、`腾讯-AI-Infra-实习-一面-1`（7）、`腾讯-AI-Infra-实习-一面-2`（6）、`腾讯-AI-Infra-校招-一二面-1`（12）。TEG 数据库/OS 专项见[41 篇](./41-腾讯-teg-数据库与os.md)，编程题见[42 篇](./42-腾讯-编程题.md)。

**配套新增代码**：`solutions/python/src/answers/tencent/`（Triton PagedAttention decode kernel、双堆滑动窗口中位数等），验证：

```bash
cd solutions/python && pip install -e . && pytest tests/test_tencent.py -v
```

## 一、推理优化与模型结构

### Q1. FP4 量化的基本原理；量化场景下矩阵乘的维度如何变化（AI-Infra 推理 1/2）

FP4 用 4 bit 浮点表示一个数。主流格式两种：**FP4(e2m1)**——1 符号 + 2 指数 + 1 尾数，只有 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} 共 16 个值；**NVFP4/MXFP4**——每 16/32 个元素一组（block），组内共享一个 **FP8(E4M3) scale**，弥补 FP4 动态范围极窄的问题（详见[06 篇 Q5](./06-量化.md)）。基础原理可拆成三步答：

1. **表示**：$w \approx s_{block} \cdot fp4\_dequant(q)$，反查表得到真实值；每 block 的 $s$ 存 FP8，因此"有效精度"是 block 内相对误差可控；
2. **矩阵乘维度变化**：权重 $W \in \mathbb{R}^{K \times N}$、激活 $X \in \mathbb{R}^{M \times K}$。量化后 **K、N、M 本身不变**，变的是每元素位宽与**scale 维度**：$W_q$ 为 uint8 打包（每字节装 2 个 FP4），另加 $s \in \mathbb{R}^{(K/B) \times N}$（B=16 为 block size）。计算时 kernel 边加载边反量化：$Y = (X \cdot diag(s 按块广播)) \cdot dequant(W_q)$，输出仍为 FP16/BF16（**激活通常 FP8/FP16，仅 weight FP4** 的 weight-only 场景更常见）；
3. **收益**：权重显存/带宽 ÷4（对比 FP16），decode 阶段（带宽瓶颈）收益最大。

### Q2. 硬件层面如何执行量化；量化推理对硬件的要求（AI-Infra 推理 3/4）

**执行方式（按硬件代际）**：

- 无原生支持（Ampere 及更早）：FP4/INT4 权重从显存读出后在寄存器/共享内存里**反量化成 FP16 再进 Tensor Core**（Marlin 类 kernel，见[06 篇 Q2](./06-量化.md)），收益主要来自带宽节省；
- 部分支持（Hopper）：FP8 Tensor Core（W8A8）；
- 原生支持（**Blackwell，第五代 Tensor Core**）：NVFP4/MXFP4 数据路径直接进 MMA，反量化在 tensor core 内部完成，`cvt` 类指令 + LUT 在寄存器级转换，scale 走独立 operand fragment。

**对硬件的要求**：① MMA 单元支持对应低精度格式（否则只能软件反量化，吃 ALU）；② 高带宽显存 + 大 L2（低精度意义在于搬得快）；③ scale 的存储/广播通路（block scale 作为第二操作数）；④ **软件栈**：量化感知训练/校准工具、kernel 库（cuBLASLt/TensorRT-LLM/FlashInfer）与调度器的支持；⑤ 数值上需要混合精度累加（FP4 乘积用 FP32/FP16 累加）。

### Q3. FlashAttention 的作用与核心优化点；注意力矩阵的维度推导；K 是否包含 Q 对应的那个位置（AI-Infra 推理 5/6/7）

核心原理详见[04 篇 Q1](./04-attention与kvcache.md)，这里补题面的三个具体追问：

- **作用**：精确注意力（非近似）的 IO 感知实现，把注意力对 HBM 的访问从 $O(N^2)$ 降到 $O(N)$，长序列下显著提速省显存（不物化 $N \times N$ 矩阵）。
- **维度推导**：$Q \in \mathbb{R}^{B \times H \times S_q \times d}$，$K, V \in \mathbb{R}^{B \times H \times S_{kv} \times d}$。score $= QK^\top \in \mathbb{R}^{B \times H \times S_q \times S_{kv}}$，softmax 沿最后一维；输出 $(softmax(score) \cdot V) \in \mathbb{R}^{B \times H \times S_q \times d}$。FA 做 tiling：外层遍历 K/V 块（列块 $B_c$），内层载入 Q 行块 $B_r$，**score 矩阵退化为 $B_r \times B_c$ 的 SRAM 级小矩阵**，配合 online softmax（running max $m$ 与归一化因子 $l$）逐块修正。
- **K 是否包含 Q 对应位置**：包含。训练/预填充阶段 $S_q = S_{kv}$，因果掩码下第 $i$ 个 query 只看到 $K_{0..i}$（含自身），即上三角置 $-\infty$；**decode 阶段**每步 query 长度为 1，但 KV Cache 里已追加当前 token 的 K/V，所以 K 的长度 = 已生成序列长度（含当前步），attention 是对全部历史 + 当前位置做的。

### Q4. FlashAttention 的加速原理，优化的是算力还是访存（实习-一面-2 推理 6）

**访存**。标准注意力的 HBM 流量：$Q,K,V$ 读 $O(Nd)$、score 与 softmax 中间矩阵写+读 $O(N^2)$；FA 把分块计算全部留在 SRAM，只写回最终 $O(Nd)$ 输出。长序列时 $N^2 \gg Nd$，省的是**带宽**而非 FLOPs（FLOPs 甚至略增——多了 rescale）。所以 FA 在 decode（bandwidth-bound）和大 N 场景收益最大；而 GPU利用率提升的来源是把"写读中间矩阵"的时间换成了对 KV 的持续计算。答题框架参考[03 篇 Q1 Roofline](./03-gemm与算子优化.md)。

### Q5. 多头注意力与多 batch 场景如何并行（AI-Infra 推理 8）

- **数学上**：各 head 的 attention 相互独立（各自 Q/K/V 投影与 softmax），batch 间也独立 → 天然两层并行维。
- **实现上**：把 $(B, H)$ 折叠成一个维度。PyTorch：`x.view(B*H, S, d)` 后按 3D 批量算；Triton：`grid = (B*H, ceil(S/B_r))`，program_id 反解 `b = pid // H`；CUDA：`blockIdx.z = b*H + h`（或 y/z 各管一维）。
- **GQA 注意**：KV head 数 $H_{kv} < H_q$ 时，同一 KV head 被 $g = H_q/H_{kv}$ 个 query head 共享，并行仍按 $B \times H_q$（读 KV 时按 `kv_head = q_head // g` 映射，见 Q17 的 Triton 实现），或按 KV head 分组减少重复加载（FA v2 的 grouped 布局）。
- 手写参考：[04 篇 Q4/Q5](./04-attention与kvcache.md)（MHA/GQA PyTorch 实现，`answers/attention/`）。

### Q6. GQA、MQA 与标准 MHA 的核心区别；线上推理为何更关注 GQA（实习-一面-2 推理 4）

MHA：$H$ 个 head 各有独立 K/V；**MQA**：所有 head 共享 1 组 KV；**GQA**：分组共享，$H_q$ 个 query head 映射到 $H_{kv}$ 组 KV（介于两者之间）。KV Cache 大小 $= 2 \times S \times H_{kv} \times d \times bs$ 字节，随 $H_{kv}$ 线性下降：MQA 降到 1/H，GQA 降到 $H_{kv}/H$。

线上更关注 GQA 的原因：① decode 是**带宽瓶颈**，每 token 要把整个 KV Cache 读一遍，KV 越小吞吐/延迟越好；② 显存占用决定**并发 batch 与最大上下文**（PagedAttention 也要配合 GQA 才能塞更多序列）；③ MQA 质量损失明显，GQA（如 8:1）质量几乎追平 MHA 而成本接近 MQA——精度/成本甜点。详见[04 篇 Q4/Q7](./04-attention与kvcache.md)。

### Q7. RoPE 的工作原理；长上下文外推为何失真（实习-一面-2 推理 5）

原理（详见[04 篇 Q8](./04-attention与kvcache.md)）：对 $q, k$ 按维度两两分组，第 $i$ 组乘旋转矩阵 $R(\theta_i m)$，$\theta_i = base^{-2i/d}$。内积 $q_m^\top k_n$ 只依赖相对位置 $m-n$（旋转矩阵正交性），从而把绝对位置编码变成相对位置信息。

**外推失真的原因**：训练时 $m \in [0, L_{train}]$，推理超出后——① 低维（高频 $\theta_i$ 大）分量的旋转角超出模型见过的范围，注意力 logits 进入未训练区域，分布漂移；② 高频分量在长距离下旋转角剧烈震荡，等效"位置混叠"，注意力熵异常升高/聚焦失效；③ 相对距离的覆盖范围外推，softmax 温度效应失衡。

**补救**：位置插值 PI（位置除以压缩比，等价缩小旋转角）、NTK-aware（调大 base，如 LLaMA 8k→32768 时 base 1e4→1e6，保高频插低频）、**YaRN**（按维度频率分段插值 + attention 温度补偿）、RoPE 基频外推 + 微调。可加一句工程现状：主流模型直接用 YaRN 或 q-format（NoPE/部分层不加 RoPE）。

### Q8. MoE 为何能控制成本；真正的技术难点（实习-一面-2 推理 2）

**成本控制原理**（对照[07 篇 Q8](./07-分布式训练.md) DeepSeek-V3）：MoE 把 FFN 换成 N 个专家 + 路由，每个 token 只激活 top-k 个专家（如 256 选 8）。参数量可以做到数百 B（容量大），而**单 token 的 FLOPs ≈ 稠密模型激活参数量**（如 V3 激活 37B/总 671B）——"存得大、算得少"。训练成本同理：每步只有被路由到的专家参与前反向。

**真正的难点**（答题要有层次）：

1. **推理显存并不省**：所有专家权重必须驻留（或 offload），总显存 = 总参数量；成本节省的是计算/带宽，不是权重存储——这是面试最常见的追问点；
2. **负载均衡**：路由不均 → 少数专家过热、其他退化，batch 内计算倾斜，EP 下 all-to-all 也要按均摊假设分配带宽（见 Q9）；
3. **路由的训练不稳定**：argmax 不可导（Gumbel/softmax 近似）、router 漂移、专家坍缩；
4. **all-to-all 通信**：EP 下 dispatch/combine 两次全互联通信，与计算重叠（DualPipe/TE-Overlap）才不拖尾；
5. kernel 效率：grouped GEMM（变长 per-expert M）、细粒度专家的 dispatch 索引构建。

### Q9. MoE 负载均衡的方法；为何 loss 正常但 expert 已退化（实习-一面-2 推理 3）

**均衡方法**：① auxiliary loss（switch/DeepSeekV2：对每个 expert 的负载占比与平均路由概率乘积加惩罚）；② router z-loss（惩罚 router logit 过大，稳定训练间接均衡）；③ **DeepSeekV3 的 aux-loss-free**：给每个专家一个 bias，负载低的 bias 调高、仅参与 top-k 选择不影响 gate 概率值，配合全局 bias 更新步长；③ capacity factor/drop-and-compute（丢弃超额 token，推理 dropless）；⑤ EP 层面：动态重排 expert 副本、跨节点 expert placement 感知拓扑。

**loss 正常但 expert 退化的判别**：主 loss 由少数"强势专家"支撑即可收敛——路由坍缩（大部分 token 挤到 k 个专家，其余几乎不接收 token）时模型仍可拟合训练分布，loss 看不出异常，但容量利用率和下游泛化受损。监控指标：**每 step 各 expert token 计数分布（基尼系数/最大占比）、router 输出熵、expert 梯度范数分布、负载 skew 在 EP 下的 all-to-all 时间**。loss 只是结果指标，路由健康度必须看分布指标。

### Q10. Chunked Prefill 的设计动机与解决的核心问题（实习-一面-1 推理 2）

要点复述（详见[05 篇 Q3](./05-推理引擎与优化.md)，该篇本题即以腾讯为题源）：prefill 是 compute-bound、decode 是 memory-bound，混跑时长 prefill 会把后到请求的 decode TTFT/ITP 拖长几十倍。chunked prefill 把长 prompt 切成固定 token 数的块，逐块做 prefill，块间穿插 decode 迭代——用轻微的 prefill 吞吐损失（ attent ion 块间 KV 重复加载）换取 decode 的低尾延迟，同时天然避开"一个 32k prompt 独占 batch"的调度空洞，是 continuous batching 之上的时延/吞吐折中。

### Q11. NVLink 节点内 + RDMA 节点间的集群如何设计分布式推理（实习-一面-1 推理 3）

分层并行 + 分层通信（结合[05 篇 Q4/Q5](./05-推理引擎与优化.md) PD 分离/Mooncake）：

1. **节点内（NVLink 900GB/s 级）**：TP 切模型层内权重（all-reduce 走 NVLink，通信/计算比可控），大模型首选 8 卡 TP；节点内也可叠 EP（MoE all-to-all 在 NVLink 上开销小）；
2. **节点间（RDMA，慢一个量级）**：避免跨节点 all-reduce——用 **PP**（点对点 activation 传送，通信量 = batch × hidden）或 **DP + 各自完整副本**（零通信，靠负载均衡）；MoE 用 EP 时 dispatch/combine 走 RDMA，需要 overlap；
3. **PD 分离**：prefill 池与 decode 池分离部署，decode 换卡时 **KV Cache 通过 RDMA 传输/共享**（Mooncake 式 KV 中心：GPU→CPU→远端 NVMe 分层存储，传输引擎多路 RDMA 打满网卡）；
4. **调度感知拓扑**：请求路由到 KV 已缓存/同机架的分片（局部性），网内拓扑感知的 expert placement；
5. 兜底讨论：节点间带宽低时"通信量最小"优先于"负载绝对均衡"，宁可副本冗余（权重盘便宜）也不做跨机 TP。

### Q12. GPU 架构相关；如何分析系统性能瓶颈（AI-Infra 基础 9/10）

- GPU 架构：SM/缓存层级/warp/ Tensor Core 的体系结构见[01 篇 §2/§3/§4](./01-cuda-基础与体系结构.md)，答"了解到什么程度"建议按"编程模型 → 内存层级 → 微架构（warp 调度/吞吐结构）→ 我做过的对应优化"四层递进，诚实标注边界。
- 瓶颈分析方法论：**先分层（应用/调度/kernel/通信）再量化**。工具链 Nsight Systems（时间线找 gap、H2D、launch 空隙）→ Nsight Compute（SOL、stall 原因、带宽）→ nccl/torch profiler（通信占比）。判读：SM% 与 Mem% 谁逼近 100 谁是瓶颈；两者都低 → 并行度/launch 间隙问题；多卡先看通信-计算 overlap。详见[08 篇 Q5](./08-编译器与性能分析.md)、[03 篇 Q4](./03-gemm与算子优化.md)。

## 二、推理引擎源码

### Q13. TensorRT-LLM、AGI 与 vLLM 的源码实现差异（校招-一二面-1 一面 1）

题面中 "AGI" 一般指面试官所在团队的自研引擎（或口误），遇到没听过的引擎先确认；答题以 **TRT-LLM vs vLLM** 的架构对比为主体，再给"自研引擎对照框架"：

| 维度 | TensorRT-LLM | vLLM |
|---|---|---|
| 语言/形态 | C++ 核心 + Python API，编译期把 graph 序列化成 engine | Python 调度器 + C++/CUDA/Triton kernel（PyTorch 生态） |
| 图优化 | 强：多层 fusion/normalize/quantize（NV 专调），engine 离线编译 | 中：torch.compile/手写融合算子，动态执行 |
| 批处理 | in-flight batching（continuous batching 的 TRT 叫法） | continuous batching 原创实现 |
| KV 管理 | 自带 paged KV（对齐 vLLM 思路） | PagedAttention（原创，见 Q16） |
| kernel | 闭源 hand-tuned（fp8/fp4/speculative 全套） | FlashInfer/vLLM 自研 + 社区 Triton，开源可改 |
| 部署 | 单一 engine 文件、延迟极致、绑定 NVIDIA | Python 服务化灵活、多硬件后端（ROCm/XPU/TPU） |

**自研引擎对照框架**（遇到任何内部引擎都按这 6 维追问/对比）：调度粒度（req/iter）、KV 布局与置换、kernel 来源（自研/FlashInfer/cuBLAS）、量化路径、图捕获策略（CUDA Graph 覆盖哪些 shape）、PD 是否分离。差异本质：**工程取向不同**——TRT-LLM 用"离线编译换在线延迟"，vLLM 用"动态调度换通用性与迭代速度"。

### Q14. Continuous batching 的设计动机（校招-一二面-1 一面 2）

static batching 的两个浪费：① 短请求被同 batch 长请求拖住（都等最长的算完）；② batch 内槽位随完成被闲置。continuous batching 把调度粒度从"请求级"降到**迭代级**：每次 forward 前重算 batch——完成的立刻移出、排队的立刻填入、KV Cache 按各自长度增长。配合 chunked prefill（Q10）与 PagedAttention（Q16）形成现代 serving 栈的铁三角。详见[05 篇 Q2](./05-推理引擎与优化.md)。

### Q15. fastllm.cpp 的源码结构与实现思路（校招-一二面-1 一面 3）

fastllm 是国产纯 C++ LLM 推理框架（零深度依赖、CPU/GPU 双后端、支持 chatglm/qwen/moss 等早期国产模型）。以开源主线为参照：

```
fastllm/
├── src/
│   ├── fastllm.cpp/.h      # 核心：Tensor 类型（自研多维数组 + 引用计数内存）、
│   │                       #   算子表 ops: string → 函数指针/lambda 的注册分发
│   ├── model.cpp + 各模型 cpp  # basellm 基类（weights map + 虚函数 Forward），
│   │                       #   每个模型一个 cpp 手写前向（无图 IR，命令式调用算子）
│   ├── tokenizer.cpp       # BPE/sentencepiece 词表加载与编解码
│   └── devices/
│       ├── cpu/  fastllm-cpu.cpp + computethreadpool.cpp   # 算子 CPU 实现 + 线程池分块并行 GEMM
│       └── cuda/ fastllm-cuda.cu                            # 逐算子 CUDA kernel（GEMM 走 cuBLAS）
├── main.cpp                # CLI 对话入口（socket 服务可选）
└── tools/fastllm_pytorch.py  # HF 权重转换 + 量化脚本（唯一用到 PyTorch 的地方）
```

**实现思路要点**：① 极简依赖：推理核心只靠标准库，权重格式自定义（fp16/int4/int8 打包 + per-channel scale）；② weight-only 量化的 GEMM **边加载边反量化**（混合精度累加），省带宽；③ CPU 端用 compute thread pool 把 GEMM 按行块切给多线程，GPU 端每算子一个实现、通过算子注册表按 device 分发；④ KV Cache 按请求连续存放、多轮对话 append；⑤ 模型结构不建图，直接 C++ 写 forward 顺序——简单可读，代价是没有图级优化（fusion/内存复用弱于 vLLM/TRT-LLM，这也是面试可点评的"缺点"）。验证：`git clone github.com/ztxz16/fastllm && mkdir build && cd build && cmake .. && make -j && ./main -p model.flm`。

### Q16. vLLM 与 PagedAttention 的核心原理（校招-一二面-1 二面 1）

要点复述（详见[04 篇 Q6](./04-attention与kvcache.md)、[05 篇 Q2](./05-推理引擎与优化.md)）：借鉴操作系统虚拟内存分页——KV Cache 按固定 block（16 token）切分，逻辑序列 → 物理块通过 **block table** 映射，块按需分配、不必连续。收益：① 显存碎片/预留浪费从 60-80% 降到 <4%，同样显存并发翻数倍；② copy-on-write（共享 prefix）与 prefix caching（前缀块复用）；③ 配合 continuous batching 实现迭代级调度。decode kernel 按 block table 间接寻址访问 KV（见 Q17）。

### Q17. 使用 Triton 实现 PagedAttention 的思路（校招-一二面-1 二面 2）★

**数据布局**：KV Cache 不再是 `[batch, seq, head, dim]` 连续张量，而是 `[num_blocks, block_size, num_kv_heads, dim]` 的物理块池 + `block_tables[batch, max_blocks]`（int32，存逻辑块→物理块 id）+ `seq_lens[batch]`。

**kernel 设计（decode，一个 program 处理一个 (seq, q_head)）**：

1. `grid = (num_seqs, num_q_heads)`；`kv_head = q_head // group_size`（GQA 映射）；
2. 载入该 head 的 q 向量 `[HEAD_DIM]` 进寄存器；
3. 循环 KV 块：`block_id = load(block_tables[seq, i])` → 按 `block_id*stride` 间接寻址载入 K/V tile `[BLOCK_SIZE, HEAD_DIM]`；
4. 每 tile 算 `qk = sum(q * k, axis=1) * scale`，**online softmax**（running max `m`、归一化和 `l`、`acc = acc * α + p @ v`）——不物化整行 score；
5. 末块 mask 掉 `offs_n >= seq_len`（置 −inf），最终 `out = acc / l` 写回。

预填充则换成 Q 外层块 × KV 内层块的二维 tiling（同 FlashAttention v2，KV 侧改间接寻址）。完整可运行实现（与 torch 参考对拍）：

```bash
cd solutions/python && pytest tests/test_tencent.py -v -k paged   # Triton PagedAttention vs torch 参考对拍
```

代码：`solutions/python/src/answers/tencent/triton_paged_attention.py`（含 GQA/块表乱序/尾块 mask 的测试）。

### Q18. CUDA 内存模型的层次结构与特点（校招-一二面-1 二面 3）

复述要点（详见[01 篇 §2/§21](./01-cuda-基础与体系结构.md)）：寄存器（per-thread，最快，溢出→local memory 实为 Global）→ Shared Memory（per-SM，~百 KB，可显式管理，bank conflict 注意）→ L1/L2（硬件缓存，L1 per-SM 可与 smem 复制配置，L2 全卡共享）→ Global/HBM（大而慢，合并访问）→ 常量内存/纹理（缓存广播读）。每层差异三件事：**容量、延迟、可见性**（smem 需要 `__syncthreads`，global 需要 fence/原子）。

### Q19. YOLO 推理加速的常见方法（校招-一二面-1 一面 6）

- **模型侧**：剪枝（通道/层结构性剪枝 + 微调）、蒸馏（大模型教小检测头）、更小 backbone（MobileNet/GhostNet 替换 CSP）、REPVGG 式重参数化（训练多分支→部署单分支）；
- **量化**：PTQ int8/fp16（TensorRT calibration）、QAT 保 mAP；检测头回归分支对量化敏感可混合精度；
- **部署/图优化**：TensorRT/ONNX Runtime 算子融合（Conv+BN+SiLU）、fp16、CUDA Graph 消 launch、kernel autotune；
- **流水线**：预处理（resize/letterbox/归一化/NHWC）GPU 化 + zero-copy（pinned memory / CUDA-GPU 解码 NVDEC）、NMS 融合进网络尾部或 plugin 化、后处理与下帧前处理异步多流；
- **调度**：动态 batch、多路视频流共享 GPU、输入分辨率自适应（远处小目标才用大分辨率）。
答题主线：先 profile（预处理/推理/后处理占比），再针对瓶颈选手段，mAP 回退用蒸馏/QAT 兜底。

## 三、高性能计算与 CUDA

### Q20. MPI 的基本概念与使用方式（实习-一二面 一面 2）

MPI = 消息传递接口的**标准 + 实现**（OpenMPI/MPICH/IntelMPI），SPMD 编程模型：同一程序启动 N 个 rank，各自有独立地址空间，靠消息通信。最小 API 集（"六个函数写完 MPI 程序"）：`MPI_Init/Finalize`、`MPI_Comm_rank/size`（拿自己在 `MPI_COMM_WORLD` 里的编号）、`MPI_Send/Recv`；再补集合通信 `MPI_Bcast/Scatter/Gather/Allgather/Reduce/Allreduce/Reduce_scatter/Alltoall`、同步 `MPI_Barrier`，以及非阻塞 `Isend/Irecv`。

```c
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
// ... 本地计算 + Allreduce 梯度 ...
MPI_Allreduce(local, global, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
MPI_Finalize();
```

与 AI Infra 的关系：**NCCL 就是"GPU 版 MPI"的集合通信库**（同一通信原语族，数据在显存间直接搬）；torchrun/ddp 的世界大小与 rank 概念直接继承 MPI；跨机训练的 host 通信、存储 IO 也常用 MPI。运行 `mpirun -np 8 --hostfile hosts ./a.out`。

### Q21. 多线程的基本原理；常见的锁；死锁的条件与解决（实习-一二面 一面 3/4/5）

- **多线程原理**：同进程内多条执行流共享地址空间，OS 调度器按时间片/优先级在核间分派；切换只需保存寄存器/栈指针（无地址空间切换），但共享数据需要同步（可见性：store buffer/cache 一致性由 MESI 保证，重排序需内存屏障约束）。
- **锁机制**（详见[09 篇 Q14](./09-cpp与系统基础.md)）：互斥锁 `std::mutex`（阻塞，futex 实现：先用户态自旋再陷入内核）、自旋锁（忙等，临界区极短/持有期间不想睡眠）、读写锁（读多写少）、条件变量（等待-通知，配 mutex 防**虚假唤醒**必须 while 重新判条件）、信号量（资源计数）、`std::scoped_lock`/RAII 防忘记解锁；乐观并发：原子变量/CAS、RCU。
- **死锁四条件**：互斥、持有并等待、不可剥夺、循环等待。预防：**破坏循环等待**——全局锁序（按地址/编号排序加锁）、`std::scoped_lock(l1, l2)` 死锁避免算法、try_lock + 退避、一次性申请全部资源。定位：gdb attach 看各线程栈 + `pthreads mutex owner`，或 pstack/coredump。

### Q22. UDP 与 TCP 的异同；TCP 靠哪些机制保证可靠传输（实习-一二面 一面 6/7）

| | TCP | UDP |
|---|---|---|
| 连接 | 面向连接（三次握手/四次挥手） | 无连接 |
| 可靠 | 可靠有序、不丢不重 | 尽力而为 |
| 传输 | 字节流（有黏包问题） | 数据报（有边界） |
| 拥塞/流控 | 滑动窗口 + 拥塞控制 | 无 |
| 头部 | 20B+ | 8B |
| 适用 | HTTP/RPC/训练同步 | 视频/游戏/KV 传输(QUIC 前) |

**TCP 可靠性机制**：① 序号 + 确认应答（ACK 累计确认）；② **超时重传**（RTO 自适应 RTT 估计）+ 快速重传（3 个冗余 ACK）；③ 滑动窗口做**流量控制**（接收方 rwnd 通告，零窗口探测）；④ **拥塞控制**（慢启动 → 拥塞避免 → 快重传 → 快恢复，cwnd 调节发送速率，丢包视为拥塞信号）；⑤ 校验和检错；⑥ 连接管理（握手同步初始序号，挥手+TIME_WAIT 保证最后 ACK 可达与旧报文消亡）。AI Infra 加分点：RDMA/NCCL 走 RoCE 时恰恰**绕过** TCP 栈换取微秒级延迟，TCP 拥塞控制反而是大规模训练同步的瓶颈（所以要专用网络/自适应路由）。

### Q23. 对 CPU/GPU 架构了解到什么程度；是否有写 CUDA kernel 的经验（实习-一二面 二面 1/2）

开放题，答法是"分层陈述 + 证据"。CPU：乱序执行/分支预测/SIMD(AVX)/三级缓存/NUMA，Evidence 举优化过的例子（向量化、cache blocking）；GPU：SIMT/warp/SM 层级/内存层级/Tensor Core（引[01 篇](./01-cuda-基础与体系结构.md)）。kernel 经验按本仓库素材组织即可：归约（原子 vs 树形 + warp shuffle，124×）、Softmax 在线版、GEMM tiling、转置消 bank conflict、直方图私有化（19.9×）——全部在 `solutions/cuda/` 可复现，见[02 篇](./02-cuda-编程题.md)。

### Q24. C++ POD 类型的定义与特性（实习-一二面 二面 3）

POD（Plain Old Data）= **trivial（平凡）+ standard-layout（标准布局）**：

- trivial：无自定义默认构造/拷贝/移动/析构（或全部 trivial）、无虚函数虚基类——"行为像 C 结构体"；
- standard-layout：所有非静态成员同一访问控制、无虚函数/虚基类、基类也是 standard-layout、无与首个成员同类型的基类等——"布局像 C 结构体"。

**特性/意义**：可以 `memcpy`/`memset`、与 C ABI 互通（跨语言/dlopen 边界）、聚合初始化 `T obj{...}`、保证成员按声明序排布（可有 padding）、可安全union 化、序列化/网络传输直接按字节拷。判断：`std::is_pod_v<T>`（C++20 起废弃，拆成 `is_trivial_v && is_standard_layout_v` 更精确）。陷阱：加了构造函数/虚函数/成员访问控制混杂 → 非 POD，`vector::resize` 等性能路径行为随之改变。

### Q25. 除 priority_queue 外，还有哪些 STL 容器可实现堆（实习-一二面 二面 4）

- **`vector`/`deque` + `<algorithm>` 堆算法**：`make_heap/push_heap/pop_heap/sort_heap/is_heap`——priority_queue 本身就是 vector+这些算法的适配器，直接用容器可以**同时保留堆序和中途遍历**（priority_queue 不行）；
- **`multiset`/`multimap`**（红黑树）：O(log n) 插删、`*begin()/*rbegin()` 取最值，还支持按值删除（`erase(key)`）与迭代（有序）——代价是指针开销与缓存不友好，常数比二叉堆大；
- 手写：数组即可（堆本质是数组上的完全二叉树），`deque` 当底层数组也行；
- 对比点：堆只能取最值 O(1)+插入 O(log n)；有序 set 支持任意删除/查询第 k 小（配合 order-statistics tree）。答"双堆求中位数"（[42 篇 Q4](./42-腾讯-编程题.md)）时两个 priority_queue 就是标准姿势。

### Q26. 解释 shuffle、Hyper-Q、SM、slot（实习-一二面 二面 5）

- **shuffle**：CUDA 语境指 **warp shuffle**——同 warp 内 lane 间寄存器直接交换（`__shfl_sync/__shfl_down_sync/...`），不经共享内存，是 warp 内归约/广播的基础原语（[02 篇 Q1](./02-cuda-编程题.md) 的树形归约每层用它）；
- **Hyper-Q**：Kepler 引入的**多硬件工作队列**（默认 32 个）——Fermi 时代单队列时，同一流内等待会让后续无依赖的 kernel 也被假性阻塞（false serialization）；Hyper-Q 用多队列让就绪 kernel 真正并发，配合 MPS 让多进程/多 CUDA 上下文共享 GPU（每 SM 上下文 slot）；
- **SM**：流式多处理器，GPU 的"核心"调度执行单元，含 warp 调度器、寄存器堆、INT32/FP32/Tensor Core、共享内存/L1（[01 篇 §4](./01-cuda-基础与体系结构.md)）；
- **slot**：语境相关——① Hyper-Q/MPS 的上下文 slot（每 SM 可驻留的 client 数）；② warp scheduler 的发射 slot（每 SM sub-partition 每周期发射指令槽）；③ 推理引擎的 batching slot。回答时先反问语境再对号入座。

### Q27. Nsight 的用途与使用；SOL 指标的含义（实习-一二面 二面 6/7）

详见[08 篇 Q5](./08-编译器与性能分析.md)。要点：Nsight Systems 看时间线（kernel 间隙/H2D/多流重叠/launch 延迟），Nsight Compute 看 kernel 内部（占用率、stall 原因、L1/L2/DRAM 命中与事务）。**SOL（Speed of Light）**= kernel 的实际吞吐占硬件理论峰值的百分比，两个核心数：Compute SOL 与 Memory SOL——**谁接近 100% 谁是瓶颈**，两者都低说明 neither：受并行度（occupancy 不足/串行依赖/分支发散）或调度空隙限制。优化就用它定方向：Memory 打满→省流量/重布局，Compute 打满→少算/换算法（如 Tensor Core/精度）。

### Q28. 多面体模型、TVM、Halide（实习-一二面 二面 8/9/11）

三题都已在[08 篇 Q1/Q2](./08-编译器与性能分析.md)详解，面试要点复述：

- **多面体模型**：循环嵌套建模成整数多面体（迭代空间+依赖不等式），仿射变换自动推导交换/分块/倾斜/融合且**保证依赖保持**（isl/Polly/MLIR affine）；
- **TVM**：计算与调度分离（算什么 `T.compute` vs 怎么算 schedule），Relay 图级 → TIR 张量级 → codegen，AutoTVM/Ansor 自动搜调度；
- **Halide**：计算-调度分离的鼻祖，函数式图像 DSL，强调 **producer-consumer pipeline 级**融合调度与 rule-based 自动搜索（auto-scheduler）。
串场答法：三者共同命题是"**算法与硬件映射解耦**"——多面体给变换的数学保证，TVM/Halide 把它工程化成可写、可搜的调度语言。

### Q29. AoS 与 SoA 的区别及适用场景（实习-一二面 二面 10）

AoS（Array of Struct）：`struct P{float x,y,z;} ps[N]`，一条记录连续；SoA（Struct of Array）：`float xs[N], ys[N], zs[N]`，一个字段连续（[01 篇](./01-cuda-基础与体系结构.md)访存节同考点）。

- **GPU/SIMD 选 SoA**：访问模式是"对全部元素的同一字段"（如只更新 x）时，SoA 每次 warp 访问 32 个连续 float → **coalesced/SIMD lane 高效**；AoS 会把 y/z 也搬进来浪费 2/3 带宽；
- **AoS 优势**：单对象处理（物理仿真中频繁访问同粒子全字段）、按记录传输/序列化、指针传递一条记录、构造析构成组发生；
- 折中：AoSoA（`struct{float x[8],y[8],z[8]}` 每 8 个对象一组）——组内 SoA 利于 SIMD、组间 AoS 利于局部性；数据库行列存储就是同构问题（[41 篇 Q11](./41-腾讯-teg-数据库与os.md)）。

### Q30. Warp 级 reduce 的实现方式（实习-一二面 二面 12）

同 warp 32 lane 的树形归约，每步折半：`for (off=16; off; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);`——`__shfl_down_sync` 让 lane i 取 lane i+off 的值，5 步后 lane0 持和，无共享内存无同步开销。跨 warp 再上一层：各 warp 结果写 shared memory → 指定 warp 重做一遍（或 atomicAdd）。完整实现与 3090 实测（vs 原子版 124×）：`solutions/cuda/src/reduce.cu`，讲解见[02 篇 Q1](./02-cuda-编程题.md)。变体追问：`__shfl_xor_sync`（蝶形，所有 lane 都拿到和）、`__reduce_add_sync`（sm_80 原生指令）、block 级两阶段。

### Q31. reduce-scatter 与 all-to-all 的比较（实习-一面-1 高性能 4）

集合通信语义（详见[07 篇 Q2](./07-分布式训练.md)）：

- **reduce-scatter**：各 rank 持等长向量，逐元素求和后**按 rank 切片分发**——每个 rank 拿到总和自己负责的那 1/N 段。是 all-reduce 的组成半部（RS + all-gather = ring all-reduce，通信量 $2\alpha + 2(N-1)/N \cdot S$）；用于 ZeRO 数据并行的梯度分桶（每卡只更新自己参数分片）；
- **all-to-all**：每个 rank 给每个其他 rank 发**不同且不等长**的数据（per-rank 消息），总通信量 $\sum s_{ij}$，实现上比规则 collective 难——消息大小不均时网络负载倾斜；用于 MoE 的 dispatch/combine（token 按 router 目的专家发往对应 rank）与 TP 的序列维重排（sequence parallel 的 gather/scatter）。
- 一句话对比：reduce-scatter 是"**同构归约+分片**"，all-to-all 是"**异构交换**"；前者优化目标是带宽利用率（ring），后者是负载均衡（deepseek 的 dispatch 均衡 token 数 + 传输 overlap）。

### Q32. 有哪些方法可以降低 kernel launch 的开销（实习-一面-1 高性能 5）

launch 开销 ≈ 每次 3-10μs 的 CPU 侧成本（驱动校验/参数拷贝/队列提交），小 kernel 密集场景成为瓶颈：

1. **减少 launch 次数**：算子融合（[03 篇 Q6](./03-gemm与算子优化.md)）、一个 kernel 干完；
2. **CUDA Graph**：捕获一次、整图重放，N 次 launch 合并为 1 次图提交（消除 CPU 逐个提交，也是 decode 延迟稳定的关键，vLLM 全图捕获）——详见[01 篇 §9](./01-cuda-基础与体系结构.md)；
3. **消除同步**：异步提交、避免 stream query/pinned 读回打断流水、事件代替忙等；
4. **persistent kernel**：一个常驻 kernel 自旋消费任务队列（零 launch，权衡占用 SM）；
5. 减小 host 侧开销：PyTorch 场景 `torch.compile(mode="reduce-overhead")`（即 CUDA Graph）、C++ extension 直接 launch 避开 Python 调度、kernel 参数放 constant；
6. 多流并不能降单次开销，但能藏 launch 间隙（与计算重叠）。

### Q33. bank conflict 的产生原因与解决方案（实习-一面-1 高性能 6）

详见[01 篇 §11](./01-cuda-基础与体系结构.md)、[02 篇 Q4](./02-cuda-编程题.md)。原因：shared memory 32 个 bank × 4B 交错划分，同一 warp 内 ≥2 lane 访问**同一 bank 的 不同地址** → 硬件串行化（n 路 conflict 拖 n 倍）。典型：按列访问 `s[threadIdx.x][k]`（stride 32 float = 32 bank 全撞）、矩阵转置直通写。解决：**padding**（行宽 +1 float 打乱 stride）、swizzle/异或重排线程-数据映射、按行访问改布局（`[k][threadIdx.x]`）、`cuda::memcpy_async`/向量化 `float4` 访问（128bit 一次事务拆 4 bank）、广播访问不算 conflict（同地址）。

## 四、C++ 与系统基础

### Q34. 指针与引用的区别；虚函数实现机制；继承的底层实现（实习 基础 1/2、校招 基础 5）

前两问详见[09 篇 Q1/Q8](./09-cpp与系统基础.md)，要点复述：

- **指针 vs 引用**：指针是对象（有地址、可重指向、可为空、可算术）；引用是别名（必须初始化、不可重绑、无空引用、sizeof 是被引对象）；函数传参语义差异与汇编层面（引用≈自动解引用的指针）。
- **虚函数**：对象头部 vptr → vtable（函数指针表，类一级一份）→ 虚调用 = 两次间接寻址 `(*(p->vptr)[i])(p)`；代价：一次额外访存 + 不可内联。多继承时多个 vptr，跨非首基类调用经 thunk 调整 this。
- **继承底层**：单继承下派生类对象 = 基类子对象（在前，含 vptr）+ 自己成员；基类指针指向派生对象地址不变（首基类）；多继承按声明序拼接多个基类子对象，各带 vptr，转换到不同基类时 this 指针做常量偏移；虚继承加 vtt/vbptr 间接定位虚基类（共享一份）。空基类优化（EBO）允许空基类占 0 字节。对二进制的影响：加基类成员/虚函数会改布局，跨 so 边界要固定 ABI。

### Q35. C++ 函数调用时的栈帧压栈过程（实习 基础 3）

x86-64 System V 调用约定，`f(a,b,c)` 的完整时序：

1. **caller**：前 6 个整型参数放 rdi/rsi/rdx/rcx/r8/r9（浮点走 xmm0-7），多余参数**从右向左压栈**；`call f` 指令把**返回地址**压栈并跳转；
2. **callee prologue**：`push rbp; mov rbp, rsp`（保存调用者帧基址，建立本帧 rbp）；`sub rsp, N` 给局部变量留空间；再保存被调用者保存寄存器（rbx/r12-r15 等）；需要时开栈保护 canary（`fs:0x28`）；
3. 函数体执行，局部变量、临时对象都在 `[rbp-x]` 寻址；
4. **返回**：返回值放 rax（>16B 交给 caller 传隐藏指针 sret）；epilogue `leave`（mov rsp,rbp; pop rbp）+ `ret` 弹返回地址；caller 负责清参数区（或平栈）。

要点：**栈从高向低生长**；rbp 固定便于调试器回溯调用链（frame pointer chain）；栈帧大小通常编译期定；深递归栈溢出与爆栈检测（guard page）、栈上大数组（alloca/VLA）都是这个模型的推论。

### Q36. 如何将函数作用域内的局部变量返回到外部使用（实习 基础 4）

- **按值返回**（默认正确姿势）：C++17 起 prvalue **强制省略拷贝**（RVO），局部对象直接在外部调用者的栈上构造，零拷贝；即使 NRVO 不发生，还有移动构造兜底（`return std::move(local)` 反而**禁用 NRVO**，不要画蛇添足）；
- **返回堆指针**：`return std::make_unique<T>(...)`——所有权明确（unique_ptr），堆对象生命周期逃逸函数栈帧；
- **out 参数**：caller 传入引用/指针，函数内填充（大对象避免任何拷贝争议）；
- **错误做法**：返回局部变量的**引用/指针**（`return &local` / `return local_ref`）→ 悬垂，UB；返回静态局部对象（线程不安全 + 状态共享）；返回 lambda 捕获局部引用同理悬垂。
- 补充：`std::function`/闭包按值捕获是安全的（拷贝进闭包）；协程场景局部变量被搬到堆上的协程帧。

### Q37. Python 计算密集型任务用多进程还是多线程（校招 基础 4）

**多进程**。GIL 保证同一时刻仅一线程执行 Python 字节码，计算密集线程无法并行（还因切换/缓存抖动更慢）；多进程各自独立解释器绕开 GIL，可占满多核（`multiprocessing.Pool`/`concurrent.futures.ProcessPoolExecutor`，注意 pickle 开销与写时复制内存）。多线程适合 **IO 密集**（等待时释放 GIL）。例外：计算已下沉到 native（NumPy/torch 的 C 扩展在计算时释放 GIL），多线程也有效；或 `torch.set_num_threads` + 进程级并行。详见[09 篇 Q19](./09-cpp与系统基础.md)。

### Q38. C++ 编译期运行的实现方式（constexpr 等）（校招-二面 基础 4）

- **`constexpr` 函数/变量**：标注后可在编译期求值（传常量实参时）；C++14 起允许循环/分支，C++20 起允许 try/虚调用/`new`/`std::vector`（constexpr 容器，编译期建表运行期零成本）；
- **`consteval`（C++20）**：立即函数，**只许**编译期调用；`constinit`：保证静态初始化（防静态初始化顺序灾难）；
- **模板元编程**：模板特化 + 递归实例化就是编译期图灵机（编译期斐波那契/阶乘/类型计算）；`std::integral_constant`、变量模板、C++17 fold 表达式；
- **编译期断言/查表**：`static_assert(f())`；编译期生成查找表 `constexpr std::array table = make_table();`（CRC 表/softmax 查表）；
- **SFINAE / `if constexpr` / concepts**：按类型在编译期选择分支，裁剪实例化代码（`if constexpr` 编译期剪枝 else 分支）；
- 其他"编译期计算"技术：宏（预处理器级，弱类型）、模板生成代码（CRTP 静态多态见[09 篇 Q9](./09-cpp与系统基础.md)）。

```cpp
constexpr uint32_t crc32_table = build_table();  // 编译期建表
static_assert(is_power2(N));                     // 编译期校验
consteval int sq(int x) { return x * x; }        // 只能编译期
int arr[sq(4)];                                  // 用作常量表达式
```

工程化：`solutions/cpp/tests/test_language_basics.cpp`（constexpr CRC32 建表 + `static_assert` 编译期校验；consteval 为 C++20，C++17 以 constexpr+static_assert 等效约束）。

### Q39. C# 事件机制；如何设计一个事件系统；事件如何与 GameObject 生命周期绑定（实习 基础 5/6/7）

- **C# 事件机制**：`delegate`（类型安全函数指针，可组播 MulticastDelegate，`+ =`/`- =` 增删调用列表）+ `event` 关键字修饰委托字段——外部只能 `+=`/`-=` 订阅退订，**不能直接触发或清空**（封装）；触发即按调用列表顺序回调。本质对比 C++：委托 ≈ `std::vector<std::function>` 的语法糖 + 语法级限制。
- **设计事件系统**（游戏/引擎通用）：① 定义强类型事件 `Event<TArgs>` 或中心总线 `EventBus.Subscribe(topic, handler)/Publish(topic, args)`；② 关键设计点：**退订安全**（回调中增删订阅 → 迭代副本或延迟队列）、异常隔离（单个 handler 抛错不中断广播）、线程模型（主线程派发 vs 加锁）、优先级、**弱引用持有**防泄漏；③ 数据驱动（topic 用枚举/int id 而非字符串）便于裁剪。
- **与 GameObject 生命周期绑定**：规则是 **OnEnable 订阅 / OnDisable（或 OnDestroy）成对退订**；静态事件/单例总线持有实例方法的委托会**阻止 GC 回收对象**（委托强引用 target），漏退订 = 内存泄漏 + 对已销毁对象回调（MissingReferenceException）；工程做法：封装 `SubscribeAutoDispose`（ IDisposable + UsingScope，对象销毁时自动退订）、或弱引用委托表。

### Q40. UI 优化中的合批如何实现（实习 基础 9）

合批（batching）= 把多次 DrawCall 合成一次，减少 CPU 提交与状态切换。要点：① **合批条件**：同材质/同纹理图集（UV 偏移进图集）+ 同 shader 变体 + 相邻渲染顺序；做法是图集合并、材质合并、打断排序重排；② **动静分离**：静态 UI 与动态 UI 拆不同 Canvas/层级，避免动态元素变化触发整个 Canvas 网格重建（Canvas rebuild 是 UI 卡顿主因）；③ **Canvas 拆分**：按功能模块拆，重建代价局部化；④ 现代管线：**SRP Batcher**（同 shader 批次化材质 CBUFFER）、**GPU Instancing**（同网格同材质 N 实例一次画）、静态合并 Mesh；⑤ 减少 Overdraw（透明叠加区域）与 Raycast Target 开销。该题属游戏客户端方向，AI Infra 岗答到"DrawCall 合并的条件 + 材质/图集 + 实例化"即可。
