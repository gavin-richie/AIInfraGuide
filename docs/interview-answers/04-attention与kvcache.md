---
title: "Attention 与 KV Cache"
description: "FlashAttention v1/v2/v3、GQA/MQA/MLA、PagedAttention、Flash Decoding、KV Cache 计算与优化题解"
pubDate: 2026-10-05
topic: "推理优化"
order: 4
tags: ["推理优化", "算子优化"]
---

覆盖题库：`AI-Infra-综合面经题库-1`（基础 2/3）、`AI-Infra-综合面经题库-2`（基础 1、PyTorch 编程 1/2）、`AI-Infra-一面`（5/7）。

**配套可运行代码**（`solutions/python/src/answers/attention/`）：`mha.py / gqa.py / mla.py / online_softmax.py / flash_attention.py / paged_attention.py`，全部与 PyTorch 参考实现对拍通过（`pytest solutions/python/tests/test_attention.py`，76 项测试全绿）。

## Q1. FlashAttention 核心原理与实现思路（综合-2 基础 1、AI-Infra-一面 5、百度/快手/美团高频）

**问题**：标准 attention 需要物化 S×S 的 attention 矩阵，HBM 读写 O(N²) 次，而数学上只输出 O(N)——访存浪费。

**三个关键技术**：

1. **Tiling（分块）**：Q/K/V 切成块（如 64/128），块装进 SRAM（shared memory）计算，中间结果不落 HBM。
2. **Online Softmax（在线 softmax）**：softmax 分母 $\ell = \sum e^{x - m}$ 无法一遍得到。维护运行中的 $(m, \ell, O)$，新块到来时重归一化：$m_{new}=\max(m, m_b)$，$\ell_{new} = \ell \cdot e^{m-m_{new}} + \sum e^{x_b - m_{new}}$，$O_{new} = O \cdot e^{m-m_{new}} + P_b V_b$。全程只需常量额外内存。
3. **kernel fusion**：QK^T、mask、softmax、PV 全在一个 kernel。

**效果**：HBM 访存从 O(N²) 降到 O(N²·d/BLOCK)（按块装载），显存 O(N)（不存矩阵），长序列训练可行。

**v1 → v2 → v3 演进**（高频追问）：

| 版本 | 关键改进 |
|---|---|
| v1 | 外层遍历 **KV**、内层 Q：一个 KV 块被所有 Q 块使用，中间 $O_{ij}/\ell_{ij}$ 要写回 HBM 再 reduce |
| v2 | **内外循环交换**：外层 Q、内层 KV——(m,ℓ,O) 驻留寄存器，一个 Q 块算完只写回一次，HBM 访存再降 ~半；非 matmul FLOPs 大幅减少；Q 块间/头/batch 多维并行，长序列小 batch 也打满 SM |
| v3 | Hopper 专属：warp specialization + TMA 异步装载 + mma 与 softmax 的 **intra-warpgroup pipelining**，FP8 支持，1.5-2× 于 v2 |

**手写实现**（面试要求）：`solutions/python/src/answers/attention/flash_attention.py` 中 `flash_attention_v1`（外层 KV）与 `flash_attention_v2`（外层 Q + causal 块裁剪）与 `F.scaled_dot_product_attention(is_causal=True)` 对拍，最大误差 < 1e-4。CUDA Online Softmax 单行版见 `solutions/cuda/src/softmax.cu`。

## Q2. FA v2 外层为何遍历 Q；Flash Decoding combine 占比（综合-1 基础 2）

**为何外层 Q**：并行写回与状态驻留。若外层是 KV，每个 Q 分块的中间输出 $O^{(j)}$ 与 $\ell^{(j)}$ 需要写回 HBM（因为不同 KV 块的计算分布在不同 CTA/时刻，最终要合并），访存与同步翻倍。外层 Q 时，一个 Q 块顺序吃掉所有 KV 块，中间状态始终在寄存器，**只在结束时写一次最终输出**——这正是 v2 论文标题 "Faster Attention" 的核心。同时 Q 块数 = 并行度，天然适配 GPU。

**Flash Decoding**：decode 时 Q 长度为 1，Q 维没有并行度 → 把** KV 序列维切分**给不同 block 并行算部分 $(m_i, \ell_i, O_i)$，再用一个小 **combine kernel** 合并（与 online softmax 的合并公式一致）。combine kernel 的耗时占比：它是 O(batch×head×num_split×d) 的小归约，通常 **<5%**（如序列 16K、split 8 时），绝大部分时间在主 kernel 的 KV 扫描——但 split 过多（小 KV）时 combine 占比上升甚至负优化。

## Q3. MLA decode 的计算访存比与 seq/batch 的关系（综合-1 基础 3）

MLA 每层每步（吸收后）对每个 token 的 KV 访存 = $d_c$（latent 维，如 512）+ RoPE 部分 $d_r$（如 64），远小于 MHA 的 $2 n_h d_h$。

- **访存量**：$\text{KV bytes} \propto (d_c+d_r) \cdot S \cdot B$；decode 单 token 的**计算量**：$QK^T + PV \approx 2 S (d_c d_h + d_h d_c) n_h B$（吸收后在 latent 空间）。
- 单 token 访存/计算比与 S 无关的常数比 + 随 batch 的计算复用：**batch 增大 → 同一份 KV 被 B 个请求复用（若同 batch 共享序列如系统 prompt，L2 命中）→ 更偏计算瓶颈；S 增大 → KV 线性涨而计算也线性涨，比值近似不变，但绝对延迟线性变差**。
- 结论表述：MLA 把 decode 的 KV 访存压缩 ~1/ n_h 量级，使 decode 更接近 compute bound；但在 batch 小、序列极长时仍 memory bound（KV 读主导）。

配套：`mla.py` 的 `forward_naive`（展开路径）与 `forward_absorb`（吸收路径）数值等价（测试 `test_mla_absorb_equivalence`）。

## Q4. 手写含 GQA 的 Attention（AI-Infra-一面 7、阿里/京东）

关键点：KV 头数 $n_{kv} <$ Q 头数 $n_h$，每组 `group = n_h / n_kv` 个 Q 头共享一个 KV 头——计算前把 KV 头按组扩张（`repeat_interleave(group, dim=head)` 或 expand+reshape），其余与 MHA 相同。KV Cache 显存 = $2 \cdot L \cdot n_{kv} \cdot d_h \cdot S \cdot B \cdot \text{bytes}$，GQA 相对 MHA 直接按 $n_{kv}/n_h$ 缩减。

**代码**：`gqa.py` → `GroupedQueryAttention`（`n_kv_heads=1` 即 MQA，`=n_heads` 即 MHA），与 SDPA 对拍通过；`kv_cache_size()` 封装显存公式。

## Q5. 手写 MHA 三变体（综合-2 PyTorch 编程 1）

**代码**：`mha.py` → `MultiHeadAttention`，一次实现覆盖三种问法：

1. **self-attention**：Q=K=V=X；
2. **cross-attention**：Q 来自解码器，K/V 来自编码器（S_q ≠ S_kv）；
3. **causal（decoder-only）**：上三角 mask 置 -inf。

必备细节：`(B,S,E)→(B,S,H,D)→transpose(1,2)`；scale $1/\sqrt{d_h}$；softmax 前减 max（`F.softmax` 内建）；输出 `transpose(1,2).contiguous().view()`（transpose 后非连续必须 contiguous 才能 view——高频追问，见《算法题》篇 torch 语义题）。

## Q6. PagedAttention 思想与实现（字节、快手、京东）

**问题**：传统推理按 max_len 连续预留 KV → 内碎片（预留未用）+ 外碎片（新请求放不下要整理拷贝），显存利用率 20-40%。

**方案**（借鉴操作系统分页）：KV Cache 切成固定 block（vLLM 默认 16 token/block）；逻辑连续、物理离散，`block_table[req][逻辑块号] = 物理块号`；copy-on-write 支持 beam search 共享前缀；显存利用率 >90%。

**kernel 视角**：decode 单步第 i 个 token 的 KV 寻址 = `k_pool[block_table[i//block_size], i%block_size]`——kernel 内直接查表，无 gather 物化。

**代码**：`paged_attention.py` 实现了 BlockAllocator / block_table / decode 寻址，测试验证"两请求交错写入后物理块离散、逻辑连续，decode 输出与稠密 attention 一致"。

## Q7. KV Cache 优化全景（快手、蚂蚁、百度高频）

1. **减少每 token 大小**：GQA/MQA/MLA（结构层）；KV 量化 FP8/INT4（W8A8 KV 或 W4 KV，需 per-block/per-channel scale，attention score 对量化敏感→校准更难）。
2. **压缩/淘汰**：滑动窗口、StreamingLLM（sink tokens + 近窗）、H2O 重打分驱逐、KIVI（按通道非对称量化 KV）。
3. **存储位置**：KV offload 到 CPU/SSD（Mooncake），重计算（layer-wise recompute）换取零 KV 存储。
4. **复用**：Prefix cache（系统提示前缀块共享，hash 前缀索引）、prompt cache。
5. **调度**：PagedAttention 打底 + 抢占换出（vLLM swap/recompute）。

显存估算公式（多家考）：$\text{bytes} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot B \cdot S \cdot \text{dtype}$，例：Qwen-7B（32 层、32 头、d_h 128、fp16）B=1 S=2048 → 1.07GB；换 GQA 8 头 → 268MB。`gqa.py::kv_cache_size` 可直接算。

## Q8. RoPE 与 Attention 结合的数值细节（快手-一面-2 常考补充）

旋转位置编码对 Q、K 按二维旋转 $\binom{q_{2i}}{q_{2i+1}} \to R(\theta_i m)\binom{q_{2i}}{q_{2i+1}}$，使 $q_m^T k_n$ 只依赖相对位置 $m-n$。长上下文失真原因：高频分量旋转角累积大、插值（linear/NTK/YaRN）改变 θ 基频会损失短程分辨率。MLA 中因为"吸收后无法保持相对位置结构"，RoPE 部分必须单独缓存不参与吸收（decoupled RoPE）——这也是 Q4 吸收推导的边界条件。
