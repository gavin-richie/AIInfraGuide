---
title: "字节跳动 · RLHF 与场景/开放题"
description: "字节跳动面经 RLHF 方向与场景题解：SFT 数据与流程 / PPO 全流程 / DPO 与 PPO 对比 / GRPO / 微调评估 / 刷赞识别 / 项目题方法论 / 编程题总索引"
pubDate: 2026-10-05
topic: "公司题解-字节跳动"
order: 24
tags: ["RLHF", "对齐", "场景题", "公司题解"]
draft: false
---

覆盖题库：`抖音电商-AI-Infra`（训练优化 1-7、项目 15、编程 16）、`豆包-实习-二面`（基础知识 3-4、项目 6、编程 7）、`抖音-AI-Infra`（场景题 6、项目 7）、以及 13 个文件中的全部编程题与项目经历题（汇总于 Q9/Q10）。

**配套代码**：`solutions/python/src/answers/bytedance/rlhf.py`（PPO clip / KL k3 估计 / DPO / GRPO advantage，全部解析对拍）；SFT loss 见 `solutions/python/src/answers/torch_impl/sft_loss.py`。

```bash
cd solutions/python && pip install -e . && python -m pytest tests/test_bytedance.py -q   # 27 passed
```

## Q1. SFT 阶段使用的模型、数据来源、数据处理与训练框架（抖音电商 训练优化 1）

- **底座模型**：预训练 base 模型（如 Qwen 系列），选型看任务域与部署预算（7B 微调 + LoRA 起步，全参需多卡）。
- **数据来源**：人工标注（质量最高、贵）、强模型蒸馏（GPT-4o/Claude 生成指令-回答对，注意许可与"模型同质化"）、开源指令集清洗（OpenHermes/WildChat/UltraChat）、业务日志回流（线上真实 query + 人工/规则打分筛选，最贴合业务分布）、合成数据（Self-Instruct/Evol-Instruct 扩增）。
- **数据处理**：去重（MinHash/嵌入相似度）→ 清洗（长度/语言/敏感词/格式校验）→ 质量过滤（打分模型/规则置信度）→ 配比（通用:领域指令混合，防灾难遗忘）→ 模板化（chat template，mask prompt 段 label，只对回答算 loss——`sft_loss.py` 的实现要点）→ 长度分桶与 padding/packing。
- **训练框架**：LLaMA-Factory / Axolotl / ms-swift（配置驱动）或 DeepSpeed/FSDP/Megatron 手搭（可控性）；LoRA/QLoRA 是单卡首选（Qwen2 底座 + LoRA r=16 典型配置见《23》Q17）。
- 追问"灾难遗忘"：混入预训练语料 replay、限制 LoRA 秩、小学习率。

## Q2. PPO 的完整训练流程：数据准备、奖励模型、损失函数、计算资源（抖音电商 训练优化 2）

**四模型流水线**（每个 batch 一步）：

1. **数据准备**：prompt 池（SFT 分布采样 + 业务回流）→ rollout：**actor（策略模型）**自回归生成回答；
2. **奖励模型（RM）训练**（前置阶段）：偏序对数据（同一 prompt 的 chosen/rejected 回答，人工或强模型标注），Bradley-Terry 损失 $L_{RM} = -\log\sigma(r(x,y_c) - r(x,y_r))$；RM 通常 SFT 模型换头（标量输出），与策略同量级或稍小；
3. **PPO 更新**：打分——RM 给 outcome 奖励（末 token），**ref model（冻结的初始策略）**与 actor 的逐 token logprob 算 KL 惩罚：$r_t = r_{env,t} - \beta\cdot \widehat{KL}(\pi_\theta\|\pi_{ref})$（k3 无偏估计：$\hat{kl}_t = e^{\log p_{ref} - \log p_\theta} - (\log p_{ref} - \log p_\theta) - 1$）；GAE 算 advantage；**critic（价值模型）**学 $V(s)$；
4. **损失**：clip 目标 $L^{CLIP} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t)]$，$r_t = \pi_\theta(a_t|s_t)/\pi_{old}$ + value loss + entropy bonus；
5. **计算资源**：四模型同时在显存（actor 训练态 + actor 推理态可共享、ref 冻结、RM 冻结、critic 训练）——7B 级 PPO 通常 8-16 卡 A100（actor+critic 训练 + LoRA 降配可到 4 卡）；rollout 阶段用 vLLM 加速再权重同步（开源框架 OpenRLM/TRL/veRL 的核心工程点）。

**代码**：`solutions/python/src/answers/bytedance/rlhf.py` —— `ppo_clip_loss`（clip 两侧边界的手算测试）、`k3_kl_estimator`（与解析 KL 对拍的无偏估计）、`ppo_lm_reward`（KL 惩罚合成 token 奖励）。

## Q3. DPO 的训练流程；与 PPO 的区别；对比实验（抖音电商 训练优化 3、豆包 基础知识 3）

**DPO 流程**：无需 RM 与在线采样——离线偏好对 $(x, y_c, y_r)$ 上直接微调：$L_{DPO} = -\mathbb{E}[\log\sigma(\beta((\log\pi_\theta(y_c|x) - \log\pi_\theta(y_r|x)) - (\log\pi_{ref}(y_c|x) - \log\pi_{ref}(y_r|x))))]$；超参 β 控制偏离 ref 的强度（大 = 保守小 = 激进）。部署同 SFT 模型。

| 维度 | PPO | DPO |
|---|---|---|
| 数据 | 在线 rollout + RM 打分 | 离线静态偏好对 |
| 模型 | actor+critic+RM+ref 四件套 | policy + ref 两件套 |
| 优化对象 | max E[r] - βKL（RL 目标） | BT 模型下同一最优解的闭式重参数化 |
| 稳定性 | 超参敏感（KL 系数/裁剪/学习率）、reward hacking 风险 | 简单稳定，但易过拟合偏好集、有分布偏移（学不到新策略外的模式） |
| 成本 | 高（采样+四模型） | 低（一个 SFT 级训练） |
| 上限 | 在线探索，上限高 | 受限于偏好数据覆盖 |

**对比实验设计**（字节问"是否做过对比"）：同底座同数据源（若可比：PPO 用 RM 训练集同源标注转偏好对）→ 在留出集上评：胜率（GPT-4 as judge + 人工抽检双盲）、奖励模型分布外基准（避免 judge 偏置）、KL 散度（漂移监控）、推理长度/重复率等行为指标。

**代码**：`rlhf.py::dpo_loss` 返回 loss/隐式奖励/准确率（手算样例测试：β=0.1、Δπ=2 时 logits=0.2 的解析校验；ref 整体平移不变性测试）。

## Q4. 如何评估微调后模型的性能提升（抖音电商 训练优化 4）

分层评估体系（避免只看一两个分数）：

1. **自动基准**：任务域 benchmark（MMLU/CMMLU 通用，GSM8K/MATH 数学，HumanEval/MBPP 代码，C-Eval 中文）+ 业务域自建评测集（线下来源、防污染）；
2. **模型评审（LLM-as-judge）**：成对胜率/平局率（judge 用更强模型，注意 position bias——AB 位置交换各评一次）；rubric 打分（维度：正确性/完整性/格式/安全）；
3. **人工评估**：金标准。抽检双盲 + 多标注者一致性（Cohen's κ）；Likert 或成对比较；
4. **对齐专项**：AlpacaEval 2（长度控制胜率）、MT-Bench 多轮；有害率/拒答率（安全评测集）；
5. **行为与回归指标**：通用能力回退（微调后 MMLU 不掉）、重复率、长度分布、语言混出、拒答过度；
6. **线上 A/B**：最终裁决——留存/采纳率/人工接管率/延迟成本一起看。
方法学要点：**对照组要同训练配方**（同数据量 LoRA vs 全参才是干净对比）、多 seed 报均值方差、评测集与训练数据严格隔离。

## Q5. PPO 效果优于 DPO 的原因；如何理解多轮 DPO 的提升（抖音电商 训练优化 5）

**PPO 占优的场景与原因**：

1. **在线数据分布**：PPO 优化的是当前策略采样的分布（on-policy），DPO 训练分布冻结在偏好集——策略改进后 DPO 数据变 off-policy，迭代失配；偏好数据覆盖不足时 DPO 天花板低；
2. **细粒度信用分配**：PPO 用逐 token 的 KL+GAE 在序列内部定位问题，DPO 只有序列级 logprob 差，长回答信号稀释；
3. **reward hacking 可控**：PPO 有显式 RM 可以迭代重训/清洗；DPO 的"奖励"是隐式的，错了只能换数据。

**多轮 DPO（iterative DPO）的提升机理**：本质是把 DPO "在线化"——每轮用当前策略采样多回答 → RM/ judge 标偏好 → DPO 一轮 → 更新采样分布（Self-Rewarding / SPIN 同族）。提升来自：① 数据分布跟随策略演进（缓解失配）；② 负样本质量高（自己生成的 rejected 而非离线静态对，对比更"知情"）；③ 累积的 ref 重锚定控制漂移。所以"多轮 DPO ≈ 穷人版在线 RLHF"——答出这个等价视角是加分点。风险：误差逐轮累积、熵坍缩（回答多样性掉）需要每轮控 β 与数据新鲜度。

## Q6. GRPO 的原理（抖音电商 训练优化 6）

DeepSeekMath 提出的 PPO 简化（DeepSeek-R1 的 RL 算法）：

1. **组内相对奖励**：同一 prompt 采样 **G 个回答**为一组，advantage = 组内标准化 $A_i = (r_i - \text{mean}(r_{1..G}))/\text{std}(r_{1..G})$——组均值当基线，**去掉 critic/value model**（PPO 最重的组件）；
2. **损失**：与 PPO 相同的 clip 目标作用于组内相对 advantage（+ KL 正则到 ref，早期版本直接加在 loss，R1 用无偏估计）；
3. **为什么合理**：同 prompt 的 G 个回答共享同一个"难度基线"，组内相对比较即控制了 prompt 间奖励分布差异（reward 的 prompt 方差被消掉）；价值函数学的不就是这个基线，直接用组均值估计——蒙特卡洛版的优势；
4. **适用**：outcome 可精确判定/廉价判定的任务（数学答案对错、代码单测通过）最佳；G 取 8-64；**Dr.GRPO** 变体发现除以 std 会引入长度偏置（短回答被放大），去掉 std 只减均值；
5. 对比：GRPO vs PPO 省一个 critic 训练（显存/工程都轻），代价是采样 G 倍（rollout 便宜，通常划算）。

**代码**：`rlhf.py::grpo_advantages`（含 Dr.GRPO 开关、零方差组归零处理；测试：[1,2,3,4] 两两成组 → ±1 的手算校验）。

## Q7. 为什么有了 SFT 之后还需要 RLHF（抖音电商 训练优化 7、豆包 基础知识 4）

1. **监督信号的本质差异**：SFT 的 loss 是"逐 token 逼近示范答案"，只学**正例**；人类偏好是**相对的**（好于什么、差于什么），RLHF 的 reward 直接优化"人类更偏好"这一目标——目标对齐而非模仿；
2. **示范的次优性**：SFT 只能模仿到示范者水平（且示范内部不自洽，多版本答案互相拉扯均匀化）；RL 在奖励引导下可**超越**示范（AlphaGo/RLHF 数学模型超过标注员水平、R1 展示的 self-emergent 能力）；
3. **信用分配**：序列级偏好（整体好坏）无法用逐 token 交叉熵表达——哪个 token 导致了差评？RL（advantage/KL 的逐 token 分解）给出回答；
4. **行为约束**：RLHF 的 KL-to-ref + reward 惩罚是注入安全/风格偏好的柔性手段（拒绝有害请求、简洁性），SFT 硬塞负例容易学成"模式匹配关键词"；
5. **分布偏移修正**：SFT 学的是"人写的分布"，部署后模型自回归生成的分布与训练分布不同（exposure bias），RLHF 直接在模型自己的生成上优化，修正这个 gap。
一句话总结：SFT 教"怎么说话"，RLHF 教"哪种说法更好"。

## Q8. 场景题：如何设计方案识别抖音平台上的刷赞行为（抖音 场景题 6）

按"定义 → 信号 → 模型 → 对抗 → 评估"的框架答：

1. **问题定义**：识别非真实互动（买赞/互赞群/机器刷量），输出 = 账号/作品/互动三元组的可疑分。难点：无干净标注（正样本要人工确认）、对抗性强（刷方持续演化）、量级（十亿级互动/天）。
2. **信号体系**：
   - *互动侧*：点赞时间序列的突发性（短窗爆发）、设备/IP 聚簇（同设备农场）、行为节奏异常（点击间隔方差过低的"机器指纹"）；
   - *账号侧*：新注册-无社交关系-昵称头像模板化、活跃时间片集中、关注/粉丝比异常；
   - *图结构*：点赞二部图上的稠密子图（一组账号集中刷一批作品 = quasi-clique 挖掘）、异构图上刷量团伙的社区发现；
   - *内容侧*：被刷作品与作者历史数据的突增偏离（涨幅超出粉丝结构支撑）。
3. **模型**：无监督打底（图社区挖掘 + 异常检测 Isolation Forest/密度）产出候选团伙 → 弱监督（规则+人工抽检确认的种子）训练 GBDT/异构图 GNN（账号-设备-IP-作品多关系）打分 → 主动学习闭环（高不确定性样本送人审回流）。图算法细节引《12 算法题》图论节。
4. **对抗鲁棒**：特征做时间衰减与近窗化（防刷方用长期养号绕过）、模型周级重训、多模型投票防单点被逆向、阈值分级处置（限流/封号分级）。
5. **评估**：精确率优先（误伤真实用户代价大），人工抽检校准；离线 PR 曲线 + 在线处置后"刷量供给下降"的 A/B 反事实；监控团伙演化速度（模型迭代 vs 绕过周期）。

## Q9. 项目经历与开放经验题的答法（各文件"项目经历"题 + AI 部署实践 + 消融设计）

字节 13 个文件里每个都有"请介绍你的项目/实习经历"，另有两道开放经验题：**"是否有 AI 模型部署实践经验"**（一面-2 11）、**"项目创新点与代码实现细节、消融实验设计"**（抖音电商 15）、**"Efficient Transformer 研究工作"**（实习-一二三面 1/10/16）、**"上线推理服务做过哪些优化"**（实习-一面-1 10）、**"Agent 生成 CUDA kernel 的方案"**（实习-一面-1 11，同《03》Q12）、**AML 的 MPS/训练内存管理/GPU 利用率**（5/9）。统一答法（STAR 变体，AI Infra 岗特化）：

1. **一句话定位**：项目解决什么瓶颈（"decode 延迟 3 秒 → 800ms"），先给量化结论抓注意力；
2. **背景与约束**：模型/硬件/SLO/成本约束——体现工程判断；
3. **方案与取舍**：候选方案对比（为什么选 A 不选 B），落在具体技术点（量化选型/并行策略/kernel 结构），引用本库语言体系（如"PagedAttention+prefix cache 组合"，见《20》）；关键实现细节准备 2-3 个深挖点（面试官必追问）：一行公式或一段代码级别；
4. **量化结果**：吞吐/延迟/显存/MFU 数字 + 测量方法（避免"快了很多"这种表述）；
5. **反思与泛化**：什么条件下方案失效、如果重做什么。
- **部署经验题**素材链：量化（W8A8/FP8）+ kernel 融合 + CUDA Graph + continuous batching + prefix cache + 监控（QPS/P99/缓存命中率）——对应《20》《21》各题。
- **消融实验设计**：控制变量（单因素轮换）、基线选强（naive 与 SOTA 双基线）、多 seed 显著性、指标成套（主指标+资源指标+回归指标）、按模块拆（结构/数据/训练配方三层各自消融）。

## Q10. 编程题总索引（13 个文件的全部编程题）

字节编程题与综合题库高度重合，全部已有工程化实现与测试，此处只给映射（题目→已有篇目/代码）：

| 题目 | 来源 | 解答与代码 |
|---|---|---|
| 棋盘第一行到最后一行连通性 | 实习-一二三面 一面 9 | 《12 算法题》树与图节（BFS/DFS 四连通扩散），`algorithms/graphs.py` |
| 给定 Graph 类实现拓扑排序 | 实习-一二三面 二面 15 | 《12》Kahn 入度 BFS + DFS 三色双实现，`graphs.py` |
| 下一个排列（LC 31） | 实习-一二三面 三面 21 | 《12》DP 与数学节（从右找升序对→换稍大→反转后缀），`math_bits.py` |
| O(1) 最小栈 | 实习-一二三面 三面 20 | 《12》LRU/栈节（辅助栈同步存当前最小），`min_stack.py` |
| 整数平方根（LC 69）+ 优化 | 实习-一面-4 8 | 《12》（二分 + 牛顿迭代双版），`math_bits.py` |
| 环上 10 节点走 n 步回原点 | 校招-1 8 | 《12》（DP `dp[s][v]=dp[s-1][v±1]`，10 节点 2 步=2；矩阵快速幂 O(log S)），`dp.py` |
| 两个有序数组中位数（LC 4） | 校招-一面 4 | 《12》（二分短数组分割线 O(log min(m,n))），`dp.py` |
| 手写 CUDA GEMM + 指标 | 校招-一面 2 | 《21》Q9 /《02 CUDA 编程题》Q3，`solutions/cuda/src/gemm.cu`（3090 实测 GFLOPS） |
| 手写 MLA | 校招-一面 3 | 《21》Q10 /《04》Q3，`attention/mla.py`（naive/absorb 等价测试） |
| 链表反转 | AML 一面 6 | 《12》链表节（三指针迭代），`linked_list.py` |
| LRU Cache | AML 二面 10 | 《10 C++ 编程题》Q1（哈希+双向链表 O(1)，C++ 版）+《12》（Python 双实现） |
| 非递归中序遍历 | 抖音 8 | 《12》树与图节（显式栈 `_push_left`），`trees.py` |
| 合并 K 个升序链表（LC 23） | 豆包 7 | 《12》链表节（小根堆 O(N log K)），`linked_list.py` |
| 图的最短连通路径 | 一面-2 13 | 《21》Q13 /《12》（MST Kruskal+Prim 双实现），`graphs.py` |
| "编程题考察"（未给具体题） | 一面-1 5、抖音电商 16 | 开放——字节现场常考：手写 attention/LayerNorm（《04》Q4-Q5、`mha.py`）、softmax（`online_softmax.py`）、链表/DP 类（上表） |

验证命令：

```bash
cd solutions/python && pip install -e . && pytest tests -q          # 算法/attention/torch/triton/bytedance
cd solutions/cuda  && cmake -B build && cmake --build build && ctest --test-dir build   # GEMM 等 kernel
```
