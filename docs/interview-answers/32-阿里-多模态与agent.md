---
title: "阿里题解 · 多模态与 Agent"
description: "Qwen-VL 训练流程、多模态对齐、RAG、Agent 记忆/编排/可观测性、MCP 评测、Vibe Coding 与 Claude Code 工程实践题解"
pubDate: 2026-10-05
topic: "公司题解-阿里巴巴"
order: 32
tags: ["多模态", "Agent", "RAG", "面试解答"]
draft: false
---

覆盖题库：`阿里巴巴-AI-Infra-一面-1`（Q4/Q6/Q7）、`阿里巴巴-AI-Infra-一面-2`（Q1-Q6）、`阿里巴巴-AI-Infra-实习-2`（Q2/Q3）、`阿里巴巴-云-AI-Infra-二面`（Q7）、`阿里巴巴-云-AI-Infra-实习-二面-1`（Q8）、`阿里巴巴-云-AI-Infra-实习-二面-2`（Q6）、`阿里巴巴-控股集团-AI-Infra-实习-一面`（Q1-Q4/Q6）、`阿里巴巴-淘天-AI-Infra-一面-1`（Q1-Q3/Q6/Q7）、`阿里巴巴-淘天-AI-Infra-一面-2`（Q8 编程题见《训练与算法》Q25）、`阿里巴巴-AI-Infra-1`（Q2）。

阿里 2025-2026 面试的明显趋势：**Agent 工程题占比高**（记忆/编排/MCP/可观测/Vibe Coding），且与多模态（Qwen 系）绑定。本篇无新增代码工程（Agent 题以方案设计为主），编程题引用已有篇目。

## 一、多模态模型

### Q1. Qwen-VL 的完整训练流程（一面-2 Q1）★

以 Qwen-VL（2023）三阶段为基准答，再补 Qwen2/2.5-VL 演进：

1. **阶段一：大规模图文对齐预训练**。数据：网页图文对（caption 类，数十亿级）。**冻结 LLM**，只训视觉编码器（ViT-bigG，OpenCLIP 初始化）+ 单层 cross-attention adapter（视觉特征注入 LLM 的可学习模块）；分辨率从 224 渐进提升到 448；任务 = 图像描述（captioning）。
2. **阶段二：多任务训练（全参）**。高质量 VQA/OCR/ grounding/多模态对话数据（千万级），**LLM 解冻全参数训练**，分辨率 448，输入输出格式统一（含 grounding 坐标框的文本表示）。视觉 token 化：ViT 输出 1024 个 patch 特征经 cross-attention 压到 256 个视觉 token（Qwen-VL v1 保持 cross-attn adapter；Qwen2-VL 起改用 MLP projector + 2×2 相邻 patch 合并）。
3. **阶段三：SFT 指令微调**。与 LLM 的 chat SFT 相同配方：指令-回答对（含纯文本对话保语言能力不退化），冻结 ViT，对齐对话格式（ChatML）与人类偏好。

**演进要点（追问必加分）**：Qwen2-VL 用 **Naive Dynamic Resolution**（任意分辨率直接 patch 化，原生处理高分辨率/变长宽比）+ **M-RoPE**（位置编码按 时间/高/宽 三维分解分配 id，见 Q4）+ 三个阶段简化为"大规模预训练 → SFT/GRPO"；Qwen2.5-VL ViT 加窗口注意力（高分辨率省算力）、绝对时间对齐的位置编码（视频时间戳理解）、大量 agent 定位/GUI 操作数据。答"完整流程"时强调三要素：**训练阶段划分（冻结策略）、数据课程（对齐→任务→指令）、分辨率与 token 化方案**。

### Q2. 视觉特征传递给 LLM 的常见方法（一面-2 Q2）

| 方案 | 代表 | 机制 | 优缺点 |
|---|---|---|---|
| Linear/MLP projector | LLaVA、Qwen2-VL | ViT patch 特征 → MLP 投影 → 直接当 LLM 输入 embedding | 简单、端到端、当前主流；token 数多（高分辨率开销） |
| Q-Former | BLIP-2 | 32/64 个可学习 query 经 cross-attn 压缩视觉信息 → 投影进 LLM | token 数固定（省）；压缩有信息损失，训练复杂（两阶段） |
| Cross-attention adapter | Flamingo、Qwen-VL(v1) | 视觉特征作 K/V 注入 LLM 内部插入的 gated cross-attn 层 | 不占输入序列长度；改 LLM 结构、门控初始化讲究 |
| Patch 直连 | Fuyu | 图像 patch 不经 ViT 直接线性投影进 LLM | 架构最简；无独立视觉预训练，性能上限低 |
| 交叉熵对齐（CLIP 式共享空间） | CLIP/检索系 | 图文映射到同一 embedding 空间 | 用于检索/zero-shot，不适合生成 |

答题主线："接口放哪（输入序列 vs 中间层）+ 压缩多少（token 数）+ 训练难度"，指出 LLaVA 系 MLP+token 合并是工程收敛点，Qwen2-VL 的 2×2 merge 兼顾分辨率与序列长度。

### Q3. ViT 通常采用什么方式预训练？（一面-2 Q3）

1. **有监督分类**（原始 ViT）：JFT-300M 大规模标注分类迁移；
2. **对比学习**：CLIP/DINOv2——图文对比（CLIP）/自蒸馏+多裁剪（DINO 系），得到通用视觉表征（多模态模型最常用的初始化来源）；
3. **掩码重建（MIM）**：MAE（mask 75% patch 重建像素，ViT patch 化天然适配）、BEiT（离散 token 化再 mask 预测）、SimMIM；
4. **自蒸馏**：DINO/DINOv2（teacher-student 中心/锐化，无需标注，表征质量高）。

选型回答：多模态 LLM 的视觉塔常用 CLIP 系初始化（图文对齐起点好）；检测/分割下游用 DINOv2/MAE 表征更强。追问 patch 化为何适合 MIM：patch = 天然的"视觉 token"，mask+重建与 BERT 同构。

### Q4. 不同尺寸图片/视频输入的位置编码如何设计？（一面-2 Q4）

问题本质：位置编码需要**平移不变/尺度泛化 + 可变长**。方案阶梯：

1. **插值**：训练 224 网格，推理高分辨率时对 pos embedding 双三次插值（ViT 原生做法）——简单但破坏尺度过程中细粒度错位；
2. **分解式（factorized）**：2D 位置 = 行编码 + 列编码相加/拼接，天然支持变 H×W；视频再加时间维 → (t,h,w) 三分解；
3. **Naive Dynamic Resolution + M-RoPE**（Qwen2-VL）：任意分辨率直接 patch 化（patch 数变长），RoPE 的位置 id 按 (t,h,w) 三维分别分配——同行的 patch 共享 h id，实现"坐标即位置"，支持视频帧数/长宽比任意组合；
4. **窗口/块内局部编码**：高分辨率图切块（如 NaVi/ S2 包装器），块间用序列位置、块内用 2D 位置；
5. **可学习 object queries**（DETR 系）：输出位置无关，靠内容匹配（检测路线，非生成）。

视频特有：时间维采样不均匀（fps 对齐）时用"绝对时间-RoPE"（Qwen2.5-VL，按真实时间戳分配 id）。

### Q5. ResNet / Transformer 中残差连接的作用（一面-2 Q5）

1. **优化角度**：$y = x + F(x)$，梯度 $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}(1 + \frac{\partial F}{\partial x})$——恒等项让梯度无损直通，深网络（百层）可训；学习目标变为残差 $F(x) = H(x) - x$（期望输出与输入的差），初始化在恒等附近，损失面更平滑（He 的初始化分析、Li 的损失面可视化）。
2. **表达角度**：集成视角（Veit et al.：ResNet = 不同深度子网络的隐式 ensemble）；信息高速公路（浅层特征直通深层）。
3. **Transformer 特有**：Post-LN（原始）深了不稳 → **Pre-LN**（子层输入先 LN，残差主干无归一化，梯度更稳）成为大模型标配；残差流维度不变（d_model 常数）是"信息总线"，子层只做增量读写。
4. **反例/代价**：残差使每层输出方差累积（需 LN 控制）；消融 Transformer 残差性能崩塌远快于去 attention。

### Q6. RAG 检索模块的实现方案；多模态 RAG 思路（一面-1 Q4、一面-2 Q6）

**文本 RAG 检索层**（按管线讲）：

1. **索引**：文档 → 分块（chunk，固定 512 重叠 / 语义切分 / 父子块）→ embedding（bge/e5 类 bi-encoder）→ ANN 向量库（FAISS/HNSW/Milvus）；可选稀疏侧（BM25/ES）与元数据（时间/来源/权限）过滤；
2. **查询侧**：改写扩展（多查询/HyDE 假设文档/查询分解）、意图路由（该走检索还是工具）；
3. **召回融合**：混合检索（稠密+稀疏，RRF 倒数排名融合）→ **rerank**（cross-encoder 对 top-50 精排）→ 上下文组装（top-k、去重、按引用顺序）；
4. **评估**：召回率@k、MRR 离线评估集 + 端到端 faithfulness（引用正确率）。

**多模态 RAG**：三种路线——①统一 embedding 空间（CLIP 系图文同空间，文本 query 直接检索图像/视频帧）；②caption 化（图像先 VLM 转 caption 进文本索引，检索文本化、生成时取原图）；③多模态生成（检索到的图+文一起进多模态 LLM 上下文，如表格截图直接喂 Qwen-VL）。工程要点：视频按关键帧/场景切分建索引、图表类文档用 VLM 做 layout-aware 解析（表格转结构化文本）、embedding 模型与 reranker 的模态匹配。

## 二、Agent 架构与工程

### Q7. 设计一个 Agent 系统的整体架构（一面-1 Q6）

分层回答（自上而下）：

1. **交互层**：流式输出、多轮会话管理、中断/澄清、人工确认门（高危操作）；
2. **编排层（核心）**：planner（任务分解/重规划）+ executor 循环（LLM 决策 → 工具调用 → 观察反馈，ReAct 式）；状态机或图执行（LangGraph 类），支持分支/回退/checkpoint 恢复；
3. **记忆层**：短期（当前上下文窗口 + 滚动摘要）、长期（向量库 episodic 记忆 + 结构化事实）、工作记忆（当前任务 scratchpad）——详见 Q8；
4. **工具层**：工具注册与 schema（function calling / MCP）、沙箱执行（容器/超时/权限分级）、结果截断与错误规范化；
5. **模型层**：模型路由（简单任务小模型、推理任务大模型）、fallback、成本/延迟预算控制；
6. **可观测与评估层**：全链路 trace、每步评估、护栏（输出校验、越权检测）——详见 Q14。

多 agent 拓扑（追问）：manager-worker（主 agent 派发子任务）、流水线（固定工序）、辩论/评审（生成-批评对）。设计权衡：自主性越高（agent）越灵活但不可控，确定性越高（workflow）越可靠但不灵活——按任务模糊度选择（衔接 Q10）。

### Q8. Agent 的记忆模块如何设计？有哪些管理策略？（淘天-一面-1 Q1、实习-2 Q2）

**分类学**（认知科学映射到工程）：

| 类型 | 内容 | 工程实现 |
|---|---|---|
| 工作记忆 | 当前任务的中间状态/scratchpad | 上下文窗口内（结构化 JSON 状态对象） |
| 情景记忆（episodic） | 历史对话/执行轨迹 | 向量库 embedding 检索（"以前发生过什么"） |
| 语义记忆（semantic） | 事实/用户偏好/领域知识 | 结构化 KV/知识图谱（"用户常住上海"） |
| 程序记忆（procedural） | 怎么做某类事的 SOP | Skills（Q11）——prompt 模板+脚本 |

**管理策略**：①**写入**：显式工具（`memory.write` 由 LLM 判断值得记）vs 自动抽取（会话结束离线摘要）；②**读取**：按当前任务 embedding 相似度 + 时间衰减 + 重要性打分的融合排序（Generative Agents 的 recency × importance × relevance）；③**遗忘/压缩**：滚动摘要（旧对话压成 summary 释放窗口）、分层合并（session → daily → long-term）、矛盾消解（新事实覆盖旧事实带版本）；④**容量控制**：上下文预算分配（系统提示/记忆/工具结果各占多少 token）。数据结构衔接 Q15（对话存储）。

### Q9. Skills 读取超长 SOP 的场景有哪些优化技巧？（实习-2 Q3）

SOP（标准作业程序）文档数千到数万字，超出或挤爆上下文。优化阶梯：

1. **结构化改写（最重要）**：把自然语言 SOP 离线编译成**决策表/流程图/前置条件-动作对**——SOP 的本质是条件分支，LLM 读表格比读散文准且省 token；
2. **分层渐进加载（progressive disclosure）**：SKILL.md 只放目录与触发条件，正文按需加载当前章节（Claude Skills 的设计哲学：metadata → 相关 section → 全文）；
3. **检索式读取**：SOP 分块建索引，按当前任务检索 top-k 相关条款而非全文注入；
4. **缓存与编译**：高频 SOP 预编译成 checklist 或伪代码，甚至直接编译成可执行 workflow/脚本（Agent 只填参数），从"读文档"降级为"调用"；
5. **压缩**：关键参数表/数字表格保留原文，叙述性内容摘要；版本管理（SOP 更新触发重编译）。

### Q10. Workflow、Agent、Skill 三者的区别与定位（淘天-一面-1 Q2）

- **Workflow**：**人预先编排**的执行图（DAG/状态机），节点是工具调用或 LLM 调用，路径确定——确定性、可测、可控，适合已知的稳定流程（报销审批流、固定数据管线）；
- **Agent**：**LLM 在循环中自主决策**下一步（感知-思考-行动），动态规划路径——灵活、能处理未预期情况，但不确定性高、成本大、需要护栏，适合开放任务（debug、研究、编码）；
- **Skill**：**封装的"专长"模块**——某类任务的领域知识 + 操作流程 + 资源（prompt 模板、脚本、参考文档），按需加载给 Agent 用。它不决定"做什么"（那是 Agent），只决定"这类事怎么做最好"（类似函数库/插件，但以模型可读的方式组织）。

定位关系：**Workflow 是编排层、Agent 是决策层、Skill 是能力层**。实践中三者组合：确定性骨架用 workflow，模糊子任务交给 agent，agent 调用 skill 获得领域专长。判断标准答题法："如果任务路径能画出来 → workflow；需要模型自己找路 → agent；某类子任务反复出现且有最佳实践 → 沉淀成 skill"。

### Q11. Prompt 编写的关键因素与质量评估（淘天-一面-1 Q3）

**编写要素**（按检查清单讲）：①任务与角色定义（who/what/约束边界）；②上下文与示例（few-shot 覆盖边界 case，包括反例）；③输出格式约束（结构化 schema/JSON，配合 constrained decoding）；④推理引导（思维链、"先分析再回答"）；⑤变量与模板化（占位符、防注入分隔）；⑥边界与禁止事项（不要编造、超出范围时拒答）；⑦长度预算（每段指令有注意力权重，过长稀释）。

**评估方法**：①固定 eval set（覆盖典型/边界/对抗 case）+ 自动评分（exact match / 单测通过率 / rubric 的 LLM-as-judge）；②A/B 对照（同 case 集两版 prompt，按维度对比：正确性/格式合规/简洁）；③敏感性测试（同义改写 prompt 看输出稳定性，检验是"真理解"还是过拟合措辞）；④成本指标（token/延迟）；⑤**回归管理**：prompt 版本化，每次修改跑全量回归防"改好一处坏三处"。

### Q12. Agent 系统中各 Agent 的分工与输入输出（淘天-一面-1 Q6）

结合自己项目答，给一个通用模板（数据开发 Agent 系统示例）：

| Agent | 职责 | 输入 | 输出 |
|---|---|---|---|
| Planner | 需求理解、任务分解、分派 | 用户需求 + 资源清单 | 任务 DAG + 各任务规格说明 |
| Coder | 代码生成 | 任务规格 + 代码库上下文 | 代码 diff + 单测 |
| Executor/Runner | 沙箱执行 | 代码 + 数据 | 运行日志/测试结果（成功或错误栈） |
| Debugger | 失败归因、修复建议 | 错误日志 + 代码 | 根因分析 + 补丁 |
| Reviewer | 质量与安全审查 | diff + 规范文档 | 审查意见（阻塞/建议） |
| Reporter | 结果汇总 | 各 agent 产物 | 给用户的报告 |

关键设计点：接口契约（agent 间传递结构化对象而非自由文本，防止信息损失）、失败传播（下游拿到上游失败标记而非脏数据）、token 预算（每个 agent 只拿它需要的上下文切片，planner 负责摘要不透传全量日志）。

### Q13. 开发的 MCP 如何评测效果、如何迭代优化？（控股-实习-一面 Q1）

MCP（Model Context Protocol）server 的评测分四层：

1. **协议/工具层**：schema 质量——工具描述、参数描述是否让 LLM 正确选用（用强模型做 tool-selection 准确率测试：给任务描述看模型能否选对工具+填对参数）；错误处理（非法参数返回可恢复错误信息而非崩溃）；
2. **功能层**：每个工具的单元测试（正常/边界/异常输入）；资源与 prompt 模板正确性；
3. **端到端任务层**：构造任务集（"用这个 MCP 完成 X"），指标：任务成功率、平均步数、token 消耗、重试次数、延迟；对照组（无 MCP 用 raw API 实现）证明增量价值；
4. **回归与持续**：版本更新跑固定 eval set（防 schema 变更引入回归）、线上采样 trace 审计。

**迭代循环**：bad case 归因 → 四类修复（工具描述改写 / 参数 schema 简化或拆分工具 / 实现修 bug / 增加 few-shot 示例资源）→ 重跑 eval → A/B。高频经验：模型选错工具多半是**描述问题**不是模型问题——描述要写清"什么时候用我/什么时候别用我"，参数名自解释。

### Q14. Agent 系统的链路追踪与可观测性（控股-实习-一面 Q2）

1. **Trace 模型**：借鉴 OpenTelemetry——一次 agent run = trace，LLM 调用/工具调用/子 agent = span（树结构），span 携带：输入输出、模型/参数、token 用量、成本、延迟、缓存命中；全链路 trace_id 贯穿（含 sub-agent 与异步任务）；标准落地：OTel gen-ai semantic conventions、Langfuse/LangSmith 等平台；
2. **关键指标**：任务成功率、每步延迟分解（LLM/工具/调度）、token 与成本按 agent/阶段分解、工具成功率与错误分类、循环/重试检测（卡死预警）、上下文长度增长曲线（爆窗前预警）；
3. **日志与状态**：结构化日志（JSON）+ 会话状态快照（配合 Q15 的持久化），支持回放调试（从 checkpoint 重放某步）；
4. **评估闭环**：线上采样 trace → 人工/LLM judge 打分 → bad case 进 eval set（可观测不只看系统健康，还喂训练与 prompt 迭代）；
5. **告警**：成本突增（死循环）、错误率、P95 延迟、越权操作尝试。

### Q15. Agent 对话记录的保存方式与数据结构（控股-实习-一面 Q3）

**逻辑结构**：消息数组（append-only log）——`{role: user/assistant/tool, content, tool_calls, tool_results, timestamp, tokens, metadata}`；会话-消息两级（session_id → messages[]）；**分支对话用树**（同一节点生成多个候选回复 → 对话树，节点存父指针），决策类 agent 还要存"思考-行动"对（ReAct 轨迹）。

**存储选型**：热路径 Redis（当前活跃会话，TTL）；主存储 PostgreSQL JSONB（结构可查询：谁/何时/用了什么工具）或 MongoDB（schema 灵活）；embedding 旁路进向量库支持"历史记忆检索"（Q8）；冷数据归档对象存储。**内存内结构**：list/deque 维护窗口 + 定期摘要压缩（长对话超窗时旧消息摘要替换原文，保留近 K 轮原文）。

追问点：为什么 append-only 不可变——审计/回放/训练数据抽取都要完整历史；上下文重建 = 按 session 取消息 + 应用窗口/摘要策略，而非存储层做截断。

### Q16. 主流 Agent 开发框架及特点（控股-实习-一面 Q4）

| 框架 | 定位 | 特点 |
|---|---|---|
| LangGraph | 图编排 | 状态机/图结构（节点+边+条件路由），checkpoint 与 human-in-loop，生产级控制流 |
| LangChain | 工具箱 | 组件最全（链/工具/memory），历史包袱重，偏原型 |
| LlamaIndex | 数据为中心 | 索引/检索强（RAG 首 选），agent 为辅 |
| AutoGen（微软） | 多 agent 对话 | agent 间消息协作模型，研究多 agent 拓扑方便 |
| CrewAI | 角色分工 | 简单的 role-based 多 agent（crew/task 抽象），上手快 |
| OpenAI Agents SDK | 轻量 | handoff/工具/护栏三件套，SDK 轻、与 OpenAI 生态绑定 |
| Claude Agent SDK / Claude Code | Agentic coding | 终端 agent 事实标准：工具循环+CLAUDE.md 记忆+Skills+hooks（详见 Q18） |
| smolagents（HF） | Code action | 用**写代码**代替工具调用 JSON 作为行动空间，减少循环轮数 |
| Dify/Coze | 低代码平台 | 可视化编排+知识库，业务侧快速搭建 |

选型话术：原型用 LangChain/LlamaIndex；生产控制流用 LangGraph 或自研（本质是"LLM 决策循环 + 状态机"，框架核心价值只剩可观测与生态）；coding agent 直接用 Claude Code/Agent SDK 深度定制。

## 三、AI 辅助开发工程（Claude Code 相关）

### Q17. 用 Agent 自动生成 CUDA kernel 的实践方案（AI-Infra-1 Q2）

方法论闭环见《GEMM 与算子优化》Q12（候选生成→验证闭环→性能闭环→护栏）。补充 2025-2026 的实践演化与细节：

1. **技术选型演进**：Triton 优先（编译反馈快、autotune 生态）→ 复杂 kernel 上 KernelBench 类 benchmark + reasoning 模型（先生成算法大纲再写代码）；CuTe/CUTLASS 类用 few-shot 喂模板（DSL 类型系统对 LM 友好，layout 代数可静态检查）；
2. **验证闭环细节**：随机 shape + 边界 shape（极端长宽、0/1 维、非对齐）+ 与 PyTorch 参考对拍（相对误差分级：fp32 1e-5 / fp16 1e-3 / int 精确）；compute-sanitizer 查 UB；性能回归用历史基线库（防"这次生成的快了但下次不一样"）；
3. **性能闭环**：ncu 提取 SOL/带宽/occupancy 文本喂回模型迭代（profile-text-driven 修复比盲改有效得多）；常见可修复问题：非合并访存、bank conflict、寄存器溢出——LLM 能看懂 ncu 指标但需要把指标与代码行关联（-lineinfo）；
4. **结论分层**：elementwise/reduction/简单融合已可靠超越新手手写；GEMM/Attention 级仍需人类模板+领域知识；生成后人工 review 是必要环节。

### Q18. 如何看待用 AI Agent 辅助编码、Review 与工程梳理？（云-实习-二面-1 Q8）★

以 Claude Code 为代表的 agentic coding 工具的实践认知（面试官考察你是否真的深度用过）：

**编码侧**：①**上下文工程**是效率的核心变量——CLAUDE.md 项目规则（代码规范/常用命令/禁区）、明确任务边界与验收标准（"改 X 并让测试 Y 通过"优于"优化一下"）；②**小步迭代**——让 agent 完成可验证的小单元（写→跑测试→贴错误→修），一轮大改容易失控；③**让 agent 自己读代码**（grep/read 工具）优于人贴代码片段——上下文污染是主要失败源。

**Review 侧**：①LLM review 的强项：规范性/边界 case/并发问题/安全（注入、越权）的系统性扫描、大 diff 无人愿读时的初筛；②弱项：架构合理性、跨模块影响的因果链——需要人给上下文（这次改动的意图）；③实践：PR 模板让 agent 按 checklist review + 人复核 blocker；生成测试（单测覆盖率提升比生成代码本身收益更稳）。

**工程梳理侧**：代码库问答（"这个函数在哪被调用"）、文档生成/更新、依赖梳理、迁移方案初稿（API 升级 diff）——低风险高价值，适合放手让 agent 做。

**边界认知**（加分项）：agent 生成代码的维护成本问题（无人理解的代码是负债）、安全红线（密钥/生产数据不进上下文）、"信任但验证"——测试与 review 是不可跳过的守门，AI 提效的是过程不是责任转移。

### Q19. Vibe Coding 实操要点（淘天-一面-1 Q7）

现场用 agentic 工具（Claude Code/Cursor）编码的考察点是**人机协作效率**：

1. **先想清楚再开口**：花 30 秒向 AI 说清需求/约束/验收（输入输出示例），比边写边改快数倍——vibe coding 不是"不动脑地 vibe"，快速迭代的前提是 spec 明确；
2. **任务拆解**：复杂任务拆成可独立验证的小步（数据结构→核心函数→边界），每步跑测试；
3. **错误反馈循环**：报错直接贴给 AI（含栈），让它定位修复；连续两次修不对就要求它"先写复现测试再修"；
4. **守门**：要求 AI 每步自检（边界/复杂度）、diff 快速人工扫一眼（防静默改动无关代码）、小步提交（可回滚）；
5. **现场演示的加分动作**：主动用 AI 生成测试、让 AI 解释自己写的代码（验证你理解了）、控制上下文（新问题开新会话避免污染）。

本质：面试官想看你在"AI 把写代码变便宜"之后，**读代码、验证、拆解任务的能力是否跟得上**。

### Q20. 基于现有大模型 + Agent 训练金融领域 Coder 模型（云-二面 Q7、云-实习-二面-2 Q6 开放题）

端到端方案（数据→训练→评估→部署）：

1. **领域数据构造**：金融语料（开源金融库/量化框架/SQL/监管规则文档 API）；**合成数据管线**：从 API 文档与合规规范自动生成"需求-代码-单测"三元组（用强模型生成多解 + 沙箱执行过滤只留可运行解）；真实任务众包（内部脱敏工单改造）；
2. **Agent 化 SFT**：不只教"写代码"，教**工具使用的轨迹**——多轮轨迹数据（需求→检索文档→写码→跑单测→读报错→修复），让模型学会 self-debug 循环；工具集：代码沙箱、金融 API/回测引擎、法规检索；
3. **RLVR（可验证奖励）**：天然的 verifiable reward——编译通过、单测通过率、回测指标合理性、静态检查（合规 lint：禁用 API/硬编码密钥检测）、沙箱运行安全；组内采样多解用 GRPO（同题可比，见《训练与算法》Q1）；多轮 agent 轨迹上的 credit assignment 用 outcome reward + 步级 process reward（编译错误步骤惩罚）混合；
4. **课程与课程外**：从单函数→模块→多文件工程；难度按通过率自适应（保持组内区分度——全对/全错组无梯度，DAPO 的动态采样思想）；
5. **评估**：held-out 金融任务 pass@k、与通用模型对比的领域增益、合规率（红线检测）、人工评审（代码风格/可维护性）；
6. **部署与安全**：RAG 接法规库（可更新不重训）、输出静态审查 + 沙箱执行隔离、人在环审批高危操作（交易/资金类代码）、持续数据回流（线上 bad case 进训练集）。

## 四、编程题对照

### Q21. 本篇编程题索引

- **LRU Cache**（一面-1 Q7）→ 《C++ 编程题》Q1 + `answers/algorithms/lru_cache.py`；
- **最长有效括号**（控股-实习 Q6，LeetCode 32）→ 《算法题》双指针/滑窗（`sliding_window.py::longest_valid_parentheses`，栈底哨兵解法）；
- **二叉搜索树迭代器**（淘天-一面-2 Q8，LeetCode 173）→ 《训练与算法》Q25 三解法（`answers/alibaba/bst_iterator.py`）。
