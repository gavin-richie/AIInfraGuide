---
title: "T4 车企题解：推理部署与 TensorRT"
description: "T4 车企/自动驾驶面经推理与部署题全解：TensorRT/OpenVINO/Plugin、vLLM 显存优化与 PD 分离、prefill/decode 与 KV Cache、CUDA Graph/PTX/编译流程/warp 原语、昇腾 NPU 与车载部署、BEV/跟踪/卡尔曼等感知与模型基础"
pubDate: 2026-10-05
topic: "公司题解-T4"
order: 91
tags: ["公司题解", "推理优化", "TensorRT", "高性能计算", "自动驾驶"]
draft: false
---

覆盖题库：`蔚来-AI-Infra-实习-一二三面`、`蔚来-AI-Infra-实习-一面-1/2`、`蔚来-AI-Infra-实习`、`小鹏汽车-AI-Infra-一面/实习`、`小鹏汽车-AI-Infra`、`理想汽车-AI-Infra-一面`、`理想汽车-AI-Infra-校招-一面`、`理想汽车-云-AI-Infra-实习-一面`、`卓驭-AI-Infra-实习`、`卓驭-AI-Infra-校招-一面/二面`、`小马智行-AI-Infra-实习`、`辉羲智能-AI-Infra-实习-一二三面`、`易控智驾-AI-Infra-二面`、`文远知行-AI-Infra-校招-一面`、`经纬恒润-AI-Infra-二面`。

与 01-12 篇同族的题只给 T4 视角的增量答案并标注交叉引用；编译器/MLIR 题见[90 篇](./90-t4-编译器与mlir.md)，C++/系统见[92 篇](./92-t4-cpp与系统.md)，编程题见[93 篇](./93-t4-编程与算法.md)。

## 一、TensorRT 与推理引擎

### Q1. 阐述对 TensorRT 的理解（蔚来-一二三面 一面 2）

TensorRT 是 NVIDIA 的**深度学习推理优化器 + 运行时**，核心价值是把"训练好的图"变成"针对当前 GPU、当前 shape 高度特化的可执行引擎"：

1. **工作流**：框架模型（ONNX/TF）→ **Parser** 解析成 network definition → **Builder** 做（a）图优化：算子融合（conv+BN+ReLU、GEMM+ bias+ activation）、精度校准（FP16/INT8/FP8）、kernel 自动调优（timing cache：同一算子对各种实现实测选优）、内存规划（liveness 复用）→ 序列化为 **engine**；运行时 `executeV2/V3` 执行；
2. **优化的来源**：融合减少中间 tensor 读写与 launch 次数；FP16/INT8 提升吞吐（Tensor Core）；per-shape 特化 + CUDA Graph；kernel 选优挑 cuBLAS/cuDNN/自研 kernel 中最快的实现；
3. **部署形态**：C++ API 嵌入车端/服务端进程，engine 与 GPU 架构 + TRT 版本绑定（不可跨），执行时数据经 `enqueueV2/V3` 的 bindings（输入输出指针）传入，中间显存由 engine 内部规划（liveness 复用）；缺算子用 **Plugin**（见 Q4）。

数据佐证可引用 solutions/ 的经验：推理侧 conv+BN 融合公式见[03 篇 Q13](./03-gemm与算子优化.md)。

### Q2. TensorRT 与 OpenVINO 的异同（蔚来-一二三面 一面 3）

| | TensorRT | OpenVINO |
|---|---|---|
| 厂商/硬件 | NVIDIA GPU（深度绑定） | Intel CPU/iGPU/VPU，另通过插件适配部分他卡 |
| 前端 | ONNX 为主 | ONNX/TF/PyTorch（转换模型为 IR：xml+bin） |
| 优化手段 | 融合、精度（INT8/FP8）、kernel timing、CUDA Graph | 融合、INT8（POT/NNCF 量化工具）、CPU 上的算子并行/内存复用 |
| 生态 | GPU 推理事实标准，LLM 栈（TRT-LLM）成熟 | 边缘 CPU 场景（工业/安防）强 |

相同点：都是"离线 builder 优化 + 在线 runtime 执行"的两段式，都做图级融合与低精度。差异本质是**硬件目标不同带来的优化重心不同**：TRT 吃透 GPU 的 Tensor Core/显存层级，OpenVINO 围绕 CPU 的 SIMD（AVX-512/AMX）与内存带宽做算子调度。车端选型：域控 GPU（Orin）用 TRT，MDC/CPU 备份链路可能用 OpenVINO。

### Q3. TensorRT 模型转换过程中遇到的问题（蔚来-一二三面 二面 2）

高频坑清单（按排查顺序讲）：

1. **算子不支持/解析失败**：`onnx-tensorrt` 报 unsupported operator——升 ONNX opset、用 `polygraphy surgeon sanitize` 替换等价子图、或写 Plugin；动态控制流（If/Loop/NonZero）支持差，转静态或拆图；
2. **动态 shape 与 profile：`builder` 需要 optimization profile（min/opt/max shape），漏配则 engine 只认固定 shape；shape 变化剧烈时多个 profile 或 bucketing；
3. **精度问题**：FP16 溢出/下溢（大 logits 上溢出 NaN）→ 逐层强制 FP32（`setPrecision`）；INT8 校准集不代表性 → 精度崩（校准方法见[06 篇 Q4](./06-量化.md)）；
4. **BN/推理模式**：导出 ONNX 前忘了 `model.eval()`（dropout/BN 统计不一致），输出对不上；
5. **版本绑定**：engine 跨 TRT 版本/GPU 型号不可用（trtexec 报 version mismatch），CI 里固定版本重编；
6. **性能回退**：某层没有 fast kernel 回退到 refImplementation（用 `trtexec --dumpLayerInfo` + profiler 查层耗时，per-layer 用 plugin 替换）。

### Q4. TensorRT 的优势与不足；缺算子时如何用 Plugin 解决（理想-校招一面 8）

- **优势**：NVIDIA 上性能天花板（kernel 选优 + 融合 + 低精度 + CUDA Graph）；生态成熟（trtexec/Polygraphy 分析工具链）；C++ 嵌入、确定性时延（车载关键）。
- **不足**：闭源黑盒（难定位/魔改）；engine 与 GPU/TRT 版本强绑定，升级成本高；动态 shape/复杂控制流弱；算子覆盖有限（新论文算子常缺失）——这正是 Plugin 的场景。
- **Plugin 自定义算子**：继承 `IPluginV2DynamicExt`（或 V3/IPluginV3），实现 `getOutputDimensions/supportsFormatCombination/initialize/enqueue（核心 kernel 调用）/serialize/deserialize`；用 `REGISTER_TENSORRT_PLUGIN` 注册；ONNX 图里以自定义 domain 节点存在（或 `trtexec --plugins` 加载 .so）。注意点：**显存生命周期归 TRT 管**（workspace）、format/type 组合要声明全（linear fp16/fp32）、多 stream 下 enqueue 必须无共享可变状态。写 plugin 的 kernel 本身就是普通 CUDA（93 篇的 kernel 方法论全部适用）。

### Q5. 推理阶段常见的优化手段（小鹏-AI-Infra 2）

分层 checklist：**模型级**（量化/剪枝/蒸馏/结构替换，见[06 篇](./06-量化.md)、[11 篇 Q6](./11-深度学习与训练.md)）；**图级**（融合/布局/常量折叠/内存复用，见[08 篇 Q6](./08-编译器与性能分析.md)）；**算子级**（合并访存/tiling/向量化/Tensor Core，见[01 篇 §8](./01-cuda-基础与体系结构.md)、[03 篇](./03-gemm与算子优化.md)）；**系统级**（batching/多流/CUDA Graph/pinned memory/异步流水）；**服务级**（continuous batching、PD 分离、prefix cache，见[05 篇 Q12](./05-推理引擎与优化.md)）。

### Q6. 使用过哪些推理框架，各自特点与适用场景（经纬恒润 二面 3）

- **TensorRT**：NVIDIA GPU 极致性能，车端/服务端 GPU 首选（见 Q1）；
- **OpenVINO**：Intel CPU/核显边缘部署（见 Q2）；
- **ONNX Runtime**：跨平台（CPU/CUDA/各 NPU EP）、易用性好，性能中上，扩展点 Execution Provider——车载国产芯片常提供 ORT EP；
- **TVM**：自研后端/长尾硬件，自动调优（见[90 篇 Q11](./90-t4-编译器与mlir.md)）；
- **NCNN/MNN/TNN**：移动端 CPU/GPU（arm 优化、无第三方依赖）；
- **vLLM/SGLang/TensorRT-LLM**：LLM 专用（PagedAttention、continuous batching、speculative decoding）；
- **国产 NPU 栈**：昇腾 ATC/MindSpore Lite、地平线 RDK（算法工具链转 nb 模型）。
选型逻辑：硬件归属 → 算子覆盖 → 动态 shape 需求 → 时延确定性要求。

### Q7. 权重文件如何导入到自行搭建的模型中（卓驭-实习 8）

PyTorch 路线（追问一般落在细节）：

```python
model = MyModel(config)                       # 自建结构
sd = torch.load("weights.bin", map_location="cpu", weights_only=True)
# 常见坑 1: key 对不上（带 module. 前缀/结构改名）→ 打印双方 state_dict().keys() 对齐
sd = {k.replace("module.", ""): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)   # 坑 2: strict=False 要检查返回
model.eval()                                  # 坑 3: 忘 eval 导致 BN/dropout 行为不同
```

其它格式：safetensors（`safetensors.torch.load_file`，避免 pickle 安全问题）；HF `from_pretrained`（分片 bin/safetensors 自动合并）；ONNX 权重在 graph initializer 里。**对齐验证**：导入后用固定输入跑一遍与参考实现对拍（误差 < 1e-3 量级），再谈部署——这也是排查"输出与预期不符"的第一步（见 Q9）。

### Q8. 模型输出与预期不符时如何排查；误差累积问题如何处理（卓驭-校招一面 6）

**排查二分法**：①输入对不对（预处理/归一化/颜色通道/BGR-RGB/resize 插值方式）；②权重对不对（load 是否 strict 全匹配，见 Q7）；③**逐层对拍**：reference（PyTorch）与部署实现逐中间 tensor 比 `max_abs_err/rel_err`，定位第一个发散层；④精度策略回退（该层 FP32）；⑤数值边界（exp 溢出、除零、INT8 饱和截断）。

**误差累积**：单层 fp16 相对误差 ~1e-3，深网络链式放大；对策：敏感层保 FP32（loss 归一化/大动态范围层）、累加用 FP32（kernel 里 acc 用 float）、BN 统计量用高精度存储、校准集覆盖长尾分布（量化误差，见[06 篇 Q4](./06-量化.md)）；系统级用"容差门限 + 分层回归测试"守住（CI 里每层设阈值）。

### Q9. 模型轻量化部署有哪些方法（理想-校招一面 7）

四类：**量化**（PTQ/QAT，INT8/INT4/FP8，见[06 篇](./06-量化.md)）；**剪枝**（结构化通道剪枝为主，配 BN γ 或泰勒准则选通道，剪后 finetune）；**知识蒸馏**（大模型教小模型，logits/中间特征对齐，见[11 篇 Q6](./11-深度学习与训练.md)）；**结构设计**（轻量骨干 MobileNet/GhostNet/深度可分离卷积，见[11 篇 Q3](./11-深度学习与训练.md)）。工程侧还有：算子融合、编码器-解码器非对称裁剪、早退（early-exit）。车载实际常组合：轻量骨干 + INT8 + 融合，指标是"mAP 损失 <1% 下的时延/功耗"。

## 二、大模型推理

### Q10. 大模型推理分哪几个阶段，各阶段特点（卓驭-实习 1）

**Prefill（处理 prompt）**：一次并行算完整个输入序列，GEMM 形态（大 M），**compute-bound**，产出 KV Cache；**Decode（逐 token 生成）**：每步 1 个 token，GEMV 形态（M=1），读全部权重与 KV，**memory-bound**（算术强度 ~1）。优化重心因此完全不同：prefill 吃 Tensor Core 利用率，decode 吃带宽与 batch（continuous batching 提高 MAC）。详见[05 篇 Q1](./05-推理引擎与优化.md)。

### Q11. 算子与 tensor 之间是什么关系（卓驭-实习 2）

算子 = **计算的定义**（输入输出 tensor 间的函数 + 超参属性：kernel size、axis...），tensor = **数据的容器**（shape/dtype/layout/device 的多维数组）。计算图里节点是算子、边是 tensor（producer 唯一、consumer 可多个）。落地到 kernel 层：**一个算子按 (shape, dtype, layout) 派发到具体 kernel 实现**——如 conv 算子在 NHWC fp16 1x1 时可能派发为 GEMM；反过来说 kernel 是算子在特定实例化参数下的实现。这也是为什么 dispatch 层（PyTorch dispatcher/TRT 的 tactic 选择）要带 shape 信息。

### Q12. 吞吐量的定义，推理场景如何衡量（卓驭-实习 4）

吞吐 = 单位时间完成的请求数或 token 数：LLM 用 **tokens/s**（区分 prefill tokens/s 与 decode tokens/s）、服务用 QPS、视觉用 FPS。测量要点：①**稳态测量**（warmup 若干迭代后取均值，排除 JIT/缓存冷启动）；②时延与吞吐的 trade-off 曲线（扫 batch 画 throughput-latency 帕累托前沿，车载选时延点、离线选吞吐点）；③报告**端到端**（含 tokenization/传输）还是模型前向；④多卡归一化（tokens/s/GPU）。相关指标 TTFT（首 token 时延，prefill 决定）、TPOT/ITL（每 token 时延，decode 决定）。

### Q13. 如何区分 memory-bound 和 compute-bound（卓驭-实习 5）

用 **Roofline**：算术强度 AI = FLOPs/Bytes，与机器平衡点（峰值算力/峰值带宽）比较，AI 小于平衡点为 memory-bound。实测上：Nsight Compute 的 SOL 面板 SM% vs Mem% 谁接近 100（见[08 篇 Q5](./08-编译器与性能分析.md)）。理论与工具细节见[03 篇 Q1](./03-gemm与算子优化.md)。典型：GEMV/decode/embedding 是访存受限，大 M GEMM/卷积是计算受限。

### Q14. 模型和算子确定后，影响推理速度的因素（卓驭-实习 6）

硬件（算力、**显存带宽**、频率、卡间互联）；精度（FP16/INT8/FP4，直接改 MAC）；batch/并行度（batch 提高带宽利用率，但有时延代价）；序列长度（KV 体积与 attention 复杂度，见 Q15）；kernel 质量（是否命中 cuBLAS 最优路径、融合程度）；调度（continuous batching、prefill/decode 干扰、多流重叠）；系统开销（python 解释器、launch 间隙 → CUDA Graph）；**散热降频**（车载 30 分钟推理后功耗墙，时延上涨——车规场景高频追问）。

### Q15. prompt 长度是否影响推理速度？decode/上下文呢（卓驭-实习 7）

三层影响：①**TTFT**：prefill 计算量 ∝ L²（attention）与 L（FFN），长 prompt 明显拖首 token；②**TPOT**：decode 每步 attention 要读全部 KV（长度 L 的历史），每步访存量随上下文线性增长，长上下文 decode 越来越慢；③**吞吐**：KV Cache 占显存 → 可容纳 batch 缩小 → 吞吐下降。缓解：prefix cache 复用重复前缀、GQA/MLA 压缩 KV（[04 篇 Q7](./04-attention与kvcache.md)）、量化 KV。

### Q16. prefill 阶段有没有加速 KV Cache 生成的方法（卓驭-实习 9）

prefill 是 KV Cache 的**生产**阶段（每个位置各生成一份 K/V），加速思路：①**并行本身**：prefill 内 token 无依赖可一次 forward（见 Q17），用 FlashAttention 类 kernel 把显存流量从 O(L²) 降到 O(L)；②**架构侧**：GQA/MQA/MLA 直接减少 KV 的 head 数/维度（生成量下降数倍）；③**前缀复用**：系统 prompt 等公共前缀的 KV 用 prefix cache 跨请求复用，等于"不重复生成"；④**分块**：chunked prefill 把大 prompt 切块，与 decode 混排降时延抖动（[05 篇 Q3](./05-推理引擎与优化.md)）；⑤ KV 生成后立即量化/压缩（FP8 KV）。

### Q17. prefill 阶段各 token 之间是否存在依赖，能否并行（卓驭-实习 10）

**计算上无依赖可并行**：causal attention 下位置 i 的输出只依赖 ≤i 的输入 token——输入序列全部已知（不像 decode 的输出是采样出来的），所以整个 prefill 等价于一次 teacher-forcing 的前向，一个 GEMM 批量算完，这正是 prefill compute-bound、吞吐高的原因。依赖体现在**语义/接口层**：训练时的 label 与 decode 时下一步输入才依赖前一步输出。追问点：padding 与 mask（非并行问题而是访存问题）。

### Q18. 稀疏化 KV Cache 是否需要修改训练流程，还是纯工程优化（卓驭-实习 11）

两类都有：**免训练（纯推理侧工程）**：H2O（重要度打分逐出）、SnapKV（观察窗口聚类选通道）、StreamingLLM（attention sink + 滑窗）——训练好的模型直接用，代价是长程信息有损；**需训练/改模型**：KIViT/轻型压缩（下投影层）、Native Sparse Attention（DeepSeek 式分块稀疏要训练 gate）、GQA/MQA 本身属于改架构需重训。工程上免训练方案落地快但精度敏感，训练方案稳但有成本。详见[04 篇 Q7](./04-attention与kvcache.md)的 KV 优化全景。

### Q19. 进一步提升大模型推理性能的技术手段（卓驭-实习 12）

综述见[05 篇 Q12](./05-推理引擎与优化.md)：模型侧（量化/稀疏 KV/投机解码）、调度侧（continuous batching/chunked prefill/PD 分离/prefix cache）、kernel 侧（FA/融合/低精度 GEMM）、系统侧（CUDA Graph/多卡 TP-EP/亲和性）。答此题的框架：先说清服务目标（时延 or 吞吐）再选手段。

**同族整题"KV Cache、算子融合、量化等模型级优化手段的原理"（卓驭-校招-一面 7）**一句话版：**KV Cache**——缓存历史的 K/V 免去逐 token 重算前缀注意力（原理与优化见[04 篇 Q6/Q7](./04-attention与kvcache.md)）；**算子融合**——把多个算子合成一个 kernel，消除中间 tensor 的全局内存写读与 launch 开销（规则见[08 篇 Q4/Q6](./08-编译器与性能分析.md)）；**量化**——权重/激活降到低精度表示（INT8/FP8/FP4），以精度换显存带宽与 Tensor Core 吞吐（见[06 篇 Q1](./06-量化.md)）。三者分别作用于访存量、kernel 数、位宽三个正交维度，可叠加。

### Q20. vLLM 的显存优化手段（文远-校招一面 3）

核心是 **PagedAttention**：KV Cache 按固定 block（16 token）分页存储 + block table 间接寻址，消除 fragmentation 与预留浪费（内部/外部碎片从 ~60-80% 降到 <4%），request 间还能共享前缀块（copy-on-write）。配套：**continuous batching**（迭代级调度，出一步进一个，不是等整批）提高 batch 占用；**抢占与重算**（swap/recompute）应对显存不足；权重与 KV 的显存配比（`gpu_memory_utilization` 预留）。详见[05 篇 Q2](./05-推理引擎与优化.md)。手写实现：`solutions/python/src/answers/attention/paged_attention.py`（Triton decode kernel + block table）。

### Q21. PD 分离机制的原理；两个队列的调度策略如何设计（文远-校招一面 2）

原理见[05 篇 Q4/Q5](./05-推理引擎与优化.md)：prefill（compute-bound）与 decode（memory-bound）分到不同卡/实例，KV 通过传输层（Mooncake 类）从 P 节点搬到 D 节点。**双队列调度设计**要点：

1. **队列**：prefill 队列（按请求 FIFO/ SJF——短 prompt 优先降平均 TTFT）、decode 队列（running batch，尽量保持满）；D 侧收到 KV 到位事件才把请求挂入 running；
2. **准入**：按 D 侧显存余量（KV 块数）控制 P 侧放行速率（背压），防 D 侧 OOM 或抢占风暴；
3. **干扰**：同卡混部时 prefill 大 kernel 会卡 decode 抖动，队列隔离正是为了消除这一点；比例可动态调整（如 1:3 卡数按负载）；
4. **公平与饥饿**：长 prompt 在 SJF 下可能饿 → aging（等待时间加权）。

### Q22. Chunked Prefill 的核心思想与应用场景（文远-校招一面 4）

把长 prompt 切成固定 token 数的块，逐块 prefill，剩余预算跑 decode——把"一次大计算"摊平，**消除 decode 的时延毛刺（TPOT P99）**、提高 GPU 利用率（小块填不满 SM）；场景：在线服务混跑长短请求。语义上块间就是顺序的 attention 累积，不影响正确性。详见[05 篇 Q3](./05-推理引擎与优化.md)。

### Q23. Attention 解决了什么问题，还有哪些不足（蔚来-一面-1 4）

解决：长程依赖（RNN 梯度消失/信息瓶颈）+ **训练并行**（序列维全并行 vs RNN 递归）；自注意力用 QK^T 动态路由信息。不足：①O(N²) 复杂度，长上下文贵（各种稀疏/线性注意力近似）；②推理 KV Cache 随长度线性涨显存与访存（见 Q15/Q18）；③位置信息要额外注入（RoPE/ALiBi）；④decoder 自回归无法并行生成（投机解码缓解，见 Q24）。原理与手写见[04 篇](./04-attention与kvcache.md)。

### Q24. 投机解码的原理；大小模型如何选择与搭配；是否微调（理想-一面 1）

原理：**小模型（draft）自回归猜 k 个 token → 大模型（target）一次并行 forward 验证这 k 个位置 → 按分布一致性接受最长前缀，拒绝处从修正分布重采样**——接受规则保证输出分布与 target 独立采样完全一致（无损）。每轮 1 次 target 前向换最多 k 个 token，k 个全收则 ~k 倍加速（decode 是访存瓶颈，验证 k 个 token 几乎不增加访存）。**搭配原则**：①**同 tokenizer/同家族**（LLaMA 7B 配同词表的 160M/360M draft）；②draft 接受率 α 决定收益（α≈0.8、k=4 时约 2-3×）；③新兴路线：无需独立 draft 的 Medusa（多解码头）/EAGLE（特征级自回归）/n-gram lookahead。**微调**：draft 在目标领域数据上微调可显著提高接受率（同一分布的任务文本）；target 不动——这正是"无损"的价值（详见[05 篇 Q12](./05-推理引擎与优化.md) 综述）。

### Q25. Dense 与 MoE 模型推理实现的差异；EP 如何实现（理想-一面 3）

Dense：一条 GEMM 链，权重总量固定，优化 = kernel/量化/调度。**MoE**：每层 router 选 top-k expert（如 DeepSeek-V3 每 token 选 8/256），推理差异：①**权重容量大但每次激活少**——显存放得下是前提，算力按激活算；②**动态路由**：token 要按 router 结果物理搬运到 expert 所在卡；③**负载不均**：热门 expert 拥塞（capacity/负载均衡 loss 缓解），batch 小时 expert 稀疏（每 expert 每步可能就几个 token，GEMM 退化成 GEMV）；④**batch 聚合**：vLLM/SGLang 把多步 decode 的 token 凑成 expert 级矩阵再算。**EP（Expert Parallelism）**：expert 分片到 R 张卡，每卡只存 1/R 的 expert 权重；一步 decode 内两次 **all-to-all**：dispatch（token 按 router 目的卡重排发送）→ 各卡本地 expert GEMM → combine（结果发回原卡求和）。通信量 ∝ batch×hidden×2，与 expert 数无关。训练/推理的并行全景见[07 篇](./07-分布式训练.md)。

## 三、算子优化与 GPU 体系结构

### Q26. 从 profiling 到落地的算子优化完整流程（卓驭-校招一面 1；理想-一面 2、云-一面 3 同族）

标准五步：①**定位**：Nsight Systems 看时间线，找占大头/空隙多的 kernel（[08 篇 Q5](./08-编译器与性能分析.md)）；②**剖析**：Nsight Compute 单 kernel，看 SOL/occupancy/stall 原因/bank conflict，判断 memory- or compute-bound（Roofline，[03 篇 Q1](./03-gemm与算子优化.md)）；③**假设与改写**：合并访存/向量化、smem tiling、occupancy 调整（block 尺寸/寄存器）、循环变换/ILP、融合（[03 篇 Q5-Q7](./03-gemm与算子优化.md)）；④**对拍验证**：CPU/框架参考实现比误差（数值题用相对容差，见[02 篇 Q1](./02-cuda-编程题.md)的精度讨论）；⑤**回归与沉淀**：benchmark 基线入库（防回退）、跨 shape 泛化测试。可补充的经验维度（理想"做了哪些层面的优化"）：算法层（数学等价替换/精度）、kernel 层（上述）、调度层（多流/图）、系统层（内存池/锁页）。

### Q27. 常用 profiling 工具有哪些；Nsight Systems 是否支持指令级流水线分析（卓驭-校招一面 2）

工具箱：**Nsight Systems**（系统时间线：kernel 时序/gap/流重叠/memcpy 占比/CPU launch 延迟）、**Nsight Compute**（单 kernel 深剖：SOL/warp stall/内存表/指令统计）、CUPTI（自研埋点）、PyTorch profiler / torch.profiler（算子级，nsys 后端）、nvbench；Linux 侧 perf/VTune（CPU 段）。**指令级流水线分析不在 Nsight Systems**——它是系统级时间线工具；kernel 内指令调度/占用/流水线 stall 要用 **Nsight Compute**（`--set full`，看 Scheduler/Warp State Statistics/Pipe Utilization）；再往下到 SASS 级指令重排是 nvdisasm/cuobjdump 静态分析。这个辨析正是考点。

### Q28. Warp 利用率偏低如何归因；负载不均如何解决（卓驭-校招一面 3）

归因按 NCU 的 **Warp State Statistics** 拆：`long scoreboard`（等全局内存）→ 合并访存/缓存/预取；`short scoreboard / MIO throttle`（等 smem）→ bank conflict/padding、减 smem 压力；`barrier`（等慢兄弟 block）→ 均衡 block 内分工；`selected/not selected`（warp 太少）→ 提 occupancy（减寄存器/smem、调 block 尺寸）；`branch resync` → divergence（见 Q34）；**tail effect**（grid 太小，末尾 SM 空转）→ 增大并行度或 persistent kernel。**负载不均**（数据驱动的变长任务，如 NMS/直方图热点/变长序列）：①静态均分改**动态取任务**（全局 atomic counter 做 work stealing，block 做完自取下一段）；②**分块更细**再聚合；③按 key 预排序分组（同类任务落同 block）；④persistent kernel + warp specialization（生产者-消费者分工，[03 篇 Q8](./03-gemm与算子优化.md)）。

### Q29. 如何使用 Nsight 辅助优化，关注哪些指标（卓驭-实习 13）

见[08 篇 Q5](./08-编译器与性能分析.md)：Systems 看"哪里慢/空隙"，Compute 看"为什么慢"。核心指标：Duration、Compute/Memory SOL（谁贴 100 谁是瓶颈）、Achieved Occupancy、Warp stall 主因、L1/L2 hit rate、DRAM 带宽利用率、事务/请求比（合并性）、寄存器数与溢出。方法论：先 Systems 定位（避免优化不重要的 kernel），再 Compute 对比优化前后同一 kernel 的 SOL 与时长。

### Q30. Bank Conflict 的原因与解决方案（蔚来-实习 2、辉羲-二面 2）

见[01 篇 §11](./01-cuda-基础与体系结构.md)：smem 按 32 bank × 4B 组织，同一 warp 内两个以上线程访问**同 bank 不同地址**→ 串行化（conflict 次 replay）。解法：padding（`tile[32][33]`）、swizzle（XOR 重排，cute/Triton 自动）、改数据布局（AoSoA）、对齐访问、向量化（float4 天然跨 bank）。检测：NCU 的 `l1tex__data_bank_conflicts_pipe_lsu`。经典例子：矩阵转置见[02 篇 Q4](./02-cuda-编程题.md)（2048² 实测数据）。

### Q31. 同一 Warp 内不同线程的访存约束条件（蔚来-实习 3）

**合并访存（coalescing）**：warp 的 32 线程访问若落在尽可能少的 32B 扇区（L2 粒度）/128B 缓存行内则一次事务完成。约束：①地址连续且**对齐**（首地址 32B/128B 对齐、步长为 4B/8B/16B 的连续向量最优）；②跨 128B 边界的访问拆多事务（无害但多花带宽）；③**同一地址**（所有线程读同一字）硬件广播，一次完成；④scatter/gather（indirect）不可合并，是性能杀手；⑤ stride 访问（如按列遍历行优先矩阵）每线程隔 W×4B → 32 个事务，劣化指标是"事务/请求比"，见[01 篇 §24](./01-cuda-基础与体系结构.md)。

### Q32. 共享内存的广播机制（蔚来-实习 4）

同一 warp 内多线程读**同一 smem 地址** → 一次访问广播给所有请求线程，不算 conflict（这正是 smem 的第五种访问模式 broadcast）。规则总结：同地址=广播（免费）；不同 bank=并行（理想）；**同 bank 不同地址=conflict**（串行化）。典型应用：tile 内共享的标量/权重（如 A[i][k] 行广播给整列线程）、warp 归约时 `__shfl_sync` 的替代场景；反例：多线程同时广播写同一地址是 UB（只允许一线程写）。

### Q33. 编写算子时如何最大化利用缓存；如何按 L1/L2 容量分块（辉羲-二面 3）

原理：命中取决于**复用距离**（两次访问间隔的数抛数据量）< 缓存容量。分块把循环重排成"用完再走"：GEMM 中 A 的一个 tile 被 B 的整行 tile 复用，tile 选取使 **A_tile×B_tile 的工作集 fit 进 L2**、单 block 内 tile fit 进 smem/L1。数字感（面试要背）：3090 smem/SM 100KB（可配 L1）、L2 6MB；A100 L2 40MB、H100 50MB。操作顺序：算 reuse 距离 → 选 block/tile 尺寸 → NCU 验证 L2 hit rate 与 dram 带宽；L2 命中但带宽低说明复用好，反之加大 tile 或循环重排（[01 篇 §22](./01-cuda-基础与体系结构.md)、[03 篇 Q3](./03-gemm与算子优化.md) 的 GEMM 优化路径）。

### Q34. 什么是 warp divergence，对性能的影响与对策（辉羲-二面 4）

warp 是 SIMT 执行单元，32 线程共享一个指令流；`if/else` 使两分支线程交替 disabled，**串行执行两个路径**（活跃掩码屏蔽），分支越多浪费越多。影响：有效吞吐按最短路径比例下降，还伴随 `branch resync` stall。对策：①把分支提到数据布局层（按谓词重排数据，同 block 同类）；②用算术替代分支（`x = cond ? a : b` 编译为 select/predication，无 divergence）；③warp 级原语投票代替分支统计（`__ballot_sync`）；④排序/分桶后分别处理（histogram、segment sum 的分桶思想，[02 篇 Q8](./02-cuda-编程题.md)）。注意：**不同 block 间分支无代价**，warp 内才有。

### Q35. blockDim.x 与 gridDim.x 的最大值；grid/block/thread 三级层次的理解（辉羲-二面 5/8）

`gridDim.x` ≤ **2³¹−1**（x 维 21 亿+），`gridDim.y/z` ≤ 65535；`blockDim.x/y` ≤ 1024、`blockDim.z` ≤ 64，且 **x·y·z ≤ 1024**（每 block 线程总数上限，受寄存器文件/smem 容量约束）。**三级层次的理解**：thread 是最小执行单元（私有寄存器/局部变量），block 是**唯一能通过 smem + `__syncthreads` 协作**的层级（共享 smem/L1、共同驻留同一 SM），grid 是全部线程的集合（跨 block 只能经 global memory + kernel 边界/atomic 通信，见 Q43）——层次决定"哪些资源能共享、哪种同步可用"，详见[01 篇 §1](./01-cuda-基础与体系结构.md)。追问常连着"怎么选 block/grid 尺寸"：见[01 篇 §10](./01-cuda-基础与体系结构.md)（128/256 为主流起点、occupancy 与尾部效应权衡）。

### Q36. 手写 CUDA 矩阵乘：naive 版 + 优化方向 + 最佳分块确定（辉羲-二面 6）

naive：每线程一个 C 元素，行×列各扫 K。优化路线：smem tiling → double buffering → 寄存器 tile（每线程 4-8 输出）+ float4 → warp 级 tile/mma（Tensor Core）→异步拷贝（cp.async/TMA）。**最佳分块没有解析解**：理论上由 smem 容量（两 tile 放得下）、occupancy（block 造成的寄存器/smem 占用）、复用率（tile 越大全局访存越少）联立，工程上**autotune 扫参**（128×128×32 / 128×64×64 等候选实测）。完整实现与 3090 实测（naive 2166 → tiled 2804 GFLOPS）见[02 篇 Q3](./02-cuda-编程题.md)（`solutions/cuda/src/gemm.cu`），优化方法论见[03 篇](./03-gemm与算子优化.md)。

### Q37. 共享内存与硬件缓存的区别（辉羲-三面 11）

| | Shared Memory | L1/L2 Cache |
|---|---|---|
| 管理 | **显式**：程序员决定 layout/同步（`__syncthreads`） | 隐式：硬件替换策略 |
| 空间保证 | 分配即驻留（scratchpad，确定性时延） | 不保证命中（容量竞争） |
| 物理关系 | 与 L1 共享同一块 SRAM，可按比例配置（如 100KB 里 smem+L1 分割） | L1 每 SM 私有；L2 全卡共享 |
| 访问模式 | 按 bank 并行/广播/conflict 规则（见 Q30-Q32） | 按缓存行/局部性 |
| 作用域 | block 内共享；跨 block 不通 | L2 全 grid 可见 |

车载/实时场景偏爱 smem 的**确定性**；通用算子多用"L1 顺其自然 + smem 管关键数据"。细节见[01 篇 §6/§7](./01-cuda-基础与体系结构.md)。

### Q38. Tensor Core 与 CUDA Core 哪个快；Tensor Core 原理（辉羲-三面 12）

矩阵乘用 **Tensor Core 快一到两个数量级**（A100 FP16：312 vs 19.5 TFLOPS dense）。原理：CUDA Core 是标量 FMA 单元（每周期每核 1-2 条乘加）；Tensor Core 是**矩阵乘累加专用流水线**——一条 `mma` 指令（如 `m16n8k16`）由一个 warp 协作完成：操作数 fragment 分布在各线程寄存器，硬件内部完成小矩阵乘并把累加也放寄存器（D=A·B+C），数据路径针对矩阵复用优化、无需取指译码开销。支持 FP16/BF16/INT8/TF32/FP8/FP4（代际不同）；编程上走 WMMA API/mma PTX/cuBLAS/cutlass/Triton（自动用 TC，`solutions/python/src/answers/triton_kernels/matmul.py` 实测 tf32 加速）。限制：形状对齐（16/8 倍数）、累加 FP32 才保精度。

### Q39. 常见卷积算法及优缺点（辉羲-二面 9）

1. **直接卷积（direct）**：无变换开销、缓存友好时快；小通道时常用（winograd 的门槛都不够）；
2. **im2col + GEMM**：把卷积展开成矩阵乘吃 cuBLAS/Tensor Core；通用、工程首选；代价是 im2col 的显存膨胀与拷贝（可融合进 GEMM 的 prologue 缓解）；
3. **Winograd**：代数变换减少乘法（F(2×2,3×3) 乘法降 2.25×）；stride=1、小 kernel（3×3）最优；数值精度差（INT8 谨慎）、大 tile 变换开销吃掉收益；
4. **FFT**：大 kernel/大图时 O(n²logn) 优势；小 kernel 不划算、复数运算与 padding 复杂，深度学习几乎弃用；
5. **隐式 GEMM/ strides 直接向量化**（cutlass/cudnn 现代路径）：不做物理 im2col，索引映射进 kernel，兼得两者优点。
推理框架选择见[03 篇 Q14](./03-gemm与算子优化.md)；conv+BN 融合公式见[03 篇 Q13](./03-gemm与算子优化.md)。

### Q40. 什么是 CUDA Graph；为何占更多显存；推理哪个阶段适合（小马智行-实习 1）

原理与基本收益见[01 篇 §9](./01-cuda-基础与体系结构.md)：把一串 kernel 的 launch 参数/依赖固化成图，一次 `cudaGraphLaunch` 提交，**消 CPU launch 开销与 kernel 间隙**。**为何更占显存**：①**地址固化**——capture 时每个 kernel 的指针/workspace 地址被 bake 进图，这些 buffer 必须保持唯一占有，无法与其它图/任务复用，caching allocator 在 capture 期间分配的块会被长期持有（grow-only）；②每个图独立一套 workspace（多图/多 batch 档位各建一图时叠加）；③图内部无动态释放时机（普通执行时框架可按 liveness 复用的内存，图里地址已定死）。**适合阶段**：**decode**（kernel 小而密、launch 开销占比最高）与 shape 固定的 CV 流水线（车载感知每帧同构）；不适合 prefill（变长、shape 档位爆炸），变长场景用 conditional graph/graph update 或分桶多图。

### Q41. NVIDIA GPU PTX 机器模型的核心概念（小马智行-实习 2）

PTX 是**面向虚拟 GPU 的中间表示**（与 SASS 的区别见[01 篇 §19](./01-cuda-基础与体系结构.md)），机器模型核心概念：

1. **执行模型**：SIMT 三级层次 thread/CTA（block）/grid，warp 一致执行；屏障 `bar.sync`（CTA 内）与合作组（grid 级）；
2. **状态空间（显式分级内存）**：`.reg`（虚拟寄存器，数量不限由 ptxas 分配物理寄存器）、`.global`（device 显存）、`.shared`（CTA 内）、`.const`、`.local`（线程私有栈）、`.param`（参数/ABI）；类型前缀 `.f32/.b32/.pred` 等；
3. **虚拟化与前向兼容**：寄存器虚拟化 + 指令集稳定 → 同一份 PTX 可 JIT（driver 内 ptxas）到多代架构的 SASS，新卡跑旧程序；
4. **内存一致性模型**：weakly ordered，`ld/st` 的 `.ca/.cg/.cv` cache 策略与 `atom/red`、`fence.acq_rel`、`barrier.sync` 内存序；
5. **同步与通信原语**：`shfl.sync`（warp 内寄存器交换）、`vote`、`red/atom`、`barrier.cta.sync`。

对编译器岗（小马此题的上下文）再加一句：MLIR 的 nvvm/llvm 方言最终经 NVPTX 后端生成 PTX，再交 ptxas——PTX 是编译器与驱动之间的契约层。

### Q42. CUDA 代码的完整编译流程（小马智行-实习 3）

nvcc 离线编译（.cu → 可执行）：

```
1. cudafe++ (cicc 前端 EDG): __global__ 与 host 代码分离、模板展开 → NVCC IR
2. cicc: IR 优化 → 生成 PTX（虚拟 ISA）
3. ptxas -arch=sm_XX: PTX → SASS（真实指令，物理寄存器分配/调度）→ cubin
4. fatbin: 打包 cubin（多 arch）+ PTX（兼容档）
5. host 编译器 (gcc): host 代码 + fatbin 资源 → 链接 cudart → 可执行文件
运行时: loader 按当前卡选 exact-match cubin；无匹配则 JIT fatbin 内 PTX（首启慢）
```

其它路径：`nvrtc` 运行时编译（Triton 即此路径：Python → MLIR/LLVM → PTX → cuModuleLoad）、driver API 的 JIT。分层与调试（`-ptx`/`-cubin`/`cuobjdump -sass`）见[01 篇 §19](./01-cuda-基础与体系结构.md)。

### Q43. 跨 Block 的通信方式；warp 级原语列举（小马智行-实习 5）

**跨 block**：block 间无直接通信（独立调度、无全局屏障保证），工程手段：①**global memory + kernel 边界**（最常用：两段 kernel，完成即全局可见，见[01 篇 §12](./01-cuda-基础与体系结构.md) threadfence）；②global **atomic**（计数/flag，配合 `__threadfence()` 发布）；③**cooperative groups 的 grid.sync**（要求 co-resident 启动，grid 不能超同时驻留容量）；④多流 + event；⑤persistent kernel 内用 atomic 做生产者-消费者。**warp 原语**：`__shfl_sync/mask`（lane 间寄存器交换：down/up/xor/idx 四种模式——归约/转置免 smem）、`__ballot_sync`（谓词位图）、`__any_sync/__all_sync`、`__activemask`、`__match_any_sync`（同值分组）、`__reduce_add_sync`（sm80+ 硬件归约）、`mma.sync`（Tensor Core）。用法实例：树归约 + warp shuffle 版 reduce 实测 124×（[02 篇 Q1](./02-cuda-编程题.md)）。

### Q44. 锁页内存（Pinned Memory）的作用与使用场景（经纬恒润 二面 4）

`cudaMallocHost` 分配**页锁定**的 host 内存：OS 不可换出 → device 的 DMA 可以直连（普通 pageable 内存要先拷进驱动内部 staging buffer）。作用：①H2D/D2H **带宽约 2×**；②**异步** `cudaMemcpyAsync` 真异步（pageable 版会退化同步），可与 kernel 计算重叠（流水线预取）；③mapped pinned（zero-copy）让 device 直接访问 host 地址，超大数据省一次拷贝（PCIe 带宽就是上限）。场景：训练/推理的数据加载流（pin_memory=True + non_blocking=True）、多卡间 P2P、分布式 NCCL 的 host buffer。代价：锁页占物理内存且 pinned 分配慢（要复用池子）、过量 pinned 拖垮整机（可换页内存变少），车载小内存环境要节制。

### Q45. CUDA 常用优化策略（易控智驾 二面 2）

汇总框架（细节全在 01-03 篇）：访存（合并、smem tiling/padding、向量加载、L2 复用）、并行度（occupancy、block/grid 配置、ILP）、同步（减 `__syncthreads`、warp 原语代替 smem）、指令（少分支、循环展开、`__ldg`/restrict、fast math）、精度（Tensor Core、混合精度）、系统（多流、CUDA Graph、pinned、kernel 融合、原子减热点）。入口见[01 篇 §8](./01-cuda-基础与体系结构.md)，优化路径见[03 篇](./03-gemm与算子优化.md)。

## 四、NPU 与车载部署

### Q46. CPU、GPU 与 NPU 优化策略的差异及适用场景（小鹏-实习 1；卓驭-实习 14"是否关注过计算/芯片架构"同族）

| | CPU | GPU | NPU（昇腾/地平线等） |
|---|---|---|---|
| 优化重心 | cache 局部性、SIMD 向量化（AVX/NEON）、多核绑核、避免伪共享 | SIMT 吞吐：占用率/合并访存/Tensor Core | **图编译**（算子切分/双缓冲/流水）、量化（INT8 为主）、DDR 带宽 |
| 编程模型 | 串行+向量化 intrinsics | CUDA kernel 自由度大 | 算子级/图级（kernel 通常封闭，靠工具链） |
| 内存 | 三级 cache 硬管 | 手动 smem + cache | 显式多级 on-chip（L1/Buffer）+ DDR，靠编译器编排 |
| 场景 | 通用控制、小模型、低功耗单路 | 大模型训练/推理、灵活算法 | 固定负载（车载感知/语音）极低功耗下高吞吐 |

共同点：都吃"局部性 + 复用 + 并行度"，Roofline 思想通用（[03 篇 Q1](./03-gemm与算子优化.md)）；差异本质是**可编程性 vs 能效**的取舍。

### Q47. 昇腾 NPU 与 NVIDIA GPU 的架构差异；内存层级设计（卓驭-校招一面 4）

昇腾（达芬奇架构）核心是** Cube/Vector/Scalar 三单元分离**：Cube 单元做矩阵乘（16×16×16 矩阵一条指令，INT16/FP16 峰值远超同代 GPU 的非 TC 路径），Vector 做向量后处理（非线性/归一化），Scalar 做控制流；配合 **MTE（DMA）搬数**，形成"Cube 算、Vector 处理、MTE 搬"的软件流水（编译器自动双缓冲/多级流水）。内存层级：L0A/L0B/L0C（Cube 专属输入/累加 buffer）→ L1/HBM 间 explicit buffer → L2 → HBM/DDR——比 GPU 更"显式"（GPU 的 smem/L1 由程序员选用，昇腾的 on-chip buffer 由编译器统一编排，tile 策略更关键）。其它差异：执行以**整图编译**（ATC → om）为中心而非动态 kernel launch；batch/shape 静态化要求更强；软件生态（CANN vs CUDA）成熟度差距是工程痛点。通用对比（scratchpad vs cache、确定性时延）见 Q37/Q46。

### Q48. NPU 的计算执行流程（易控智驾 二面 1）

以昇腾为例端到端走：①**离线**：模型（ONNX/Caffe）→ ATC 图编译：算子融合、tiling 切分（按 L1/Cube 尺寸）、指令生成（AI Core 的多级流水：MTE1/MTE2 搬入 → Cube → Vector → MTE3 搬出，双缓冲重叠）、内存规划 → om 文件；②**在线**：runtime 加载 om → stream 上 launch（task 粒度）→ AICore 执行（数据流驱动，event 同步）；③**同步**：host 侧 aclmdexecu 前向，异步 stream + event。本质差异：GPU 是"运行时逐 kernel 派发"，NPU 是"编译期把整个网络的指令流/数据流全编排好"——所以 NPU 时延确定性强、适合车规，但动态 shape/新算子更笨重。

### Q49. 对自动驾驶技术栈的了解（小鹏-实习 2）

分层讲：**感知**（多模态：相机/激光/毫米波 → 检测/分割/跟踪；主流 BEV 架构：LSS/BEVFormer → occupancy/稀疏 query，见 Q64-Q66）、**预测**（他车轨迹预测，向量编码/Transformer）、**规划控制**（PnC：lattice/优化式，或端到端；L2+ 城市NOA）、**定位建图**（SLAM/高精图/BEV 局部图）、**数据闭环**（影子模式采集 → 自动标注（大模型离线）→ 训练（分布式）→ 仿真/回灌测试 → OTA）。AI Infra 在其中的位置：训练集群效率（[07 篇](./07-分布式训练.md)）、推理部署（本篇 Q1-Q48：车端 Orin/NPU 上的量化与时延）、数据管线（大规模存储/吞吐）。

### Q50. 自动驾驶的高性能计算需求：时延、吞吐与功耗的权衡（卓驭-二面 6、易控-二面 3）

车端约束：**硬实时**（感知到规划端到端 ~100ms 级预算，单模型常 10-35ms）、传感器帧率固定（8-12 路相机 30fps → 吞吐下限）、**功耗墙**（域控 30-100W 散热约束，持续推理不能降频）。三者耦合：降时延（小 batch/高主频）牺牲能效；提吞吐（batch 多路）加时延；功耗决定可持续的频率。工程手段：INT8 量化（能效 ~4×）、NPU 化（能效比 GPU 高数倍）、帧间流水与流水线重叠（传感器-感知-规划 overlap）、DNN 任务分级（关键模型高频、辅助模型低频）、DVFS 按负载调频、算子/图层面优化（本篇三、四节全适用）。答题给出量化语言的例子加分："8 路 1080p@30fps、35ms 时延预算、<30W——所以选 NPU INT8 + 融合 + 分档 batch"。

### Q51. 调度器的设计思路与关键考量（易控智驾 二面 4）

先问清调度对象（推理服务请求 / 车端多任务 DNN / 训练集群 job），通用框架：**目标**（SLA：时延 P99/吞吐/公平/优先级）、**资源模型**（GPU/NPU 显存、算力、带宽、互连拓扑）、**队列与策略**（优先级队列、抢占、aging 防饿死）、**放置**（亲和性：NUMA/PCIe 拓扑、KV cache 局部性）、**自适应**（负载反馈扩缩容）。推理实例（结合 [05 篇](./05-推理引擎与优化.md)）：continuous batching 的迭代级调度 + PD 双队列（Q21）+ 显存背压；车端实例：RTOS 优先级调度（感知 > 舒适性功能）+ DNN runtime 的多流/时间片（保证关键路时延确定性）。

### Q52. 如果负责推理优化，会如何展开（理想-云 三面 2）

给一个 30/60/90 天式框架：①**摸底**：建立基准（时延/吞吐/显存/精度四指标 + 硬件监控），跑通 profiling 工具链（nsys/ncu），找出 top 耗时算子与系统空隙（Q26 流程）；②**快赢**：已有引擎的低垂果实——量化（[06 篇](./06-量化.md)）、batch/调度策略、CUDA Graph、pinned memory 数据通路；③**深入**：热点算子手写优化/融合（[03 篇](./03-gemm与算子优化.md)）、缺算子 Plugin；④**体系化**：CI 性能回归、自动化调参（autotune）、与训练侧协作（量化友好的 checkpoint）；⑤长期：编译器化（TVM/MLIR 定制 pass，[90 篇](./90-t4-编译器与mlir.md)）。关键是先讲"度量与定位"再讲"优化"——没有基准的优化无法证明。

## 五、深度学习与感知模型基础

### Q53. PyTorch DDP 的原理与实现机制（理想-校招一面 1）

DistributedDataParallel：每卡一份完整模型副本，**前向后向各卡独立跑，梯度 all-reduce 平均**。机制：①`DistributedSampler` 切分数据保证不重复；②构建时把每个参数包成 `GradBucket`，**反向传播中梯度就绪即通信**（overlap 计算与通信，Reducer hook）；③NCCL ring/树 all-reduce 均摊带宽；④find_unused_parameters 处理条件分支。对比 DP（单进程多线程、GIL+单卡聚合）与 ZeRO/FSDP（切参数，[07 篇 Q3](./07-分布式训练.md)）。通信量推导（为什么 DP 加速差、DDP 通信量 2Φ）见[07 篇 Q1/Q2](./07-分布式训练.md)。

### Q54. 模型训练有哪些并行方式（理想-校招一面 5）

数据并行 DP/DDP/ZeRO、张量并行 TP（行列切分 + all-reduce/all-gather）、序列并行 SP（长序列维切）、流水线并行 PP（stage 切分 + micro-batch bubble 优化 1F1B/interleaved）、专家并行 EP（MoE，见 Q25）、混合 5D 并行与 auto-parallel。各自通信模式与选型（小模型 DDP、大模型 TP×PP×DP、MoE 加 EP）详见[07 篇](./07-分布式训练.md) Q3-Q5/Q11。

### Q55. BN 的作用，为何加快收敛（理想-校招一面 9、云-一面 1）

计算/训练推理差异/与 LN 关系完整版见[11 篇 Q1](./11-深度学习与训练.md)。加快收敛的机制：①拉平各层输入分布（缓解内部协变量偏移，原始论文解释；现代更主流的解释是**平滑损失面/允许更大学习率**）；②β、γ 恢复表达能力；③ batch 统计带来的正则化。注意车载部署追问：BN 推理时折叠成 scale-shift（与 conv 融合零开销，[03 篇 Q13](./03-gemm与算子优化.md)）。

### Q56. 梯度消失和爆炸的成因与解决思路（理想-云-一面 2）

成因：链式法则连乘——激活函数导数 <1（sigmoid 0.25 上限）或权重谱半径失衡，深网络乘积指数级衰减/放大；RNN 上即长程依赖失效。解决：结构（残差/LSTM 门控/归一化层 BN-LN）、初始化（Xavier/He 保持方差）、激活（ReLU 族）、优化（梯度裁剪防爆、warmup）、损失尺度（FP16 loss scaling，见[06 篇](./06-量化.md) 混合精度）。诊断：逐层梯度范数直方图（hook 打点）。

### Q57. CV 发展脉络：从 AlexNet 说起（理想-校招一面 2）

AlexNet(2012, ReLU/Dropout/GPU) → VGG(小核堆深) → GoogLeNet(Inception 多尺度) → ResNet(残差，深过百层) → DenseNet → 注意力入 CV（SE/CBAM 通道空间） → **ViT**(2020, patch 序列进 Transformer) → Swin（层级窗口，稠密任务） → ConvNeXt（现代化 conv 对打） → 大一统/多模态（CLIP/DINOv2 自监督）。激活函数演进：sigmoid → ReLU → Leaky/PReLU → ELU → GELU（Transformer 标配）。轻量化支线 MobileNet/GhostNet 见[11 篇 Q3](./11-深度学习与训练.md)。答此题抓两条主线：**加深（怎么解决退化）**与**注意力化（怎么替代局部性假设）**。

### Q58. NLP 发展路径：RNN → LSTM → Transformer（理想-校招一面 3）

word2vec 静态词向量 → RNN/LM（序列递归，难并行、长程梯度问题） → LSTM/GRU（门控缓解） → Seq2Seq+Attention（对齐解码，注意力初现） → **Transformer（2017，自注意力替代递归，位置编码补序列信息）** → BERT（双向编码预训练）/GPT（自回归） → LLM 缩放定律 → 多模态 LLM。每一步的驱动力讲清楚：RNN 为何不行（并行差/梯度）、LSTM 缓解了什么没解决什么（长程仍有衰减、依旧串行）、Transformer 用什么换掉了什么（O(N²) 换全并行+全局感受野）。

### Q59. RNN 与 Transformer 优缺点，重点说明 Transformer 的并行性优势（理想-校招一面 4）

RNN：O(N) 顺序依赖 → 训练必须逐步递归（时间维串行），长程衰减，但推理逐 token 天然流式、参数省。Transformer：**训练/预填充时序列维全并行**（所有位置一次矩阵乘 + attention），GPU 大 GEMM 吃满 → 预训练提速数个量级是它取代 RNN 的根本原因；代价 O(N²) 计算与 KV Cache 显存、自回归 decode 阶段仍逐 token（并行优势主要在训练与 prefill，decode 要靠投机解码/缓存优化，见 Q23/Q24）。

### Q60. Transformer 与 CV 的结合体现在哪里，ViT 设计思路（理想-校招一面 6）

ViT：图像切 16×16 patch → 线性嵌入为 token 序列 + 位置嵌入 → 标准 Transformer 编码器 → class token 做分类。"图片即句子"的映射让 NLP 的架构/训练范式（大规模预训练+微调）平移到 CV：数据够大时超过 CNN（JFT-300M）。后续结合：Swin 的层级化做检测分割、DETR 用 object query 做集合预测（去掉 NMS 后处理，见 Q64）、CLIP 图文对齐、BEVFormer/StreamPETR 把多相机 token 化做 3D 感知（Q65）。本质上 Transformer 提供"**统一 token 接口**"，使模态/任务/编码器解码器自由组合。

### Q61. GPU 与 CPU 的核心区别与设计差异（蔚来-一面-1 5、辉羲-二面 7）

完整版见[01 篇 §3](./01-cuda-基础与体系结构.md)：CPU 少核强控（乱序/分支预测/大缓存，延迟导向），GPU 多核吞吐（数千 ALU、硬件多线程隐藏访存延迟、SIMT）。要点补数字：3090 10496 CUDA core vs CPU 几十核；GPU 用切换 warp 隐藏延迟而非预测/乱序。

### Q62. 算术强度分析与矩阵乘的性能瓶颈（小鹏-AI-Infra 3 项目题）

按 Roofline 讲：GEMM MNK 的 FLOPs=2MNK、Bytes≈2(MK+KN+MN)×dtype → AI 随 N/K 增大；对照机器平衡点（3090：35.6TFLOPS/936GB/s ≈ 38）判断 bound；再落到 M=1（GEMV，纯带宽）与 M 小（decode）为何慢、batch 为何救吞吐。公式与实例见[03 篇 Q1/Q3](./03-gemm与算子优化.md)。

### Q63. 目标检测主流方法（理想-云-二面 1）

两段系：R-CNN → Fast/Faster R-CNN（proposal+分类回归，精度高慢）；单段：YOLO 系/SSD（密集预测，快，v1-v3 演进见[11 篇 Q4](./11-深度学习与训练.md)）；**Anchor-free**：FCOS/CenterNet；**DETR 系**：Transformer set prediction（去掉 NMS/handcrafted anchor，DETR→Deformable DETR→DINO）；脉冲式新线：YOLOX/v8 工程化。车载视角：多路相机 + BEV 空间（Q64-Q66），单目检测常用 anchor-free + 量化部署。

### Q64. LSS 的后续改进；LSS 与 BEVFormer 的算力对比（理想-云-二面 2）

**LSS（Lift-Splat-Shoot）**：每相机预测**显式深度分布** + 语义，外积得到 frustum 特征，scatter 到 BEV 网格（voxel pool）。后续：**BEVDepth**（LiDAR 监督深度，修正 LSS 深度不准）、BEVDet/BEVDet4D（+时序）、**BEVFusion**（多模态融合框架化）、FB-BEV（显式隐式结合）；再往后稀疏化：**StreamPETR/Sparse4R**（3D object query，摆脱稠密 BEV）。**算力对比**：LSS 的开销在深度分布（softmax over D 桶）与 voxel scatter（大显存带宽、不规则写）；BEVFormer 的开销在 **spatial cross-attention（deformable 采样，QKV 投影与采样点循环）+ temporal self-attention**，计算密度高（GEMM 多）但对带宽压力小于 scatter。粗略量级：同分辨率下 BEVFormer FLOPs 更高、LSS 显存/带宽更重；**部署难度**上 LSS 算子常规（量化友好），BEVFormer 的 deformable attention 需定制算子/Plugin（Q4），车载量产 LSS/BEVDepth 路线更多。

### Q65. 基于 Diffusion 的 BEV 与 BEVFormer 的区别；BEV 三代演进（理想-云-二面 3）

演进脉络：一代 **LSS**（显式深度+前投影视角）、二代 **BEVFormer**（query 采样、时序融合、无显式深度）、三代 **diffusion-based / occupancy**（生成式建 BEV 或 Occ 特征，把感知从判别式转为"条件生成"）。区别：BEVFormer 是判别式一次前向（deterministic，输出单张 BEV）；diffusion BEV 把 BEV 生成建模为**迭代去噪**（从噪声出发多步 refine，以多相机图像为条件），优势：天然建模**不确定性/多假设**（可采样多个未来或多种布局）、生成精细结构（occ/长尾）；代价：多步去噪推理慢（几十步），要靠蒸馏/少步采样（一致性模型）落地——正是 [05 篇 Q8](./05-推理引擎与优化.md) Diffusion 推理优化的用武之地。

### Q66. 常见多目标跟踪算法及特点（经纬恒润 二面 1）

**Tracking-by-Detection**：SORT（卡尔曼 + IoU 匹配，快但无外观）、**DeepSORT**（+ReID 外观特征，级联匹配）、**ByteTrack**（低分框二次关联，高频考点：高置信度过卡尔文预测后，低分框再救回遮挡目标，零成本涨点）。**联合检测嵌入 JDE**：FairMOT（anchor 中心同时出检测+embedding，均衡两个任务）。**Transformer 端到端**：TrackFormer/MOTR（track query 跨帧自回归传播，免手工关联，训练数据要求高）。工程：ID 开关 vs 漏检的 trade-off、轨迹管理（出生/死亡/平滑）、车载上与卡尔曼滤波（Q67）组合做状态估计。

### Q67. 卡尔曼滤波的基本原理与应用场景（经纬恒润 二面 2）

线性高斯系统的**最优递推贝叶斯估计**：两步循环——①**预测**：x̂ₖ=A x̂ₖ₋₁，P=APAᵀ+Q（运动模型外推）；②**更新**：K=P Hᵀ(HPHᵀ+R)⁻¹，x̂=x̂+K(z−Hx̂)，P=(I−KH)P（用观测 z 修正，K 是预测与观测的可信度加权）。直观：滤波值 = 预测与观测按方差的加权平均，方差越小越信谁。应用：SORT 系跟踪的 bbox/状态平滑（恒速模型）、IMU/GPS 融合定位（EKF：非线性系统先雅可比线性化）、雷达点迹/激光配准。面试常追问 EKF/UKF 何时用（非线性强弱）与 Q/R 调参含义（过程/观测噪声）。

### Q68. Transformer 的整体结构（辉羲-三面 14）

输入嵌入（token + 位置编码 RoPE/learned）→ L×[多头自注意力（QK^T/√d softmax 加权 V，多头并行拼接）→ 残差+LayerNorm（Pre-LN 现代主流）；FFN（两层 MLP，中间 4d，GELU/SwiGLU）→ 残差+LN] → 输出头。decoder 加 causal mask + cross-attention。细节（多头维度、KV Cache、GQA/MLA 变体）见[04 篇](./04-attention与kvcache.md)；手写实现 `solutions/python/src/answers/attention/`（MHA/GQA/MLA）。

### Q69. Softmax 在深度学习中的典型应用（辉羲-三面 15）

分类输出归一化（概率化 + 交叉熵，见[11 篇 Q11](./11-深度学习与训练.md)）；**attention 权重**（QK^T 后归一化成权重——Transformer 核心）；知识蒸馏的软标签；MoE router 的 top-k 选择概率；强化学习策略输出（概率分布采样）；深度估计的离散分布（LSS，Q64）；扩散模型噪声预测参数化。共性：把任意实值向量变成合法分布；数值稳定版（减 max）与 online softmax 手写见[02 篇 Q2](./02-cuda-编程题.md)。
