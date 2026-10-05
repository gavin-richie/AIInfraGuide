"""大模型推理显存估算：权重 / KV Cache / 激活，含 MLA 与 GQA 的换算。

来源:
- 京东-AI-Infra-实习 Q7（估算 Qwen3-8B 推理显存）
- 快手-AI-Infra-实习-一面-1 Q4（Qwen 部署占用多少显存）
- 华为-AI-Infra-实习-2 Q2（KV Cache 大小如何计算）
- 蚂蚁-AI-Infra-实习-三面 Q2（batch/seq/head/dim/层数/dtype 全因子）
- 快手-AI-Infra-实习-一面-3 Q2（MLA 与 GQA 的数据对应关系、等价 head 数）

公式:
    KV bytes = 2(K和V) × L(层) × H_kv(KV头数,GQA下小于Q头数) × d_head × b_dtype × B × S
    权重 bytes = 参数量 × 位宽 / 8
    MLA 每 token 每层只缓存 latent: (kv_lora_rank + qk_rope_dim) 个元素
    （MoE/长上下文另有 experts 权重与激活峰值，本模块给主项 + 余量系数）
"""

import math

# Qwen3-8B 官方配置（Qwen3-8B 技术报告 / config.json）
QWEN3_8B = {
    "params": 8.2e9,
    "num_layers": 36,
    "num_q_heads": 32,
    "num_kv_heads": 8,  # GQA 4:1
    "head_dim": 128,
    "hidden": 4096,
    "context": 32768,
}

# DeepSeek-V2/V3 的 MLA 配置
DEEPSEEK_MLA = {
    "kv_lora_rank": 512,   # 压缩后的 latent 维度 c_kv
    "rope_dim": 64,        # qk_rope_head_dim，与 latent 分离缓存
    "head_dim": 128,       # v_head_dim / qk_nope_head_dim
}


def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    tokens: int,
    dtype_bytes: int = 2,
) -> int:
    """单请求（或 batch 展平后的总 token 数）的 KV Cache 字节数。"""
    return 2 * num_layers * num_kv_heads * head_dim * tokens * dtype_bytes


def weights_bytes(num_params: float, bits_per_weight: float = 16) -> int:
    """权重量化位宽可取 16 / 8 / 4（W4 还有 group scale 的少量元数据，忽略）。"""
    return int(num_params * bits_per_weight / 8)


def inference_memory_report(
    config: dict,
    batch: int = 1,
    seq: int = 4096,
    dtype_bytes: int = 2,
    kv_dtype_bytes: int = 2,
    weight_bits: float = 16,
    activation_gb: float = 1.0,
    runtime_overhead_gb: float = 1.5,
) -> dict[str, float]:
    """推理总显存的分项估算（GB）。

    activation_gb: 中间激活/CUDA Graph/量化 workspace 等的经验余量；
    runtime_overhead_gb: 推理引擎上下文、通信缓冲（TP 时 NCCL 缓冲）等。
    """
    n_tokens = batch * seq
    w = weights_bytes(config["params"], weight_bits) / 1024**3
    kv = (
        kv_cache_bytes(
            config["num_layers"],
            config["num_kv_heads"],
            config["head_dim"],
            n_tokens,
            kv_dtype_bytes,
        )
        / 1024**3
    )
    total = w + kv + activation_gb + runtime_overhead_gb
    return {
        "weights_gb": round(w, 2),
        "kv_cache_gb": round(kv, 2),
        "activation_gb": activation_gb,
        "runtime_overhead_gb": runtime_overhead_gb,
        "total_gb": round(total, 2),
    }


def mla_kv_per_token_per_layer(
    kv_lora_rank: int = DEEPSEEK_MLA["kv_lora_rank"],
    rope_dim: int = DEEPSEEK_MLA["rope_dim"],
    dtype_bytes: int = 1,
) -> int:
    """MLA 每 token 每层缓存的字节数：c_kv(latent) + k_rope 两段。

    DeepSeek-V3: (512 + 64) 元素，FP8 下 576 B/token/layer。
    """
    return (kv_lora_rank + rope_dim) * dtype_bytes


def mla_equivalent_gqa_heads(
    kv_lora_rank: int = DEEPSEEK_MLA["kv_lora_rank"],
    rope_dim: int = DEEPSEEK_MLA["rope_dim"],
    gqa_head_dim: int = DEEPSEEK_MLA["head_dim"],
) -> float:
    """MLA 等价于多少个 GQA KV head（KV Cache 体积口径）。

    GQA 每个 KV head 每 token 缓存 2*d（K 与 V），MLA 缓存 kv_lora_rank+rope_dim，
    令两者相等：n_gqa = (kv_lora_rank + rope_dim) / (2*d)。
    DeepSeek-V3: (512+64)/(2*128) = 2.25 个 head，对比 MHA 的 128 head，
    KV Cache 缩到 1/56.9 —— 这就是"MLA ≈ 极端 GQA"的直觉。
    """
    return (kv_lora_rank + rope_dim) / (2 * gqa_head_dim)


def mla_vs_mha_compression(
    num_q_heads: int = 128,
    head_dim: int = DEEPSEEK_MLA["head_dim"],
    kv_lora_rank: int = DEEPSEEK_MLA["kv_lora_rank"],
    rope_dim: int = DEEPSEEK_MLA["rope_dim"],
) -> float:
    """同规模 MHA 与 MLA 的 KV Cache 压缩比（MHA 每 token 每层 2*n_heads*d）。"""
    mha_bytes = 2 * num_q_heads * head_dim
    mla_bytes = kv_lora_rank + rope_dim
    return mha_bytes / mla_bytes


def fit_max_concurrent_requests(
    gpu_memory_gb: float,
    config: dict,
    seq: int = 4096,
    weight_bits: float = 16,
    kv_dtype_bytes: int = 2,
    reserved_gb: float = 2.5,
) -> int:
    """给定显存反解最大并发请求数（vLLM gpu_memory_utilization 的核心逻辑）。"""
    kv_per_request = (
        kv_cache_bytes(
            config["num_layers"], config["num_kv_heads"], config["head_dim"], seq, kv_dtype_bytes
        )
        / 1024**3
    )
    budget = gpu_memory_gb - weights_bytes(config["params"], weight_bits) / 1024**3 - reserved_gb
    return max(0, math.floor(budget / kv_per_request))
