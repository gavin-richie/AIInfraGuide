"""训练/推理显存占用估算器。

来源: 中兴-AI-Infra-二面（基础知识 4/5: 参数量 → 优化器状态/梯度/参数显存;
16 卡训 200B 的方案设计）。KV Cache 公式与 attention/gqa.py::kv_cache_size 一致。

口径（混合精度 + Adam，业界标准 16N 法则）:
  参数 fp16 副本   2N bytes   （前向/反向计算用）
  梯度 fp16        2N bytes
  优化器状态      12N bytes = fp32 master 权重 4N + Adam m 4N + Adam v 4N
  → 单卡合计 16N（DDP，不分片）。纯 fp32 训练同为 16N（4N 参数 + 4N 梯度 + 8N m/v）。
ZeRO-1/2/3 分别把优化器状态/梯度/参数在数据并行组内切分（见 07 篇 Q3 的对照表）。
"""
from __future__ import annotations

_GB = 10 ** 9          # 面试口径用十进制 GB: "7B fp16+Adam → 112GB" 即 16N

# 每 token 每层激活字节数 ≈ 34*h（fp16 口径, Korthikanti et al. 2022 / Megatron 统计）;
# activation checkpointing 只保留每层输入, 降到 2*h 每 token 每层。
ACT_COEF_FULL = 34
ACT_COEF_CKPT = 2


def training_memory(
    n_params: float,
    *,
    optimizer: str = "adam",
    mixed_precision: bool = True,
    zero_stage: int = 0,
    world_size: int = 1,
    act_batch: int = 0,
    act_seq: int = 0,
    act_hidden: int = 0,
    act_layers: int = 0,
    act_checkpointing: bool = False,
) -> dict[str, float]:
    """估算单卡显存（GB, 十进制）。n_params 单位为个（如 7e9）。

    >>> m = training_memory(7e9)   # 7B fp16+Adam DDP → 16N = 112 GB
    >>> round(m["total_gb"])
    112
    """
    if optimizer not in ("adam", "sgd"):
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    pbytes = 2 if mixed_precision else 4          # 参数/梯度的元素宽度
    params = pbytes * n_params
    grads = pbytes * n_params
    if optimizer == "adam":
        optim = 12 * n_params if mixed_precision else 8 * n_params
        # fp16 混合精度需 fp32 master(4N)+m(4N)+v(4N)=12N; 纯 fp32 参数即 master, 只存 m+v=8N
    else:
        optim = 4 * n_params                      # SGD+momentum 只有一份动量缓冲

    d = max(world_size, 1)                        # ZeRO 切分的 DP 组大小
    keep = {0: [params, grads, optim],
            1: [params, grads, optim / d],
            2: [params, grads / d, optim / d],
            3: [params / d, grads / d, optim / d]}[zero_stage]

    act = 0.0
    if act_batch and act_seq and act_hidden and act_layers:
        coef = ACT_COEF_CKPT if act_checkpointing else ACT_COEF_FULL
        # 34*h 已按 fp16 元素计; 若用其他宽度按 pbytes/2 比例折算
        act = act_batch * act_seq * act_layers * coef * act_hidden * (pbytes / 2)

    return {
        "params_gb": keep[0] / _GB,
        "grads_gb": keep[1] / _GB,
        "optimizer_gb": keep[2] / _GB,
        "activations_gb": act / _GB,
        "total_gb": (sum(keep) + act) / _GB,
    }


def kv_cache_bytes(
    layers: int, kv_heads: int, head_dim: int,
    batch: int, seq_len: int, dtype_bytes: int = 2,
) -> int:
    """KV Cache 字节数 = 2(K 和 V) × L × n_kv × d_h × B × S × dtype。

    与 answers/attention/gqa.py::kv_cache_size 同公式（04 篇 Q7 的例:
    Qwen-7B, 32 层 32 头 d=128, B=1 S=2048 fp16 → 1.07 GiB）。
    """
    return 2 * layers * kv_heads * head_dim * batch * seq_len * dtype_bytes
