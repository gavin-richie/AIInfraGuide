"""KV Cache 显存占用计算器（MHA / GQA / MLA，FP16/FP8/INT8/INT4）。

对应面试题:
    - 科大讯飞-飞星 "如何计算使用 FP8 存储时的 KV Cache 大小？"
    - 荣耀-校招-二面 "量化后的大模型（INT8/INT4）运行时内存占用约为多少"
    - 联想 "有哪些常见的 KV Cache 优化方法"（量化分支的定量部分）

公式:
    MHA/GQA:  bytes = 2(K和V) × n_layer × n_kv_head × d_head × S × B × bytes_per_elem
    MLA:      bytes = n_layer × (d_c_kv + d_rope) × S × B × bytes_per_elem
              （只缓存潜在向量 c_kv 与解耦的 RoPE K，d_head/n_head 维度被"吸收"掉）
"""
from __future__ import annotations

from dataclasses import dataclass

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "e4m3": 1, "e5m2": 1, "int8": 1, "int4": 0.5}


def kv_cache_bytes(n_layer: int, n_kv_head: int, d_head: int, seq_len: int,
                   batch: int = 1, dtype: str = "fp16") -> int:
    """MHA/GQA 的 KV Cache 字节数（GQA 体现在 n_kv_head < n_head）。"""
    return int(2 * n_layer * n_kv_head * d_head * seq_len * batch * DTYPE_BYTES[dtype])


def mla_kv_bytes(n_layer: int, d_c_kv: int = 512, d_rope: int = 64,
                 seq_len: int = 1, batch: int = 1, dtype: str = "fp16") -> int:
    """MLA 的 KV Cache 字节数（DeepSeek-V2/V3: d_c_kv=512, d_rope=64）。"""
    return int(n_layer * (d_c_kv + d_rope) * seq_len * batch * DTYPE_BYTES[dtype])


@dataclass
class ModelKVProfile:
    """常见模型的 KV Cache 剖析（面试速算用）。"""

    name: str
    n_layer: int
    n_kv_head: int
    d_head: int

    def per_token_kb(self, dtype: str = "fp16", mla: bool = False) -> float:
        if mla:
            return mla_kv_bytes(self.n_layer, seq_len=1, dtype=dtype) / 1024
        return kv_cache_bytes(self.n_layer, self.n_kv_head, self.d_head,
                              seq_len=1, dtype=dtype) / 1024

    def total_gb(self, seq_len: int, batch: int = 1, dtype: str = "fp16",
                 mla: bool = False) -> float:
        if mla:
            return mla_kv_bytes(self.n_layer, seq_len=seq_len, batch=batch, dtype=dtype) / 1024**3
        return kv_cache_bytes(self.n_layer, self.n_kv_head, self.d_head,
                              seq_len=seq_len, batch=batch, dtype=dtype) / 1024**3


# 常见模型速查表（n_kv_head 已按 GQA 计）
PROFILES = {
    "llama2-7b": ModelKVProfile("Llama-2-7B (MHA)", 32, 32, 128),
    "llama3-8b": ModelKVProfile("Llama-3-8B (GQA-8)", 32, 8, 128),
    "qwen2.5-7b": ModelKVProfile("Qwen2.5-7B (GQA-4)", 28, 4, 128),
    "deepseek-v3": ModelKVProfile("DeepSeek-V3 (MLA)", 61, 1, 576),  # n_kv_head/d_head 仅占位
}


def quantized_model_footprint(n_params_b: float, weight_dtype: str = "int8",
                              kv_gb: float = 0.0, overhead_ratio: float = 0.15) -> float:
    """量化模型运行时显存估算: 权重 + KV + 运行开销(激活/CUDA context, ~10-20%)。

    对应荣耀-校招-二面场景题：32B INT8 ≈ 32GB 权重 + KV + 开销 → 单卡放不下，
    需 TP=2（或 W4A16: 16GB 单卡勉强）。
    """
    weight_gb = n_params_b * DTYPE_BYTES[weight_dtype]
    return weight_gb + kv_gb + (weight_gb + kv_gb) * overhead_ratio


if __name__ == "__main__":
    for name, p in PROFILES.items():
        mla = name == "deepseek-v3"
        print(f"{p.name}: {p.per_token_kb('fp16', mla=mla):.1f} KB/token/seq, "
              f"4096 tokens 单序列 = {p.total_gb(4096, dtype='fp16', mla=mla):.2f} GB (fp16)")
    print(f"DeepSeek-V3 MLA fp8: {mla_kv_bytes(61, seq_len=4096, dtype='fp8') / 1024**2:.0f} MB / 4096 tokens")
    print(f"32B INT8 运行时估算: {quantized_model_footprint(32, 'int8'):.1f} GB (TP=2 每卡)")
