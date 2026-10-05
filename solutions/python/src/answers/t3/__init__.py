"""T3 芯片/硬件厂面试题编程解答（英伟达/三星/摩尔线程/沐曦/中兴等）。

对应文档: docs/interview-answers/80 ~ 83。CUDA/C++ 题的完整代码内嵌在文档中，
本包只放可用 pytest 验证语义/行为的 Python 实现。
"""
from answers.t3.clock_angle import clock_angle, parse_clock
from answers.t3.first_one import first_one
from answers.t3.gpu_memory_estimator import (
    kv_cache_bytes,
    training_memory,
)
from answers.t3.minimal_transformer import MinimalTransformer
from answers.t3.outer_product import outer_product
from answers.t3.outlier_quantization import (
    quantize_minmax,
    quantize_outlier_split,
    quantize_percentile,
    quantize_per_channel,
    rmse,
)
from answers.t3.print_in_turn import run_print_in_turn
from answers.t3.two_sum import two_sum

__all__ = [
    "clock_angle",
    "parse_clock",
    "first_one",
    "kv_cache_bytes",
    "training_memory",
    "MinimalTransformer",
    "outer_product",
    "quantize_minmax",
    "quantize_percentile",
    "quantize_per_channel",
    "quantize_outlier_split",
    "rmse",
    "run_print_in_turn",
    "two_sum",
]
