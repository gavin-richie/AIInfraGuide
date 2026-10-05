// LayerNorm(CPU 手写版) —— 综合-6 C++ 编程题、百度-实习-一面-3
// y = (x - mean) / sqrt(var + eps) * w + b
#pragma once

#include <cmath>
#include <vector>

namespace answers {

std::vector<float> layer_norm(const std::vector<float> &x, const std::vector<float> &w,
                              const std::vector<float> &b, float eps = 1e-5f);

}  // namespace answers
