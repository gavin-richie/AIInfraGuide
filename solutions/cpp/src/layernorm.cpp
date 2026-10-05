#include "answers/layernorm.hpp"

namespace answers {

std::vector<float> layer_norm(const std::vector<float> &x, const std::vector<float> &w,
                              const std::vector<float> &b, float eps) {
    const size_t n = x.size();
    std::vector<float> y(n);

    // 单遍数值稳定写法: Welford 更新 mean/M2, 比两遍少一次扫描
    float mean = 0.0f, m2 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float delta = x[i] - mean;
        mean += delta / static_cast<float>(i + 1);
        m2 += delta * (x[i] - mean);
    }
    float var = m2 / static_cast<float>(n);  // 总体方差(与 PyTorch LN 一致, 非 n-1)
    float rstd = 1.0f / std::sqrt(var + eps);

    for (size_t i = 0; i < n; ++i) {
        y[i] = (x[i] - mean) * rstd * w[i] + b[i];
    }
    return y;
}

}  // namespace answers
