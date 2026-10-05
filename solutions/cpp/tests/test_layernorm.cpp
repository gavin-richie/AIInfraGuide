// LayerNorm 测试: 均值/方差归一性质 + 与手算对拍
#include "answers/layernorm.hpp"
#include "answers/testing.h"

#include <cmath>

int main() {
    using answers::layer_norm;
    {  // 归一后 w=1,b=0 → 均值≈0, 方差≈1
        std::vector<float> x{1, 2, 3, 4, 5};
        std::vector<float> w(x.size(), 1.0f), b(x.size(), 0.0f);
        auto y = layer_norm(x, w, b, 1e-5f);
        float mean = 0;
        for (float v : y) mean += v;
        mean /= y.size();
        EXPECT_NEAR(mean, 0.0, 1e-5);
        float var = 0;
        for (float v : y) var += (v - mean) * (v - mean);
        var /= y.size();
        EXPECT_NEAR(var, 1.0, 1e-3);  // eps 会让它略小于 1
    }
    {  // 手算: x={0,2}, mean=1, var=1 → y={-1,1}(w=1,b=0)
        std::vector<float> x{0, 2};
        std::vector<float> w{1, 1}, b{0, 0};
        auto y = layer_norm(x, w, b, 0.0f);
        EXPECT_NEAR(y[0], -1.0, 1e-5);
        EXPECT_NEAR(y[1], 1.0, 1e-5);
    }
    {  // affine: w=2, b=1 → y={-1,3}
        std::vector<float> x{0, 2};
        std::vector<float> w{2, 2}, b{1, 1};
        auto y = layer_norm(x, w, b, 0.0f);
        EXPECT_NEAR(y[0], -1.0, 1e-5);
        EXPECT_NEAR(y[1], 3.0, 1e-5);
    }
    RETURN_TEST_MAIN();
}
