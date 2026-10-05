// Conv2D 测试: 手算对拍 + 全 1 核 = box sum
#include "answers/vision.hpp"
#include "answers/testing.h"

int main() {
    using answers::conv2d;
    {  // 3x3 输入 * 2x2 核, 手算验证
        // in: 1 2 3
        //     4 5 6
        //     7 8 9     kernel: 1 0
        //                       0 -1     → out[y][x] = in[y][x] - in[y+1][x+1]
        std::vector<float> in{1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<float> kern{1, 0, 0, -1};
        auto out = conv2d(in, 3, 3, kern, 2, 2);
        EXPECT(out.size() == 4);
        EXPECT_NEAR(out[0], 1 - 5, 1e-6);   // -4
        EXPECT_NEAR(out[1], 2 - 6, 1e-6);   // -4
        EXPECT_NEAR(out[2], 4 - 8, 1e-6);   // -4
        EXPECT_NEAR(out[3], 5 - 9, 1e-6);   // -4
    }
    {  // 全 1 的 3x3 核 = 3x3 框内求和(盒滤波)
        std::vector<float> in{
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        };
        std::vector<float> kern(9, 1.0f);
        auto out = conv2d(in, 4, 4, kern, 3, 3);
        EXPECT(out.size() == 4);
        for (float v : out) EXPECT_NEAR(v, 9.0, 1e-6);
    }
    {  // 恒等核(中心 1): 5x5 输入 → 3x3 输出等于中心像
        std::vector<float> in(25);
        for (int i = 0; i < 25; ++i) in[i] = static_cast<float>(i);
        std::vector<float> kern{0, 0, 0, 0, 1, 0, 0, 0, 0};
        auto out = conv2d(in, 5, 5, kern, 3, 3);
        // out[y][x] = in[y+1][x+1]
        EXPECT_NEAR(out[0], in[6], 1e-6);
        EXPECT_NEAR(out[3], in[11], 1e-6);
    }
    RETURN_TEST_MAIN();
}
