// 双线性插值测试: 整点还原 / 线性 / 中心插值
#include "answers/vision.hpp"
#include "answers/testing.h"

int main() {
    using answers::bilinear_sample;
    // 2x2 图: 10 20
    //         30 40
    std::vector<float> img{10, 20, 30, 40};
    // 整数坐标必须精确还原原像素
    EXPECT_NEAR(bilinear_sample(img, 2, 2, 0, 0), 10, 1e-6);
    EXPECT_NEAR(bilinear_sample(img, 2, 2, 0, 1), 20, 1e-6);
    EXPECT_NEAR(bilinear_sample(img, 2, 2, 1, 0), 30, 1e-6);
    EXPECT_NEAR(bilinear_sample(img, 2, 2, 1, 1), 40, 1e-6);
    // 中心 (0.5, 0.5) = 四点均值
    EXPECT_NEAR(bilinear_sample(img, 2, 2, 0.5f, 0.5f), 25.0, 1e-5);
    // 边中点: (0, 0.5) = 10 与 20 的线性中点
    EXPECT_NEAR(bilinear_sample(img, 2, 2, 0.0f, 0.5f), 15.0, 1e-5);
    // 越界 clamp 回边缘
    EXPECT_NEAR(bilinear_sample(img, 2, 2, -3.0f, -3.0f), 10, 1e-6);
    EXPECT_NEAR(bilinear_sample(img, 2, 2, 9.0f, 9.0f), 40, 1e-6);
    RETURN_TEST_MAIN();
}
