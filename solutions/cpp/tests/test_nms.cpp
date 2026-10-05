// NMS/IoU 测试
#include "answers/vision.hpp"
#include "answers/testing.h"

int main() {
    using answers::Box;
    using answers::iou;
    using answers::nms;

    {  // IoU 基础
        Box a{0, 0, 10, 10};
        EXPECT_NEAR(iou(a, a), 1.0, 1e-6);            // 自身 IoU=1
        EXPECT_NEAR(iou(a, Box{1, 1, 11, 11}), 81.0 / 119.0, 1e-4);
        EXPECT_NEAR(iou(a, Box{20, 20, 30, 30}), 0.0, 1e-6);  // 相离
        EXPECT_NEAR(iou(a, Box{5, 5, 15, 15}), 25.0 / 175.0, 1e-4);
        EXPECT_NEAR(iou(Box{0, 0, 0, 0}, Box{0, 0, 0, 0}), 0.0, 1e-6);  // 零面积安全
    }
    {  // NMS 抑制与保留
        std::vector<Box> boxes{
            {0, 0, 10, 10},
            {1, 1, 11, 11},     // 与 box0 IoU≈0.68 > 0.5 → 被抑制
            {20, 20, 30, 30},   // 独立 → 保留
            {50, 50, 60, 60},
        };
        std::vector<float> scores{0.9f, 0.8f, 0.7f, 0.6f};
        auto keep = nms(boxes, scores, 0.5f);
        EXPECT(keep.size() == 3);
        EXPECT(keep[0] == 0 && keep[1] == 2 && keep[2] == 3);
    }
    {  // 低分框先出不能抑制高分框(顺序独立性)
        // IoU = 81/119 ≈ 0.68 > 0.5 → 只留 0.9 的框
        std::vector<Box> boxes{{0, 0, 10, 10}, {1, 1, 11, 11}};
        std::vector<float> scores{0.5f, 0.9f};
        auto keep = nms(boxes, scores, 0.5f);
        EXPECT(keep.size() == 1 && keep[0] == 1);
    }
    RETURN_TEST_MAIN();
}
