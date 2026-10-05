#include "answers/vision.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace answers {

float iou(const Box &a, const Box &b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> nms(const std::vector<Box> &boxes, const std::vector<float> &scores,
                     float threshold) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int i, int j) { return scores[i] > scores[j]; });

    std::vector<int> keep;
    while (!order.empty()) {
        int top = order.front();
        keep.push_back(top);
        // 删除与 top 的 IoU 超阈值的框
        order.erase(std::remove_if(order.begin() + 1, order.end(),
                                   [&](int j) { return iou(boxes[top], boxes[j]) > threshold; }),
                    order.end());
        order.erase(order.begin());  // 移除 top 自身
    }
    return keep;
}

std::vector<float> conv2d(const std::vector<float> &in, int H, int W,
                          const std::vector<float> &kernel, int KH, int KW) {
    int OH = H - KH + 1;
    int OW = W - KW + 1;
    std::vector<float> out(static_cast<size_t>(OH) * OW, 0.0f);
    for (int oy = 0; oy < OH; ++oy) {
        for (int ox = 0; ox < OW; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < KH; ++ky) {
                for (int kx = 0; kx < KW; ++kx) {
                    acc += in[(oy + ky) * W + (ox + kx)] * kernel[ky * KW + kx];
                }
            }
            out[oy * OW + ox] = acc;
        }
    }
    return out;
}

float bilinear_sample(const std::vector<float> &img, int H, int W, float y, float x) {
    // clamp 到图像范围(对齐 OpenCV BORDER 恒等边缘的行为, 亦可改 -1 填 0)
    y = std::clamp(y, 0.0f, static_cast<float>(H - 1));
    x = std::clamp(x, 0.0f, static_cast<float>(W - 1));
    int y0 = static_cast<int>(std::floor(y));
    int x0 = static_cast<int>(std::floor(x));
    int y1 = std::min(y0 + 1, H - 1);
    int x1 = std::min(x0 + 1, W - 1);
    float dy = y - y0;
    float dx = x - x0;
    // 水平两次插值 + 垂直一次: f = (1-dy)(1-dx)f00 + (1-dy)dx f01 + dy(1-dx)f10 + dy dx f11
    float top = img[y0 * W + x0] * (1 - dx) + img[y0 * W + x1] * dx;
    float bot = img[y1 * W + x0] * (1 - dx) + img[y1 * W + x1] * dx;
    return top * (1 - dy) + bot * dy;
}

}  // namespace answers
