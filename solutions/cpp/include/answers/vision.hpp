// 计算机视觉手写题: NMS/IoU、Conv2D、双线性插值
// (综合-6 C++/CUDA 编程题、蔚来、旷视、海康)
#pragma once

#include <cstddef>
#include <vector>

namespace answers {

struct Box {
    float x1, y1, x2, y2;  // 左上/右下
};

// IoU = 交 / 并; 相离返回 0
float iou(const Box &a, const Box &b);

// NMS: score 降序贪心保留, 抑制 IoU > threshold 的框; 返回保留下标
std::vector<int> nms(const std::vector<Box> &boxes, const std::vector<float> &scores,
                     float threshold);

// 2D 卷积(NCHW, 无 padding, stride=1, 单通道) —— 手写版
// in: (H, W), kernel: (KH, KW); out: (H-KH+1, W-KW+1)
std::vector<float> conv2d(const std::vector<float> &in, int H, int W,
                          const std::vector<float> &kernel, int KH, int KW);

// 双线性插值: 在 (H,W) 图上采样 (x, y)(浮点, 像素中心对齐 origin)
float bilinear_sample(const std::vector<float> &img, int H, int W, float y, float x);

}  // namespace answers
