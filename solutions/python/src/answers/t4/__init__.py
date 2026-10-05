"""T4 车企/自动驾驶面经专属题解代码。

模块:
    concat4d            -- 四维张量 Concat(逐维索引推导 + 向量化 + Triton last-dim kernel)
    graph_memory        -- 计算图运行最小内存(liveness 分析 + 贪心调度 + 暴力对拍)
    rect_overlap        -- 任意朝向两矩形重叠面积(Sutherland-Hodgman 凸多边形裁剪)
    kth_largest_uint8   -- uint8 数组第 K 大(256 桶计数, CUDA kernel 思路的 host 对拍版)
    basics              -- 矩阵旋转 / 两数之和 / 左上到右下最短路径 DP

其余 T4 编程题(NMS+IoU/LRU/拓扑排序/归并排序/岛屿数量/背包/反转链表/
单链表冒泡排序/二叉树右视图/reduce_sum/softmax/GEMM/转置/循环缓冲区)
已收录于 answers/algorithms、answers/torch_impl 与 solutions/cpp、solutions/cuda 工程。
"""
