// CUDA 手写算子面试题公共声明
// 覆盖题库: 综合-6 CUDA 编程题(Reduction/Softmax/转置/AvgPooling/IoU)、
// AI-Infra-校招-1(NCHW→NHWC)、综合-1(broadcast 乘)、蚂蚁(Histogram)、
// 美团-北斗(前缀和/GEMM)、沐曦(行 reduce)、快手(GEMM 优化)、百度(GEMM/向量加)等。
#pragma once

#include <cstddef>

namespace answers {

// ---------------- reduce.cu ----------------
// 版本1: 全局原子加(慢, 仅演示语义)
void reduce_atomic(const float *d_in, float *d_out, size_t n);
// 版本2: block 分块 + shared memory 树归约 + warp shuffle, 两级 kernel
void reduce_two_stage(const float *d_in, float *d_out, size_t n);

// ---------------- softmax.cu ----------------
// 数值稳定 softmax: 一行一个 block(块内先求 max 再求 sum, 全程驻留 shared)
void softmax_rowblock(const float *d_in, float *d_out, int rows, int cols);
// Online softmax(单遍流式): 先在线求 (m,l), 再归一化 —— FA 的数学基础
void softmax_online(const float *d_in, float *d_out, int rows, int cols);

// ---------------- gemm.cu ----------------
void gemm_naive(const float *A, const float *B, float *C, int M, int N, int K);
void gemm_smem_tiled(const float *A, const float *B, float *C, int M, int N, int K);

// ---------------- transpose.cu ----------------
void transpose_naive(const float *in, float *out, int rows, int cols);
// shared memory + (+1) padding 消 bank conflict
void transpose_smem(const float *in, float *out, int rows, int cols);

// ---------------- layernorm.cu / rmsnorm ----------------
void layernorm(const float *x, const float *w, const float *b, float *y,
               int rows, int cols, float eps);
void rmsnorm(const float *x, const float *w, float *y, int rows, int cols, float eps);

// ---------------- layout.cu ----------------
void nchw_to_nhwc(const float *in, float *out, int N, int C, int H, int W);
// 4D 广播 elementwise 乘: out[i] = a[sa*i] * b[sb*i], 广播维 stride=0
void broadcast_mul_4d(const float *a, const float *b, float *out,
                      int on, int oc, int oh, int ow,
                      long sa_n, long sa_c, long sa_h, long sa_w,
                      long sb_n, long sb_c, long sb_h, long sb_w);

// ---------------- histogram.cu ----------------
// 版本1: global atomicAdd(所有线程抢同一组计数器)
void histogram_global(const unsigned char *in, unsigned int *out_hist, size_t n);
// 版本2: shared memory 私有直方图, block 级合并(快)
void histogram_smem(const unsigned char *in, unsigned int *out_hist, size_t n);

// ---------------- scan.cu ----------------
// 前缀和: block 内 Hillis-Steele 扫描 + block 间chained(单 kernel 简化:
// 假定 n <= blockDim 内可容纳, 超出用多 block 版本思路见解答文档)
void prefix_sum_block(const float *in, float *out, size_t n);  // inclusive scan

// ---------------- pooling_iou.cu ----------------
void avg_pool_2d(const float *in, float *out, int H, int W, int ksize, int stride);
// M 个框两两 IoU 矩阵: out[i*M+j]
void iou_pairwise(const float *boxes, float *out, int M);

// ---------------- row_reduce.cu ----------------
// 每行求和: 一行一个 warp(warp shuffle 归约)
void row_sum_warp(const float *in, float *out, int rows, int cols);

// ---------------- reduce.cu 附: 向量加(冒烟) ----------------
void vec_add(const float *a, const float *b, float *c, size_t n);

}  // namespace answers
