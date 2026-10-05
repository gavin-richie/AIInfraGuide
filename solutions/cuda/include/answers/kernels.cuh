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
// 寄存器 tile 版: BM=BN=64, 64 线程, 每线程 8x8 累加器(81-t3 Q1 寒武纪)
void gemm_regtile_8x8(const float *A, const float *B, float *C, int M, int N, int K);

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

// ---------------- elementwise.cu ----------------
// grid-stride 向量加(50-百度 Q17): 线程跨步循环覆盖任意 n
void vec_add_grid_stride(const float *a, const float *b, float *c, size_t n);
// SiLU 逐元素(81-t3 Q13 遂原): y = x / (1 + e^-x)
void silu(const float *x, float *y, size_t n);
// 向量外积(80-t3 #11 英伟达): C(MxN) = a(M) ⊗ b(N)
void outer_product(const float *a, const float *b, float *c, int M, int N);

// ---------------- concat4d.cu (93-t4 Q5 元戎) ----------------
// 通用版: 任意拼接维 dim; ins 为设备端指针数组, offs[K+1] 拼接维前缀和,
// dims 为输出四维尺寸, total 为输出元素总数; hist/dims/offs 均须已在设备端
void concat4d_generic(const float *const *ins, float *out,
                      const int *offs, const int *dims, int K, int dim, long total);
// 最后一维(W)特化: rows = N*C*H, out 每行 Wa+Wb
void concat4d_last(const float *a, const float *b, float *out,
                   int Wa, int Wb, long rows);

// ---------------- kth_largest.cu (93-t4 Q6 小鹏) ----------------
// kernel1: 全局直方图(smem 私有化 + grid-stride); hist 须先清零
void hist256_grid_stride(const unsigned char *data, long n, unsigned int *hist);
// kernel2: 单 block 反转包含性前缀和, 输出第 K 大(out 为设备端 1 字节)
void kth_from_hist(const unsigned int *hist, unsigned int K, unsigned char *out);

}  // namespace answers
