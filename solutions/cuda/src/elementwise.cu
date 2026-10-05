// 逐元素类 kernel —— 各批次文档内嵌片段的工程化版本
// 覆盖题库: 50-百度 Q17(grid-stride 向量加)、81-t3 Q13(遂原 SiLU)、
// 80-t3 #11(英伟达 向量外积)。
#include <cuda_runtime.h>
#include <math.h>

namespace answers {

// ============ grid-stride 向量加: 线程数固定, 循环跨步覆盖任意 n ============
// 与 reduce.cu 的 bound-check 版对比: grid 不必 cover 全部元素,
// launch 开销与 occupancy 解耦, 大 n 下更稳。
__global__ void vec_add_gs_kernel(const float *a, const float *b, float *c, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) c[i] = a[i] + b[i];
}

void vec_add_grid_stride(const float *a, const float *b, float *c, size_t n) {
    int threads = 256;
    size_t want = (n + threads - 1) / threads;
    int blocks = (int)(want < 4096 ? (want ? want : 1) : 4096);
    vec_add_gs_kernel<<<blocks, threads>>>(a, b, c, n);
}

// ============ SiLU(x) = x * sigmoid(x) = x / (1 + e^-x) ============
// 纯文本面试最容易漏的边界: i < n 越界保护。
// expf 为精确版; 追求吞吐可换 __expf(UAFP 快速近似, 误差 ~1e-6 相对量级)。
__global__ void silu_kernel(const float *x, float *y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / (1.0f + expf(-x[i]));
}

void silu(const float *x, float *y, size_t n) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    silu_kernel<<<blocks, threads>>>(x, y, n);
}

// ============ 向量外积: C(MxN) = a(M) ⊗ b(N), 行优先 ============
// 二维 block/grid: y 维对应行(读 a), x 维对应列(读 b)。
__global__ void outer_product_kernel(const float *a, const float *b,
                                     float *c, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 行 -> a[i]
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 列 -> b[j]
    if (i < M && j < N) c[(size_t)i * N + j] = a[i] * b[j];
}

void outer_product(const float *a, const float *b, float *c, int M, int N) {
    dim3 block(32, 8);
    dim3 grid((N + 31) / 32, (M + 7) / 8);
    outer_product_kernel<<<grid, block>>>(a, b, c, M, N);
}

}  // namespace answers
