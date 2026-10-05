// LayerNorm / RMSNorm —— 快手(CUDA 写 Norm 算子)、百度(RMSNorm 为何快)、综合-6
#include <cuda_runtime.h>
#include <math.h>

namespace answers {

// 一行一个 block, 两遍归约(mean → var) + 写回, 结构同 softmax_rowblock
__global__ void layernorm_kernel(const float *x, const float *w, const float *b,
                                 float *y, int cols, float eps) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *rx = x + (size_t)row * cols;
    float *ry = y + (size_t)row * cols;

    // mean
    float partial = 0.f;
    for (int i = tid; i < cols; i += blockDim.x) partial += rx[i];
    sdata[tid] = partial;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / cols;
    __syncthreads();

    // variance
    partial = 0.f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float d = rx[i] - mean;
        partial += d * d;
    }
    sdata[tid] = partial;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rstd = rsqrtf(sdata[0] / cols + eps);

    for (int i = tid; i < cols; i += blockDim.x)
        ry[i] = (rx[i] - mean) * rstd * w[i] + b[i];
}

void layernorm(const float *x, const float *w, const float *b, float *y,
               int rows, int cols, float eps) {
    layernorm_kernel<<<rows, 1024, 1024 * sizeof(float)>>>(x, w, b, y, cols, eps);
}

// RMSNorm: 少一次 mean 归约与减法 —— "为何比 LayerNorm 快"的代码级答案
__global__ void rmsnorm_kernel(const float *x, const float *w, float *y,
                               int cols, float eps) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *rx = x + (size_t)row * cols;
    float *ry = y + (size_t)row * cols;

    float partial = 0.f;
    for (int i = tid; i < cols; i += blockDim.x) partial += rx[i] * rx[i];
    sdata[tid] = partial;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rstd = rsqrtf(sdata[0] / cols + eps);

    for (int i = tid; i < cols; i += blockDim.x)
        ry[i] = rx[i] * rstd * w[i];
}

void rmsnorm(const float *x, const float *w, float *y, int rows, int cols, float eps) {
    rmsnorm_kernel<<<rows, 1024, 1024 * sizeof(float)>>>(x, w, y, cols, eps);
}

}  // namespace answers
