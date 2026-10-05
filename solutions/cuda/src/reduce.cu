// Reduction 归约求和 —— 综合-6 CUDA 编程题 1、B站(多级归约)、腾讯(warp reduce)、
// 原粒(Reduce 计算顺序与数值精度)、小厂(百万浮点求和)
#include <cuda_runtime.h>

namespace answers {

// ============ 版本 1: 每线程原子加(演示"为什么慢": 串行化) ============
__global__ void reduce_atomic_kernel(const float *in, float *out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, in[i]);
}
void reduce_atomic(const float *d_in, float *d_out, size_t n) {
    // 调用前 d_out 需清零
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    reduce_atomic_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

// ============ 版本 2: 树归约(经典优化路径) ============
// 每个 block 归约自己的一段, 结果写 partial[blockIdx], 第二个 kernel 再归约
__global__ void reduce_block_kernel(const float *in, float *partial, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * (blockDim.x * 2) + tid;  // 每线程先取 2 个元素

    float v = 0.f;
    if (i < n) v = in[i];
    if (i + blockDim.x < n) v += in[i + blockDim.x];

    // block 内 shared memory 树归约, 对折到只剩 64 个元素(s=64 那轮做完)
    sdata[tid] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // 最后一 warp: 先并掉另一半, 再 warp shuffle 归约(免 sync/无 bank conflict)
    if (tid < 32) {
        float w = sdata[tid] + sdata[tid + 32];
        for (int offset = 16; offset > 0; offset >>= 1)
            w += __shfl_down_sync(0xffffffffu, w, offset);
        if (tid == 0) partial[blockIdx.x] = w;
    }
}

__global__ void reduce_final_kernel(const float *partial, float *out, int nblocks) {
    // 单 block 收尾: 网格跨步装载所有 partial(nblocks 可能 > blockDim)
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float v = 0.f;
    for (int i = tid; i < nblocks; i += blockDim.x) v += partial[i];
    sdata[tid] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out = sdata[0];
}

void reduce_two_stage(const float *d_in, float *d_out, size_t n) {
    int threads = 256;
    size_t elems_per_block = (size_t)threads * 2;
    int blocks = (int)((n + elems_per_block - 1) / elems_per_block);
    float *partial;
    cudaMalloc(&partial, blocks * sizeof(float));
    reduce_block_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, partial, n);
    int ft = 256;  // 收尾块
    reduce_final_kernel<<<1, ft, ft * sizeof(float)>>>(partial, d_out, blocks);
    cudaFree(partial);
}

// ============ 附: 向量加(冒烟用) ============
__global__ void vec_add_kernel(const float *a, const float *b, float *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
void vec_add(const float *a, const float *b, float *c, size_t n) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    vec_add_kernel<<<blocks, threads>>>(a, b, c, n);
}

}  // namespace answers
