// Histogram(蚂蚁: 原子 vs 共享内存归约) + 前缀和 scan(美团-北斗、腾讯)
#include <cuda_runtime.h>

namespace answers {

// ============ 版本1: 全局原子直方图(所有 block 抢 256 个全局计数器) ============
__global__ void histogram_global_kernel(const unsigned char *in, unsigned int *hist, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[in[i]], 1u);
}

void histogram_global(const unsigned char *in, unsigned int *out_hist, size_t n) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    histogram_global_kernel<<<blocks, threads>>>(in, out_hist, n);
}

// ============ 版本2: block 私有 smem 直方图 → 合并进全局 ============
// 冲突域从"全 grid"缩小到"单 block", 高计数场景快数倍
#define HIST_BINS 256
__global__ void histogram_smem_kernel(const unsigned char *in, unsigned int *hist, size_t n) {
    __shared__ unsigned int local[HIST_BINS];
    int tid = threadIdx.x;
    for (int b = tid; b < HIST_BINS; b += blockDim.x) local[b] = 0;
    __syncthreads();

    size_t i0 = blockIdx.x * (size_t)blockDim.x * 4 + tid;  // 每线程 4 元素(I收益增益)
    for (int k = 0; k < 4; ++k) {
        size_t i = i0 + (size_t)k * blockDim.x;
        if (i < n) atomicAdd(&local[in[i]], 1u);
    }
    __syncthreads();
    // 合并: 每线程认领若干 bin, 与其他 block 在全局层原子累加
    for (int b = tid; b < HIST_BINS; b += blockDim.x)
        atomicAdd(&hist[b], local[b]);
}

void histogram_smem(const unsigned char *in, unsigned int *out_hist, size_t n) {
    int threads = 256;
    size_t per_block = (size_t)threads * 4;
    int blocks = (int)((n + per_block - 1) / per_block);
    histogram_smem_kernel<<<blocks, threads>>>(in, out_hist, n);
}

// ============ 前缀和: block 内 Hillis-Steele inclusive scan ============
// 适用 n <= blockDim; 多 block 版需 block 间偏移二次 kernel(思路见文档 02)
__global__ void scan_kernel(const float *in, float *out, size_t n) {
    extern __shared__ float temp[];
    size_t tid = threadIdx.x;
    size_t offset = blockIdx.x * blockDim.x;
    size_t i = offset + tid;

    temp[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();

    // Hillis-Steele: 每轮跨度翻倍, log2(n) 轮, 总工作量 2n(有冗余但全并行)
    for (size_t d = 1; d < blockDim.x; d <<= 1) {
        float add = 0.f;
        if (tid >= d) add = temp[tid - d];
        __syncthreads();
        temp[tid] += add;
        __syncthreads();
    }
    if (i < n) out[i] = temp[tid];
}

void prefix_sum_block(const float *in, float *out, size_t n) {
    int threads = 1024;
    int blocks = (int)((n + threads - 1) / threads);
    // 注意: 多 block 时各 block 独立扫描(块间不接续), 此处语义为"块内前缀和";
    // 全局前缀和需要第二个 kernel 把前缀块总和加回(见文档 02 讲解)。
    scan_kernel<<<blocks, threads, threads * sizeof(float)>>>(in, out, n);
}

}  // namespace answers
