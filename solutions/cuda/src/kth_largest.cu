// uint8 数组第 K 大 —— 93-t4 Q6(小鹏)文档内嵌 kernel 的工程化版本
// 两级 kernel: 全局直方图(smem 私有化) -> 单 block 反转包含性前缀和, 边界判定。
#include <cuda_runtime.h>

namespace answers {

// ============ kernel1: 全局直方图(smem 私有化, grid-stride 任意 n) ============
// 思路同 histogram_scan.cu, 但用 grid-stride 让 block 数与 n 解耦。
__global__ void hist256_gs_kernel(const unsigned char *data, long n, unsigned *hist) {
    __shared__ unsigned local[256];
    int tid = threadIdx.x;
    for (int b = tid; b < 256; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (long i = blockIdx.x * (long)blockDim.x + tid; i < n;
         i += (long)gridDim.x * blockDim.x)
        atomicAdd(&local[data[i]], 1u);
    __syncthreads();
    for (int b = tid; b < 256; b += blockDim.x) atomicAdd(&hist[b], local[b]);
}

void hist256_grid_stride(const unsigned char *data, long n, unsigned *hist) {
    int threads = 256;
    long want = (n + threads - 1) / threads;
    int blocks = (int)(want < 2048 ? (want ? want : 1) : 2048);
    hist256_gs_kernel<<<blocks, threads>>>(data, n, hist);
}

// ============ kernel2: 单 block(256 线程)反转直方图包含性前缀和 ============
// cum[i] = #{x >= 255-i}, 关于 i 单调不减;
// 第 K 大 = 最大值 v 使 #{x>=v} >= K = 255 - (最小满足 cum[i]>=K 的 i)。
__global__ void kth_from_hist_kernel(const unsigned *hist, unsigned K, unsigned char *out) {
    __shared__ unsigned cum[256];
    int tid = threadIdx.x;
    cum[tid] = hist[255 - tid];  // 反转: 下标 0 对应值 255
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {  // Hillis-Steele 包含性 scan
        unsigned v = (tid >= off) ? cum[tid] + cum[tid - off] : cum[tid];
        __syncthreads();
        cum[tid] = v;
        __syncthreads();
    }
    bool prev_small = (tid == 0) || (cum[tid - 1] < K);
    if (prev_small && cum[tid] >= K) out[0] = (unsigned char)(255 - tid);
}

void kth_from_hist(const unsigned *hist, unsigned K, unsigned char *out) {
    kth_from_hist_kernel<<<1, 256>>>(hist, K, out);
}

}  // namespace answers
