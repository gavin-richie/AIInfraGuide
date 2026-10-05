// GEMM —— 出现频率最高的 CUDA 手写题(综合-6/百度/快手/美团-北斗/商汤/英伟达)
// naive → shared memory tiling 两级实现, --bench 可看实测加速比。
#include <cuda_runtime.h>

namespace answers {

// ============ naive: 每线程算 C 的一个元素, A 行 B 列各扫 K ============
__global__ void gemm_naive_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    C[(size_t)row * N + col] = acc;
}

void gemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

// ============ shared memory tiling: 一个 block 算 C 的一个 16x16 tile ============
// 256 线程(16x16), 每线程一个输出; A/B tile 经 shared 复用 16 次,
// 全局访存量从 naive 的 2*M*N*K 降到 2*M*N*K/16。
// 进一步优化路线(文档 03): 寄存器 tile + float4 向量装载 + double buffering。
__global__ void gemm_smem_kernel(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
    constexpr int TM = 16, TK = 16;
    __shared__ float As[TM][TK];
    __shared__ float Bs[TK][TM];

    int tx = threadIdx.x, ty = threadIdx.y;     // 16x16 线程
    int row = blockIdx.y * TM + ty;
    int col = blockIdx.x * TM + tx;

    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        // 协作装载: A tile (16x16) / B tile (16x16), 越界补 0
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TK; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

void gemm_smem_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_smem_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace answers
