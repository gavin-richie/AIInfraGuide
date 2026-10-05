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

// ============ 寄存器 tile: BM=BN=64, 64 线程(8x8), 每线程 8x8 累加器 ============
// 81-t3 Q1(寒武纪)/70-t2(商汤 8x8 分块)的工程化版本:
// 64 个独立 acc -> 64 条无依赖 FMA 背靠背发射(ILP), 指令级隐藏 global 延迟;
// A/B tile 驻留 shared, 每线程从 shared 取 a[8]/b[8] 片段在寄存器里做外积累加。
__global__ void gemm_regtile_kernel(const float *A, const float *B, float *C,
                                    int M, int N, int K) {
    constexpr int BM = 64, BN = 64, BK = 16;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x % 8, ty = threadIdx.x / 8;  // 8x8 = 64 线程
    int row_base = blockIdx.y * BM, col_base = blockIdx.x * BN;

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 协作装载 A tile(64x16)/B tile(16x64), 越界补 0
        for (int i = threadIdx.x; i < BM * BK; i += 64) {
            int r = i / BK, c = i % BK;
            As[r][c] = (row_base + r < M && k0 + c < K)
                           ? A[(size_t)(row_base + r) * K + k0 + c] : 0.f;
        }
        for (int i = threadIdx.x; i < BK * BN; i += 64) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = (k0 + r < K && col_base + c < N)
                           ? B[(size_t)(k0 + r) * N + col_base + c] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[8], b[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[ty * 8 + i][k];  // A 行片段
            #pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = Bs[k][tx * 8 + j];  // B 列片段
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[i][j] += a[i] * b[j];  // 64 条 FMA 无依赖背靠背发射
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = row_base + ty * 8 + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int c = col_base + tx * 8 + j;
            if (c < N) C[(size_t)r * N + c] = acc[i][j];
        }
    }
}

void gemm_regtile_8x8(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(64, 1, 1);
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    gemm_regtile_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace answers
