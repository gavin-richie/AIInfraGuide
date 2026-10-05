// 矩阵转置 —— 综合-6、英伟达、小鹏(逐步优化) 
// naive(合并访存失败) → smem(+1 padding 消 bank conflict)
#include <cuda_runtime.h>

namespace answers {

__global__ void transpose_naive_kernel(const float *in, float *out, int rows, int cols) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;  // 源列
    int iy = blockIdx.y * blockDim.y + threadIdx.y;  // 源行
    if (ix >= cols || iy >= rows) return;
    // 读: in[iy][ix] 相邻线程连续(合并); 写: out[ix][iy] 相邻线程跨 cols 步长(不合并)
    out[(size_t)ix * rows + iy] = in[(size_t)iy * cols + ix];
}

void transpose_naive(const float *in, float *out, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    transpose_naive_kernel<<<grid, block>>>(in, out, rows, cols);
}

// smem 版: 读入/写出均合并访存; tile 尺寸 [32][33], 第二维 +1 让
// 转置读 tile[threadIdx.x][threadIdx.y] 时错开 bank → 消除 bank conflict
#define TILE 32
__global__ void transpose_smem_kernel(const float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];  // +1 padding

    // 装载源 tile: in 为 (rows, cols), 相邻线程读连续内存(合并)
    int ix = blockIdx.x * TILE + threadIdx.x;  // 源列
    int iy = blockIdx.y * TILE + threadIdx.y;  // 源行
    if (ix < cols && iy < rows)
        tile[threadIdx.y][threadIdx.x] = in[(size_t)iy * cols + ix];
    __syncthreads();

    // 写出: out 为 (cols, rows), out[bx*T+ty][by*T+tx] = tile[tx][ty]。
    // 相邻线程(tx)写相邻 out 地址(合并), smem 跨行读 → +1 padding 错开 bank
    int oc = blockIdx.x * TILE + threadIdx.y;    // 目标行(源列块+ty)
    int orow = blockIdx.y * TILE + threadIdx.x;  // 目标列(源行块+tx)
    if (oc < cols && orow < rows)
        out[(size_t)oc * rows + orow] = tile[threadIdx.x][threadIdx.y];
}

void transpose_smem(const float *in, float *out, int rows, int cols) {
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    dim3 block(TILE, TILE);  // 二维 block, threadIdx.y 才有效
    transpose_smem_kernel<<<grid, block>>>(in, out, rows, cols);
}

}  // namespace answers
