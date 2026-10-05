// Avg Pooling(综合-6) + IoU(综合-6、蔚来) + 行 reduce sum(沐曦)
#include <cuda_runtime.h>

namespace answers {

__global__ void avg_pool_kernel(const float *in, float *out,
                                int H, int W, int ksize, int stride) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int OH = (H - ksize) / stride + 1;
    int OW = (W - ksize) / stride + 1;
    if (ox >= OW || oy >= OH) return;

    float acc = 0.f;
    for (int ky = 0; ky < ksize; ++ky)
        for (int kx = 0; kx < ksize; ++kx)
            acc += in[(size_t)(oy * stride + ky) * W + ox * stride + kx];
    out[(size_t)oy * OW + ox] = acc / (ksize * ksize);
}

void avg_pool_2d(const float *in, float *out, int H, int W, int ksize, int stride) {
    int OH = (H - ksize) / stride + 1;
    int OW = (W - ksize) / stride + 1;
    dim3 block(16, 16);
    dim3 grid((OW + 15) / 16, (OH + 15) / 16);
    avg_pool_kernel<<<grid, block>>>(in, out, H, W, ksize, stride);
}

// IoU: 两组 BBox 逐对计算的综合版见 CPU 侧, 这里给 kernel 版单对 IoU(用于 NMS 上 GPU)
__device__ float iou_xyxy(const float *a, const float *b) {
    float ix1 = fmaxf(a[0], b[0]);
    float iy1 = fmaxf(a[1], b[1]);
    float ix2 = fminf(a[2], b[2]);
    float iy2 = fminf(a[3], b[3]);
    float iw = fmaxf(0.f, ix2 - ix1);
    float ih = fmaxf(0.f, iy2 - iy1);
    float inter = iw * ih;
    float aa = fmaxf(0.f, a[2] - a[0]) * fmaxf(0.f, a[3] - a[1]);
    float ab = fmaxf(0.f, b[2] - b[0]) * fmaxf(0.f, b[3] - b[1]);
    float uni = aa + ab - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// M 个框两两 IoU 矩阵: out[i*M+j] = iou(boxes[i], boxes[j])
__global__ void iou_pairwise_kernel(const float *boxes, float *out, int M) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= M) return;
    out[(size_t)i * M + j] = iou_xyxy(boxes + i * 4, boxes + j * 4);
}

void iou_pairwise(const float *boxes, float *out, int M) {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (M + 15) / 16);
    iou_pairwise_kernel<<<grid, block>>>(boxes, out, M);
}

// 行 reduce sum: 一行一个 warp(shuffle 归约), 大幅减少 block 数量调度开销
__global__ void row_sum_warp_kernel(const float *in, float *out, int rows, int cols) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= rows) return;
    const float *row = in + (size_t)warp_id * cols;

    float v = 0.f;
    for (int i = lane; i < cols; i += 32) v += row[i];
    // warp shuffle 对折
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    if (lane == 0) out[warp_id] = v;
}

void row_sum_warp(const float *in, float *out, int rows, int cols) {
    int warps_per_block = 8;                    // 256 线程 = 8 warp
    int threads = warps_per_block * 32;
    int blocks = (rows + warps_per_block - 1) / warps_per_block;
    row_sum_warp_kernel<<<blocks, threads>>>(in, out, rows, cols);
}

}  // namespace answers
