// 布局转换与广播乘 —— AI-Infra-校招-1(NCHW→NHWC)、综合-1(4D broadcast elementwise)
#include <cuda_runtime.h>

namespace answers {

// NCHW → NHWC: out[n][h][w][c] = in[n][c][h][w]
// 特点: 纯访存型 kernel, 优化目标是合并访存
// NHWC 线性地址 = (((n*H + h)*W + w)*C + c), c stride 为 1(最快维)
__global__ void nchw_to_nhwc_kernel(const float *in, float *out, int C, int H, int W) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= H * W) return;
    int h = hw / W, w = hw % W;
    size_t src = ((size_t)n * C + c) * H * W + hw;          // in: c 的 stride 是 H*W
    size_t dst = (((size_t)n * H + h) * W + w) * C + c;     // out: c 的 stride 是 1
    out[dst] = in[src];
}

void nchw_to_nhwc(const float *in, float *out, int N, int C, int H, int W) {
    dim3 block(256);
    dim3 grid((H * W + 255) / 256, C, N);
    nchw_to_nhwc_kernel<<<grid, block>>>(in, out, C, H, W);
}

// 4D 广播 elementwise 乘: 广播维的 stride 传 0, 输出线性下标逐维反解输入下标。
// 这是 "实现 torch broadcast 语义 4D 乘法" 的 CUDA 标准做法。
__global__ void broadcast_mul_4d_kernel(const float *a, const float *b, float *out,
                                        int oc, int oh, int ow,
                                        long sa_n, long sa_c, long sa_h, long sa_w,
                                        long sb_n, long sb_c, long sb_h, long sb_w) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.y * gridDim.z;  // 不用; 用显式总数更清晰
    // 由调用方保证 i < on*oc*oh*ow(见 launcher 的 grid 覆盖)
    int n = i / ((size_t)oc * oh * ow);
    size_t rem = i % ((size_t)oc * oh * ow);
    int c = rem / ((size_t)oh * ow);
    rem %= (size_t)oh * ow;
    int h = rem / ow;
    int w = rem % ow;
    (void)total;
    out[i] = a[(size_t)n * sa_n + (size_t)c * sa_c + (size_t)h * sa_h + (size_t)w * sa_w] *
             b[(size_t)n * sb_n + (size_t)c * sb_c + (size_t)h * sb_h + (size_t)w * sb_w];
}

void broadcast_mul_4d(const float *a, const float *b, float *out,
                      int on, int oc, int oh, int ow,
                      long sa_n, long sa_c, long sa_h, long sa_w,
                      long sb_n, long sb_c, long sb_h, long sb_w) {
    size_t total = (size_t)on * oc * oh * ow;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    broadcast_mul_4d_kernel<<<blocks, threads>>>(a, b, out, oc, oh, ow,
                                                 sa_n, sa_c, sa_h, sa_w,
                                                 sb_n, sb_c, sb_h, sb_w);
}

}  // namespace answers
