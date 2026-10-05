// 四维张量 Concat —— 93-t4 Q5(元戎)文档内嵌 kernel 的工程化版本
// 通用版(任意拼接维, 坐标折算) + 最后一维特化版(纯下标运算, 无除法/取模)。
#include <cuda_runtime.h>

namespace answers {

// ============ 通用版: ins 为 K 个输入指针(设备端数组), offs[K+1] 为拼接维前缀和 ============
// dims 为输出四维尺寸, dim 为拼接维(0..3), total = 产出元素总数。
// 每线程处理一个输出元素: 线性下标 -> 坐标 -> 定位输入 k -> 源线性下标。
__global__ void concat4d_generic_kernel(const float *const *ins, float *out,
                                        const int *offs, const int *dims,
                                        int K, int dim, long total) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    // 1) 输出线性下标 -> 坐标(行优先连续布局)
    int coord[4], t = (int)i;
    for (int d = 3; d >= 0; --d) {
        coord[d] = t % dims[d];
        t /= dims[d];
    }
    // 2) 定位输入 k: offs 单调, K 小线性扫即可(大 K 可换二分)
    int k = 0;
    while (k + 1 < K && coord[dim] >= offs[k + 1]) ++k;
    // 3) 源坐标 -> 输入线性下标(仅 dim 维尺寸不同), 逐步折叠 row-major
    int src = 0;
    for (int d = 0; d < 4; ++d) {
        int s = (d == dim) ? coord[d] - offs[k] : coord[d];
        int dim_size = (d == dim) ? (offs[k + 1] - offs[k]) : dims[d];
        src = src * dim_size + s;
    }
    out[i] = ins[k][src];
}

void concat4d_generic(const float *const *ins, float *out,
                      const int *offs, const int *dims, int K, int dim, long total) {
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    concat4d_generic_kernel<<<blocks, threads>>>(ins, out, offs, dims, K, dim, total);
}

// ============ 最后一维(W)特化: 两输入, 无坐标折算, 一个除法一个取模 ============
// rows = N*C*H; out 每行长度 Wa+Wb, 前 Wa 个来自 a, 其余来自 b。
__global__ void concat4d_last_kernel(const float *a, const float *b, float *out,
                                     int Wa, int Wb, long rows) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = rows * (long)(Wa + Wb);
    if (i >= total) return;
    long row = i / (Wa + Wb), w = i % (Wa + Wb);
    bool fromA = w < Wa;
    const float *src = fromA ? a : b;
    out[i] = src[row * (fromA ? Wa : Wb) + (fromA ? w : w - Wa)];
}

void concat4d_last(const float *a, const float *b, float *out,
                   int Wa, int Wb, long rows) {
    long total = rows * (long)(Wa + Wb);
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    concat4d_last_kernel<<<blocks, threads>>>(a, b, out, Wa, Wb, rows);
}

}  // namespace answers
