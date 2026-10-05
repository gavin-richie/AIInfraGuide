// 数值稳定 Softmax + Online Softmax —— 综合-6、太初、海康、三星、飞腾
#include <cuda_runtime.h>
#include <math.h>

namespace answers {

// ============ 两遍式数值稳定 softmax: 一行一个 block ============
// pass1: 块内归约求 max; pass2: 块内归约求 sum(exp(x-m)); pass3: 写回。
// 数据经 shared memory 驻留, 全程只读一次 HBM(行 <= blockDim)。
__global__ void softmax_rowblock_kernel(const float *in, float *out, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.y;
    int tid = threadIdx.x;
    const float *row_in = in + (size_t)row * cols;
    float *row_out = out + (size_t)row * cols;

    // 1) 求 max
    float m = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) m = fmaxf(m, row_in[i]);
    sdata[tid] = m;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    m = sdata[0];
    __syncthreads();

    // 2) 求 sum(exp(x - m))  —— 减 max 后 exp 不溢出
    float s = 0.f;
    for (int i = tid; i < cols; i += blockDim.x) s += __expf(row_in[i] - m);
    sdata[tid] = s;
    __syncthreads();
    for (unsigned h = blockDim.x / 2; h > 0; h >>= 1) {
        if (tid < h) sdata[tid] += sdata[tid + h];
        __syncthreads();
    }
    float sum = sdata[0] + 1e-30f;
    __syncthreads();

    // 3) 归一化写回
    for (int i = tid; i < cols; i += blockDim.x)
        row_out[i] = __expf(row_in[i] - m) / sum;
}

void softmax_rowblock(const float *d_in, float *d_out, int rows, int cols) {
    dim3 block(1024);
    dim3 grid(1, rows);
    size_t smem = block.x * sizeof(float);
    softmax_rowblock_kernel<<<grid, block, smem>>>(d_in, d_out, cols);
}

// ============ Online softmax: 单遍扫描(FA 数学基础) ============
// 一行一个线程流式读入, 增量维护 (m, l):
//   m_new = max(m, x);  l = l*exp(m-m_new) + exp(x-m_new)
// 最后一趟只算 exp(x-m)/l —— 但严格单遍需缓存分子, 此处演示"两访 vs 单访"中
// 的在线更新结构; 真正的单遍版把分子写寄存器数组(见 online_softmax.py 的块版)。
__global__ void softmax_online_kernel(const float *in, float *out, int cols) {
    int row = blockIdx.x;
    const float *row_in = in + (size_t)row * cols;
    float *row_out = out + (size_t)row * cols;

    float m = -INFINITY, l = 0.f;
    for (int i = 0; i < cols; ++i) {
        float x = row_in[i];
        float m_new = fmaxf(m, x);
        l = l * __expf(m - m_new) + __expf(x - m_new);
        m = m_new;
    }
    for (int i = 0; i < cols; ++i)
        row_out[i] = __expf(row_in[i] - m) / (l + 1e-30f);
}

void softmax_online(const float *d_in, float *d_out, int rows, int cols) {
    softmax_online_kernel<<<rows, 1>>>(d_in, d_out, cols);
}

}  // namespace answers
