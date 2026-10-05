// 四维张量 Concat 测试: 通用版(拼 C 维 / W 维) + 最后一维特化版对拍
#include "answers/kernels.cuh"
#include "test_utils.cuh"

#include <vector>

// CPU 参考: K 个输入按 dim 维拼接
static std::vector<float> ref_concat(const std::vector<std::vector<float>> &ins,
                                     const int dims_in[4][4], int K, int dim,
                                     const int dims_out[4]) {
    long total = 1;
    for (int d = 0; d < 4; ++d) total *= dims_out[d];
    std::vector<float> out(total);
    int offs[5] = {0};
    for (int k = 0; k < K; ++k) offs[k + 1] = offs[k] + dims_in[k][dim];
    for (long i = 0; i < total; ++i) {
        int coord[4], t = (int)i;
        for (int d = 3; d >= 0; --d) {
            coord[d] = t % dims_out[d];
            t /= dims_out[d];
        }
        int k = 0;
        while (k + 1 < K && coord[dim] >= offs[k + 1]) ++k;
        int src = 0;
        for (int d = 0; d < 4; ++d) {
            int s = (d == dim) ? coord[d] - offs[k] : coord[d];
            src = src * dims_in[k][d] + s;
        }
        out[i] = ins[k][src];
    }
    return out;
}

static void run_case(const std::vector<std::vector<float>> &ins_h,
                     const int dims_in[4][4], int K, int dim, const int dims_out[4]) {
    long total = 1;
    for (int d = 0; d < 4; ++d) total *= dims_out[d];

    // 组设备端指针数组 / offs / dims
    std::vector<float *> dptrs(K);
    for (int k = 0; k < K; ++k) dptrs[k] = test::to_gpu(ins_h[k]);
    float **d_ins;
    CUDA_CHK(cudaMalloc(&d_ins, K * sizeof(float *)));
    CUDA_CHK(cudaMemcpy(d_ins, dptrs.data(), K * sizeof(float *), cudaMemcpyHostToDevice));

    int offs_h[5] = {0};
    for (int k = 0; k < K; ++k) offs_h[k + 1] = offs_h[k] + dims_in[k][dim];
    int *d_offs, *d_dims;
    CUDA_CHK(cudaMalloc(&d_offs, (K + 1) * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_offs, offs_h, (K + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMalloc(&d_dims, 4 * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_dims, dims_out, 4 * sizeof(int), cudaMemcpyHostToDevice));

    float *d_out;
    CUDA_CHK(cudaMalloc(&d_out, total * sizeof(float)));
    answers::concat4d_generic(d_ins, d_out, d_offs, d_dims, K, dim, total);
    auto got = test::to_host(d_out, total);
    auto ref = ref_concat(ins_h, dims_in, K, dim, dims_out);
    bool ok = got == ref;
    if (!ok)
        for (long i = 0; i < total; ++i)
            if (got[i] != ref[i]) {
                std::fprintf(stderr, "  concat mismatch @%ld: %g vs %g\n", i, got[i], ref[i]);
                break;
            }
    EXPECT(ok);
    for (auto p : dptrs) CUDA_CHK(cudaFree(p));
    CUDA_CHK(cudaFree(d_ins)); CUDA_CHK(cudaFree(d_offs));
    CUDA_CHK(cudaFree(d_dims)); CUDA_CHK(cudaFree(d_out));
}

int main() {
    test::Rng rng;

    {  // 拼通道维(C): 3 输入 C=2/3/4 -> out 2x9x4x5
        int K = 3;
        int dims_in[4][4] = {{2, 2, 4, 5}, {2, 3, 4, 5}, {2, 4, 4, 5}};
        int dims_out[4] = {2, 9, 4, 5};
        std::vector<std::vector<float>> ins(K);
        for (int k = 0; k < K; ++k) {
            long sz = 1;
            for (int d = 0; d < 4; ++d) sz *= dims_in[k][d];
            ins[k].resize(sz);
            for (auto &v : ins[k]) v = rng.uniform(-1.f, 1.f);
        }
        run_case(ins, dims_in, K, 1, dims_out);
    }
    {  // 拼最后一维(W): 2 输入 -> out 2x3x4x11
        int K = 2;
        int dims_in[4][4] = {{2, 3, 4, 5}, {2, 3, 4, 6}};
        int dims_out[4] = {2, 3, 4, 11};
        std::vector<std::vector<float>> ins(K);
        for (int k = 0; k < K; ++k) {
            long sz = 1;
            for (int d = 0; d < 4; ++d) sz *= dims_in[k][d];
            ins[k].resize(sz);
            for (auto &v : ins[k]) v = rng.uniform(-1.f, 1.f);
        }
        run_case(ins, dims_in, K, 3, dims_out);

        // 同一 case 用特化版对拍
        long rows = 2L * 3 * 4, Wa = 5, Wb = 6;
        float *da = test::to_gpu(ins[0]), *db = test::to_gpu(ins[1]), *d_out;
        CUDA_CHK(cudaMalloc(&d_out, rows * (Wa + Wb) * sizeof(float)));
        answers::concat4d_last(da, db, d_out, (int)Wa, (int)Wb, rows);
        auto got = test::to_host(d_out, rows * (Wa + Wb));
        auto ref = ref_concat(ins, dims_in, K, 3, dims_out);
        EXPECT(got == ref);
        CUDA_CHK(cudaFree(da)); CUDA_CHK(cudaFree(db)); CUDA_CHK(cudaFree(d_out));
    }
    RETURN_TEST_MAIN();
}
