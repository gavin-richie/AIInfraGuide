// LayerNorm / RMSNorm / NCHW→NHWC / 4D 广播乘 测试
#include "answers/kernels.cuh"
#include "test_utils.cuh"

#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    bool bench = test::has_arg(argv, argc, "--bench");
    test::Rng rng;

    {  // LayerNorm 对拍
        int rows = 32, cols = 1024;
        std::vector<float> x((size_t)rows * cols), w(cols), b(cols), y((size_t)rows * cols);
        for (auto &v : x) v = rng.uniform(-3.f, 3.f);
        for (auto &v : w) v = rng.uniform(0.5f, 1.5f);
        for (auto &v : b) v = rng.uniform(-0.1f, 0.1f);
        float *dx = test::to_gpu(x), *dw = test::to_gpu(w), *db = test::to_gpu(b), *dy;
        CUDA_CHK(cudaMalloc(&dy, x.size() * sizeof(float)));
        answers::layernorm(dx, dw, db, dy, rows, cols, 1e-5f);
        auto got = test::to_host(dy, x.size());

        for (int r = 0; r < rows; ++r) {
            const float *rx = &x[(size_t)r * cols];
            double mean = 0, var = 0;
            for (int i = 0; i < cols; ++i) mean += rx[i];
            mean /= cols;
            for (int i = 0; i < cols; ++i) var += (rx[i] - mean) * (rx[i] - mean);
            var /= cols;
            float rstd = 1.f / std::sqrt((float)var + 1e-5f);
            for (int i = 0; i < cols; ++i)
                if (std::fabs(got[(size_t)r * cols + i] -
                              ((rx[i] - mean) * rstd * w[i] + b[i])) > 1e-2f) {
                    std::fprintf(stderr, "  LN mismatch r=%d i=%d\n", r, i);
                    ++test::failures();
                    r = rows;
                    break;
                }
        }
        if (bench) {
            float t = test::bench_ms([&] { answers::layernorm(dx, dw, db, dy, rows, cols, 1e-5f); });
            std::printf("[bench layernorm %dx%d] %.3f ms\n", rows, cols, t);
        }
        CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dw)); CUDA_CHK(cudaFree(db));
        CUDA_CHK(cudaFree(dy));
    }
    {  // RMSNorm 对拍
        int rows = 16, cols = 512;
        std::vector<float> x((size_t)rows * cols), w(cols);
        for (auto &v : x) v = rng.uniform(-2.f, 2.f);
        for (auto &v : w) v = rng.uniform(0.5f, 1.5f);
        float *dx = test::to_gpu(x), *dw = test::to_gpu(w), *dy;
        CUDA_CHK(cudaMalloc(&dy, x.size() * sizeof(float)));
        answers::rmsnorm(dx, dw, dy, rows, cols, 1e-6f);
        auto got = test::to_host(dy, x.size());
        for (int r = 0; r < rows; ++r) {
            const float *rx = &x[(size_t)r * cols];
            double ms = 0;
            for (int i = 0; i < cols; ++i) ms += (double)rx[i] * rx[i];
            ms /= cols;
            float rstd = 1.f / std::sqrt((float)ms + 1e-6f);
            for (int i = 0; i < cols; ++i)
                if (std::fabs(got[(size_t)r * cols + i] - rx[i] * rstd * w[i]) > 1e-2f) {
                    std::fprintf(stderr, "  RMS mismatch r=%d i=%d\n", r, i);
                    ++test::failures();
                    r = rows;
                    break;
                }
        }
        CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dw)); CUDA_CHK(cudaFree(dy));
    }
    {  // NCHW → NHWC
        int N = 2, C = 3, H = 4, W = 5;
        std::vector<float> h((size_t)N * C * H * W);
        for (auto &v : h) v = rng.uniform(-1.f, 1.f);
        float *din = test::to_gpu(h), *dout;
        CUDA_CHK(cudaMalloc(&dout, h.size() * sizeof(float)));
        answers::nchw_to_nhwc(din, dout, N, C, H, W);
        auto got = test::to_host(dout, h.size());
        auto at = [&](int n, int c, int hh, int ww) {
            return h[((size_t)n * C + c) * H * W + (size_t)hh * W + ww];
        };
        bool ok = true;
        for (int n = 0; n < N && ok; ++n)
            for (int hh = 0; hh < H && ok; ++hh)
                for (int ww = 0; ww < W && ok; ++ww)
                    for (int c = 0; c < C; ++c)
                        if (got[(((size_t)n * H + hh) * W + ww) * C + c] != at(n, c, hh, ww)) {
                            ok = false;
                            break;
                        }
        EXPECT(ok);
        CUDA_CHK(cudaFree(din)); CUDA_CHK(cudaFree(dout));
    }
    {  // 4D 广播乘: (2,1,6,1) * (1,3,1,5) -> (2,3,6,5)
        int on = 2, oc = 3, oh = 6, ow = 5;
        std::vector<float> a((size_t)on * 1 * oh * 1), b((size_t)1 * oc * 1 * ow);
        for (auto &v : a) v = rng.uniform(-1.f, 1.f);
        for (auto &v : b) v = rng.uniform(-1.f, 1.f);
        float *da = test::to_gpu(a), *db = test::to_gpu(b), *dc;
        CUDA_CHK(cudaMalloc(&dc, (size_t)on * oc * oh * ow * sizeof(float)));
        // 广播维 stride=0
        answers::broadcast_mul_4d(da, db, dc, on, oc, oh, ow,
                                  /*a strides*/ oh, 0, 1, 0,
                                  /*b strides*/ 0, ow, 0, 1);
        auto got = test::to_host(dc, (size_t)on * oc * oh * ow);
        bool ok = true;
        for (int n = 0; n < on && ok; ++n)
            for (int c = 0; c < oc && ok; ++c)
                for (int hh = 0; hh < oh && ok; ++hh)
                    for (int w = 0; w < ow; ++w) {
                        float expect = a[(size_t)n * oh + hh] * b[(size_t)c * ow + w];
                        size_t idx = (((size_t)n * oc + c) * oh + hh) * ow + w;
                        if (std::fabs(got[idx] - expect) > 1e-5f) {
                            ok = false;
                            std::fprintf(stderr, "  bcast mismatch n%d c%d h%d w%d: %g vs %g\n",
                                         n, c, hh, w, got[idx], expect);
                            break;
                        }
                    }
        EXPECT(ok);
        CUDA_CHK(cudaFree(da)); CUDA_CHK(cudaFree(db)); CUDA_CHK(cudaFree(dc));
    }
    RETURN_TEST_MAIN();
}
