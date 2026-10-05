// Histogram / Scan / AvgPool / IoU / 行求和 测试
#include "answers/kernels.cuh"
#include "test_utils.cuh"

#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    bool bench = test::has_arg(argv, argc, "--bench");
    test::Rng rng;

    {  // Histogram: 两个版本计数一致
        size_t n = 1 << 20;
        std::vector<unsigned char> h(n);
        for (auto &v : h) v = (unsigned char)(rng.next() & 0xFF);
        unsigned int hist_ref[256] = {0};
        for (auto v : h) ++hist_ref[v];

        unsigned char *din;
        CUDA_CHK(cudaMalloc(&din, n));
        CUDA_CHK(cudaMemcpy(din, h.data(), n, cudaMemcpyHostToDevice));
        unsigned int *dhist;
        CUDA_CHK(cudaMalloc(&dhist, 256 * sizeof(unsigned int)));

        CUDA_CHK(cudaMemset(dhist, 0, 256 * sizeof(unsigned int)));
        answers::histogram_global(din, dhist, n);
        std::vector<unsigned int> got(256);
        CUDA_CHK(cudaMemcpy(got.data(), dhist, 256 * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < 256; ++i)
            if (got[i] != hist_ref[i]) { ok = false; break; }
        EXPECT(ok);

        CUDA_CHK(cudaMemset(dhist, 0, 256 * sizeof(unsigned int)));
        answers::histogram_smem(din, dhist, n);
        CUDA_CHK(cudaMemcpy(got.data(), dhist, 256 * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));
        ok = true;
        for (int i = 0; i < 256; ++i)
            if (got[i] != hist_ref[i]) { ok = false; break; }
        EXPECT(ok);

        if (bench) {
            float t1 = test::bench_ms([&] {
                CUDA_CHK(cudaMemset(dhist, 0, 256 * sizeof(unsigned int)));
                answers::histogram_global(din, dhist, n);
            });
            float t2 = test::bench_ms([&] {
                CUDA_CHK(cudaMemset(dhist, 0, 256 * sizeof(unsigned int)));
                answers::histogram_smem(din, dhist, n);
            });
            std::printf("[bench histogram n=%zu] global-atomic %.3f ms | smem-privatized %.3f ms | 加速比 %.1fx\n",
                        n, t1, t2, t1 / t2);
        }
        CUDA_CHK(cudaFree(din)); CUDA_CHK(cudaFree(dhist));
    }
    {  // 前缀和: 单 block 内(n=1024)
        size_t n = 1024;
        std::vector<float> h(n);
        for (auto &v : h) v = rng.uniform(0.f, 1.f);
        float *din = test::to_gpu(h), *dout;
        CUDA_CHK(cudaMalloc(&dout, n * sizeof(float)));
        answers::prefix_sum_block(din, dout, n);
        auto got = test::to_host(dout, n);
        double acc = 0;
        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            acc += h[i];
            if (std::fabs(got[i] - acc) > 1e-2 * std::max(1.0, std::fabs(acc))) {
                ok = false;
                std::fprintf(stderr, "  scan mismatch @%zu: %g vs %g\n", i, got[i], acc);
                break;
            }
        }
        EXPECT(ok);
        CUDA_CHK(cudaFree(din)); CUDA_CHK(cudaFree(dout));
    }
    {  // AvgPool 3x3 stride 1
        int H = 16, W = 16, K = 3, S = 1;
        int OH = (H - K) / S + 1, OW = (W - K) / S + 1;
        std::vector<float> h((size_t)H * W);
        for (auto &v : h) v = rng.uniform(-1.f, 1.f);
        float *din = test::to_gpu(h), *dout;
        CUDA_CHK(cudaMalloc(&dout, (size_t)OH * OW * sizeof(float)));
        answers::avg_pool_2d(din, dout, H, W, K, S);
        auto got = test::to_host(dout, (size_t)OH * OW);
        bool ok = true;
        for (int oy = 0; oy < OH && ok; ++oy)
            for (int ox = 0; ox < OW; ++ox) {
                float acc = 0;
                for (int ky = 0; ky < K; ++ky)
                    for (int kx = 0; kx < K; ++kx)
                        acc += h[(size_t)(oy * S + ky) * W + ox * S + kx];
                if (std::fabs(got[(size_t)oy * OW + ox] - acc / (K * K)) > 1e-4f) {
                    ok = false;
                    break;
                }
            }
        EXPECT(ok);
        CUDA_CHK(cudaFree(din)); CUDA_CHK(cudaFree(dout));
    }
    {  // IoU 两两矩阵
        std::vector<float> boxes{
            0, 0, 10, 10,
            1, 1, 11, 11,    // IoU with #0 = 81/119 ≈ 0.6807
            20, 20, 30, 30,  // 与 #0 相离 = 0
        };
        float *dboxes = test::to_gpu(boxes);
        float *dout;
        CUDA_CHK(cudaMalloc(&dout, 9 * sizeof(float)));
        answers::iou_pairwise(dboxes, dout, 3);
        auto got = test::to_host(dout, 9);
        EXPECT(std::fabs(got[0] - 1.f) < 1e-5);
        EXPECT(std::fabs(got[1] - 81.f / 119.f) < 1e-4);
        EXPECT(std::fabs(got[2] - 0.f) < 1e-5);
        EXPECT(std::fabs(got[4] - 1.f) < 1e-5);
        CUDA_CHK(cudaFree(dboxes)); CUDA_CHK(cudaFree(dout));
    }
    {  // 行求和(一行一个 warp)
        int rows = 200, cols = 300;
        std::vector<float> h((size_t)rows * cols);
        for (auto &v : h) v = rng.uniform(-1.f, 1.f);
        float *din = test::to_gpu(h), *dout;
        CUDA_CHK(cudaMalloc(&dout, rows * sizeof(float)));
        answers::row_sum_warp(din, dout, rows, cols);
        auto got = test::to_host(dout, rows);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int c = 0; c < cols; ++c) acc += h[(size_t)r * cols + c];
            if (std::fabs(got[r] - acc) > 1e-2) { ok = false; break; }
        }
        EXPECT(ok);
        CUDA_CHK(cudaFree(din)); CUDA_CHK(cudaFree(dout));
    }
    RETURN_TEST_MAIN();
}
