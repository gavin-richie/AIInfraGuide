// 向量加(grid-stride) / SiLU / 向量外积 测试 —— 各批次内嵌片段的对拍
#include "answers/kernels.cuh"
#include "test_utils.cuh"

#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    bool bench = test::has_arg(argv, argc, "--bench");
    test::Rng rng;

    {  // 向量加 grid-stride: 整块 + 非整块尺寸
        for (size_t n : {size_t(1 << 20), size_t(1000003)}) {
            std::vector<float> ha(n), hb(n);
            for (auto &v : ha) v = rng.uniform(-2.f, 2.f);
            for (auto &v : hb) v = rng.uniform(-2.f, 2.f);
            float *da = test::to_gpu(ha), *db = test::to_gpu(hb), *dc;
            CUDA_CHK(cudaMalloc(&dc, n * sizeof(float)));
            answers::vec_add_grid_stride(da, db, dc, n);
            std::vector<float> ref(n);
            for (size_t i = 0; i < n; ++i) ref[i] = ha[i] + hb[i];
            EXPECT(test::allclose(test::to_host(dc, n), ref));
            CUDA_CHK(cudaFree(da)); CUDA_CHK(cudaFree(db)); CUDA_CHK(cudaFree(dc));
        }
    }
    {  // SiLU: 与 CPU 精确 expf 对拍
        size_t n = 1 << 20;
        std::vector<float> hx(n);
        for (auto &v : hx) v = rng.uniform(-8.f, 8.f);
        float *dx = test::to_gpu(hx), *dy;
        CUDA_CHK(cudaMalloc(&dy, n * sizeof(float)));
        answers::silu(dx, dy, n);
        auto got = test::to_host(dy, n);
        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            float ref = hx[i] / (1.0f + std::exp(-hx[i]));
            if (std::fabs(got[i] - ref) > 1e-5f * std::max(1.f, std::fabs(ref))) {
                ok = false;
                std::fprintf(stderr, "  silu mismatch @%zu: %g vs %g\n", i, got[i], ref);
                break;
            }
        }
        EXPECT(ok);
        if (bench) {
            float t = test::bench_ms([&] { answers::silu(dx, dy, n); });
            std::printf("[bench silu n=%zu] %.3f ms | %.1f GB/s(有效 8B/元素)\n",
                        n, t, 8.0 * n / (t / 1e3) / 1e9);
        }
        CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    }
    {  // 向量外积: M/N 均非 32/8 整数倍, 考验边界
        int M = 33, N = 67;
        std::vector<float> ha(M), hb(N);
        for (auto &v : ha) v = rng.uniform(-1.f, 1.f);
        for (auto &v : hb) v = rng.uniform(-1.f, 1.f);
        float *da = test::to_gpu(ha), *db = test::to_gpu(hb), *dc;
        CUDA_CHK(cudaMalloc(&dc, (size_t)M * N * sizeof(float)));
        answers::outer_product(da, db, dc, M, N);
        auto got = test::to_host(dc, (size_t)M * N);
        bool ok = true;
        for (int i = 0; i < M && ok; ++i)
            for (int j = 0; j < N; ++j)
                if (std::fabs(got[(size_t)i * N + j] - ha[i] * hb[j]) > 1e-5f) {
                    ok = false;
                    break;
                }
        EXPECT(ok);
        CUDA_CHK(cudaFree(da)); CUDA_CHK(cudaFree(db)); CUDA_CHK(cudaFree(dc));
    }
    RETURN_TEST_MAIN();
}
