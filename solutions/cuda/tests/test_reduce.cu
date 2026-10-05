// Reduce / vec_add 测试 + naive vs 优化基准
#include "answers/kernels.cuh"
#include "test_utils.cuh"

#include <algorithm>
#include <numeric>

int main(int argc, char **argv) {
    bool bench = test::has_arg(argv, argc, "--bench");
    test::Rng rng;
    size_t n = 1 << 20;  // 百万级(对应"百万浮点求和"面试题)

    std::vector<float> h(n);
    for (auto &v : h) v = rng.uniform(-1.f, 1.f);

    float *d_in = test::to_gpu(h);
    float *d_out;
    CUDA_CHK(cudaMalloc(&d_out, sizeof(float)));

    // 两遍归约
    answers::reduce_two_stage(d_in, d_out, n);
    float got = test::to_host(d_out, 1)[0];
    // 双精度 CPU 参考(浮点归约顺序不同, 用 rtol 对拍 —— 顺带讲数值精度考点)
    double ref = std::accumulate(h.begin(), h.end(), 0.0);
    EXPECT(std::fabs(got - ref) < 1e-2 * std::max(1.0, std::fabs(ref)));

    // 原子版(先清零)
    CUDA_CHK(cudaMemset(d_out, 0, sizeof(float)));
    answers::reduce_atomic(d_in, d_out, n);
    float got2 = test::to_host(d_out, 1)[0];
    EXPECT(std::fabs(got2 - ref) < 1e-2 * std::max(1.0, std::fabs(ref)));

    // vec_add 冒烟
    {
        std::vector<float> a(1024), b(1024), c(1024);
        for (int i = 0; i < 1024; ++i) { a[i] = i; b[i] = 2 * i; }
        float *da = test::to_gpu(a), *db = test::to_gpu(b), *dc;
        CUDA_CHK(cudaMalloc(&dc, 1024 * sizeof(float)));
        answers::vec_add(da, db, dc, 1024);
        c = test::to_host(dc, 1024);
        EXPECT(c[100] == 300 && c[1023] == 3069);
        CUDA_CHK(cudaFree(da)); CUDA_CHK(cudaFree(db)); CUDA_CHK(cudaFree(dc));
    }

    if (bench) {
        float t2 = test::bench_ms([&] { answers::reduce_two_stage(d_in, d_out, n); });
        CUDA_CHK(cudaMemset(d_out, 0, sizeof(float)));
        float t1 = test::bench_ms([&] { answers::reduce_atomic(d_in, d_out, n); });
        float gbps = (float)(n * sizeof(float)) / (t2 / 1e3f) / 1e9f;
        std::printf(
            "[bench reduce n=%zu] atomic %.3f ms | two-stage(tree+warp shuffle) %.3f ms"
            " | 有效带宽 %.0f GB/s | 加速比 %.1fx\n",
            n, t1, t2, gbps, t1 / t2);
    }

    CUDA_CHK(cudaFree(d_in));
    CUDA_CHK(cudaFree(d_out));
    RETURN_TEST_MAIN();
}
