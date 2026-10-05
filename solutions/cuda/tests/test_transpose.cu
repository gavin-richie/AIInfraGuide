// 矩阵转置测试: 正确性 + naive vs smem 实测带宽(bank conflict 优化收益)
#include "answers/kernels.cuh"
#include "test_utils.cuh"

int main(int argc, char **argv) {
    bool bench = test::has_arg(argv, argc, "--bench");
    test::Rng rng;
    int rows = 2048, cols = 2048;

    std::vector<float> h((size_t)rows * cols);
    for (auto &v : h) v = rng.uniform(-1.f, 1.f);
    float *d_in = test::to_gpu(h);
    float *d_out;
    CUDA_CHK(cudaMalloc(&d_out, h.size() * sizeof(float)));

    answers::transpose_naive(d_in, d_out, rows, cols);
    auto got = test::to_host(d_out, h.size());
    bool ok = true;
    for (int r = 0; r < rows && ok; ++r)
        for (int c = 0; c < cols; ++c)
            if (got[(size_t)c * rows + r] != h[(size_t)r * cols + c]) {
                ok = false;
                std::fprintf(stderr, "  mismatch at (%d,%d)\n", r, c);
                break;
            }
    EXPECT(ok);

    answers::transpose_smem(d_in, d_out, rows, cols);
    auto got2 = test::to_host(d_out, h.size());
    EXPECT(got == got2);

    if (bench) {
        float t1 = test::bench_ms([&] { answers::transpose_naive(d_in, d_out, rows, cols); });
        float t2 = test::bench_ms([&] { answers::transpose_smem(d_in, d_out, rows, cols); });
        float bytes = (float)(h.size() * sizeof(float) * 2);
        std::printf("[bench transpose %dx%d] naive %.3f ms (%.0f GB/s) | smem+padding %.3f ms (%.0f GB/s) | 加速比 %.1fx\n",
                    rows, cols, t1, bytes / (t1 / 1e3f) / 1e9f,
                    t2, bytes / (t2 / 1e3f) / 1e9f, t1 / t2);
    }

    CUDA_CHK(cudaFree(d_in));
    CUDA_CHK(cudaFree(d_out));
    RETURN_TEST_MAIN();
}
