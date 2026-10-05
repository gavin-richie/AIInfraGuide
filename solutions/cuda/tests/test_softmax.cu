// Softmax / Online Softmax 测试(含大数值稳定对拍)
#include "answers/kernels.cuh"
#include "test_utils.cuh"

#include <cmath>
#include <vector>

static std::vector<float> ref_softmax(const std::vector<float> &row) {
    float m = -INFINITY;
    for (float v : row) m = std::fmax(m, v);
    double s = 0;
    std::vector<float> out(row.size());
    for (size_t i = 0; i < row.size(); ++i) { out[i] = std::exp(row[i] - m); s += out[i]; }
    for (auto &v : out) v /= (float)s;
    return out;
}

int main(int argc, char **argv) {
    bool bench = test::has_arg(argv, argc, "--bench");
    test::Rng rng;
    int rows = 64, cols = 4096;

    std::vector<float> h((size_t)rows * cols);
    for (auto &v : h) v = rng.uniform(-10.f, 10.f);
    // 加几个大 logit: naive exp(1000) 会溢出, 稳定版不受影响
    for (int r = 0; r < rows; ++r) h[(size_t)r * cols] = 1000.f;

    float *d_in = test::to_gpu(h);
    float *d_out;
    CUDA_CHK(cudaMalloc(&d_out, h.size() * sizeof(float)));

    answers::softmax_rowblock(d_in, d_out, rows, cols);
    auto got = test::to_host(d_out, h.size());
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row(h.begin() + (size_t)r * cols,
                               h.begin() + (size_t)(r + 1) * cols);
        auto ref = ref_softmax(row);
        EXPECT(test::allclose(
            std::vector<float>(got.begin() + (size_t)r * cols,
                               got.begin() + (size_t)(r + 1) * cols),
            ref, 1e-4f, 1e-3f));
        if (test::failures()) break;
    }

    answers::softmax_online(d_in, d_out, rows, cols);
    auto got2 = test::to_host(d_out, h.size());
    EXPECT(test::allclose(got, got2, 1e-4f, 1e-3f));  // 两个版本一致

    if (bench) {
        float t_block = test::bench_ms([&] { answers::softmax_rowblock(d_in, d_out, rows, cols); });
        float t_online = test::bench_ms([&] { answers::softmax_online(d_in, d_out, rows, cols); });
        float gbps = (float)(h.size() * sizeof(float) * 2) / (t_block / 1e3f) / 1e9f;
        std::printf("[bench softmax %dx%d] rowblock %.3f ms(有效带宽 %.0f GB/s) | online(1 warp/行演示) %.3f ms\n",
                    rows, cols, t_block, gbps, t_online);
    }

    CUDA_CHK(cudaFree(d_in));
    CUDA_CHK(cudaFree(d_out));
    RETURN_TEST_MAIN();
}
