// GEMM 测试: naive vs smem tiled 对拍 + 实测加速比与 TFLOPS
#include "answers/kernels.cuh"
#include "test_utils.cuh"

#include <vector>

static std::vector<float> ref_gemm(const std::vector<float> &A, const std::vector<float> &B,
                                   int M, int N, int K) {
    std::vector<float> C((size_t)M * N, 0.f);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            float a = A[(size_t)i * K + k];
            for (int j = 0; j < N; ++j) C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
        }
    return C;
}

int main(int argc, char **argv) {
    bool bench = test::has_arg(argv, argc, "--bench");
    test::Rng rng;

    {  // 小尺寸正确性
        int M = 33, N = 17, K = 64;  // 非对齐尺寸, 考验边界
        std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
        for (auto &v : hA) v = rng.uniform(-1.f, 1.f);
        for (auto &v : hB) v = rng.uniform(-1.f, 1.f);
        float *dA = test::to_gpu(hA), *dB = test::to_gpu(hB), *dC1, *dC2;
        CUDA_CHK(cudaMalloc(&dC1, (size_t)M * N * sizeof(float)));
        CUDA_CHK(cudaMalloc(&dC2, (size_t)M * N * sizeof(float)));

        answers::gemm_naive(dA, dB, dC1, M, N, K);
        answers::gemm_smem_tiled(dA, dB, dC2, M, N, K);
        auto ref = ref_gemm(hA, hB, M, N, K);
        EXPECT(test::allclose(test::to_host(dC1, (size_t)M * N), ref, 1e-3f, 1e-3f));
        EXPECT(test::allclose(test::to_host(dC2, (size_t)M * N), ref, 1e-3f, 1e-3f));
        CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dB));
        CUDA_CHK(cudaFree(dC1)); CUDA_CHK(cudaFree(dC2));
    }

    if (bench) {
        int M = 1024, N = 1024, K = 1024;
        std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
        for (auto &v : hA) v = rng.uniform(-1.f, 1.f);
        for (auto &v : hB) v = rng.uniform(-1.f, 1.f);
        float *dA = test::to_gpu(hA), *dB = test::to_gpu(hB), *dC;
        CUDA_CHK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));

        float t_naive = test::bench_ms([&] { answers::gemm_naive(dA, dB, dC, M, N, K); });
        float t_smem = test::bench_ms([&] { answers::gemm_smem_tiled(dA, dB, dC, M, N, K); });
        double flops = 2.0 * M * N * K;
        std::printf("[bench gemm %dx%dx%d] naive %.3f ms (%.1f GFLOPS) | smem-tiled %.3f ms (%.1f GFLOPS) | 加速比 %.1fx\n",
                    M, N, K, t_naive, flops / (t_naive / 1e3) / 1e9,
                    t_smem, flops / (t_smem / 1e3) / 1e9, t_naive / t_smem);
        CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dB)); CUDA_CHK(cudaFree(dC));
    }
    RETURN_TEST_MAIN();
}
