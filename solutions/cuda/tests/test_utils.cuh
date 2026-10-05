// CUDA 测试公共工具: CUDA 检查宏 / 随机数据 / 对拍 / ms 计时
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHK(expr)                                                          \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",              \
                         cudaGetErrorString(_e), __FILE__, __LINE__, #expr);    \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace test {

inline int &failures() {
    static int n = 0;
    return n;
}

inline bool allclose(const std::vector<float> &a, const std::vector<float> &b,
                     float atol = 1e-3, float rtol = 1e-3) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > atol + rtol * std::fabs(b[i])) {
            std::fprintf(stderr, "  mismatch @%zu: %g vs %g\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

// 简单 xorshift 随机数(跨平台可复现)
struct Rng {
    unsigned long s = 88172645463325252ULL;
    unsigned long next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() & 0xFFFFFF) / (float)0x1000000);
    }
};

inline float *to_gpu(const std::vector<float> &v) {
    float *p = nullptr;
    CUDA_CHK(cudaMalloc(&p, v.size() * sizeof(float)));
    CUDA_CHK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    return p;
}
inline std::vector<float> to_host(const float *p, size_t n) {
    std::vector<float> v(n);
    CUDA_CHK(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

// cudaEvent 计时(毫秒), runs 次取平均
template <typename F>
float bench_ms(F &&launch, int runs = 20, int warmup = 3) {
    for (int i = 0; i < warmup; ++i) launch();
    cudaEvent_t s, e;
    CUDA_CHK(cudaEventCreate(&s));
    CUDA_CHK(cudaEventCreate(&e));
    CUDA_CHK(cudaEventRecord(s));
    for (int i = 0; i < runs; ++i) launch();
    CUDA_CHK(cudaEventRecord(e));
    CUDA_CHK(cudaEventSynchronize(e));
    float ms = 0.f;
    CUDA_CHK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHK(cudaEventDestroy(s));
    CUDA_CHK(cudaEventDestroy(e));
    return ms / runs;
}

inline bool has_arg(char **argv, int argc, const char *flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

}  // namespace test

#define EXPECT(cond)                                                            \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "  FAIL %s:%d  EXPECT(%s)\n", __FILE__,        \
                         __LINE__, #cond);                                      \
            ++test::failures();                                                 \
        }                                                                       \
    } while (0)

#define RETURN_TEST_MAIN()                                                        \
    do {                                                                          \
        CUDA_CHK(cudaGetLastError());                                             \
        if (test::failures() == 0) {                                              \
            std::printf("[  PASSED  ] %s\n", __FILE__);                           \
            return 0;                                                             \
        }                                                                         \
        std::printf("[  FAILED  ] %s (%d assertion(s))\n", __FILE__,              \
                    test::failures());                                            \
        return 1;                                                                 \
    } while (0)
