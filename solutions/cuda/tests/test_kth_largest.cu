// uint8 数组第 K 大测试: 直方图 + 反转前缀和两段 kernel 对拍 std::nth_element
#include "answers/kernels.cuh"
#include "test_utils.cuh"

#include <algorithm>
#include <vector>

static unsigned char run_kth(const std::vector<unsigned char> &h, unsigned K) {
    unsigned char *din;
    CUDA_CHK(cudaMalloc(&din, h.size()));
    CUDA_CHK(cudaMemcpy(din, h.data(), h.size(), cudaMemcpyHostToDevice));
    unsigned int *dhist;
    CUDA_CHK(cudaMalloc(&dhist, 256 * sizeof(unsigned int)));
    CUDA_CHK(cudaMemset(dhist, 0, 256 * sizeof(unsigned int)));
    answers::hist256_grid_stride(din, h.size(), dhist);
    unsigned char *dout;
    CUDA_CHK(cudaMalloc(&dout, 1));
    answers::kth_from_hist(dhist, K, dout);
    unsigned char host;
    CUDA_CHK(cudaMemcpy(&host, dout, 1, cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaFree(din)); CUDA_CHK(cudaFree(dhist)); CUDA_CHK(cudaFree(dout));
    return host;
}

int main() {
    test::Rng rng;
    {  // 随机大数组: K=1(最大)/中位数/K=n(最小)
        size_t n = 1 << 20;
        std::vector<unsigned char> h(n);
        for (auto &v : h) v = (unsigned char)(rng.next() & 0xFF);
        std::vector<unsigned char> s = h;
        std::nth_element(s.begin(), s.begin(), s.end(), std::greater<unsigned char>());
        EXPECT(run_kth(h, 1) == s[0]);  // 最大值
        std::nth_element(s.begin(), s.begin() + n / 2, s.end(), std::greater<unsigned char>());
        EXPECT(run_kth(h, (unsigned)(n / 2 + 1)) == s[n / 2]);  // 第 n/2+1 大
        std::nth_element(s.begin(), s.end() - 1, s.end(), std::greater<unsigned char>());
        EXPECT(run_kth(h, (unsigned)n) == s[n - 1]);            // 最小值
    }
    {  // 全相同值: 任意 K 都是该值
        std::vector<unsigned char> h(100000, 137);
        EXPECT(run_kth(h, 1) == 137);
        EXPECT(run_kth(h, 99999) == 137);
        EXPECT(run_kth(h, 100000) == 137);
    }
    {  // 恰好越过 bin 边界: 值域只占 250..255
        std::vector<unsigned char> h(6);
        h[0] = 250; h[1] = 255; h[2] = 250; h[3] = 253; h[4] = 251; h[5] = 250;
        EXPECT(run_kth(h, 1) == 255);
        EXPECT(run_kth(h, 2) == 253);
        EXPECT(run_kth(h, 3) == 251);
        EXPECT(run_kth(h, 4) == 250);
        EXPECT(run_kth(h, 6) == 250);
    }
    RETURN_TEST_MAIN();
}
