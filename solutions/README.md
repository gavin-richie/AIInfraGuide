# 面试题解答配套代码工程

对应文档：[docs/interview-answers](../docs/interview-answers)。本目录不是单文件示例集合，而是**三个可构建、可测试、可复现实测数据**的工程，覆盖 AI Infra 面试中的全部高频手写编程题。

```
solutions/
├── cpp/     C++17 + CMake：LRU/内存池(new[]语义)/单例(Meyers+DCL)/手写智能指针/
│            NMS+IoU/Conv2D/双线性插值/LayerNorm(Welford)/环形缓冲区
│            → ctest 9 个测试全绿
├── cuda/    CUDA 12 + CMake(sm_86, 3090 实测)：归约(原子 vs 树+warp shuffle)、
│            数值稳定/Online Softmax、GEMM(naive vs smem tiling)、矩阵转置
│            (smem+padding 消 bank conflict)、LayerNorm/RMSNorm、NCHW→NHWC、
│            4D 广播乘、直方图(全局原子 vs smem 私有化)、前缀和、AvgPooling、
│            IoU、行归约(一行一 warp)、向量加
│            → ctest 6 个测试全绿，--bench 输出 3090 实测带宽数据
└── python/  PyTorch + Triton：MHA/GQA/MQA/MLA(含权重吸收)/Online Softmax/
             FlashAttention v1&v2(与 SDPA 对拍)/PagedAttention(block table)、
             Triton fused softmax/matmul/layernorm、LoRA/SFT loss/广播语义/
             Flow Matching、LeetCode 算法 ~45 题
             → pytest 76 个测试全绿（GPU 上运行）
```

## 环境要求

| 组件 | 验证过的版本 |
|---|---|
| GPU | RTX 3090 ×2（Ampere sm_86；其他架构改 `solutions/cuda/CMakeLists.txt` 的 `CMAKE_CUDA_ARCHITECTURES`） |
| CUDA Toolkit | 12.9（≥11.8 均可） |
| gcc / cmake | 11.4 / ≥3.18 |
| Python | 3.13（≥3.10），PyTorch 2.8+cu128、Triton 3.4、pytest |

## 一键验证

```bash
bash solutions/verify_all.sh
```

脚本依次：cpp 构建测试 → cuda 构建测试 → python 安装测试，任一失败立即退出非零。

## 分工程运行

```bash
# C++ 手写题
cd cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure

# CUDA kernel(正确性对拍 + 性能实测)
cd ../cuda && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build
./build/test_gemm --bench      # 任意测试加 --bench 输出 3090 实测

# Python(PyTorch/Triton/算法)
cd ../python && pip install -e . && pytest tests -v
```

## 实测数据（RTX 3090，Release 构建）

| 测试 | naive | 优化 | 加速比 |
|---|---|---|---|
| test_reduce（1M float 求和） | atomic 1.962 ms | 树归约+warp shuffle 0.016 ms（265 GB/s） | 124× |
| test_histogram（1M byte 计数） | 全局原子 0.330 ms | smem 私有直方图 0.017 ms | 19.9× |
| test_gemm（1024³ FP32） | naive 2166 GFLOPS | 16×16 smem tiling 2804 GFLOPS | 1.3× |
| test_transpose（2048²） | naive 548 GB/s | smem+padding 547 GB/s（4096²：601 vs 537 GB/s） | 1.0-1.12× |
| test_triton（softmax 4096²） | torch 参考对照 | 0.163 ms，825 GB/s | — |

> 转置优化版在 2048² 下加速不明显的原因：16MB 数据在 3090 的 6MB L2 中大量命中，掩盖了合并访存的差异；放大到 4096²（64MB）后 smem 版带宽优势显现。这个现象本身就是 L2 与测量方法论的好教材（详见 docs/interview-answers/02）。

## 与文档的对应关系

解答文档中每道编程题都标注了代码位置（如 `solutions/cuda/src/reduce.cu`）与构建验证命令，文档叙述与代码实现一一对应；代码内的注释解释了对应面试题的考点。
