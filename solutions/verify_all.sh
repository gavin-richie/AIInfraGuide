#!/usr/bin/env bash
# 面试题解答代码工程一键验证：cpp → cuda → python
# 用法: bash solutions/verify_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------- C++ ----------------
say "[1/3] C++ 工程 (ctest 9 项)"
cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/cpp/build" -j >/dev/null
ctest --test-dir "$ROOT/cpp/build" --output-on-failure

# ---------------- CUDA ----------------
say "[2/3] CUDA 工程 (ctest 6 项, 需 GPU)"
cmake -S "$ROOT/cuda" -B "$ROOT/cuda/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/cuda/build" -j >/dev/null
ctest --test-dir "$ROOT/cuda/build" --output-on-failure

# ---------------- Python ----------------
say "[3/3] Python 工程 (pytest, GPU 优先)"
python3 -c "import answers" 2>/dev/null || pip install -e "$ROOT/python" -q
python3 -m pytest "$ROOT/python/tests" -q

say "全部通过 ✓ (cpp 9/9, cuda 6/6, python 76/76)"
