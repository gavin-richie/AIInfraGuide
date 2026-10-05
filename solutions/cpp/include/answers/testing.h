// 轻量测试断言(不依赖 gtest), 所有 test_*.cpp 共用。
// 用法: main() 里直接写 EXPECT(...); 结尾 RETURN_TEST_MAIN()
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace answers {
inline int &failures() {
    static int n = 0;
    return n;
}
}  // namespace answers

#define EXPECT(cond)                                                              \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "  FAIL %s:%d  EXPECT(%s)\n", __FILE__, __LINE__, \
                         #cond);                                                  \
            ++answers::failures();                                                \
        }                                                                         \
    } while (0)

#define EXPECT_NEAR(a, b, eps)                                                       \
    do {                                                                             \
        double _da = static_cast<double>(a), _db = static_cast<double>(b);           \
        if (std::fabs(_da - _db) > (eps)) {                                          \
            std::fprintf(stderr, "  FAIL %s:%d  EXPECT_NEAR(%s=%g, %s=%g, eps=%g)\n",\
                         __FILE__, __LINE__, #a, _da, #b, _db,                       \
                         static_cast<double>(eps));                                  \
            ++answers::failures();                                                   \
        }                                                                            \
    } while (0)

#define EXPECT_THROW(expr, ExType)                                              \
    do {                                                                        \
        bool _caught = false;                                                   \
        try {                                                                   \
            (void)(expr);                                                       \
        } catch (const ExType &) {                                              \
            _caught = true;                                                     \
        } catch (...) {                                                         \
            std::fprintf(stderr, "  FAIL %s:%d  EXPECT_THROW wrong type\n",      \
                         __FILE__, __LINE__);                                   \
            ++answers::failures();                                              \
        }                                                                       \
        if (!_caught) {                                                         \
            std::fprintf(stderr, "  FAIL %s:%d  EXPECT_THROW(%s, %s) no throw\n",\
                         __FILE__, __LINE__, #expr, #ExType);                   \
            ++answers::failures();                                              \
        }                                                                       \
    } while (0)

#define RETURN_TEST_MAIN()                                            \
    do {                                                              \
        if (answers::failures() == 0) {                               \
            std::printf("[  PASSED  ] %s\n", __FILE__);               \
            return 0;                                                 \
        }                                                             \
        std::printf("[  FAILED  ] %s (%d assertion(s))\n", __FILE__,  \
                    answers::failures());                             \
        return 1;                                                     \
    } while (0)
