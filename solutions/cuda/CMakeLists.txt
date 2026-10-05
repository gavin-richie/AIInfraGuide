cmake_minimum_required(VERSION 3.18)

# 3090 = sm_86; 必须在 project() 之前设置, 否则 CMake 会用 toolkit 默认(52)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(ai_infra_cuda_answers LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 静态链接 cudart, 避免与驱动版本的动态库不匹配
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(Threads REQUIRED)

set(TESTS
    test_reduce
    test_softmax
    test_gemm
    test_transpose
    test_norm_layout
    test_misc
)
foreach(t IN LISTS TESTS)
    add_executable(${t} tests/${t}.cu src/reduce.cu src/softmax.cu src/gemm.cu
                                   src/transpose.cu src/norm.cu src/layout.cu
                                   src/histogram_scan.cu src/pool_iou.cu)
    target_include_directories(${t} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(${t} PRIVATE Threads::Threads)
    target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

enable_testing()
