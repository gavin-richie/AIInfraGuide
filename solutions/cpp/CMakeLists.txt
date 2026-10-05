cmake_minimum_required(VERSION 3.16)
project(ai_infra_cpp_answers CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(answers STATIC
    src/vision.cpp
    src/layernorm.cpp
)
target_include_directories(answers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)

# 每道手写题一个测试可执行文件, 全部注册进 CTest
set(TESTS
    test_lru
    test_singleton
    test_smart_ptr
    test_memory_pool
    test_ring_buffer
    test_nms
    test_conv2d
    test_bilinear
    test_layernorm
    test_language_basics
)
foreach(t IN LISTS TESTS)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE answers Threads::Threads)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

enable_testing()
