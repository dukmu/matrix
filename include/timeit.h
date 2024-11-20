#pragma once
#include <chrono>
#include <iostream>

// #define DEBUG_DO_ENABLE
#if defined(DEBUG_DO_ENABLE) && !defined(DEBUG_DO_DISABLE)
#define DEBUG_DO(x) \
    {               \
        x;          \
    }
#else
#define DEBUG_DO(x)
#endif

#define timeit(id, code, times)                                                              \
    {                                                                                        \
        auto start = std::chrono::high_resolution_clock::now();                              \
        for (int i = 0; i < times; ++i)                                                      \
        {                                                                                    \
            code;                                                                            \
        }                                                                                    \
        auto end = std::chrono::high_resolution_clock::now();                                \
        std::chrono::duration<double> duration = end - start;                                \
        std::cout << id << " Time taken: " << duration.count() * 1000/times << " ms" << std::endl; \
    }
#if defined(TIMEIT_ENABLE) && !defined(TIMEIT_DISABLE)
#define TIMEIT_BEGIN(id) \
    auto timeit_start_##id = std::chrono::high_resolution_clock::now();

#define TIMEIT_END(id) \
    std::chrono::duration<double> timeit_duration_##id = std::chrono::high_resolution_clock::now() - timeit_start_##id;

#define TIMEIT_PRINT(id, tab_num, n) \
    std::cout << std::string(tab_num, '\t') << #id << " Time taken: " << timeit_duration_##id.count() * 1000 << " ms" << std::string(n + 1, '\n');
#else
#define TIMEIT_BEGIN(id) {}
#define TIMEIT_END(id) {}
#define TIMEIT_PRINT(id, tab_num, n) {}
#endif
