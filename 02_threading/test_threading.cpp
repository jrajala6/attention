#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include "thread_pool.h"

int main() {
    size_t len = 50000000; // 50 million elements
    std::vector<float> data(len, 1.0f);
    
    // Benchmark single thread scalar
    auto start = std::chrono::high_resolution_clock::now();
    float sum_scalar = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        sum_scalar += data[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> scalar_ms = end - start;
    
    // Benchmark thread pool parallel reduce
    start = std::chrono::high_resolution_clock::now();
    float sum_threaded = parallel_reduce(
        len, 
        [&data](size_t i) { return data[i]; }, 
        0.0f, 
        [](float a, float b) { return a + b; }
    );
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> threaded_ms = end - start;
    
    std::cout << "--- 02 Threading Benchmark (N = " << len << ") ---\n";
    std::cout << "Single Thread Reduce: " << scalar_ms.count() << " ms (sum=" << sum_scalar << ")\n";
    std::cout << "Thread Pool Reduce:   " << threaded_ms.count() << " ms (sum=" << sum_threaded << ")\n";
    std::cout << "Speedup:              " << scalar_ms.count() / threaded_ms.count() << "x\n\n";
    
    return 0;
}
