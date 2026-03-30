#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "neon_ops.h"

int main() {
    size_t len = 10000000; // 10 million elements
    std::vector<__fp16> a(len), b(len);
    
    for (size_t i = 0; i < len; ++i) {
        a[i] = static_cast<__fp16>((rand() % 100) / 100.0f);
        b[i] = static_cast<__fp16>((rand() % 100) / 100.0f);
    }
    
    // Benchmark scalar
    auto start = std::chrono::high_resolution_clock::now();
    float sum_scalar = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        sum_scalar += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> scalar_ms = end - start;
    
    // Benchmark NEON
    start = std::chrono::high_resolution_clock::now();
    float sum_neon = dot_product_f16(a.data(), b.data(), len);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> neon_ms = end - start;
    
    std::cout << "--- 01 NEON Basics Benchmark (N = " << len << ") ---\n";
    std::cout << "Scalar Dot Product: " << scalar_ms.count() << " ms (sum=" << sum_scalar << ")\n";
    std::cout << "NEON Dot Product:   " << neon_ms.count() << " ms (sum=" << sum_neon << ")\n";
    std::cout << "Speedup:            " << scalar_ms.count() / neon_ms.count() << "x\n\n";
    
    return 0;
}
