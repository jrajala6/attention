#include "../01_neon_basics/simple_ops.h"
#include "../01_neon_basics/neon_ops.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>

int main() {
    const size_t len = 10000000;
    std::vector<__fp16> a(len), b(len);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < len; ++i) {
        a[i] = static_cast<__fp16>(dist(gen));
        b[i] = static_cast<__fp16>(dist(gen));
    }

    // Warmup
    for (int i = 0; i < 3; ++i) {
        volatile float d1 = dot_product_f16_simple(a.data(), b.data(), len);
        volatile float d2 = dot_product_f16(a.data(), b.data(), len);
        (void)d1; (void)d2;
    }

    std::cout << "\n=== Simple Ops: NEON vs Auto-Vectorized Dot Product ===\n";
    std::cout << "Vector size: " << len << " elements\n\n";

    std::cout << std::setw(16) << "Implementation"
              << std::setw(14) << "Time (ms)"
              << std::setw(14) << "Throughput"
              << "\n";
    std::cout << std::string(44, '-') << "\n";

    // Benchmark auto-vectorized
    std::vector<double> simple_times;
    for (int run = 0; run < 20; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        volatile float r = dot_product_f16_simple(a.data(), b.data(), len);
        auto end = std::chrono::high_resolution_clock::now();
        simple_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        (void)r;
    }

    // Benchmark NEON
    std::vector<double> neon_times;
    for (int run = 0; run < 20; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        volatile float r = dot_product_f16(a.data(), b.data(), len);
        auto end = std::chrono::high_resolution_clock::now();
        neon_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        (void)r;
    }

    std::sort(simple_times.begin(), simple_times.end());
    std::sort(neon_times.begin(), neon_times.end());

    double median_simple = simple_times[10];
    double median_neon = neon_times[10];
    double bytes = len * 2 * sizeof(__fp16);

    std::cout << std::setw(16) << "Auto-vectorized"
              << std::setw(14) << std::fixed << std::setprecision(2) << median_simple
              << std::setw(12) << std::fixed << std::setprecision(1) << bytes / (median_simple / 1000.0) / 1e9 << " GB/s"
              << "\n";
    std::cout << std::setw(16) << "NEON intrinsics"
              << std::setw(14) << std::fixed << std::setprecision(2) << median_neon
              << std::setw(12) << std::fixed << std::setprecision(1) << bytes / (median_neon / 1000.0) / 1e9 << " GB/s"
              << "\n";
    std::cout << std::endl;

    return 0;
}
