#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include "neon_ops.h"

struct BenchResult {
    double scalar_ms;
    double neon_ms;
    double speedup;
    double throughput_gb_s;
};

BenchResult benchmark_dot_product(size_t len, int warmup_runs = 5, int test_runs = 20) {
    std::vector<__fp16> a(len), b(len);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < len; ++i) {
        a[i] = static_cast<__fp16>(dist(gen));
        b[i] = static_cast<__fp16>(dist(gen));
    }

    int actual_runs = (len < 100000) ? test_runs * 10 : test_runs;

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        volatile float d1 = 0.0f;
        for (size_t j = 0; j < len; ++j)
            d1 += static_cast<float>(a[j]) * static_cast<float>(b[j]);
        volatile float d2 = dot_product_f16(a.data(), b.data(), len);
        (void)d1; (void)d2;
    }

    // Scalar baseline
    std::vector<double> scalar_times;
    for (int run = 0; run < actual_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        float sum = 0.0f;
        for (size_t i = 0; i < len; ++i)
            sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        auto end = std::chrono::high_resolution_clock::now();
        scalar_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        volatile float p = sum; (void)p;
    }

    // NEON implementation
    std::vector<double> neon_times;
    for (int run = 0; run < actual_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        float sum = dot_product_f16(a.data(), b.data(), len);
        auto end = std::chrono::high_resolution_clock::now();
        neon_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        volatile float p = sum; (void)p;
    }

    std::sort(scalar_times.begin(), scalar_times.end());
    std::sort(neon_times.begin(), neon_times.end());
    double median_scalar = scalar_times[actual_runs / 2];
    double median_neon = neon_times[actual_runs / 2];
    double speedup = median_scalar / median_neon;
    double bytes = len * 2 * sizeof(__fp16);
    double throughput = bytes / (median_neon / 1000.0) / 1e9;

    return {median_scalar, median_neon, speedup, throughput};
}

int main() {
    std::cout << "\n=== NEON SIMD: Dot Product ===\n";
    std::cout << "Baseline: scalar fp16 loop (compiled with -fno-vectorize)\n\n";

    std::cout << std::setw(14) << "Vector Size"
              << std::setw(14) << "Scalar (ms)"
              << std::setw(14) << "NEON (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "Throughput"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    size_t sizes[] = {1000000, 10000000, 50000000};

    for (size_t i = 0; i < 3; ++i) {
        BenchResult r = benchmark_dot_product(sizes[i]);
        std::cout << std::setw(14) << sizes[i]
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.scalar_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.neon_ms
                  << std::setw(9) << std::fixed << std::setprecision(1) << r.speedup << "x"
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.throughput_gb_s << " GB/s"
                  << "\n";
    }
    std::cout << std::endl;

    return 0;
}
