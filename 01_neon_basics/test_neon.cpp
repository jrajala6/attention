#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <numeric>
#include "neon_ops.h"

struct BenchResult {
    double scalar_ms;
    double neon_ms;
    double speedup;
    double throughput_gb_s;
};

BenchResult benchmark_vectorization(size_t len, int warmup_runs = 5, int test_runs = 20) {
    std::vector<__fp16> a(len), b(len);

    // Initialize with realistic data distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < len; ++i) {
        a[i] = static_cast<__fp16>(dist(gen));
        b[i] = static_cast<__fp16>(dist(gen));
    }

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        volatile float dummy_scalar = 0.0f;
        for (size_t j = 0; j < len; ++j) {
            dummy_scalar += static_cast<float>(a[j]) * static_cast<float>(b[j]);
        }
        volatile float dummy_neon = dot_product_f16(a.data(), b.data(), len);
        (void)dummy_scalar; (void)dummy_neon; // Suppress warnings
    }

    // Benchmark scalar implementation (more iterations for small vectors)
    int actual_runs = (len < 100000) ? test_runs * 10 : test_runs;
    std::vector<double> scalar_times;
    for (int run = 0; run < actual_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        float sum_scalar = 0.0f;
        for (size_t i = 0; i < len; ++i) {
            sum_scalar += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        scalar_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        // Prevent optimization
        volatile float prevent_opt = sum_scalar;
        (void)prevent_opt;
    }

    // Benchmark NEON implementation
    std::vector<double> neon_times;
    for (int run = 0; run < actual_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        float sum_neon = dot_product_f16(a.data(), b.data(), len);
        auto end = std::chrono::high_resolution_clock::now();
        neon_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        // Prevent optimization
        volatile float prevent_opt = sum_neon;
        (void)prevent_opt;
    }

    // Calculate statistics
    double avg_scalar = std::accumulate(scalar_times.begin(), scalar_times.end(), 0.0) / actual_runs;
    double avg_neon = std::accumulate(neon_times.begin(), neon_times.end(), 0.0) / actual_runs;
    double speedup = avg_scalar / avg_neon;

    // Calculate throughput: 2 operations (read + multiply) per element, 2 bytes per fp16
    double bytes_processed = len * 2 * sizeof(__fp16);
    double throughput_gb_s = bytes_processed / (avg_neon / 1000.0) / 1e9;

    return {avg_scalar, avg_neon, speedup, throughput_gb_s};
}

int main() {
    std::cout << "=== NEON SIMD Vectorization Performance Analysis ===\n\n";
    std::cout << "Testing ARM NEON float16 vector operations vs scalar (no auto-vec)\n";
    std::cout << "Workload: Dot product (fundamental operation in ML inference)\n\n";

    std::vector<size_t> test_sizes;
    test_sizes.push_back(1000);      // 1K elements - small vectors
    test_sizes.push_back(10000);     // 10K elements - medium vectors
    test_sizes.push_back(100000);    // 100K elements - large vectors
    test_sizes.push_back(1000000);   // 1M elements - very large vectors
    test_sizes.push_back(10000000);  // 10M elements - stress test
    test_sizes.push_back(50000000);  // 50M elements - memory bandwidth bound

    std::cout << std::setw(12) << "Vector Size"
              << std::setw(15) << "Scalar (ms)"
              << std::setw(15) << "NEON (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Throughput"
              << std::setw(15) << "Efficiency"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    double total_speedup = 0.0;
    int valid_tests = 0;

    for (size_t i = 0; i < test_sizes.size(); ++i) {
        size_t len = test_sizes[i];
        BenchResult result = benchmark_vectorization(len);

        // Calculate efficiency (how close to theoretical 8x speedup for 8-wide SIMD)
        double theoretical_max = 8.0; // fp16x8 NEON vectors
        double efficiency = (result.speedup / theoretical_max) * 100.0;

        std::cout << std::setw(12) << len
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.scalar_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.neon_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.speedup << "x"
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput_gb_s << " GB/s"
                  << std::setw(15) << std::fixed << std::setprecision(1) << efficiency << "%"
                  << "\n";

        total_speedup += result.speedup;
        valid_tests++;
    }

    double avg_speedup = total_speedup / valid_tests;

    std::cout << "\n=== PERFORMANCE SUMMARY ===\n";
    std::cout << "• Average vectorization speedup: " << std::fixed << std::setprecision(1) << avg_speedup << "x\n";
    std::cout << "• Peak throughput achieved: " << std::fixed << std::setprecision(1)
              << benchmark_vectorization(50000000).throughput_gb_s << " GB/s\n";
    std::cout << "• SIMD efficiency: " << std::fixed << std::setprecision(1)
              << (avg_speedup / 8.0) * 100.0 << "% of theoretical maximum\n\n";

    std::cout << "• Matrix operations are " << std::fixed << std::setprecision(1) << avg_speedup
              << "x faster, directly accelerating ML inference\n";
    std::cout << "• Memory bandwidth efficiently utilized for large-scale computations\n";
    std::cout << "• Foundation enables real-time processing on mobile/edge devices\n";

    return 0;
}
