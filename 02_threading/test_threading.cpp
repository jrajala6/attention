#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <thread>
#include <random>
#include <algorithm>
#include "thread_pool.h"

float complex_operation(float x) {
    return std::sin(x) * std::cos(x * 0.5f) + std::sqrt(std::abs(x));
}

struct ThreadingResult {
    double serial_ms;
    double parallel_ms;
    double speedup;
    double throughput_gops;
};

ThreadingResult benchmark_parallel(size_t workload_size, int test_runs = 5) {
    std::vector<float> data(workload_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t i = 0; i < workload_size; ++i)
        data[i] = dist(gen);

    // Warmup
    for (int i = 0; i < 2; ++i) {
        float w = 0.0f;
        for (size_t j = 0; j < std::min(workload_size, size_t(1000)); ++j)
            w += complex_operation(data[j]);
        volatile float p = w; (void)p;
    }

    // Serial baseline
    std::vector<double> serial_times;
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        float sum = 0.0f;
        for (size_t i = 0; i < workload_size; ++i)
            sum += complex_operation(data[i]);
        auto end = std::chrono::high_resolution_clock::now();
        serial_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        volatile float p = sum; (void)p;
    }

    // Parallel implementation
    std::vector<double> parallel_times;
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        float sum = parallel_reduce(
            workload_size,
            [&data](size_t i) { return complex_operation(data[i]); },
            0.0f,
            [](float a, float b) { return a + b; }
        );
        auto end = std::chrono::high_resolution_clock::now();
        parallel_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        volatile float p = sum; (void)p;
    }

    std::sort(serial_times.begin(), serial_times.end());
    std::sort(parallel_times.begin(), parallel_times.end());
    double median_serial = serial_times[test_runs / 2];
    double median_parallel = parallel_times[test_runs / 2];
    double speedup = median_serial / median_parallel;
    double throughput = workload_size / (median_parallel / 1000.0) / 1e9;

    return {median_serial, median_parallel, speedup, throughput};
}

int main() {
    unsigned int cores = std::thread::hardware_concurrency();

    std::cout << "\n=== Thread Pool: parallel_reduce ===\n";
    std::cout << "Baseline: single-threaded serial loop (" << cores << " cores available)\n\n";

    std::cout << std::setw(14) << "Problem Size"
              << std::setw(14) << "Serial (ms)"
              << std::setw(14) << "Parallel (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "Throughput"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    size_t sizes[] = {1000000, 10000000, 100000000};
    const char* labels[] = {"1M ops", "10M ops", "100M ops"};

    for (size_t i = 0; i < 3; ++i) {
        ThreadingResult r = benchmark_parallel(sizes[i]);
        std::cout << std::setw(14) << labels[i]
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.serial_ms
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.parallel_ms
                  << std::setw(9) << std::fixed << std::setprecision(1) << r.speedup << "x"
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.throughput_gops << " GOp/s"
                  << "\n";
    }
    std::cout << std::endl;

    return 0;
}
