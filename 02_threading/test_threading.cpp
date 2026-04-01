#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <thread>
#include <random>
#include "thread_pool.h"

struct ThreadingResult {
    double serial_ms;
    double parallel_ms;
    double speedup;
    double efficiency;
    double throughput_gops;
};

// Simulate a more complex workload than simple addition
float complex_operation(float x) {
    return std::sin(x) * std::cos(x * 0.5f) + std::sqrt(std::abs(x));
}

ThreadingResult benchmark_parallel_workload(size_t workload_size, const std::string& workload_name, int test_runs = 5) {
    std::vector<float> data(workload_size);

    // Initialize with realistic data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t i = 0; i < workload_size; ++i) {
        data[i] = dist(gen);
    }

    std::vector<double> serial_times, parallel_times;

    // Warmup
    for (int i = 0; i < 2; ++i) {
        float warmup_result = 0.0f;
        for (size_t j = 0; j < std::min(workload_size, size_t(1000)); ++j) {
            warmup_result += complex_operation(data[j]);
        }
        volatile float prevent_opt = warmup_result;
        (void)prevent_opt;
    }

    // Benchmark serial execution
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();

        float sum_serial = 0.0f;
        for (size_t i = 0; i < workload_size; ++i) {
            sum_serial += complex_operation(data[i]);
        }

        auto end = std::chrono::high_resolution_clock::now();
        serial_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());

        // Prevent optimization
        volatile float prevent_opt = sum_serial;
        (void)prevent_opt;
    }

    // Benchmark parallel execution
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();

        float sum_parallel = parallel_reduce(
            workload_size,
            [&data](size_t i) { return complex_operation(data[i]); },
            0.0f,
            [](float a, float b) { return a + b; }
        );

        auto end = std::chrono::high_resolution_clock::now();
        parallel_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());

        // Prevent optimization
        volatile float prevent_opt = sum_parallel;
        (void)prevent_opt;
    }

    // Calculate statistics (use median for robustness)
    std::sort(serial_times.begin(), serial_times.end());
    std::sort(parallel_times.begin(), parallel_times.end());

    double median_serial = serial_times[test_runs / 2];
    double median_parallel = parallel_times[test_runs / 2];
    double speedup = median_serial / median_parallel;

    // Calculate efficiency (how well we use available cores)
    unsigned int num_cores = std::thread::hardware_concurrency();
    double efficiency = (speedup / num_cores) * 100.0;

    // Calculate throughput in billion operations per second
    double throughput_gops = workload_size / (median_parallel / 1000.0) / 1e9;

    return {median_serial, median_parallel, speedup, efficiency, throughput_gops};
}

int main() {
    unsigned int num_cores = std::thread::hardware_concurrency();

    std::cout << "=== THREADING PERFORMANCE ===\n";

    std::vector<std::pair<size_t, std::string> > test_sizes;
    test_sizes.push_back(std::make_pair(1000000, "1M ops"));
    test_sizes.push_back(std::make_pair(10000000, "10M ops"));
    test_sizes.push_back(std::make_pair(100000000, "100M ops"));

    std::cout << std::setw(12) << "Problem Size"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Efficiency"
              << std::setw(15) << "Throughput"
              << "\n";
    std::cout << std::string(55, '-') << "\n";

    double total_efficiency = 0.0;

    for (size_t i = 0; i < test_sizes.size(); ++i) {
        ThreadingResult result = benchmark_parallel_workload(test_sizes[i].first, test_sizes[i].second);

        std::cout << std::setw(12) << test_sizes[i].second
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.speedup << "x"
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.efficiency << "%"
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.throughput_gops << " GOp/s"
                  << "\n";

        total_efficiency += result.efficiency;
    }
    std::cout << std::endl;

    return 0;
}
