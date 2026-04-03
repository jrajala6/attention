#include "quant.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>

struct QuantResult {
    double throughput_gb_s;
    double compression_ratio;
    double mse_error;
};

void dequantize_int8_to_fp16(const int8_t* quantized, __fp16* output,
                             const float* scales, size_t total_elements, size_t group_size) {
    for (size_t i = 0; i < total_elements; ++i) {
        size_t group_idx = i / group_size;
        output[i] = static_cast<__fp16>(quantized[i] * scales[group_idx]);
    }
}

QuantResult benchmark_quantization(size_t total_elements, size_t group_size, int test_runs = 5) {
    size_t num_groups = (total_elements + group_size - 1) / group_size;

    std::vector<__fp16> src(total_elements);
    std::vector<int8_t> dst(total_elements);
    std::vector<float> scales(num_groups);
    std::vector<__fp16> reconstructed(total_elements);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < total_elements; ++i)
        src[i] = static_cast<__fp16>(dist(gen));

    // Warmup
    for (int i = 0; i < 2; ++i)
        quantize_fp16_to_int8_grouped(src.data(), dst.data(), scales.data(), total_elements, group_size);

    // Benchmark
    std::vector<double> times;
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        quantize_fp16_to_int8_grouped(src.data(), dst.data(), scales.data(), total_elements, group_size);
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(times.begin(), times.end());
    double median_time = times[test_runs / 2];

    double bytes = total_elements * sizeof(__fp16);
    double throughput = bytes / (median_time / 1000.0) / 1e9;

    double original_size = total_elements * sizeof(__fp16);
    double compressed_size = total_elements * sizeof(int8_t) + num_groups * sizeof(float);
    double compression = original_size / compressed_size;

    dequantize_int8_to_fp16(dst.data(), reconstructed.data(), scales.data(), total_elements, group_size);
    double mse = 0.0;
    for (size_t i = 0; i < total_elements; ++i) {
        double err = std::abs(static_cast<float>(src[i]) - static_cast<float>(reconstructed[i]));
        mse += err * err;
    }
    mse /= total_elements;

    return {throughput, compression, mse};
}

int main() {
    size_t test_elements = 10000000;

    std::cout << "\n=== Quantization: FP16 -> INT8 ===\n";
    std::cout << "Baseline: uncompressed FP16 (2 bytes/element)\n\n";

    std::cout << std::setw(20) << "Group Size"
              << std::setw(14) << "Throughput"
              << std::setw(14) << "Compression"
              << std::setw(14) << "MSE"
              << std::setw(14) << "Size Saved"
              << "\n";
    std::cout << std::string(76, '-') << "\n";

    size_t groups[] = {32, 64, 128, 256, 512};
    const char* labels[] = {"Ultra-fine (32)", "Fine (64)", "Standard (128)", "Coarse (256)", "Very coarse (512)"};

    for (size_t i = 0; i < 5; ++i) {
        QuantResult r = benchmark_quantization(test_elements, groups[i]);
        double size_saved = ((r.compression_ratio - 1.0) / r.compression_ratio) * 100.0;

        std::cout << std::setw(20) << labels[i]
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.throughput_gb_s << " GB/s"
                  << std::setw(13) << std::fixed << std::setprecision(2) << r.compression_ratio << "x"
                  << std::setw(14) << std::scientific << std::setprecision(1) << r.mse_error
                  << std::setw(13) << std::fixed << std::setprecision(1) << size_saved << "%"
                  << "\n";
    }
    std::cout << std::endl;

    return 0;
}
