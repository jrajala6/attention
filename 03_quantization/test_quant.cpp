#include "quant.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>

struct QuantizationResult {
    double quantize_ms;
    double throughput_gb_s;
    double compression_ratio;
    double mse_error;
    double max_error;
};

// Dequantize for error measurement
void dequantize_int8_to_fp16(const int8_t* quantized, __fp16* output,
                            const float* scales, size_t total_elements, size_t group_size) {
    for (size_t i = 0; i < total_elements; ++i) {
        size_t group_idx = i / group_size;
        float scale = scales[group_idx];
        output[i] = static_cast<__fp16>(quantized[i] * scale);
    }
}

QuantizationResult benchmark_quantization(size_t total_elements, size_t group_size,
                                         const std::string& data_type, int test_runs = 5) {
    size_t num_groups = (total_elements + group_size - 1) / group_size;

    std::vector<__fp16> src(total_elements);
    std::vector<int8_t> dst(total_elements);
    std::vector<float> scales(num_groups);
    std::vector<__fp16> reconstructed(total_elements);

    // Initialize with different data distributions
    std::random_device rd;
    std::mt19937 gen(rd());

    if (data_type == "normal") {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < total_elements; ++i) {
            src[i] = static_cast<__fp16>(dist(gen));
        }
    } else if (data_type == "uniform") {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (size_t i = 0; i < total_elements; ++i) {
            src[i] = static_cast<__fp16>(dist(gen));
        }
    } else if (data_type == "weights") {
        // Simulate typical model weight distribution
        std::normal_distribution<float> dist(0.0f, 0.02f); // Small variance like trained weights
        for (size_t i = 0; i < total_elements; ++i) {
            src[i] = static_cast<__fp16>(std::max(-2.0f, std::min(2.0f, dist(gen))));
        }
    }

    // Warmup
    for (int i = 0; i < 2; ++i) {
        quantize_fp16_to_int8_grouped(src.data(), dst.data(), scales.data(),
                                    total_elements, group_size);
    }

    // Benchmark quantization
    std::vector<double> quant_times;
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        quantize_fp16_to_int8_grouped(src.data(), dst.data(), scales.data(),
                                    total_elements, group_size);
        auto end = std::chrono::high_resolution_clock::now();
        quant_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Calculate median time
    std::sort(quant_times.begin(), quant_times.end());
    double median_time = quant_times[test_runs / 2];

    // Calculate throughput
    double bytes_processed = total_elements * sizeof(__fp16);
    double throughput_gb_s = bytes_processed / (median_time / 1000.0) / 1e9;

    // Calculate compression ratio
    double original_size = total_elements * sizeof(__fp16);
    double compressed_size = total_elements * sizeof(int8_t) + num_groups * sizeof(float);
    double compression_ratio = original_size / compressed_size;

    // Measure quantization error
    dequantize_int8_to_fp16(dst.data(), reconstructed.data(), scales.data(), total_elements, group_size);

    double mse = 0.0;
    double max_error = 0.0;
    for (size_t i = 0; i < total_elements; ++i) {
        double error = std::abs(static_cast<float>(src[i]) - static_cast<float>(reconstructed[i]));
        mse += error * error;
        max_error = std::max(max_error, error);
    }
    mse /= total_elements;

    return {median_time, throughput_gb_s, compression_ratio, mse, max_error};
}

int main() {

    // Test different group sizes
    std::vector<std::pair<size_t, std::string> > group_configs;
    group_configs.push_back(std::make_pair(32,  std::string("Ultra-fine (32)")));
    group_configs.push_back(std::make_pair(64,  std::string("Fine (64)")));
    group_configs.push_back(std::make_pair(128, std::string("Standard (128)")));
    group_configs.push_back(std::make_pair(256, std::string("Coarse (256)")));
    group_configs.push_back(std::make_pair(512, std::string("Very coarse (512)")));

    std::cout << "=== QUANTIZATION PERFORMANCE ===\n";

    size_t test_elements = 10000000;

    std::cout << std::setw(12) << "Group Size"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "Compression"
              << std::setw(12) << "MSE Error"
              << "\n";
    std::cout << std::string(55, '-') << "\n";

    double best_throughput = 0.0;
    double best_compression = 0.0;

    for (size_t i = 0; i < group_configs.size(); ++i) {
        size_t group_size = group_configs[i].first;
        const std::string& name = group_configs[i].second;

        QuantizationResult result = benchmark_quantization(test_elements, group_size, "normal");

        std::cout << std::setw(12) << name
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput_gb_s << " GB/s"
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.compression_ratio << "x"
                  << std::setw(12) << std::scientific << std::setprecision(1) << result.mse_error
                  << "\n";

        best_throughput = std::max(best_throughput, result.throughput_gb_s);
        best_compression = std::max(best_compression, result.compression_ratio);
    }
    std::cout << std::endl;



    std::cout << "Quantization: " << std::fixed << std::setprecision(1)
              << best_compression << "x compression, " << best_throughput << " GB/s, "
              << ((best_compression - 1.0) / best_compression) * 100.0 << "% size reduction" << std::endl;

    return 0;
}
