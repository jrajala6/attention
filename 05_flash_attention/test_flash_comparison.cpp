#include "05_flash_attention/flash_attention.h"
#include "04_naive_attention/naive_attention.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>

void fill_random_realistic(std::vector<__fp16>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<__fp16>(dist(gen));
    }
}

struct ComparisonResult {
    double neon_ms;
    double simple_ms;
    double speedup;
    double accuracy_diff;
};

ComparisonResult benchmark_flash_comparison(size_t batch_size, size_t seq_len, size_t num_q_heads,
                                          size_t num_kv_heads, size_t head_dim, bool is_causal,
                                          int warmup_runs = 3, int test_runs = 8) {
    size_t kv_seq_len = seq_len;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t window_size = 0;

    size_t q_size = batch_size * num_q_heads * seq_len * head_dim;
    size_t kv_size = batch_size * num_kv_heads * kv_seq_len * head_dim;
    size_t out_size = q_size;

    std::vector<__fp16> Q(q_size);
    std::vector<__fp16> K(kv_size);
    std::vector<__fp16> V(kv_size);
    std::vector<__fp16> O_neon(out_size, static_cast<__fp16>(0.0f));
    std::vector<__fp16> O_simple(out_size, static_cast<__fp16>(0.0f));

    fill_random_realistic(Q);
    fill_random_realistic(K);
    fill_random_realistic(V);

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        flash_attention_f16(Q.data(), K.data(), V.data(), O_neon.data(),
                           batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads,
                           head_dim, scale, is_causal, 0, window_size);
        flash_attention_f16_simple(Q.data(), K.data(), V.data(), O_simple.data(),
                                   batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads,
                                   head_dim, scale, is_causal, 0, window_size);
    }

    // Benchmark NEON version
    std::vector<double> neon_times;
    for (int run = 0; run < test_runs; ++run) {
        std::fill(O_neon.begin(), O_neon.end(), static_cast<__fp16>(0.0f));
        auto start = std::chrono::high_resolution_clock::now();
        flash_attention_f16(Q.data(), K.data(), V.data(), O_neon.data(),
                           batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads,
                           head_dim, scale, is_causal, 0, window_size);
        auto end = std::chrono::high_resolution_clock::now();
        neon_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Benchmark simple auto-vectorized version
    std::vector<double> simple_times;
    for (int run = 0; run < test_runs; ++run) {
        std::fill(O_simple.begin(), O_simple.end(), static_cast<__fp16>(0.0f));
        auto start = std::chrono::high_resolution_clock::now();
        flash_attention_f16_simple(Q.data(), K.data(), V.data(), O_simple.data(),
                                   batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads,
                                   head_dim, scale, is_causal, 0, window_size);
        auto end = std::chrono::high_resolution_clock::now();
        simple_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Calculate median times
    std::sort(neon_times.begin(), neon_times.end());
    std::sort(simple_times.begin(), simple_times.end());
    double median_neon = neon_times[test_runs / 2];
    double median_simple = simple_times[test_runs / 2];
    double speedup = median_simple / median_neon;  // How much faster is simple vs NEON

    // Calculate accuracy difference
    double max_diff = 0.0;
    for (size_t i = 0; i < out_size; ++i) {
        double diff = std::abs(static_cast<float>(O_neon[i]) - static_cast<float>(O_simple[i]));
        max_diff = std::max(max_diff, diff);
    }

    return {median_neon, median_simple, speedup, max_diff};
}

int main() {
    std::cout << "=== FLASH ATTENTION: NEON vs AUTO-VECTORIZATION COMPARISON ===\n\n";
    std::cout << "Testing hypothesis: Simple loops with auto-vectorization vs hand-written NEON\n";
    std::cout << "Comparing function call overhead vs compiler optimization\n\n";

    // Test different configurations
    std::vector<std::pair<std::pair<size_t, size_t>, std::string> > configs;
    configs.push_back(std::make_pair(std::make_pair(512, 64), std::string("Small (512, h=64)")));
    configs.push_back(std::make_pair(std::make_pair(1024, 128), std::string("Medium (1K, h=128)")));
    configs.push_back(std::make_pair(std::make_pair(2048, 128), std::string("Large (2K, h=128)")));
    configs.push_back(std::make_pair(std::make_pair(4096, 128), std::string("XLarge (4K, h=128)")));
    configs.push_back(std::make_pair(std::make_pair(8192, 128), std::string("XXLarge (8K, h=128)")));

    std::cout << std::setw(20) << "Configuration"
              << std::setw(15) << "NEON (ms)"
              << std::setw(15) << "Simple (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Accuracy"
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    double total_speedup = 0.0;
    int num_tests = 0;

    for (size_t i = 0; i < configs.size(); ++i) {
        size_t seq_len = configs[i].first.first;
        size_t head_dim = configs[i].first.second;
        const std::string& name = configs[i].second;

        ComparisonResult result = benchmark_flash_comparison(1, seq_len, 32, 8, head_dim, true);

        std::string accuracy_status = (result.accuracy_diff < 1e-4) ? "EXCELLENT" :
                                     (result.accuracy_diff < 1e-3) ? "GOOD" :
                                     (result.accuracy_diff < 1e-2) ? "OK" : "POOR";

        std::string speedup_winner = (result.speedup > 1.0) ? " (NEON wins)" : " (Simple wins)";

        std::cout << std::setw(20) << name
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.neon_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.simple_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup << "x" << speedup_winner
                  << std::setw(15) << accuracy_status
                  << "\n";

        total_speedup += result.speedup;
        num_tests++;
    }

    double avg_speedup = total_speedup / num_tests;

    std::string winner = (avg_speedup > 1.05) ? "NEON" : (avg_speedup < 0.95) ? "Auto-vec" : "Tie";
    std::cout << "NEON vs Auto-vectorization: " << std::fixed << std::setprecision(2)
              << avg_speedup << "x speedup, " << winner << " wins" << std::endl;

    return 0;
}