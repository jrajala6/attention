#include "naive_attention.h"
#include "../05_flash_attention/flash_attention.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <random>

struct ComparisonResult {
    double naive_ms;
    double flash_ms;
    double speedup;
    double memory_saved_gb;
};

void fill_random_realistic(std::vector<__fp16>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<__fp16>(dist(gen));
    }
}

ComparisonResult benchmark_naive_vs_flash(size_t batch_size, size_t seq_len, size_t num_q_heads,
                                          size_t num_kv_heads, size_t head_dim, bool is_causal,
                                          int warmup_runs = 2, int test_runs = 5) {
    size_t kv_seq_len = seq_len;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t window_size = 0;

    size_t q_size = batch_size * num_q_heads * seq_len * head_dim;
    size_t kv_size = batch_size * num_kv_heads * kv_seq_len * head_dim;
    size_t out_size = q_size;

    std::vector<__fp16> Q(q_size);
    std::vector<__fp16> K(kv_size);
    std::vector<__fp16> V(kv_size);
    std::vector<__fp16> O_naive(out_size, static_cast<__fp16>(0.0f));
    std::vector<__fp16> O_flash(out_size, static_cast<__fp16>(0.0f));

    fill_random_realistic(Q);
    fill_random_realistic(K);
    fill_random_realistic(V);

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        naive_attention(Q.data(), K.data(), V.data(), O_naive.data(), batch_size, seq_len, kv_seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, is_causal, window_size);
        flash_attention_f16(Q.data(), K.data(), V.data(), O_flash.data(), batch_size, seq_len, kv_seq_len,
                           num_q_heads, num_kv_heads, head_dim, scale, is_causal, 0, window_size);
    }

    // Benchmark naive attention
    std::vector<double> naive_times;
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        naive_attention(Q.data(), K.data(), V.data(), O_naive.data(), batch_size, seq_len, kv_seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, is_causal, window_size);
        auto end = std::chrono::high_resolution_clock::now();
        naive_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Benchmark flash attention
    std::vector<double> flash_times;
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        flash_attention_f16(Q.data(), K.data(), V.data(), O_flash.data(), batch_size, seq_len, kv_seq_len,
                           num_q_heads, num_kv_heads, head_dim, scale, is_causal, 0, window_size);
        auto end = std::chrono::high_resolution_clock::now();
        flash_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Calculate median times
    std::sort(naive_times.begin(), naive_times.end());
    std::sort(flash_times.begin(), flash_times.end());
    double median_naive = naive_times[test_runs / 2];
    double median_flash = flash_times[test_runs / 2];
    double speedup = median_naive / median_flash;

    // Calculate memory savings (naive materializes full attention matrix)
    double naive_attention_matrix_gb = static_cast<double>(batch_size) * num_q_heads * seq_len * seq_len * sizeof(float) / 1e9;
    double flash_intermediate_gb = static_cast<double>(batch_size) * num_q_heads * 128 * 128 * sizeof(float) / 1e9;
    double memory_saved = naive_attention_matrix_gb - flash_intermediate_gb;

    return {median_naive, median_flash, speedup, memory_saved};
}

int main() {
    std::cout << "=== NAIVE vs FLASH ATTENTION COMPARISON ===\n";

    // Test sequence length scaling
    std::vector<std::pair<size_t, std::string> > seq_configs;
    seq_configs.push_back(std::make_pair(512, "512"));
    seq_configs.push_back(std::make_pair(1024, "1K"));
    seq_configs.push_back(std::make_pair(2048, "2K"));

    std::cout << std::setw(12) << "Seq Length"
              << std::setw(12) << "Naive (ms)"
              << std::setw(12) << "Flash (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Memory Saved"
              << "\n";
    std::cout << std::string(65, '-') << "\n";

    for (size_t i = 0; i < seq_configs.size(); ++i) {
        size_t seq_len = seq_configs[i].first;
        const std::string& name = seq_configs[i].second;

        ComparisonResult result = benchmark_naive_vs_flash(1, seq_len, 32, 8, 128, true);

        std::cout << std::setw(12) << name
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.naive_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.flash_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.speedup << "x"
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_saved_gb << " GB"
                  << "\n";
    }

    std::cout << "\n";

    // Test head count scaling
    std::vector<std::pair<std::pair<size_t, size_t>, std::string> > head_configs;
    head_configs.push_back(std::make_pair(std::make_pair(16, 4), "16→4"));
    head_configs.push_back(std::make_pair(std::make_pair(32, 8), "32→8"));
    head_configs.push_back(std::make_pair(std::make_pair(64, 8), "64→8"));

    std::cout << std::setw(12) << "Head Config"
              << std::setw(12) << "Naive (ms)"
              << std::setw(12) << "Flash (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Memory Saved"
              << "\n";
    std::cout << std::string(65, '-') << "\n";

    for (size_t i = 0; i < head_configs.size(); ++i) {
        size_t q_heads = head_configs[i].first.first;
        size_t kv_heads = head_configs[i].first.second;
        const std::string& name = head_configs[i].second;

        ComparisonResult result = benchmark_naive_vs_flash(1, 1024, q_heads, kv_heads, 128, true);

        std::cout << std::setw(12) << name
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.naive_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.flash_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.speedup << "x"
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.memory_saved_gb << " GB"
                  << "\n";
    }

    std::cout << std::endl;

    return 0;
}