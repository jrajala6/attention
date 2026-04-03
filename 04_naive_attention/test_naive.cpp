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

void fill_random(std::vector<__fp16>& vec) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] = static_cast<__fp16>(dist(gen));
}

ComparisonResult benchmark_naive_vs_flash(size_t batch_size, size_t seq_len, size_t num_q_heads,
                                          size_t num_kv_heads, size_t head_dim,
                                          int warmup_runs = 2, int test_runs = 5) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t q_size = batch_size * num_q_heads * seq_len * head_dim;
    size_t kv_size = batch_size * num_kv_heads * seq_len * head_dim;

    std::vector<__fp16> Q(q_size), K(kv_size), V(kv_size);
    std::vector<__fp16> O_naive(q_size, 0), O_flash(q_size, 0);
    fill_random(Q); fill_random(K); fill_random(V);

    for (int i = 0; i < warmup_runs; ++i) {
        naive_attention(Q.data(), K.data(), V.data(), O_naive.data(), batch_size, seq_len, seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, true, 0);
        flash_attention_f16(Q.data(), K.data(), V.data(), O_flash.data(), batch_size, seq_len, seq_len,
                           num_q_heads, num_kv_heads, head_dim, scale, true, 0, 0);
    }

    std::vector<double> naive_times, flash_times;
    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        naive_attention(Q.data(), K.data(), V.data(), O_naive.data(), batch_size, seq_len, seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, true, 0);
        auto end = std::chrono::high_resolution_clock::now();
        naive_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    for (int run = 0; run < test_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        flash_attention_f16(Q.data(), K.data(), V.data(), O_flash.data(), batch_size, seq_len, seq_len,
                           num_q_heads, num_kv_heads, head_dim, scale, true, 0, 0);
        auto end = std::chrono::high_resolution_clock::now();
        flash_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(naive_times.begin(), naive_times.end());
    std::sort(flash_times.begin(), flash_times.end());
    double median_naive = naive_times[test_runs / 2];
    double median_flash = flash_times[test_runs / 2];

    double naive_mem = static_cast<double>(batch_size) * num_q_heads * seq_len * seq_len * sizeof(float) / 1e9;
    double flash_mem = static_cast<double>(batch_size) * num_q_heads * 128 * 128 * sizeof(float) / 1e9;

    return {median_naive, median_flash, median_naive / median_flash, naive_mem - flash_mem};
}

int main() {
    std::cout << "\n=== Flash Attention vs Naive: Sequence Length Scaling ===\n";
    std::cout << "Config: batch=1, q_heads=32, kv_heads=8, head_dim=128, causal=true\n\n";

    std::cout << std::setw(12) << "Seq Length"
              << std::setw(14) << "Naive (ms)"
              << std::setw(14) << "Flash (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "Mem Saved"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    size_t seq_sizes[] = {512, 1024, 2048};
    const char* seq_labels[] = {"512", "1K", "2K"};

    for (size_t i = 0; i < 3; ++i) {
        ComparisonResult r = benchmark_naive_vs_flash(1, seq_sizes[i], 32, 8, 128);
        std::cout << std::setw(12) << seq_labels[i]
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.naive_ms
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.flash_ms
                  << std::setw(9) << std::fixed << std::setprecision(1) << r.speedup << "x"
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.memory_saved_gb << " GB"
                  << "\n";
    }

    std::cout << "\n=== Flash Attention vs Naive: GQA Head Scaling ===\n";
    std::cout << "Config: batch=1, seq_len=1024, head_dim=128, causal=true\n\n";

    std::cout << std::setw(12) << "Heads (Q/KV)"
              << std::setw(14) << "Naive (ms)"
              << std::setw(14) << "Flash (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "Mem Saved"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    size_t q_heads[] = {16, 32, 64};
    size_t kv_heads[] = {4, 8, 8};
    const char* head_labels[] = {"16/4", "32/8", "64/8"};

    for (size_t i = 0; i < 3; ++i) {
        ComparisonResult r = benchmark_naive_vs_flash(1, 1024, q_heads[i], kv_heads[i], 128);
        std::cout << std::setw(12) << head_labels[i]
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.naive_ms
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.flash_ms
                  << std::setw(9) << std::fixed << std::setprecision(1) << r.speedup << "x"
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.memory_saved_gb << " GB"
                  << "\n";
    }
    std::cout << std::endl;

    return 0;
}
