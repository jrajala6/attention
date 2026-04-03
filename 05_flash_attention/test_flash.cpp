#include "flash_attention.h"
#include "../04_naive_attention/naive_attention.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>

struct FlashResult {
    double naive_ms;
    double flash_ms;
    double speedup;
    double max_error;
    double throughput_tokens_s;
};

void fill_random(std::vector<__fp16>& vec) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] = static_cast<__fp16>(dist(gen));
}

FlashResult benchmark_flash_vs_naive(size_t batch_size, size_t seq_len, size_t num_q_heads,
                                     size_t num_kv_heads, size_t head_dim,
                                     int warmup_runs = 2, int test_runs = 6) {
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
        std::fill(O_naive.begin(), O_naive.end(), static_cast<__fp16>(0.0f));
        auto start = std::chrono::high_resolution_clock::now();
        naive_attention(Q.data(), K.data(), V.data(), O_naive.data(), batch_size, seq_len, seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, true, 0);
        auto end = std::chrono::high_resolution_clock::now();
        naive_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    for (int run = 0; run < test_runs; ++run) {
        std::fill(O_flash.begin(), O_flash.end(), static_cast<__fp16>(0.0f));
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

    double max_diff = 0.0;
    for (size_t i = 0; i < q_size; ++i) {
        double diff = std::abs(static_cast<float>(O_naive[i]) - static_cast<float>(O_flash[i]));
        max_diff = std::max(max_diff, diff);
    }

    double throughput = (batch_size * seq_len) / (median_flash / 1000.0);

    return {median_naive, median_flash, median_naive / median_flash, max_diff, throughput};
}

int main() {
    std::cout << "\n=== Flash Attention vs Naive: Large Sequence Scaling ===\n";
    std::cout << "Config: batch=1, q_heads=32, kv_heads=8, head_dim=128, causal=true\n\n";

    std::cout << std::setw(12) << "Seq Length"
              << std::setw(14) << "Naive (ms)"
              << std::setw(14) << "Flash (ms)"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "Max Error"
              << std::setw(14) << "Throughput"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    size_t sizes[] = {2048, 4096};
    const char* labels[] = {"2K", "4K"};

    for (size_t i = 0; i < 2; ++i) {
        FlashResult r = benchmark_flash_vs_naive(1, sizes[i], 32, 8, 128);
        std::cout << std::setw(12) << labels[i]
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.naive_ms
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.flash_ms
                  << std::setw(9) << std::fixed << std::setprecision(1) << r.speedup << "x"
                  << std::setw(14) << std::scientific << std::setprecision(1) << r.max_error
                  << std::setw(12) << std::fixed << std::setprecision(0) << r.throughput_tokens_s / 1000 << "K tok/s"
                  << "\n";
    }
    std::cout << std::endl;

    return 0;
}
