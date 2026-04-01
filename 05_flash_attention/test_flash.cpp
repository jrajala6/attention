#include "flash_attention.h"
#include "../04_naive_attention/naive_attention.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>

struct FlashAttentionResult {
    double naive_ms;
    double flash_ms;
    double speedup;
    double memory_saved_gb;
    double accuracy_loss;
    double throughput_tokens_s;
};

void fill_random_realistic(std::vector<__fp16>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<__fp16>(dist(gen));
    }
}

FlashAttentionResult benchmark_flash_vs_naive(size_t batch_size, size_t seq_len, size_t num_q_heads,
                                             size_t num_kv_heads, size_t head_dim, bool is_causal,
                                             int warmup_runs = 2, int test_runs = 6) {
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
        std::fill(O_naive.begin(), O_naive.end(), static_cast<__fp16>(0.0f));
        auto start = std::chrono::high_resolution_clock::now();
        naive_attention(Q.data(), K.data(), V.data(), O_naive.data(), batch_size, seq_len, kv_seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, is_causal, window_size);
        auto end = std::chrono::high_resolution_clock::now();
        naive_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Benchmark flash attention
    std::vector<double> flash_times;
    for (int run = 0; run < test_runs; ++run) {
        std::fill(O_flash.begin(), O_flash.end(), static_cast<__fp16>(0.0f));
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

    // Calculate memory savings
    // Naive: stores full attention matrix (batch × heads × seq × seq × sizeof(float))
    double naive_attention_matrix_gb = static_cast<double>(batch_size) * num_q_heads * seq_len * seq_len * sizeof(float) / 1e9;
    // Flash: only stores intermediate blocks (much smaller)
    double flash_intermediate_gb = static_cast<double>(batch_size) * num_q_heads * 128 * 128 * sizeof(float) / 1e9; // 128 = block size
    double memory_saved = naive_attention_matrix_gb - flash_intermediate_gb;

    // Calculate accuracy (using final outputs from last benchmark run)
    double mse = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < out_size; ++i) {
        double diff = std::abs(static_cast<float>(O_naive[i]) - static_cast<float>(O_flash[i]));
        mse += diff * diff;
        max_diff = std::max(max_diff, diff);
    }
    mse /= out_size;

    // Calculate throughput
    double total_tokens = batch_size * seq_len;
    double throughput_tokens_s = total_tokens / (median_flash / 1000.0);

    return {median_naive, median_flash, speedup, memory_saved, mse, throughput_tokens_s};
}

int main() {
    std::vector<size_t> test_sizes;
    test_sizes.push_back(2048);
    test_sizes.push_back(4096);

    double total_speedup = 0.0;
    double total_memory_saved = 0.0;
    double max_throughput = 0.0;

    for (size_t i = 0; i < test_sizes.size(); ++i) {
        FlashAttentionResult result = benchmark_flash_vs_naive(1, test_sizes[i], 32, 8, 128, true);
        total_speedup += result.speedup;
        total_memory_saved += result.memory_saved_gb;
        max_throughput = std::max(max_throughput, result.throughput_tokens_s);
    }

    double avg_speedup = total_speedup / test_sizes.size();

    std::cout << "Flash Attention: " << std::fixed << std::setprecision(1)
              << avg_speedup << "x speedup, " << (int)(max_throughput/1000) << "K tokens/s, "
              << total_memory_saved << " GB saved" << std::endl;

    return 0;
}
