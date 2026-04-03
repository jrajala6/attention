#include "flash_attention.h"
#include "../04_naive_attention/naive_attention.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>

void fill_random(std::vector<__fp16>& vec) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] = static_cast<__fp16>(dist(gen));
}

struct CompResult {
    double naive_ms;
    double neon_ms;
    double simple_ms;
    double neon_speedup;
    double simple_speedup;
};

CompResult benchmark_three_way(size_t seq_len, size_t head_dim,
                               int warmup_runs = 3, int test_runs = 8) {
    size_t num_q_heads = 32, num_kv_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t q_size = num_q_heads * seq_len * head_dim;
    size_t kv_size = num_kv_heads * seq_len * head_dim;

    std::vector<__fp16> Q(q_size), K(kv_size), V(kv_size);
    std::vector<__fp16> O_naive(q_size, 0), O_neon(q_size, 0), O_simple(q_size, 0);
    fill_random(Q); fill_random(K); fill_random(V);

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        naive_attention(Q.data(), K.data(), V.data(), O_naive.data(), 1, seq_len, seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, true, 0);
        flash_attention_f16(Q.data(), K.data(), V.data(), O_neon.data(), 1, seq_len, seq_len,
                           num_q_heads, num_kv_heads, head_dim, scale, true, 0, 0);
        flash_attention_f16_simple(Q.data(), K.data(), V.data(), O_simple.data(), 1, seq_len, seq_len,
                                   num_q_heads, num_kv_heads, head_dim, scale, true, 0, 0);
    }

    std::vector<double> naive_times, neon_times, simple_times;

    for (int run = 0; run < test_runs; ++run) {
        std::fill(O_naive.begin(), O_naive.end(), static_cast<__fp16>(0.0f));
        auto start = std::chrono::high_resolution_clock::now();
        naive_attention(Q.data(), K.data(), V.data(), O_naive.data(), 1, seq_len, seq_len,
                       num_q_heads, num_kv_heads, head_dim, scale, true, 0);
        auto end = std::chrono::high_resolution_clock::now();
        naive_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    for (int run = 0; run < test_runs; ++run) {
        std::fill(O_neon.begin(), O_neon.end(), static_cast<__fp16>(0.0f));
        auto start = std::chrono::high_resolution_clock::now();
        flash_attention_f16(Q.data(), K.data(), V.data(), O_neon.data(), 1, seq_len, seq_len,
                           num_q_heads, num_kv_heads, head_dim, scale, true, 0, 0);
        auto end = std::chrono::high_resolution_clock::now();
        neon_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    for (int run = 0; run < test_runs; ++run) {
        std::fill(O_simple.begin(), O_simple.end(), static_cast<__fp16>(0.0f));
        auto start = std::chrono::high_resolution_clock::now();
        flash_attention_f16_simple(Q.data(), K.data(), V.data(), O_simple.data(), 1, seq_len, seq_len,
                                   num_q_heads, num_kv_heads, head_dim, scale, true, 0, 0);
        auto end = std::chrono::high_resolution_clock::now();
        simple_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(naive_times.begin(), naive_times.end());
    std::sort(neon_times.begin(), neon_times.end());
    std::sort(simple_times.begin(), simple_times.end());

    double median_naive = naive_times[test_runs / 2];
    double median_neon = neon_times[test_runs / 2];
    double median_simple = simple_times[test_runs / 2];

    return {median_naive, median_neon, median_simple,
            median_naive / median_neon, median_naive / median_simple};
}

int main() {
    std::cout << "\n=== Flash NEON vs Auto-Vectorized vs Naive ===\n";
    std::cout << "All speedups measured against naive attention baseline\n";
    std::cout << "Config: batch=1, q_heads=32, kv_heads=8, causal=true\n\n";

    std::cout << std::setw(20) << "Configuration"
              << std::setw(14) << "Naive (ms)"
              << std::setw(14) << "NEON (ms)"
              << std::setw(14) << "AutoVec (ms)"
              << std::setw(12) << "NEON vs N"
              << std::setw(12) << "AV vs N"
              << "\n";
    std::cout << std::string(86, '-') << "\n";

    size_t seq_sizes[] = {512, 1024, 2048, 4096};
    size_t head_dims[] = {64, 128, 128, 128};
    const char* labels[] = {"512, h=64", "1K, h=128", "2K, h=128", "4K, h=128"};

    for (size_t i = 0; i < 4; ++i) {
        CompResult r = benchmark_three_way(seq_sizes[i], head_dims[i]);
        std::cout << std::setw(20) << labels[i]
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.naive_ms
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.neon_ms
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.simple_ms
                  << std::setw(11) << std::fixed << std::setprecision(1) << r.neon_speedup << "x"
                  << std::setw(11) << std::fixed << std::setprecision(1) << r.simple_speedup << "x"
                  << "\n";
    }
    std::cout << std::endl;

    return 0;
}
