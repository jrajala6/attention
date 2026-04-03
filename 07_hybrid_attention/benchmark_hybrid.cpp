#include "hybrid_attention.h"
#include "../04_naive_attention/naive_attention.h"
#include "../05_flash_attention/flash_attention.h"
#include "../06_kv_cache/kv_cache.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iomanip>

std::vector<__fp16> gen_fp16(size_t size) {
    std::vector<__fp16> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i)
        data[i] = (__fp16)dis(gen);
    return data;
}

// --- Correctness: Hybrid vs Naive ---

bool test_correctness() {
    const size_t batch = 1, seq = 32, cache_len = 16, new_len = 8;
    const size_t q_heads = 4, kv_heads = 2, head_dim = 64;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const size_t group_size = 8;

    auto Q = gen_fp16(batch * q_heads * seq * head_dim);
    auto K_full = gen_fp16(batch * kv_heads * (cache_len + new_len) * head_dim);
    auto V_full = gen_fp16(batch * kv_heads * (cache_len + new_len) * head_dim);

    // Extract new tokens
    std::vector<__fp16> K_new(batch * kv_heads * new_len * head_dim);
    std::vector<__fp16> V_new(batch * kv_heads * new_len * head_dim);
    for (size_t h = 0; h < kv_heads; ++h)
        for (size_t t = 0; t < new_len; ++t)
            for (size_t d = 0; d < head_dim; ++d) {
                size_t fi = h * (cache_len + new_len) * head_dim + (cache_len + t) * head_dim + d;
                size_t ni = h * new_len * head_dim + t * head_dim + d;
                K_new[ni] = K_full[fi]; V_new[ni] = V_full[fi];
            }

    // Populate cache
    KVCache cache;
    cache.init(1, cache_len + 10, kv_heads, head_dim, CachePrecision::INT8, group_size);
    for (size_t t = 0; t < cache_len; ++t) {
        std::vector<__fp16> kt(kv_heads * head_dim), vt(kv_heads * head_dim);
        for (size_t h = 0; h < kv_heads; ++h)
            for (size_t d = 0; d < head_dim; ++d) {
                kt[h * head_dim + d] = K_full[h * (cache_len + new_len) * head_dim + t * head_dim + d];
                vt[h * head_dim + d] = V_full[h * (cache_len + new_len) * head_dim + t * head_dim + d];
            }
        cache.append(0, kt.data(), vt.data(), 1);
    }

    std::vector<int8_t> K_cached(cache_len * kv_heads * head_dim);
    std::vector<int8_t> V_cached(cache_len * kv_heads * head_dim);
    cache.get_keys_int8(0, K_cached.data()); cache.get_values_int8(0, V_cached.data());

    // Naive baseline
    std::vector<__fp16> O_naive(batch * q_heads * seq * head_dim);
    naive_attention(Q.data(), K_full.data(), V_full.data(), O_naive.data(),
                   batch, seq, cache_len + new_len, q_heads, kv_heads, head_dim, scale, true);

    // Hybrid
    std::vector<__fp16> O_hybrid(batch * q_heads * seq * head_dim);
    hybrid_attention_f16(Q.data(), K_cached.data(), V_cached.data(),
                        cache.get_key_scales(0), cache.get_value_scales(0),
                        K_new.data(), V_new.data(), O_hybrid.data(),
                        batch, seq, cache_len, new_len, q_heads, kv_heads, head_dim,
                        scale, 0, true, 0, group_size);

    // Compare (positions that see full context)
    float max_error = 0.0f;
    for (size_t h = 0; h < q_heads; ++h)
        for (size_t t = cache_len; t < std::min(seq, cache_len + new_len); ++t)
            for (size_t d = 0; d < head_dim; ++d) {
                size_t idx = h * seq * head_dim + t * head_dim + d;
                float err = std::abs((float)O_naive[idx] - (float)O_hybrid[idx]);
                max_error = std::max(max_error, err);
            }

    return max_error < 0.1f;
}

// --- Performance Benchmark ---

struct BenchConfig {
    size_t seq_len, cache_len, new_len, q_heads, kv_heads, head_dim;
    const char* label;
};

struct BenchResult {
    double naive_us;
    double hybrid_us;
    double speedup;
    double throughput_gops;
};

BenchResult benchmark(const BenchConfig& cfg, size_t num_trials = 10) {
    const size_t batch = 1;
    const float scale = 1.0f / std::sqrt((float)cfg.head_dim);

    auto Q = gen_fp16(batch * cfg.q_heads * cfg.seq_len * cfg.head_dim);
    auto K_full = gen_fp16(batch * cfg.kv_heads * (cfg.cache_len + cfg.new_len) * cfg.head_dim);
    auto V_full = gen_fp16(batch * cfg.kv_heads * (cfg.cache_len + cfg.new_len) * cfg.head_dim);
    auto K_new = gen_fp16(batch * cfg.kv_heads * cfg.new_len * cfg.head_dim);
    auto V_new = gen_fp16(batch * cfg.kv_heads * cfg.new_len * cfg.head_dim);

    // Setup cache
    KVCache cache;
    cache.init(1, cfg.cache_len + 10, cfg.kv_heads, cfg.head_dim, CachePrecision::INT8, 8);
    for (size_t t = 0; t < cfg.cache_len; ++t) {
        std::vector<__fp16> kt(cfg.kv_heads * cfg.head_dim), vt(cfg.kv_heads * cfg.head_dim);
        for (size_t h = 0; h < cfg.kv_heads; ++h)
            for (size_t d = 0; d < cfg.head_dim; ++d) {
                kt[h * cfg.head_dim + d] = K_full[h * (cfg.cache_len + cfg.new_len) * cfg.head_dim + t * cfg.head_dim + d];
                vt[h * cfg.head_dim + d] = V_full[h * (cfg.cache_len + cfg.new_len) * cfg.head_dim + t * cfg.head_dim + d];
            }
        cache.append(0, kt.data(), vt.data(), 1);
    }

    std::vector<int8_t> K_cached(cfg.cache_len * cfg.kv_heads * cfg.head_dim);
    std::vector<int8_t> V_cached(cfg.cache_len * cfg.kv_heads * cfg.head_dim);
    cache.get_keys_int8(0, K_cached.data()); cache.get_values_int8(0, V_cached.data());

    std::vector<__fp16> O(batch * cfg.q_heads * cfg.seq_len * cfg.head_dim);

    // Warmup
    for (size_t i = 0; i < 3; ++i) {
        hybrid_attention_f16(Q.data(), K_cached.data(), V_cached.data(),
                            cache.get_key_scales(0), cache.get_value_scales(0),
                            K_new.data(), V_new.data(), O.data(),
                            batch, cfg.seq_len, cfg.cache_len, cfg.new_len,
                            cfg.q_heads, cfg.kv_heads, cfg.head_dim, scale, 0, true, 0, 8);
        naive_attention(Q.data(), K_full.data(), V_full.data(), O.data(),
                       batch, cfg.seq_len, cfg.cache_len + cfg.new_len,
                       cfg.q_heads, cfg.kv_heads, cfg.head_dim, scale, true);
    }

    // Benchmark hybrid
    auto start_h = std::chrono::high_resolution_clock::now();
    for (size_t trial = 0; trial < num_trials; ++trial) {
        hybrid_attention_f16(Q.data(), K_cached.data(), V_cached.data(),
                            cache.get_key_scales(0), cache.get_value_scales(0),
                            K_new.data(), V_new.data(), O.data(),
                            batch, cfg.seq_len, cfg.cache_len, cfg.new_len,
                            cfg.q_heads, cfg.kv_heads, cfg.head_dim, scale, 0, true, 0, 8);
    }
    auto end_h = std::chrono::high_resolution_clock::now();

    // Benchmark naive
    auto start_n = std::chrono::high_resolution_clock::now();
    for (size_t trial = 0; trial < num_trials; ++trial) {
        naive_attention(Q.data(), K_full.data(), V_full.data(), O.data(),
                       batch, cfg.seq_len, cfg.cache_len + cfg.new_len,
                       cfg.q_heads, cfg.kv_heads, cfg.head_dim, scale, true);
    }
    auto end_n = std::chrono::high_resolution_clock::now();

    double hybrid_us = std::chrono::duration_cast<std::chrono::microseconds>(end_h - start_h).count() / (double)num_trials;
    double naive_us = std::chrono::duration_cast<std::chrono::microseconds>(end_n - start_n).count() / (double)num_trials;

    size_t total_ops = batch * cfg.q_heads * cfg.seq_len * (cfg.cache_len + cfg.new_len) * cfg.head_dim * 2;
    double gops = total_ops / (hybrid_us / 1e6) / 1e9;

    return {naive_us, hybrid_us, naive_us / hybrid_us, gops};
}

int main() {
    // Correctness
    bool correct = test_correctness();
    std::cout << "Correctness vs naive: " << (correct ? "PASS" : "FAIL") << "\n";
    if (!correct) return 1;

    // Performance
    std::cout << "\n=== Hybrid Attention vs Naive ===\n";
    std::cout << "INT8-cached KV + FP16-new tokens, all speedups vs naive baseline\n\n";

    std::cout << std::setw(30) << "Configuration"
              << std::setw(14) << "Naive (us)"
              << std::setw(14) << "Hybrid (us)"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "GOPS"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    BenchConfig configs[] = {
        {128, 64, 32, 8, 2, 128, "128seq, 8/2h, d=128"},
        {256, 128, 64, 8, 2, 128, "256seq, 8/2h, d=128"},
        {128, 64, 32, 32, 8, 128, "128seq, 32/8h, d=128"},
        {64, 32, 16, 4, 2, 64, "64seq, 4/2h, d=64"},
    };

    for (size_t i = 0; i < 4; ++i) {
        BenchResult r = benchmark(configs[i]);
        std::cout << std::setw(30) << configs[i].label
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.naive_us
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.hybrid_us
                  << std::setw(9) << std::fixed << std::setprecision(1) << r.speedup << "x"
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.throughput_gops
                  << "\n";
    }
    std::cout << std::endl;

    return 0;
}
