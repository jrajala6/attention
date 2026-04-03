#include "kv_cache.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cassert>

void fill_random_fp16(std::vector<__fp16>& vec, float scale = 0.02f) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, scale);
    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] = static_cast<__fp16>(dist(gen));
}


// --- Correctness Tests ---

bool test_basic_functionality() {
    size_t num_layers = 4, max_seq_len = 64, num_kv_heads = 8, head_dim = 64;

    KVCache fp16_cache;
    fp16_cache.init(num_layers, max_seq_len, num_kv_heads, head_dim, CachePrecision::FP16);

    size_t token_elements = num_kv_heads * head_dim;
    std::vector<__fp16> keys(token_elements), values(token_elements);
    fill_random_fp16(keys); fill_random_fp16(values);

    for (size_t layer = 0; layer < num_layers; ++layer)
        fp16_cache.append(layer, keys.data(), values.data(), 1);
    assert(fp16_cache.effective_len() == 1);

    KVCache int8_cache;
    int8_cache.init(num_layers, max_seq_len, num_kv_heads, head_dim, CachePrecision::INT8);

    for (size_t layer = 0; layer < num_layers; ++layer)
        int8_cache.append(layer, keys.data(), values.data(), 1);
    assert(int8_cache.effective_len() == 1);

    return true;
}

bool test_sliding_window() {
    size_t window_size = 8, sink_size = 2, num_kv_heads = 2, head_dim = 64;

    KVCache cache;
    cache.init(1, 32, num_kv_heads, head_dim, CachePrecision::FP16);
    cache.set_window(window_size, sink_size);

    size_t token_elements = num_kv_heads * head_dim;
    std::vector<__fp16> keys(token_elements), values(token_elements);

    for (size_t token = 0; token < 12; ++token) {
        fill_random_fp16(keys, 0.01f); fill_random_fp16(values, 0.01f);
        if (token < sink_size) {
            for (size_t h = 0; h < num_kv_heads; ++h)
                for (size_t d = 0; d < head_dim; ++d)
                    keys[h * head_dim + d] = static_cast<__fp16>(100.0f + token + h * 0.1f);
        }
        cache.append(0, keys.data(), values.data(), 1);
    }

    if (cache.effective_len() != window_size) return false;

    std::vector<__fp16> retrieved(num_kv_heads * cache.effective_len() * head_dim);
    cache.get_keys_fp16(0, retrieved.data());

    for (size_t token = 0; token < sink_size; ++token) {
        for (size_t h = 0; h < num_kv_heads; ++h) {
            size_t idx = h * cache.effective_len() * head_dim + token * head_dim;
            float expected = 100.0f + token + h * 0.1f;
            if (std::abs(static_cast<float>(retrieved[idx]) - expected) > 1.0f) return false;
        }
    }
    return true;
}

// --- Performance Benchmark ---

struct KVCacheResult {
    double fp16_ms;
    double int8_ms;
    double memory_saved_pct;
};

KVCacheResult benchmark_kv_cache(size_t num_layers, size_t seq_len, size_t num_kv_heads,
                                 size_t head_dim, size_t group_size = 64,
                                 int warmup_runs = 2, int test_runs = 5) {
    size_t token_elements = num_kv_heads * head_dim;
    std::vector<__fp16> fp16_keys(token_elements), fp16_values(token_elements);

    fill_random_fp16(fp16_keys); fill_random_fp16(fp16_values);

    size_t max_seq = seq_len + 100;
    double fp16_total = 0, int8_total = 0;

    for (int run = 0; run < warmup_runs + test_runs; ++run) {
        KVCache cache;
        cache.init(num_layers, max_seq, num_kv_heads, head_dim, CachePrecision::FP16);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t t = 0; t < seq_len; ++t)
            for (size_t l = 0; l < num_layers; ++l)
                cache.append(l, fp16_keys.data(), fp16_values.data(), 1);
        auto end = std::chrono::high_resolution_clock::now();
        if (run >= warmup_runs)
            fp16_total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    for (int run = 0; run < warmup_runs + test_runs; ++run) {
        KVCache cache;
        cache.init(num_layers, max_seq, num_kv_heads, head_dim, CachePrecision::INT8, group_size);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t t = 0; t < seq_len; ++t)
            for (size_t l = 0; l < num_layers; ++l)
                cache.append(l, fp16_keys.data(), fp16_values.data(), 1);
        auto end = std::chrono::high_resolution_clock::now();
        if (run >= warmup_runs)
            int8_total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double fp16_ms = fp16_total / test_runs;
    double int8_ms = int8_total / test_runs;

    size_t fp16_mem = num_layers * seq_len * num_kv_heads * head_dim * sizeof(__fp16) * 2;
    size_t int8_mem = num_layers * seq_len * num_kv_heads * head_dim * sizeof(int8_t) * 2 +
                      num_layers * seq_len * sizeof(float) * 2;
    double mem_saved = (1.0 - static_cast<double>(int8_mem) / fp16_mem) * 100.0;

    return {fp16_ms, int8_ms, mem_saved};
}

int main() {
    // Run correctness tests
    bool ok = true;
    ok &= test_basic_functionality();
    ok &= test_sliding_window();

    if (!ok) {
        std::cout << "FAIL: Correctness tests failed\n";
        return 1;
    }
    std::cout << "Correctness tests passed\n";

    // Performance benchmarks
    std::cout << "\n=== KV Cache: FP16 Append vs INT8 Append (with auto-quantization) ===\n";
    std::cout << "Both receive FP16 input. INT8 cache quantizes on the fly.\n\n";

    std::cout << std::setw(30) << "Configuration"
              << std::setw(12) << "FP16 (ms)"
              << std::setw(12) << "INT8 (ms)"
              << std::setw(14) << "Quant Cost"
              << std::setw(12) << "Mem Saved"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    struct Config {
        size_t layers, seq_len, heads, head_dim;
        const char* label;
    };

    Config configs[] = {
        {4, 64, 8, 64, "4L, 64T, 8H, d=64"},
        {12, 256, 12, 64, "12L, 256T, 12H, d=64"},
        {32, 512, 32, 128, "32L, 512T, 32H, d=128"},
    };

    for (size_t i = 0; i < 3; ++i) {
        Config& c = configs[i];
        KVCacheResult r = benchmark_kv_cache(c.layers, c.seq_len, c.heads, c.head_dim);
        std::cout << std::setw(30) << c.label
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.fp16_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.int8_ms
                  << std::setw(13) << std::fixed << std::setprecision(0) << (r.int8_ms / r.fp16_ms) << "x"
                  << std::setw(11) << std::fixed << std::setprecision(1) << r.memory_saved_pct << "%"
                  << "\n";
    }
    std::cout << std::endl;

    return 0;
}
