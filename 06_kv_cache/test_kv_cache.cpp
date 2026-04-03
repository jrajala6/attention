#include "kv_cache.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cassert>

struct KVCacheResult {
    double fp16_latency_ms;
    double int8_latency_ms;
    double memory_saved_percent;
    double accuracy_loss;
    double append_throughput_gb_s;
    double retrieval_throughput_gb_s;
    bool correctness_passed;
};

void fill_random_fp16(std::vector<__fp16>& vec, float scale = 0.02f) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, scale);

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<__fp16>(dist(gen));
    }
}

void fill_random_int8(std::vector<int8_t>& vec) {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-127, 127);

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<int8_t>(dist(gen));
    }
}

void fill_random_scales(std::vector<float>& vec) {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.001f, 0.1f);

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = dist(gen);
    }
}

bool test_basic_functionality() {
    std::cout << "🧪 Testing basic KV cache functionality..." << std::endl;

    // Test parameters
    size_t num_layers = 4;
    size_t max_seq_len = 64;
    size_t num_kv_heads = 8;
    size_t head_dim = 64;

    // Test FP16 cache
    KVCache fp16_cache;
    fp16_cache.init(num_layers, max_seq_len, num_kv_heads, head_dim, CachePrecision::FP16);

    size_t token_elements = num_kv_heads * head_dim;
    std::vector<__fp16> keys(token_elements);
    std::vector<__fp16> values(token_elements);
    fill_random_fp16(keys);
    fill_random_fp16(values);

    // Test single token append
    try {
        for (size_t layer = 0; layer < num_layers; ++layer) {
            fp16_cache.append(layer, keys.data(), values.data(), 1);
        }
        assert(fp16_cache.effective_len() == 1);
    } catch (const std::exception& e) {
        std::cout << "❌ FP16 append failed: " << e.what() << std::endl;
        return false;
    }

    // Test INT8 cache
    KVCache int8_cache;
    int8_cache.init(num_layers, max_seq_len, num_kv_heads, head_dim, CachePrecision::INT8);

    std::vector<int8_t> int8_keys(token_elements);
    std::vector<int8_t> int8_values(token_elements);
    std::vector<float> key_scales(1);
    std::vector<float> value_scales(1);

    fill_random_int8(int8_keys);
    fill_random_int8(int8_values);
    fill_random_scales(key_scales);
    fill_random_scales(value_scales);

    try {
        for (size_t layer = 0; layer < num_layers; ++layer) {
            int8_cache.append_int8(layer, int8_keys.data(), int8_values.data(),
                                  key_scales.data(), value_scales.data(), 1);
        }
        assert(int8_cache.effective_len() == 1);
    } catch (const std::exception& e) {
        std::cout << "❌ INT8 append failed: " << e.what() << std::endl;
        return false;
    }

    std::cout << "✅ Basic functionality tests passed" << std::endl;
    return true;
}

bool test_input_validation() {
    std::cout << "🛡️  Testing input validation..." << std::endl;

    KVCache cache;
    cache.init(2, 32, 4, 64, CachePrecision::FP16);

    std::vector<__fp16> keys(4 * 64);
    std::vector<__fp16> values(4 * 64);
    fill_random_fp16(keys);
    fill_random_fp16(values);

    // Test invalid layer index
    try {
        cache.append(5, keys.data(), values.data(), 1);
        std::cout << "❌ Should have thrown for invalid layer" << std::endl;
        return false;
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test null pointers
    try {
        cache.append(0, nullptr, values.data(), 1);
        std::cout << "❌ Should have thrown for null keys" << std::endl;
        return false;
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test type mismatch
    KVCache int8_cache;
    int8_cache.init(2, 32, 4, 64, CachePrecision::INT8);

    try {
        int8_cache.get_keys_fp16(0, keys.data());
        std::cout << "❌ Should have thrown for type mismatch" << std::endl;
        return false;
    } catch (const std::runtime_error& e) {
        // Expected
    }

    std::cout << "✅ Input validation tests passed" << std::endl;
    return true;
}

bool test_sliding_window_eviction() {
    std::cout << "📊 Testing sliding window eviction..." << std::endl;

    size_t window_size = 8;
    size_t sink_size = 2;
    size_t num_kv_heads = 2;
    size_t head_dim = 64;

    KVCache cache;
    cache.init(1, 32, num_kv_heads, head_dim, CachePrecision::FP16);
    cache.set_window(window_size, sink_size);

    size_t token_elements = num_kv_heads * head_dim;
    std::vector<__fp16> keys(token_elements);
    std::vector<__fp16> values(token_elements);

    // Fill cache beyond window size
    for (size_t token = 0; token < 12; ++token) {
        fill_random_fp16(keys, 0.01f);
        fill_random_fp16(values, 0.01f);

        // Make first few tokens distinctive for sink test
        if (token < sink_size) {
            for (size_t head = 0; head < num_kv_heads; ++head) {
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    size_t idx = head * head_dim + dim;
                    keys[idx] = static_cast<__fp16>(100.0f + token + head * 0.1f);
                    values[idx] = static_cast<__fp16>(200.0f + token + head * 0.1f);
                }
            }
        }

        cache.append(0, keys.data(), values.data(), 1);
    }

    // Should have evicted to window size
    if (cache.effective_len() != window_size) {
        std::cout << "❌ Eviction failed: expected len=" << window_size
                  << ", got len=" << cache.effective_len() << std::endl;
        return false;
    }

    // Verify sink tokens are preserved
    std::vector<__fp16> retrieved_keys(num_kv_heads * cache.effective_len() * head_dim);
    cache.get_keys_fp16(0, retrieved_keys.data());

    // Check if sink tokens are still there
    // Head-major layout: [num_kv_heads][seq_len][head_dim]
    bool sink_preserved = true;
    for (size_t token = 0; token < sink_size; ++token) {
        for (size_t head = 0; head < num_kv_heads; ++head) {
            // Access pattern: head * cache.effective_len() * head_dim + token * head_dim
            size_t base_idx = head * cache.effective_len() * head_dim + token * head_dim;
            float expected_val = 100.0f + token + head * 0.1f;

            if (std::abs(static_cast<float>(retrieved_keys[base_idx]) - expected_val) > 1.0f) {
                std::cout << "❌ Sink token mismatch at head=" << head << ", token=" << token
                          << ", expected=" << expected_val
                          << ", got=" << static_cast<float>(retrieved_keys[base_idx]) << std::endl;
                sink_preserved = false;
                break;
            }
        }
        if (!sink_preserved) break;
    }

    if (!sink_preserved) {
        std::cout << "❌ Sink tokens not preserved during eviction" << std::endl;
        return false;
    }

    std::cout << "✅ Sliding window eviction tests passed" << std::endl;
    return true;
}

bool test_head_major_layout() {
    std::cout << "🔍 Testing head-major layout consistency..." << std::endl;

    size_t num_layers = 1;
    size_t max_seq_len = 10;
    size_t num_kv_heads = 3;
    size_t head_dim = 4;

    KVCache cache;
    cache.init(num_layers, max_seq_len, num_kv_heads, head_dim, CachePrecision::FP16);

    // Create distinctive test data
    size_t token_elements = num_kv_heads * head_dim;
    std::vector<__fp16> keys(token_elements);
    std::vector<__fp16> values(token_elements);

    // Fill with distinctive pattern: head_index * 100 + dim_index
    for (size_t head = 0; head < num_kv_heads; ++head) {
        for (size_t dim = 0; dim < head_dim; ++dim) {
            size_t idx = head * head_dim + dim;
            keys[idx] = static_cast<__fp16>(head * 100.0f + dim);
            values[idx] = static_cast<__fp16>(head * 100.0f + dim + 1000.0f);
        }
    }

    // Append 3 tokens
    for (size_t token = 0; token < 3; ++token) {
        cache.append(0, keys.data(), values.data(), 1);
    }

    // Retrieve data and verify head-major layout
    std::vector<__fp16> retrieved_keys(num_kv_heads * cache.effective_len() * head_dim);
    cache.get_keys_fp16(0, retrieved_keys.data());

    // Verify layout: [num_kv_heads][seq_len][head_dim]
    for (size_t head = 0; head < num_kv_heads; ++head) {
        for (size_t token = 0; token < cache.effective_len(); ++token) {
            for (size_t dim = 0; dim < head_dim; ++dim) {
                size_t retrieved_idx = head * cache.effective_len() * head_dim + token * head_dim + dim;
                float expected_val = head * 100.0f + dim;
                float retrieved_val = static_cast<float>(retrieved_keys[retrieved_idx]);

                if (std::abs(retrieved_val - expected_val) > 0.1f) {
                    std::cout << "❌ Layout mismatch at head=" << head << ", token=" << token
                              << ", dim=" << dim << std::endl;
                    std::cout << "   Expected: " << expected_val << ", Got: " << retrieved_val << std::endl;
                    return false;
                }
            }
        }
    }

    std::cout << "✅ Head-major layout test passed" << std::endl;
    return true;
}

bool test_int8_buffer_functions() {
    std::cout << "🔢 Testing INT8 buffer-based functions..." << std::endl;

    size_t num_layers = 1;
    size_t max_seq_len = 5;
    size_t num_kv_heads = 2;
    size_t head_dim = 8;

    KVCache cache;
    cache.init(num_layers, max_seq_len, num_kv_heads, head_dim, CachePrecision::INT8);

    // Create test data
    size_t token_elements = num_kv_heads * head_dim;
    std::vector<int8_t> keys(token_elements);
    std::vector<int8_t> values(token_elements);
    std::vector<float> key_scales(1, 0.05f);
    std::vector<float> value_scales(1, 0.07f);

    // Fill with distinctive pattern
    for (size_t i = 0; i < token_elements; ++i) {
        keys[i] = static_cast<int8_t>(i % 127);
        values[i] = static_cast<int8_t>((i + 50) % 127);
    }

    // Append 2 tokens
    for (size_t token = 0; token < 2; ++token) {
        cache.append_int8(0, keys.data(), values.data(),
                         key_scales.data(), value_scales.data(), 1);
    }

    // Test buffer-based get functions
    std::vector<int8_t> retrieved_keys(num_kv_heads * cache.effective_len() * head_dim);
    std::vector<int8_t> retrieved_values(num_kv_heads * cache.effective_len() * head_dim);

    try {
        cache.get_keys_int8(0, retrieved_keys.data());
        cache.get_values_int8(0, retrieved_values.data());
    } catch (const std::exception& e) {
        std::cout << "❌ INT8 get functions failed: " << e.what() << std::endl;
        return false;
    }

    // Verify some data
    bool data_correct = true;
    for (size_t head = 0; head < num_kv_heads; ++head) {
        size_t head_base = head * cache.effective_len() * head_dim;
        size_t expected_val = head * head_dim;

        if (retrieved_keys[head_base] != static_cast<int8_t>(expected_val % 127)) {
            std::cout << "❌ INT8 data mismatch at head " << head << std::endl;
            data_correct = false;
            break;
        }
    }

    if (!data_correct) {
        return false;
    }

    std::cout << "✅ INT8 buffer functions test passed" << std::endl;
    return true;
}

bool test_multi_token_append() {
    std::cout << "🎯 Testing multi-token append functionality..." << std::endl;

    size_t num_layers = 1;
    size_t max_seq_len = 20;
    size_t num_kv_heads = 2;
    size_t head_dim = 4;
    size_t num_tokens = 3;

    KVCache cache;
    cache.init(num_layers, max_seq_len, num_kv_heads, head_dim, CachePrecision::FP16);

    // Create multi-token test data: [num_kv_heads][num_tokens][head_dim]
    size_t total_elements = num_kv_heads * num_tokens * head_dim;
    std::vector<__fp16> keys(total_elements);
    std::vector<__fp16> values(total_elements);

    // Fill with pattern: head * 1000 + token * 100 + dim
    for (size_t head = 0; head < num_kv_heads; ++head) {
        for (size_t token = 0; token < num_tokens; ++token) {
            for (size_t dim = 0; dim < head_dim; ++dim) {
                size_t idx = head * num_tokens * head_dim + token * head_dim + dim;
                float val = head * 1000.0f + token * 100.0f + dim;
                keys[idx] = static_cast<__fp16>(val);
                values[idx] = static_cast<__fp16>(val + 10000.0f);
            }
        }
    }

    // Append all tokens at once
    cache.append(0, keys.data(), values.data(), num_tokens);

    if (cache.effective_len() != num_tokens) {
        std::cout << "❌ Multi-token append failed: expected len=" << num_tokens
                  << ", got len=" << cache.effective_len() << std::endl;
        return false;
    }

    // Retrieve and verify
    std::vector<__fp16> retrieved_keys(num_kv_heads * cache.effective_len() * head_dim);
    cache.get_keys_fp16(0, retrieved_keys.data());

    // Verify data layout: [num_kv_heads][seq_len][head_dim]
    for (size_t head = 0; head < num_kv_heads; ++head) {
        for (size_t token = 0; token < num_tokens; ++token) {
            for (size_t dim = 0; dim < head_dim; ++dim) {
                size_t retrieved_idx = head * cache.effective_len() * head_dim + token * head_dim + dim;
                float expected_val = head * 1000.0f + token * 100.0f + dim;
                float retrieved_val = static_cast<float>(retrieved_keys[retrieved_idx]);

                if (std::abs(retrieved_val - expected_val) > 0.1f) {
                    std::cout << "❌ Multi-token data mismatch at head=" << head
                              << ", token=" << token << ", dim=" << dim << std::endl;
                    std::cout << "   Expected: " << expected_val << ", Got: " << retrieved_val << std::endl;
                    return false;
                }
            }
        }
    }

    std::cout << "✅ Multi-token append test passed" << std::endl;
    return true;
}

KVCacheResult benchmark_kv_cache(size_t num_layers, size_t seq_len, size_t num_kv_heads,
                                size_t head_dim, int warmup_runs = 2, int test_runs = 5) {
    KVCacheResult result = {};

    // Setup data
    size_t token_elements = num_kv_heads * head_dim;
    std::vector<__fp16> fp16_keys(token_elements);
    std::vector<__fp16> fp16_values(token_elements);
    std::vector<int8_t> int8_keys(token_elements);
    std::vector<int8_t> int8_values(token_elements);
    std::vector<float> key_scales(1, 0.05f);
    std::vector<float> value_scales(1, 0.05f);

    fill_random_fp16(fp16_keys);
    fill_random_fp16(fp16_values);
    fill_random_int8(int8_keys);
    fill_random_int8(int8_values);

    size_t max_seq_len = seq_len + 100; // Extra buffer

    // Test FP16 performance
    for (int run = 0; run < warmup_runs + test_runs; ++run) {
        KVCache fp16_cache;
        fp16_cache.init(num_layers, max_seq_len, num_kv_heads, head_dim, CachePrecision::FP16);

        std::chrono::high_resolution_clock::time_point run_start = std::chrono::high_resolution_clock::now();

        for (size_t token = 0; token < seq_len; ++token) {
            for (size_t layer = 0; layer < num_layers; ++layer) {
                fp16_cache.append(layer, fp16_keys.data(), fp16_values.data(), 1);
            }
        }

        std::chrono::high_resolution_clock::time_point run_end = std::chrono::high_resolution_clock::now();

        if (run >= warmup_runs) {
            double run_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
            result.fp16_latency_ms += run_ms;
        }
    }
    result.fp16_latency_ms /= test_runs;

    // Test INT8 performance
    for (int run = 0; run < warmup_runs + test_runs; ++run) {
        KVCache int8_cache;
        int8_cache.init(num_layers, max_seq_len, num_kv_heads, head_dim, CachePrecision::INT8);

        std::chrono::high_resolution_clock::time_point run_start = std::chrono::high_resolution_clock::now();

        for (size_t token = 0; token < seq_len; ++token) {
            for (size_t layer = 0; layer < num_layers; ++layer) {
                int8_cache.append_int8(layer, int8_keys.data(), int8_values.data(),
                                     key_scales.data(), value_scales.data(), 1);
            }
        }

        std::chrono::high_resolution_clock::time_point run_end = std::chrono::high_resolution_clock::now();

        if (run >= warmup_runs) {
            double run_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
            result.int8_latency_ms += run_ms;
        }
    }
    result.int8_latency_ms /= test_runs;

    // Memory savings calculation
    size_t fp16_memory = num_layers * seq_len * num_kv_heads * head_dim * sizeof(__fp16) * 2; // keys + values
    size_t int8_memory = num_layers * seq_len * num_kv_heads * head_dim * sizeof(int8_t) * 2; // keys + values
    int8_memory += num_layers * seq_len * sizeof(float) * 2; // scales

    result.memory_saved_percent = (1.0 - static_cast<double>(int8_memory) / fp16_memory) * 100.0;

    // Throughput calculation (GB/s for append operations)
    double total_data_gb = (num_layers * seq_len * token_elements * sizeof(__fp16) * 2) / (1024.0 * 1024.0 * 1024.0);
    result.append_throughput_gb_s = total_data_gb / (result.fp16_latency_ms / 1000.0);

    result.correctness_passed = true;
    result.accuracy_loss = 0.0; // No conversion happening in these tests

    return result;
}

void print_results(const KVCacheResult& result, size_t num_layers, size_t seq_len,
                  size_t num_kv_heads, size_t head_dim) {
    std::cout << "\n📊 KV Cache Benchmark Results" << std::endl;
    std::cout << "=============================" << std::endl;
    std::cout << "Configuration: " << num_layers << " layers, " << seq_len << " tokens, "
              << num_kv_heads << " heads, " << head_dim << " dims" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "FP16 Latency:     " << result.fp16_latency_ms << " ms" << std::endl;
    std::cout << "INT8 Latency:     " << result.int8_latency_ms << " ms" << std::endl;
    std::cout << "Memory Saved:     " << result.memory_saved_percent << "%" << std::endl;
    std::cout << "Append Throughput:" << result.append_throughput_gb_s << " GB/s" << std::endl;
    std::cout << "Correctness:      " << (result.correctness_passed ? "✅ PASS" : "❌ FAIL") << std::endl;
}

int main() {
    std::cout << "🎯 KV CACHE PERFORMANCE & CORRECTNESS TESTS" << std::endl;
    std::cout << "===========================================" << std::endl;

    bool all_passed = true;

    // Correctness tests
    all_passed &= test_basic_functionality();
    all_passed &= test_input_validation();
    all_passed &= test_head_major_layout();
    all_passed &= test_multi_token_append();
    all_passed &= test_int8_buffer_functions();
    all_passed &= test_sliding_window_eviction();

    if (!all_passed) {
        std::cout << "\n❌ Some correctness tests failed!" << std::endl;
        return 1;
    }

    // Performance benchmarks
    struct Config {
        size_t layers, seq_len, heads, head_dim;
        Config(size_t l, size_t s, size_t h, size_t d) : layers(l), seq_len(s), heads(h), head_dim(d) {}
    };

    std::vector<Config> configs;
    configs.push_back(Config(4, 64, 8, 64));      // Small config
    configs.push_back(Config(12, 256, 12, 64));   // Medium config
    configs.push_back(Config(32, 512, 32, 128));  // Large config

    for (size_t i = 0; i < configs.size(); ++i) {
        const Config& config = configs[i];
        KVCacheResult result = benchmark_kv_cache(config.layers, config.seq_len, config.heads, config.head_dim);
        print_results(result, config.layers, config.seq_len, config.heads, config.head_dim);
    }

    std::cout << "\n✅ All KV cache tests completed successfully!" << std::endl;
    return 0;
}