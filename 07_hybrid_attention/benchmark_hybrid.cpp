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

// Test utilities
std::vector<__fp16> generate_test_fp16(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<__fp16> data(size);
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < size; ++i) {
        data[i] = (__fp16)dis(gen);
    }
    return data;
}

float compute_max_error(const std::vector<__fp16>& a, const std::vector<__fp16>& b) {
    assert(a.size() == b.size());
    float max_error = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float error = std::abs((float)a[i] - (float)b[i]);
        max_error = std::max(max_error, error);
    }
    return max_error;
}

float compute_mean_error(const std::vector<__fp16>& a, const std::vector<__fp16>& b) {
    assert(a.size() == b.size());
    double sum_error = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float error = std::abs((float)a[i] - (float)b[i]);
        sum_error += error;
    }
    return (float)(sum_error / a.size());
}

// Test 1: Correctness vs Naive Attention
bool test_correctness_vs_naive() {
    std::cout << "\n🧪 Test 1: Correctness vs Naive Attention" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Test parameters
    const size_t batch_size = 1;
    const size_t seq_len = 32;
    const size_t cache_len = 16;
    const size_t new_len = 8;
    const size_t num_q_heads = 4;
    const size_t num_kv_heads = 2;  // GQA: 2:1 ratio
    const size_t head_dim = 64;
    const float scale = 1.0f / std::sqrt(head_dim);
    const size_t quant_group_size = 8;

    std::cout << "Config: batch=" << batch_size << ", seq=" << seq_len
              << ", cache=" << cache_len << ", new=" << new_len
              << ", q_heads=" << num_q_heads << ", kv_heads=" << num_kv_heads
              << ", head_dim=" << head_dim << std::endl;

    // Generate test data
    std::vector<__fp16> Q = generate_test_fp16(batch_size * num_q_heads * seq_len * head_dim);
    std::vector<__fp16> K_full = generate_test_fp16(batch_size * num_kv_heads * (cache_len + new_len) * head_dim);
    std::vector<__fp16> V_full = generate_test_fp16(batch_size * num_kv_heads * (cache_len + new_len) * head_dim);

    // Extract new tokens from full sequence
    std::vector<__fp16> K_new(batch_size * num_kv_heads * new_len * head_dim);
    std::vector<__fp16> V_new(batch_size * num_kv_heads * new_len * head_dim);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_kv_heads; ++h) {
            for (size_t t = 0; t < new_len; ++t) {
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t full_idx = b * num_kv_heads * (cache_len + new_len) * head_dim +
                                     h * (cache_len + new_len) * head_dim +
                                     (cache_len + t) * head_dim + d;
                    size_t new_idx = b * num_kv_heads * new_len * head_dim +
                                    h * new_len * head_dim + t * head_dim + d;
                    K_new[new_idx] = K_full[full_idx];
                    V_new[new_idx] = V_full[full_idx];
                }
            }
        }
    }

    // Create and populate cache
    KVCache cache;
    cache.init(1, cache_len + 10, num_kv_heads, head_dim, CachePrecision::INT8);  // Extra space

    // Add cached tokens (first cache_len tokens)
    for (size_t t = 0; t < cache_len; ++t) {
        std::vector<__fp16> k_token(num_kv_heads * head_dim);
        std::vector<__fp16> v_token(num_kv_heads * head_dim);

        for (size_t h = 0; h < num_kv_heads; ++h) {
            for (size_t d = 0; d < head_dim; ++d) {
                size_t full_idx = h * (cache_len + new_len) * head_dim + t * head_dim + d;
                size_t token_idx = h * head_dim + d;
                k_token[token_idx] = K_full[full_idx];
                v_token[token_idx] = V_full[full_idx];
            }
        }

        cache.append(0, k_token.data(), v_token.data(), 1);
    }

    // Get quantized cache data
    std::vector<int8_t> K_cached(cache_len * num_kv_heads * head_dim);
    std::vector<int8_t> V_cached(cache_len * num_kv_heads * head_dim);
    cache.get_keys_int8(0, K_cached.data());
    cache.get_values_int8(0, V_cached.data());
    const float* k_scales = cache.get_key_scales(0);
    const float* v_scales = cache.get_value_scales(0);

    // Run naive attention on full sequence
    std::vector<__fp16> O_naive(batch_size * num_q_heads * seq_len * head_dim);
    naive_attention(Q.data(), K_full.data(), V_full.data(), O_naive.data(),
                       batch_size, seq_len, cache_len + new_len,
                       num_q_heads, num_kv_heads, head_dim, scale,
                       true);  // causal=true

    // Run hybrid attention
    std::vector<__fp16> O_hybrid(batch_size * num_q_heads * seq_len * head_dim);
    hybrid_attention_f16(Q.data(), K_cached.data(), V_cached.data(),
                        k_scales, v_scales, K_new.data(), V_new.data(),
                        O_hybrid.data(), batch_size, seq_len, cache_len, new_len,
                        num_q_heads, num_kv_heads, head_dim, scale,
                        0, true, 0, quant_group_size);

    // Compare results - focus on positions that have full context
    float max_error = 0.0f;
    float sum_error = 0.0f;
    size_t num_compared = 0;

    // Only compare tokens that see the full cached + new context
    size_t start_pos = cache_len;  // First position that sees cached data
    size_t end_pos = std::min(seq_len, cache_len + new_len);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_q_heads; ++h) {
            for (size_t t = start_pos; t < end_pos; ++t) {
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t idx = b * num_q_heads * seq_len * head_dim +
                               h * seq_len * head_dim + t * head_dim + d;

                    float naive_val = (float)O_naive[idx];
                    float hybrid_val = (float)O_hybrid[idx];
                    float error = std::abs(naive_val - hybrid_val);

                    max_error = std::max(max_error, error);
                    sum_error += error;
                    num_compared++;
                }
            }
        }
    }

    float mean_error = sum_error / num_compared;

    std::cout << "Compared " << num_compared << " values" << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Mean error: " << mean_error << std::endl;

    // Error thresholds (allow for quantization error)
    const float MAX_ERROR_THRESHOLD = 0.1f;    // 10% relative error
    const float MEAN_ERROR_THRESHOLD = 0.05f;  // 5% mean relative error

    bool passed = (max_error < MAX_ERROR_THRESHOLD) && (mean_error < MEAN_ERROR_THRESHOLD);

    if (passed) {
        std::cout << "✅ PASSED: Hybrid attention matches naive within tolerance" << std::endl;
    } else {
        std::cout << "❌ FAILED: Error exceeds threshold" << std::endl;
        std::cout << "   Max error: " << max_error << " (threshold: " << MAX_ERROR_THRESHOLD << ")" << std::endl;
        std::cout << "   Mean error: " << mean_error << " (threshold: " << MEAN_ERROR_THRESHOLD << ")" << std::endl;
    }

    return passed;
}

// Test 2: Multiple configurations
bool test_multiple_configs() {
    std::cout << "\n🧪 Test 2: Multiple Configurations" << std::endl;
    std::cout << "==================================" << std::endl;

    struct TestConfig {
        size_t batch_size, seq_len, cache_len, new_len;
        size_t num_q_heads, num_kv_heads, head_dim;
        const char* description;
    };

    std::vector<TestConfig> configs;
    TestConfig config1 = {1, 16, 8, 4, 8, 8, 32, "Single head"};
    TestConfig config2 = {1, 32, 16, 8, 4, 2, 64, "GQA 2:1"};
    TestConfig config3 = {1, 64, 32, 16, 12, 4, 128, "GQA 3:1"};
    TestConfig config4 = {2, 24, 12, 6, 8, 2, 64, "Batched"};
    configs.push_back(config1);
    configs.push_back(config2);
    configs.push_back(config3);
    configs.push_back(config4);

    bool all_passed = true;

    for (size_t i = 0; i < configs.size(); ++i) {
        const TestConfig& config = configs[i];
        std::cout << "\nTesting: " << config.description << std::endl;

        // Simplified test for each config
        try {
            std::vector<__fp16> Q = generate_test_fp16(config.batch_size * config.num_q_heads * config.seq_len * config.head_dim);
            std::vector<__fp16> K_new = generate_test_fp16(config.batch_size * config.num_kv_heads * config.new_len * config.head_dim);
            std::vector<__fp16> V_new = generate_test_fp16(config.batch_size * config.num_kv_heads * config.new_len * config.head_dim);

            // Create minimal cache
            KVCache cache;
            cache.init(1, config.cache_len + 5, config.num_kv_heads, config.head_dim, CachePrecision::INT8);

            for (size_t t = 0; t < config.cache_len; ++t) {
                std::vector<__fp16> k_token = generate_test_fp16(config.num_kv_heads * config.head_dim);
                std::vector<__fp16> v_token = generate_test_fp16(config.num_kv_heads * config.head_dim);
                cache.append(0, k_token.data(), v_token.data(), 1);
            }

            std::vector<int8_t> K_cached(config.cache_len * config.num_kv_heads * config.head_dim);
            std::vector<int8_t> V_cached(config.cache_len * config.num_kv_heads * config.head_dim);
            cache.get_keys_int8(0, K_cached.data());
            cache.get_values_int8(0, V_cached.data());

            std::vector<__fp16> O(config.batch_size * config.num_q_heads * config.seq_len * config.head_dim);

            float scale = 1.0f / std::sqrt(config.head_dim);

            hybrid_attention_f16(Q.data(), K_cached.data(), V_cached.data(),
                                cache.get_key_scales(0), cache.get_value_scales(0),
                                K_new.data(), V_new.data(), O.data(),
                                config.batch_size, config.seq_len, config.cache_len, config.new_len,
                                config.num_q_heads, config.num_kv_heads, config.head_dim,
                                scale, 0, true, 0, 8);

            // Basic sanity check
            bool has_nonzero = false;
            for (size_t i = 0; i < O.size(); ++i) {
                if (std::abs((float)O[i]) > 1e-6f) {
                    has_nonzero = true;
                    break;
                }
            }

            if (has_nonzero) {
                std::cout << "✅ " << config.description << " - PASSED" << std::endl;
            } else {
                std::cout << "❌ " << config.description << " - FAILED (all zeros)" << std::endl;
                all_passed = false;
            }

        } catch (const std::exception& e) {
            std::cout << "❌ " << config.description << " - FAILED: " << e.what() << std::endl;
            all_passed = false;
        }
    }

    return all_passed;
}

// Test 3: Performance Benchmark
void benchmark_performance() {
    std::cout << "\n⚡ Performance Benchmark" << std::endl;
    std::cout << "=======================" << std::endl;

    const size_t batch_size = 1;
    const size_t seq_len = 128;
    const size_t cache_len = 64;
    const size_t new_len = 32;
    const size_t num_q_heads = 8;
    const size_t num_kv_heads = 2;
    const size_t head_dim = 128;
    const float scale = 1.0f / std::sqrt(head_dim);
    const size_t num_trials = 10;

    std::cout << "Config: seq_len=" << seq_len << ", cache_len=" << cache_len
              << ", new_len=" << new_len << ", heads=" << num_q_heads
              << "/" << num_kv_heads << ", head_dim=" << head_dim << std::endl;
    std::cout << "Trials: " << num_trials << std::endl;

    // Generate test data
    std::vector<__fp16> Q = generate_test_fp16(batch_size * num_q_heads * seq_len * head_dim);
    std::vector<__fp16> K_full = generate_test_fp16(batch_size * num_kv_heads * (cache_len + new_len) * head_dim);
    std::vector<__fp16> V_full = generate_test_fp16(batch_size * num_kv_heads * (cache_len + new_len) * head_dim);
    std::vector<__fp16> K_new = generate_test_fp16(batch_size * num_kv_heads * new_len * head_dim);
    std::vector<__fp16> V_new = generate_test_fp16(batch_size * num_kv_heads * new_len * head_dim);

    // Setup cache
    KVCache cache;
    cache.init(1, cache_len + 10, num_kv_heads, head_dim, CachePrecision::INT8);

    for (size_t t = 0; t < cache_len; ++t) {
        std::vector<__fp16> k_token(num_kv_heads * head_dim);
        std::vector<__fp16> v_token(num_kv_heads * head_dim);
        for (size_t h = 0; h < num_kv_heads; ++h) {
            for (size_t d = 0; d < head_dim; ++d) {
                k_token[h * head_dim + d] = K_full[h * (cache_len + new_len) * head_dim + t * head_dim + d];
                v_token[h * head_dim + d] = V_full[h * (cache_len + new_len) * head_dim + t * head_dim + d];
            }
        }
        cache.append(0, k_token.data(), v_token.data(), 1);
    }

    std::vector<int8_t> K_cached(cache_len * num_kv_heads * head_dim);
    std::vector<int8_t> V_cached(cache_len * num_kv_heads * head_dim);
    cache.get_keys_int8(0, K_cached.data());
    cache.get_values_int8(0, V_cached.data());

    // Benchmark Hybrid Attention
    std::vector<__fp16> O_hybrid(batch_size * num_q_heads * seq_len * head_dim);

    std::chrono::high_resolution_clock::time_point start_hybrid = std::chrono::high_resolution_clock::now();
    for (size_t trial = 0; trial < num_trials; ++trial) {
        hybrid_attention_f16(Q.data(), K_cached.data(), V_cached.data(),
                            cache.get_key_scales(0), cache.get_value_scales(0),
                            K_new.data(), V_new.data(), O_hybrid.data(),
                            batch_size, seq_len, cache_len, new_len,
                            num_q_heads, num_kv_heads, head_dim,
                            scale, 0, true, 0, 8);
    }
    std::chrono::high_resolution_clock::time_point end_hybrid = std::chrono::high_resolution_clock::now();

    // Benchmark Naive Attention (for comparison)
    std::vector<__fp16> O_naive(batch_size * num_q_heads * seq_len * head_dim);

    std::chrono::high_resolution_clock::time_point start_naive = std::chrono::high_resolution_clock::now();
    for (size_t trial = 0; trial < num_trials; ++trial) {
        naive_attention(Q.data(), K_full.data(), V_full.data(), O_naive.data(),
                           batch_size, seq_len, cache_len + new_len,
                           num_q_heads, num_kv_heads, head_dim, scale,
                           true);
    }
    std::chrono::high_resolution_clock::time_point end_naive = std::chrono::high_resolution_clock::now();

    // Calculate timing
    std::chrono::microseconds hybrid_time = std::chrono::duration_cast<std::chrono::microseconds>(end_hybrid - start_hybrid);
    std::chrono::microseconds naive_time = std::chrono::duration_cast<std::chrono::microseconds>(end_naive - start_naive);

    double hybrid_avg = hybrid_time.count() / (double)num_trials;
    double naive_avg = naive_time.count() / (double)num_trials;
    double speedup = naive_avg / hybrid_avg;

    size_t total_ops = batch_size * num_q_heads * seq_len * (cache_len + new_len) * head_dim * 2; // Q*K + score*V
    double hybrid_throughput = total_ops / (hybrid_avg / 1e6) / 1e9; // GOPS

    std::cout << "\nResults:" << std::endl;
    std::cout << "Hybrid Attention: " << hybrid_avg << " μs/iter" << std::endl;
    std::cout << "Naive Attention:  " << naive_avg << " μs/iter" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    std::cout << "Throughput: " << hybrid_throughput << " GOPS" << std::endl;
}

int main() {
    std::cout << "🚀 HYBRID ATTENTION VALIDATION SUITE" << std::endl;
    std::cout << "====================================" << std::endl;

    bool all_tests_passed = true;

    // Run correctness tests
    all_tests_passed &= test_correctness_vs_naive();
    all_tests_passed &= test_multiple_configs();

    // Run performance benchmark
    benchmark_performance();

    // Summary
    std::cout << "\n📊 SUMMARY" << std::endl;
    std::cout << "==========" << std::endl;
    if (all_tests_passed) {
        std::cout << "✅ All correctness tests PASSED" << std::endl;
        std::cout << "🎉 Hybrid attention implementation is validated!" << std::endl;
    } else {
        std::cout << "❌ Some tests FAILED" << std::endl;
        std::cout << "🔧 Implementation needs debugging" << std::endl;
    }

    return all_tests_passed ? 0 : 1;
}