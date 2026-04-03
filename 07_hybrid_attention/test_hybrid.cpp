#include "hybrid_attention.h"
#include "../06_kv_cache/kv_cache.h"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

// Simple test data generator
std::vector<__fp16> generate_random_fp16(size_t size) {
    std::vector<__fp16> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        data[i] = (__fp16)dis(gen);
    }
    return data;
}

std::vector<int8_t> generate_random_int8(size_t size) {
    std::vector<int8_t> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-127, 127);

    for (size_t i = 0; i < size; ++i) {
        data[i] = (int8_t)dis(gen);
    }
    return data;
}

std::vector<float> generate_random_scales(size_t size) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.001f, 0.1f);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

void test_basic_compilation() {
    std::cout << "Testing hybrid attention with KVCache integration..." << std::endl;

    // Small test parameters
    const size_t batch_size = 1;  // Single batch for simplicity
    const size_t seq_len = 16;
    const size_t cache_len = 8;
    const size_t new_len = 4;
    const size_t num_q_heads = 2;
    const size_t num_kv_heads = 1;  // Simple case: 2:1 ratio
    const size_t head_dim = 32;
    const float scale = 1.0f / std::sqrt(head_dim);
    const size_t quant_group_size = 4;

    std::cout << "Running hybrid attention with:" << std::endl;
    std::cout << "  Batch: " << batch_size << ", Seq: " << seq_len << std::endl;
    std::cout << "  Cache len: " << cache_len << ", New len: " << new_len << std::endl;
    std::cout << "  Q heads: " << num_q_heads << ", KV heads: " << num_kv_heads << std::endl;
    std::cout << "  Head dim: " << head_dim << std::endl;

    try {
        // Create and populate KV cache
        KVCache cache;
        cache.init(1, cache_len, num_kv_heads, head_dim, CachePrecision::INT8, quant_group_size);

        // Generate and add cached tokens (simulate past tokens)
        for (size_t t = 0; t < cache_len; ++t) {
            auto k_token = generate_random_fp16(num_kv_heads * head_dim);
            auto v_token = generate_random_fp16(num_kv_heads * head_dim);
            cache.append(0, k_token.data(), v_token.data(), 1);
        }

        // Generate Q and new KV data
        auto Q = generate_random_fp16(batch_size * num_q_heads * seq_len * head_dim);
        auto K_new = generate_random_fp16(batch_size * num_kv_heads * new_len * head_dim);
        auto V_new = generate_random_fp16(batch_size * num_kv_heads * new_len * head_dim);

        // Get cached data from KVCache (this does the copy and packing)
        std::vector<int8_t> K_cached(cache_len * num_kv_heads * head_dim);
        std::vector<int8_t> V_cached(cache_len * num_kv_heads * head_dim);
        cache.get_keys_int8(0, K_cached.data());
        cache.get_values_int8(0, V_cached.data());

        // Get quantization scales
        const float* k_scales = cache.get_key_scales(0);
        const float* v_scales = cache.get_value_scales(0);

        // Output buffer
        std::vector<__fp16> O(batch_size * num_q_heads * seq_len * head_dim, 0.0f);

        // Call hybrid attention
        hybrid_attention_f16(
            Q.data(), K_cached.data(), V_cached.data(),
            k_scales, v_scales,
            K_new.data(), V_new.data(), O.data(),
            batch_size, seq_len, cache_len, new_len,
            num_q_heads, num_kv_heads, head_dim,
            scale, 0, true, 0, quant_group_size
        );

        std::cout << "✅ Hybrid attention executed successfully!" << std::endl;

        // Basic sanity check - output should not be all zeros
        bool has_nonzero = false;
        for (size_t i = 0; i < O.size(); ++i) {
            if (std::abs((float)O[i]) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }

        if (has_nonzero) {
            std::cout << "✅ Output contains non-zero values (sanity check passed)" << std::endl;
        } else {
            std::cout << "⚠️  Warning: Output is all zeros" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "❌ Unknown error occurred" << std::endl;
    }
}

int main() {
    std::cout << "🎯 HYBRID ATTENTION TEST" << std::endl;
    std::cout << "========================" << std::endl;

    test_basic_compilation();

    std::cout << "Test completed." << std::endl;
    return 0;
}