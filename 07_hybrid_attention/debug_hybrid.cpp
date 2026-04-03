#include "hybrid_attention.h"
#include "../06_kv_cache/kv_cache.h"
#include <iostream>
#include <vector>
#include <random>

std::vector<__fp16> generate_debug_fp16(size_t size) {
    std::vector<__fp16> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = (__fp16)(0.1f * ((int)(i % 10) - 5)); // Simple pattern: -0.5 to 0.4
    }
    return data;
}

int main() {
    std::cout << "🐛 DEBUGGING HYBRID ATTENTION" << std::endl;
    std::cout << "=============================" << std::endl;

    try {
        // Very simple parameters
        const size_t batch_size = 1;
        const size_t seq_len = 4;
        const size_t cache_len = 2;
        const size_t new_len = 2;
        const size_t num_q_heads = 1;
        const size_t num_kv_heads = 1;
        const size_t head_dim = 8;
        const float scale = 1.0f / std::sqrt(head_dim);
        const size_t quant_group_size = 2;

        std::cout << "Step 1: Generate test data..." << std::endl;
        std::vector<__fp16> Q = generate_debug_fp16(batch_size * num_q_heads * seq_len * head_dim);
        std::vector<__fp16> K_new = generate_debug_fp16(batch_size * num_kv_heads * new_len * head_dim);
        std::vector<__fp16> V_new = generate_debug_fp16(batch_size * num_kv_heads * new_len * head_dim);
        std::cout << "✓ Generated data - Q: " << Q.size() << ", K_new: " << K_new.size() << ", V_new: " << V_new.size() << std::endl;

        std::cout << "Step 2: Create KV cache..." << std::endl;
        KVCache cache;
        cache.init(1, cache_len + 5, num_kv_heads, head_dim, CachePrecision::INT8, quant_group_size);
        std::cout << "✓ Created cache" << std::endl;

        std::cout << "Step 3: Populate cache..." << std::endl;
        for (size_t t = 0; t < cache_len; ++t) {
            std::vector<__fp16> k_token = generate_debug_fp16(num_kv_heads * head_dim);
            std::vector<__fp16> v_token = generate_debug_fp16(num_kv_heads * head_dim);
            cache.append(0, k_token.data(), v_token.data(), 1);
            std::cout << "  ✓ Added token " << t << std::endl;
        }
        std::cout << "✓ Cache populated, effective_len: " << cache.effective_len() << std::endl;

        std::cout << "Step 4: Extract cache data..." << std::endl;
        std::vector<int8_t> K_cached(cache_len * num_kv_heads * head_dim);
        std::vector<int8_t> V_cached(cache_len * num_kv_heads * head_dim);
        cache.get_keys_int8(0, K_cached.data());
        cache.get_values_int8(0, V_cached.data());
        std::cout << "✓ Extracted cache data - K: " << K_cached.size() << ", V: " << V_cached.size() << std::endl;

        const float* k_scales = cache.get_key_scales(0);
        const float* v_scales = cache.get_value_scales(0);
        std::cout << "✓ Got scales - k_scales[0]: " << k_scales[0] << ", v_scales[0]: " << v_scales[0] << std::endl;

        std::cout << "Step 5: Prepare output buffer..." << std::endl;
        std::vector<__fp16> O(batch_size * num_q_heads * seq_len * head_dim, 0.0f);
        std::cout << "✓ Output buffer size: " << O.size() << std::endl;

        std::cout << "Step 6: Call hybrid attention..." << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "  batch_size=" << batch_size << ", seq_len=" << seq_len << std::endl;
        std::cout << "  cache_len=" << cache_len << ", new_len=" << new_len << std::endl;
        std::cout << "  num_q_heads=" << num_q_heads << ", num_kv_heads=" << num_kv_heads << std::endl;
        std::cout << "  head_dim=" << head_dim << ", scale=" << scale << std::endl;

        hybrid_attention_f16(
            Q.data(), K_cached.data(), V_cached.data(),
            k_scales, v_scales,
            K_new.data(), V_new.data(), O.data(),
            batch_size, seq_len, cache_len, new_len,
            num_q_heads, num_kv_heads, head_dim,
            scale, 0, true, 0, quant_group_size
        );

        std::cout << "✅ SUCCESS: Hybrid attention completed!" << std::endl;

        // Check output
        bool has_nonzero = false;
        float sum = 0.0f;
        for (size_t i = 0; i < O.size(); ++i) {
            float val = (float)O[i];
            sum += std::abs(val);
            if (std::abs(val) > 1e-6f) {
                has_nonzero = true;
            }
        }

        std::cout << "Output check: has_nonzero=" << has_nonzero << ", sum_abs=" << sum << std::endl;
        std::cout << "First few values: ";
        for (size_t i = 0; i < std::min((size_t)8, O.size()); ++i) {
            std::cout << (float)O[i] << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cout << "❌ ERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "❌ UNKNOWN ERROR" << std::endl;
        return 1;
    }

    return 0;
}