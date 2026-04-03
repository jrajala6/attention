#include "hybrid_attention.h"
#include "../06_kv_cache/kv_cache.h"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

std::vector<__fp16> gen_fp16(size_t size) {
    std::vector<__fp16> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i)
        data[i] = (__fp16)dis(gen);
    return data;
}

int main() {
    std::cout << "\n=== Hybrid Attention: Smoke Test ===\n";

    const size_t batch = 1, seq = 16, cache_len = 8, new_len = 4;
    const size_t q_heads = 2, kv_heads = 1, head_dim = 32;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const size_t group_size = 4;

    KVCache cache;
    cache.init(1, cache_len, kv_heads, head_dim, CachePrecision::INT8, group_size);

    for (size_t t = 0; t < cache_len; ++t) {
        auto k = gen_fp16(kv_heads * head_dim);
        auto v = gen_fp16(kv_heads * head_dim);
        cache.append(0, k.data(), v.data(), 1);
    }

    auto Q = gen_fp16(batch * q_heads * seq * head_dim);
    auto K_new = gen_fp16(batch * kv_heads * new_len * head_dim);
    auto V_new = gen_fp16(batch * kv_heads * new_len * head_dim);

    std::vector<int8_t> K_cached(cache_len * kv_heads * head_dim);
    std::vector<int8_t> V_cached(cache_len * kv_heads * head_dim);
    cache.get_keys_int8(0, K_cached.data());
    cache.get_values_int8(0, V_cached.data());

    std::vector<__fp16> O(batch * q_heads * seq * head_dim, 0.0f);

    hybrid_attention_f16(
        Q.data(), K_cached.data(), V_cached.data(),
        cache.get_key_scales(0), cache.get_value_scales(0),
        K_new.data(), V_new.data(), O.data(),
        batch, seq, cache_len, new_len,
        q_heads, kv_heads, head_dim,
        scale, 0, true, 0, group_size
    );

    bool has_nonzero = false;
    for (size_t i = 0; i < O.size(); ++i) {
        if (std::abs((float)O[i]) > 1e-6f) { has_nonzero = true; break; }
    }

    std::cout << "Output: " << (has_nonzero ? "PASS (non-zero)" : "FAIL (all zeros)") << "\n\n";
    return has_nonzero ? 0 : 1;
}
