#include "flash_attention.h"
#include "../04_naive_attention/naive_attention.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

void fill_random(std::vector<__fp16>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<__fp16>(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }
}

int main() {
    size_t batch_size = 1;
    size_t seq_len = 1024;
    size_t kv_seq_len = 1024;
    size_t num_q_heads = 32;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    
    float scale = 1.0f / std::sqrt((float)head_dim);
    bool is_causal = true;
    size_t window_size = 0;

    size_t q_size = batch_size * num_q_heads * seq_len * head_dim;
    size_t kv_size = batch_size * num_kv_heads * kv_seq_len * head_dim;
    size_t out_size = q_size;

    std::vector<__fp16> Q(q_size);
    std::vector<__fp16> K(kv_size);
    std::vector<__fp16> V(kv_size);
    std::vector<__fp16> O_naive(out_size, 0.0f);
    std::vector<__fp16> O_flash(out_size, 0.0f);

    std::cout << "Initializing randomized inputs...\n";
    fill_random(Q);
    fill_random(K);
    fill_random(V);

    std::cout << "\n--- 05 Flash Attention Testing ---\n";
    std::cout << "Running Naive Attention...\n";
    auto start = std::chrono::high_resolution_clock::now();
    naive_attention(Q.data(), K.data(), V.data(), O_naive.data(), batch_size, seq_len, kv_seq_len,
                    num_q_heads, num_kv_heads, head_dim, scale, is_causal, window_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> naive_ms = end - start;
    std::cout << "Naive time: " << naive_ms.count() << " ms\n";

    std::cout << "Running Flash Attention...\n";
    start = std::chrono::high_resolution_clock::now();
    flash_attention_f16(Q.data(), K.data(), V.data(), O_flash.data(), batch_size, seq_len, kv_seq_len,
                        num_q_heads, num_kv_heads, head_dim, scale, is_causal, 0, window_size);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> flash_ms = end - start;
    std::cout << "Flash time: " << flash_ms.count() << " ms\n";

    std::cout << "Comparing outputs...\n";
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_size; ++i) {
        float diff = std::abs((float)O_naive[i] - (float)O_flash[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    std::cout << "Max absolute difference: " << max_diff << "\n";
    if (max_diff < 0.01f) {
        std::cout << "TEST PASSED: FlashAttention matches NaiveAttention!\n";
    } else {
        std::cout << "TEST FAILED: Outputs differ significantly.\n";
    }

    return 0;
}
