#include "naive_attention.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

void fill_random(std::vector<__fp16>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<__fp16>(static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }
}

int main() {
    // Standard LLM shapes for a single layer
    size_t batch_size = 1;
    size_t seq_len = 1024;
    size_t kv_seq_len = 1024;
    size_t num_q_heads = 32;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    
    float scale = 1.0f / std::sqrt((float)head_dim);
    bool is_causal = true;
    size_t window_size = 0; // 0 = standard causal, infinite window

    size_t q_size = batch_size * num_q_heads * seq_len * head_dim;
    size_t kv_size = batch_size * num_kv_heads * kv_seq_len * head_dim;
    size_t out_size = q_size;

    std::vector<__fp16> Q(q_size);
    std::vector<__fp16> K(kv_size);
    std::vector<__fp16> V(kv_size);
    std::vector<__fp16> O(out_size, 0.0f);

    std::cout << "Initializing randomized inputs...\n";
    fill_random(Q);
    fill_random(K);
    fill_random(V);

    std::cout << "\n--- 04 Naive Attention Benchmark ---\n";
    std::cout << "Batch: " << batch_size << "\n"
              << "Seq_len: " << seq_len << "\n"
              << "Q_Heads: " << num_q_heads << "\n"
              << "KV_Heads: " << num_kv_heads << " (Grouped Query Ratio: " << (num_q_heads / num_kv_heads) << ")\n"
              << "Head_dim: " << head_dim << "\n\n";
              
    std::cout << "Running Warmup...\n";
    
    // Warmup
    naive_attention(Q.data(), K.data(), V.data(), O.data(), batch_size, seq_len, kv_seq_len,
                    num_q_heads, num_kv_heads, head_dim, scale, is_causal, window_size);

    int num_iters = 10;
    std::cout << "Running " << num_iters << " iterations to measure average latency...\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iters; ++i) {
        naive_attention(Q.data(), K.data(), V.data(), O.data(), batch_size, seq_len, kv_seq_len,
                        num_q_heads, num_kv_heads, head_dim, scale, is_causal, window_size);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total_ms = end - start;
    double avg_ms = total_ms.count() / num_iters;

    std::cout << "Average Latency: " << avg_ms << " ms per forward pass\n";
    std::cout << "Done.\n";

    return 0;
}
